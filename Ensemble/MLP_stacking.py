import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from pytorch_widedeep.models import SAINT, WideDeep

warnings.filterwarnings("ignore", category=FutureWarning)


def model_predict_proba(clf, X):
	if hasattr(clf, "predict_proba"):
		return clf.predict_proba(X)
	return clf["predict_proba"](X)


def model_predict(clf, X):
	if hasattr(clf, "predict"):
		return clf.predict(X)
	return clf["predict"](X)


def make_torch_tabular_predictor(model, preprocessor, device="cpu", batch_size=128):
	model = model.to(device)
	model.eval()

	def predict_proba(X):
		X_processed = preprocessor.transform(X)
		X_tensor = torch.tensor(X_processed, dtype=torch.float32)

		prob_chunks = []
		with torch.no_grad():
			for start in range(0, X_tensor.shape[0], batch_size):
				end = start + batch_size
				batch = X_tensor[start:end].to(device)
				try:
					logits = model({"deeptabular": batch})
				except KeyError:
					logits = model({"X_tab": batch})
				probs = torch.softmax(logits, dim=1).cpu().numpy()
				prob_chunks.append(probs)

		return np.vstack(prob_chunks)

	def predict(X):
		return np.argmax(predict_proba(X), axis=1)

	return {"predict_proba": predict_proba, "predict": predict}


def load_saint_predictor(models_dir):
	saint_dir = os.path.join(models_dir, "Saint")
	saint_model_path = os.path.join(saint_dir, "model_state_dict.pt")
	saint_preprocessor_path = os.path.join(saint_dir, "tab_preprocessor.joblib")
	saint_config_path = os.path.join(saint_dir, "config.joblib")

	if not (os.path.exists(saint_model_path) and os.path.exists(saint_preprocessor_path)):
		print("✗ SAINT model not found")
		print(f"  Expected files: {saint_model_path} and {saint_preprocessor_path}")
		print("  Run: python3 Trainings/train_saint.py")
		return None

	try:
		saint_preprocessor = joblib.load(saint_preprocessor_path)
		saint_config = joblib.load(saint_config_path) if os.path.exists(saint_config_path) else {}
		features = saint_config.get("features", {})
		continuous_cols = features.get(
			"all_continuous_cols", getattr(saint_preprocessor, "continuous_cols", [])
		)

		saint = SAINT(
			column_idx=saint_preprocessor.column_idx,
			cat_embed_input=saint_preprocessor.cat_embed_input,
			continuous_cols=continuous_cols,
			input_dim=64,
			n_heads=8,
			n_blocks=3,
			attn_dropout=0.05,
			ff_dropout=0.05,
		)
		saint_model = WideDeep(deeptabular=saint, pred_dim=5)
		state_dict = torch.load(saint_model_path, map_location=torch.device("cpu"))
		saint_model.load_state_dict(state_dict, strict=True)

		print(f"✓ SAINT loaded from {saint_model_path}")
		return make_torch_tabular_predictor(saint_model, saint_preprocessor)
	except Exception as exc:
		print(f"✗ Error loading SAINT: {exc}")
		return None


def build_meta_features(classifiers, X):
	proba_blocks = [model_predict_proba(clf, X) for _, clf in classifiers]
	return np.concatenate(proba_blocks, axis=1)


def summarize_model_usage(mlp_meta, classifiers, n_classes):
	"""Approximate contribution per base model from first-layer absolute weights."""
	first_layer = mlp_meta.coefs_[0]  # shape: (n_meta_features, hidden_dim)
	feature_strength = np.mean(np.abs(first_layer), axis=1)

	model_scores = []
	for model_idx, (name, _) in enumerate(classifiers):
		start = model_idx * n_classes
		end = start + n_classes
		score = feature_strength[start:end].mean()
		model_scores.append((name, score))

	total = sum(score for _, score in model_scores) + 1e-12
	return [(name, score, score / total) for name, score in model_scores]


def ablation_impact(mlp_meta, classifiers, X, y):
	"""Measure performance drop when each model's probability block is neutralized."""
	proba_blocks = [model_predict_proba(clf, X) for _, clf in classifiers]
	n_classes = proba_blocks[0].shape[1]

	X_meta_full = np.concatenate(proba_blocks, axis=1)
	y_pred_full = mlp_meta.predict(X_meta_full)
	base_acc = accuracy_score(y, y_pred_full)
	base_f1 = f1_score(y, y_pred_full, average="macro")

	results = []
	uniform = np.full((X.shape[0], n_classes), 1.0 / n_classes)
	for idx, (name, _) in enumerate(classifiers):
		ablated_blocks = [block.copy() for block in proba_blocks]
		ablated_blocks[idx] = uniform
		X_meta_ablate = np.concatenate(ablated_blocks, axis=1)
		y_pred_ablate = mlp_meta.predict(X_meta_ablate)
		acc = accuracy_score(y, y_pred_ablate)
		f1m = f1_score(y, y_pred_ablate, average="macro")
		results.append((name, base_acc - acc, base_f1 - f1m))

	return base_acc, base_f1, results