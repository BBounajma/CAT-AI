import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
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

	def _extract_logits(output):
		if isinstance(output, (tuple, list)):
			return output[0]
		if isinstance(output, dict):
			for key in ("logits", "preds", "y_pred"):
				if key in output:
					return output[key]
		return output

	def predict_proba(X):
		X_processed = preprocessor.transform(X)
		X_tensor = torch.tensor(X_processed, dtype=torch.float32)

		prob_chunks = []
		with torch.no_grad():
			for start in range(0, X_tensor.shape[0], batch_size):
				end = start + batch_size
				batch = X_tensor[start:end].to(device)
				try:
					model_output = model({"deeptabular": batch})
				except KeyError:
					model_output = model({"X_tab": batch})
				logits = _extract_logits(model_output)
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
	return _build_enhanced_meta_features(proba_blocks)


def _build_enhanced_meta_features(proba_blocks, eps=1e-12):
	"""
	Build richer meta-features from base-model class probabilities.

	Feature groups:
	1) Raw probability blocks for each model (backward-compatible prefix)
	2) Per-model confidence stats: max prob, margin(top1-top2), entropy
	3) Cross-model average probabilities and uncertainty stats
	"""
	if len(proba_blocks) == 0:
		raise ValueError("proba_blocks cannot be empty")

	base = np.concatenate(proba_blocks, axis=1)
	aux_blocks = []

	for probs in proba_blocks:
		sorted_probs = np.sort(probs, axis=1)
		top1 = sorted_probs[:, -1]
		top2 = sorted_probs[:, -2] if probs.shape[1] > 1 else np.zeros_like(top1)
		margin = top1 - top2
		entropy = -np.sum(np.clip(probs, eps, 1.0) * np.log(np.clip(probs, eps, 1.0)), axis=1)
		aux_blocks.append(np.column_stack([top1, margin, entropy]))

	mean_probs = np.mean(np.stack(proba_blocks, axis=0), axis=0)
	sorted_mean = np.sort(mean_probs, axis=1)
	mean_top1 = sorted_mean[:, -1]
	mean_top2 = sorted_mean[:, -2] if mean_probs.shape[1] > 1 else np.zeros_like(mean_top1)
	mean_margin = mean_top1 - mean_top2
	mean_entropy = -np.sum(
		np.clip(mean_probs, eps, 1.0) * np.log(np.clip(mean_probs, eps, 1.0)), axis=1
	)

	aux = np.concatenate(aux_blocks + [mean_probs, np.column_stack([mean_top1, mean_margin, mean_entropy])], axis=1)
	return np.concatenate([base, aux], axis=1)


def build_meta_features_oof(classifiers, X, y, n_splits=5, random_state=42):
	"""
	Build OOF meta-features when base estimators are trainable.

	For sklearn-like models (fit + predict_proba), generates proper OOF probabilities.
	For pre-fitted predictors (dict wrappers, e.g. torch models), falls back to direct
	predict_proba on X and keeps the original predictor for test-time meta-features.
	"""
	skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
	meta_blocks = []
	fitted_classifiers = []

	for name, clf in classifiers:
		if hasattr(clf, "fit") and hasattr(clf, "predict_proba"):
			n_classes = model_predict_proba(clf, X.iloc[:1]).shape[1]
			oof = np.zeros((len(X), n_classes), dtype=np.float64)

			for train_idx, val_idx in skf.split(X, y):
				X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
				y_tr = y.iloc[train_idx]

				model_fold = clone(clf)
				model_fold.fit(X_tr, y_tr)
				oof[val_idx] = model_fold.predict_proba(X_val)

			model_full = clone(clf)
			model_full.fit(X, y)
			fitted_classifiers.append((name, model_full))
			meta_blocks.append(oof)
		else:
			print(f"[OOF fallback] {name}: using direct predictions (model is not fit-capable).")
			meta_blocks.append(model_predict_proba(clf, X))
			fitted_classifiers.append((name, clf))

	X_meta = np.concatenate(meta_blocks, axis=1)
	X_meta = _build_enhanced_meta_features(meta_blocks)
	return X_meta, fitted_classifiers


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

	X_meta_full = _build_enhanced_meta_features(proba_blocks)
	y_pred_full = mlp_meta.predict(X_meta_full)
	base_acc = accuracy_score(y, y_pred_full)
	base_f1 = f1_score(y, y_pred_full, average="macro")

	results = []
	uniform = np.full((X.shape[0], n_classes), 1.0 / n_classes)
	for idx, (name, _) in enumerate(classifiers):
		ablated_blocks = [block.copy() for block in proba_blocks]
		ablated_blocks[idx] = uniform
		X_meta_ablate = _build_enhanced_meta_features(ablated_blocks)
		y_pred_ablate = mlp_meta.predict(X_meta_ablate)
		acc = accuracy_score(y, y_pred_ablate)
		f1m = f1_score(y, y_pred_ablate, average="macro")
		results.append((name, base_acc - acc, base_f1 - f1m))

	return base_acc, base_f1, results