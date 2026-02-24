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


if __name__ == "__main__":
	script_dir = os.path.dirname(os.path.abspath(__file__))
	project_root = os.path.dirname(script_dir)
	models_dir = os.path.join(project_root, "Models")
	data_path = os.path.join(project_root, "Data/processed_new_data2.csv")

	print("=" * 70)
	print("MLP Stacking: XGBoost + Random Forest + SAINT")
	print("=" * 70)

	print("\nLoading data...")
	df = pd.read_csv(data_path)
	y = df["damage_grade"]
	X = df.drop(columns=["damage_grade"])

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.3, stratify=y, random_state=42
	)
	X_train, X_val, y_train, y_val = train_test_split(
		X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
	)

	print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

	print("\nLoading trained models...")

	xgb_path = os.path.join(models_dir, "xgb_classifier_model.joblib")
	if os.path.exists(xgb_path):
		xgb_model = joblib.load(xgb_path)
		print(f"✓ XGBoost loaded from {xgb_path}")
	else:
		print(f"✗ XGBoost model not found at {xgb_path}")
		print("  Run: python Trainings/train_XGboost.py")
		xgb_model = None

	rf_path = os.path.join(models_dir, "rf_classifier_model.joblib")
	if os.path.exists(rf_path):
		rf_model = joblib.load(rf_path)
		print(f"✓ Random Forest loaded from {rf_path}")
	else:
		print(f"✗ Random Forest model not found at {rf_path}")
		print("  Run: python Trainings/train_rf.py")
		rf_model = None

	saint_predictor = load_saint_predictor(models_dir)

	classifiers = []
	if xgb_model is not None:
		classifiers.append(("XGBoost", xgb_model))
	if rf_model is not None:
		classifiers.append(("Random Forest", rf_model))
	if saint_predictor is not None:
		classifiers.append(("SAINT", saint_predictor))

	if len(classifiers) < 2:
		print("\n✗ Need at least 2 trained base models for stacking.")
		sys.exit(1)

	print(f"\nUsing {len(classifiers)} models for stacking")

	print("\n" + "=" * 70)
	print("Individual Model Performance on Test Set")
	print("=" * 70)
	for name, clf in classifiers:
		y_pred = model_predict(clf, X_test)
		acc = accuracy_score(y_test, y_pred)
		f1_macro = f1_score(y_test, y_pred, average="macro")
		f1_weighted = f1_score(y_test, y_pred, average="weighted")
		print(
			f"{name:20s} | Acc: {acc:.4f} | F1-Macro: {f1_macro:.4f} | F1-Weighted: {f1_weighted:.4f}"
		)

	print("\n" + "=" * 70)
	print("MLP Stacking Performance")
	print("=" * 70)

	print("\nBuilding meta-features...")
	X_meta_train = build_meta_features(classifiers, X_val)
	X_meta_test = build_meta_features(classifiers, X_test)

	mlp_meta = MLPClassifier(
		hidden_layer_sizes=(64, 32),
		activation="relu",
		solver="adam",
		alpha=1e-4,
		batch_size=64,
		learning_rate_init=1e-3,
		max_iter=300,
		early_stopping=True,
		n_iter_no_change=15,
		random_state=42,
	)

	print("Training MLP meta-learner on validation predictions...")
	mlp_meta.fit(X_meta_train, y_val)

	y_pred_stack = mlp_meta.predict(X_meta_test)
	acc = accuracy_score(y_test, y_pred_stack)
	f1_macro = f1_score(y_test, y_pred_stack, average="macro")
	f1_weighted = f1_score(y_test, y_pred_stack, average="weighted")

	print(f"\nStacking Test Accuracy:    {acc:.4f}")
	print(f"Stacking Test F1-Macro:    {f1_macro:.4f}")
	print(f"Stacking Test F1-Weighted: {f1_weighted:.4f}")

	n_classes = model_predict_proba(classifiers[0][1], X_test.iloc[:1]).shape[1]

	print("\n" + "=" * 70)
	print("MLP Usage by Base Model (first-layer weight strength)")
	print("=" * 70)
	for name, raw_score, pct in summarize_model_usage(mlp_meta, classifiers, n_classes):
		print(f"{name:20s} | strength: {raw_score:.6f} | share: {pct*100:6.2f}%")

	print("\n" + "=" * 70)
	print("Model Ablation Impact on Stacking (drop after neutralization)")
	print("=" * 70)
	base_acc, base_f1, ablation = ablation_impact(mlp_meta, classifiers, X_test, y_test)
	print(f"Baseline stacking -> Acc: {base_acc:.4f} | F1-Macro: {base_f1:.4f}")
	for name, d_acc, d_f1 in ablation:
		print(f"{name:20s} | ΔAcc: {d_acc:+.4f} | ΔF1-Macro: {d_f1:+.4f}")

	meta_model_path = os.path.join(models_dir, "mlp_stacking_model.joblib")
	meta_info_path = os.path.join(models_dir, "mlp_stacking_info.joblib")
	joblib.dump(mlp_meta, meta_model_path)
	joblib.dump(
		{
			"base_models": [name for name, _ in classifiers],
			"meta_feature_dim": X_meta_train.shape[1],
			"hidden_layer_sizes": (64, 32),
		},
		meta_info_path,
	)
	print(f"\n✓ Saved meta-learner: {meta_model_path}")
	print(f"✓ Saved metadata:     {meta_info_path}")

	print("\n" + "=" * 70)
	print("Done!")
	print("=" * 70)