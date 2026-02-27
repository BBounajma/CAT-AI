import os
import sys

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

try:
	from Ensemble.utils_stacking import (
		ablation_impact,
		build_meta_features,
		build_meta_features_oof,
		load_saint_predictor,
		model_predict,
		model_predict_proba,
		summarize_model_usage,
	)
except ModuleNotFoundError:
	from utils_stacking import (
		ablation_impact,
		build_meta_features,
		build_meta_features_oof,
		load_saint_predictor,
		model_predict,
		model_predict_proba,
		summarize_model_usage,
	)


def main():
	script_dir = os.path.dirname(os.path.abspath(__file__))
	project_root = os.path.dirname(script_dir)
	models_dir = os.path.join(project_root, "Models")
	data_path = os.path.join(project_root, "Data/processed_new_data2.csv")

	print("=" * 70)
	print("MLP Stacking: XGBoost + Random Forest + SAINT")
	print("=" * 70)

	print("\nLoading data...")
	df = pd.read_csv(data_path)
	multi_class_cat_cols = [
		"foundation_type",
		"roof_type",
		"ground_floor_type",
	]
	for col in multi_class_cat_cols:
		if col in df.columns:
			df[col] = df[col].astype("category").cat.codes

	y = df["damage_grade"]
	X = df.drop(columns=["damage_grade"])

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.3, random_state=42, stratify=y
	)
	X_train, X_val, y_train, y_val = train_test_split(
		X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
	)

	print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

	print("\nLoading trained models...")

	xgb_path = os.path.join(models_dir, "XG_boost/xgb_classifier_model.joblib")
	if os.path.exists(xgb_path):
		xgb_model = joblib.load(xgb_path)
		print(f"✓ XGBoost loaded from {xgb_path}")
	else:
		print(f"✗ XGBoost model not found at {xgb_path}")
		print("  Run: python Trainings/train_XGboost.py")
		xgb_model = None

	rf_path = os.path.join(models_dir, "Random_Forest/rf_classifier_model.joblib")
	if os.path.exists(rf_path):
		rf_model = joblib.load(rf_path)
		print(f"✓ Random Forest loaded from {rf_path}")
	else:
		print(f"✗ Random Forest model not found at {rf_path}")
		print("  Run: python Trainings/train_rf.py")
		rf_model = None

	catboost_path = os.path.join(models_dir, "CatBoost/cat_classifier_model.joblib")
	if os.path.exists(catboost_path):
		cat_model = joblib.load(catboost_path)
		print(f"✓ CatBoost loaded from {catboost_path}")
	else:
		cat_model = None

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

	X_meta_source = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
	y_meta_source = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

	print("\nBuilding OOF meta-features on train+val split...")
	X_meta_train, stacked_classifiers = build_meta_features_oof(
		classifiers,
		X_meta_source,
		y_meta_source,
		n_splits=5,
		random_state=42,
	)
	X_meta_test = build_meta_features(stacked_classifiers, X_test)

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

	print("Training MLP meta-learner on OOF predictions...")
	mlp_meta.fit(X_meta_train, y_meta_source)

	y_pred_stack = mlp_meta.predict(X_meta_test)
	acc = accuracy_score(y_test, y_pred_stack)
	f1_macro = f1_score(y_test, y_pred_stack, average="macro")
	f1_weighted = f1_score(y_test, y_pred_stack, average="weighted")

	print(f"\nStacking Test Accuracy:    {acc:.4f}")
	print(f"Stacking Test F1-Macro:    {f1_macro:.4f}")
	print(f"Stacking Test F1-Weighted: {f1_weighted:.4f}")

	n_classes = model_predict_proba(stacked_classifiers[0][1], X_test.iloc[:1]).shape[1]

	print("\n" + "=" * 70)
	print("MLP Usage by Base Model (first-layer weight strength)")
	print("=" * 70)
	for name, raw_score, pct in summarize_model_usage(mlp_meta, stacked_classifiers, n_classes):
		print(f"{name:20s} | strength: {raw_score:.6f} | share: {pct*100:6.2f}%")

	print("\n" + "=" * 70)
	print("Model Ablation Impact on Stacking (drop after neutralization)")
	print("=" * 70)
	base_acc, base_f1, ablation = ablation_impact(mlp_meta, stacked_classifiers, X_test, y_test)
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


if __name__ == "__main__":
	main()
