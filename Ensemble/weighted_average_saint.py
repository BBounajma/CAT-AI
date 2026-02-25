import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from pytorch_widedeep.models import SAINT, WideDeep

try:
	from Ensemble.weighted_average import WeightedEnsembleLearner
except ModuleNotFoundError:
	from weighted_average import WeightedEnsembleLearner


def make_torch_tabular_predictor(model, preprocessor, device='cpu', batch_size=256):
	"""Return minimal predict/predict_proba callables."""
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

	return {'predict_proba': predict_proba, 'predict': predict}


def main():
	script_dir = os.path.dirname(os.path.abspath(__file__))
	project_root = os.path.dirname(script_dir)
	models_dir = os.path.join(project_root, 'Models')
	data_path = os.path.join(project_root, 'Data/processed_new_data2.csv')

	print("=" * 70)
	print("Weighted Ensemble Learner: XGBoost + Random Forest + SAINT")
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

	xgb_path = os.path.join(models_dir, 'XG_boost/xgb_classifier_model.joblib')
	if os.path.exists(xgb_path):
		xgb_model = joblib.load(xgb_path)
		print(f"✓ XGBoost loaded from {xgb_path}")
	else:
		print(f"✗ XGBoost model not found at {xgb_path}")
		print("  Run: python Trainings/train_XGboost.py")
		xgb_model = None

	rf_path = os.path.join(models_dir, 'Random_Forest/rf_classifier_model.joblib')
	if os.path.exists(rf_path):
		rf_model = joblib.load(rf_path)
		print(f"✓ Random Forest loaded from {rf_path}")
	else:
		print(f"✗ Random Forest model not found at {rf_path}")
		print("  Run: python Trainings/train_rf.py")
		rf_model = None

	saint_dir = os.path.join(models_dir, 'Saint')
	saint_model_path = os.path.join(saint_dir, 'model_state_dict.pt')
	saint_preprocessor_path = os.path.join(saint_dir, 'tab_preprocessor.joblib')
	saint_config_path = os.path.join(saint_dir, 'config.joblib')

	saint_predictor = None
	if os.path.exists(saint_model_path) and os.path.exists(saint_preprocessor_path):
		try:
			saint_preprocessor = joblib.load(saint_preprocessor_path)
			saint_config = (
				joblib.load(saint_config_path) if os.path.exists(saint_config_path) else {}
			)
			features = saint_config.get('features', {})
			continuous_cols = features.get(
				'all_continuous_cols',
				getattr(saint_preprocessor, 'continuous_cols', []),
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

			state_dict = torch.load(saint_model_path, map_location=torch.device('cpu'))
			saint_model.load_state_dict(state_dict, strict=True)
			saint_model.eval()

			saint_predictor = make_torch_tabular_predictor(saint_model, saint_preprocessor)
			print(f"✓ SAINT loaded from {saint_model_path}")
		except Exception as e:
			print(f"✗ Error loading SAINT: {e}")
	else:
		print("✗ SAINT model not found")
		print(f"  Expected files: {saint_model_path} and {saint_preprocessor_path}")
		print("  Run: python3 Trainings/train_saint.py")

	classifiers = []
	if xgb_model is not None:
		classifiers.append(('XGBoost', xgb_model))
	if rf_model is not None:
		classifiers.append(('Random Forest', rf_model))
	if saint_predictor is not None:
		classifiers.append(('SAINT', saint_predictor))

	if len(classifiers) == 0:
		print("\n✗ No models available. Please train models first.")
		sys.exit(1)

	print(f"\nUsing {len(classifiers)} models for ensemble")

	print("\n" + "=" * 70)
	print("Individual Model Performance on Test Set")
	print("=" * 70)
	for name, clf in classifiers:
		if hasattr(clf, 'predict'):
			y_pred = clf.predict(X_test)
		else:
			y_pred = clf['predict'](X_test)
		acc = accuracy_score(y_test, y_pred)
		f1_macro = f1_score(y_test, y_pred, average='macro')
		f1_weighted = f1_score(y_test, y_pred, average='weighted')
		print(f"{name:20s} | Acc: {acc:.4f} | F1-Macro: {f1_macro:.4f} | F1-Weighted: {f1_weighted:.4f}")

	print("\n" + "=" * 70)
	print("Ensemble Performance")
	print("=" * 70)

	for method in ['uniform', 'grid_search', 'optimization']:
		print(f"\n{'-' * 70}")
		print(f"Method: {method.upper()}")
		print(f"{'-' * 70}")

		ensemble = WeightedEnsembleLearner(
			classifiers,
			method=method,
			metric='f1',
			use_proba=True,
			random_state=42,
		)

		ensemble.fit(X_train, y_train, X_val, y_val)

		y_pred_ensemble = ensemble.predict(X_test)
		acc = accuracy_score(y_test, y_pred_ensemble)
		f1_macro = f1_score(y_test, y_pred_ensemble, average='macro')
		f1_weighted = f1_score(y_test, y_pred_ensemble, average='weighted')

		print(f"\nTest Accuracy:       {acc:.4f}")
		print(f"Test F1-Macro:       {f1_macro:.4f}")
		print(f"Test F1-Weighted:    {f1_weighted:.4f}")
		print("\nLearned Weights:")
		for model_name, weight in ensemble.get_weights().items():
			print(f"  {model_name:20s}: {weight:.4f}")

	print("\n" + "=" * 70)
	print("Done!")
	print("=" * 70)


if __name__ == "__main__":
	main()
