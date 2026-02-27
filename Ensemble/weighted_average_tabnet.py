import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from pytorch_widedeep.models import TabNet, WideDeep

try:
	from Ensemble.weighted_average import WeightedEnsembleLearner, StackingMetaEnsemble
except ModuleNotFoundError:
	from weighted_average import WeightedEnsembleLearner, StackingMetaEnsemble


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


def load_tabnet_predictor(models_dir):
	tabnet_dir = os.path.join(models_dir, "TabNet")
	tabnet_model_path = os.path.join(tabnet_dir, "model_state_dict.pt")
	tabnet_preprocessor_path = os.path.join(tabnet_dir, "tab_preprocessor.joblib")

	if not (os.path.exists(tabnet_model_path) and os.path.exists(tabnet_preprocessor_path)):
		print("✗ TabNet model not found")
		print(f"  Expected files: {tabnet_model_path} and {tabnet_preprocessor_path}")
		print("  Run: python3 Trainings/train_tabnet.py")
		return None

	try:
		tab_preprocessor = joblib.load(tabnet_preprocessor_path)
		continuous_cols = getattr(
			tab_preprocessor,
			"continuous_cols",
			[
				"PGA_g",
				"count_floors_pre_eq",
				"age_building",
				"plinth_area_sq_ft",
				"per-height_ft_pre_eq",
			],
		)

		tabnet = TabNet(
			column_idx=tab_preprocessor.column_idx,
			cat_embed_input=tab_preprocessor.cat_embed_input,
			continuous_cols=continuous_cols,
			n_steps=7,
			step_dim=128,
			attn_dim=128,
			dropout=0.2,
			n_glu_step_dependent=2,
			n_glu_shared=2,
			gamma=1.3,
			epsilon=1e-15,
			ghost_bn=True,
			virtual_batch_size=128,
			momentum=0.02,
			mask_type="sparsemax",
		)
		tabnet_model = WideDeep(deeptabular=tabnet, pred_dim=5)
		state_dict = torch.load(tabnet_model_path, map_location=torch.device("cpu"))
		tabnet_model.load_state_dict(state_dict, strict=True)
		tabnet_model.eval()

			print(f"✓ TabNet loaded from {tabnet_model_path}")
			predictor = make_torch_tabular_predictor(tabnet_model, tab_preprocessor)
			tabnet_oof_path = os.path.join(tabnet_dir, "tabnet_oof_preds.npy")
			if os.path.exists(tabnet_oof_path):
				try:
					predictor["oof"] = np.load(tabnet_oof_path)
				except Exception:
					pass
			return predictor
	except Exception as exc:
		print(f"✗ Error loading TabNet: {exc}")
		return None


	catboost_path = os.path.join(models_dir, 'CatBoost/cat_classifier_model.joblib')
	if os.path.exists(catboost_path):
		cat_model = joblib.load(catboost_path)
		print(f"✓ CatBoost loaded from {catboost_path}")
	else:
		cat_model = None

def main():
	script_dir = os.path.dirname(os.path.abspath(__file__))
	project_root = os.path.dirname(script_dir)
	models_dir = os.path.join(project_root, 'Models')
	data_path = os.path.join(project_root, 'Data/processed_new_data2.csv')

	print("=" * 70)
	print("Weighted Ensemble Learner: XGBoost + Random Forest + TabNet")
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

	tabnet_predictor = load_tabnet_predictor(models_dir)

	classifiers = []
	if xgb_model is not None:
		classifiers.append(('XGBoost', xgb_model))
	if rf_model is not None:
		classifiers.append(('Random Forest', rf_model))
	if tabnet_predictor is not None:
		classifiers.append(('TabNet', tabnet_predictor))

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
	print("Weighted Ensemble Performance")
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

	print("\n" + "=" * 70)
	print("Stacking Meta-Ensemble (Class-Aware)")
	print("=" * 70)

	stacker = StackingMetaEnsemble(
		base_models=classifiers,
		n_classes=5,
		n_folds=5,
		random_state=42,
	)

	stacker.fit(X_train, y_train)
	stacker.evaluate(X_test, y_test, name="Test")

	print("\n" + "=" * 70)
	print("Done!")
	print("=" * 70)


if __name__ == "__main__":
	main()
