import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, log_loss

# ------------------------------------------------------------------
# Import stacking utilities
# ------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENSEMBLE_DIR = os.path.join(ROOT, "Ensemble")
if ENSEMBLE_DIR not in sys.path:
    sys.path.insert(0, ENSEMBLE_DIR)

from utils_stacking import (
    build_meta_features,
    load_saint_predictor,
    make_torch_tabular_predictor,
    model_predict_proba,
)

try:
    from MLP_stacking_saint_tabnet import load_tabnet_predictor
except Exception:
    load_tabnet_predictor = None


# ------------------------------------------------------------------
# Load sklearn models
# ------------------------------------------------------------------

def load_sklearn_model(path):
    if os.path.exists(path):
        print(f"✓ Loaded {path}")
        return joblib.load(path)
    else:
        print(f"✗ Missing {path}")
        return None


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    project_root = ROOT
    models_dir = os.path.join(project_root, "Models")
    turkey_data_path = os.path.join(project_root, "Data", "processed_Turkey_data.csv")

    print("=" * 70)
    print("Testing MLP Stacking on Turkey Data")
    print("=" * 70)

    # --------------------------------------------------------------
    # Load Turkey dataset
    # --------------------------------------------------------------

    if not os.path.exists(turkey_data_path):
        raise RuntimeError(f"Turkey data not found: {turkey_data_path}")

    df = pd.read_csv(turkey_data_path)

    cat_cols = ["foundation_type", "roof_type", "ground_floor_type"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    if "damage_grade" not in df.columns:
        raise RuntimeError("Target column 'damage_grade' missing")

    y = df["damage_grade"]
    X = df.drop(columns=["damage_grade"])

    print(f"Turkey samples: {len(X)}")

    # --------------------------------------------------------------
    # Load base models
    # --------------------------------------------------------------

    print("\nLoading base models...")

    xgb = load_sklearn_model(
        os.path.join(models_dir, "XG_boost/xgb_classifier_model.joblib")
    )
    rf = load_sklearn_model(
        os.path.join(models_dir, "Random_Forest/rf_classifier_model.joblib")
    )
    cat = load_sklearn_model(
        os.path.join(models_dir, "CatBoost/cat_classifier_model.joblib")
    )

    saint = load_saint_predictor(models_dir)

    tabnet = None
    if load_tabnet_predictor is not None:
        tabnet = load_tabnet_predictor(models_dir)

    classifiers = []
    if xgb is not None:
        classifiers.append(("XGBoost", xgb))
    if rf is not None:
        classifiers.append(("Random Forest", rf))
    if cat is not None:
        classifiers.append(("CatBoost", cat))
    if saint is not None:
        classifiers.append(("SAINT", saint))
    if tabnet is not None:
        classifiers.append(("TabNet", tabnet))

    if len(classifiers) == 0:
        raise RuntimeError("No base models found")

    print(f"Using {len(classifiers)} base models")

    # --------------------------------------------------------------
    # Build meta-features
    # --------------------------------------------------------------

    print("\nBuilding meta-features from Turkey data...")
    X_meta = build_meta_features(classifiers, X)

    print(f"Meta-feature shape: {X_meta.shape}")

    # --------------------------------------------------------------
    # Load meta-learner
    # --------------------------------------------------------------

    meta_model_path = os.path.join(
        models_dir, "mlp_stacking_saint_tabnet_model.joblib"
    )
    meta_info_path = os.path.join(
        models_dir, "mlp_stacking_saint_tabnet_info.joblib"
    )

    if not os.path.exists(meta_model_path):
        raise RuntimeError("Saved meta-learner not found")

    mlp_meta = joblib.load(meta_model_path)
    print(f"✓ Loaded meta-learner")

    expected_dim = getattr(mlp_meta, "n_features_in_", None)

    if expected_dim is None and os.path.exists(meta_info_path):
        meta_info = joblib.load(meta_info_path)
        expected_dim = meta_info.get("meta_feature_dim")

    if expected_dim is None:
        raise RuntimeError("Could not determine expected meta feature dimension")

    if X_meta.shape[1] != expected_dim:
        raise RuntimeError(
            f"Meta feature mismatch: got {X_meta.shape[1]}, expected {expected_dim}"
        )

    # --------------------------------------------------------------
    # Evaluate stacking
    # --------------------------------------------------------------

    print("\nEvaluating MLP stacking on Turkey...")

    y_pred = mlp_meta.predict(X_meta)

    acc = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average="macro")
    f1_weighted = f1_score(y, y_pred, average="weighted")

    print("\n" + "=" * 70)
    print(f"Turkey Accuracy:     {acc:.4f}")
    print(f"Turkey F1-Macro:     {f1_macro:.4f}")
    print(f"Turkey F1-Weighted:  {f1_weighted:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()