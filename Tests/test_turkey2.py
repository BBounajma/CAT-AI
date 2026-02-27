import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, log_loss

# Ensure Ensemble utilities are importable when running from Tests/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENSEMBLE_DIR = os.path.join(ROOT, "Ensemble")
if ENSEMBLE_DIR not in sys.path:
    sys.path.insert(0, ENSEMBLE_DIR)

from utils_stacking import (
    model_predict,
    model_predict_proba,
    build_meta_features,
    load_saint_predictor,
)

try:
    from Ensemble.MLP_stacking_saint_tabnet import load_tabnet_predictor
except Exception:
    try:
        from MLP_stacking_saint_tabnet import load_tabnet_predictor
    except Exception:
        load_tabnet_predictor = None


# ============================================================
# Load Models
# ============================================================

def load_models(models_dir):
    models = {}

    xgb_path = os.path.join(models_dir, "XG_boost", "xgb_classifier_model.joblib")
    rf_path = os.path.join(models_dir, "Random_Forest", "rf_classifier_model.joblib")
    cat_path = os.path.join(models_dir, "CatBoost", "cat_classifier_model.joblib")

    if os.path.exists(xgb_path):
        models["xgb"] = joblib.load(xgb_path)

    if os.path.exists(rf_path):
        models["rf"] = joblib.load(rf_path)

    if os.path.exists(cat_path):
        models["catboost"] = joblib.load(cat_path)

    saint = load_saint_predictor(models_dir)
    if saint is not None:
        models["saint"] = saint

    tabnet = None
    if load_tabnet_predictor is not None:
        tabnet = load_tabnet_predictor(models_dir)
    if tabnet is not None:
        models["tabnet"] = tabnet

    mlp_path = os.path.join(models_dir, "mlp_stacking_saint_tabnet_model.joblib")
    if os.path.exists(mlp_path):
        models["mlp_stack"] = joblib.load(mlp_path)

    return models


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(name, clf, X, y):
    proba = model_predict_proba(clf, X)
    pred = np.argmax(proba, axis=1)

    acc = accuracy_score(y, pred)
    f1m = f1_score(y, pred, average="macro")
    f1w = f1_score(y, pred, average="weighted")

    try:
        ll = log_loss(y, proba)
    except Exception:
        ll = None

    print(
        f"{name:12s} | "
        f"Acc: {acc:.4f} | "
        f"F1-m: {f1m:.4f} | "
        f"F1-w: {f1w:.4f} | "
        f"LogLoss: {ll}"
    )

    return {"acc": acc, "f1m": f1m, "f1w": f1w, "logloss": ll}


# ============================================================
# Main
# ============================================================

def main():

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "Data", "processed_Turkey_data.csv")
    models_dir = os.path.join(project_root, "Models")

    if not os.path.exists(data_path):
        raise RuntimeError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)

    # Encode categorical columns consistently
    cat_cols = ["foundation_type", "roof_type", "ground_floor_type"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category").cat.codes

    if "damage_grade" not in df.columns:
        raise RuntimeError("Target column 'damage_grade' not found")

    y = df["damage_grade"]
    X = df.drop(columns=["damage_grade"])

    # --------------------------------------------------------
    # Proper Train/Test Split
    # --------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    models = load_models(models_dir)
    print("Loaded models:", list(models.keys()))

    # --------------------------------------------------------
    # Evaluate Base Models on TEST SET ONLY
    # --------------------------------------------------------
    for k in ["xgb", "rf", "catboost", "saint", "tabnet"]:
        if k in models:
            print(f"Evaluating {k}...")
            evaluate_model(k, models[k], X_test, y_test)

    # --------------------------------------------------------
    # Weighted Average (SAINT + TabNet)
    # --------------------------------------------------------
    if "saint" in models and "tabnet" in models:
        print("Evaluating weighted average (SAINT+TabNet) — equal weights")

        p1 = model_predict_proba(models["saint"], X_test)
        p2 = model_predict_proba(models["tabnet"], X_test)

        pa = 0.5 * p1 + 0.5 * p2
        pred = np.argmax(pa, axis=1)

        acc = accuracy_score(y_test, pred)
        f1m = f1_score(y_test, pred, average="macro")
        f1w = f1_score(y_test, pred, average="weighted")

        try:
            ll = log_loss(y_test, pa)
        except Exception:
            ll = None

        print(
            f"weighted_avg | "
            f"Acc: {acc:.4f} | "
            f"F1-m: {f1m:.4f} | "
            f"F1-w: {f1w:.4f} | "
            f"LogLoss: {ll}"
        )

    # MLP stacking evaluation removed; only base learners are evaluated in this test.


if __name__ == "__main__":
    main()