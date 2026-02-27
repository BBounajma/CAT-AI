import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, log_loss

# Ensure Ensemble utilities are importable when running from Tests/
import sys
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
    # fallback: try local module
    try:
        from MLP_stacking_saint_tabnet import load_tabnet_predictor
    except Exception:
        load_tabnet_predictor = None


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

    # torch predictors
    saint = load_saint_predictor(models_dir)
    if saint is not None:
        models["saint"] = saint

    tabnet = None
    if load_tabnet_predictor is not None:
        tabnet = load_tabnet_predictor(models_dir)
    if tabnet is not None:
        models["tabnet"] = tabnet

    # mlp stacking meta-learner (if saved)
    mlp_path = os.path.join(models_dir, "mlp_stacking_saint_tabnet_model.joblib")
    if os.path.exists(mlp_path):
        models["mlp_stack"] = joblib.load(mlp_path)

    return models


def evaluate_model(name, clf, X, y):
    try:
        proba = model_predict_proba(clf, X)
    except Exception:
        proba = model_predict_proba(clf, X)
    pred = np.argmax(proba, axis=1)
    acc = accuracy_score(y, pred)
    f1m = f1_score(y, pred, average="macro")
    f1w = f1_score(y, pred, average="weighted")
    ll = None
    try:
        ll = log_loss(y, proba)
    except Exception:
        ll = None
    print(f"{name:12s} | Acc: {acc:.4f} | F1-m: {f1m:.4f} | F1-w: {f1w:.4f} | LogLoss: {ll}")
    return {"acc": acc, "f1m": f1m, "f1w": f1w, "logloss": ll}


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "Data", "processed_Turkey_data.csv")
    models_dir = os.path.join(project_root, "Models")

    if not os.path.exists(data_path):
        raise RuntimeError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    cat_cols = ["foundation_type", "roof_type", "ground_floor_type"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category").cat.codes

    if "damage_grade" not in df.columns:
        raise RuntimeError("Target column 'damage_grade' not found in data")

    y = df["damage_grade"]
    X = df.drop(columns=["damage_grade"])

    models = load_models(models_dir)
    print("Loaded models:", list(models.keys()))

    results = {}

    # Evaluate individual models
    for k in ["xgb", "rf", "catboost", "saint", "tabnet"]:
        if k in models:
            print(f"Evaluating {k}...")
            results[k] = evaluate_model(k, models[k], X, y)

    # Weighted average SAINT+TabNet
    if "saint" in models and "tabnet" in models:
        print("Evaluating weighted average (SAINT+TabNet) — equal weights")
        p1 = model_predict_proba(models["saint"], X)
        p2 = model_predict_proba(models["tabnet"], X)
        pa = 0.5 * p1 + 0.5 * p2
        pred = np.argmax(pa, axis=1)
        acc = accuracy_score(y, pred)
        f1m = f1_score(y, pred, average="macro")
        f1w = f1_score(y, pred, average="weighted")
        ll = None
        try:
            ll = log_loss(y, pa)
        except Exception:
            pass
        print(f"weighted_avg | Acc: {acc:.4f} | F1-m: {f1m:.4f} | F1-w: {f1w:.4f} | LogLoss: {ll}")

    # MLP stacking meta-learner (if available)
    if "mlp_stack" in models:
        print("Evaluating MLP stacking meta-learner (SAINT+TabNet + sklearns)...")
        # build sklearn stack list (use loaded xgb and rf if present)
        sklearn_stack = []
        for name in ["XGBoost", "Random Forest"]:
            if name == "XGBoost" and "xgb" in models:
                sklearn_stack.append((name, models["xgb"]))
            if name == "Random Forest" and "rf" in models:
                sklearn_stack.append((name, models["rf"]))

        torch_preds = []
        if "saint" in models:
            torch_preds.append(("SAINT", models["saint"]))
        if "tabnet" in models:
            torch_preds.append(("TabNet", models["tabnet"]))

        if len(sklearn_stack) > 0:
            X_meta_test_sklearn = build_meta_features(sklearn_stack, X)
        else:
            X_meta_test_sklearn = np.empty((len(X), 0))

        torch_blocks = [model_predict_proba(p, X) for _, p in torch_preds]
        if len(torch_blocks) > 0:
            X_meta_test = np.hstack([X_meta_test_sklearn] + torch_blocks)
        else:
            X_meta_test = X_meta_test_sklearn

        y_pred = models["mlp_stack"].predict(X_meta_test)
        acc = accuracy_score(y, y_pred)
        f1m = f1_score(y, y_pred, average="macro")
        f1w = f1_score(y, y_pred, average="weighted")
        print(f"mlp_stack   | Acc: {acc:.4f} | F1-m: {f1m:.4f} | F1-w: {f1w:.4f}")


if __name__ == "__main__":
    main()
