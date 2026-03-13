import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# determine project root so script can be run from repository root
# prefer current working directory when it looks like the repo root
cwd = os.getcwd()
candidate_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(cwd, "Ensemble")) and os.path.isdir(os.path.join(cwd, "Models")):
    PROJECT_ROOT = cwd
else:
    PROJECT_ROOT = candidate_root

# ensure project root is importable so `Ensemble` pack
# age can be imported
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Ensemble.utils_stacking import (
    model_predict,
    model_predict_proba,
    load_saint_predictor,
    make_torch_tabular_predictor,
    build_meta_features,
)


class SklearnFeatureSafeWrapper:
    """Wrap a sklearn estimator to ensure input DataFrame columns match
    the estimator's `feature_names_in_` during predict/predict_proba calls.
    Missing columns are filled with zeros and extra columns are dropped.
    """

    def __init__(self, clf):
        self.clf = clf
        self.expected = getattr(clf, "feature_names_in_", None)

    def _align(self, X):
        # If we don't have expected names, pass through
        if self.expected is None:
            return X
        # ensure X is a DataFrame
        if not hasattr(X, "reindex"):
            # accept numpy arrays — assume order matches
            return X
        X2 = X.reindex(columns=self.expected)
        # fill missing cols with zeros
        X2 = X2.fillna(0)
        return X2

    def predict(self, X):
        return self.clf.predict(self._align(X))

    def predict_proba(self, X):
        return self.clf.predict_proba(self._align(X))


try:
    # Weighted/stacking utilities
    from weighted_average import WeightedEnsembleLearner, StackingMetaEnsemble
except Exception:
    from Ensemble.weighted_average import WeightedEnsembleLearner, StackingMetaEnsemble

try:
    from logreg_stacking_saint_tabnet import load_tabnet_predictor
except Exception:
    # fallback to weighted_average_saint_tabnet loader
    try:
        from weighted_average_saint_tabnet import load_tabnet_predictor
    except Exception:
        load_tabnet_predictor = None


def first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def compute_metrics(y_true, y_pred):
    acc = float(accuracy_score(y_true, y_pred))
    f1m = float(f1_score(y_true, y_pred, average="macro"))
    f1w = float(f1_score(y_true, y_pred, average="weighted"))
    return {"accuracy": acc, "f1_macro": f1m, "f1_weighted": f1w}


def save_cm(results_dir, prefix, labels, cm):
    path = os.path.join(results_dir, f"{prefix}_confusion_matrix.csv")
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(path)
    return path


def main():
    project_root = PROJECT_ROOT
    models_dir = os.path.join(project_root, "Models")
    data_path = os.path.join(project_root, "Data", "processed_Turkey_data.csv")

    if not os.path.exists(data_path):
        raise RuntimeError(f"Turkey data not found: {data_path}")

    df = pd.read_csv(data_path)
    cat_cols = ["foundation_type", "roof_type", "ground_floor_type"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category").cat.codes

    if "damage_grade" not in df.columns:
        raise RuntimeError("Target column 'damage_grade' missing")
   
    n_before = len(df)
    df = df.dropna(subset=["damage_grade"]).reset_index(drop=True)
    n_after = len(df)
    if n_after < n_before:
        print(f"Warning: dropped {n_before - n_after} rows with missing 'damage_grade'")

    y = df["damage_grade"]
    X = df.drop(columns=["damage_grade"])

    # split turkey set into train/val/test (reuse same ratios used elsewhere)
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.3, stratify=y_train, random_state=42
    )

    results = {"individual": {}, "ensembles": {}, "stacking": {}}
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    # -------------------------
    # Load base models
    # -------------------------
    models = {}
    # XGBoost (several possible locations)
    # prefer model files directly in Models/ (root) before legacy subfolders
    xgb_candidates = [
        os.path.join(models_dir, "xgb_classifier_model.joblib"),
        os.path.join(models_dir, "XG_boost", "xgb_classifier_model.joblib"),
    ]
    xgb_path = first_existing(*xgb_candidates)
    if xgb_path:
        try:
            clf = joblib.load(xgb_path)
            models["XGBoost"] = SklearnFeatureSafeWrapper(clf) if hasattr(clf, "feature_names_in_") else clf
        except Exception:
            models["XGBoost"] = None

    # Random Forest
    rf_path = first_existing(os.path.join(models_dir, "rf_classifier_model.joblib"), os.path.join(models_dir, "Random_Forest", "rf_classifier_model.joblib"))
    if rf_path:
        try:
            rf_clf = joblib.load(rf_path)
        except Exception as exc:
            # try to recover from a grid-search artifact that may contain the estimator
            gs_candidates = [
                os.path.join(models_dir, "rf_grid_search.joblib"),
                os.path.join(models_dir, "Random_Forest", "rf_grid_search.joblib"),
            ]
            gs_path = first_existing(*gs_candidates)
            rf_clf = None
            if gs_path:
                try:
                    gs_obj = joblib.load(gs_path)
                    if hasattr(gs_obj, "best_estimator_"):
                        rf_clf = gs_obj.best_estimator_
                    elif isinstance(gs_obj, dict) and "best_estimator_" in gs_obj:
                        rf_clf = gs_obj["best_estimator_"]
                except Exception:
                    rf_clf = None
            if rf_clf is None:
                results.setdefault("load_errors", {})["Random Forest"] = str(exc)
        if rf_clf is not None:
            models["Random Forest"] = SklearnFeatureSafeWrapper(rf_clf) if hasattr(rf_clf, "feature_names_in_") else rf_clf

    # CatBoost
    cat_path = first_existing(os.path.join(models_dir, "cat_classifier_model.joblib"), os.path.join(models_dir, "CatBoost", "cat_classifier_model.joblib"))
    if cat_path:
        cat_clf = joblib.load(cat_path)
        models["CatBoost"] = SklearnFeatureSafeWrapper(cat_clf) if hasattr(cat_clf, "feature_names_in_") else cat_clf

    # SAINT predictor (torch wrapper)
    saint_pred = load_saint_predictor(models_dir)
    if saint_pred is not None:
        models["SAINT"] = saint_pred

    # TabNet predictor
    tabnet_pred = None
    if load_tabnet_predictor is not None:
        try:
            tabnet_pred = load_tabnet_predictor(models_dir)
        except Exception:
            tabnet_pred = None
    if tabnet_pred is not None:
        models["TabNet"] = tabnet_pred

    # -------------------------
    # Evaluate individual models on Turkey test set
    # -------------------------
    individual_cms = {}
    for name, clf in models.items():
        try:
            if hasattr(clf, "predict"):
                y_pred = clf.predict(X_test)
            else:
                y_pred = model_predict(clf, X_test)
            metrics = compute_metrics(y_test, y_pred)
            results["individual"][name] = metrics
            try:
                labels = np.unique(np.concatenate([y_test.values, y_pred]))
                cm = confusion_matrix(y_test, y_pred, labels=labels)
                individual_cms[name] = (labels, cm)
            except Exception:
                individual_cms[name] = None
        except Exception as exc:
            results["individual"][name] = {"error": str(exc)}

    # Also evaluate saint_oof / tabnet_oof files if present (direct argmax)
    saint_oof_path = os.path.join(models_dir, "Saint", "saint_oof_preds.npy")
    if os.path.exists(saint_oof_path):
        try:
            arr = np.load(saint_oof_path)
            if arr.shape[0] == len(df):
                preds = np.argmax(arr, axis=1)
                m = compute_metrics(y, preds)
                results["individual"]["SAINT_OOF"] = m
            else:
                results["individual"]["SAINT_OOF"] = {"note": "OOF shape mismatch"}
        except Exception as exc:
            results["individual"]["SAINT_OOF"] = {"error": str(exc)}

    tabnet_oof_path = os.path.join(models_dir, "TabNet", "tabnet_oof_preds.npy")
    if os.path.exists(tabnet_oof_path):
        try:
            arr = np.load(tabnet_oof_path)
            if arr.shape[0] == len(df):
                preds = np.argmax(arr, axis=1)
                m = compute_metrics(y, preds)
                results["individual"]["TabNet_OOF"] = m
            else:
                results["individual"]["TabNet_OOF"] = {"note": "OOF shape mismatch"}
        except Exception as exc:
            results["individual"]["TabNet_OOF"] = {"error": str(exc)}

    # -------------------------
    # MLP stacking (if pretrained mlp exists). Build meta-features and predict
    # -------------------------
    mlp_path = os.path.join(models_dir, "mlp_stacking_saint_tabnet_model.joblib")
    if os.path.exists(mlp_path):
        try:
            mlp = joblib.load(mlp_path)
            # build meta-features for test: enhanced sklearn block + raw torch prob blocks
            sklearn_models = [(n, models[n]) for n in ["XGBoost", "Random Forest", "CatBoost"] if n in models]
            # calibrate not needed for inference
            X_meta_sklearn = build_meta_features(sklearn_models, X_test) if len(sklearn_models) > 0 else np.empty((len(X_test), 0))
            torch_blocks_test = []
            for n in ["SAINT", "TabNet"]:
                if n in models:
                    torch_blocks_test.append(model_predict_proba(models[n], X_test))
            if len(torch_blocks_test) > 0:
                X_meta_test = np.hstack([X_meta_sklearn] + torch_blocks_test)
            else:
                X_meta_test = X_meta_sklearn
            y_pred = mlp.predict(X_meta_test)
            results["individual"]["MLP_Stacking_OOF"] = compute_metrics(y_test, y_pred)
        except Exception as exc:
            results["individual"]["MLP_Stacking_OOF"] = {"error": str(exc)}

    # -------------------------
    # Logistic stacking: try to load saved logreg meta model, otherwise train a stacking meta on turkey train/val
    # -------------------------
    logreg_path = os.path.join(models_dir, "logreg_stacking_saint_tabnet_model.joblib")
    if os.path.exists(logreg_path):
        try:
            meta = joblib.load(logreg_path)
            # build meta features same as above but using full set of available base models
            sklearn_models = [(n, models[n]) for n in ["XGBoost", "Random Forest", "CatBoost"] if n in models]
            X_meta_sklearn = build_meta_features(sklearn_models, X_test) if len(sklearn_models) > 0 else np.empty((len(X_test), 0))
            torch_blocks_test = [model_predict_proba(models[n], X_test) for n in ["SAINT", "TabNet"] if n in models]
            X_meta_test = np.hstack([X_meta_sklearn] + torch_blocks_test) if len(torch_blocks_test) > 0 else X_meta_sklearn
            y_pred = meta.predict(X_meta_test)
            results["stacking"]["logreg"] = compute_metrics(y_test, y_pred)
        except Exception as exc:
            results["stacking"]["logreg"] = {"error": str(exc)}
    else:
        # train a StackingMetaEnsemble on Turkey train/val and evaluate on test
        try:
            classifiers = [(n, models[n]) for n in ["XGBoost", "Random Forest", "CatBoost", "SAINT", "TabNet"] if n in models]
            if len(classifiers) >= 2:
                stacker = StackingMetaEnsemble(classifiers, n_classes=len(np.unique(y)))
                stacker.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
                res = stacker.evaluate(X_test, y_test, name="Turkey")
                results["stacking"]["logreg_trained_on_turkey"] = res
        except Exception as exc:
            results["stacking"]["logreg_trained_on_turkey"] = {"error": str(exc)}

    # -------------------------
    # Weighted average via grid_search
    # -------------------------
    try:
        classifiers = [(n, models[n]) for n in ["XGBoost", "Random Forest", "CatBoost", "SAINT", "TabNet"] if n in models]
        if len(classifiers) >= 2:
            wel = WeightedEnsembleLearner(classifiers, method="grid_search", metric="f1", use_proba=True, random_state=42)
            wel.fit(X_train, y_train, X_val, y_val)
            y_pred = wel.predict(X_test)
            results["ensembles"]["grid_search"] = compute_metrics(y_test, y_pred)
        else:
            results["ensembles"]["grid_search"] = {"note": "not enough base models"}
    except Exception as exc:
        results["ensembles"]["grid_search"] = {"error": str(exc)}

    # -------------------------
    # Save results and confusion matrices
    # -------------------------
    results_path = os.path.join(results_dir, "turkey_all_models_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    # save confusion matrices for individual models
    try:
        for name, entry in individual_cms.items():
            if entry is None:
                continue
            labels, cm = entry
            key = f"individual_{name.replace(' ', '_')}"
            cm_path = save_cm(results_dir, key, labels, cm)
    except Exception:
        pass

    print(f"Results written to {results_path}")


if __name__ == "__main__":
    main()
