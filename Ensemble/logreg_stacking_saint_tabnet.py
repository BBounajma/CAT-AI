import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression

from utils_stacking import (
    build_meta_features,
    build_meta_features_oof,
    load_saint_predictor,
    make_torch_tabular_predictor,
    model_predict,
    model_predict_proba,
    summarize_model_usage,
    ablation_impact,
    calibrate_sklearn_models,
    temperature_scale_torch_models,
)

import torch
from pytorch_widedeep.models import TabNet, WideDeep


# ============================================================
# TabNet loader (FINAL MODEL ONLY)
# ============================================================
def load_tabnet_predictor(models_dir):
    tabnet_dir = os.path.join(models_dir, "TabNet")
    model_path = os.path.join(tabnet_dir, "model_state_dict.pt")
    prep_path = os.path.join(tabnet_dir, "tab_preprocessor.joblib")

    if not (os.path.exists(model_path) and os.path.exists(prep_path)):
        return None

    tab_preprocessor = joblib.load(prep_path)
    continuous_cols = getattr(tab_preprocessor, "continuous_cols", [])

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
        virtual_batch_size=128,
        mask_type="sparsemax",
    )

    model = WideDeep(deeptabular=tabnet, pred_dim=5)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    predictor = make_torch_tabular_predictor(model, tab_preprocessor)
    tabnet_oof_path = os.path.join(tabnet_dir, "tabnet_oof_preds.npy")
    if os.path.exists(tabnet_oof_path):
        try:
            predictor["oof"] = np.load(tabnet_oof_path)
        except Exception:
            pass

    return predictor


# ============================================================
# Main
# ============================================================
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    models_dir = os.path.join(project_root, "Models")
    data_path = os.path.join(project_root, "Data/processed_new_data2.csv")

    print("=" * 70)
    print("Logistic Regression Stacking (Leakage-Free)")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    df = pd.read_csv(data_path)

    cat_cols = [
        "foundation_type",
        "roof_type",
        "ground_floor_type",
    ]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category").cat.codes

    y = df["damage_grade"]
    X = df.drop(columns=["damage_grade"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    X_meta_source = pd.concat([X_train, X_val]).reset_index(drop=True)
    y_meta_source = pd.concat([y_train, y_val]).reset_index(drop=True)

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # --------------------------------------------------------
    # Load base models
    # --------------------------------------------------------
    sklearn_models = []
    torch_models = []

    xgb_path = os.path.join(models_dir, "XG_boost/xgb_classifier_model.joblib")
    rf_path = os.path.join(models_dir, "Random_Forest/rf_classifier_model.joblib")

    if os.path.exists(xgb_path):
        sklearn_models.append(("XGBoost", joblib.load(xgb_path)))
        print("✓ XGBoost loaded")

    if os.path.exists(rf_path):
        sklearn_models.append(("Random Forest", joblib.load(rf_path)))
        print("✓ Random Forest loaded")

    catboost_path = os.path.join(models_dir, "CatBoost/cat_classifier_model.joblib")
    if os.path.exists(catboost_path):
        sklearn_models.append(("CatBoost", joblib.load(catboost_path)))
        print("✓ CatBoost loaded")

    saint = load_saint_predictor(models_dir)
    if saint is not None:
        torch_models.append(("SAINT", saint))
        print("✓ SAINT loaded")

    tabnet = load_tabnet_predictor(models_dir)
    if tabnet is not None:
        torch_models.append(("TabNet", tabnet))
        print("✓ TabNet loaded")

    # Temperature-scale torch predictors (using validation set)
    if len(torch_models) > 0:
        torch_models = temperature_scale_torch_models(torch_models, X_val, y_val)

    if len(sklearn_models) + len(torch_models) < 2:
        raise RuntimeError("Need at least two base models for stacking")

    # --------------------------------------------------------
    # Build OOF meta-features (TRAIN)
    # --------------------------------------------------------
    print("\nBuilding meta-features (true OOF)...")

    X_meta_sklearn, sklearn_stack = build_meta_features_oof(
        sklearn_models,
        X_meta_source,
        y_meta_source,
        n_splits=5,
        random_state=42,
    )

    # Calibrate the fitted sklearn stack using the validation set
    if len(sklearn_stack) > 0:
        sklearn_stack = calibrate_sklearn_models(sklearn_stack, X_val, y_val)

    oof_blocks = [X_meta_sklearn]

    included_torch_names = []
    included_torch_predictors = []

    saint_oof_path = os.path.join(models_dir, "Saint", "saint_oof_preds.npy")
    if os.path.exists(saint_oof_path):
        oof_blocks.append(np.load(saint_oof_path))
        included_torch_names.append("SAINT")

    tabnet_oof_path = os.path.join(models_dir, "TabNet", "tabnet_oof_preds.npy")
    if os.path.exists(tabnet_oof_path):
        oof_blocks.append(np.load(tabnet_oof_path))
        included_torch_names.append("TabNet")

    for name, pred in torch_models:
        if name in included_torch_names:
            included_torch_predictors.append((name, pred))

    X_meta_train = np.hstack(oof_blocks)

    # --------------------------------------------------------
    # Build meta-features (TEST)
    # --------------------------------------------------------
    if len(sklearn_stack) > 0:
        X_meta_test_sklearn = build_meta_features(sklearn_stack, X_test)
    else:
        X_meta_test_sklearn = np.empty((len(X_test), 0))

    torch_blocks_test = [
        model_predict_proba(pred, X_test)
        for _, pred in included_torch_predictors
    ]

    if len(torch_blocks_test) > 0:
        X_meta_test = np.hstack([X_meta_test_sklearn] + torch_blocks_test)
    else:
        X_meta_test = X_meta_test_sklearn

    # --------------------------------------------------------
    # Train meta-learner (LOGISTIC REGRESSION)
    # --------------------------------------------------------
    meta = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        C=0.2,
        class_weight="balanced",
        max_iter=2000,
        n_jobs=-1,
        random_state=42,
    )

    print("Training Logistic Regression meta-learner...")
    meta.fit(X_meta_train, y_meta_source)

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------
    y_pred = meta.predict(X_meta_test)

    print("\n" + "=" * 70)
    print("Stacking Performance (Test)")
    print("=" * 70)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1-Macro:  {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"F1-Weight: {f1_score(y_test, y_pred, average='weighted'):.4f}")

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------
    stacked_classifiers = sklearn_stack + included_torch_predictors

    print("\nStacked classifiers order:")
    for i, (n, _) in enumerate(stacked_classifiers):
        print(f"  {i}: {n}")

    n_classes = model_predict_proba(
        stacked_classifiers[0][1], X_test.iloc[:1]
    ).shape[1]

    print("\n" + "=" * 70)
    print("Model Usage (absolute coefficient mass)")
    print("=" * 70)

    # Compute model usage directly from LogisticRegression `coef_` to avoid
    # depending on MLP-specific attributes in `MLP_stacking.summarize_model_usage`.
    # coef_ shape: (n_classes, n_meta_features)
    if not hasattr(meta, "coef_"):
        raise ValueError("Meta-learner does not expose `coef_`; cannot compute usage")

    feature_strength = np.mean(np.abs(meta.coef_), axis=0)

    # Count sklearn-like base models (those that were OOF-trained)
    sklearn_count = sum(
        1 for _, clf in stacked_classifiers if hasattr(clf, "fit") and hasattr(clf, "predict_proba")
    )

    # Compute enhanced block size used during training: see _build_enhanced_meta_features
    if sklearn_count > 0:
        enhanced_block_size = sklearn_count * n_classes + 3 * sklearn_count + n_classes + 3
    else:
        enhanced_block_size = 0

    model_scores = []
    for model_idx, (name, _) in enumerate(stacked_classifiers):
        if model_idx < sklearn_count:
            start = model_idx * n_classes
        else:
            torch_idx = model_idx - sklearn_count
            start = enhanced_block_size + torch_idx * n_classes
        end = start + n_classes
        score = feature_strength[start:end].mean()
        model_scores.append((name, score))

    total = sum(s for _, s in model_scores) + 1e-12
    for name, score in model_scores:
        print(f"{name:20s} | strength: {score:.6f} | share: {score/total*100:6.2f}%")

    print("\n" + "=" * 70)
    print("Ablation Impact")
    print("=" * 70)
    base_acc, base_f1, drops = ablation_impact(
        meta, stacked_classifiers, X_test, y_test
    )
    print(f"Baseline -> Acc: {base_acc:.4f} | F1: {base_f1:.4f}")
    for name, d_acc, d_f1 in drops:
        print(f"{name:20s} | ΔAcc: {d_acc:+.4f} | ΔF1: {d_f1:+.4f}")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    joblib.dump(
        meta,
        os.path.join(models_dir, "logreg_stacking_saint_tabnet_model.joblib"),
    )
    joblib.dump(
        {
            "base_models": [n for n, _ in stacked_classifiers],
            "meta_dim": X_meta_train.shape[1],
        },
        os.path.join(
            models_dir, "logreg_stacking_saint_tabnet_info.joblib"
        ),
    )

    print("\n✓ Meta-learner saved")
    print("=" * 70)


if __name__ == "__main__":
    main()