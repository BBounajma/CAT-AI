"""
Script to load and test the SAINT model trained in train_saint_v2.py
(using trainer.save(..., save_state_dict=False))
"""

import os

# ------------------------------------------------------------------
# Force single GPU (must match training environment)
# ------------------------------------------------------------------
if os.environ.get("FORCE_SINGLE_GPU", "1") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"

import json
import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from pytorch_widedeep import Trainer


# ------------------------------------------------------------------
# Data loading & preprocessing (MUST match training exactly)
# ------------------------------------------------------------------
def load_and_preprocess_data():
    df = pd.read_csv("Data/new_data2.csv")

    df.rename(
        columns={
            "roof_type(Bamboo/Timber-Heavy roof=0; Bamboo/Timber-Light roof=1; RCC/RB/RBC=2)": "roof_type",
            "district_distance_to_earthquakecenter(mi)": "distance",
        },
        inplace=True,
    )

    num_cols = [
        "PGA_g",
        "count_floors_pre_eq",
        "age_building",
        "plinth_area_sq_ft",
        "per-height_ft_pre_eq",
        "has_superstructure_mud_mortar_stone",
        "has_superstructure_stone_flag",
        "has_superstructure_cement_mortar_stone",
        "has_superstructure_cement_mortar_brick",
        "has_superstructure_timber",
        "has_superstructure_bamboo",
        "has_superstructure_rc_non_engineered",
        "has_superstructure_rc_engineered",
        "has_superstructure_other",
    ]

    to_drop = [
        "technical_solution_proposed",
        "condition_post_eq",
        "vdcmun_id",
        "ward_id",
        "land_surface_condition",
        "has_superstructure_mud_mortar_brick",
        "position",
        "has_superstructure_adobe_mud",
    ]

    df.drop(columns=to_drop, inplace=True)

    # Target: zero-based classes
    y = df["damage_grade"].astype(int) - 1
    df.drop(columns=["damage_grade"], inplace=True)

    multi_class_cat_cols = [
        "foundation_type",
        "roof_type",
        "ground_floor_type",
    ]

    # Missing values (same logic as training)
    df[multi_class_cat_cols] = df[multi_class_cat_cols].fillna(
        df[multi_class_cat_cols].mode().iloc[0]
    )
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    return df, y.to_numpy()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print("=" * 70)
    print("SAINT Model Evaluation (Full Trainer Checkpoint)")
    print("=" * 70)

    # --------------------------------------------------
    # Load tabular preprocessor
    # --------------------------------------------------
    print("\n1. Loading tab preprocessor...")
    if not os.path.exists("tab_preprocessor.joblib"):
        raise FileNotFoundError("tab_preprocessor.joblib not found")

    tab_preprocessor = joblib.load("tab_preprocessor.joblib")
    print("   ✓ Preprocessor loaded")

    # --------------------------------------------------
    # Load and preprocess data
    # --------------------------------------------------
    print("\n2. Loading and preprocessing data...")
    df, y = load_and_preprocess_data()
    X_tab = tab_preprocessor.transform(df)

    print(f"   ✓ Samples: {X_tab.shape[0]}")
    print(f"   ✓ Features: {X_tab.shape[1]}")

    # --------------------------------------------------
    # Train / valid / test split (same seed as training)
    # --------------------------------------------------
    print("\n3. Creating data splits...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_tab, y, test_size=0.3, random_state=42, stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
    )

    print(f"   ✓ Train: {len(y_train)}")
    print(f"   ✓ Valid: {len(y_valid)}")
    print(f"   ✓ Test:  {len(y_test)}")

    # --------------------------------------------------
    # Load trained Trainer (FULL checkpoint)
    # --------------------------------------------------
    print("\n4. Loading trained model checkpoint...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = Trainer.load(
        path="widedeep_saint_model",
        device=device,
    )

    trainer.model.eval()
    print(f"   ✓ Trainer + model loaded on {device}")

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    print("\n5. Running predictions...")

    with torch.no_grad():
        y_pred_train = trainer.predict(X_tab=X_train)
        y_pred_valid = trainer.predict(X_tab=X_valid)
        y_pred_test = trainer.predict(X_tab=X_test)

    # Safety check
    assert y_pred_test.ndim == 1, "predict() must return class labels"

    train_acc = accuracy_score(y_train, y_pred_train)
    valid_acc = accuracy_score(y_valid, y_pred_valid)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"   ✓ Train accuracy: {train_acc:.4f}")
    print(f"   ✓ Valid accuracy: {valid_acc:.4f}")
    print(f"   ✓ Test  accuracy: {test_acc:.4f}")

    # --------------------------------------------------
    # Detailed metrics
    # --------------------------------------------------
    print("\n6. Test set diagnostics")
    print("-" * 70)

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred_test,
            target_names=[
                "Grade 1",
                "Grade 2",
                "Grade 3",
                "Grade 4",
                "Grade 5",
            ],
        )
    )

    print("\nConfusion matrix:")
    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)

    print("\nPer-class accuracy:")
    for cls in range(5):
        mask = y_test == cls
        if mask.any():
            acc = accuracy_score(y_test[mask], y_pred_test[mask])
            print(f"   Grade {cls + 1}: {acc:.4f} ({mask.sum()} samples)")

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------
    results = {
        "train_accuracy": train_acc,
        "validation_accuracy": valid_acc,
        "test_accuracy": test_acc,
        "n_test_samples": int(len(y_test)),
        "checkpoint_path": "widedeep_saint_model",
        "save_mode": "full_trainer_checkpoint",
    }

    with open("test_results_saint_v2.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n✓ Results saved to test_results_saint_v2.json")
    print("=" * 70)
    print("Evaluation complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
