import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

import torch
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.models import WideDeep, TabNet
from pytorch_widedeep import Trainer

torch.manual_seed(42)
np.random.seed(42)


if __name__ == '__main__':
    # ------------------------------------------------------------------
    # Load data (same as training)
    # ------------------------------------------------------------------
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Models'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Data'))

    df = pd.read_csv("Data/processed_new_data2.csv")

    multi_class_cat_cols = [
        'foundation_type',
        'roof_type',
        'ground_floor_type'
    ]

    y = df["damage_grade"]
    X_processed = df.drop("damage_grade", axis=1)

    num_cols = [
        "PGA_g",
        "count_floors_pre_eq",
        "age_building",
        "plinth_area_sq_ft",
        "per-height_ft_pre_eq"
    ]

    # Split data (same split as training with random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.3, random_state=42, stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
    )

    # Reset indices
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    y_test = y_test.astype(np.int64)

    # ------------------------------------------------------------------
    # Load preprocessor
    # ------------------------------------------------------------------
    model_dir = Path("Models/TabNet")
    
    print("Loading saved model artifacts...")
    tab_preprocessor = joblib.load(model_dir / "tab_preprocessor.joblib")
    print(f"✓ Loaded preprocessor from {model_dir / 'tab_preprocessor.joblib'}")

    # Transform test data
    X_test_tab = tab_preprocessor.transform(X_test)
    print(f"✓ Transformed test data: {X_test_tab.shape}")

    # ------------------------------------------------------------------
    # Reconstruct model architecture (must match training)
    # ------------------------------------------------------------------
    tabnet = TabNet(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=num_cols,
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

    model = WideDeep(
        deeptabular=tabnet,
        pred_dim=5
    )

    # ------------------------------------------------------------------
    # Load weights
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    state_dict = torch.load(model_dir / "model_state_dict.pt", map_location=device)
    model.load_state_dict(state_dict)
    print(f"✓ Loaded model weights from {model_dir / 'model_state_dict.pt'}")
    
    model.eval()  # Set to evaluation mode

    # ------------------------------------------------------------------
    # Create trainer for prediction
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        objective="multiclass",
    )

    # ------------------------------------------------------------------
    # Test set evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EVALUATING ON TEST SET")
    print("=" * 70)
    
    y_pred = trainer.predict(X_tab=X_test_tab, batch_size=1024)

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score
    )

    # Calculate metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    print(f"\nTest Set Results:")
    print(f"  Accuracy:      {test_accuracy:.4f}")
    print(f"  F1-Macro:      {f1_macro:.4f}")
    print(f"  F1-Weighted:   {f1_weighted:.4f}")
    
    print("\n" + "-" * 70)
    print("Classification Report:")
    print("-" * 70)
    print(classification_report(y_test, y_pred, target_names=[
        "Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5"
    ]))
    
    print("\n" + "-" * 70)
    print("Confusion Matrix:")
    print("-" * 70)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Per-class accuracy
    print("\n" + "-" * 70)
    print("Per-Class Accuracy:")
    print("-" * 70)
    for i in range(5):
        class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"  Grade {i+1}: {class_acc:.4f} ({cm[i, i]}/{cm[i].sum()})")

    print("\n" + "=" * 70)
    print("Testing Complete!")
    print("=" * 70)

