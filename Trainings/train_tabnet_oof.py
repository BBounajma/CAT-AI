import os
import sys
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split

import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassF1Score, MulticlassFBetaScore

from pytorch_widedeep import Trainer
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.models import WideDeep, TabNet
from pytorch_widedeep.callbacks import EarlyStopping

# ------------------------------------------------------------------
# Repro
# ------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)

# Training configuration
N_SPLITS = 5
EPOCHS_OOF = 15
EPOCHS_FINAL = 40
BATCH_SIZE = 1024

# ------------------------------------------------------------------
# Focal Loss
# ------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


# ------------------------------------------------------------------
# Build TabNet model
# ------------------------------------------------------------------
def build_tabnet(tab_preprocessor, num_cols):
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

    return WideDeep(deeptabular=tabnet, pred_dim=5)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":

    df = pd.read_csv("Data/processed_new_data2.csv")

    cat_cols = [
        "foundation_type",
        "roof_type",
        "ground_floor_type",
    ]

    num_cols = [
        "PGA_g",
        "count_floors_pre_eq",
        "age_building",
        "plinth_area_sq_ft",
        "per-height_ft_pre_eq",
    ]

    onehot_cols = [
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

    all_continuous_cols = num_cols + onehot_cols

    y = df["damage_grade"].astype(np.int64)
    X = df.drop(columns=["damage_grade"])

    # --------------------------------------------------------------
    # Hold-out test split (never touched by OOF)
    # --------------------------------------------------------------
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    X_tv = X_tv.reset_index(drop=True)
    y_tv = y_tv.reset_index(drop=True)

    # --------------------------------------------------------------
    # Tab preprocessing (fit ONCE on train+val)
    # --------------------------------------------------------------
    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_cols,
        continuous_cols=all_continuous_cols,
        cols_to_scale=num_cols,
        with_cls_token=False,
    )

    X_tv_tab = tab_preprocessor.fit_transform(X_tv)

    # --------------------------------------------------------------
    # OOF loop
    # --------------------------------------------------------------
    n_classes = y_tv.nunique()
    oof_preds = np.zeros((len(X_tv), n_classes))

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tv_tab, y_tv)):
        print(f"\n===== TabNet OOF fold {fold + 1} / 5 =====")

        X_tr, X_val = X_tv_tab[tr_idx], X_tv_tab[val_idx]
        y_tr, y_val = y_tv.iloc[tr_idx], y_tv.iloc[val_idx]

        model = build_tabnet(tab_preprocessor, all_continuous_cols).to(device)

        counts = np.bincount(y_tr.values)
        counts[counts == 0] = 1

        class_weights = torch.tensor(
            np.log1p(counts.sum() / counts),
            dtype=torch.float32,
        ).to(device)

        loss_fn = FocalLoss(gamma=2.0, weight=class_weights)

        trainer = Trainer(
            model=model,
            objective="multiclass",
            custom_loss_function=loss_fn,
            metrics=[
                MulticlassF1Score(num_classes=5, average="macro").to(device)
            ],
            learning_rate=3e-3,
        )

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
        )

        trainer.fit(
            X_tab=X_tr,
            target=y_tr.values,
            X_tab_val=X_val,
            target_val=y_val.values,
            n_epochs=EPOCHS_OOF,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping],
        )

        preds = trainer.predict_proba(X_tab=X_val)
        oof_preds[val_idx] = preds

    # --------------------------------------------------------------
    # Save OOF predictions
    # --------------------------------------------------------------
    target_dir = Path("Models/TabNet")
    target_dir.mkdir(parents=True, exist_ok=True)

    np.save(target_dir / "tabnet_oof_preds.npy", oof_preds)
    joblib.dump(tab_preprocessor, target_dir / "tab_preprocessor.joblib")

    print("\n✓ TabNet OOF predictions saved")

    # --------------------------------------------------------------
    # Train FINAL model on full train+val
    # --------------------------------------------------------------
    final_model = build_tabnet(tab_preprocessor, all_continuous_cols).to(device)

    counts = np.bincount(y_tv.values)
    counts[counts == 0] = 1

    class_weights = torch.tensor(
        np.log1p(counts.sum() / counts),
        dtype=torch.float32,
    ).to(device)

    trainer = Trainer(
        model=final_model,
        objective="multiclass",
        custom_loss_function=FocalLoss(gamma=2.0, weight=class_weights),
        metrics=[
            MulticlassF1Score(num_classes=5, average="macro").to(device)
        ],
        learning_rate=3e-3,
    )

    trainer.fit(
        X_tab=X_tv_tab,
        target=y_tv.values,
        n_epochs=EPOCHS_FINAL,
        batch_size=BATCH_SIZE,
    )

    torch.save(
        final_model.state_dict(),
        target_dir / "model_state_dict.pt",
    )

    print("✓ Final TabNet model saved")