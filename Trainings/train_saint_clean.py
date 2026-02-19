import os
import sys
import einops

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassF1Score, MulticlassFBetaScore
from pytorch_widedeep import Trainer
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.models import WideDeep
from pytorch_widedeep.metrics import Accuracy, F1Score
from pytorch_widedeep.models import TabNet
from pytorch_widedeep.callbacks import EarlyStopping, ModelCheckpoint

torch.manual_seed(42)
np.random.seed(42)


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


if __name__ == '__main__':
    # ------------------------------------------------------------------
    # Load data
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

     onehot_cols = [
        "has_superstructure_mud_mortar_stone",
        "has_superstructure_stone_flag",
        "has_superstructure_cement_mortar_stone",
        "has_superstructure_cement_mortar_brick",
        "has_superstructure_timber",
        "has_superstructure_bamboo",
        "has_superstructure_rc_non_engineered",
        "has_superstructure_rc_engineered",
        "has_superstructure_other"
    ]

    all_continuous_cols = num_cols + onehot_cols

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.3, random_state=42, stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
    )

    X_train = X_train.reset_index(drop=True)
    X_valid = X_valid.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_valid = y_valid.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    y_train = (y_train ).astype(np.int64)
    y_valid = (y_valid ).astype(np.int64)
    y_test  = (y_test  ).astype(np.int64)

    # ------------------------------------------------------------------
    # Tabular preprocessing
    # ------------------------------------------------------------------
    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=multi_class_cat_cols,
        continuous_cols=all_continuous_cols,
        cols_to_scale=num_cols,
        with_cls_token=False
    )

    X_train_tab = tab_preprocessor.fit_transform(X_train)
    X_valid_tab = tab_preprocessor.transform(X_valid)
    X_test_tab  = tab_preprocessor.transform(X_test)

    # ------------------------------------------------------------------
    # TabNet + WideDeep model (PARAMS IMPROVED)
    # ------------------------------------------------------------------
    saint = TabNet(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=num_cols,

        n_steps=7,                    # ↑ more decision steps
        step_dim=128,                 # ↑ representation capacity
        attn_dim=128,                 # ↑ attention capacity
        dropout=0.2,                  # ↑ regularization

        n_glu_step_dependent=2,
        n_glu_shared=2,

        gamma=1.3,                    # ↓ less aggressive feature reuse
        epsilon=1e-15,

        ghost_bn=True,
        virtual_batch_size=128,
        momentum=0.02,

        mask_type="sparsemax",
    )

    model = WideDeep(
        deeptabular=saint,
        pred_dim=5
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    counts = np.bincount(y_train)
    counts[counts == 0] = 1

    class_weights = torch.tensor(
        np.log1p(counts.sum() / counts),
        dtype=torch.float32
    ).to(device)

    f1_macro = MulticlassF1Score(num_classes=5, average="macro").to(device)
    f1_per_class = MulticlassF1Score(num_classes=5, average=None).to(device)
    f2_macro = MulticlassFBetaScore(num_classes=5, beta=2.0, average="macro").to(device)

    loss_fn = FocalLoss(gamma=2.0, weight=class_weights)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=6,
        verbose=1,
        restore_best_weights=True
    )

    trainer = Trainer(
        model=model,
        objective="multiclass",
        custom_loss_function=loss_fn,
        metrics=[f1_macro, f1_per_class, f2_macro],
        learning_rate=3e-3,   # ↓ for larger TabNet
        lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        lr_scheduler_params={
            "mode": "min",
            "factor": 0.5,
            "patience": 3
        }
    )

    model_checkpoint = ModelCheckpoint(
        filepath='models/tabnet_best',
        monitor='val_loss',
        save_best_only=True,
        max_save=1
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer.fit(
        X_tab=X_train_tab,
        target=y_train,
        X_tab_val=X_valid_tab,
        target_val=y_valid,
        n_epochs=50,
        batch_size=1024,
        clip_grad_norm=1.0,
        callbacks=[early_stopping, model_checkpoint]
    )

    # ------------------------------------------------------------------
    # Proper saving
    # ------------------------------------------------------------------
    target_dir = Path("Models/TabNet")
    target_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(tab_preprocessor, target_dir / "tab_preprocessor.joblib")
    torch.save(model.state_dict(), target_dir / "model_state_dict.pt")

    print("✓ Model artifacts saved")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------