import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
from torch.optim import AdamW
from torchmetrics.classification import MulticlassF1Score

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

from pytorch_widedeep import Trainer
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.models import WideDeep, SAINT
from pytorch_widedeep.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_widedeep.initializers import XavierNormal

# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
N_SPLITS = 5
N_CLASSES = 5
BATCH_SIZE = 256
EPOCHS_OOF = 15
EPOCHS_FINAL = 20

DATA_PATH = "Data/processed_new_data2.csv"
OUT_DIR = Path("Models/Saint")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

multi_class_cat_cols = [
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

y = df["damage_grade"]
if y.min() == 1:
    y = y - 1
y = y.astype(np.int64)

X = df.drop(columns=["damage_grade"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# Loss & metrics
# ------------------------------------------------------------------
counts = np.bincount(y_train.values)
counts[counts == 0] = 1
weights = 1.0 / counts
weights = weights / weights.sum() * len(counts)
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

loss_fn = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.05,
)

f1_macro = MulticlassF1Score(
    num_classes=N_CLASSES,
    average="macro"
).to(device)

# ------------------------------------------------------------------
# Model builder
# ------------------------------------------------------------------
def build_saint(tab_preprocessor):
    saint = SAINT(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=all_continuous_cols,
        input_dim=64,
        n_heads=8,
        n_blocks=3,
        attn_dropout=0.05,
        ff_dropout=0.05,
    )
    return WideDeep(deeptabular=saint, pred_dim=N_CLASSES)

# ------------------------------------------------------------------
# OOF training
# ------------------------------------------------------------------
print("\n===== SAINT OOF TRAINING =====")

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=42
)

oof_preds = np.zeros((len(X_train), N_CLASSES), dtype=np.float32)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    print(f"\n[Fold {fold}/{N_SPLITS}]")

    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    # Ensure targets use a contiguous RangeIndex so DataLoader worker
    # positional indexing (0..n-1) does not raise a KeyError when
    # pytorch_widedeep accesses self.Y[idx]
    y_tr = y_tr.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)

    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=multi_class_cat_cols,
        continuous_cols=all_continuous_cols,
        cols_to_scale=num_cols,
        with_cls_token=False,
    )

    X_tr_tab = tab_preprocessor.fit_transform(X_tr)
    X_val_tab = tab_preprocessor.transform(X_val)

    model = build_saint(tab_preprocessor).to(device)

    optimizer = AdamW(
        model.deeptabular.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    trainer = Trainer(
        model=model,
        objective="multiclass",
        custom_loss_function=loss_fn,
        metrics=[f1_macro],
        optimizers={"deeptabular": optimizer},
        initializers={"deeptabular": XavierNormal},
    )

    early_stopping = EarlyStopping(
        monitor="val_f1_macro",
        mode="max",
        patience=4,
        restore_best_weights=True,
    )

    trainer.fit(
        X_tab=X_tr_tab,
        target=y_tr.values,
        X_tab_val=X_val_tab,
        target_val=y_val.values,
        n_epochs=EPOCHS_OOF,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=0,
    )

    probs = trainer.predict_proba(X_tab=X_val_tab)
    oof_preds[val_idx] = probs

# Save OOF predictions
np.save(OUT_DIR / "saint_oof_preds.npy", oof_preds)
print("\n✓ Saved OOF predictions")

# ------------------------------------------------------------------
# Final model (full training data)
# ------------------------------------------------------------------
print("\n===== SAINT FINAL TRAINING =====")

tab_preprocessor = TabPreprocessor(
    cat_embed_cols=multi_class_cat_cols,
    continuous_cols=all_continuous_cols,
    cols_to_scale=num_cols,
    with_cls_token=False,
)

X_train_tab = tab_preprocessor.fit_transform(X_train)
X_test_tab = tab_preprocessor.transform(X_test)

model = build_saint(tab_preprocessor).to(device)

optimizer = AdamW(
    model.deeptabular.parameters(),
    lr=1e-4,
    weight_decay=1e-4,
)

trainer = Trainer(
    model=model,
    objective="multiclass",
    custom_loss_function=loss_fn,
    metrics=[f1_macro],
    optimizers={"deeptabular": optimizer},
    initializers={"deeptabular": XavierNormal},
)

early_stopping = EarlyStopping(
    monitor="val_f1_macro",
    mode="max",
    patience=5,
    restore_best_weights=True,
)

checkpoint = ModelCheckpoint(
    filepath=str(OUT_DIR / "saint_best"),
    monitor="val_loss",
    save_best_only=True,
    max_save=1,
)

trainer.fit(
    X_tab=X_train_tab,
    target=y_train.values,
    X_tab_val=X_test_tab,   # only for monitoring
    target_val=y_test.values,
    n_epochs=EPOCHS_FINAL,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping, checkpoint],
)

# ------------------------------------------------------------------
# Save artifacts
# ------------------------------------------------------------------
joblib.dump(tab_preprocessor, OUT_DIR / "tab_preprocessor.joblib")
torch.save(model.state_dict(), OUT_DIR / "model_state_dict.pt")

joblib.dump(
    {
        "model_type": "SAINT",
        "n_splits_oof": N_SPLITS,
        "features": {
            "num_cols": num_cols,
            "onehot_cols": onehot_cols,
            "cat_cols": multi_class_cat_cols,
            "all_continuous_cols": all_continuous_cols,
        },
    },
    OUT_DIR / "config.joblib",
)

print("\n✓ SAINT artifacts saved")

# ------------------------------------------------------------------
# Test evaluation
# ------------------------------------------------------------------
y_pred = trainer.predict(X_tab=X_test_tab)

print("\n===== FINAL TEST PERFORMANCE =====")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))