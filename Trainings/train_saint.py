import os
import sys

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler, AdamW
from torchmetrics.classification import MulticlassF1Score, MulticlassFBetaScore, AUROC

from pytorch_widedeep import Trainer
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.models import WideDeep, SAINT
from pytorch_widedeep.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_widedeep.initializers import XavierNormal

torch.manual_seed(42)
np.random.seed(42)


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
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Models'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Data'))

    df = pd.read_csv('Data/processed_new_data2.csv')

    multi_class_cat_cols = [
        'foundation_type',
        'roof_type',
        'ground_floor_type'
    ]

    num_cols = [
        'PGA_g',
        'count_floors_pre_eq',
        'age_building',
        'plinth_area_sq_ft',
        'per-height_ft_pre_eq'
    ]

    onehot_cols = [
        'has_superstructure_mud_mortar_stone',
        'has_superstructure_stone_flag',
        'has_superstructure_cement_mortar_stone',
        'has_superstructure_cement_mortar_brick',
        'has_superstructure_timber',
        'has_superstructure_bamboo',
        'has_superstructure_rc_non_engineered',
        'has_superstructure_rc_engineered',
        'has_superstructure_other'
    ]

    all_continuous_cols = num_cols + onehot_cols

    y = df['damage_grade']
    if y.min() == 1:
        y = y - 1
    y = y.astype(np.int64)

    X_processed = df.drop('damage_grade', axis=1)

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

    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=multi_class_cat_cols,
        continuous_cols=all_continuous_cols,
        cols_to_scale=num_cols,
        with_cls_token=False
    )

    X_train_tab = tab_preprocessor.fit_transform(X_train)
    X_valid_tab = tab_preprocessor.transform(X_valid)
    X_test_tab = tab_preprocessor.transform(X_test)

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

    model = WideDeep(
        deeptabular=saint,
        pred_dim=5
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    counts = np.bincount(y_train)
    counts[counts == 0] = 1

    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)

    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    f1_macro = MulticlassF1Score(num_classes=5, average='macro').to(device)
    f1_per_class = MulticlassF1Score(num_classes=5, average=None).to(device)
    f2_macro = MulticlassFBetaScore(num_classes=5, beta=2.0, average='macro').to(device)



    auroc = AUROC(num_classes=len(counts), task="multiclass").to(device)

    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05
    )

    #loss_fn = FocalLoss(gamma=2.0, weight=class_weights)

    deep_opt = AdamW(
        model.deeptabular.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )


    trainer = Trainer(
        model=model,
        objective="multiclass",
        custom_loss_function=loss_fn,
        metrics=[f1_macro],
        optimizers={"deeptabular": deep_opt},
        initializers={"deeptabular": XavierNormal},
        lr_scheduler_params={          
            "mode": "min",
            "factor": 0.5,
            "patience": 3
        }
    )

    early_stopping=EarlyStopping(
        monitor='val_f1_macro',
        mode="max",
        patience=5,
        restore_best_weights=True
    )

    model_checkpoint = ModelCheckpoint(
        filepath='models/saint_best',
        monitor='val_loss',
        save_best_only=True,
        max_save=1
    )

    print("Train distribution:", np.bincount(y_train))
    print("Valid distribution:", np.bincount(y_valid))

    trainer.fit(
        X_tab=X_train_tab,
        target=y_train,
        X_tab_val=X_valid_tab,
        target_val=y_valid,
        n_epochs=20,
        batch_size=256,
        clip_grad_norm=1.0,
        callbacks=[early_stopping, model_checkpoint]
    )

    target_dir = Path('Models/Saint')
    target_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(tab_preprocessor, target_dir / 'tab_preprocessor.joblib')
    torch.save(model.state_dict(), target_dir / 'model_state_dict.pt')

    model_config = {
        'model_type': 'SAINT',
        'architecture': {
            'input_dim': 32,
            'n_heads': 4,
            'n_blocks':24,
            'attn_dropout': 0.05,
            'ff_dropout': 0.05,
            'pred_dim': 5
        },
        'features': {
            'num_cols': num_cols,
            'onehot_cols': onehot_cols,
            'cat_cols': multi_class_cat_cols,
            'all_continuous_cols': all_continuous_cols
        },
        'files': {
            'preprocessor': 'tab_preprocessor.joblib',
            'weights': 'model_state_dict.pt'
        }
    }
    joblib.dump(model_config, target_dir / 'config.joblib')

    print('✓ Model artifacts saved')

    print('\nEvaluating on test set...')
    y_pred = trainer.predict(X_tab=X_test_tab)

    print(f'Test Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))
