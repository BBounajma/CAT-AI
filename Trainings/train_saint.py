import os
import sys

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassF1Score,MulticlassFBetaScore
from pytorch_widedeep import Trainer
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.models import SAINT, WideDeep
from pytorch_widedeep.metrics import Accuracy, F1Score


if __name__ == '__main__':
    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    # Script path setup
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Models'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Data'))

    # Load data - adjust path as per your setup
    df = pd.read_csv("Data/processed_new_data2.csv")  # Linux version
    #df = pd.read_csv("Data\processed_new_data2.csv")  # Windows version


    #multi_class_cat_cols to be embedded by TabPreprocessor
    multi_class_cat_cols = [
    'foundation_type',
    'roof_type',
    'ground_floor_type'
    ]

    y=df["damage_grade"] 
    
    X_processed = df.drop("damage_grade", axis=1)

    #scale numerical features to be scaled by TabPreprocessor
    num_cols = [
    "PGA_g",
    "count_floors_pre_eq",
    "age_building",
    "plinth_area_sq_ft",
    "per-height_ft_pre_eq"

    ]
    


    

    # Split data into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    # Reset indices to 0-based sequential indexing
    X_train = X_train.reset_index(drop=True)
    X_valid = X_valid.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_valid = y_valid.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Tabular preprocessing
    # ------------------------------------------------------------------
    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=multi_class_cat_cols,
        continuous_cols=num_cols,
        cols_to_scale=num_cols,
        with_cls_token=False
    )


    X_train_tab = tab_preprocessor.fit_transform(X_train)
    X_valid_tab = tab_preprocessor.transform(X_valid)
    X_test_tab = tab_preprocessor.transform(X_test)


    # ------------------------------------------------------------------
    # SAINT + WideDeep model
    # ------------------------------------------------------------------
    saint = SAINT(
    column_idx=tab_preprocessor.column_idx,
    cat_embed_input=tab_preprocessor.cat_embed_input,
    continuous_cols=num_cols,
    input_dim=32,
    n_heads=4,
    n_blocks=2,
    attn_dropout=0.1
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
    

    """
    class_weights = torch.tensor(
        1.0 / np.bincount(y_train),
        dtype=torch.float32
    )
    class_weights /= class_weights.sum()
    """
    counts = np.bincount(y_train)
    counts[counts == 0] = 1
    beta = 0.999  # try 0.99–0.9999
    effective_num = 1.0 - np.power(beta, counts)
    class_weights = (1.0 - beta) / effective_num
    class_weights = class_weights / class_weights.sum() * len(counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)


    

    class_weights = torch.tensor(
        counts.sum() / counts,
        dtype=torch.float32
    )
    class_weights = class_weights.to(device)

    f1_macro = F1Score(
    average="macro",
    )

    f1_per_class = MulticlassF1Score(
    num_classes=5,
    average=None   
    ).to(device)

    f2_macro = MulticlassFBetaScore(
    num_classes=5,
    beta=2.0,
    average="macro"
    ).to(device)

    
    weighted_ce = nn.CrossEntropyLoss(weight=class_weights)

    trainer = Trainer(
    model=model,
    objective="multiclass",          
    custom_loss_function=weighted_ce,
    metrics=[f1_macro, f1_per_class, f2_macro],
    )
    

    loss = trainer.loss_fn

    print(type(loss))
    print(hasattr(loss, "weight"))
    print(loss.weight)




    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer.fit(
        X_tab=X_train_tab,
        target=y_train,
        X_tab_val=X_valid_tab,
        target_val=y_valid,
        n_epochs=100,
        batch_size=512,
        early_stopping=True,
        early_stopping_metric="f1_macro",
        patience=1,
    )

    # ------------------------------------------------------------------
    # Proper saving
    # ------------------------------------------------------------------

    # Create a directory for all model artifacts
    model_dir = Path("Models/saint_model")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Ensure target artifacts directory exists
    target_dir = Path("Models/Saint")
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1) Save the preprocessor
    joblib.dump(tab_preprocessor, str(target_dir / "tab_preprocessor.joblib"))

    # 2) Save model state dict
    torch.save(model.state_dict(), str(target_dir / "model_state_dict.pt"))

    # 3) Save metadata/config as a dictionary
    model_config = {
        "model_type": "SAINT",
        "architecture": {
        "input_dim": 16,
        "n_heads": 2,
        "n_blocks": 1,
        "attn_dropout": 0.0,
        "pred_dim": 5
    },
    "features": {
        "num_cols": num_cols,
        "cat_cols": multi_class_cat_cols
    },
    "files": {
        "preprocessor": "tab_preprocessor.joblib",
        "weights": "model_state_dict.pt"
        }
    }

    joblib.dump(model_config, str(target_dir / "config.joblib"))

    print(f"✓ Saved model artifacts to {target_dir}/")
    print(f"  - tab_preprocessor.joblib")
    print(f"  - model_state_dict.pt")
    print(f"  - config.joblib")
    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    print("\nEvaluating on test set...")
    y_pred = trainer.predict(X_tab=X_test_tab)

    from sklearn.metrics import accuracy_score, classification_report
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)

