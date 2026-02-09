import os
import sys

from anyio import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

import torch
from pytorch_widedeep import Trainer
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.models import SAINT, WideDeep
from pytorch_widedeep.metrics import Accuracy


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

    X_processed = df.drop("damage_grade", axis=1)

    #scale numerical features to be scaled by TabPreprocessor
    num_cols = [
    "PGA_g",
    "count_floors_pre_eq",
    "age_building",
    "plinth_area_sq_ft",
    "per-height_ft_pre_eq"

    ]
    


    y=df["damage_grade"] 

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
    input_dim=16,
    n_heads=2,
    n_blocks=1,
    attn_dropout=0.0
)


    model = WideDeep(
        deeptabular=saint,
        pred_dim=5
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    class_weights = torch.tensor(
        1.0 / np.bincount(y_train),
        dtype=torch.float32
    )
    class_weights /= class_weights.sum()

    trainer = Trainer(
        model=model,
        objective="multiclass",
        metrics=[Accuracy],
        class_weights=class_weights
    )


    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer.fit(
        X_tab=X_train_tab,
        target=y_train,
        X_tab_val=X_valid_tab,
        target_val=y_valid,
        n_epochs=10,
        batch_size=256
    )

    # ------------------------------------------------------------------
    # Proper saving
    # ------------------------------------------------------------------

    # Create a directory for all model artifacts
    model_dir = Path("Models/saint_model")
    model_dir.mkdir(exist_ok=True)

    # 1) Save the preprocessor
    joblib.dump(tab_preprocessor, model_dir / "../Models/Saint/tab_preprocessor.joblib")

    # 2) Save model state dict
    torch.save(model.state_dict(), model_dir / "../Models/Saint/model_state_dict.pt")

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

    joblib.dump(model_config, model_dir / "../Models/Saint/config.joblib")

    print(f"✓ Saved model artifacts to {model_dir}/")
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

