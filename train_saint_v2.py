import os
if os.environ.get("FORCE_SINGLE_GPU", "1") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"

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
    df = pd.read_csv("Data/new_data2.csv")
    df.rename(columns={
        'roof_type(Bamboo/Timber-Heavy roof=0; Bamboo/Timber-Light roof=1; RCC/RB/RBC=2)': 'roof_type',
        'district_distance_to_earthquakecenter(mi)': 'distance'
    }, inplace=True)

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
        "has_superstructure_adobe_mud"
    ]

    df.drop(columns=to_drop, inplace=True)

    # Target: make it zero-based
    y = df["damage_grade"] - 1
    df.drop(columns=["damage_grade"], inplace=True)

    multi_class_cat_cols = [
        "foundation_type",
        "roof_type",
        "ground_floor_type",
    ]

    # ------------------------------------------------------------------
    # Missing values
    # ------------------------------------------------------------------
    df[multi_class_cat_cols] = df[multi_class_cat_cols].fillna(
        df[multi_class_cat_cols].mode().iloc[0]
    )
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # ------------------------------------------------------------------
    # Tabular preprocessing
    # ------------------------------------------------------------------
    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=multi_class_cat_cols,
        continuous_cols=num_cols,
        scale=True,
        with_cls_token=True
    )

    X_tab = tab_preprocessor.fit_transform(df)
    y = y.to_numpy()

    # ------------------------------------------------------------------
    # Train / val / test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_tab, y, test_size=0.3, random_state=42
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.3, random_state=42
    )

    # ------------------------------------------------------------------
    # SAINT + WideDeep model
    # ------------------------------------------------------------------
    saint = SAINT(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=num_cols,
        input_dim=32,
        n_heads=8,
        n_blocks=3,
        attn_dropout=0.1
    )

    model = WideDeep(
        deeptabular=saint,
        pred_dim=5
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        objective="multiclass",
        metrics=[Accuracy],
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer.fit(
        X_tab=X_train,
        target=y_train,
        X_tab_val=X_valid,
        target_val=y_valid,
        n_epochs=10,
        batch_size=256
    )

    # ------------------------------------------------------------------
    # Proper saving
    # ------------------------------------------------------------------

    # 1) Save preprocessing (joblib is correct here)
    joblib.dump(tab_preprocessor, "tab_preprocessor.joblib")

    # 2) Save model via Trainer (recommended)
    trainer.save(
        path="widedeep_saint_model",
        save_state_dict=True
    )

    # (This creates: widedeep_saint_model/model.pt)

    print("Saved:")
    print("  - tab_preprocessor.joblib")
    print("  - widedeep_saint_model/model.pt")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    print("\nEvaluating on test set...")
    y_pred = trainer.predict(X_tab=X_test)

    from sklearn.metrics import accuracy_score, classification_report
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
