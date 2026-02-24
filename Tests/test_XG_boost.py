import os
import sys

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    model_path = os.path.join(project_root, "Models", "xgb_classifier_model.joblib")
    data_path = os.path.join(project_root, "Data", "processed_new_data2.csv")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Missing model: {model_path}. Train first with python3 Trainings/train_XGboost.py"
        )

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing data file: {data_path}")

    model = joblib.load(model_path)

    df = pd.read_csv(data_path)

    multi_class_cat_cols = [
        "foundation_type",
        "roof_type",
        "ground_floor_type",
    ]
    for col in multi_class_cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    y = df["damage_grade"]
    X = df.drop(columns=["damage_grade"])

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    print("XGBoost test metrics")
    print(f"Accuracy:    {acc:.4f}")
    print(f"F1-Macro:    {f1_macro:.4f}")
    print(f"F1-Weighted: {f1_weighted:.4f}")


if __name__ == "__main__":
    main()