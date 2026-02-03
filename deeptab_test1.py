import pandas as pd
import numpy as np
import sys
import os
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from deeptab.models import SAINTClassifier


sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Data'))

df=pd.read_csv(r"Data\new_data2.csv") #local version
#df=pd.read_csv("/content/new_data2.csv") #colab version
df.rename(columns={'roof_type(Bamboo/Timber-Heavy roof=0; Bamboo/Timber-Light roof=1; RCC/RB/RBC=2)':'roof_type','district_distance_to_earthquakecenter(mi)':'distance'}, inplace=True)

num_cols=["PGA_g",
          "count_floors_pre_eq",
          "age_building",
          "plinth_area_sq_ft",
          "per-height_ft_pre_eq"
]

target="damage_grade"


to_drop=["technical_solution_proposed",
         "condition_post_eq",
         "vdcmun_id",
         "ward_id",
         "land_surface_condition",
         "has_superstructure_mud_mortar_brick",
         "position",
         "has_superstructure_adobe_mud"
]

cat_cols = ['foundation_type',
    'roof_type',
    'ground_floor_type',
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

df.drop(columns=to_drop, inplace=True)

y = df["damage_grade"] - 1

df.drop("damage_grade", axis=1, inplace=True)

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_cols),
    
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

X_processed= preprocessor.fit_transform(df)

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=42)  

classifier = SAINTClassifier(d_model=32, n_heads=8, n_layers=3, attn_dropout=0.1)

classifier.fit(X_train, y_train, X_val=X_valid, y_val=y_valid, max_epochs=10, batch_size=256)

joblib.dump(preprocessor, "preprocessor.joblib")
joblib.dump(classifier, "saint_model.joblib")