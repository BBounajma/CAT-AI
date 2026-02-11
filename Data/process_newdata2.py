"""
Minimal processing of new_data2.csv for model training.
Compatible with tree based models, catboost and neural networks
from widedeep.
Handles missing values by imputing with median (numerical) and mode (categorical).
Binary features are already one hot encoded.
Label encoded of multi-class features is done at training time,
as it is not compatible with catboost.

Scaling of numerical features is not done here, 
as tree based models do not require it.
"""



import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Data'))


df = pd.read_csv("Data/new_data2.csv")  # Linux version
#df = pd.read_csv("Data\new_data2.csv")  # Windows version

# Rename columns
df.rename(columns={
    'roof_type(Bamboo/Timber-Heavy roof=0; Bamboo/Timber-Light roof=1; RCC/RB/RBC=2)': 'roof_type',
    'district_distance_to_earthquakecenter(mi)': 'distance'
}, inplace=True)

# Feature and target setup
num_cols = [
    "PGA_g",
    "count_floors_pre_eq",
    "age_building",
    "plinth_area_sq_ft",
    "per-height_ft_pre_eq"

]

binary_cols=[
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

multi_class_cat_cols = [
    'foundation_type',
    'roof_type',
    'ground_floor_type'
]

target = "damage_grade"

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

# Drop unnecessary columns
df.drop(columns=to_drop, inplace=True)

# Adjust target variable
y = df["damage_grade"] - 1
df.drop("damage_grade", axis=1, inplace=True)


# Make a copy for preprocessing
df_processed = df.copy()

# Impute categorical features with mode
for col in multi_class_cat_cols:
    df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])


# Impute numerical features with median
df_processed[num_cols] = df_processed[num_cols].fillna(df_processed[num_cols].median())



X_processed = df_processed

# Save the processed data to a new CSV file
processed_data = pd.concat([X_processed, y], axis=1)
processed_data.to_csv("Data/processed_new_data2.csv", index=False)