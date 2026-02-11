"""
Minimal processing of Turkey_data.csv for model testing.
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

df = pd.read_csv("Data/Turkey_data.csv", sep=";", decimal=",", skiprows=[0,2])  # Linux version
#df = pd.read_csv("Data/Turkey_data.csv", sep=";", decimal=",")  # Windows version

#print the names of the columns in the dataset
print("Columns in the dataset:")
print(df.columns.tolist())

# Keep only the relevant columns
df = df[[
    "Age",
    "No Stories",
    "PGA_max",
    "Total Height",
    "Floor Area",
    "Structural Damage (5-class)",
]]

# Convert meters to feet

df["Total Height"] = df["Total Height"] * 3.28084
df["Floor Area"] = df["Floor Area"] * 10.7639

# Compute the height per floor
df["per-height_ft_pre_eq"] = df["Total Height"] / df["No Stories"]

# Enconde the target variable to match Nepal data
damage_grade_mapping = {
    "N": 0,
    "Se": 1,
    "M": 2,
    "E": 3,
    "C": 4
}

df["Structural Damage (5-class)"] = df["Structural Damage (5-class)"].map(damage_grade_mapping)

# Rename columns to match Nepal data
df.rename(columns={
    "Age": "age_building",
    "No Stories": "count_floors_pre_eq",
    "PGA_max": "PGA_g",
    "Floor Area": "plinth_area_sq_ft",
    "Structural Damage (5-class)": "damage_grade"
}, inplace=True)

# Add the missing columns with realistic default values

df["foundation_type"] = 4
df["roof_type"] = 2
df["ground_floor_type"] = 4

df["has_superstructure_mud_mortar_stone"]=0
df["has_superstructure_stone_flag"]=0
df["has_superstructure_cement_mortar_stone"]=1
df["has_superstructure_cement_mortar_brick"]=1
df["has_superstructure_timber"]=0
df["has_superstructure_bamboo"]=0
df["has_superstructure_other"]=0

#Pre 2000 buildings are assumed to be non-engineered
df["has_superstructure_rc_non_engineered"] = (df["age_building"] > 23).astype(int)
df["has_superstructure_rc_engineered"] = (df["age_building"] <= 23).astype(int)


#print the names of the columns in the processed dataset
print("Columns in the processed dataset:")
print(df.columns.tolist())

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

y = df["damage_grade"] 
df.drop("damage_grade", axis=1, inplace=True)


# Make a copy for preprocessing
df_processed = df.copy()

# Impute categorical features with mode
for col in multi_class_cat_cols + binary_cols:
    df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])


# Impute numerical features with median
df_processed[num_cols] = df_processed[num_cols].fillna(df_processed[num_cols].median())



X_processed = df_processed

# Save the processed data to a new CSV file
processed_data = pd.concat([X_processed, y], axis=1)
processed_data.to_csv("Data/processed_Turkey_data.csv", index=False)