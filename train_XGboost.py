import os
import pandas as pd
import numpy as np
import sys
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier

# Script path setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Data'))

# Load data - adjust path as per your setup
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
    "per-height_ft_pre_eq",
    # Binary indicator features (already 0/1, no need for one-hot encoding)
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
# Removed cat_cols definition - now using multi_class_cat_cols directly in preprocessing

# Drop unnecessary columns
df.drop(columns=to_drop, inplace=True)

# Adjust target variable
y = df["damage_grade"] - 1
df.drop("damage_grade", axis=1, inplace=True)

# Preprocessing: Separate numerical and categorical features
multi_class_cat_cols = [
    'foundation_type',
    'roof_type',
    'ground_floor_type'
]

# Make a copy for preprocessing
df_processed = df.copy()

# Impute categorical features
for col in multi_class_cat_cols:
    df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])

# Convert categorical columns to category dtype (deeptab recognizes this)
for col in multi_class_cat_cols:
    df_processed[col] = df_processed[col].astype('category')

# Impute numerical features with median
df_processed[num_cols] = df_processed[num_cols].fillna(df_processed[num_cols].median())

# Scale numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_processed[num_cols] = scaler.fit_transform(df_processed[num_cols])

X_processed = df_processed

# Split data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# Hyperparameter Optimization for XGBoost using GridSearchCV
print("Starting hyperparameter optimization for XGBoost...")

# Define parameter grid for XGBoost (optimized for faster search)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0],
}

# Create XGBoost classifier base model
xgb_base = XGBClassifier(eval_metric='mlogloss', enable_categorical=True, random_state=42)

# Perform GridSearchCV for hyperparameter optimization
grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

# Fit grid search on training data
grid_search.fit(X_train, y_train)

# Get best model
xgb_model = grid_search.best_estimator_

print(f"\nBest parameters found: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}") 

# Evaluate model
train_accuracy = xgb_model.score(X_train, y_train)
valid_accuracy = xgb_model.score(X_valid, y_valid)
test_accuracy = xgb_model.score(X_test, y_test)
print(f"\nXGBoost Classifier - Train Accuracy: {train_accuracy:.4f}")
print(f"XGBoost Classifier - Validation Accuracy: {valid_accuracy:.4f}")
print(f"XGBoost Classifier - Test Accuracy: {test_accuracy:.4f}")

# Save the trained model and grid search results
joblib.dump(xgb_model, 'xgb_classifier_model.joblib')
joblib.dump(grid_search, 'xgb_grid_search.joblib')
print("\nModel and grid search results saved successfully!")