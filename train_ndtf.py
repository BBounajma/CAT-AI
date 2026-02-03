import os
if os.environ.get("FORCE_SINGLE_GPU", "1") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"

import pandas as pd
import numpy as np
import sys
import joblib
import torch
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from deeptab.models import NDTFClassifier

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

# Get number of classes
n_classes = len(np.unique(y))
print(f"Number of classes: {n_classes}")
print(f"Classes: {np.unique(y)}")

# Initialize the NDTF classifier

classifier = NDTFClassifier(
    min_depth=4,
    max_depth=16,
    temperature=0.1,
    node_sampling=0.3,
    lamda=0.3,
    n_ensembles=12,
    penalty_factor=1e-8
)


# Train the classifier
classifier.fit(X_train, y_train, X_val=X_valid, y_val=y_valid, max_epochs=10, batch_size=256)

# Save preprocessing components
preprocessing_info = {
    'scaler': scaler,
    'numerical_cols': num_cols,
    'categorical_cols': multi_class_cat_cols
}
joblib.dump(preprocessing_info, "preprocessor.joblib")
print("Preprocessor saved to 'preprocessor.joblib'")

# Save the trained classifier with joblib
joblib.dump(classifier, "trained_ndtf_gpu.joblib")
print("GPU classifier saved to 'trained_ndtf_gpu.joblib'")

# Move model to CPU and save CPU version
classifier.task_model.to('cpu')
joblib.dump(classifier, "trained_ndtf_cpu.joblib")
print("CPU classifier saved to 'trained_ndtf_cpu.joblib'")

# Evaluate on test set
print("\nEvaluating on test set...")
try:
    test_predictions = classifier.predict(X_test)
    from sklearn.metrics import accuracy_score, classification_report
    test_accuracy = accuracy_score(y_test, test_predictions)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_predictions))
except Exception as e:
    print(f"Warning: Could not evaluate on test set: {e}")
    print("Model saved successfully anyway.")

print("\n" + "="*70)
print("Training Complete!")
print("="*70)
print("\nSaved files:")
print("  - preprocessor.joblib")
print("  - trained_ndtf_gpu.joblib")
print("  - trained_ndtf_cpu.joblib")