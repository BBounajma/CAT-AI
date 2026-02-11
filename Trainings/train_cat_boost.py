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
from catboost import CatBoostClassifier

# Script path setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Data'))

# Load data - adjust path as per your setup
df = pd.read_csv("Data/processed_new_data2.csv")  # Linux version
#df = pd.read_csv("Data\processed_new_data2.csv")  # Windows version


#Label encode the categorical features
multi_class_cat_cols = [
    'foundation_type',
    'roof_type',
    'ground_floor_type'
]

y=df["damage_grade"] 

X_processed = df.drop("damage_grade", axis=1)



# Split data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# Hyperparameter Optimization for CatBoost using GridSearchCV
print("Starting hyperparameter optimization for CatBoost...")

# Define parameter grid for CatBoost (optimized for faster search)
param_grid = {
    'iterations': [100, 200],
    'depth': [4, 6],
    'learning_rate': [0.01, 0.1],
    'l2_leaf_reg': [1, 5],
    'subsample': [0.7, 1.0],
}

# Create CatBoost classifier base model with cat_features as column names
cat_base = CatBoostClassifier(
    random_state=42,
    verbose=0,
    cat_features=multi_class_cat_cols,
    bootstrap_type='Bernoulli'
)

# Perform GridSearchCV for hyperparameter optimization
grid_search = GridSearchCV(
    estimator=cat_base,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

# Fit grid search on training data
grid_search.fit(X_train, y_train)

# Get best model
cat_model = grid_search.best_estimator_

print(f"\nBest parameters found: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}") 

# Evaluate model
train_accuracy = cat_model.score(X_train, y_train)
valid_accuracy = cat_model.score(X_valid, y_valid)
test_accuracy = cat_model.score(X_test, y_test)
print(f"\nCatBoost Classifier - Train Accuracy: {train_accuracy:.4f}")
print(f"CatBoost Classifier - Validation Accuracy: {valid_accuracy:.4f}")
print(f"CatBoost Classifier - Test Accuracy: {test_accuracy:.4f}")

# Save the trained model and grid search results
models_dir = os.path.join(os.path.dirname(__file__), '..', 'Models')
os.makedirs(models_dir, exist_ok=True)
joblib.dump(cat_model, os.path.join(models_dir, 'cat_classifier_model.joblib'))
joblib.dump(grid_search, os.path.join(models_dir, 'cat_grid_search.joblib'))
