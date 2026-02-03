import pandas as pd
import numpy as np
import sys
import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Data'))

# Load data
df = pd.read_csv(r"Data\new_data2.csv")
df.rename(columns={'roof_type(Bamboo/Timber-Heavy roof=0; Bamboo/Timber-Light roof=1; RCC/RB/RBC=2)':'roof_type','district_distance_to_earthquakecenter(mi)':'distance'}, inplace=True)

to_drop = ["technical_solution_proposed", "condition_post_eq", "vdcmun_id", "ward_id", "land_surface_condition", "has_superstructure_mud_mortar_brick", "position", "has_superstructure_adobe_mud"]

df.drop(columns=to_drop, inplace=True)
y = df["damage_grade"] - 1
df.drop("damage_grade", axis=1, inplace=True)

# Load preprocessor
preprocessor = joblib.load("preprocessor.joblib")

# Try loading as joblib first
try:
    model = joblib.load("trained_saint_cpu.pth")
except:
    # Fall back to torch load
    try:
        model = torch.load("trained_saint_cpu.pth", map_location=torch.device('cpu'), weights_only=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("File may be corrupted. Check if file exists and is valid.")
        sys.exit(1)

model.eval()  # Set to evaluation mode

# Transform data
X_processed = preprocessor.transform(df)

# Split data (same split as training)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)

# Convert to torch tensors
X_test_tensor = torch.FloatTensor(X_test)

# Make predictions
with torch.no_grad():
    y_pred = model(X_test_tensor).argmax(dim=1).numpy()

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
cm = confusion_matrix(y_test, y_pred)

# Print results
print("=" * 50)
print("MODEL EVALUATION ON TEST SET")
print("=" * 50)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("=" * 50)