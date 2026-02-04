import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from deeptab.models import SAINTClassifier

# Generate mock data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    *make_classification(n_samples=1000, n_features=20, n_classes=2), test_size=0.2
)

# Step 1: Initialize and Train the Model
model = SAINTClassifier(d_model=64, n_layers=8)  # Initialize SAINTClassifier
model.fit(X_train, y_train)  # Train using the sklearn-compatible fit method

# Step 2: Save the Model's Parameters
save_path = Path("saved_model.pth")
if model.task_model and model.task_model.estimator:
    model.task_model.estimator.save_model(save_path)  # Save the estimator's parameters
    print(f"Model parameters successfully saved to: {save_path}")
else:
    print("Error: TaskModel or its estimator is not properly initialized.")

# Step 3: Load the Saved Model into a New Instance
loaded_model = SAINTClassifier(d_model=64, n_layers=8)
if loaded_model.task_model and loaded_model.task_model.estimator:
    loaded_model.task_model.estimator.load_model(save_path)  # Load into the estimator
    print(f"Model parameters successfully loaded from: {save_path}")

    # Step 4: Make Predictions to Verify Restored Model
    predictions = loaded_model.predict(X_test)
    print("Predictions:", predictions)
else:
    print("Error: TaskModel or its estimator is not properly initialized in the loaded model.")