from ast import Raise
import sys
import os
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier, XGBRegressor


# Add paths to modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Data'))

from Data.preprocessor import Preprocessor
from Models.Saint.wrapper import SaintWrapper
from Models.model_XG_boost import multiclass_XG_regressor_model, xgb_ensemble

# Import L2BoostingEnsemble so pickle can find it when loading models
try:
    from L2BoostingEnsemble import L2BoostingEnsemble
except ImportError:
    raise ImportError("L2BoostingEnsemble module not found. Ensure L2BoostingEnsemble.py is in the same directory.")


if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'new_l2boost_xgb_saint.pkl')
    

    # Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_data_path = os.path.join(script_dir, 'Data', 'processed_new_data2.pkl')
        
    print(f"\nLoading data from: {pickle_data_path}")
    with open(pickle_data_path, 'rb') as f:
        X_train, X_val, X_test, y_train, y_val, y_test = pickle.load(f)
        
    # Convert to numpy arrays
    X_train = X_train.values.astype(np.float32) if hasattr(X_train, 'values') else X_train.astype(np.float32)
    y_train = y_train.values.astype(np.int64) if hasattr(y_train, 'values') else y_train.astype(np.int64)
    X_valid = X_val.values.astype(np.float32) if hasattr(X_val, 'values') else X_val.astype(np.float32)
    y_valid = y_val.values.astype(np.int64) if hasattr(y_val, 'values') else y_val.astype(np.int64)
        
    print(f"Data loaded successfully!")
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_valid.shape}")

    # Check if trained model exists
    if os.path.exists(model_path):
        print("="*60)
        print("Loading Trained Boosting Ensemble Model")
        print("="*60)
        print(f"\nLoading model from: {model_path}")
        
        # Load the trained model
        with open(model_path, 'rb') as f:
            trained_model = pickle.load(f)
        
        print(f"Successfully loaded trained ensemble model")
        print(f"Model type: {type(trained_model).__name__}")
        
        # Evaluate ensemble on training and validation data
        print("\n" + "="*60)
        print("Ensemble Evaluation on Full Data")
        print("="*60)
        
        # Get ensemble predictions on training data
        train_proba = trained_model.predict_ensemble_proba(X_train)
        train_pred  = trained_model.predict_ensemble(X_train)

        # Validation set
        valid_proba = trained_model.predict_ensemble_proba(X_valid)
        valid_pred  = trained_model.predict_ensemble(X_valid)

        print("\nFull Boosting Ensemble Accuracy:")
        print(f"  Training Accuracy: {accuracy_score(y_train, train_pred):.4f}")
        print(f"  Validation Accuracy: {accuracy_score(y_valid, valid_pred):.4f}")

        print("\nFull Boosting Ensemble LogLoss:")
        print(f"  Training LogLoss: {log_loss(y_train, train_proba):.4f}")
        print(f"  Validation LogLoss: {log_loss(y_valid, valid_proba):.4f}")
        
        print("\n" + "="*60)
        print("Model loaded and evaluated successfully!")
        print("="*60)
    
    else:
        print("="*60)
        print("Training New Boosting Ensemble Model")
        print("="*60)
        print(f"\nNo trained model found at: {model_path}")
        print("Proceeding with training new models...\n")

        models = [
            xgb_ensemble(),
            SaintWrapper(
                num_categorical=12,
                num_continuous=5,
                num_classes=5,
                cat_dims=[2]*12,
                task="regression"
            ),
            RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
        ]

        boost = L2BoostingEnsemble(
            base_models=models,
            learning_rates=[1.0, 0.3, 0.2]
        )

        boost.fit(X_train, y_train)

        # Save the trained model to pickle file
        print("\n" + "="*60)
        print("Saving Trained Model")
        print("="*60)

        with open(model_path, "wb") as f:
            pickle.dump(boost, f)

        print(f"Model saved to: {model_path}")
        
        # Evaluate ensemble on full training and validation data
        print("\n" + "="*60)
        print("Ensemble Evaluation:")
        print("="*60)
        
        # Get logits and convert to probabilities
        train_logits = boost.predict_logits(X_train)
        valid_logits = boost.predict_logits(X_valid)
        
        # Convert logits to probabilities using softmax
        train_proba = np.exp(train_logits) / np.sum(np.exp(train_logits), axis=1, keepdims=True)
        valid_proba = np.exp(valid_logits) / np.sum(np.exp(valid_logits), axis=1, keepdims=True)

        # Convert to labels
        train_pred = train_proba.argmax(axis=1)
        valid_pred = valid_proba.argmax(axis=1)

        print("=== L2Boosting Evaluation ===")
        print(f"Train Accuracy : {accuracy_score(y_train, train_pred):.4f}")
        print(f"Valid Accuracy : {accuracy_score(y_valid, valid_pred):.4f}")
        print(f"Train LogLoss  : {log_loss(y_train, train_proba):.4f}")
        print(f"Valid LogLoss  : {log_loss(y_valid, valid_proba):.4f}")
        
        print("\n" + "="*60)
        print("Models trained and ready for use!")
        print("="*60)