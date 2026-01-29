import numpy as np
from sklearn.base import clone
import copy
import os
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

class L2BoostingEnsemble:
    """
    Proper L2 Gradient Boosting in logit space (multiclass).
    
    A unified class to train ensemble models on data using gradient boosting.
    Compatible with:
    - xgb_ensemble (multi-output regression)
    - SAINT (multi-output regression)
    - sklearn regressors wrapped for multi-output

    Parameters:
    -----------
    base_models : list
        List of model instances to train. Each model should have fit() and predict() methods.
    learning_rates : list, default=None
        Learning rates for each model. If None, uses 0.1 for all models.
    
    Attributes:
    -----------
    base_models : list
        List of base models
    learning_rates : list
        Learning rates for each model
    models_ : list
        List of trained models
    num_classes_ : int
        Number of classes
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    X_valid : np.ndarray
        Validation features
    y_valid : np.ndarray
        Validation labels
    trained_models : list
        List of trained models
    results : dict
        Dictionary containing training results for each model
    boosting_acc : np.ndarray
        Array of [train_acc, valid_acc] for each boosting step
    """

    def __init__(self, base_models, learning_rates=None):
        self.base_models = base_models
        self.learning_rates = learning_rates or [0.1] * len(base_models)
        self.models_ = []
        self.num_classes_ = None
        
        # Additional attributes for trainer functionality
        self.trained_models = []
        self.results = {}
        self.boosting_acc = None
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.num_classes = None
        
        # Initialize model weights: equal weights for all models
        num_models = len(base_models)
        self.model_weights = np.ones(num_models) / num_models

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    def _onehot(self, y, C):
        y = y.astype(int)
        oh = np.zeros((len(y), C))
        oh[np.arange(len(y)), y] = 1.0
        return oh

    def _softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def _onehot_encode(self, y):
        """
        One-hot encode labels.
        
        Parameters:
        -----------
        y : np.ndarray
            Class labels (0 to num_classes-1)
            
        Returns:
        --------
        np.ndarray : One-hot encoded labels
        """
        y = y.reshape(-1)
        y = y.astype('int')
        y_max = np.max(y)
        y_encode = np.zeros((len(y), y_max + 1)).astype('int')
        for i in range(len(y)):
            y_encode[i, y[i]] = 1
        return y_encode
    
    def _labels_to_proba(self, labels, num_classes):
        """
        Convert class labels to one-hot probability matrix.
        
        Parameters:
        -----------
        labels : np.ndarray
            Class labels
        num_classes : int
            Number of classes
            
        Returns:
        --------
        np.ndarray : One-hot probability matrix
        """
        proba = np.zeros((len(labels), num_classes))
        for i, label in enumerate(labels):
            proba[i, int(label)] = 1.0
        return proba
    
    def _ce_gradient_cal(self, logits, yt):
        """
        Calculate cross-entropy gradients g = p - y
        
        Parameters:
        -----------
        logits : np.ndarray
            Raw logits from model (batch_size, num_classes)
        yt : np.ndarray
            True labels (one-hot encoded or regular)
            
        Returns:
        --------
        np.ndarray : CE gradients (p - y)
        """
        if yt.ndim == 1:
            yt = self._onehot_encode(yt)
        
        # Compute softmax from logits
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        softmax_p = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Gradients: g = p - y
        gradients = softmax_p - yt
        return gradients

    # --------------------------------------------------
    # Core API
    # --------------------------------------------------

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : np.ndarray (n_samples, n_features)
        y : np.ndarray (n_samples,) class labels
        """

        self.num_classes_ = int(np.max(y)) + 1
        Y = self._onehot(y, self.num_classes_)

        # Initial logits = 0 → uniform prior
        F = np.zeros((len(X), self.num_classes_))

        self.models_ = []

        for m, (model, eta) in enumerate(zip(self.base_models, self.learning_rates)):
            print(f"[L2Boost] Training model {m+1}/{len(self.base_models)}")

            # P = softmax(F)
            P = self._softmax(F)

            # Residuals = y - p  (L2 loss on logits)
            R = Y - P

            # Try to clone, but if it fails use the model directly
            try:
                learner = clone(model)
            except (TypeError, AttributeError):
                # For non-sklearn models, use directly
                learner = copy.deepcopy(model)

            # Train as multi-output regression
            learner.fit(X, R)

            # Update logits
            F += eta * learner.predict(X)

            self.models_.append(learner)

        return self

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    def predict_logits(self, X):
        F = np.zeros((len(X), self.num_classes_))
        for model, eta in zip(self.models_, self.learning_rates):
            F += eta * model.predict(X)
        return F

    def predict_proba(self, X):
        return self._softmax(self.predict_logits(X))

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
    
    # --------------------------------------------------
    # Data Loading and Training (from BoostingEnsembleTrainer)
    # --------------------------------------------------
    
    def load_and_preprocess_data(self):
        """
        Load preprocessed data from pickle file.
        """
        print("Loading preprocessed data from pickle file...")
        
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pickle_path = os.path.join(script_dir, 'Data', 'processed_data2.pkl')
        
        # Load preprocessed data from pickle file
        with open(pickle_path, 'rb') as f:
            X_train, X_val, X_test, y_train, y_val, y_test = pickle.load(f)
        
        print(f"Data loaded successfully!")
        print(f"Training data shape: {X_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        
        # Convert to numpy arrays
        self.X_train = X_train.values.astype(np.float32) if hasattr(X_train, 'values') else X_train.astype(np.float32)
        self.y_train = y_train.values.astype(np.int64) if hasattr(y_train, 'values') else y_train.astype(np.int64)
        self.X_valid = X_val.values.astype(np.float32) if hasattr(X_val, 'values') else X_val.astype(np.float32)
        self.y_valid = y_val.values.astype(np.int64) if hasattr(y_val, 'values') else y_val.astype(np.int64)
        
        # Determine number of classes
        self.num_classes = len(np.unique(self.y_train))
        print(f"Number of classes: {self.num_classes}")
    
    def train_models(self, test_size=0.3, random_state=1, verbose=10, learning_rates=None, one_hot=True):
        """
        Train all models using gradient boosting with residuals.
        Each model trains on scaled residuals of previous models' probability predictions.
        
        Parameters:
        -----------
        test_size : float, default=0.3
            Proportion of data to use for validation
        random_state : int, default=1
            Random state for reproducibility
        verbose : int, default=10
            Verbosity level for model training
        learning_rates : list, default=None
            Learning rate for each model (scales residuals). If None, uses 0.6 for all
        one_hot : bool, default=True
            Whether to use one-hot encoding for targets
        """
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_and_preprocess_data() first.")
        
        if learning_rates is None:
            learning_rates = [0.6] * len(self.base_models)
        
        print("\n" + "="*60)
        print("Training Gradient Boosting Ensemble")
        print("="*60)
        
        # One-hot encode labels if needed
        y_train_encoded = self._onehot_encode(self.y_train) if one_hot else self.y_train
        
        # Split data for training and validation
        X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
            self.X_train, y_train_encoded, test_size=test_size, random_state=random_state
        )
        
        boosting_acc = []
        
        for i, model in enumerate(self.base_models):
            model_name = model.__class__.__name__
            learning_rate = learning_rates[i]
            
            print(f"\n[Model {i + 1}/{len(self.base_models)}] Training {model_name}...")
            print("-" * 60)
            
            if i == 0:
                # First model: train on original data
                print("Training first model on original data...")
                
                try:
                    if hasattr(model, 'fit'):
                        model.fit(X_train_split, y_train_split.argmax(1) if one_hot else y_train_split)
                except Exception as e:
                    print(f"Error training model: {e}")
                    raise
                
                # Get logit predictions (before softmax)
                yp_train = model.predict_proba(X_train_split) if hasattr(model, 'predict_proba') else self._labels_to_proba(model.predict(X_train_split), self.num_classes)
                yp_valid = model.predict_proba(X_valid_split) if hasattr(model, 'predict_proba') else self._labels_to_proba(model.predict(X_valid_split), self.num_classes)
                logits_train = np.log(yp_train + 1e-10)
                logits_valid = np.log(yp_valid + 1e-10)
                
                # Calculate accuracy from logits
                preds_train = logits_train.argmax(1)
                preds_valid = logits_valid.argmax(1)
                acc_train = np.mean(preds_train == y_train_split.argmax(1))
                acc_valid = np.mean(preds_valid == y_valid_split.argmax(1))
                
                print(f'Model {i} trained')
                print(f'[**Boosting step {i}**]: [Train]: {acc_train:.3f}]   [Valid]: {acc_valid:.3f}')
                boosting_acc.append([acc_train, acc_valid])
                
                # Add first model to trained models list
                self.trained_models.append(model)
                
            else:
                # Subsequent models: train to predict negative CE gradients in logit space
                print("Computing CE gradients from previous models...")
                
                # Accumulate logits from all previous models (train set)
                logits_train_accumulated = None
                for j in range(i):
                    if hasattr(self.trained_models[j], 'predict'):
                        logits = self.trained_models[j].predict(X_train_split)
                    else:
                        yp = self._labels_to_proba(self.trained_models[j].predict(X_train_split), self.num_classes)
                        logits = np.log(yp + 1e-10)
                    
                    if logits_train_accumulated is None:
                        logits_train_accumulated = logits
                    else:
                        logits_train_accumulated += logits
                
                # Accumulate logits from all previous models (valid set)
                logits_valid_accumulated = None
                for j in range(i):
                    if hasattr(self.trained_models[j], 'predict'):
                        logits = self.trained_models[j].predict(X_valid_split)
                    else:
                        yp = self._labels_to_proba(self.trained_models[j].predict(X_valid_split), self.num_classes)
                        logits = np.log(yp + 1e-10)
                    
                    if logits_valid_accumulated is None:
                        logits_valid_accumulated = logits
                    else:
                        logits_valid_accumulated += logits
                
                # Calculate CE gradients: g = p - y
                gradients_train = self._ce_gradient_cal(logits_train_accumulated, y_train_split)
                gradients_valid = self._ce_gradient_cal(logits_valid_accumulated, y_valid_split)
                
                # Train model to predict negative gradients
                neg_gradients_train = -learning_rate * gradients_train
                neg_gradients_valid = -learning_rate * gradients_valid
                
                print(f"Training model {i} to predict negative CE gradients (learning_rate={learning_rate})...")
                
                try:
                    model.fit(X_train_split, neg_gradients_train.argmax(1))
                except Exception as e:
                    print(f"Error training model: {e}")
                    raise
                
                # Add trained model to list
                self.trained_models.append(model)
                
                # Get weighted accumulated predictions from all models
                yp_train = None
                yp_valid = None
                
                for j in range(i + 1):
                    yp_train_j = self.trained_models[j].predict_proba(X_train_split) if hasattr(self.trained_models[j], 'predict_proba') else self._labels_to_proba(self.trained_models[j].predict(X_train_split), self.num_classes)
                    yp_valid_j = self.trained_models[j].predict_proba(X_valid_split) if hasattr(self.trained_models[j], 'predict_proba') else self._labels_to_proba(self.trained_models[j].predict(X_valid_split), self.num_classes)
                    logits_train_j = np.log(yp_train_j + 1e-10)
                    logits_valid_j = np.log(yp_valid_j + 1e-10)
                    
                    if yp_train is None:
                        yp_train = logits_train_j
                        yp_valid = logits_valid_j
                    else:
                        yp_train = yp_train + logits_train_j
                        yp_valid = yp_valid + logits_valid_j
                
                # Apply softmax to accumulated logits
                exp_train = np.exp(yp_train - np.max(yp_train, axis=1, keepdims=True))
                softmax_train = exp_train / np.sum(exp_train, axis=1, keepdims=True)
                
                exp_valid = np.exp(yp_valid - np.max(yp_valid, axis=1, keepdims=True))
                softmax_valid = exp_valid / np.sum(exp_valid, axis=1, keepdims=True)
                
                # Calculate accuracy on final softmax predictions
                train_labels = softmax_train.argmax(1)
                valid_labels = softmax_valid.argmax(1)
                acc_train = np.mean(train_labels == y_train_split.argmax(1))
                acc_valid = np.mean(valid_labels == y_valid_split.argmax(1))
                
                print(f'Model {i} trained')
                print(f'[**Boosting step {i}**]: [Train]: {acc_train:.3f}]   [Valid]: {acc_valid:.3f}')
                boosting_acc.append([acc_train, acc_valid])
            
            # Store results
            self.results[f"{model_name} (Model {i + 1})"] = {
                'train_accuracy': acc_train,
                'valid_accuracy': acc_valid,
                'model_index': i
            }
        
        self.boosting_acc = np.array(boosting_acc)
        
        print("\n" + "="*60)
        print("Boosting ensemble training completed!")
        print("="*60)
        
    def print_summary(self):
        """
        Print a summary of training results for all models.
        """
        print("\n" + "="*60)
        print("Training Summary")
        print("="*60)
        
        for model_name, result in self.results.items():
            print(f"\n{model_name}:")
            print(f"  Train Accuracy: {result['train_accuracy']:.4f}")
            print(f"  Valid Accuracy: {result['valid_accuracy']:.4f}")
        
    def get_trained_models(self):
        """
        Get list of trained models.
        
        Returns:
        --------
        list : List of trained models
        """
        return self.trained_models
    
    def get_results(self):
        """
        Get training results dictionary.
        
        Returns:
        --------
        dict : Dictionary containing results for each model
        """
        return self.results
    
    def predict_ensemble(self, X):
        """
        Make ensemble predictions by combining all model predictions.
        
        Parameters:
        -----------
        X : np.ndarray
            Input features
            
        Returns:
        --------
        np.ndarray : Ensemble predictions (sum of all model predictions)
        """
        if len(self.trained_models) == 0:
            raise ValueError("No trained models available. Train models first.")
        
        # Use probabilities for ensemble prediction
        ensemble_proba = self.predict_ensemble_proba(X)
        
        if ensemble_proba is None:
            raise ValueError("No models with predict_proba available.")
        
        # Return class with highest accumulated probability
        return ensemble_proba.argmax(axis=1)
    
    def predict_ensemble_proba(self, X):
        """
        Get weighted ensemble probability predictions by accumulating logits from all models.
        
        Accumulates weighted logits from all models, then applies softmax once at the end.
        
        Parameters:
        -----------
        X : np.ndarray
            Input features
            
        Returns:
        --------
        np.ndarray : Ensemble probability predictions after softmax
        """
        if len(self.trained_models) == 0:
            raise ValueError("No trained models available. Train models first.")
        
        ensemble_logits = None
        
        for model_idx, model in enumerate(self.trained_models):
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                logits = np.log(proba + 1e-10)
            else:
                proba = self._labels_to_proba(model.predict(X), self.num_classes)
                logits = np.log(proba + 1e-10)
            
            if ensemble_logits is None:
                ensemble_logits = logits
            else:
                ensemble_logits += logits
        
        # Apply softmax to accumulated logits once at the end
        exp_logits = np.exp(ensemble_logits - np.max(ensemble_logits, axis=1, keepdims=True))
        ensemble_proba = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        return ensemble_proba
