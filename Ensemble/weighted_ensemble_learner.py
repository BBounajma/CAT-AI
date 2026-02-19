"""
Weighted Ensemble Learner
A class that learns optimal weights for a list of classifiers to improve overall accuracy.
Supports multiple weight optimization strategies.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')


class WeightedEnsembleLearner:
    """
    A meta-learner that optimizes weights for an ensemble of pre-trained classifiers.
    
    Parameters
    ----------
    classifiers : list of tuples
        List of (name, classifier) tuples where each classifier has fit, predict, 
        and predict_proba methods (or predict_proba for probability-based ensemble)
    
    method : str, default='optimization'
        Weight optimization method:
        - 'optimization': Uses scipy.optimize.minimize to find optimal weights
        - 'grid_search': Uses grid search with validation set
        - 'uniform': Uses uniform weights (baseline)
    
    metric : str, default='accuracy'
        Metric to optimize: 'accuracy', 'f1', or 'auc'
    
    use_proba : bool, default=True
        If True, uses predict_proba for ensemble (soft voting)
        If False, uses predict for ensemble (hard voting)
    
    random_state : int, default=None
        Random seed for reproducibility
    """
    
    def __init__(self, classifiers, method='optimization', metric='accuracy', 
                 use_proba=True, random_state=None):
        self.classifiers = classifiers
        self.method = method
        self.metric = metric
        self.use_proba = use_proba
        self.random_state = random_state
        self.n_classifiers = len(classifiers)
        self.weights = np.ones(self.n_classifiers) / self.n_classifiers
        self.is_trained = False
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _get_metric_function(self):
        """Return the metric function based on self.metric."""
        if self.metric == 'accuracy':
            return accuracy_score
        elif self.metric == 'f1':
            return f1_score
        elif self.metric == 'auc':
            return roc_auc_score
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _get_predictions(self, X, method='predict'):
        """Get predictions from all classifiers."""
        predictions = []
        for name, clf in self.classifiers:
            if method == 'predict_proba':
                pred = clf.predict_proba(X)
            else:
                pred = clf.predict(X)
            predictions.append(pred)
        return predictions
    
    def _ensemble_predict(self, predictions, weights=None):
        """
        Combine predictions using weights.
        
        Parameters
        ----------
        predictions : list of arrays
            Predictions from each classifier
        weights : array-like, optional
            Weights for each classifier. If None, uses self.weights
        
        Returns
        -------
        ensemble_pred : array
            Ensemble predictions
        """
        if weights is None:
            weights = self.weights
        
        # Normalize weights to sum to 1
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        if self.use_proba and predictions[0].ndim == 2:
            # Soft voting: weighted average of probabilities
            weighted_proba = np.zeros_like(predictions[0])
            for i, pred in enumerate(predictions):
                weighted_proba += weights[i] * pred
            ensemble_pred = np.argmax(weighted_proba, axis=1)
        else:
            # Hard voting: weighted majority vote
            ensemble_pred = np.zeros(len(predictions[0]))
            for i, pred in enumerate(predictions):
                ensemble_pred += weights[i] * (pred == pred)
            ensemble_pred = np.argmax(np.column_stack([
                np.sum([weights[i] * (pred == c) for i, pred in enumerate(predictions)], axis=0)
                for c in np.unique(predictions[0])
            ]), axis=1)
        
        return ensemble_pred
    
    def _ensemble_predict_simple(self, predictions, weights=None):
        """Simplified ensemble prediction for soft voting."""
        if weights is None:
            weights = self.weights
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Weighted average of probabilities
        weighted_proba = np.zeros_like(predictions[0], dtype=np.float64)
        for i, pred in enumerate(predictions):
            weighted_proba += weights[i] * pred
        
        return np.argmax(weighted_proba, axis=1)
    
    def _objective_function(self, weights, predictions, y_true):
        """
        Objective function to minimize (negative metric, since we want to maximize).
        
        Parameters
        ----------
        weights : array
            Weights for each classifier
        predictions : list of arrays
            Predictions from each classifier
        y_true : array
            True labels
        
        Returns
        -------
        loss : float
            Negative performance metric (for minimization)
        """
        # Normalize weights
        weights = np.abs(weights)  # Ensure non-negative weights
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        # Get ensemble prediction
        ensemble_pred = self._ensemble_predict_simple(predictions, weights)
        
        # Calculate metric
        metric_func = self._get_metric_function()
        try:
            if self.metric == 'auc':
                # For multi-class, use weighted average
                score = metric_func(y_true, ensemble_pred, multi_class='weighted', 
                                   average='weighted', labels=np.unique(y_true))
            elif self.metric == 'f1':
                score = metric_func(y_true, ensemble_pred, average='weighted')
            else:
                score = metric_func(y_true, ensemble_pred)
        except:
            score = accuracy_score(y_true, ensemble_pred)
        
        return -score  # Negative because we minimize
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the ensemble to learn optimal weights.
        
        Parameters
        ----------
        X_train : array-like, shape (n_samples, n_features)
            Training features
        y_train : array-like, shape (n_samples,)
            Training labels
        X_val : array-like, optional
            Validation features for weight optimization
        y_val : array-like, optional
            Validation labels for weight optimization
        
        Returns
        -------
        self : returns self
        """
        # Use validation set if provided, otherwise use training set
        X_opt = X_val if X_val is not None else X_train
        y_opt = y_val if y_val is not None else y_train
        
        # Get predictions from all classifiers
        predictions = self._get_predictions(X_opt, method='predict_proba' if self.use_proba else 'predict')
        
        if self.method == 'optimization':
            self._fit_optimization(predictions, y_opt)
        elif self.method == 'grid_search':
            self._fit_grid_search(predictions, y_opt)
        elif self.method == 'uniform':
            self.weights = np.ones(self.n_classifiers) / self.n_classifiers
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.is_trained = True
        return self
    
    def _fit_optimization(self, predictions, y_true):
        """Fit weights using scipy optimization."""
        # Initial weights: uniform
        initial_weights = np.ones(self.n_classifiers) / self.n_classifiers
        
        # Constraints: weights sum to 1 and are non-negative
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(self.n_classifiers)]
        
        # Optimize
        result = minimize(
            self._objective_function,
            initial_weights,
            args=(predictions, y_true),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100}
        )
        
        self.weights = result.x / np.sum(result.x)
        print(f"Optimization converged: {result.success}")
        print(f"Learned weights: {self.weights}")
    
    def _fit_grid_search(self, predictions, y_true):
        """Fit weights using grid search."""
        n_steps = 11  # 0.0, 0.1, 0.2, ..., 1.0
        best_score = -np.inf
        best_weights = np.ones(self.n_classifiers) / self.n_classifiers
        
        # For simplicity, only grid search with 2-3 classifiers
        if self.n_classifiers > 3:
            print(f"Grid search is too slow for {self.n_classifiers} classifiers. Using uniform weights.")
            self.weights = best_weights
            return
        
        def generate_weights(n_classifiers, n_steps):
            """Generate all possible weight combinations."""
            if n_classifiers == 1:
                yield np.array([1.0])
            elif n_classifiers == 2:
                for w1 in np.linspace(0, 1, n_steps):
                    yield np.array([w1, 1 - w1])
            elif n_classifiers == 3:
                for w1 in np.linspace(0, 1, n_steps):
                    for w2 in np.linspace(0, 1 - w1, n_steps):
                        yield np.array([w1, w2, 1 - w1 - w2])
        
        for weights in generate_weights(self.n_classifiers, n_steps):
            score = -self._objective_function(weights, predictions, y_true)
            if score > best_score:
                best_score = score
                best_weights = weights
        
        self.weights = best_weights
        print(f"Grid search best score: {best_score:.4f}")
        print(f"Learned weights: {self.weights}")
    
    def predict(self, X):
        """
        Make predictions using the trained ensemble.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict
        
        Returns
        -------
        predictions : array, shape (n_samples,)
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction. Call fit() first.")
        
        predictions = self._get_predictions(X, method='predict_proba' if self.use_proba else 'predict')
        return self._ensemble_predict_simple(predictions, self.weights)
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the trained ensemble.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples to predict
        
        Returns
        -------
        probabilities : array, shape (n_samples, n_classes)
            Predicted class probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction. Call fit() first.")
        
        predictions = self._get_predictions(X, method='predict_proba')
        
        # Weighted average of probabilities
        weights = self.weights / np.sum(self.weights)
        weighted_proba = np.zeros_like(predictions[0], dtype=np.float64)
        for i, pred in enumerate(predictions):
            weighted_proba += weights[i] * pred
        
        return weighted_proba
    
    def score(self, X, y):
        """
        Calculate accuracy score on test data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Test features
        y : array-like, shape (n_samples,)
            Test labels
        
        Returns
        -------
        score : float
            Accuracy score
        """
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
    def get_weights(self):
        """Return the learned weights for each classifier."""
        return dict(zip([name for name, _ in self.classifiers], self.weights))


# Example usage
if __name__ == "__main__":
    import os
    import joblib
    import torch
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, f1_score
    from pytorch_widedeep.preprocessing import TabPreprocessor
    from pytorch_widedeep.models import SAINT
    import sys
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    models_dir = os.path.join(project_root, 'Models')
    data_path = os.path.join(project_root, 'Data/processed_new_data2.csv')
    
    print("="*70)
    print("Weighted Ensemble Learner: XGBoost + Random Forest + SAINT")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(data_path)
    y = df["damage_grade"]
    X = df.drop(columns=["damage_grade"])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Load models
    print("\nLoading trained models...")
    
    # 1. XGBoost
    xgb_path = os.path.join(models_dir, 'xgb_classifier_model.joblib')
    if os.path.exists(xgb_path):
        xgb_model = joblib.load(xgb_path)
        print(f"✓ XGBoost loaded from {xgb_path}")
    else:
        print(f"✗ XGBoost model not found at {xgb_path}")
        print("  Run: python Trainings/train_XGboost.py")
        xgb_model = None
    
    # 2. Random Forest
    rf_path = os.path.join(models_dir, 'rf_classifier_model.joblib')
    if os.path.exists(rf_path):
        rf_model = joblib.load(rf_path)
        print(f"✓ Random Forest loaded from {rf_path}")
    else:
        print(f"✗ Random Forest model not found at {rf_path}")
        print("  Run: python Trainings/train_rf.py")
        rf_model = None
    
    # 3. SAINT - requires custom wrapper
    class SAINTClassifier(torch.nn.Module):
        """Standard classification head for SAINT."""
        def __init__(self, saint_model, n_classes=5):
            super().__init__()
            self.saint = saint_model
            self.fc = torch.nn.Linear(saint_model.output_dim, n_classes)

        def forward(self, x):
            h = self.saint(x)
            return self.fc(h)
    
    class SAINTWrapper:
        """Wrapper to make SAINT compatible with sklearn interface."""
        def __init__(self, model, preprocessor, device='cpu'):
            self.model = model
            self.preprocessor = preprocessor
            self.device = device
            self.n_classes = 5
            
        def predict_proba(self, X):
            """Return class probabilities."""
            self.model.eval()
            X_processed = self.preprocessor.transform(X)
            X_tensor = torch.tensor(X_processed, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                logits = self.model(X_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                
            return probs
        
        def predict(self, X):
            """Return class predictions."""
            probs = self.predict_proba(X)
            return np.argmax(probs, axis=1)
    
    # Try to load SAINT model
    saint_model_path = os.path.join(models_dir, 'saint_model.pt')
    saint_preprocessor_path = os.path.join(models_dir, 'saint_tab_preprocessor.joblib')
    
    saint_wrapper = None
    if os.path.exists(saint_model_path) and os.path.exists(saint_preprocessor_path):
        try:
            # Load preprocessor
            saint_preprocessor = joblib.load(saint_preprocessor_path)
            
            # Recreate SAINT architecture
            cat_embed_cols = ["foundation_type", "roof_type", "ground_floor_type"]
            continuous_cols = [c for c in X.columns if c not in cat_embed_cols]
            
            saint = SAINT(
                column_idx=saint_preprocessor.column_idx,
                cat_embed_input=saint_preprocessor.cat_embed_input,
                continuous_cols=continuous_cols,
                input_dim=32,
                n_heads=4,
                n_blocks=3,
                attn_dropout=0.1,
                ff_dropout=0.1,
            )
            
            saint_model = SAINTClassifier(saint)
            saint_model.load_state_dict(torch.load(saint_model_path, map_location=torch.device('cpu')))
            saint_model.eval()
            
            saint_wrapper = SAINTWrapper(saint_model, saint_preprocessor)
            print(f"✓ SAINT loaded from {saint_model_path}")
        except Exception as e:
            print(f"✗ Error loading SAINT: {e}")
    else:
        print(f"✗ SAINT model not found")
        print(f"  Expected: {saint_model_path}")
        print(f"  Run train_saint_v2.py and add model saving code")
    
    # Collect available models
    classifiers = []
    if xgb_model is not None:
        classifiers.append(('XGBoost', xgb_model))
    if rf_model is not None:
        classifiers.append(('Random Forest', rf_model))
    if saint_wrapper is not None:
        classifiers.append(('SAINT', saint_wrapper))
    
    if len(classifiers) == 0:
        print("\n✗ No models available. Please train models first.")
        sys.exit(1)
    
    print(f"\nUsing {len(classifiers)} models for ensemble")
    
    # Evaluate individual models
    print("\n" + "="*70)
    print("Individual Model Performance on Test Set")
    print("="*70)
    for name, clf in classifiers:
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        print(f"{name:20s} | Acc: {acc:.4f} | F1-Macro: {f1_macro:.4f} | F1-Weighted: {f1_weighted:.4f}")
    
    # Train ensemble with different methods
    print("\n" + "="*70)
    print("Ensemble Performance")
    print("="*70)
    
    for method in ['uniform', 'grid_search', 'optimization']:
        print(f"\n{'-'*70}")
        print(f"Method: {method.upper()}")
        print(f"{'-'*70}")
        
        ensemble = WeightedEnsembleLearner(
            classifiers, 
            method=method, 
            metric='f1',  # Optimize for F1 score
            use_proba=True, 
            random_state=42
        )
        
        # Train ensemble on validation set
        ensemble.fit(X_train, y_train, X_val, y_val)
        
        # Evaluate on test set
        y_pred_ensemble = ensemble.predict(X_test)
        acc = accuracy_score(y_test, y_pred_ensemble)
        f1_macro = f1_score(y_test, y_pred_ensemble, average='macro')
        f1_weighted = f1_score(y_test, y_pred_ensemble, average='weighted')
        
        print(f"\nTest Accuracy:       {acc:.4f}")
        print(f"Test F1-Macro:       {f1_macro:.4f}")
        print(f"Test F1-Weighted:    {f1_weighted:.4f}")
        print(f"\nLearned Weights:")
        for model_name, weight in ensemble.get_weights().items():
            print(f"  {model_name:20s}: {weight:.4f}")
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)

