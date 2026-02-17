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
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    
    # Create synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                              n_redundant=5, n_classes=3, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                         random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, 
                                                       random_state=42)
    
    # Create and train base classifiers
    clf1 = RandomForestClassifier(n_estimators=100, random_state=42)
    clf2 = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf3 = LogisticRegression(max_iter=1000, random_state=42)
    
    clf1.fit(X_train, y_train)
    clf2.fit(X_train, y_train)
    clf3.fit(X_train, y_train)
    
    # Print individual classifier scores
    print("Individual Classifier Scores:")
    print(f"Random Forest: {clf1.score(X_test, y_test):.4f}")
    print(f"Gradient Boosting: {clf2.score(X_test, y_test):.4f}")
    print(f"Logistic Regression: {clf3.score(X_test, y_test):.4f}")
    print()
    
    # Create ensemble learner with different methods
    for method in ['uniform', 'grid_search', 'optimization']:
        print(f"\n{'='*50}")
        print(f"Method: {method}")
        print(f"{'='*50}")
        
        classifiers = [
            ('Random Forest', clf1),
            ('Gradient Boosting', clf2),
            ('Logistic Regression', clf3)
        ]
        
        ensemble = WeightedEnsembleLearner(classifiers, method=method, metric='accuracy',
                                           use_proba=True, random_state=42)
        
        # Train ensemble
        ensemble.fit(X_train, y_train, X_val, y_val)
        
        # Evaluate
        train_score = ensemble.score(X_train, y_train)
        test_score = ensemble.score(X_test, y_test)
        
        print(f"Training Accuracy: {train_score:.4f}")
        print(f"Test Accuracy: {test_score:.4f}")
        print(f"Learned Weights: {ensemble.get_weights()}")
