import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
)
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


class WeightedEnsembleLearner:
    """
    A meta-learner that optimizes weights for an ensemble of pre-trained classifiers.
    """

    def __init__(
        self,
        classifiers,
        method="optimization",
        metric="accuracy",
        use_proba=True,
        random_state=None,
    ):
        self.classifiers = classifiers
        self.method = method
        self.metric = metric
        self.use_proba = use_proba
        self.random_state = random_state
        self.n_classifiers = len(classifiers)
        self.weights = np.ones(self.n_classifiers) / self.n_classifiers
        self.is_trained = False

        if not self.use_proba:
            raise NotImplementedError(
                "Hard voting is disabled: use_proba=True is required."
            )

        if random_state is not None:
            np.random.seed(random_state)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _get_predictions(self, X):
        """Get predict_proba outputs from all classifiers."""
        preds = []
        for _, clf in self.classifiers:
            if hasattr(clf, "predict_proba"):
                preds.append(clf.predict_proba(X))
            else:
                preds.append(clf["predict_proba"](X))
        return preds

    def _normalize_weights(self, w):
        w = np.maximum(w, 0.0)
        s = w.sum()
        return w / s if s > 0 else np.ones_like(w) / len(w)

    def _weighted_proba(self, predictions, weights):
        """Weighted average of probabilities."""
        weights = self._normalize_weights(weights)
        proba = np.zeros_like(predictions[0], dtype=np.float64)
        for i, p in enumerate(predictions):
            proba += weights[i] * p
        return proba

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def _objective_function(self, weights, predictions, y_true):
        """
        Smooth objective suitable for optimization.
        """
        proba = self._weighted_proba(predictions, weights)

        try:
            if self.metric == "auc":
                score = roc_auc_score(
                    y_true,
                    proba,
                    multi_class="ovr",
                    average="weighted",
                )
                return -score

            elif self.metric == "f1":
                y_pred = np.argmax(proba, axis=1)
                return -f1_score(y_true, y_pred, average="weighted")

            else:  # accuracy
                y_pred = np.argmax(proba, axis=1)
                return -accuracy_score(y_true, y_pred)

        except Exception:
            # FIX: safe fallback
            return log_loss(y_true, proba)

    # ------------------------------------------------------------------
    # Fit methods
    # ------------------------------------------------------------------

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_opt = X_val if X_val is not None else X_train
        y_opt = y_val if y_val is not None else y_train

        predictions = self._get_predictions(X_opt)

        if self.method == "optimization":
            self._fit_optimization(predictions, y_opt)
        elif self.method == "grid_search":
            self._fit_grid_search(predictions, y_opt)
        elif self.method == "uniform":
            self.weights = np.ones(self.n_classifiers) / self.n_classifiers
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.is_trained = True
        return self

    def _fit_optimization(self, predictions, y_true):
        init = np.ones(self.n_classifiers) / self.n_classifiers

        bounds = [(0.0, 1.0)] * self.n_classifiers
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        result = minimize(
            self._objective_function,
            init,
            args=(predictions, y_true),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 100},
        )

        self.weights = self._normalize_weights(result.x)

    def _fit_grid_search(self, predictions, y_true):
        n_steps = 11
        best_score = -np.inf
        best_weights = np.ones(self.n_classifiers) / self.n_classifiers

        if self.n_classifiers > 3:
            self.weights = best_weights
            return

        grid = np.linspace(0, 1, n_steps)

        if self.n_classifiers == 2:
            for w1 in grid:
                w = np.array([w1, 1 - w1])
                score = -self._objective_function(w, predictions, y_true)
                if score > best_score:
                    best_score, best_weights = score, w

        elif self.n_classifiers == 3:
            for w1 in grid:
                for w2 in grid:
                    w3 = 1 - w1 - w2
                    if w3 < 0:
                        continue  # FIX: enforce simplex
                    w = np.array([w1, w2, w3])
                    score = -self._objective_function(w, predictions, y_true)
                    if score > best_score:
                        best_score, best_weights = score, w

        self.weights = self._normalize_weights(best_weights)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Call fit() first.")

        preds = self._get_predictions(X)
        proba = self._weighted_proba(preds, self.weights)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Call fit() first.")

        preds = self._get_predictions(X)
        return self._weighted_proba(preds, self.weights)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def get_weights(self):
        return dict(zip([n for n, _ in self.classifiers], self.weights))

# Example usage
if __name__ == "__main__":
    import os
    import joblib
    import torch
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, f1_score
    from pytorch_widedeep.models import SAINT, WideDeep
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
    
    def make_torch_tabular_predictor(model, preprocessor, device='cpu', batch_size=256):
        """Return minimal predict/predict_proba callables."""
        model = model.to(device)
        model.eval()

        def predict_proba(X):
            X_processed = preprocessor.transform(X)
            X_tensor = torch.tensor(X_processed, dtype=torch.float32)

            prob_chunks = []
            with torch.no_grad():
                for start in range(0, X_tensor.shape[0], batch_size):
                    end = start + batch_size
                    batch = X_tensor[start:end].to(device)
                    try:
                        logits = model({"deeptabular": batch})
                    except KeyError:
                        logits = model({"X_tab": batch})
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    prob_chunks.append(probs)

            return np.vstack(prob_chunks)

        def predict(X):
            return np.argmax(predict_proba(X), axis=1)

        return {'predict_proba': predict_proba, 'predict': predict}
    
    # Try to load SAINT model
    saint_dir = os.path.join(models_dir, 'Saint')
    saint_model_path = os.path.join(saint_dir, 'model_state_dict.pt')
    saint_preprocessor_path = os.path.join(saint_dir, 'tab_preprocessor.joblib')
    saint_config_path = os.path.join(saint_dir, 'config.joblib')
    
    saint_predictor = None
    if os.path.exists(saint_model_path) and os.path.exists(saint_preprocessor_path):
        try:
            # Load preprocessor
            saint_preprocessor = joblib.load(saint_preprocessor_path)
            saint_config = joblib.load(saint_config_path) if os.path.exists(saint_config_path) else {}
            features = saint_config.get('features', {})
            continuous_cols = features.get('all_continuous_cols', getattr(saint_preprocessor, 'continuous_cols', []))

            # Must match Trainings/train_saint.py exactly
            saint = SAINT(
                column_idx=saint_preprocessor.column_idx,
                cat_embed_input=saint_preprocessor.cat_embed_input,
                continuous_cols=continuous_cols,
                input_dim=64,
                n_heads=8,
                n_blocks=3,
                attn_dropout=0.05,
                ff_dropout=0.05,
            )
            saint_model = WideDeep(deeptabular=saint, pred_dim=5)

            state_dict = torch.load(saint_model_path, map_location=torch.device('cpu'))
            saint_model.load_state_dict(state_dict, strict=True)
            saint_model.eval()
            
            saint_predictor = make_torch_tabular_predictor(saint_model, saint_preprocessor)
            print(f"✓ SAINT loaded from {saint_model_path}")
        except Exception as e:
            print(f"✗ Error loading SAINT: {e}")
    else:
        print(f"✗ SAINT model not found")
        print(f"  Expected files: {saint_model_path} and {saint_preprocessor_path}")
        print(f"  Run: python3 Trainings/train_saint.py")
    
    # Collect available models
    classifiers = []
    if xgb_model is not None:
        classifiers.append(('XGBoost', xgb_model))
    if rf_model is not None:
        classifiers.append(('Random Forest', rf_model))
    if saint_predictor is not None:
        classifiers.append(('SAINT', saint_predictor))
    
    if len(classifiers) == 0:
        print("\n✗ No models available. Please train models first.")
        sys.exit(1)
    
    print(f"\nUsing {len(classifiers)} models for ensemble")
    
    # Evaluate individual models
    print("\n" + "="*70)
    print("Individual Model Performance on Test Set")
    print("="*70)
    for name, clf in classifiers:
        if hasattr(clf, 'predict'):
            y_pred = clf.predict(X_test)
        else:
            y_pred = clf['predict'](X_test)
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

