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

