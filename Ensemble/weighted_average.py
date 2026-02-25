import numpy as np
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    log_loss,
)
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def _predict_proba_from_model(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return model["predict_proba"](X)


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
            preds.append(_predict_proba_from_model(clf, X))
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
        print("\nLearned Weights:")
        for model_name, weight in self.get_weights().items():
            print(f"  {model_name:20s}: {weight:.4f}")
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

        if self.n_classifiers > 4:
            print("✗ Grid search is not feasible for more than 4 classifiers. Falling back to uniform weights.")
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

        elif self.n_classifiers == 4:
            for w1 in grid:
                for w2 in grid:
                    for w3 in grid:
                        w4 = 1 - w1 - w2 - w3
                        if w4 < 0:
                            continue
                        w = np.array([w1, w2, w3, w4])
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


class StackingMetaEnsemble:
    """
    Class-aware stacking ensemble using class-probability features.

    If all base models are sklearn-like estimators (fit + predict_proba),
    this class trains with out-of-fold meta-features.
    If any base model is pre-fitted/predict-only (e.g. torch wrapper dict),
    it falls back to direct meta-feature fitting on the provided data.
    """

    def __init__(
        self,
        base_models,
        n_classes,
        n_folds=5,
        random_state=42,
        max_iter=2000,
    ):
        self.base_models = base_models
        self.n_classes = n_classes
        self.n_folds = n_folds
        self.random_state = random_state

        self.meta_model = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=max_iter,
            n_jobs=-1,
        )

        self.fitted_base_models_ = None

    def _stack_probas(self, probas_list):
        return np.hstack(probas_list)

    def _is_trainable_model(self, model):
        return hasattr(model, "fit") and hasattr(model, "predict_proba")

    def _get_base_probas(self, models, X):
        probas = []
        for _, model in models:
            p = _predict_proba_from_model(model, X)
            if p.shape[1] != self.n_classes:
                raise ValueError(
                    f"Expected {self.n_classes} classes but got {p.shape[1]} from base model."
                )
            probas.append(p)
        return probas

    def fit(self, X, y):
        if len(self.base_models) == 0:
            raise ValueError("base_models cannot be empty")

        all_trainable = all(self._is_trainable_model(model) for _, model in self.base_models)

        if all_trainable:
            n_samples = len(X)
            n_models = len(self.base_models)
            oof_probas = np.zeros((n_samples, n_models * self.n_classes))

            skf = StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_state,
            )

            print("Generating out-of-fold predictions for stacking...")
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
                print(f"  Fold {fold}/{self.n_folds}")
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr = y.iloc[train_idx]

                fold_probas = []
                for _, model in self.base_models:
                    model_clone = clone(model)
                    model_clone.fit(X_tr, y_tr)
                    p = _predict_proba_from_model(model_clone, X_val)
                    if p.shape[1] != self.n_classes:
                        raise ValueError(
                            f"Expected {self.n_classes} classes but got {p.shape[1]} from base model."
                        )
                    fold_probas.append(p)

                oof_probas[val_idx, :] = self._stack_probas(fold_probas)

            print("Fitting meta-learner...")
            self.meta_model.fit(oof_probas, y)

            self.fitted_base_models_ = []
            for name, model in self.base_models:
                fitted = clone(model)
                fitted.fit(X, y)
                self.fitted_base_models_.append((name, fitted))
        else:
            print(
                "Using direct stacked probabilities for meta-learner "
                "(detected non-trainable pre-fitted base models)."
            )
            stacked = self._stack_probas(self._get_base_probas(self.base_models, X))
            self.meta_model.fit(stacked, y)
            self.fitted_base_models_ = list(self.base_models)

        return self

    def predict_proba(self, X):
        if self.fitted_base_models_ is None:
            raise ValueError("Call fit() first.")

        stacked = self._stack_probas(self._get_base_probas(self.fitted_base_models_, X))
        return self.meta_model.predict_proba(stacked)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def get_model_weights(self):
        if self.fitted_base_models_ is None:
            raise ValueError("Call fit() first.")

        coef = self.meta_model.coef_
        model_scores = []
        for idx, (name, _) in enumerate(self.fitted_base_models_):
            start = idx * self.n_classes
            end = start + self.n_classes
            block = coef[:, start:end]
            score = float(np.mean(np.abs(block)))
            model_scores.append((name, score))

        total = sum(score for _, score in model_scores) + 1e-12
        return {name: score / total for name, score in model_scores}

    def evaluate(self, X, y, name="Set"):
        y_pred = self.predict(X)

        acc = accuracy_score(y, y_pred)
        f1_macro = f1_score(y, y_pred, average="macro")
        f1_weighted = f1_score(y, y_pred, average="weighted")

        print(f"\n{name} Performance")
        print("-" * 50)
        print(f"Accuracy     : {acc:.4f}")
        print(f"F1 Macro     : {f1_macro:.4f}")
        print(f"F1 Weighted  : {f1_weighted:.4f}")
        print("\nModel Weights (normalized coefficient strength)")
        print("-" * 50)
        for model_name, weight in self.get_model_weights().items():
            print(f"{model_name:20s}: {weight:.4f}")

        return {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        }

