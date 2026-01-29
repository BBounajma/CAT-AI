# ============================================================
# Correct Gradient Boosting for Multi-class Classification
# Models: XGBClassifier (base) + SaintWrapper (gradient booster)
# NO SaintBooster, NO weighting, NO averaging
# ============================================================

import os
import pickle
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from Models.Saint.wrapper import SaintWrapper


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def one_hot(y, num_classes):
    out = np.zeros((len(y), num_classes), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


# ------------------------------------------------------------
# CE Gradient Boosting Trainer
# ------------------------------------------------------------

class CEGradientBoosting:
    def __init__(self, num_classes, eta=0.1):
        self.num_classes = num_classes
        self.eta = eta
        self.base_model = None
        self.saint = None

    def fit(self, X_train, y_train, X_val, y_val, saint_kwargs, saint_epochs=20):
        # ------------------
        # Base model (XGB)
        # ------------------
        print("Training XGBClassifier (base model)...")
        self.base_model = XGBClassifier(
            objective="multi:softprob",
            num_class=self.num_classes,
            eval_metric="mlogloss",
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            random_state=42,
        )

        self.base_model.fit(X_train, y_train)

        F_tr = self.base_model.predict(X_train, output_margin=True)
        F_va = self.base_model.predict(X_val, output_margin=True)

        base_acc = np.mean(F_va.argmax(1) == y_val)
        print(f"Base validation accuracy: {base_acc:.4f}")

        # ------------------
        # Compute CE gradients
        # ------------------
        y_tr_oh = one_hot(y_train, self.num_classes)
        grad_tr = softmax(F_tr) - y_tr_oh
        residual_tr = -self.eta * grad_tr

        # ------------------
        # Train SAINT on gradients with epoch-wise validation
        # ------------------
        print("Training SAINT on CE gradients...")
        self.saint = SaintWrapper(**saint_kwargs)
        
        # Store base logits for validation during training
        F_va_base = F_va.copy()
        
        # Train SAINT with manual epoch-wise reporting every 4 epochs
        for epoch in range(saint_epochs):
            # Train one epoch
            self.saint.fit(
                X_train, 
                residual_tr, 
                epochs=1,
            )
            
            # Report accuracy every 4 epochs
            if (epoch + 1) % 4 == 0:
                pred_residual = self.saint.predict(X_val)
                ensemble_logits = F_va_base + pred_residual
                ensemble_acc = np.mean(ensemble_logits.argmax(1) == y_val)
                print(f"Epoch {epoch+1}/{saint_epochs} | ensemble validation accuracy: {ensemble_acc:.4f}")

        # ------------------
        # Update logits
        # ------------------
        F_tr = F_tr + self.saint.predict(X_train)
        F_va = F_va_base + self.saint.predict(X_val)

        boost_acc = np.mean(F_va.argmax(1) == y_val)
        print(f"\nAfter SAINT boost | validation accuracy: {boost_acc:.4f}")

    def predict(self, X):
        F = self.base_model.predict(X, output_margin=True)
        F = F + self.saint.predict(X)
        return F.argmax(1)

    def predict_proba(self, X):
        F = self.base_model.predict(X, output_margin=True)
        F = F + self.saint.predict(X)
        return softmax(F)


# ------------------------------------------------------------
# Main (mirrors train_boosting_ensemble_data2.py)
# ------------------------------------------------------------

if __name__ == "__main__":
    print("Loading preprocessed data2 from pickle file...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_path = os.path.join(script_dir, "Data", "processed_data2.pkl")

    with open(pickle_path, "rb") as f:
        X_train, X_val, X_test, y_train, y_val, y_test = pickle.load(f)

    X_train = X_train.values.astype(np.float32) if hasattr(X_train, "values") else X_train.astype(np.float32)
    X_val   = X_val.values.astype(np.float32)   if hasattr(X_val, "values")   else X_val.astype(np.float32)
    y_train = y_train.values.astype(np.int64)   if hasattr(y_train, "values") else y_train.astype(np.int64)
    y_val   = y_val.values.astype(np.int64)     if hasattr(y_val, "values")   else y_val.astype(np.int64)

    num_classes = len(np.unique(y_train))
    print(f"Number of classes: {num_classes}")

    # ---- SAINT configuration (must match wrapper.py) ----
    # Data is already one-hot encoded, so no categorical features remain
    # X_train shape: (batch_size, num_features) where num_features = 5 continuous + one-hot categorical
    saint_kwargs = dict(
        num_categorical=0,
        num_continuous=X_train.shape[1],
        num_classes=num_classes,
        cat_dims=(),
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        task="regression",
    )

    trainer = CEGradientBoosting(num_classes=num_classes, eta=0.1)
    trainer.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        saint_kwargs,
        saint_epochs=20,
    )

    train_preds = trainer.predict(X_train)
    val_preds = trainer.predict(X_val)

    print("\n============================================================")
    print("Boosting Ensemble Evaluation")
    print("============================================================")
    print(f"Training Accuracy:   {np.mean(train_preds == y_train):.4f}")
    print(f"Validation Accuracy: {np.mean(val_preds == y_val):.4f}")
