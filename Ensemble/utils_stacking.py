import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

from pytorch_widedeep.models import SAINT, WideDeep

warnings.filterwarnings("ignore", category=FutureWarning)


def model_predict_proba(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)
    return clf["predict_proba"](X)


def model_predict(clf, X):
    if hasattr(clf, "predict"):
        return clf.predict(X)
    return clf["predict"](X)


def make_torch_tabular_predictor(model, preprocessor, device="cpu", batch_size=128):
    model = model.to(device)
    model.eval()

    def _extract_logits(output):
        if isinstance(output, (tuple, list)):
            return output[0]
        if isinstance(output, dict):
            for key in ("logits", "preds", "y_pred"):
                if key in output:
                    return output[key]
        return output

    def predict_logits(X):
        X_processed = preprocessor.transform(X)
        X_tensor = torch.tensor(X_processed, dtype=torch.float32)

        logit_chunks = []
        with torch.no_grad():
            for start in range(0, X_tensor.shape[0], batch_size):
                end = start + batch_size
                batch = X_tensor[start:end].to(device)
                try:
                    model_output = model({"deeptabular": batch})
                except KeyError:
                    model_output = model({"X_tab": batch})
                logits = _extract_logits(model_output)
                logit_chunks.append(logits.cpu().numpy())

        return np.vstack(logit_chunks)

    def predict_proba(X):
        logits = predict_logits(X)
        # numerical stable softmax
        e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def predict(X):
        return np.argmax(predict_proba(X), axis=1)

    return {"predict_logits": predict_logits, "predict_proba": predict_proba, "predict": predict}


def load_saint_predictor(models_dir):
    saint_dir = os.path.join(models_dir, "Saint")
    saint_model_path = os.path.join(saint_dir, "model_state_dict.pt")
    saint_preprocessor_path = os.path.join(saint_dir, "tab_preprocessor.joblib")
    saint_config_path = os.path.join(saint_dir, "config.joblib")

    if not (os.path.exists(saint_model_path) and os.path.exists(saint_preprocessor_path)):
        print("✗ SAINT model not found")
        print(f"  Expected files: {saint_model_path} and {saint_preprocessor_path}")
        print("  Run: python3 Trainings/train_saint.py")
        return None

    try:
        saint_preprocessor = joblib.load(saint_preprocessor_path)
        saint_config = joblib.load(saint_config_path) if os.path.exists(saint_config_path) else {}
        features = saint_config.get("features", {})
        continuous_cols = features.get(
            "all_continuous_cols", getattr(saint_preprocessor, "continuous_cols", [])
        )

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
        state_dict = torch.load(saint_model_path, map_location=torch.device("cpu"))
        saint_model.load_state_dict(state_dict, strict=True)

        print(f"✓ SAINT loaded from {saint_model_path}")
        predictor = make_torch_tabular_predictor(saint_model, saint_preprocessor)

        # Attach precomputed OOF predictions if available
        saint_oof_path = os.path.join(saint_dir, "saint_oof_preds.npy")
        if os.path.exists(saint_oof_path):
            try:
                predictor["oof"] = np.load(saint_oof_path)
            except Exception:
                pass

        return predictor
    except Exception as exc:
        print(f"✗ Error loading SAINT: {exc}")
        return None


def build_meta_features(classifiers, X):
    proba_blocks = [model_predict_proba(clf, X) for _, clf in classifiers]
    return _build_enhanced_meta_features(proba_blocks)


def _build_enhanced_meta_features(proba_blocks, eps=1e-12):
    """
    Build richer meta-features from base-model class probabilities.

    Feature groups:
    1) Raw probability blocks for each model (backward-compatible prefix)
    2) Per-model confidence stats: max prob, margin(top1-top2), entropy
    3) Cross-model average probabilities and uncertainty stats
    """
    if len(proba_blocks) == 0:
        raise ValueError("proba_blocks cannot be empty")

    base = np.concatenate(proba_blocks, axis=1)
    aux_blocks = []

    for probs in proba_blocks:
        sorted_probs = np.sort(probs, axis=1)
        top1 = sorted_probs[:, -1]
        top2 = sorted_probs[:, -2] if probs.shape[1] > 1 else np.zeros_like(top1)
        margin = top1 - top2
        entropy = -np.sum(np.clip(probs, eps, 1.0) * np.log(np.clip(probs, eps, 1.0)), axis=1)
        aux_blocks.append(np.column_stack([top1, margin, entropy]))

    mean_probs = np.mean(np.stack(proba_blocks, axis=0), axis=0)
    sorted_mean = np.sort(mean_probs, axis=1)
    mean_top1 = sorted_mean[:, -1]
    mean_top2 = sorted_mean[:, -2] if mean_probs.shape[1] > 1 else np.zeros_like(mean_top1)
    mean_margin = mean_top1 - mean_top2
    mean_entropy = -np.sum(
        np.clip(mean_probs, eps, 1.0) * np.log(np.clip(mean_probs, eps, 1.0)), axis=1
    )

    aux = np.concatenate(aux_blocks + [mean_probs, np.column_stack([mean_top1, mean_margin, mean_entropy])], axis=1)
    return np.concatenate([base, aux], axis=1)


def build_meta_features_oof(classifiers, X, y, n_splits=5, random_state=42):
    """
    Build OOF meta-features when base estimators are trainable.

    For sklearn-like models (fit + predict_proba), generates proper OOF probabilities.
    For pre-fitted predictors (dict wrappers, e.g. torch models), falls back to direct
    predict_proba on X and keeps the original predictor for test-time meta-features.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    meta_blocks = []
    fitted_classifiers = []

    for name, clf in classifiers:
        if hasattr(clf, "fit") and hasattr(clf, "predict_proba"):
            # Avoid calling predict_proba on possibly-unfitted wrappers (e.g. pre-saved
            # CalibratedClassifierCV). Infer number of classes from the target `y`.
            n_classes = len(np.unique(y))
            oof = np.zeros((len(X), n_classes), dtype=np.float64)

            for train_idx, val_idx in skf.split(X, y):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr = y.iloc[train_idx]

                model_fold = clone(clf)
                model_fold.fit(X_tr, y_tr)
                oof[val_idx] = model_fold.predict_proba(X_val)

            model_full = clone(clf)
            model_full.fit(X, y)
            fitted_classifiers.append((name, model_full))
            meta_blocks.append(oof)
        else:
            # If the predictor provides precomputed OOF predictions, use them.
            if isinstance(clf, dict) and "oof" in clf and getattr(clf["oof"], "shape", (None,))[0] == len(X):
                meta_blocks.append(clf["oof"])  # use precomputed OOF block
                fitted_classifiers.append((name, clf))
            else:
                # First try several locations on the predictor object for precomputed OOF:
                #  - mapping key: clf["oof"]
                #  - attribute: clf.oof
                oof_arr = None
                try:
                    if isinstance(clf, dict) and "oof" in clf:
                        oof_arr = np.asarray(clf["oof"])
                    elif hasattr(clf, "oof"):
                        oof_arr = np.asarray(getattr(clf, "oof"))
                except Exception:
                    oof_arr = None

                # Next, try loading from repo Models folder as a fallback
                if oof_arr is None:
                    try:
                        repo_root = os.path.dirname(os.path.dirname(__file__))
                        candidate = os.path.join(repo_root, "Models", name, f"{name.lower()}_oof_preds.npy")
                        if os.path.exists(candidate):
                            oof_arr = np.load(candidate)
                            print(f"[OOF file] {name}: loaded {candidate} with shape {getattr(oof_arr,'shape',None)}")
                    except Exception:
                        oof_arr = None

                if oof_arr is not None and getattr(oof_arr, "shape", (None,))[0] == len(X):
                    print(f"[OOF found] {name}: using OOF with shape {oof_arr.shape}")
                    meta_blocks.append(oof_arr)
                    fitted_classifiers.append((name, clf))
                else:
                    if oof_arr is not None:
                        print(f"[OOF fallback] {name}: provided OOF shape {getattr(oof_arr,'shape',None)} does not match {len(X)}")
                    else:
                        # extra diagnostics for why we didn't find OOF
                        if isinstance(clf, dict):
                            print(f"[OOF fallback] {name}: predictor keys = {list(clf.keys())}")
                        if hasattr(clf, "oof"):
                            try:
                                print(f"[OOF fallback] {name}: predictor.oof shape = {np.asarray(getattr(clf,'oof')).shape}")
                            except Exception as exc:
                                print(f"[OOF fallback] {name}: error reading predictor.oof: {exc}")
                    print(f"[OOF fallback] {name}: using direct predictions (model is not fit-capable).")
                    meta_blocks.append(model_predict_proba(clf, X))
                    fitted_classifiers.append((name, clf))

    X_meta = np.concatenate(meta_blocks, axis=1)
    X_meta = _build_enhanced_meta_features(meta_blocks)
    return X_meta, fitted_classifiers


def summarize_model_usage(mlp_meta, classifiers, n_classes):
    """Approximate contribution per base model from first-layer absolute weights."""
    first_layer = mlp_meta.coefs_[0]  # shape: (n_meta_features, hidden_dim)
    feature_strength = np.mean(np.abs(first_layer), axis=1)

    # Determine how many sklearn-like models were used when building the enhanced
    # meta-features. Training uses enhanced features for the sklearn block (which
    # contains the raw probs for each sklearn model at the front of the enhanced
    # block), and then appends raw torch probability blocks. We need to compute
    # the correct slice indices for each model accordingly.
    sklearn_count = sum(1 for _, clf in classifiers if hasattr(clf, "fit") and hasattr(clf, "predict_proba"))

    if sklearn_count > 0:
        # enhanced block layout (for K sklearn models, C classes):
        # base probs: K*C
        # per-model aux: 3*K
        # mean_probs: C
        # mean aux: 3
        enhanced_block_size = sklearn_count * n_classes + 3 * sklearn_count + n_classes + 3
    else:
        enhanced_block_size = 0

    model_scores = []
    for model_idx, (name, _) in enumerate(classifiers):
        if model_idx < sklearn_count:
            # sklearn models: their raw-prob block is inside the enhanced block
            start = model_idx * n_classes
        else:
            # torch models: located after the enhanced block
            torch_idx = model_idx - sklearn_count
            start = enhanced_block_size + torch_idx * n_classes

        end = start + n_classes
        score = feature_strength[start:end].mean()
        model_scores.append((name, score))

    total = sum(score for _, score in model_scores) + 1e-12
    return [(name, score, score / total) for name, score in model_scores]


def ablation_impact(mlp_meta, classifiers, X, y):
    """Measure performance drop when each model's probability block is neutralized."""
    # Split classifiers into sklearn-like (were enhanced during training) and
    # torch-like (raw prob blocks appended). This mirrors how the training
    # meta-features were constructed: enhanced(sklearn) + raw(torch).
    sklearn_indices = [i for i, (_, clf) in enumerate(classifiers) if hasattr(clf, "fit") and hasattr(clf, "predict_proba")]
    torch_indices = [i for i in range(len(classifiers)) if i not in sklearn_indices]

    # collect probability blocks separately
    proba_sklearn = [model_predict_proba(classifiers[i][1], X) for i in sklearn_indices]
    proba_torch = [model_predict_proba(classifiers[i][1], X) for i in torch_indices]

    # Debug: print shapes
    try:
        print("DEBUG ablation: proba_sklearn shapes:", [p.shape for p in proba_sklearn])
        print("DEBUG ablation: proba_torch shapes:", [p.shape for p in proba_torch])
    except Exception:
        pass

    # determine number of classes
    n_classes = proba_sklearn[0].shape[1] if len(proba_sklearn) > 0 else proba_torch[0].shape[1]

    # Build full meta features in the same layout used for training
    if len(proba_sklearn) > 0:
        X_meta_sklearn = _build_enhanced_meta_features(proba_sklearn)
    else:
        X_meta_sklearn = np.empty((X.shape[0], 0))

    X_meta_full = np.hstack([X_meta_sklearn] + proba_torch) if len(proba_torch) > 0 else X_meta_sklearn

    print(f"DEBUG ablation: X_meta_full.shape = {X_meta_full.shape}")
    y_pred_full = mlp_meta.predict(X_meta_full)
    base_acc = accuracy_score(y, y_pred_full)
    base_f1 = f1_score(y, y_pred_full, average="macro")

    uniform = np.full((X.shape[0], n_classes), 1.0 / n_classes)
    results = []

    # For each model, create ablated meta features by replacing the appropriate
    # block (inside the sklearn proba list or the torch proba list) with uniform.
    for idx, (name, _) in enumerate(classifiers):
        if idx in sklearn_indices:
            # ablate within the sklearn proba blocks
            sk_idx = sklearn_indices.index(idx)
            ablated_sklearn = [b.copy() for b in proba_sklearn]
            ablated_sklearn[sk_idx] = uniform
            if len(ablated_sklearn) > 0:
                X_meta_sklearn_ab = _build_enhanced_meta_features(ablated_sklearn)
            else:
                X_meta_sklearn_ab = np.empty((X.shape[0], 0))
            X_meta_ablate = np.hstack([X_meta_sklearn_ab] + proba_torch) if len(proba_torch) > 0 else X_meta_sklearn_ab
        else:
            # ablate a torch block
            torch_idx = torch_indices.index(idx)
            ablated_torch = [b.copy() for b in proba_torch]
            ablated_torch[torch_idx] = uniform
            X_meta_ablate = np.hstack([X_meta_sklearn] + ablated_torch) if len(ablated_torch) > 0 else X_meta_sklearn

        y_pred_ablate = mlp_meta.predict(X_meta_ablate)
        acc = accuracy_score(y, y_pred_ablate)
        f1m = f1_score(y, y_pred_ablate, average="macro")
        results.append((name, base_acc - acc, base_f1 - f1m))

    return base_acc, base_f1, results


def calibrate_sklearn_models(sklearn_models, X_val, y_val, method="isotonic"):
    """Calibrate fitted sklearn classifiers using `cv='prefit'` on validation data.

    Uses isotonic calibration by default. Returns a list of (name, calibrated_estimator)
    maintaining order. If calibration fails (e.g. insufficient validation samples),
    falls back to leaving the original estimator unchanged.
    """
    from sklearn.calibration import CalibratedClassifierCV

    calibrated = []
    for name, clf in sklearn_models:
        if hasattr(clf, "predict_proba"):
            try:
                calib = CalibratedClassifierCV(base_estimator=clf, cv="prefit", method=method)
                calib.fit(X_val, y_val)
                calibrated.append((name, calib))
            except Exception:
                # fallback: keep original if calibration fails
                calibrated.append((name, clf))
        else:
            calibrated.append((name, clf))

    return calibrated


def temperature_scale_torch_models(torch_models, X_val, y_val, device="cpu", max_iter=200):
    """Apply temperature scaling to torch-based predictors.

    torch_models: list of (name, predictor) where predictor exposes `predict_logits`.
    Returns list of (name, calibrated_predictor).
    """
    calibrated = []
    for name, pred in torch_models:
        if not hasattr(pred, "predict_logits"):
            calibrated.append((name, pred))
            continue

        logits = pred["predict_logits"](X_val)
        labels = np.array(y_val)

        import torch

        logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
        labels_t = torch.tensor(labels, dtype=torch.long, device=device)

        T = torch.nn.Parameter(torch.ones(1, device=device))
        optimizer = torch.optim.LBFGS([T], max_iter=50, line_search_fn="strong_wolfe")

        loss_fn = torch.nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            scaled = logits_t / T.clamp(min=1e-6)
            loss = loss_fn(scaled, labels_t)
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
        except Exception:
            # fall back to simple SGD if LBFGS fails
            T.data = torch.tensor([1.0], device=device)

        T_val = float(T.detach().cpu().numpy())

        # build calibrated predictor wrapper
        def make_scaled(pred, T_val):
            def predict_logits(X):
                return pred["predict_logits"](X)

            def predict_proba(X):
                logits = predict_logits(X)
                scaled = logits / T_val
                e = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
                return e / np.sum(e, axis=1, keepdims=True)

            def predict(X):
                return np.argmax(predict_proba(X), axis=1)

            return {"predict_logits": predict_logits, "predict_proba": predict_proba, "predict": predict}

        calibrated.append((name, make_scaled(pred, T_val)))

    return calibrated
