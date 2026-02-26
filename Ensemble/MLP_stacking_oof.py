import os
import joblib
import numpy as np
import torch

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# Generic predict helpers (sklearn + torch wrappers)
# ============================================================
def model_predict_proba(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)
    return clf["predict_proba"](X)


def model_predict(clf, X):
    if hasattr(clf, "predict"):
        return clf.predict(X)
    return clf["predict"](X)


# ============================================================
# Torch → sklearn-style predictor wrapper
# ============================================================
def make_torch_tabular_predictor(model, preprocessor, device="cpu", batch_size=256):
    model = model.to(device)
    model.eval()

    def _extract_logits(out):
        if isinstance(out, (tuple, list)):
            return out[0]
        if isinstance(out, dict):
            for k in ("logits", "preds", "y_pred"):
                if k in out:
                    return out[k]
        return out

    def predict_proba(X):
        Xp = preprocessor.transform(X)
        Xt = torch.tensor(Xp, dtype=torch.float32)

        probs = []
        with torch.no_grad():
            for i in range(0, len(Xt), batch_size):
                batch = Xt[i : i + batch_size].to(device)
                out = model({"deeptabular": batch})
                logits = _extract_logits(out)
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(probs)

    def predict(X):
        return np.argmax(predict_proba(X), axis=1)

    return {"predict_proba": predict_proba, "predict": predict}


# ============================================================
# SAINT loader (NO TRAINING HERE)
# ============================================================
def load_saint_predictor(models_dir):
    saint_dir = os.path.join(models_dir, "Saint")
    model_path = os.path.join(saint_dir, "model_state_dict.pt")
    prep_path = os.path.join(saint_dir, "tab_preprocessor.joblib")

    if not (os.path.exists(model_path) and os.path.exists(prep_path)):
        return None

    from pytorch_widedeep.models import SAINT, WideDeep

    preprocessor = joblib.load(prep_path)

    saint = SAINT(
        column_idx=preprocessor.column_idx,
        cat_embed_input=preprocessor.cat_embed_input,
        continuous_cols=preprocessor.continuous_cols,
        input_dim=64,
        n_heads=8,
        n_blocks=3,
        attn_dropout=0.05,
        ff_dropout=0.05,
    )

    model = WideDeep(deeptabular=saint, pred_dim=5)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    return make_torch_tabular_predictor(model, preprocessor)


# ============================================================
# Load precomputed OOF predictions (STRICT)
# ============================================================
def load_oof_predictions(models_dir, model_name):
    path = os.path.join(
        models_dir,
        model_name,
        f"{model_name.lower()}_oof_preds.npy",
    )
    if not os.path.exists(path):
        raise RuntimeError(
            f"Missing OOF predictions for {model_name}. "
            "Stacking would be invalid."
        )
    return np.load(path)


# ============================================================
# Meta-feature construction
# ============================================================
def build_meta_features(classifiers, X):
    proba_blocks = [model_predict_proba(clf, X) for _, clf in classifiers]
    return _build_enhanced_meta_features(proba_blocks)


def build_meta_features_oof(
    sklearn_models,
    X,
    y,
    torch_model_names=None,
    models_dir=None,
    n_splits=5,
    random_state=42,
):
    """
    OOF for sklearn models + disk-loaded OOF for torch models.
    NO FALLBACKS.
    """
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    meta_blocks = []
    fitted_models = []

    # -------------------------
    # sklearn CV OOF
    # -------------------------
    for name, clf in sklearn_models:
        n_classes = clf.predict_proba(X.iloc[:1]).shape[1]
        oof = np.zeros((len(X), n_classes))

        for tr, va in skf.split(X, y):
            model = clone(clf)
            model.fit(X.iloc[tr], y.iloc[tr])
            oof[va] = model.predict_proba(X.iloc[va])

        meta_blocks.append(oof)

        model_full = clone(clf)
        model_full.fit(X, y)
        fitted_models.append((name, model_full))

    # -------------------------
    # torch OOF from disk
    # -------------------------
    if torch_model_names:
        assert models_dir is not None
        for name in torch_model_names:
            oof = load_oof_predictions(models_dir, name)
            if oof.shape[0] != len(X):
                raise RuntimeError(f"OOF size mismatch for {name}")
            meta_blocks.append(oof)

    X_meta = _build_enhanced_meta_features(meta_blocks)
    return X_meta, fitted_models


# ============================================================
# Feature engineering on probabilities
# ============================================================
def _build_enhanced_meta_features(proba_blocks, eps=1e-12):
    base = np.concatenate(proba_blocks, axis=1)
    aux = []

    for p in proba_blocks:
        s = np.sort(p, axis=1)
        top1, top2 = s[:, -1], s[:, -2]
        entropy = -np.sum(np.clip(p, eps, 1) * np.log(np.clip(p, eps, 1)), axis=1)
        aux.append(np.c_[top1, top1 - top2, entropy])

    mean_p = np.mean(proba_blocks, axis=0)
    s = np.sort(mean_p, axis=1)
    aux.append(mean_p)
    aux.append(np.c_[s[:, -1], s[:, -1] - s[:, -2]])

    return np.hstack([base] + aux)


# ============================================================
# Diagnostics
# ============================================================
def summarize_model_usage(mlp, classifiers, n_classes):
    w = np.mean(np.abs(mlp.coefs_[0]), axis=1)
    out = []
    for i, (name, _) in enumerate(classifiers):
        s = w[i * n_classes : (i + 1) * n_classes].mean()
        out.append((name, s))
    tot = sum(x[1] for x in out) + 1e-12
    return [(n, s, s / tot) for n, s in out]


def ablation_impact(mlp, classifiers, X, y):
    probs = [model_predict_proba(clf, X) for _, clf in classifiers]
    n_classes = probs[0].shape[1]

    base_X = _build_enhanced_meta_features(probs)
    base_pred = mlp.predict(base_X)
    base_acc = accuracy_score(y, base_pred)
    base_f1 = f1_score(y, base_pred, average="macro")

    uniform = np.full_like(probs[0], 1.0 / n_classes)
    results = []

    for i, (name, _) in enumerate(classifiers):
        p = probs.copy()
        p[i] = uniform
        X_ab = _build_enhanced_meta_features(p)
        pred = mlp.predict(X_ab)
        results.append(
            (
                name,
                base_acc - accuracy_score(y, pred),
                base_f1 - f1_score(y, pred, average="macro"),
            )
        )

    return base_acc, base_f1, results