import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import DataLoader, TensorDataset

from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.models import SAINT

torch.manual_seed(42)
np.random.seed(42)

# --------------------------------------------------
# Ordinal utilities
# --------------------------------------------------

def make_ordinal_targets(y, n_classes=5):
    y = y.values if isinstance(y, pd.Series) else y
    out = np.zeros((len(y), n_classes - 1), dtype=np.float32)
    for k in range(n_classes - 1):
        out[:, k] = (y > k).astype(np.float32)
    return out


def ordinal_predict(logits, thresholds):
    probs = torch.sigmoid(logits)
    t = torch.tensor(thresholds, device=logits.device)
    return (probs > t).sum(dim=1)


def tune_thresholds(logits, y_ord):
    probs = torch.sigmoid(logits).cpu().numpy()
    y_ord = y_ord.cpu().numpy()

    thresholds = []
    for k in range(probs.shape[1]):
        best_f1, best_t = -1, 0.5
        for t in np.linspace(0.2, 0.8, 61):
            pred = (probs[:, k] > t).astype(int)
            f1 = f1_score(y_ord[:, k], pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds.append(best_t)
    return thresholds


# --------------------------------------------------
# Monotonic ordinal head
# --------------------------------------------------

class SAINTOrdinal(nn.Module):
    def __init__(self, saint_model, n_classes=5):
        super().__init__()
        self.saint = saint_model
        self.fc = nn.Linear(saint_model.output_dim, n_classes - 1)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        h = self.saint(x)
        logits = self.fc(h)
        return torch.cumsum(logits, dim=1)

# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    data_path = os.path.join(project_root, "Data/processed_new_data2.csv")
    df = pd.read_csv(data_path)

    y = df["damage_grade"] 
    X = df.drop(columns=["damage_grade"])

    # --------------------------------------------------
    # Column definition
    # --------------------------------------------------

    cat_embed_cols = [
        "foundation_type",
        "roof_type",
        "ground_floor_type",
    ]

    # binary flags are CONTINUOUS
    continuous_cols = [c for c in X.columns if c not in cat_embed_cols]

    # --------------------------------------------------
    # Split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.3, stratify=y_train, random_state=42
    )

    # --------------------------------------------------
    # Preprocessing
    # --------------------------------------------------

    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=cat_embed_cols,
        continuous_cols=continuous_cols,
        cols_to_scale=continuous_cols,
        with_cls_token=False,
    )

    X_train_tab = tab_preprocessor.fit_transform(X_train)
    X_val_tab = tab_preprocessor.transform(X_val)
    X_test_tab = tab_preprocessor.transform(X_test)

    # Ordinal targets
    y_train_ord = make_ordinal_targets(y_train)
    y_val_ord   = make_ordinal_targets(y_val)


    # --------------------------------------------------
    # Class imbalance handling
    # --------------------------------------------------

    ppos = y_train_ord.sum(axis=0)
    neg = len(y_train_ord) - ppos
    ratio = neg / ppos

    pos_weight = torch.tensor(
        np.clip(ratio, 1.0, 3.0),
        dtype=torch.float32
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --------------------------------------------------
    # Dataloaders
    # --------------------------------------------------

    train_ds = TensorDataset(
        torch.tensor(X_train_tab, dtype=torch.float32),
        torch.tensor(y_train_ord, dtype=torch.float32),
    )

    val_ds = TensorDataset(
        torch.tensor(X_val_tab, dtype=torch.float32),
        torch.tensor(y_val_ord, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)  # Reduced from 1024
    val_loader = DataLoader(val_ds, batch_size=512)  # Reduced from 2048
    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    saint = SAINT(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=continuous_cols,
        input_dim=32,        
        n_heads=8,           
        n_blocks=4,          
        attn_dropout=0.1,
        ff_dropout=0.1,
    )

    model = SAINTOrdinal(saint).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    
    # Add mixed precision for memory efficiency
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    print(f"Using mixed precision: {scaler is not None}")

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    EPOCHS = 30

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            
            # Use mixed precision
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        model.eval()
        all_logits, all_y = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                
                # Use mixed precision for inference
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        logits = model(xb)
                else:
                    logits = model(xb)
                    
                all_logits.append(logits.cpu())
                all_y.append(yb.cpu())
        
        # Clear CUDA cache to prevent memory accumulation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        all_logits = torch.cat(all_logits)
        all_y = torch.cat(all_y)

        preds = ordinal_predict(all_logits, [0.5] * 4)
        true = all_y.sum(dim=1).long()

        macro_f1 = f1_score(true.cpu(), preds.cpu(), average="macro", zero_division=0)
        
        # Compute validation loss (move to device for loss computation)
        val_loss = criterion(all_logits.to(device), all_y.to(device)).item()
        
        # Debug: logit statistics
        logit_stats = f"Logits - mean: {all_logits.mean():.3f}, std: {all_logits.std():.3f}, min: {all_logits.min():.3f}, max: {all_logits.max():.3f}"
        
        # Debug: prediction distribution vs true distribution
        pred_dist = [int((preds == i).sum()) for i in range(5)]
        true_dist = [int((true == i).sum()) for i in range(5)]

        print(
            f"Epoch {epoch+1:02d} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Macro-F1: {macro_f1:.4f}"
        )
        print(f"  {logit_stats}")
        print(f"  Pred dist: {pred_dist} | True dist: {true_dist}")

    # --------------------------------------------------
    # Final evaluation
    # --------------------------------------------------

    final_thresholds = tune_thresholds(all_logits, all_y)
    print("Optimal thresholds:", final_thresholds)

    model.eval()
    with torch.no_grad():
        logits_test = model(torch.tensor(X_test_tab, dtype=torch.float32).to(device))

    y_pred = ordinal_predict(logits_test, final_thresholds).cpu().numpy()

    print("\nTest Classification Report:")
    print(classification_report(y_test, y_pred))
