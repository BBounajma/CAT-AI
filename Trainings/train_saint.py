import os
import sys
import einops

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.utils.data
from torchmetrics.classification import MulticlassF1Score, MulticlassFBetaScore
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.models import SAINT

torch.manual_seed(42)
np.random.seed(42)


if __name__ == '__main__':
    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    # Script path setup
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Models'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Data'))

    # Load data - adjust path as per your setup
    df = pd.read_csv("Data/processed_new_data2.csv")  # Linux version
    #df = pd.read_csv("Data\processed_new_data2.csv")  # Windows version


    #multi_class_cat_cols to be embedded by TabPreprocessor
    multi_class_cat_cols = [
        'foundation_type',
        'roof_type',
        'ground_floor_type'
    ]
    
    # One-hot encoded superstructure columns
    onehot_cols = [
        "has_superstructure_mud_mortar_stone",
        "has_superstructure_stone_flag",
        "has_superstructure_cement_mortar_stone",
        "has_superstructure_cement_mortar_brick",
        "has_superstructure_timber",
        "has_superstructure_bamboo",
        "has_superstructure_rc_non_engineered",
        "has_superstructure_rc_engineered",
        "has_superstructure_other"
    ]

    y = df["damage_grade"]
    
    X_processed = df.drop("damage_grade", axis=1)

    #scale numerical features to be scaled by TabPreprocessor
    num_cols = [
        "PGA_g",
        "count_floors_pre_eq",
        "age_building",
        "plinth_area_sq_ft",
        "per-height_ft_pre_eq"
    ]
    
    # Combine numerical and one-hot as continuous features
    all_continuous_cols = num_cols + onehot_cols

    # Split data into training, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    # Reset indices to 0-based sequential indexing
    X_train = X_train.reset_index(drop=True)
    X_valid = X_valid.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_valid = y_valid.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    # Convert labels to integer type and ensure 0-indexed (damage_grade is 1-5, needs to be 0-4)
    print(f"\nBefore conversion - y_train range: [{y_train.min()}, {y_train.max()}]")
    print(f"Unique values: {sorted(y_train.unique())}")
    
    # Check if labels need to be converted from 1-5 to 0-4
    if y_train.min() == 1:
        print("Converting labels from 1-5 to 0-4...")
        y_train = (y_train - 1).astype(np.int64)
        y_valid = (y_valid - 1).astype(np.int64)
        y_test = (y_test - 1).astype(np.int64)
    else:
        y_train = y_train.astype(np.int64)
        y_valid = y_valid.astype(np.int64)
        y_test = y_test.astype(np.int64)
    
    print(f"After conversion - y_train range: [{y_train.min()}, {y_train.max()}]")
    print(f"Unique values: {sorted(np.unique(y_train))}")
    
    # Debug: Check label distribution
    print(f"\nLabel distribution (y_train):")
    for i in range(5):
        count = (y_train == i).sum()
        print(f"  Class {i}: {count} ({count/len(y_train)*100:.1f}%)")



    # ------------------------------------------------------------------
    # Tabular preprocessing
    # ------------------------------------------------------------------
    tab_preprocessor = TabPreprocessor(
        cat_embed_cols=multi_class_cat_cols,
        continuous_cols=all_continuous_cols,  # Use all continuous features
        cols_to_scale=num_cols,  # Only scale numerical, not one-hot
        with_cls_token=False
    )


    X_train_tab = tab_preprocessor.fit_transform(X_train)
    X_valid_tab = tab_preprocessor.transform(X_valid)
    X_test_tab = tab_preprocessor.transform(X_test)
    
    print(f"\nFeature shapes after preprocessing:")
    print(f"  X_train_tab: {X_train_tab.shape}")
    print(f"  X_valid_tab: {X_valid_tab.shape}")
    print(f"  X_test_tab: {X_test_tab.shape}")


    # ------------------------------------------------------------------
    # SAINT model (direct, not wrapped in WideDeep to avoid device issues)
    # ------------------------------------------------------------------
    saint = SAINT(
        column_idx=tab_preprocessor.column_idx,
        cat_embed_input=tab_preprocessor.cat_embed_input,
        continuous_cols=all_continuous_cols,  # Use all continuous features
        input_dim=64,
        n_heads=8,
        n_blocks=3,
        attn_dropout=0.1,
        ff_dropout=0.1,
    )

    # Add a final classification head
    class SAINTClassifier(nn.Module):
        def __init__(self, saint_model, num_classes=5, input_sample=None):
            super().__init__()
            self.saint = saint_model
            
            # Determine SAINT output dimension by running a forward pass
            if input_sample is not None:
                with torch.no_grad():
                    sample_output = saint_model(input_sample[:1])
                    saint_output_dim = sample_output.shape[1]
            else:
                # Fallback: SAINT output is concatenation of embeddings + continuous features
                saint_output_dim = saint_model.input_dim
            
            print(f"SAINT output dimension: {saint_output_dim}")
            self.classifier = nn.Linear(saint_output_dim, num_classes)
            
        def forward(self, x):
            features = self.saint(x)
            return self.classifier(features)

    # Create sample input to determine SAINT output dimension
    sample_input = torch.tensor(X_train_tab[:1], dtype=torch.float32)
    model = SAINTClassifier(saint, num_classes=5, input_sample=sample_input)

    # ------------------------------------------------------------------
    # Setup training
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    model = model.to(device)
    
    # Debug: Test forward pass
    print(f"\nTesting model forward pass...")
    test_batch = torch.tensor(X_train_tab[:2], dtype=torch.float32).to(device)
    with torch.no_grad():
        test_output = model(test_batch)
        print(f"  Input shape: {test_batch.shape}")
        print(f"  Output shape: {test_output.shape}")
        print(f"  Output sample: {test_output[0].cpu().numpy()}")
        print(f"  Output range: [{test_output.min():.3f}, {test_output.max():.3f}]")

    counts = np.bincount(y_train)
    counts[counts == 0] = 1
    
    # Calculate class weights using inverse frequency
    class_weights = torch.tensor(
        counts.sum() / counts,
        dtype=torch.float32
    )
    # Cap weights to prevent extreme imbalance
    class_weights = torch.clamp(class_weights, min=0.5, max=5.0)
    print(f"\nClass weights (capped at 5.0): {class_weights.numpy()}")
    
    class_weights = class_weights.to(device)

    weighted_ce = nn.CrossEntropyLoss(weight=class_weights)

    # Custom training loop (avoid WideDeep Trainer device issues)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Create dataloaders
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train_tab, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_valid_tab, dtype=torch.float32),
        torch.tensor(y_valid, dtype=torch.long)
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print(f"\nStarting training with:")
    print(f"  Learning rate: 1e-3")
    print(f"  Batch size: 128")
    print(f"  Gradient clip: 5.0")
    print(f"  Epochs: 20")
    print(f"  Class weights: {class_weights.cpu().numpy()}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 3
    
    for epoch in range(20):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = weighted_ce(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x)
                loss = weighted_ce(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1}/20 - Loss: {train_loss:.4f} Acc: {train_acc:.2f}% - Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss - 0.001:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'Models/Saint/model_state_dict.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    # ------------------------------------------------------------------
    # Proper saving
    # ------------------------------------------------------------------

    # Create a directory for all model artifacts
    model_dir = Path("Models/saint_model")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Ensure target artifacts directory exists
    target_dir = Path("Models/Saint")
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1) Save the preprocessor
    joblib.dump(tab_preprocessor, str(target_dir / "tab_preprocessor.joblib"))

    # 2) Save model state dict
    torch.save(model.state_dict(), str(target_dir / "model_state_dict.pt"))

    # 3) Save metadata/config as a dictionary
    model_config = {
        "model_type": "SAINT",
        "architecture": {
            "input_dim": 16,
            "n_heads": 2,
            "n_blocks": 1,
            "attn_dropout": 0.0,
            "pred_dim": 5
        },
        "features": {
            "num_cols": num_cols,
            "cat_cols": multi_class_cat_cols
        },
        "files": {
            "preprocessor": "tab_preprocessor.joblib",
            "weights": "model_state_dict.pt"
        }
    }

    joblib.dump(model_config, str(target_dir / "config.joblib"))

    print(f"✓ Saved model artifacts to {target_dir}/")
    print(f"  - tab_preprocessor.joblib")
    print(f"  - model_state_dict.pt")
    print(f"  - config.joblib")
    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    print("\nEvaluating on test set...")
    y_pred = trainer.predict(X_tab=X_test_tab)

    from sklearn.metrics import accuracy_score, classification_report
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)

