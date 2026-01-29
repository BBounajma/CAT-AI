import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class new_SAINTWrapper:
    def __init__(
        self,
        model,
        cat_idx,
        cat_dims,
        num_continuous,
        device="cpu",
        lr=1e-3
    ):
        self.model = model.to(device)
        self.cat_idx = cat_idx
        self.cat_dims = cat_dims
        self.num_continuous = num_continuous
        self.device = device

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    # ------------------------------------------------------------------
    # ONLY ADDITION: embedding creation (matches SAINT.forward signature)
    # ------------------------------------------------------------------
    def _encode(self, x_cat, x_cont):
    
        device = x_cat.device

    # categorical embeddings (with offset!)
        x_cat_enc = self.model.embeds(
            x_cat + self.model.categories_offset.to(device)
        )

    # continuous embeddings (MLP-based, as in SAINT)
        x_cont_enc = torch.stack(
            [
                mlp(x_cont[:, i:i+1])
                for i, mlp in enumerate(self.model.simple_MLP)
            ],
            dim=1
        )
        return x_cat_enc, x_cont_enc

    # ------------------------------------------------------------------
    # Fit (structure unchanged, only forward call fixed)
    # ------------------------------------------------------------------
    def fit(
        self,
        X_train,
        y_train,
        X_valid=None,
        y_valid=None,
        epochs=10,
        batch_size=1024
    ):
        self.model.train()

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0.0

            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                # Extract one-hot encoded categorical and continuous features
                x_cat_onehot = xb[:, self.cat_idx]
                x_cont = xb[:, [i for i in range(xb.shape[1]) if i not in self.cat_idx]]

                # Convert one-hot encoded to categorical indices
                # For each categorical variable (group of 2 columns), get the argmax to get the category index
                x_cat_idx = torch.zeros((xb.shape[0], len(self.cat_dims)), dtype=torch.long, device=self.device)
                col_idx = 0
                for i, card in enumerate(self.cat_dims):
                    x_cat_idx[:, i] = x_cat_onehot[:, col_idx:col_idx+card].argmax(dim=1)
                    col_idx += card

                # Embed categorical features with offset
                x_cat_embedded = self.model.embeds(
                    x_cat_idx + self.model.categories_offset.to(self.device)
                )  # Shape: [batch_size, num_categories, dim]

                # Embed continuous features through MLPs
                x_cont_embedded_list = []
                for i, mlp in enumerate(self.model.simple_MLP):
                    x_cont_embedded_list.append(mlp(x_cont[:, i:i+1]))
                x_cont_embedded = torch.stack(x_cont_embedded_list, dim=1)  # Shape: [batch_size, num_continuous, dim]

                # ---- FIX ----
                # Concatenate embeddings and pass to model
                x_combined = torch.cat((x_cat_embedded, x_cont_embedded), dim=1)  # [batch_size, num_categories + num_continuous, dim]
                
                # Pass through transformer
                x_transformed = self.model.transformer(x_combined)
                
                # Get predictions - model returns cat_outs, con_outs
                cat_outs, con_outs = self.model.mlp1(x_transformed[:,:self.model.num_categories,:]), self.model.mlp2(x_transformed[:,self.model.num_categories:,:])
                
                # For regression, use con_outs or a weighted combination
                preds = con_outs.squeeze(-1) if con_outs.shape[-1] == 1 else con_outs.mean(dim=1)

                # -------------

                loss = self.criterion(preds, yb.float())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1} | Train loss: {total_loss / len(train_loader):.6f}")

            if X_valid is not None:
                val_loss = self.evaluate(X_valid, y_valid, batch_size)
                print(f"           | Val loss:   {val_loss:.6f}")

    # ------------------------------------------------------------------
    # Evaluate (same structure, fixed forward)
    # ------------------------------------------------------------------
    def evaluate(self, X, y, batch_size=2048):
        self.model.eval()

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        ds = TensorDataset(X, y)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        total_loss = 0.0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                x_cat = xb[:, self.cat_idx].long()
                x_cont = xb[:, [i for i in range(xb.shape[1]) if i not in self.cat_idx]]

                x_cat_enc, x_cont_enc = self._create_embeddings(x_cat, x_cont)
                preds = self.model(
                    x_cat,
                    x_cont,
                    x_cat_enc,
                    x_cont_enc
                )

                total_loss += self.criterion(preds, yb).item()

        self.model.train()
        return total_loss / len(loader)

    # ------------------------------------------------------------------
    # Predict (same structure, fixed forward)
    # ------------------------------------------------------------------
    def predict(self, X, batch_size=2048):
        self.model.eval()

        X = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(X, batch_size=batch_size, shuffle=False)

        preds_all = []
        with torch.no_grad():
            for xb in loader:
                xb = xb.to(self.device)

                x_cat = xb[:, self.cat_idx].long()
                x_cont = xb[:, [i for i in range(xb.shape[1]) if i not in self.cat_idx]]

                x_cat_enc, x_cont_enc = self._encode(x_cat, x_cont)
                preds = self.model(
                    x_cat,
                    x_cont,
                    x_cat_enc,
                    x_cont_enc
                )

                preds_all.append(preds.cpu())

        self.model.train()
        return torch.cat(preds_all, dim=0).numpy()
