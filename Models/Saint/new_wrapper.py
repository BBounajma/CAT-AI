import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SAINTWrapper:
    def __init__(self, model, cat_idx, num_continuous, lr=1e-3, device=None):
        self.model = model
        self.cat_idx = cat_idx
        self.num_cont = num_continuous
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    # --------------------------------------------------
    # embedding creation (matches SAINT.forward exactly)
    # --------------------------------------------------
    def _encode(self, x_cat, x_cont):
        x_cat_enc = self.model.embeds(
            x_cat + self.model.categories_offset
        )

        x_cont_enc = torch.stack(
            [mlp(x_cont[:, i:i+1]) for i, mlp in enumerate(self.model.simple_MLP)],
            dim=1
        )
        return x_cat_enc, x_cont_enc

    # --------------------------------------------------
    # training
    # --------------------------------------------------
    def fit(self, X, y, epochs=10, batch_size=1024):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        loader = DataLoader(
            TensorDataset(X, y),
            batch_size=batch_size,
            shuffle=True
        )

        self.model.train()
        for ep in range(epochs):
            total = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)

                x_cat = xb[:, self.cat_idx].long()
                x_cont = xb[:, [i for i in range(xb.shape[1]) if i not in self.cat_idx]]

                x_cat_enc, x_cont_enc = self._encode(x_cat, x_cont)

                preds = self.model(x_cat, x_cont, x_cat_enc, x_cont_enc)
                loss = self.loss_fn(preds, yb)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                total += loss.item()

            print(f"Epoch {ep+1} | loss {total / len(loader):.6f}")

    # --------------------------------------------------
    # inference
    # --------------------------------------------------
    def predict(self, X, batch_size=2048):
        X = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(X, batch_size=batch_size)

        self.model.eval()
        preds = []

        with torch.no_grad():
            for xb in loader:
                xb = xb.to(self.device)

                x_cat = xb[:, self.cat_idx].long()
                x_cont = xb[:, [i for i in range(xb.shape[1]) if i not in self.cat_idx]]

                x_cat_enc, x_cont_enc = self._encode(x_cat, x_cont)
                preds.append(
                    self.model(x_cat, x_cont, x_cat_enc, x_cont_enc).cpu()
                )

        self.model.train()
        return torch.cat(preds).numpy()
