"""
neural.py
---------
LSTM and Transformer forecasters — extracted from forecast_hnd_lax.py.
Both accept (B, T, F) tensors and return a scalar per sample.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── LSTM ──────────────────────────────────────────────────────────────────────

class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128,
                 layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):           # (B, T, F) → (B,)
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


# ── Transformer ───────────────────────────────────────────────────────────────

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-np.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, : x.size(1)])


class TransformerForecaster(nn.Module):
    def __init__(self, n_features: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dim_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.proj    = nn.Linear(n_features, d_model)
        self.pe      = _PositionalEncoding(d_model, dropout=dropout)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head    = nn.Linear(d_model, 1)

    def forward(self, x):           # (B, T, F) → (B,)
        x = self.pe(self.proj(x))
        return self.head(self.encoder(x)[:, -1]).squeeze(-1)


# ── Shared training loop ──────────────────────────────────────────────────────

def train_neural(
    model: nn.Module,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 12,
) -> nn.Module:
    model = model.to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    crit  = nn.MSELoss()

    Xt = torch.tensor(X_tr).to(DEVICE)
    yt = torch.tensor(y_tr).to(DEVICE)
    Xv = torch.tensor(X_val).to(DEVICE)
    yv = torch.tensor(y_val).to(DEVICE)

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    best_val, wait, best_state = float("inf"), 0, None

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            crit(model(xb), yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = crit(model(Xv), yv).item()

        if val_loss < best_val - 1e-6:
            best_val, wait = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        if wait >= patience:
            print(f"    Early stop @ epoch {epoch}  (best val={best_val:.5f})")
            break
        if epoch % 25 == 0:
            print(f"    Epoch {epoch:3d}  val={val_loss:.5f}")

    model.load_state_dict(best_state)
    return model


def predict_neural(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X).to(DEVICE)).cpu().numpy()
