"""
forecast_hnd_lax.py
-------------------
Forecast daily load factor for HND->LAX (JP market) using three models:

  1. SARIMA(2,0,2)(1,1,1)[7]   — statsmodels; captures weekly DOW seasonality
  2. LSTM (2-layer, hidden=128) — PyTorch; learns temporal patterns from sequences
  3. Transformer (encoder-only) — PyTorch; multi-head self-attention over sequences

Train: 2015-01-01 → 2022-12-31  (2,922 days)
Test:  2023-01-01 → 2024-12-31  (731 days)

Metrics: MAE, RMSE, MAPE on held-out test set.
Output:
  forecasting/results/comparison.csv
  forecasting/results/forecast_comparison.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── Paths ───────────────────────────────────────────────────────────────────────

HERE        = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(HERE, "..", "output", "simulation_slim.csv")
RESULTS_DIR = os.path.join(HERE, "results")

# ── Config ──────────────────────────────────────────────────────────────────────

ROUTE      = "HND->LAX"
TARGET     = "load_factor"
SEQ_LEN    = 30         # lookback window (days) for neural models
TRAIN_END  = "2022-12-31"
TEST_START = "2023-01-01"

FEATURE_COLS = ["load_factor", "month_sin", "month_cos", "dow_sin", "dow_cos", "year_norm"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Data ────────────────────────────────────────────────────────────────────────

def load_and_prepare(path: str, route: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["flight_od"] == route].sort_values("date").reset_index(drop=True)
    df = df[["date", "load_factor", "month", "dow", "year"]].copy()
    # Cyclical encodings so the model sees continuity across month/DOW boundaries
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dow"] / 7)
    df["year_norm"] = (df["year"] - df["year"].min()) / max(df["year"].max() - df["year"].min(), 1)
    return df


def make_sequences(arr: np.ndarray, seq_len: int):
    """Sliding-window (X, y). y = load_factor (column 0) at step t."""
    X, y = [], []
    for i in range(seq_len, len(arr)):
        X.append(arr[i - seq_len : i])
        y.append(arr[i, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ── Metrics ─────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-6, None))) * 100)
    print(f"  {name:<30}  MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%")
    return {"model": name, "MAE": round(mae, 5), "RMSE": round(rmse, 5), "MAPE": round(mape, 3)}


# ── SARIMA ──────────────────────────────────────────────────────────────────────

def run_sarima(train_y: np.ndarray, test_len: int) -> np.ndarray:
    print("\n[1/3] Fitting SARIMA(2,0,2)(1,1,1)[7]...")
    model = SARIMAX(
        train_y,
        order=(2, 0, 2),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False, maxiter=300)
    pred = result.forecast(steps=test_len)
    return np.clip(np.asarray(pred), 0.0, 1.0)


# ── Neural model training ───────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 12,
) -> nn.Module:
    model = model.to(DEVICE)
    opt  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    crit = nn.MSELoss()

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
            print(f"    Early stop at epoch {epoch}  (best val_loss={best_val:.5f})")
            break
        if epoch % 25 == 0:
            print(f"    Epoch {epoch:3d}  val_loss={val_loss:.5f}")

    model.load_state_dict(best_state)
    return model


def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X).to(DEVICE)).cpu().numpy()


# ── LSTM ────────────────────────────────────────────────────────────────────────

class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):           # x: (B, T, F)
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


# ── Transformer ─────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, : x.size(1)])


class TransformerForecaster(nn.Module):
    def __init__(self, n_features: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dim_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.proj   = nn.Linear(n_features, d_model)
        self.pe     = PositionalEncoding(d_model, dropout=dropout)
        enc_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head    = nn.Linear(d_model, 1)

    def forward(self, x):           # x: (B, T, F)
        x = self.pe(self.proj(x))
        x = self.encoder(x)
        return self.head(x[:, -1]).squeeze(-1)   # last timestep → prediction


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")

    # 1. Load & split ─────────────────────────────────────────────────────────
    print(f"\nLoading {ROUTE} daily load factor...")
    df       = load_and_prepare(DATA_PATH, ROUTE)
    train_df = df[df["date"] <= TRAIN_END].reset_index(drop=True)
    test_df  = df[df["date"] >= TEST_START].reset_index(drop=True)
    print(f"  Total: {len(df)} days  ({df['date'].min().date()} → {df['date'].max().date()})")
    print(f"  Train: {len(train_df)} days  |  Test: {len(test_df)} days")

    y_test = test_df["load_factor"].values

    # 2. Scale (fit on train only) ────────────────────────────────────────────
    scaler    = StandardScaler()
    train_arr = scaler.fit_transform(train_df[FEATURE_COLS].values).astype(np.float32)
    test_arr  = scaler.transform(test_df[FEATURE_COLS].values).astype(np.float32)

    lf_mean, lf_std = scaler.mean_[0], scaler.scale_[0]
    inv = lambda x: np.clip(x * lf_std + lf_mean, 0.0, 1.0)

    # Build neural sequences: stitch last SEQ_LEN train rows onto test for continuity
    border_arr = np.concatenate([train_arr[-SEQ_LEN:], test_arr], axis=0)
    X_te, _    = make_sequences(border_arr, SEQ_LEN)

    X_all, y_all = make_sequences(train_arr, SEQ_LEN)
    val_cut      = int(len(X_all) * 0.90)
    X_tr, y_tr   = X_all[:val_cut], y_all[:val_cut]
    X_val, y_val = X_all[val_cut:], y_all[val_cut:]

    n_feat = len(FEATURE_COLS)
    results = []

    # 3. SARIMA ───────────────────────────────────────────────────────────────
    sarima_pred = run_sarima(train_df["load_factor"].values, len(test_df))
    results.append(compute_metrics(y_test, sarima_pred, "SARIMA(2,0,2)(1,1,1)[7]"))

    # 4. LSTM ─────────────────────────────────────────────────────────────────
    print("\n[2/3] Training LSTM...")
    lstm      = train_model(LSTMForecaster(n_feat), X_tr, y_tr, X_val, y_val)
    lstm_pred = inv(predict(lstm, X_te))
    results.append(compute_metrics(y_test, lstm_pred, "LSTM"))

    # 5. Transformer ──────────────────────────────────────────────────────────
    print("\n[3/3] Training Transformer...")
    tsfm      = train_model(TransformerForecaster(n_feat), X_tr, y_tr, X_val, y_val, lr=5e-4)
    tsfm_pred = inv(predict(tsfm, X_te))
    results.append(compute_metrics(y_test, tsfm_pred, "Transformer"))

    # 6. Summary ──────────────────────────────────────────────────────────────
    res_df = pd.DataFrame(results).set_index("model")
    print("\n" + "=" * 60)
    print(f"MODEL COMPARISON — {ROUTE}  (test: 2023–2024)")
    print("=" * 60)
    print(res_df.to_string())
    winner = res_df["RMSE"].idxmin()
    print(f"\n  Best model (RMSE): {winner}")
    res_df.to_csv(os.path.join(RESULTS_DIR, "comparison.csv"))

    # 7. Plot ─────────────────────────────────────────────────────────────────
    test_dates = test_df["date"].values
    models = [
        (sarima_pred, "SARIMA(2,0,2)(1,1,1)[7]", "#3B6FA0"),
        (lstm_pred,   "LSTM",                     "#2E8B57"),
        (tsfm_pred,   "Transformer",               "#C05050"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(16, 13), sharex=True)
    for ax, (preds, label, color) in zip(axes, models):
        ax.plot(test_dates, y_test, color="black", alpha=0.35, lw=0.9, label="Actual")
        ax.plot(test_dates, preds,  color=color,   alpha=0.90, lw=0.9, label=label)
        row = res_df.loc[label]
        ax.set_title(
            f"{label}   —   MAE={row['MAE']:.4f}  |  RMSE={row['RMSE']:.4f}  |  MAPE={row['MAPE']:.2f}%",
            fontsize=10, loc="left",
        )
        ax.set_ylabel("Load Factor", fontsize=9)
        ax.set_ylim(0.0, 1.10)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.25)

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=25, ha="right")

    fig.suptitle(
        f"HND→LAX Daily Load Factor Forecast  |  Test period: 2023–2024",
        fontsize=13, y=1.005,
    )
    plt.tight_layout()

    plot_path = os.path.join(RESULTS_DIR, "forecast_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
