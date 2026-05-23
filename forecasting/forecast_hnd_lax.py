"""
forecast_hnd_lax.py
-------------------
Detailed single-route forecast for HND->LAX using three models:

  1. SARIMA(2,0,2)(1,1,1)[7]   — statsmodels; captures weekly DOW seasonality
  2. LSTM (2-layer, hidden=128) — PyTorch
  3. Transformer (encoder-only) — PyTorch

This script forecasts load_factor (constrained, 0–1) as a detailed
per-route analysis. For the full unconstraining → forecasting pipeline
across all O-D pairs, see experiment.py.

Train: 2015-01-01 → 2022-12-31
Test:  2023-01-01 → 2024-12-31

Outputs:
  forecasting/results/comparison.csv
  forecasting/results/forecast_comparison.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE      = os.path.dirname(os.path.abspath(__file__))
ROOT      = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)

from forecasting.models import (
    SARIMAForecaster,
    LSTMForecaster, TransformerForecaster,
    train_neural, predict_neural,
)
from forecasting.utils import make_sequences, summary as compute_metrics
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH   = os.path.join(ROOT, "output", "simulation_slim.csv")
RESULTS_DIR = os.path.join(HERE, "results")

ROUTE      = "HND->LAX"
SEQ_LEN    = 30
TRAIN_END  = "2022-12-31"
TEST_START = "2023-01-01"
FEATURE_COLS = ["load_factor", "month_sin", "month_cos", "dow_sin", "dow_cos", "year_norm"]


# ── Data ──────────────────────────────────────────────────────────────────────

def load_and_prepare(path: str, route: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["flight_od"] == route].sort_values("date").reset_index(drop=True)
    df = df[["date", "load_factor", "month", "dow", "year"]].copy()
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dow"] / 7)
    df["year_norm"] = (df["year"] - df["year"].min()) / max(df["year"].max() - df["year"].min(), 1)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Device: {device}")

    # 1. Load & split
    print(f"\nLoading {ROUTE} daily load factor...")
    df       = load_and_prepare(DATA_PATH, ROUTE)
    train_df = df[df["date"] <= TRAIN_END].reset_index(drop=True)
    test_df  = df[df["date"] >= TEST_START].reset_index(drop=True)
    print(f"  Total: {len(df)} days  |  Train: {len(train_df)}  |  Test: {len(test_df)}")

    y_test = test_df["load_factor"].values

    # 2. Scale (fit on train only)
    scaler    = StandardScaler()
    train_arr = scaler.fit_transform(train_df[FEATURE_COLS].values).astype("float32")
    test_arr  = scaler.transform(test_df[FEATURE_COLS].values).astype("float32")

    lf_mean, lf_std = scaler.mean_[0], scaler.scale_[0]
    inv = lambda x: np.clip(x * lf_std + lf_mean, 0.0, 1.0)

    border_arr    = np.concatenate([train_arr[-SEQ_LEN:], test_arr], axis=0)
    X_te, _       = make_sequences(border_arr, SEQ_LEN)
    X_all, y_all  = make_sequences(train_arr, SEQ_LEN)
    val_cut       = int(len(X_all) * 0.90)
    X_tr, y_tr    = X_all[:val_cut], y_all[:val_cut]
    X_val, y_val  = X_all[val_cut:], y_all[val_cut:]
    n_feat        = len(FEATURE_COLS)

    results = []

    # 3. SARIMA
    print("\n[1/3] Fitting SARIMA(2,0,2)(1,1,1)[7]...")
    sarima      = SARIMAForecaster(order=(2,0,2), seasonal_order=(1,1,1,7))
    sarima.fit(train_df["load_factor"].values)
    sarima_pred = np.clip(sarima.predict(len(test_df)), 0.0, 1.0)
    results.append(compute_metrics(y_test, sarima_pred, "SARIMA(2,0,2)(1,1,1)[7]"))

    # 4. LSTM
    print("\n[2/3] Training LSTM...")
    lstm      = train_neural(LSTMForecaster(n_feat), X_tr, y_tr, X_val, y_val)
    lstm_pred = inv(predict_neural(lstm, X_te))
    results.append(compute_metrics(y_test, lstm_pred, "LSTM"))

    # 5. Transformer
    print("\n[3/3] Training Transformer...")
    tsfm      = train_neural(TransformerForecaster(n_feat), X_tr, y_tr, X_val, y_val, lr=5e-4)
    tsfm_pred = inv(predict_neural(tsfm, X_te))
    results.append(compute_metrics(y_test, tsfm_pred, "Transformer"))

    # 6. Summary
    res_df = pd.DataFrame(results).set_index("model")
    print("\n" + "=" * 60)
    print(f"MODEL COMPARISON — {ROUTE}  (test: 2023–2024)")
    print("=" * 60)
    print(res_df.to_string())
    print(f"\n  Best (RMSE): {res_df['RMSE'].idxmin()}")
    res_df.to_csv(os.path.join(RESULTS_DIR, "comparison.csv"))

    # 7. Plot
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
            f"{label}   —   MAE={row['MAE']:.4f}  RMSE={row['RMSE']:.4f}  MAPE={row['MAPE']:.2f}%",
            fontsize=10, loc="left",
        )
        ax.set_ylabel("Load Factor", fontsize=9)
        ax.set_ylim(0.0, 1.10)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.25)

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=25, ha="right")
    fig.suptitle(f"HND→LAX Daily Load Factor Forecast  |  Test: 2023–2024",
                 fontsize=13, y=1.005)
    plt.tight_layout()

    out = os.path.join(RESULTS_DIR, "forecast_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Plot → {out}")


if __name__ == "__main__":
    main()
