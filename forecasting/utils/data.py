"""
data.py
-------
Shared data loading and feature engineering for all forecasting scripts.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time encodings and normalised year trend."""
    df = df.copy()
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)
    df["dow"]        = pd.to_datetime(df["date"]).dt.dayofweek
    df["dow_sin"]    = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["dow"] / 7)
    yr_min, yr_max   = df["year"].min(), df["year"].max()
    df["year_norm"]  = (df["year"] - yr_min) / max(yr_max - yr_min, 1)
    return df


def add_lag_features(df: pd.DataFrame, target_col: str,
                     lags: list[int] | None = None) -> pd.DataFrame:
    """
    Add lagged values of target_col as features.
    Rows where any lag is NaN (start of series) are dropped.
    """
    if lags is None:
        lags = [7, 14, 28, 365]
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    return df.dropna().reset_index(drop=True)


def make_sequences(arr: np.ndarray, seq_len: int):
    """
    Sliding-window sequences for LSTM / Transformer.
    y = value of column 0 (target) at the next step.
    """
    X, y = [], []
    for i in range(seq_len, len(arr)):
        X.append(arr[i - seq_len : i])
        y.append(arr[i, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def load_od(path: str, od_pair: str, signal_col: str = "constrained_seats") -> pd.DataFrame:
    """
    Load one O-D pair from a batch_unconstrained CSV or parquet.
    Returns a tidy DataFrame sorted by date with features added.
    """
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=["date"])

    df = df[df["od_pair"] == od_pair].sort_values("date").reset_index(drop=True)
    df = add_features(df)
    return df


def train_test_split(df: pd.DataFrame, train_end: str, test_start: str):
    train = df[df["date"] <= train_end].reset_index(drop=True)
    test  = df[df["date"] >= test_start].reset_index(drop=True)
    return train, test


def scale_signal(train: np.ndarray, test: np.ndarray):
    """Fit StandardScaler on train, apply to both. Returns (train_s, test_s, scaler)."""
    sc = StandardScaler()
    train_s = sc.fit_transform(train.reshape(-1, 1)).ravel().astype(np.float32)
    test_s  = sc.transform(test.reshape(-1, 1)).ravel().astype(np.float32)
    return train_s, test_s, sc
