import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(y_true - y_pred).mean())


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 1e-6
    return float(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]).mean() * 100)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error — robust when actuals include zeros."""
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / denom * 100) if denom > 0 else np.nan


def summary(y_true: np.ndarray, y_pred: np.ndarray, name: str = "") -> dict:
    return {
        "model": name,
        "MAE":   round(mae(y_true, y_pred), 2),
        "RMSE":  round(rmse(y_true, y_pred), 2),
        "MAPE":  round(mape(y_true, y_pred), 3),
        "WAPE":  round(wape(y_true, y_pred), 3),
    }
