"""
statistical.py
--------------
SARIMA forecaster — extracted from forecast_hnd_lax.py so experiment.py
can reuse it without duplicating model code.
"""

import warnings
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .base import BaseForecaster

warnings.filterwarnings("ignore")


class SARIMAForecaster(BaseForecaster):
    """
    Seasonal ARIMA wrapper.

    Default order matches the detailed single-route analysis:
      SARIMA(2,0,2)(1,1,1)[7]  — weekly seasonal period

    For batch runs over many O-D pairs, use a simpler order to reduce
    fitting time:
      SARIMAForecaster(order=(1,1,1), seasonal_order=(0,1,1,7))
    """

    def __init__(
        self,
        order: tuple = (2, 0, 2),
        seasonal_order: tuple = (1, 1, 1, 7),
        maxiter: int = 300,
    ):
        self.order          = order
        self.seasonal_order = seasonal_order
        self.maxiter        = maxiter
        self._result        = None

    def fit(self, y_train: np.ndarray, **kwargs) -> "SARIMAForecaster":
        model = SARIMAX(
            y_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._result = model.fit(disp=False, maxiter=self.maxiter)
        return self

    def predict(self, steps: int) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("Call fit() before predict().")
        return np.asarray(self._result.forecast(steps=steps))
