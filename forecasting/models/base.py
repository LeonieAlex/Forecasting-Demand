from abc import ABC, abstractmethod
import numpy as np


class BaseForecaster(ABC):
    """Minimal interface shared by all forecasting models."""

    @abstractmethod
    def fit(self, y_train: np.ndarray, **kwargs) -> "BaseForecaster":
        ...

    @abstractmethod
    def predict(self, steps: int) -> np.ndarray:
        ...
