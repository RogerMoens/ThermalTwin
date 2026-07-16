"""Shared predictor interface for all ThermalTwin models."""

from abc import ABC, abstractmethod
from pathlib import Path


class BasePredictor(ABC):
    """
    Common interface implemented by every predictor in ``models/``.

    Subclasses that don't use validation data or extra keyword options
    should still accept ``X_val``/``y_val``/``**kwargs`` (even if
    ignored) so the signature stays interchangeable across the model
    family.
    """

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def save(self, folder):
        pass

    @classmethod
    @abstractmethod
    def load(cls, folder):
        pass

    @staticmethod
    def _prepare_folder(folder):
        """Resolve ``folder`` to a ``Path`` and ensure it exists."""

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        return folder


class FlattenMixin:
    """Shared flattening of windowed ``(samples, timesteps, features)`` input."""

    @staticmethod
    def _flatten(X):
        return X.reshape(X.shape[0], -1)
