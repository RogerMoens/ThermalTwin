"""Ridge regression predictor over windowed weather-only features."""

from pathlib import Path

import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .base_predictor import BasePredictor, FlattenMixin


class RidgeWeatherOnlyPredictor(FlattenMixin, BasePredictor):
    """
    L2-regularized predictor over a flattened
    ``(samples, timesteps, features)`` window.
    """

    def __init__(self, alpha=1.0):

        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):

        X = self._flatten(X_train)
        X = self.scaler.fit_transform(X)

        self.model.fit(X, y_train)

    def predict(self, X):

        X = self._flatten(X)
        X = self.scaler.transform(X)

        return self.model.predict(X)

    def save(self, folder):

        folder = self._prepare_folder(folder)
        joblib.dump(self.model, folder / "ridge.pkl")
        joblib.dump(self.scaler, folder / "scaler.pkl")

    @classmethod
    def load(cls, folder):

        folder = Path(folder)

        obj = cls.__new__(cls)
        obj.model = joblib.load(folder / "ridge.pkl")
        obj.scaler = joblib.load(folder / "scaler.pkl")

        return obj
