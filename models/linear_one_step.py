"""One-step-ahead linear regression predictor."""

from pathlib import Path

import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from .base_predictor import BasePredictor


class LinearOneStepPredictor(BasePredictor):
    """Ordinary least squares predictor for single-step-ahead forecasts."""

    def __init__(self):

        self.scaler = StandardScaler()
        self.model = LinearRegression()

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):

        X = self.scaler.fit_transform(X_train)
        self.model.fit(X, y_train)

    def predict(self, X):

        X = self.scaler.transform(X)
        return self.model.predict(X)

    def save(self, folder):

        folder = self._prepare_folder(folder)
        joblib.dump(self.model, folder / "linear.pkl")
        joblib.dump(self.scaler, folder / "scaler.pkl")

    @classmethod
    def load(cls, folder):

        folder = Path(folder)

        obj = cls.__new__(cls)
        obj.model = joblib.load(folder / "linear.pkl")
        obj.scaler = joblib.load(folder / "scaler.pkl")

        return obj
