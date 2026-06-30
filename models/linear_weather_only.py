import joblib
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from .base_predictor import BasePredictor


class LinearWeatherOnlyPredictor(BasePredictor):

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LinearRegression()

    def _flatten(self, X):
        return X.reshape(X.shape[0], -1)

    def fit(self, X_train, y_train, **kwargs):

        X = self._flatten(X_train)
        X = self.scaler.fit_transform(X)

        self.model.fit(X, y_train)

    def predict(self, X):

        X = self._flatten(X)
        X = self.scaler.transform(X)

        return self.model.predict(X)

    def save(self, folder):

        joblib.dump(self.model, folder / "linear.pkl")
        joblib.dump(self.scaler, folder / "scaler.pkl")

    @classmethod
    def load(cls, folder):

        obj = cls()
        obj.model = joblib.load(folder / "linear.pkl")
        obj.scaler = joblib.load(folder / "scaler.pkl")
        return obj