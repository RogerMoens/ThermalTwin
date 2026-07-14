from pathlib import Path

import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .base_predictor import BasePredictor

class RidgeOneStepPredictor(BasePredictor):

    def __init__(self, alpha=1.0):
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)

    def fit(self, X_train, y_train, **kwargs):

        X = self.scaler.fit_transform(X_train)
        self.model.fit(X, y_train)

    def predict(self, X):

        X = self.scaler.transform(X)
        return self.model.predict(X)

    def save(self, folder):

        joblib.dump(self.model, folder / "ridge.pkl")
        joblib.dump(self.scaler, folder / "scaler.pkl")

    @classmethod
    def load(cls, folder):

        obj = cls()
        obj.model = joblib.load(folder / "ridge.pkl")
        obj.scaler = joblib.load(folder / "scaler.pkl")
        return obj