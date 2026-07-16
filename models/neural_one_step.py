"""One-step-ahead feed-forward neural network predictor."""

from pathlib import Path

import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

from .base_predictor import BasePredictor


class NeuralOneStepPredictor(BasePredictor):
    """Small feed-forward network for single-step-ahead forecasts."""

    def __init__(self, n_features=102):

        self.n_features = n_features
        self.scaler = StandardScaler()
        self.model = self._build_model()

    def _build_model(self):

        model = Sequential([
            Input(shape=(self.n_features,)),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(1),
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss="mse",
            metrics=["mae"],
        )

        return model

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):

        X = self.scaler.fit_transform(X_train)

        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (self.scaler.transform(X_val), y_val)

        self.model.fit(
            X,
            y_train,
            epochs=kwargs.get("epochs", 200),
            batch_size=kwargs.get("batch_size", 256),
            validation_data=validation_data,
            verbose=kwargs.get("verbose", 1),
        )

    def predict(self, X):

        X = self.scaler.transform(X)
        return self.model.predict(X, verbose=0).reshape(-1)

    def save(self, folder):

        folder = self._prepare_folder(folder)
        self.model.save(folder / "model.keras")
        joblib.dump(self.scaler, folder / "scaler.pkl")

    @classmethod
    def load(cls, folder):

        folder = Path(folder)

        obj = cls.__new__(cls)
        obj.model = tf.keras.models.load_model(folder / "model.keras")
        obj.scaler = joblib.load(folder / "scaler.pkl")
        obj.n_features = obj.model.input_shape[-1]

        return obj
