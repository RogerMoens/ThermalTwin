"""Multi-step LSTM predictor over windowed weather-only features."""

from pathlib import Path

import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model, layers

from .base_predictor import BasePredictor


class NeuralWeatherOnlyPredictor(BasePredictor):
    """LSTM predictor mapping a windowed weather history to a multi-step horizon."""

    def __init__(
        self,
        lookback=48,
        n_features=6,
        horizon=12,
    ):

        self.lookback = lookback
        self.n_features = n_features
        self.horizon = horizon

        self.scaler = StandardScaler()
        self.model = self._build_model()

    def _build_model(self):

        inputs = layers.Input(
            shape=(self.lookback, self.n_features)
        )

        x = layers.LSTM(64, return_sequences=True)(inputs)
        x = layers.LSTM(32)(x)

        x = layers.Dense(64, activation="relu")(x)

        outputs = layers.Dense(self.horizon)(x)

        model = Model(inputs, outputs)

        model.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mae"],
        )

        return model

    def _scale(self, X, fit=False):
        """Scale a windowed 3D array feature-wise, preserving its shape."""

        n_samples, lookback, n_features = X.shape
        flat = X.reshape(-1, n_features)

        flat = (
            self.scaler.fit_transform(flat)
            if fit
            else self.scaler.transform(flat)
        )

        return flat.reshape(n_samples, lookback, n_features).astype(
            X.dtype, copy=False
        )

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):

        X = self._scale(X_train, fit=True)

        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (self._scale(X_val), y_val)

        self.model.fit(
            X,
            y_train,
            epochs=kwargs.get("epochs", 30),
            batch_size=kwargs.get("batch_size", 256),
            validation_data=validation_data,
            verbose=kwargs.get("verbose", 1),
        )

    def predict(self, X):

        X = self._scale(X)
        return self.model.predict(X, verbose=0)

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
        obj.lookback = obj.model.input_shape[1]
        obj.n_features = obj.model.input_shape[2]
        obj.horizon = obj.model.output_shape[-1]

        return obj
