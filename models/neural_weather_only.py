import tensorflow as tf
from tensorflow.keras import Model, layers

from .base_predictor import BasePredictor


class NeuralWeatherOnlyPredictor(BasePredictor):

    def __init__(
        self,
        lookback=48,
        n_features=6,
        horizon=12,
    ):

        self.lookback = lookback
        self.n_features = n_features
        self.horizon = horizon

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

    def fit(self, X_train, y_train, **kwargs):

        self.model.fit(
            X_train,
            y_train,
            epochs=kwargs.get("epochs", 30),
            batch_size=kwargs.get("batch_size", 256),
            validation_data=kwargs.get("validation_data"),
            verbose=kwargs.get("verbose", 1),
        )

    def predict(self, X):

        return self.model.predict(X, verbose=0)

    def save(self, folder):

        self.model.save(folder / "model.keras")

    @classmethod
    def load(cls, folder):

        obj = cls()
        obj.model = tf.keras.models.load_model(folder / "model.keras")
        return obj