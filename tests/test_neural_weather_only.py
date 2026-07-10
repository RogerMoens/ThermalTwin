import numpy as np

from models.neural_weather_only import NeuralWeatherOnlyPredictor


def test_fit_predict():
    rng = np.random.default_rng(42)

    lookback = 48
    n_features = 6
    horizon = 12
    n_samples = 32

    X = rng.normal(size=(n_samples, lookback, n_features)).astype(np.float32)
    y = rng.normal(size=(n_samples, horizon)).astype(np.float32)

    model = NeuralWeatherOnlyPredictor(
        lookback=lookback,
        n_features=n_features,
        horizon=horizon,
    )

    model.fit(
        X,
        y,
        epochs=1,
        batch_size=8,
        verbose=0,
    )

    preds = model.predict(X)

    assert preds.shape == y.shape
    assert np.isfinite(preds).all()


def test_save_and_load(tmp_path):
    rng = np.random.default_rng(42)

    lookback = 48
    n_features = 6
    horizon = 12
    n_samples = 32

    X = rng.normal(size=(n_samples, lookback, n_features)).astype(np.float32)
    y = rng.normal(size=(n_samples, horizon)).astype(np.float32)

    model = NeuralWeatherOnlyPredictor(
        lookback=lookback,
        n_features=n_features,
        horizon=horizon,
    )

    model.fit(
        X,
        y,
        epochs=1,
        batch_size=8,
        verbose=0,
    )

    expected = model.predict(X)

    model.save(tmp_path)

    loaded_model = NeuralWeatherOnlyPredictor.load(tmp_path)

    actual = loaded_model.predict(X)

    np.testing.assert_allclose(actual, expected, atol=1e-6)