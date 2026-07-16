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


def test_scaler_is_fit_during_training():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=10.0, scale=3.0, size=(20, 48, 6)).astype(np.float32)
    y = rng.normal(size=(20, 12)).astype(np.float32)

    model = NeuralWeatherOnlyPredictor(lookback=48, n_features=6, horizon=12)
    assert not hasattr(model.scaler, "mean_")

    model.fit(X, y, epochs=1, batch_size=8, verbose=0)

    assert hasattr(model.scaler, "mean_")
    assert model.scaler.mean_.shape == (6,)


def test_save_and_load_preserves_scaler_and_config(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(loc=10.0, scale=3.0, size=(16, 48, 6)).astype(np.float32)
    y = rng.normal(size=(16, 12)).astype(np.float32)

    model = NeuralWeatherOnlyPredictor(lookback=48, n_features=6, horizon=12)
    model.fit(X, y, epochs=1, batch_size=8, verbose=0)
    model.save(tmp_path)

    loaded = NeuralWeatherOnlyPredictor.load(tmp_path)

    np.testing.assert_allclose(loaded.scaler.mean_, model.scaler.mean_)
    np.testing.assert_allclose(loaded.scaler.scale_, model.scaler.scale_)
    assert loaded.lookback == 48
    assert loaded.n_features == 6
    assert loaded.horizon == 12


def test_fit_accepts_validation_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(16, 48, 6)).astype(np.float32)
    y = rng.normal(size=(16, 12)).astype(np.float32)
    X_val = rng.normal(size=(4, 48, 6)).astype(np.float32)
    y_val = rng.normal(size=(4, 12)).astype(np.float32)

    model = NeuralWeatherOnlyPredictor(lookback=48, n_features=6, horizon=12)

    # should not raise
    model.fit(X, y, X_val=X_val, y_val=y_val, epochs=1, batch_size=8, verbose=0)