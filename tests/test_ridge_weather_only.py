import numpy as np

from models.ridge_weather_only import RidgeWeatherOnlyPredictor


def test_fit_predict():
    rng = np.random.default_rng(42)

    X = rng.normal(size=(50, 3, 4))

    # Target is a linear function of the flattened input
    y = X.reshape(50, -1).sum(axis=1)

    model = RidgeWeatherOnlyPredictor(alpha=1e-8)
    model.fit(X, y)

    preds = model.predict(X)

    assert preds.shape == y.shape
    np.testing.assert_allclose(preds, y, atol=1e-6)


def test_save_and_load(tmp_path):
    rng = np.random.default_rng(42)

    X = rng.normal(size=(50, 3, 4))
    y = X.reshape(50, -1).sum(axis=1)

    model = RidgeWeatherOnlyPredictor(alpha=1e-8)
    model.fit(X, y)

    expected = model.predict(X)

    model.save(tmp_path)

    loaded_model = RidgeWeatherOnlyPredictor.load(tmp_path)

    actual = loaded_model.predict(X)

    np.testing.assert_allclose(actual, expected, atol=1e-6)