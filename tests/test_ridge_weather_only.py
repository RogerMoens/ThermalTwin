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


def test_save_creates_missing_nested_directory(tmp_path):
    X = np.random.default_rng(0).normal(size=(50, 3, 4))
    y = X.reshape(50, -1).sum(axis=1)

    model = RidgeWeatherOnlyPredictor(alpha=1e-8)
    model.fit(X, y)

    target = tmp_path / "nested" / "does_not_exist_yet"
    model.save(target)

    assert (target / "ridge.pkl").exists()
    assert (target / "scaler.pkl").exists()


def test_save_accepts_string_path(tmp_path):
    X = np.random.default_rng(0).normal(size=(50, 3, 4))
    y = X.reshape(50, -1).sum(axis=1)

    model = RidgeWeatherOnlyPredictor(alpha=1e-8)
    model.fit(X, y)

    model.save(str(tmp_path / "as_string"))

    assert (tmp_path / "as_string" / "ridge.pkl").exists()


def test_fit_accepts_validation_data_kwargs():
    X = np.random.default_rng(0).normal(size=(50, 3, 4))
    y = X.reshape(50, -1).sum(axis=1)

    model = RidgeWeatherOnlyPredictor(alpha=1e-8)

    # X_val/y_val are unused by this model but must be accepted, not error out
    model.fit(X, y, X_val=X, y_val=y)

    preds = model.predict(X)
    np.testing.assert_allclose(preds, y, atol=1e-6)


def test_does_not_store_redundant_alpha_attribute():
    model = RidgeWeatherOnlyPredictor(alpha=0.5)

    assert not hasattr(model, "alpha")
    assert model.model.alpha == 0.5