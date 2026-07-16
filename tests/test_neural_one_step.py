import numpy as np

from models.neural_one_step import NeuralOneStepPredictor


def test_fit_predict():
    rng = np.random.default_rng(42)

    X = rng.normal(size=(100, 102))
    y = X.sum(axis=1)

    model = NeuralOneStepPredictor()
    model.fit(X, y)

    preds = model.predict(X)

    assert preds.shape == y.shape

    # The model should have learned something.
    mse = np.mean((preds - y) ** 2)
    assert mse < 1.0


def test_save_and_load(tmp_path):
    rng = np.random.default_rng(42)

    X = rng.normal(size=(100, 102))
    y = X.sum(axis=1)

    model = NeuralOneStepPredictor()
    model.fit(X, y)

    expected = model.predict(X)

    model.save(tmp_path)

    loaded_model = NeuralOneStepPredictor.load(tmp_path)

    actual = loaded_model.predict(X)

    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_configurable_n_features():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 10))
    y = X.sum(axis=1)

    model = NeuralOneStepPredictor(n_features=10)
    model.fit(X, y, epochs=1, verbose=0)

    preds = model.predict(X)
    assert preds.shape == y.shape


def test_save_and_load_preserves_n_features(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 7))
    y = X.sum(axis=1)

    model = NeuralOneStepPredictor(n_features=7)
    model.fit(X, y, epochs=1, verbose=0)
    model.save(tmp_path)

    loaded = NeuralOneStepPredictor.load(tmp_path)

    assert loaded.n_features == 7
    preds = loaded.predict(X)
    assert preds.shape == y.shape


def test_fit_accepts_validation_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))
    y = X.sum(axis=1)
    X_val = rng.normal(size=(5, 5))
    y_val = X_val.sum(axis=1)

    model = NeuralOneStepPredictor(n_features=5)

    # should not raise
    model.fit(X, y, X_val=X_val, y_val=y_val, epochs=1, verbose=0)


def test_save_creates_missing_nested_directory(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 4))
    y = X.sum(axis=1)

    model = NeuralOneStepPredictor(n_features=4)
    model.fit(X, y, epochs=1, verbose=0)

    target = tmp_path / "nested" / "missing"
    model.save(target)

    assert (target / "model.keras").exists()
    assert (target / "scaler.pkl").exists()