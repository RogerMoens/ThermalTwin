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