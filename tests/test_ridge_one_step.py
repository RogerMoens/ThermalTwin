import numpy as np

from models.ridge_one_step import RidgeOneStepPredictor


def test_fit_predict():
    X = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ])

    y = np.array([2.0, 4.0, 6.0, 8.0])

    model = RidgeOneStepPredictor(alpha=1.0)
    model.fit(X, y)

    preds = model.predict(X)

    assert preds.shape == y.shape
    np.testing.assert_allclose(preds, y, atol=1e-6)


def test_save_and_load(tmp_path):
    X = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ])

    y = np.array([2.0, 4.0, 6.0, 8.0])

    model = RidgeOneStepPredictor(alpha=1.0)
    model.fit(X, y)

    expected = model.predict(X)

    model.save(tmp_path)

    loaded_model = RidgeOneStepPredictor.load(tmp_path)

    actual = loaded_model.predict(X)

    np.testing.assert_allclose(actual, expected)