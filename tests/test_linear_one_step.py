import numpy as np

from models.linear_one_step import LinearOneStepPredictor


def test_fit_predict():
    X = np.array([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ])

    y = np.array([2.0, 4.0, 6.0, 8.0])

    model = LinearOneStepPredictor()
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

    model = LinearOneStepPredictor()
    model.fit(X, y)

    expected = model.predict(X)

    model.save(tmp_path)

    loaded_model, = (LinearOneStepPredictor.load(tmp_path),)
    # or simply:
    # loaded_model = LinearOneStepPredictor.load(tmp_path)

    actual = loaded_model.predict(X)

    np.testing.assert_allclose(actual, expected)


def test_save_creates_missing_nested_directory(tmp_path):
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0])

    model = LinearOneStepPredictor()
    model.fit(X, y)

    target = tmp_path / "nested" / "does_not_exist_yet"
    model.save(target)

    assert (target / "linear.pkl").exists()
    assert (target / "scaler.pkl").exists()


def test_save_accepts_string_path(tmp_path):
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0])

    model = LinearOneStepPredictor()
    model.fit(X, y)

    model.save(str(tmp_path / "as_string"))

    assert (tmp_path / "as_string" / "linear.pkl").exists()


def test_fit_accepts_validation_data_kwargs():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([2.0, 4.0, 6.0, 8.0])

    model = LinearOneStepPredictor()

    # X_val/y_val are unused by this model but must be accepted, not error out
    model.fit(X, y, X_val=X, y_val=y)

    preds = model.predict(X)
    np.testing.assert_allclose(preds, y, atol=1e-6)