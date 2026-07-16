import numpy as np
import pandas as pd

from utils.split import DatasetSplitter


def _make_xy(n):
    X = pd.DataFrame({"a": np.arange(n)})
    y = pd.Series(np.arange(n), name="y")
    return X, y


def test_split_sizes_and_chronological_order():
    X, y = _make_xy(100)

    splitter = DatasetSplitter(train_size=0.7, val_size=0.15)
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(X, y)

    assert len(X_train) == 70
    assert len(X_val) == 15
    assert len(X_test) == 15

    assert len(X_train) + len(X_val) + len(X_test) == len(X)
    assert len(y_train) + len(y_val) + len(y_test) == len(y)

    assert X_train["a"].max() < X_val["a"].min()
    assert X_val["a"].max() < X_test["a"].min()


def test_split_keeps_x_and_y_aligned():
    X, y = _make_xy(50)

    splitter = DatasetSplitter(train_size=0.6, val_size=0.2)
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(X, y)

    np.testing.assert_array_equal(X_train["a"].values, y_train.values)
    np.testing.assert_array_equal(X_val["a"].values, y_val.values)
    np.testing.assert_array_equal(X_test["a"].values, y_test.values)


def test_split_respects_custom_fractions():
    X, y = _make_xy(20)

    splitter = DatasetSplitter(train_size=0.5, val_size=0.25)
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(X, y)

    assert len(X_train) == 10
    assert len(X_val) == 5
    assert len(X_test) == 5


def test_split_accepts_plain_ndarrays():
    X = np.arange(100 * 3 * 2).reshape(100, 3, 2)  # e.g. windowed (samples, timesteps, features)
    y = np.arange(100 * 4).reshape(100, 4)

    splitter = DatasetSplitter(train_size=0.7, val_size=0.15)
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(X, y)

    assert isinstance(X_train, np.ndarray)
    assert X_train.shape == (70, 3, 2)
    assert X_val.shape == (15, 3, 2)
    assert X_test.shape == (15, 3, 2)

    assert y_train.shape == (70, 4)
    assert y_val.shape == (15, 4)
    assert y_test.shape == (15, 4)

    # chronological, non-overlapping
    np.testing.assert_array_equal(X_train, X[:70])
    np.testing.assert_array_equal(X_val, X[70:85])
    np.testing.assert_array_equal(X_test, X[85:])
