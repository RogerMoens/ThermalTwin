import numpy as np
import pandas as pd
import pytest

from utils.windowing import make_windows


def _make_df(n):
    return pd.DataFrame({
        "feat_a": np.arange(n, dtype=float),
        "feat_b": np.arange(n, dtype=float) * 10,
        "target": np.arange(n, dtype=float) * 100,
    })


def test_make_windows_shapes():
    df = _make_df(20)

    X, y = make_windows(
        df,
        feature_columns=["feat_a", "feat_b"],
        target_column="target",
        lookback=5,
        horizon=3,
    )

    n_expected = 20 - 5 - 3 + 1
    assert X.shape == (n_expected, 5, 2)
    assert y.shape == (n_expected, 3)


def test_make_windows_content_is_correctly_aligned():
    df = _make_df(10)

    X, y = make_windows(
        df,
        feature_columns=["feat_a", "feat_b"],
        target_column="target",
        lookback=3,
        horizon=2,
    )

    # window 0 covers rows [0, 1, 2] of feat_a/feat_b, target rows [3, 4]
    np.testing.assert_array_equal(X[0, :, 0], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(X[0, :, 1], [0.0, 10.0, 20.0])
    np.testing.assert_array_equal(y[0], [300.0, 400.0])

    # window 1 shifts everything forward by one row
    np.testing.assert_array_equal(X[1, :, 0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(y[1], [400.0, 500.0])


def test_make_windows_raises_when_not_enough_rows():
    df = _make_df(5)

    with pytest.raises(ValueError):
        make_windows(
            df,
            feature_columns=["feat_a"],
            target_column="target",
            lookback=3,
            horizon=5,
        )
