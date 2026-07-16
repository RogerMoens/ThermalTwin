import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils.recursive import predict_one_day_recursive, recursive_predict


class _IdentityScaler:
    """Pass-through stand-in for a fitted StandardScaler."""

    def transform(self, X):
        return X.values


class _SumModel:
    """Sklearn-style stub model: predicts the row-wise feature sum."""

    def predict(self, X):
        return np.asarray(X).sum(axis=1)


class _SumKerasLikeModel:
    """Keras-style stub model exposing a `.layers` attribute."""

    layers = ["dummy"]

    def predict(self, X, verbose=0):
        return np.asarray(X).sum(axis=1, keepdims=True)


FEATURE_COLUMNS = ["temp_out", "temp_in_lag_1", "temp_in_lag_2"]


def _make_x_data():
    return pd.DataFrame({
        "temp_out": [10.0, 11.0, 12.0],
        "temp_in_lag_1": [20.0, 0.0, 0.0],  # rows 1-2 are overwritten every step
        "temp_in_lag_2": [19.5, 0.0, 0.0],
    })


def test_recursive_predict_feeds_predictions_back_as_lags():
    X_data = _make_x_data()

    preds = recursive_predict(
        model=_SumModel(),
        X_data=X_data,
        scaler=_IdentityScaler(),
        feature_columns=FEATURE_COLUMNS,
        indoor_lags=2,
    )

    # step 0: 10 + 20.0 + 19.5 = 49.5
    # step 1: 11 + 49.5 + 20.0 = 80.5
    # step 2: 12 + 80.5 + 49.5 = 142.0
    np.testing.assert_allclose(preds, [49.5, 80.5, 142.0])


def test_recursive_predict_keras_and_sklearn_style_models_agree():
    X_data = _make_x_data()

    sklearn_preds = recursive_predict(
        _SumModel(), X_data, _IdentityScaler(), FEATURE_COLUMNS, indoor_lags=2,
    )
    keras_preds = recursive_predict(
        _SumKerasLikeModel(), X_data, _IdentityScaler(), FEATURE_COLUMNS, indoor_lags=2,
    )

    np.testing.assert_allclose(sklearn_preds, keras_preds)


def test_predict_one_day_recursive_clips_predictions():
    X_data = _make_x_data()
    y_day = pd.Series([50.0, 50.0, 55.0])

    mae, rmse, valid_ratio, preds = predict_one_day_recursive(
        model=_SumModel(),
        X_day=X_data,
        y_day=y_day,
        scaler=_IdentityScaler(),
        feature_columns=FEATURE_COLUMNS,
        indoor_lags=2,
    )

    # same trajectory as the unclipped test, but clamped to [-10, 50]:
    # step 0: 49.5 (unclipped)
    # step 1: 80.5 -> clipped to 50.0, which then feeds back into step 2
    # step 2: 12 + 50.0 + 49.5 = 111.5 -> clipped to 50.0
    expected_preds = np.array([49.5, 50.0, 50.0])
    np.testing.assert_allclose(preds, expected_preds)

    assert valid_ratio == 1.0
    assert mae == pytest.approx(mean_absolute_error(y_day, expected_preds))
    assert rmse == pytest.approx(np.sqrt(mean_squared_error(y_day, expected_preds)))


def test_predict_one_day_recursive_valid_ratio_drops_with_nan_predictions():
    X_data = _make_x_data()
    y_day = pd.Series([50.0, 50.0, 50.0])

    class _FlakyModel:
        def __init__(self):
            self.calls = 0

        def predict(self, X):
            self.calls += 1
            if self.calls == 3:
                return np.array([np.nan])
            return np.asarray(X).sum(axis=1)

    mae, rmse, valid_ratio, preds = predict_one_day_recursive(
        model=_FlakyModel(),
        X_day=X_data,
        y_day=y_day,
        scaler=_IdentityScaler(),
        feature_columns=FEATURE_COLUMNS,
        indoor_lags=2,
    )

    assert np.isnan(preds[2])
    assert valid_ratio == pytest.approx(2 / 3)

    valid_mask = np.isfinite(preds)
    expected_mae = mean_absolute_error(y_day[valid_mask], preds[valid_mask])
    assert mae == pytest.approx(expected_mae)
