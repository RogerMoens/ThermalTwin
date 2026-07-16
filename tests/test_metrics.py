import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils.metrics import check_array, compute_metrics, cut_at_first_invalid, safe_metrics


def test_compute_metrics_matches_sklearn():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.5, 2.0, 2.5, 5.0])

    mae, rmse = compute_metrics(y_true, y_pred)

    assert mae == pytest.approx(mean_absolute_error(y_true, y_pred))
    assert rmse == pytest.approx(np.sqrt(mean_squared_error(y_true, y_pred)))


def test_compute_metrics_perfect_prediction_is_zero():
    y = np.array([1.0, 2.0, 3.0])

    mae, rmse = compute_metrics(y, y)

    assert mae == 0.0
    assert rmse == 0.0


def test_safe_metrics_masks_non_finite_predictions():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, np.nan, np.inf, 4.5])

    mae, rmse = safe_metrics(y_true, y_pred)

    valid = [0, 3]
    assert mae == pytest.approx(mean_absolute_error(y_true[valid], y_pred[valid]))
    assert rmse == pytest.approx(np.sqrt(mean_squared_error(y_true[valid], y_pred[valid])))


def test_check_array_reports_nan_and_inf_counts(capsys):
    arr = np.array([1.0, np.nan, np.inf, -np.inf, 2.0])

    check_array("sample", arr)

    captured = capsys.readouterr()
    assert "NaN: 1" in captured.out
    assert "Inf: 2" in captured.out


def test_cut_at_first_invalid_truncates_at_first_bad_prediction():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    time = pd.Series(pd.date_range("2026-01-01", periods=5, freq="5min"))
    predictions = np.array([1.1, 2.1, np.nan, 4.1, 5.1])

    time_cut, y_true_cut, preds_cut = cut_at_first_invalid(y_true, time, predictions)

    assert len(time_cut) == 2
    np.testing.assert_array_equal(y_true_cut, y_true[:2])
    np.testing.assert_array_equal(preds_cut[0], predictions[:2])


def test_cut_at_first_invalid_keeps_everything_when_all_valid():
    y_true = np.array([1.0, 2.0, 3.0])
    time = pd.Series(pd.date_range("2026-01-01", periods=3, freq="5min"))
    predictions = np.array([1.1, 2.1, 3.1])

    time_cut, y_true_cut, preds_cut = cut_at_first_invalid(y_true, time, predictions)

    assert len(time_cut) == 3
    np.testing.assert_array_equal(y_true_cut, y_true)
    np.testing.assert_array_equal(preds_cut[0], predictions)
