"""
Utility functions for evaluating forecasting models.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_metrics(y_true, y_pred):
    """
    Compute MAE and RMSE.

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    tuple[float, float]
        (MAE, RMSE)
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def safe_metrics(y_true, y_pred):
    """
    Compute metrics while ignoring NaN and infinite predictions.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    y_pred = np.where(np.isfinite(y_pred), y_pred, np.nan)

    mask = np.isfinite(y_pred)

    mae = mean_absolute_error(y_true[mask], y_pred[mask])
    rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))

    return mae, rmse


def check_array(name, arr):
    """
    Print simple diagnostics for a prediction array.
    """
    arr = np.asarray(arr)

    print(name)
    print("NaN:", np.isnan(arr).sum())
    print("Inf:", np.isinf(arr).sum())
    print("Max:", np.nanmax(arr))
    print("Min:", np.nanmin(arr))
    print()


def cut_at_first_invalid(y_true, time, *predictions):
    """
    Truncate predictions and targets at the first invalid prediction.

    Parameters
    ----------
    y_true : array-like
    time : pandas.Series
        Time axis.
    *predictions : array-like
        Prediction arrays.

    Returns
    -------
    tuple
        (time_cut, y_true_cut, predictions_cut)
    """
    y_true = np.asarray(y_true)

    valid_mask = np.isfinite(predictions[0])

    first_bad = np.where(~valid_mask)[0]
    end = first_bad[0] if len(first_bad) else len(y_true)

    return (
        time.iloc[:end],
        y_true[:end],
        [pred[:end] for pred in predictions],
    )