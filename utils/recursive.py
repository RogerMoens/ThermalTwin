"""
Recursive forecasting utilities.
"""

from __future__ import annotations

import numpy as np

from .metrics import safe_metrics


def recursive_predict(
    model,
    X_data,
    scaler,
    feature_columns,
    indoor_lags,
):
    """
    Perform closed-loop recursive prediction.
    """

    indoor_history = [
        X_data.iloc[0][f"temp_in_lag_{lag}"]
        for lag in range(1, indoor_lags + 1)
    ]

    predictions = []

    for i in range(len(X_data)):

        row = X_data.iloc[i].copy()

        for lag in range(1, indoor_lags + 1):
            row[f"temp_in_lag_{lag}"] = indoor_history[lag - 1]

        x = row[feature_columns].to_frame().T
        x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        x_scaled = scaler.transform(x)

        if hasattr(model, "layers"):
            pred = model.predict(x_scaled, verbose=0)[0, 0]
        else:
            pred = model.predict(x_scaled)[0]

        predictions.append(pred)

        indoor_history = [pred] + indoor_history[:-1]

    return np.asarray(predictions)


def predict_one_day_recursive(
    model,
    X_day,
    y_day,
    scaler,
    feature_columns,
    indoor_lags,
):
    """
    Run recursive prediction for a single day.
    """

    indoor_history = [
        X_day.iloc[0][f"temp_in_lag_{lag}"]
        for lag in range(1, indoor_lags + 1)
    ]

    predictions = []

    for i in range(len(X_day)):

        row = X_day.iloc[i].copy()

        for lag in range(1, indoor_lags + 1):
            row[f"temp_in_lag_{lag}"] = indoor_history[lag - 1]

        x = row[feature_columns].to_frame().T
        x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        x_scaled = scaler.transform(x)

        if hasattr(model, "layers"):
            pred = model.predict(x_scaled, verbose=0)[0, 0]
        else:
            pred = model.predict(x_scaled)[0]

        pred = np.clip(pred, -10, 50)

        predictions.append(pred)

        indoor_history = [pred] + indoor_history[:-1]

    predictions = np.asarray(predictions)

    mae, rmse = safe_metrics(y_day, predictions)

    valid_ratio = (
        np.isfinite(predictions) &
        np.isfinite(np.asarray(y_day))
    ).mean()

    return mae, rmse, valid_ratio, predictions