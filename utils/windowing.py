"""Utilities for building windowed (samples, timesteps, features) datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_windows(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sliding windows of ``feature_columns`` and a multi-step-ahead
    target from ``target_column``.

    Used by the ``*_weather_only`` model family, which expects a 3D
    ``(samples, timesteps, features)`` input rather than the flat,
    lag-column layout ``FeatureEngineer`` produces for the ``*_one_step``
    models.

    Parameters
    ----------
    df : pandas.DataFrame
        Must already be sorted chronologically.
    feature_columns : list[str]
        Columns used as the per-timestep feature vector.
    target_column : str
        Column to forecast.
    lookback : int
        Number of past timesteps included in each window.
    horizon : int
        Number of future timesteps to predict.

    Returns
    -------
    X : ndarray, shape (samples, lookback, len(feature_columns))
    y : ndarray, shape (samples, horizon)
    """

    features = df[feature_columns].to_numpy()
    target = df[target_column].to_numpy()

    n_samples = len(df) - lookback - horizon + 1

    if n_samples <= 0:
        raise ValueError(
            "Not enough rows to build a single window: need at least "
            f"lookback + horizon = {lookback + horizon} rows, got {len(df)}."
        )

    X = np.stack([features[i : i + lookback] for i in range(n_samples)])
    y = np.stack([target[i + lookback : i + lookback + horizon] for i in range(n_samples)])

    return X, y
