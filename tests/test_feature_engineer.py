import numpy as np
import pandas as pd

from utils.feature_engineer import FeatureEngineer


def _make_df(n=10):
    dates = pd.date_range("2026-01-01", periods=n, freq="5min")
    return pd.DataFrame({
        "datetime": dates,
        "temp_in": np.arange(n, dtype=float) + 20.0,
        "temp_out": np.arange(n, dtype=float) + 5.0,
        "hum_out": np.full(n, 80.0),
    })


def test_create_time_features():
    df = _make_df(3)
    df.loc[0, "datetime"] = pd.Timestamp("2026-01-01 00:00:00")
    df.loc[1, "datetime"] = pd.Timestamp("2026-01-01 06:00:00")
    df.loc[2, "datetime"] = pd.Timestamp("2026-01-01 12:00:00")

    fe = FeatureEngineer()
    out = fe.create_time_features(df)

    assert list(out["hour"]) == [0, 6, 12]

    np.testing.assert_allclose(out["hour_sin"].iloc[0], 0.0, atol=1e-9)
    np.testing.assert_allclose(out["hour_cos"].iloc[0], 1.0, atol=1e-9)

    np.testing.assert_allclose(out["hour_sin"].iloc[1], 1.0, atol=1e-9)
    np.testing.assert_allclose(out["hour_cos"].iloc[1], 0.0, atol=1e-9)


def test_create_lag_features_drops_leading_rows():
    df = _make_df(10)

    fe = FeatureEngineer(indoor_lags=2, outdoor_lags=3)
    out = fe.create_lag_features(df)

    # rows without a full lag history (max(indoor_lags, outdoor_lags)) are dropped
    assert len(out) == len(df) - 3

    first_kept_original_index = 3
    assert out.iloc[0]["temp_in_lag_1"] == df.iloc[first_kept_original_index - 1]["temp_in"]
    assert out.iloc[0]["temp_out_lag_3"] == df.iloc[first_kept_original_index - 3]["temp_out"]


def test_create_target_shifts_and_drops_trailing_rows():
    df = _make_df(5)

    fe = FeatureEngineer(target="temp_in", target_shift=2)
    out = fe.create_target(df)

    assert len(out) == len(df) - 2
    np.testing.assert_allclose(out["temp_in"].values, df["temp_in"].values[2:])


def test_transform_pipeline_has_no_missing_values():
    df = _make_df(20)

    fe = FeatureEngineer(indoor_lags=3, outdoor_lags=3, target_shift=1)
    out = fe.transform(df)

    assert len(out) == 20 - 3 - 1
    assert not out.isna().any().any()

    for lag in range(1, 4):
        assert f"temp_in_lag_{lag}" in out.columns
        assert f"temp_out_lag_{lag}" in out.columns


def test_create_dataset_matches_feature_columns():
    df = _make_df(20)

    fe = FeatureEngineer(indoor_lags=2, outdoor_lags=2, target_shift=1)
    transformed = fe.transform(df)
    X, y = fe.create_dataset(transformed)

    assert list(X.columns) == fe.feature_columns
    assert y.name == fe.target_column
    assert len(X) == len(y) == len(transformed)


def test_feature_columns_length_and_target_column():
    fe = FeatureEngineer(indoor_lags=5, outdoor_lags=7)

    assert len(fe.feature_columns) == 6 + 5 + 7
    assert fe.target_column == "temp_in"
