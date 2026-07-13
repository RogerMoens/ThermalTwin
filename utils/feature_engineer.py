from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngineer:
    """
    Feature engineering for Netatmo weather station data.

    This class transforms a raw dataframe into a machine-learning
    ready dataset by:

        - Creating cyclical time features
        - Creating indoor temperature lag features
        - Creating outdoor temperature lag features
        - Selecting feature columns
        - Creating feature matrix (X) and target vector (y)

    Parameters
    ----------
    indoor_lags : int
        Number of indoor temperature lag features.

    outdoor_lags : int
        Number of outdoor temperature lag features.

    target : str
        Target column to predict.
    """

    def __init__(
        self,
        indoor_lags: int = 48,
        outdoor_lags: int = 48,
        target: str = "temp_in",
    ):

        self.indoor_lags = indoor_lags
        self.outdoor_lags = outdoor_lags
        self.target = target

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform all feature engineering steps.

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
        """

        df = df.copy()

        df = self.create_time_features(df)
        df = self.create_lag_features(df)

        return df

    def create_dataset(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Create feature matrix X and target vector y.

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------
        X : pandas.DataFrame

        y : pandas.Series
        """

        X = df[self.feature_columns]
        y = df[self.target]

        return X, y

    # ------------------------------------------------------------------
    # Feature creation
    # ------------------------------------------------------------------

    def create_time_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create cyclical time features.
        """

        df = df.copy()

        df["hour"] = df["datetime"].dt.hour
        df["minute"] = df["datetime"].dt.minute

        df["hour_sin"] = np.sin(
            2 * np.pi * df["hour"] / 24
        )

        df["hour_cos"] = np.cos(
            2 * np.pi * df["hour"] / 24
        )

        df["dayofyear"] = df["datetime"].dt.dayofyear

        df["day_sin"] = np.sin(
            2 * np.pi * df["dayofyear"] / 365.25
        )

        df["day_cos"] = np.cos(
            2 * np.pi * df["dayofyear"] / 365.25
        )

        return df

    def create_lag_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create lag features for indoor and outdoor temperatures.
        """

        df = df.copy()

        for lag in range(1, self.indoor_lags + 1):
            df[f"temp_in_lag_{lag}"] = (
                df["temp_in"].shift(lag)
            )

        for lag in range(1, self.outdoor_lags + 1):
            df[f"temp_out_lag_{lag}"] = (
                df["temp_out"].shift(lag)
            )

        df = df.dropna().reset_index(drop=True)

        return df

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def feature_columns(self) -> list[str]:
        """
        List of model input features.
        """

        features = [
            "temp_out",
            "hum_out",
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
        ]

        features.extend(
            [
                f"temp_in_lag_{lag}"
                for lag in range(1, self.indoor_lags + 1)
            ]
        )

        features.extend(
            [
                f"temp_out_lag_{lag}"
                for lag in range(1, self.outdoor_lags + 1)
            ]
        )

        return features

    @property
    def target_column(self) -> str:
        """
        Name of prediction target.
        """

        return self.target

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """
        Print feature engineering configuration.
        """

        print("FeatureEngineer")
        print("----------------")
        print(f"Target          : {self.target}")
        print(f"Indoor lags     : {self.indoor_lags}")
        print(f"Outdoor lags    : {self.outdoor_lags}")
        print(f"Total features  : {len(self.feature_columns)}")