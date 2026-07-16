from pathlib import Path

import joblib
import numpy as np

from scipy.optimize import minimize

from .base_predictor import BasePredictor


class Thermal2R2CPredictor(BasePredictor):
    """
    Physics-based 2R2C thermal model.

    Parameters to estimate
    ----------------------
    R_ow : Outside-wall thermal resistance
    R_iw : Wall-indoor thermal resistance
    C_w  : Wall thermal capacitance
    C_a  : Indoor air thermal capacitance
    T_wall0 : Initial wall temperature
    """

    DEFAULT_DT = 300.0  # 5-minute timestep, matches Netatmo's native sampling rate

    # Physical constants must be positive; T_wall0 is a free state variable.
    DEFAULT_BOUNDS = [
        (1e-6, None),  # R_ow > 0
        (1e-6, None),  # R_iw > 0
        (1e-6, None),  # C_w > 0
        (1e-6, None),  # C_a > 0
        (None, None),  # T_wall0
    ]

    def __init__(
        self,
        initial_guess=None,
        bounds="auto",
        dt=DEFAULT_DT,
    ):

        self.initial_guess = (
            initial_guess
            if initial_guess is not None
            else [
                2.0,
                0.5,
                5e5,
                5e4,
                20.0,
            ]
        )

        self.bounds = self.DEFAULT_BOUNDS if bounds == "auto" else bounds
        self.dt = dt

        self.theta_ = None

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    @staticmethod
    def _simulate(theta, T_out, T_air0, dt=DEFAULT_DT):

        R_ow, R_iw, C_w, C_a, T_wall0 = theta

        n = len(T_out)

        T_air = np.zeros(n)
        T_wall = np.zeros(n)

        T_air[0] = T_air0
        T_wall[0] = T_wall0

        for k in range(n - 1):

            q_out = (T_out[k] - T_wall[k]) / R_ow
            q_wall = (T_wall[k] - T_air[k]) / R_iw

            dT_wall = (q_out - q_wall) / C_w
            dT_air = q_wall / C_a

            T_wall[k + 1] = T_wall[k] + dt * dT_wall
            T_air[k + 1] = T_air[k] + dt * dT_air

        return T_air

    @staticmethod
    def _objective(theta, T_out, T_air, dt=DEFAULT_DT):

        prediction = Thermal2R2CPredictor._simulate(
            theta,
            T_out,
            T_air[0],
            dt=dt,
        )

        return np.mean((prediction - T_air) ** 2)

    # ------------------------------------------------------------------
    # BasePredictor API
    # ------------------------------------------------------------------

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):

        """
        Parameters
        ----------
        X_train : DataFrame or ndarray

            Must contain outdoor temperature as first column or a column
            named 'temp_out'.

        y_train : indoor temperature
        """

        if hasattr(X_train, "columns"):
            T_out = X_train["temp_out"].values
        else:
            T_out = np.asarray(X_train).reshape(-1)

        T_air = np.asarray(y_train)

        guess = self.initial_guess.copy()

        # initialize wall temperature using first indoor temperature
        guess[-1] = T_air[0]

        result = minimize(
            self._objective,
            guess,
            args=(T_out, T_air, self.dt),
            method="L-BFGS-B",
            bounds=self.bounds,
        )

        self.theta_ = result.x

        return self

    def predict(self, X, T0=None):

        """
        Parameters
        ----------
        X : DataFrame or ndarray
            Outdoor temperature. Must contain a 'temp_out' column if a
            DataFrame, otherwise a 1D array of outdoor temperatures
            (mirrors the ndarray handling in ``fit``).

        T0 : float, optional
            Initial indoor temperature used to seed the simulation. If
            ``X`` is a DataFrame with a 'temp_in' column, it is inferred
            from ``X['temp_in'].iloc[0]`` when not given explicitly.
            Required when ``X`` is not a DataFrame.
        """

        if self.theta_ is None:
            raise RuntimeError("Model has not been fitted.")

        if hasattr(X, "columns"):
            T_out = X["temp_out"].values

            if T0 is None:
                if "temp_in" not in X.columns:
                    raise ValueError(
                        "Prediction requires an initial indoor temperature: "
                        "pass T0 explicitly or include a 'temp_in' column."
                    )
                T0 = X["temp_in"].iloc[0]
        else:
            T_out = np.asarray(X).reshape(-1)

            if T0 is None:
                raise ValueError(
                    "Prediction requires an initial indoor temperature: "
                    "pass T0 explicitly when X is not a DataFrame."
                )

        return self._simulate(
            self.theta_,
            T_out,
            T0,
            dt=self.dt,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, folder):

        folder = self._prepare_folder(folder)

        joblib.dump(
            {
                "theta": self.theta_,
                "initial_guess": self.initial_guess,
                "bounds": self.bounds,
                "dt": self.dt,
            },
            folder / "thermal2r2c.pkl",
        )

    @classmethod
    def load(cls, folder):

        folder = Path(folder)

        data = joblib.load(folder / "thermal2r2c.pkl")

        obj = cls.__new__(cls)

        obj.initial_guess = data["initial_guess"]
        obj.bounds = data["bounds"]
        obj.dt = data.get("dt", cls.DEFAULT_DT)
        obj.theta_ = data["theta"]

        return obj
