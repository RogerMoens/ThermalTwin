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

    DT = 300.0  # 5-minute timestep

    def __init__(
        self,
        initial_guess=None,
        bounds=None,
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

        self.bounds = bounds

        self.theta_ = None

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------

    @staticmethod
    def _simulate(theta, T_out, T_air0):

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

            T_wall[k + 1] = T_wall[k] + Thermal2R2CPredictor.DT * dT_wall
            T_air[k + 1] = T_air[k] + Thermal2R2CPredictor.DT * dT_air

        return T_air

    @staticmethod
    def _objective(theta, T_out, T_air):

        prediction = Thermal2R2CPredictor._simulate(
            theta,
            T_out,
            T_air[0],
        )

        return np.mean((prediction - T_air) ** 2)

    # ------------------------------------------------------------------
    # BasePredictor API
    # ------------------------------------------------------------------

    def fit(self, X_train, y_train, X_val=None, y_val=None):

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
            args=(T_out, T_air),
            method="L-BFGS-B",
            bounds=self.bounds,
        )

        self.theta_ = result.x

        return self

    def predict(self, X):

        if self.theta_ is None:
            raise RuntimeError("Model has not been fitted.")

        if hasattr(X, "columns"):
            T_out = X["temp_out"].values
        else:
            T_out = np.asarray(X).reshape(-1)

        # prediction requires an initial indoor temperature
        if hasattr(X, "columns") and "temp_in" in X.columns:
            T0 = X["temp_in"].iloc[0]
        else:
            raise ValueError(
                "Prediction requires the first indoor temperature "
                "(column 'temp_in')."
            )

        return self._simulate(
            self.theta_,
            T_out,
            T0,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, folder):

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "theta": self.theta_,
                "initial_guess": self.initial_guess,
                "bounds": self.bounds,
            },
            folder / "thermal2r2c.pkl",
        )

    @classmethod
    def load(cls, folder):

        folder = Path(folder)

        data = joblib.load(folder / "thermal2r2c.pkl")

        obj = cls(
            initial_guess=data["initial_guess"],
            bounds=data["bounds"],
        )

        obj.theta_ = data["theta"]

        return obj