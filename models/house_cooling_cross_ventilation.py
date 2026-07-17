"""House cooling via natural cross ventilation.

Estimates the natural airflow driven by wind through two window
openings (cross ventilation), optionally boosted by a mechanical fan,
and simulates the resulting cooling of the indoor air temperature:

    dT/dt = -k * (T - T_out),    k = ACH / thermal_mass_factor

The house geometry and thermal-mass parameters describe the physical
building rather than being fitted to data, so they can be reused by
any of the other cooling-oriented models in this package.
"""

from pathlib import Path

import joblib
import numpy as np

from .base_predictor import BasePredictor


class HouseCoolingCrossVentilationPredictor(BasePredictor):
    """
    Natural cross-ventilation cooling model.

    Parameters
    ----------
    floor_area_m2 : float
        Floor area of the house.
    ceiling_height_m : float
        Ceiling height, used with ``floor_area_m2`` to get house volume.
    window_area_side_a_m2, window_area_side_b_m2 : float
        Net open window area on each of the two ventilation sides.
        Cross-ventilation flow is limited by the smaller of the two.
    discharge_coefficient : float, default 0.015
        Discharge coefficient (Cd) of the window openings. Typical
        textbook values are 0.5-0.7; this default matches the
        empirically-tuned value from the original house measurements.
    thermal_mass_factor : float, default 8.0
        Governs how slowly the house cools for a given air-change
        rate: 1-2 for a light-weight home, 3-5 for an average
        apartment, 5-8 for a heavy-weight home.
    fan_efficiency : float, default 0.65
        Fraction of a fan's nominal m3/h capacity that is realized as
        effective airflow.
    dt : float, default 300.0
        Simulation timestep in seconds (matches Netatmo's native
        5-minute sampling rate).

    Notes
    -----
    Unlike the data-fitted models in this package, the house/window
    geometry and thermal mass factor are physical properties supplied
    at construction time rather than learned parameters, so ``fit()``
    is a no-op kept only to satisfy the shared ``BasePredictor``
    interface.
    """

    DEFAULT_DT = 300.0  # seconds, matches Netatmo's native sampling rate

    def __init__(
        self,
        floor_area_m2,
        ceiling_height_m,
        window_area_side_a_m2,
        window_area_side_b_m2,
        discharge_coefficient=0.015,
        thermal_mass_factor=8.0,
        fan_efficiency=0.65,
        dt=DEFAULT_DT,
    ):
        self.floor_area_m2 = floor_area_m2
        self.ceiling_height_m = ceiling_height_m
        self.window_area_side_a_m2 = window_area_side_a_m2
        self.window_area_side_b_m2 = window_area_side_b_m2
        self.discharge_coefficient = discharge_coefficient
        self.thermal_mass_factor = thermal_mass_factor
        self.fan_efficiency = fan_efficiency
        self.dt = dt

        self.fitted_ = False

    # ------------------------------------------------------------------
    # Geometry / ventilation physics
    # ------------------------------------------------------------------

    @property
    def house_volume_m3(self):
        return self.floor_area_m2 * self.ceiling_height_m

    @property
    def effective_window_area_m2(self):
        return min(self.window_area_side_a_m2, self.window_area_side_b_m2)

    def natural_flow_m3_h(self, wind_speed_m_s):
        """Wind-driven natural airflow through the window openings, in m3/h."""

        flow_m3_s = (
            self.discharge_coefficient
            * self.effective_window_area_m2
            * np.asarray(wind_speed_m_s, dtype=float)
        )

        return flow_m3_s * 3600

    def air_changes_per_hour(self, wind_speed_m_s, fan_capacity_m3_h=0.0):
        """Total ACH from natural ventilation plus an optional fan."""

        effective_fan_flow = np.asarray(fan_capacity_m3_h, dtype=float) * self.fan_efficiency
        total_flow = self.natural_flow_m3_h(wind_speed_m_s) + effective_fan_flow

        return total_flow / self.house_volume_m3

    # ------------------------------------------------------------------
    # BasePredictor API
    # ------------------------------------------------------------------

    def fit(self, X_train=None, y_train=None, X_val=None, y_val=None, **kwargs):
        """No-op: house geometry and thermal mass are physical inputs, not fitted parameters."""

        self.fitted_ = True

        return self

    def predict(self, X, T0=None, wind_speed_m_s=None, fan_capacity_m3_h=0.0):
        """
        Parameters
        ----------
        X : DataFrame or ndarray
            Outdoor temperature. Must contain a 'temp_out' column if a
            DataFrame, otherwise a 1D array of outdoor temperatures
            (mirrors the ndarray handling used by the other models).

        T0 : float, optional
            Initial indoor temperature. If ``X`` is a DataFrame with a
            'temp_in' column, it is inferred from
            ``X['temp_in'].iloc[0]`` when not given explicitly.
            Required when ``X`` is not a DataFrame.

        wind_speed_m_s : float or array-like, optional
            Wind speed driving natural ventilation. If ``X`` is a
            DataFrame with a 'wind_speed' column, it is inferred from
            that column when not given explicitly. Required otherwise.

        fan_capacity_m3_h : float, default 0.0
            Nominal mechanical fan capacity boosting ventilation.
        """

        if not self.fitted_:
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

            if wind_speed_m_s is None:
                if "wind_speed" not in X.columns:
                    raise ValueError(
                        "Prediction requires a wind speed: pass "
                        "wind_speed_m_s explicitly or include a "
                        "'wind_speed' column."
                    )
                wind_speed_m_s = X["wind_speed"].values
        else:
            T_out = np.asarray(X).reshape(-1)

            if T0 is None:
                raise ValueError(
                    "Prediction requires an initial indoor temperature: "
                    "pass T0 explicitly when X is not a DataFrame."
                )

            if wind_speed_m_s is None:
                raise ValueError(
                    "Prediction requires a wind speed: pass "
                    "wind_speed_m_s explicitly when X is not a DataFrame."
                )

        return self.simulate(T_out, T0, wind_speed_m_s, fan_capacity_m3_h)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(self, T_out, T0, wind_speed_m_s, fan_capacity_m3_h=0.0, dt=None):
        """
        Simulate indoor temperature driven by cross ventilation.

        ``T_out`` and ``wind_speed_m_s`` may be scalars (held constant
        for the whole run) or arrays of the same length, one value per
        timestep.
        """

        dt = self.dt if dt is None else dt
        dt_h = dt / 3600

        T_out = np.atleast_1d(np.asarray(T_out, dtype=float))
        n = len(T_out)

        wind_speed = np.broadcast_to(np.asarray(wind_speed_m_s, dtype=float), (n,))
        fan_capacity = np.broadcast_to(np.asarray(fan_capacity_m3_h, dtype=float), (n,))

        ach = self.air_changes_per_hour(wind_speed, fan_capacity)
        k = ach / self.thermal_mass_factor

        T = np.empty(n)
        T[0] = T0

        for i in range(n - 1):
            T[i + 1] = T[i] - k[i] * (T[i] - T_out[i]) * dt_h

        return T

    def simulate_constant_conditions(
        self,
        outside_temp_C,
        T0,
        wind_speed_m_s,
        fan_capacity_m3_h=0.0,
        duration_hours=6.0,
        dt=None,
    ):
        """Convenience wrapper: simulate at a fixed outdoor temperature for a given duration."""

        dt = self.dt if dt is None else dt
        steps = int(duration_hours * 3600 / dt)

        T_out = np.full(steps, float(outside_temp_C))

        return self.simulate(T_out, T0, wind_speed_m_s, fan_capacity_m3_h, dt=dt)

    def compare_fan_capacities(
        self,
        outside_temp_C,
        T0,
        wind_speed_m_s,
        fan_capacities_m3_h,
        duration_hours=6.0,
        dt=None,
    ):
        """
        Simulate cooling under a fixed outdoor temperature for several
        fan capacities, for side-by-side comparison.

        Returns
        -------
        dict mapping each fan capacity to a dict with keys
        ``temps``, ``ach`` and ``flow_m3_h``.
        """

        results = {}

        for fan_capacity in fan_capacities_m3_h:
            temps = self.simulate_constant_conditions(
                outside_temp_C,
                T0,
                wind_speed_m_s,
                fan_capacity_m3_h=fan_capacity,
                duration_hours=duration_hours,
                dt=dt,
            )

            results[fan_capacity] = {
                "temps": temps,
                "ach": float(self.air_changes_per_hour(wind_speed_m_s, fan_capacity)),
                "flow_m3_h": float(
                    self.natural_flow_m3_h(wind_speed_m_s)
                    + fan_capacity * self.fan_efficiency
                ),
            }

        return results

    @staticmethod
    def time_to_reach_temperature(temps, target_temp_C, dt=None):
        """Hours until ``temps`` first reaches ``target_temp_C``, or NaN if it never does."""

        dt = HouseCoolingCrossVentilationPredictor.DEFAULT_DT if dt is None else dt
        dt_h = dt / 3600

        for i, T in enumerate(temps):
            if T <= target_temp_C:
                return i * dt_h

        return float("nan")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, folder):
        folder = self._prepare_folder(folder)

        joblib.dump(
            {
                "floor_area_m2": self.floor_area_m2,
                "ceiling_height_m": self.ceiling_height_m,
                "window_area_side_a_m2": self.window_area_side_a_m2,
                "window_area_side_b_m2": self.window_area_side_b_m2,
                "discharge_coefficient": self.discharge_coefficient,
                "thermal_mass_factor": self.thermal_mass_factor,
                "fan_efficiency": self.fan_efficiency,
                "dt": self.dt,
                "fitted_": self.fitted_,
            },
            folder / "house_cooling_cross_ventilation.pkl",
        )

    @classmethod
    def load(cls, folder):
        folder = Path(folder)

        data = joblib.load(folder / "house_cooling_cross_ventilation.pkl")

        obj = cls.__new__(cls)

        obj.floor_area_m2 = data["floor_area_m2"]
        obj.ceiling_height_m = data["ceiling_height_m"]
        obj.window_area_side_a_m2 = data["window_area_side_a_m2"]
        obj.window_area_side_b_m2 = data["window_area_side_b_m2"]
        obj.discharge_coefficient = data["discharge_coefficient"]
        obj.thermal_mass_factor = data["thermal_mass_factor"]
        obj.fan_efficiency = data["fan_efficiency"]
        obj.dt = data.get("dt", cls.DEFAULT_DT)
        obj.fitted_ = data["fitted_"]

        return obj
