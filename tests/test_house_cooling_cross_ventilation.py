import numpy as np
import pandas as pd
import pytest

from models.house_cooling_cross_ventilation import HouseCoolingCrossVentilationPredictor


def _make_model(**overrides):
    params = dict(
        floor_area_m2=57,
        ceiling_height_m=2.6,
        window_area_side_a_m2=1030 * 700 / 1000**2,
        window_area_side_b_m2=1400 * 700 / 1000**2 + 1400 * 1100 / 1000**2,
        discharge_coefficient=0.015,
        thermal_mass_factor=8,
        fan_efficiency=0.65,
    )
    params.update(overrides)

    return HouseCoolingCrossVentilationPredictor(**params)


def test_house_volume():
    model = _make_model()

    assert model.house_volume_m3 == pytest.approx(57 * 2.6)


def test_effective_window_area_is_smaller_side():
    model = _make_model()

    side_a = 1030 * 700 / 1000**2
    side_b = 1400 * 700 / 1000**2 + 1400 * 1100 / 1000**2

    assert model.effective_window_area_m2 == pytest.approx(min(side_a, side_b))
    assert model.effective_window_area_m2 == pytest.approx(side_a)


def test_natural_flow_scales_with_wind_speed():
    model = _make_model()

    flow_at_3 = model.natural_flow_m3_h(3)
    flow_at_6 = model.natural_flow_m3_h(6)

    assert flow_at_6 == pytest.approx(2 * flow_at_3)


def test_air_changes_per_hour_includes_fan_contribution():
    model = _make_model()

    ach_no_fan = model.air_changes_per_hour(wind_speed_m_s=3, fan_capacity_m3_h=0)
    ach_with_fan = model.air_changes_per_hour(wind_speed_m_s=3, fan_capacity_m3_h=1000)

    assert ach_with_fan > ach_no_fan

    expected_flow = model.natural_flow_m3_h(3) + 1000 * model.fan_efficiency
    assert ach_with_fan == pytest.approx(expected_flow / model.house_volume_m3)


def test_predict_before_fit_raises():
    model = _make_model()

    with pytest.raises(RuntimeError):
        model.predict(np.array([20.0, 20.0]), T0=24.0, wind_speed_m_s=3)


def test_simulate_cools_toward_outdoor_temperature():
    model = _make_model()
    model.fit()

    temps = model.simulate_constant_conditions(
        outside_temp_C=20,
        T0=24,
        wind_speed_m_s=3,
        fan_capacity_m3_h=0,
        duration_hours=6,
    )

    # monotonically cooling toward, but never below, the outdoor temperature
    assert np.all(np.diff(temps) <= 0)
    assert temps[-1] < temps[0]
    assert temps[-1] >= 20


def test_more_ventilation_cools_faster():
    model = _make_model()
    model.fit()

    no_fan = model.simulate_constant_conditions(
        outside_temp_C=20,
        T0=24,
        wind_speed_m_s=3,
        fan_capacity_m3_h=0,
        duration_hours=6,
    )

    with_fan = model.simulate_constant_conditions(
        outside_temp_C=20,
        T0=24,
        wind_speed_m_s=3,
        fan_capacity_m3_h=3000,
        duration_hours=6,
    )

    assert with_fan[-1] < no_fan[-1]


def test_predict_with_dataframe_infers_T0_and_wind_speed():
    model = _make_model()
    model.fit()

    X = pd.DataFrame(
        {
            "temp_out": np.full(10, 20.0),
            "temp_in": np.full(10, 24.0),
            "wind_speed": np.full(10, 3.0),
        }
    )

    preds = model.predict(X)
    expected = model.simulate(X["temp_out"].values, 24.0, X["wind_speed"].values)

    assert preds.shape == (10,)
    np.testing.assert_allclose(preds, expected)


def test_predict_dataframe_without_wind_speed_requires_explicit_value():
    model = _make_model()
    model.fit()

    X = pd.DataFrame({"temp_out": np.full(5, 20.0), "temp_in": np.full(5, 24.0)})

    with pytest.raises(ValueError):
        model.predict(X)

    preds = model.predict(X, wind_speed_m_s=3)
    assert preds.shape == (5,)


def test_compare_fan_capacities_returns_expected_keys():
    model = _make_model()
    model.fit()

    results = model.compare_fan_capacities(
        outside_temp_C=20,
        T0=24,
        wind_speed_m_s=3,
        fan_capacities_m3_h=[0, 1000, 3000],
        duration_hours=2,
    )

    assert set(results.keys()) == {0, 1000, 3000}

    for fan_capacity, result in results.items():
        assert "temps" in result
        assert "ach" in result
        assert "flow_m3_h" in result

    # larger fan capacity -> more cooling over the same duration
    assert results[3000]["temps"][-1] < results[0]["temps"][-1]


def test_time_to_reach_temperature():
    model = _make_model()
    model.fit()

    temps = model.simulate_constant_conditions(
        outside_temp_C=20,
        T0=24,
        wind_speed_m_s=3,
        fan_capacity_m3_h=3000,
        duration_hours=6,
    )

    time_h = HouseCoolingCrossVentilationPredictor.time_to_reach_temperature(
        temps, target_temp_C=21, dt=model.dt
    )

    assert time_h == time_h  # not NaN
    assert 0 <= time_h <= 6


def test_time_to_reach_temperature_returns_nan_when_never_reached():
    model = _make_model()
    model.fit()

    temps = model.simulate_constant_conditions(
        outside_temp_C=20,
        T0=20.1,
        wind_speed_m_s=0,
        fan_capacity_m3_h=0,
        duration_hours=1,
    )

    time_h = HouseCoolingCrossVentilationPredictor.time_to_reach_temperature(
        temps, target_temp_C=-100, dt=model.dt
    )

    assert np.isnan(time_h)


def test_save_and_load_roundtrip(tmp_path):
    model = _make_model()
    model.fit()

    model.save(tmp_path)
    loaded = HouseCoolingCrossVentilationPredictor.load(tmp_path)

    assert loaded.fitted_ is True
    assert loaded.floor_area_m2 == model.floor_area_m2
    assert loaded.thermal_mass_factor == model.thermal_mass_factor

    expected = model.simulate_constant_conditions(20, 24, 3, fan_capacity_m3_h=1000, duration_hours=2)
    actual = loaded.simulate_constant_conditions(20, 24, 3, fan_capacity_m3_h=1000, duration_hours=2)

    np.testing.assert_allclose(actual, expected)
