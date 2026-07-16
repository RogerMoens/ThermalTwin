import pandas as pd
import pytest

from utils.data_loader import NetatmoDataLoader


WEATHER_CSV_CONTENT = (
    "Netatmo export\n"
    "generated 2026-01-01\n"
    "station: Home\n"
    "module: Indoor\n"
    "unit: metric\n"
    "Timezone : Europe/Amsterdam,Temperature °C,Humidity %,Temperature °C,Humidity %\n"
    "2026-01-01 00:05:00,20.6,45,5.1,81\n"
    "2026-01-01 00:00:00,20.5,46,5.2,80\n"
    "2026-01-01 00:10:00,20.7,44,5.0,82\n"
)


def _write_station(root, name):
    station_dir = root / name
    station_dir.mkdir(parents=True, exist_ok=True)
    (station_dir / "Weather.csv").write_text(WEATHER_CSV_CONTENT, encoding="utf-8")
    return station_dir


def test_load_renames_columns_and_sorts_chronologically(tmp_path):
    station_dir = _write_station(tmp_path, "Home")

    loader = NetatmoDataLoader(tmp_path)
    df = loader.load(station_dir / "Weather.csv")

    assert list(df.columns) == ["datetime", "temp_in", "hum_in", "temp_out", "hum_out"]
    assert df["datetime"].is_monotonic_increasing

    first = df.iloc[0]
    assert first["datetime"] == pd.Timestamp("2026-01-01 00:00:00")
    assert first["temp_in"] == 20.5
    assert first["hum_in"] == 46
    assert first["temp_out"] == 5.2
    assert first["hum_out"] == 80


def test_list_weather_files_and_station_names(tmp_path):
    _write_station(tmp_path, "Home")
    _write_station(tmp_path, "Office")

    loader = NetatmoDataLoader(tmp_path)

    files = loader.list_weather_files()
    assert [f.parent.name for f in files] == ["Home", "Office"]

    assert loader.station_names() == ["Home", "Office"]


def test_load_station(tmp_path):
    _write_station(tmp_path, "Home")

    loader = NetatmoDataLoader(tmp_path)
    df = loader.load_station("Home")

    assert len(df) == 3


def test_load_station_missing_raises(tmp_path):
    loader = NetatmoDataLoader(tmp_path)

    with pytest.raises(FileNotFoundError):
        loader.load_station("Nonexistent")


def test_load_by_index(tmp_path):
    _write_station(tmp_path, "Home")
    _write_station(tmp_path, "Office")

    loader = NetatmoDataLoader(tmp_path)

    df = loader.load_by_index(1)
    assert len(df) == 3


def test_load_by_index_out_of_range_raises(tmp_path):
    _write_station(tmp_path, "Home")

    loader = NetatmoDataLoader(tmp_path)

    with pytest.raises(IndexError):
        loader.load_by_index(5)
