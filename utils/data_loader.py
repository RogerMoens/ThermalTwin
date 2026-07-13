from pathlib import Path

import pandas as pd


class NetatmoDataLoader:
    """
    Utility class for loading Netatmo Weather.csv files.

    The loader automatically:
        - finds Weather.csv files
        - parses timestamps
        - renames columns
        - keeps only relevant variables
        - sorts chronologically
    """

    COLUMN_MAPPING = {
        "Timezone : Europe/Amsterdam": "datetime",
        "Temperature °C": "temp_in",
        "Humidity %": "hum_in",
        "Temperature °C.1": "temp_out",
        "Humidity %.1": "hum_out",
    }

    KEEP_COLUMNS = [
        "datetime",
        "temp_in",
        "hum_in",
        "temp_out",
        "hum_out",
    ]

    def __init__(self, data_root):
        self.data_root = Path(data_root)

    def list_weather_files(self):
        """Return all Weather.csv files."""
        return sorted(self.data_root.glob("*/Weather.csv"))

    def station_names(self):
        """Return available station names."""
        return [f.parent.name for f in self.list_weather_files()]

    def load(self, path):
        """Load a single Weather.csv file."""

        df = pd.read_csv(
            path,
            skiprows=5,
            sep=",",
            engine="python"
        )

        df.columns = [str(c).strip() for c in df.columns]

        df = df.rename(columns=self.COLUMN_MAPPING)

        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])

        df = df[self.KEEP_COLUMNS]

        df = df.sort_values("datetime").reset_index(drop=True)

        return df

    def load_station(self, station):
        """
        Load by station name.

        Example:
            loader.load_station("Home")
        """

        path = self.data_root / station / "Weather.csv"

        if not path.exists():
            raise FileNotFoundError(path)

        return self.load(path)

    def load_by_index(self, index):
        """
        Load station by index.

        Example:
            loader.load_by_index(2)
        """

        files = self.list_weather_files()

        if index >= len(files):
            raise IndexError("Invalid station index.")

        return self.load(files[index])