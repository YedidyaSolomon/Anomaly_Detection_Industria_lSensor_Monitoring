"""
data_loader.py
--------------
Loads machine_sensors.csv and returns a clean DataFrame with a proper DatetimeIndex.
"""

import pandas as pd
from pathlib import Path


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "machine_sensors.csv"

FEATURE_COLS = [
    "temperature_c",
    "pressure_kpa",
    "vibration_mm_s",
    "rotation_rpm",
    "power_kw",
]


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load sensor CSV, parse timestamps, sort, and return a clean DataFrame."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.set_index("timestamp")
    # Ensure all feature columns are float
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
