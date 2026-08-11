"""
univariate.py
-------------
Statistical univariate anomaly detectors.

Provides:
  - z_score_flag      : global z-score threshold
  - rolling_zscore_flag : rolling mean ± k·std (better for drifting signals)
  - iqr_flag          : IQR fence method
  - flatline_flag     : detects stuck / flatlined sensors (std ≈ 0 in window)
  - pressure_drop_flag: sudden large negative delta in pressure
"""

import numpy as np
import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────────────

def _zscore(series: pd.Series) -> pd.Series:
    mu, sigma = series.mean(), series.std()
    return (series - mu) / (sigma + 1e-12)


# ── detectors ────────────────────────────────────────────────────────────────

def z_score_flag(
    df: pd.DataFrame,
    col: str = "vibration_mm_s",
    threshold: float = 3.5,
) -> pd.Series:
    """
    Flag rows where |z-score| > threshold.
    Returns a boolean Series aligned with df.index.
    """
    z = _zscore(df[col])
    return (z.abs() > threshold).rename(f"zscore_{col}")


def rolling_zscore_flag(
    df: pd.DataFrame,
    col: str = "vibration_mm_s",
    window: int = 120,
    k: float = 3.5,
) -> pd.Series:
    """
    Rolling z-score: flag points that deviate > k·σ from the local rolling mean.
    Good at catching sudden spikes without being tripped by slow drift.
    """
    roll_mean = df[col].rolling(window, min_periods=30).mean()
    roll_std = df[col].rolling(window, min_periods=30).std()
    z = (df[col] - roll_mean) / (roll_std + 1e-12)
    return (z.abs() > k).rename(f"rolling_zscore_{col}")


def iqr_flag(
    df: pd.DataFrame,
    col: str = "vibration_mm_s",
    k: float = 3.0,
) -> pd.Series:
    """
    IQR-fence method: flag points outside [Q1 - k·IQR, Q3 + k·IQR].
    More robust to heavy-tailed distributions than z-score.
    """
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return ((df[col] < lower) | (df[col] > upper)).rename(f"iqr_{col}")


def flatline_flag(
    df: pd.DataFrame,
    col: str = "rotation_rpm",
    window: int = 15,
    std_threshold: float = 0.05,
) -> pd.Series:
    """
    Detect stuck / flatlined sensors: rolling std < std_threshold over *window*.
    A perfectly constant signal in a normally noisy sensor is a fault.
    """
    roll_std = df[col].rolling(window, min_periods=window).std()
    # Normalise by global std so the threshold is scale-independent
    global_std = df[col].std()
    normalised_std = roll_std / (global_std + 1e-12)
    flag = (normalised_std < std_threshold).rename(f"flatline_{col}")
    return flag


def pressure_drop_flag(
    df: pd.DataFrame,
    col: str = "pressure_kpa",
    delta_threshold: float = -3.0,
) -> pd.Series:
    """
    Flag sudden large single-step drops in pressure (valve fault signature).
    delta_threshold is in the same units as col (kPa).
    """
    delta = df[col].diff()
    return (delta < delta_threshold).rename(f"pressure_drop_{col}")


def run_all_univariate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience: run every univariate detector and return a DataFrame of flags.
    Each column is a boolean (True = anomaly).
    """
    flags = pd.DataFrame(index=df.index)
    flags["vib_zscore"]       = z_score_flag(df, "vibration_mm_s", threshold=3.5)
    flags["vib_rolling_z"]    = rolling_zscore_flag(df, "vibration_mm_s", window=120, k=3.5)
    flags["vib_iqr"]          = iqr_flag(df, "vibration_mm_s", k=3.0)
    flags["pressure_drop"]    = pressure_drop_flag(df, "pressure_kpa", delta_threshold=-3.0)
    flags["rpm_flatline"]     = flatline_flag(df, "rotation_rpm", window=15, std_threshold=0.05)
    flags["temp_zscore"]      = z_score_flag(df, "temperature_c", threshold=4.0)
    return flags
