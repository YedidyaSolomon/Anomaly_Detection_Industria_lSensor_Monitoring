"""
preprocessing.py
----------------
Scales features for ML-based detectors and provides a helper that builds
a rolling-window feature matrix for sequential/collective detection.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data_loader import FEATURE_COLS


def preprocess(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """
    Standard-scale the five sensor columns.

    Returns
    -------
    X_scaled : np.ndarray  shape (n, 5)
    scaler   : fitted StandardScaler (kept for inverse transforms)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_COLS].values)
    return X_scaled, scaler


def rolling_features(
    df: pd.DataFrame,
    window: int = 60,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build a rolling-window feature matrix.

    For each column in *features* compute the rolling mean and rolling std
    over *window* minutes, then drop rows that have NaN (the first window-1
    rows).

    Returns a DataFrame aligned with df.index (NaN rows dropped at the head).
    """
    if features is None:
        features = FEATURE_COLS

    parts = {}
    for col in features:
        parts[f"{col}_rmean"] = df[col].rolling(window, min_periods=window).mean()
        parts[f"{col}_rstd"] = df[col].rolling(window, min_periods=window).std()

    roll_df = pd.DataFrame(parts, index=df.index).dropna()
    return roll_df


def add_power_rpm_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a derived feature: power_kw / (rotation_rpm + 1e-9).
    Useful for detecting contextual power-vs-load anomalies.
    """
    df = df.copy()
    df["power_per_rpm"] = df["power_kw"] / (df["rotation_rpm"] + 1e-9)
    return df
