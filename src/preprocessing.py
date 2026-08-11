

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data_loader import FEATURE_COLS


def preprocess(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
   
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_COLS].values)
    return X_scaled, scaler


def rolling_features(
    df: pd.DataFrame,
    window: int = 60,
    features: list[str] | None = None,
) -> pd.DataFrame:
    
    if features is None:
        features = FEATURE_COLS

    parts = {}
    for col in features:
        parts[f"{col}_rmean"] = df[col].rolling(window, min_periods=window).mean()
        parts[f"{col}_rstd"] = df[col].rolling(window, min_periods=window).std()

    roll_df = pd.DataFrame(parts, index=df.index).dropna()
    return roll_df


def add_power_rpm_ratio(df: pd.DataFrame) -> pd.DataFrame:
   
    df = df.copy()
    df["power_per_rpm"] = df["power_kw"] / (df["rotation_rpm"] + 1e-9)
    return df
