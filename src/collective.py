import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ── Rolling-window Isolation Forest ──────────────────────────────────────────

def rolling_window_detector(
    df: pd.DataFrame,
    window: int = 60,
    contamination: float = 0.03,
    random_state: int = 42,
) -> pd.DataFrame:
    
    roll_features = pd.DataFrame(index=df.index)
    for col in ["temperature_c", "vibration_mm_s", "pressure_kpa"]:
        roll_features[f"{col}_rmean"] = (
            df[col].rolling(window, min_periods=window).mean()
        )
        roll_features[f"{col}_rstd"] = (
            df[col].rolling(window, min_periods=window).std()
        )
    roll_features = roll_features.dropna()

    scaler = StandardScaler()
    X = scaler.fit_transform(roll_features.values)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)
    preds = model.predict(X)
    scores = -model.decision_function(X)

    # Re-index to full df index, fill leading NaN rows with False / 0
    roll_anomaly = pd.Series(preds == -1, index=roll_features.index, name="roll_anomaly")
    roll_score   = pd.Series(scores,      index=roll_features.index, name="roll_score")

    result_df = pd.DataFrame(
        {
            "roll_anomaly": roll_anomaly.reindex(df.index, fill_value=False),
            "roll_score":   roll_score.reindex(df.index,   fill_value=0.0),
        },
        index=df.index,
    )
    return result_df


# ── CUSUM change-point ────────────────────────────────────────────────────────

def cusum_flag(
    df: pd.DataFrame,
    col: str = "temperature_c",
    k: float = 0.5,
    h: float = 5.0,
    window: int = 60,
) -> pd.Series:
    """
    Two-sided CUSUM on a rolling-mean-smoothed signal.

    k : allowance (slack) in units of signal std
    h : decision threshold in units of signal std

    Returns boolean Series (True = change detected).
    """
    smooth = df[col].rolling(window, min_periods=1).mean()
    mu = smooth.mean()
    sigma = smooth.std() + 1e-12

    # Standardise
    x = (smooth - mu) / sigma

    cusum_pos = np.zeros(len(x))
    cusum_neg = np.zeros(len(x))
    for i in range(1, len(x)):
        cusum_pos[i] = max(0.0, cusum_pos[i - 1] + x.iloc[i] - k)
        cusum_neg[i] = max(0.0, cusum_neg[i - 1] - x.iloc[i] - k)

    flag = pd.Series(
        (cusum_pos > h) | (cusum_neg > h),
        index=df.index,
        name=f"cusum_{col}",
    )
    return flag


# ── Bearing-wear heuristic ────────────────────────────────────────────────────

def bearing_wear_flag(
    df: pd.DataFrame,
    window: int = 60,
    temp_rise_threshold: float = 1.5,   # °C above rolling baseline
    vib_rise_threshold: float  = 0.15,  # mm/s above rolling baseline
    min_duration: int = 90,             # minutes both must be elevated
) -> pd.Series:
    """
    Flag sustained periods where BOTH temperature and vibration rise together.

    Strategy:
      1. Compute a slow baseline (2× window rolling mean).
      2. Compute a fast signal (window rolling mean).
      3. Both being above their respective thresholds for >= min_duration
         consecutive minutes flags a bearing-wear collective anomaly.
    """
    # Fast-moving average
    temp_fast = df["temperature_c"].rolling(window, min_periods=1).mean()
    vib_fast  = df["vibration_mm_s"].rolling(window, min_periods=1).mean()

    # Slow baseline
    temp_slow = df["temperature_c"].rolling(window * 4, min_periods=window).mean().ffill()
    vib_slow  = df["vibration_mm_s"].rolling(window * 4, min_periods=window).mean().ffill()

    temp_elevated = (temp_fast - temp_slow) > temp_rise_threshold
    vib_elevated  = (vib_fast  - vib_slow)  > vib_rise_threshold
    both_elevated = temp_elevated & vib_elevated

    # Require sustained elevation: rolling sum >= min_duration
    sustained = (
        both_elevated.rolling(min_duration, min_periods=1).sum() >= min_duration
    )

    return sustained.rename("bearing_wear")


# ── Convenience wrapper ───────────────────────────────────────────────────────

def run_all_collective(df: pd.DataFrame) -> pd.DataFrame:
    """Run all collective detectors and return a combined flag DataFrame."""
    roll_df   = rolling_window_detector(df, window=60, contamination=0.03)
    cusum_temp = cusum_flag(df, col="temperature_c", k=0.5, h=5.0, window=60)
    cusum_vib  = cusum_flag(df, col="vibration_mm_s", k=0.5, h=4.0, window=30)
    bearing    = bearing_wear_flag(df, window=60, min_duration=90)

    result = pd.DataFrame(
        {
            "roll_anomaly":  roll_df["roll_anomaly"],
            "roll_score":    roll_df["roll_score"],
            "cusum_temp":    cusum_temp,
            "cusum_vib":     cusum_vib,
            "bearing_wear":  bearing,
            # combined collective flag: any of the above
            "collective_anomaly": (
                roll_df["roll_anomaly"] | cusum_temp | cusum_vib | bearing
            ),
        },
        index=df.index,
    )
    return result
