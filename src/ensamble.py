"""
ensamble.py
-----------
Combines all individual detector outputs into a unified anomaly result.

Strategy
--------
1. Collect every boolean anomaly flag from all detectors.
2. Count how many detectors agree on each timestamp (anomaly_votes).
3. Apply a vote threshold to produce the final label.
4. Also tag each anomaly with its *type* based on which detectors fired:
   - point_vib       : vibration spike (impact event)
   - point_pressure  : sudden pressure drop (valve fault)
   - contextual_power: power/load mismatch
   - collective_drift : sustained temp+vib rise (bearing wear)
   - sensor_fault    : flatlined rotation_rpm
"""

import pandas as pd
import numpy as np

from .univariate        import run_all_univariate
from .isolation_forest  import isolation_forest_detection
from .lof               import lof_detection
from .pca               import pca_detection
from .collective        import run_all_collective


# ── Type-tagging helpers ──────────────────────────────────────────────────────

def _tag_anomaly_type(row: pd.Series) -> str:
    """Return a comma-separated string of anomaly types for a flagged row."""
    types = []
    # Point: vibration spike
    if row.get("vib_zscore", False) or row.get("vib_iqr", False):
        types.append("point_vib")
    # Point: pressure drop
    if row.get("pressure_drop", False):
        types.append("point_pressure")
    # Sensor fault: flatline
    if row.get("rpm_flatline", False):
        types.append("sensor_fault")
    # Collective / drift
    if row.get("bearing_wear", False) or row.get("collective_anomaly", False):
        types.append("collective_drift")
    # Contextual power — flagged by IF/LOF/PCA but NOT by simple univariate
    uni_flagged = any([
        row.get("vib_zscore", False),
        row.get("vib_iqr", False),
        row.get("pressure_drop", False),
        row.get("rpm_flatline", False),
    ])
    ml_flagged = (
        row.get("if_anomaly", False)
        or row.get("lof_anomaly", False)
        or row.get("pca_anomaly", False)
    )
    if ml_flagged and not uni_flagged and not types:
        types.append("contextual_power")

    return ", ".join(types) if types else "unknown"


# ── Main combiner ─────────────────────────────────────────────────────────────

def combine_detectors(
    df: pd.DataFrame,
    vote_threshold: int = 2,
    contamination: float = 0.03,
) -> pd.DataFrame:
    """
    Run all detectors and merge results.

    Parameters
    ----------
    df              : raw sensor DataFrame (DatetimeIndex)
    vote_threshold  : minimum number of detector agreements to label anomaly
    contamination   : passed to ML detectors

    Returns
    -------
    full DataFrame with all flag columns plus:
        - anomaly_votes   : int  (how many detectors flagged this point)
        - final_anomaly   : bool (votes >= vote_threshold)
        - anomaly_type    : str  (best-guess type label)
    Also includes all original sensor columns for easy downstream use.
    """
    # ── Run each detector ──
    uni_df   = run_all_univariate(df)
    if_df    = isolation_forest_detection(df, contamination=contamination)
    lof_df   = lof_detection(df, contamination=contamination)
    pca_df   = pca_detection(df, contamination=contamination)
    coll_df  = run_all_collective(df)

    # ── Collect boolean flag columns for voting ──
    vote_cols = [
        uni_df["vib_zscore"],
        uni_df["vib_iqr"],
        uni_df["pressure_drop"],
        uni_df["rpm_flatline"],
        uni_df["temp_zscore"],
        if_df["if_anomaly"],
        lof_df["lof_anomaly"],
        pca_df["pca_anomaly"],
        coll_df["roll_anomaly"],
        coll_df["collective_anomaly"],
    ]

    votes = pd.concat(vote_cols, axis=1).sum(axis=1).rename("anomaly_votes")

    # ── Merge everything ──
    result = pd.concat(
        [
            df,            # original sensor readings
            uni_df,
            if_df,
            lof_df,
            pca_df[["pca_anomaly", "pca_recon_error"]],
            coll_df,
            votes,
        ],
        axis=1,
    )

    result["final_anomaly"] = result["anomaly_votes"] >= vote_threshold

    # Tag type only for final anomalies
    result["anomaly_type"] = result.apply(
        lambda row: _tag_anomaly_type(row) if row["final_anomaly"] else "normal",
        axis=1,
    )

    return result
