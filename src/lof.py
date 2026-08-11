

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from .data_loader import FEATURE_COLS
from .preprocessing import add_power_rpm_ratio


def lof_detection(
    df: pd.DataFrame,
    contamination: float = 0.03,
    n_neighbors: int = 30,
    use_power_ratio: bool = True,
) -> pd.DataFrame:
   
    work_df = df.copy()
    if use_power_ratio:
        work_df = add_power_rpm_ratio(work_df)
        feat_cols = FEATURE_COLS + ["power_per_rpm"]
    else:
        feat_cols = FEATURE_COLS

    scaler = StandardScaler()
    X = scaler.fit_transform(work_df[feat_cols].values)

    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=False,
        n_jobs=-1,
    )
    predictions = model.fit_predict(X)   # -1 = anomaly, 1 = normal
    # negative_outlier_factor_: more negative = more anomalous; flip it
    lof_scores = -model.negative_outlier_factor_

    result_df = pd.DataFrame(
        {
            "lof_anomaly": predictions == -1,
            "lof_score": lof_scores,
        },
        index=df.index,
    )
    return result_df
