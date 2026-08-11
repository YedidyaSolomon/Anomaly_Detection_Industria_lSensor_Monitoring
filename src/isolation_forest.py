
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .preprocessing import preprocess, add_power_rpm_ratio
from .data_loader import FEATURE_COLS


def isolation_forest_detection(
    df: pd.DataFrame,
    contamination: float = 0.03,
    n_estimators: int = 200,
    random_state: int = 42,
    use_power_ratio: bool = True,
) -> pd.DataFrame:
   
    work_df = df.copy()
    if use_power_ratio:
        work_df = add_power_rpm_ratio(work_df)
        feat_cols = FEATURE_COLS + ["power_per_rpm"]
    else:
        feat_cols = FEATURE_COLS

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(work_df[feat_cols].values)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)

    raw_scores = model.decision_function(X)   # higher = more normal
    predictions = model.predict(X)            # -1 = anomaly, 1 = normal

    result_df = pd.DataFrame(
        {
            "if_anomaly": predictions == -1,
            "if_score": -raw_scores,          # flip: higher = more anomalous
        },
        index=df.index,
    )
    return result_df
