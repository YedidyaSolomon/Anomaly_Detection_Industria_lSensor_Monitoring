

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .data_loader import FEATURE_COLS
from .preprocessing import add_power_rpm_ratio


def pca_detection(
    df: pd.DataFrame,
    n_components: int = 3,
    contamination: float = 0.03,
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

    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(X_reduced)

    # Mean squared reconstruction error per sample
    recon_error = np.mean((X - X_reconstructed) ** 2, axis=1)

    threshold = np.quantile(recon_error, 1.0 - contamination)
    anomaly = recon_error > threshold

    result_df = pd.DataFrame(
        {
            "pca_anomaly": anomaly,
            "pca_recon_error": recon_error,
            "pca_threshold": threshold,
        },
        index=df.index,
    )

    # Store explained variance for reporting
    result_df.attrs["explained_variance_ratio"] = pca.explained_variance_ratio_
    result_df.attrs["n_components"] = n_components

    return result_df
