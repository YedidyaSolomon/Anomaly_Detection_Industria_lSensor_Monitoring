"""
evaluation.py
-------------
Summarises detection results, saves metrics to JSON, and generates
all result plots saved to results/plots/ and results/metrics/.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


RESULTS_DIR  = Path(__file__).resolve().parent.parent / "results"
PLOTS_DIR    = RESULTS_DIR / "plots"
METRICS_DIR  = RESULTS_DIR / "metrics"
PRED_DIR     = RESULTS_DIR / "predictions"

for d in [PLOTS_DIR, METRICS_DIR, PRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SENSOR_LABELS = {
    "temperature_c":  "Temperature (°C)",
    "pressure_kpa":   "Pressure (kPa)",
    "vibration_mm_s": "Vibration (mm/s)",
    "rotation_rpm":   "Rotation (RPM)",
    "power_kw":       "Power (kW)",
}

TYPE_COLORS = {
    "point_vib":       "#e74c3c",
    "point_pressure":  "#e67e22",
    "sensor_fault":    "#9b59b6",
    "collective_drift":"#2ecc71",
    "contextual_power":"#3498db",
    "unknown":         "#95a5a6",
}


# ── Metrics ───────────────────────────────────────────────────────────────────

def summarize_results(result_df: pd.DataFrame) -> dict:
    """
    Compute summary statistics and save to JSON.
    Returns the metrics dict.
    """
    total       = len(result_df)
    n_anomalies = int(result_df["final_anomaly"].sum())
    pct         = round(100.0 * n_anomalies / total, 2)

    type_counts = (
        result_df[result_df["final_anomaly"]]
        ["anomaly_type"]
        .str.split(", ")
        .explode()
        .value_counts()
        .to_dict()
    )

    detector_counts = {
        col: int(result_df[col].sum())
        for col in [
            "vib_zscore", "vib_iqr", "pressure_drop", "rpm_flatline",
            "temp_zscore", "if_anomaly", "lof_anomaly", "pca_anomaly",
            "roll_anomaly", "collective_anomaly",
        ]
        if col in result_df.columns
    }

    metrics = {
        "total_samples":    total,
        "total_anomalies":  n_anomalies,
        "anomaly_pct":      pct,
        "by_type":          type_counts,
        "by_detector":      detector_counts,
    }

    out_path = METRICS_DIR / "summary_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluation] Metrics saved → {out_path}")
    return metrics


# ── Plots ─────────────────────────────────────────────────────────────────────

def _fmt_xaxis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)


def plot_all_sensors_overview(result_df: pd.DataFrame):
    """5-panel time-series with final anomalies highlighted."""
    sensors = list(SENSOR_LABELS.keys())
    fig, axes = plt.subplots(len(sensors), 1, figsize=(16, 14), sharex=True)
    fig.suptitle("All Sensors — Anomalies Highlighted", fontsize=14, fontweight="bold")

    anomalies = result_df[result_df["final_anomaly"]]

    for ax, col in zip(axes, sensors):
        ax.plot(result_df.index, result_df[col], lw=0.6, color="#34495e", label=col)
        ax.scatter(
            anomalies.index, anomalies[col],
            c="red", s=12, zorder=5, label="anomaly", alpha=0.7,
        )
        ax.set_ylabel(SENSOR_LABELS[col], fontsize=8)
        ax.grid(True, alpha=0.3)

    _fmt_xaxis(axes[-1])
    axes[0].legend(loc="upper right", fontsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = PLOTS_DIR / "overview_all_sensors.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[evaluation] Plot saved → {path}")


def plot_anomaly_types(result_df: pd.DataFrame):
    """Scatter overlay coloured by anomaly type on vibration and temperature."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.suptitle("Anomaly Types Over Time", fontsize=14, fontweight="bold")

    ax1.plot(result_df.index, result_df["vibration_mm_s"], lw=0.6, color="#7f8c8d")
    ax1.set_ylabel("Vibration (mm/s)", fontsize=9)

    ax2.plot(result_df.index, result_df["temperature_c"], lw=0.6, color="#7f8c8d")
    ax2.set_ylabel("Temperature (°C)", fontsize=9)

    for atype, color in TYPE_COLORS.items():
        mask = result_df["anomaly_type"].str.contains(atype, na=False)
        subset = result_df[mask]
        if subset.empty:
            continue
        ax1.scatter(subset.index, subset["vibration_mm_s"],
                    c=color, s=20, zorder=5, label=atype, alpha=0.8)
        ax2.scatter(subset.index, subset["temperature_c"],
                    c=color, s=20, zorder=5, alpha=0.8)

    ax1.legend(loc="upper right", fontsize=8, ncol=3)
    _fmt_xaxis(ax2)
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = PLOTS_DIR / "anomaly_types.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[evaluation] Plot saved → {path}")


def plot_detector_comparison(result_df: pd.DataFrame):
    """Heatmap: each row = detector, each column = time bin."""
    bool_cols = [
        "vib_zscore", "vib_iqr", "pressure_drop", "rpm_flatline",
        "temp_zscore", "if_anomaly", "lof_anomaly", "pca_anomaly",
        "roll_anomaly", "collective_anomaly",
    ]
    bool_cols = [c for c in bool_cols if c in result_df.columns]

    # Resample to 15-min bins for readability
    bin_df = result_df[bool_cols].resample("15min").mean()

    fig, ax = plt.subplots(figsize=(18, 5))
    im = ax.imshow(
        bin_df.T.values,
        aspect="auto",
        cmap="YlOrRd",
        vmin=0, vmax=1,
        interpolation="nearest",
    )
    ax.set_yticks(range(len(bool_cols)))
    ax.set_yticklabels(bool_cols, fontsize=8)

    # X-axis: show every 24th bin (~6 hours)
    n_bins = bin_df.shape[0]
    step = max(1, n_bins // 20)
    ax.set_xticks(range(0, n_bins, step))
    ax.set_xticklabels(
        [str(t)[:16] for t in bin_df.index[::step]],
        rotation=40, ha="right", fontsize=7,
    )
    plt.colorbar(im, ax=ax, label="Anomaly fraction in 15-min bin")
    ax.set_title("Detector Agreement Heatmap (15-min bins)", fontsize=13)
    fig.tight_layout()
    path = PLOTS_DIR / "detector_comparison_heatmap.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[evaluation] Plot saved → {path}")


def plot_vote_distribution(result_df: pd.DataFrame):
    """Bar chart of anomaly vote counts."""
    counts = result_df["anomaly_votes"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(counts.index, counts.values, color="#3498db", edgecolor="white")
    ax.axvline(x=1.5, color="red", linestyle="--", label="vote threshold = 2")
    ax.set_xlabel("Number of detectors flagging anomaly", fontsize=10)
    ax.set_ylabel("Count of timestamps", fontsize=10)
    ax.set_title("Anomaly Vote Distribution", fontsize=12)
    ax.legend(fontsize=9)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(int(bar.get_height())),
            ha="center", va="bottom", fontsize=8,
        )
    fig.tight_layout()
    path = PLOTS_DIR / "vote_distribution.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[evaluation] Plot saved → {path}")


def plot_power_vs_rpm(result_df: pd.DataFrame):
    """Scatter: power_kw vs rotation_rpm, coloured by anomaly type."""
    fig, ax = plt.subplots(figsize=(9, 6))
    normal = result_df[~result_df["final_anomaly"]]
    ax.scatter(normal["rotation_rpm"], normal["power_kw"],
               s=4, color="#bdc3c7", alpha=0.4, label="normal")

    for atype, color in TYPE_COLORS.items():
        mask = result_df["anomaly_type"].str.contains(atype, na=False)
        subset = result_df[mask]
        if subset.empty:
            continue
        ax.scatter(subset["rotation_rpm"], subset["power_kw"],
                   s=20, color=color, alpha=0.8, label=atype, zorder=5)

    ax.set_xlabel("Rotation (RPM)", fontsize=10)
    ax.set_ylabel("Power (kW)", fontsize=10)
    ax.set_title("Power vs RPM — Contextual Anomalies", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = PLOTS_DIR / "power_vs_rpm_anomalies.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[evaluation] Plot saved → {path}")


def plot_collective_detail(result_df: pd.DataFrame):
    """Zoom into collective/drift anomalies: temp + vib with shaded region."""
    bearing_mask = result_df["anomaly_type"].str.contains("collective_drift", na=False)
    if not bearing_mask.any():
        print("[evaluation] No collective drift anomalies found, skipping detail plot.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    fig.suptitle("Collective Anomaly Detail — Bearing Wear Drift", fontsize=13)

    ax1.plot(result_df.index, result_df["temperature_c"], lw=0.8, color="#e74c3c")
    ax1.set_ylabel("Temperature (°C)", fontsize=9)

    ax2.plot(result_df.index, result_df["vibration_mm_s"], lw=0.8, color="#2980b9")
    ax2.set_ylabel("Vibration (mm/s)", fontsize=9)

    # Shade detected collective periods
    in_region = False
    start = None
    for ts, flag in bearing_mask.items():
        if flag and not in_region:
            start = ts
            in_region = True
        elif not flag and in_region:
            for ax in (ax1, ax2):
                ax.axvspan(start, ts, color="#f39c12", alpha=0.25)
            in_region = False
    if in_region:
        for ax in (ax1, ax2):
            ax.axvspan(start, result_df.index[-1], color="#f39c12", alpha=0.25)

    _fmt_xaxis(ax2)
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3)
    # Legend patch
    import matplotlib.patches as mpatches
    patch = mpatches.Patch(color="#f39c12", alpha=0.4, label="collective anomaly")
    ax1.legend(handles=[patch], fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = PLOTS_DIR / "collective_detail.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[evaluation] Plot saved → {path}")


def save_predictions(result_df: pd.DataFrame):
    """Save the full annotated result DataFrame to CSV."""
    path = PRED_DIR / "anomaly_predictions.csv"
    result_df.to_csv(path)
    print(f"[evaluation] Predictions saved → {path}")


def run_all_evaluation(result_df: pd.DataFrame) -> dict:
    """Run all evaluation steps: metrics + all plots + save predictions."""
    metrics = summarize_results(result_df)
    plot_all_sensors_overview(result_df)
    plot_anomaly_types(result_df)
    plot_detector_comparison(result_df)
    plot_vote_distribution(result_df)
    plot_power_vs_rpm(result_df)
    plot_collective_detail(result_df)
    save_predictions(result_df)
    return metrics
