"""
run_pipeline.py
---------------
End-to-end anomaly detection pipeline.

Run from project root:
    python run_pipeline.py

Outputs
-------
results/predictions/anomaly_predictions.csv
results/metrics/summary_metrics.json
results/plots/*.png
"""

import sys
from pathlib import Path

# Make sure src/ is importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader  import load_data
from src.ensamble     import combine_detectors
from src.evaluation   import run_all_evaluation


def main():
    print("=" * 60)
    print("  Anomaly Detection Pipeline — Industrial Sensor Monitoring")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────
    print("\n[1/3] Loading data …")
    df = load_data()
    print(f"      {len(df):,} samples | {df.index[0]} → {df.index[-1]}")

    # ── 2. Run all detectors + combine ────────────────────────────
    print("\n[2/3] Running detectors …")
    result_df = combine_detectors(df, vote_threshold=2, contamination=0.03)

    n_anom = result_df["final_anomaly"].sum()
    pct    = 100.0 * n_anom / len(result_df)
    print(f"      Total anomalies flagged: {n_anom:,} ({pct:.1f}%)")

    type_summary = (
        result_df[result_df["final_anomaly"]]["anomaly_type"]
        .str.split(", ").explode().value_counts()
    )
    print("\n      Anomaly type breakdown:")
    for t, c in type_summary.items():
        print(f"        {t:<25} {c:>5}")

    # ── 3. Evaluate & save ────────────────────────────────────────
    print("\n[3/3] Generating plots and saving results …")
    metrics = run_all_evaluation(result_df)

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print(f"  Metrics  → results/metrics/summary_metrics.json")
    print(f"  Plots    → results/plots/")
    print(f"  CSV      → results/predictions/anomaly_predictions.csv")
    print("=" * 60)

    return result_df, metrics


if __name__ == "__main__":
    main()
