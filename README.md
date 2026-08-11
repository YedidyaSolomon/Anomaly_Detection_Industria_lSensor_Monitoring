# Anomaly Detection — Industrial Sensor Monitoring

Detects anomalies in a 5-sensor industrial pump dataset (~8,000 samples, 5.5 days)
using an ensemble of statistical and ML methods.

## Anomaly types detected

| Type | Description | Method |
|---|---|---|
| **point_vib** | Sudden vibration spikes (impact events) | Z-score, IQR |
| **point_pressure** | Sudden pressure drops (valve faults) | Delta threshold |
| **sensor_fault** | RPM flatline — stuck sensor | Rolling std |
| **collective_drift** | Sustained temp+vib rise (bearing wear) | CUSUM, rolling IF, bearing heuristic |
| **contextual_power** | Power too high for current RPM | Isolation Forest, LOF, PCA |

## Project structure

```
anomaly-detection/
├── data/
│   └── machine_sensors.csv
├── src/
│   ├── data_loader.py        # CSV → clean DataFrame
│   ├── preprocessing.py      # Scaling, rolling features, derived features
│   ├── univariate.py         # Z-score, IQR, flatline, pressure-drop detectors
│   ├── isolation_forest.py   # Multivariate Isolation Forest
│   ├── lof.py                # Local Outlier Factor
│   ├── pca.py                # PCA reconstruction error
│   ├── collective.py         # Rolling-window IF, CUSUM, bearing heuristic
│   ├── ensamble.py           # Combines all detectors + type tagging
│   └── evaluation.py         # Metrics, plots, CSV export
├── UI/
│   └── app.py                # Streamlit dashboard
├── results/
│   ├── metrics/              # summary_metrics.json
│   ├── plots/                # PNG plots
│   └── predictions/          # anomaly_predictions.csv
├── run_pipeline.py           # CLI entry point
└── requirement.txt
```

## Quick start

```bash
# Install dependencies
pip install -r requirement.txt

# Run the detection pipeline (generates all results)
python run_pipeline.py

# Launch the interactive UI
streamlit run UI/app.py
```

## How the ensemble works

Each timestamp is voted on by up to 10 detectors:

```
vib_zscore · vib_iqr · pressure_drop · rpm_flatline · temp_zscore
if_anomaly · lof_anomaly · pca_anomaly · roll_anomaly · collective_anomaly
```

A point is labelled **anomaly** when `votes ≥ vote_threshold` (default = 2).
The vote threshold is tunable live in the UI sidebar.

## Precision vs Recall trade-off

In production, **recall should be prioritised** for bearing-wear and valve faults
because a missed failure carries far higher cost (unplanned downtime, equipment
damage, safety risk) than a false positive (an unnecessary inspection).
The vote threshold lets operators tune this trade-off: lower → higher recall,
higher → higher precision.
