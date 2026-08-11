

import sys
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import streamlit as st

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_data, FEATURE_COLS
from src.ensamble    import combine_detectors

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Anomaly Detection — Industrial Sensors",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── colour palette ────────────────────────────────────────────────────────────
TYPE_COLORS = {
    "point_vib":        "#e74c3c",
    "point_pressure":   "#e67e22",
    "sensor_fault":     "#9b59b6",
    "collective_drift": "#27ae60",
    "contextual_power": "#2980b9",
    "unknown":          "#95a5a6",
    "normal":           "#bdc3c7",
}

SENSOR_UNITS = {
    "temperature_c":  "°C",
    "pressure_kpa":   "kPa",
    "vibration_mm_s": "mm/s",
    "rotation_rpm":   "RPM",
    "power_kw":       "kW",
}

SENSOR_LABELS = {
    "temperature_c":  "Temperature",
    "pressure_kpa":   "Pressure",
    "vibration_mm_s": "Vibration",
    "rotation_rpm":   "Rotation",
    "power_kw":       "Power",
}

# ── caching ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading & running detectors …")
def get_results(vote_threshold: int, contamination: float) -> pd.DataFrame:
    df = load_data()
    result = combine_detectors(df, vote_threshold=vote_threshold,
                               contamination=contamination)
    return result


@st.cache_data(show_spinner=False)
def load_metrics() -> dict | None:
    path = ROOT / "results" / "metrics" / "summary_metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/settings.png", width=50)
    st.title("Controls")
    st.markdown("---")

    st.subheader("⚙️ Detection Settings")
    vote_threshold = st.slider(
        "Vote threshold (min detectors to agree)",
        min_value=1, max_value=8, value=2, step=1,
        help="A point is flagged as anomaly only if this many detectors agree.",
    )
    contamination = st.slider(
        "Contamination rate",
        min_value=0.01, max_value=0.10, value=0.03, step=0.01,
        help="Expected fraction of anomalies — used by ML detectors.",
    )

    st.markdown("---")
    st.subheader("🔍 Sensor Filter")
    selected_sensors = st.multiselect(
        "Sensors to display",
        options=FEATURE_COLS,
        default=FEATURE_COLS,
        format_func=lambda x: SENSOR_LABELS[x],
    )

    st.markdown("---")
    st.subheader("🏷️ Anomaly Type Filter")
    all_types = list(TYPE_COLORS.keys())
    all_types.remove("normal")
    selected_types = st.multiselect(
        "Show anomaly types",
        options=all_types,
        default=all_types,
    )

    st.markdown("---")
    st.caption("Built with Streamlit + Plotly")


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
result_df = get_results(vote_threshold, contamination)

# Apply type filter
def type_matches(atype: str) -> bool:
    if atype == "normal":
        return True
    return any(t in atype for t in selected_types)

display_df = result_df.copy()
display_df["_show"] = display_df["anomaly_type"].apply(type_matches)

anomalies = display_df[display_df["final_anomaly"] & display_df["_show"]]
normals   = display_df[~display_df["final_anomaly"]]


# ─────────────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("🔧 Industrial Sensor Anomaly Detection")
st.markdown(
    "Real-time style dashboard for monitoring an industrial pump across **5 sensors** "
    "over ~5.5 days. Detects point anomalies, collective drift, contextual mismatch, "
    "and sensor faults using an ensemble of statistical and ML methods."
)
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
#  KPI ROW
# ─────────────────────────────────────────────────────────────────────────────
total        = len(result_df)
n_anom       = int(result_df["final_anomaly"].sum())
pct_anom     = 100.0 * n_anom / total

type_counts = (
    result_df[result_df["final_anomaly"]]["anomaly_type"]
    .str.split(", ").explode().value_counts()
)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Samples",   f"{total:,}")
col2.metric("Anomalies Found", f"{n_anom:,}",  f"{pct_anom:.1f}%")
col3.metric("Vibration Spikes",  str(type_counts.get("point_vib", 0)),       delta_color="off")
col4.metric("Pressure Drops",    str(type_counts.get("point_pressure", 0)),  delta_color="off")
col5.metric("Collective Drift",  str(type_counts.get("collective_drift", 0)),delta_color="off")

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Overview",
    "🔴 Anomaly Types",
    "🌡️ Sensor Deep-Dive",
    "🤖 Detector Comparison",
    "📊 Analysis",
    "📋 Data Table",
])


# ════════════════════════════════════════════════════════════════════════════
#  TAB 1 — Overview: all sensors with anomaly overlay
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("All Sensors Over Time")
    st.caption("Red markers = final anomaly flags. Use the sidebar to filter sensor and type.")

    if not selected_sensors:
        st.warning("Select at least one sensor in the sidebar.")
    else:
        fig = make_subplots(
            rows=len(selected_sensors), cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            subplot_titles=[f"{SENSOR_LABELS[s]} ({SENSOR_UNITS[s]})" for s in selected_sensors],
        )

        for i, col in enumerate(selected_sensors, start=1):
            # Normal trace
            fig.add_trace(
                go.Scattergl(
                    x=normals.index, y=normals[col],
                    mode="lines", name=col,
                    line=dict(width=0.8, color="#636e72"),
                    showlegend=(i == 1),
                    legendgroup="normal",
                    legendgrouptitle_text="Normal" if i == 1 else None,
                ),
                row=i, col=1,
            )
            # Anomaly scatter coloured by type
            for atype, color in TYPE_COLORS.items():
                if atype == "normal":
                    continue
                if atype not in selected_types:
                    continue
                mask = anomalies["anomaly_type"].str.contains(atype, na=False)
                sub  = anomalies[mask]
                if sub.empty:
                    continue
                fig.add_trace(
                    go.Scattergl(
                        x=sub.index, y=sub[col],
                        mode="markers",
                        marker=dict(size=6, color=color, opacity=0.85),
                        name=atype,
                        showlegend=(i == 1),
                        legendgroup=atype,
                        legendgrouptitle_text=atype if i == 1 else None,
                    ),
                    row=i, col=1,
                )

        fig.update_layout(
            height=220 * len(selected_sensors),
            margin=dict(l=60, r=20, t=40, b=40),
            legend=dict(orientation="h", y=-0.05, x=0),
            hovermode="x unified",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 2 — Anomaly Types
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Anomaly Types Breakdown")

    left, right = st.columns([1, 2])

    with left:
        # Donut chart of anomaly type distribution
        labels = [t for t in type_counts.index if t in selected_types]
        values = [type_counts[t] for t in labels]
        colors = [TYPE_COLORS.get(t, "#95a5a6") for t in labels]

        if labels:
            fig_pie = go.Figure(go.Pie(
                labels=labels, values=values,
                hole=0.45,
                marker=dict(colors=colors, line=dict(color="white", width=2)),
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Pct: %{percent}<extra></extra>",
            ))
            fig_pie.update_layout(
                title="Anomaly Type Distribution",
                height=350,
                margin=dict(l=10, r=10, t=40, b=10),
                template="plotly_white",
                showlegend=False,
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No anomaly types match current filter.")

        # Type legend table
        st.markdown("**Type Descriptions**")
        desc = {
            "point_vib":        "Sudden vibration spike (impact event)",
            "point_pressure":   "Sudden pressure drop (valve fault)",
            "sensor_fault":     "RPM flatline — stuck sensor",
            "collective_drift": "Sustained temp+vib rise (bearing wear)",
            "contextual_power": "Power too high for current RPM load",
        }
        for t, d in desc.items():
            color = TYPE_COLORS[t]
            st.markdown(
                f'<span style="background:{color};color:white;padding:2px 8px;'
                f'border-radius:4px;font-size:12px">{t}</span> {d}',
                unsafe_allow_html=True,
            )
            st.markdown("")

    with right:
        st.markdown("#### Vibration & Temperature coloured by anomaly type")
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                             subplot_titles=["Vibration (mm/s)", "Temperature (°C)"])

        fig2.add_trace(go.Scattergl(
            x=normals.index, y=normals["vibration_mm_s"],
            mode="lines", name="normal",
            line=dict(width=0.7, color="#aab2bd"),
            showlegend=True, legendgroup="normal",
        ), row=1, col=1)

        fig2.add_trace(go.Scattergl(
            x=normals.index, y=normals["temperature_c"],
            mode="lines", name="normal",
            line=dict(width=0.7, color="#aab2bd"),
            showlegend=False, legendgroup="normal",
        ), row=2, col=1)

        for atype, color in TYPE_COLORS.items():
            if atype in ("normal", "unknown"):
                continue
            if atype not in selected_types:
                continue
            mask = anomalies["anomaly_type"].str.contains(atype, na=False)
            sub  = anomalies[mask]
            if sub.empty:
                continue
            fig2.add_trace(go.Scattergl(
                x=sub.index, y=sub["vibration_mm_s"],
                mode="markers",
                marker=dict(size=7, color=color, opacity=0.9),
                name=atype, legendgroup=atype, showlegend=True,
            ), row=1, col=1)
            fig2.add_trace(go.Scattergl(
                x=sub.index, y=sub["temperature_c"],
                mode="markers",
                marker=dict(size=7, color=color, opacity=0.9),
                name=atype, legendgroup=atype, showlegend=False,
            ), row=2, col=1)

        fig2.update_layout(
            height=480, template="plotly_white",
            margin=dict(l=60, r=20, t=40, b=40),
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.08),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 3 — Sensor Deep-Dive
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Single-Sensor Deep-Dive")

    sensor_choice = st.selectbox(
        "Choose sensor",
        options=FEATURE_COLS,
        format_func=lambda x: f"{SENSOR_LABELS[x]} ({SENSOR_UNITS[x]})",
    )

    show_rolling = st.checkbox("Show rolling mean (60-min window)", value=True)

    fig3 = go.Figure()

    # Normal signal
    fig3.add_trace(go.Scattergl(
        x=result_df.index, y=result_df[sensor_choice],
        mode="lines", name=SENSOR_LABELS[sensor_choice],
        line=dict(width=0.9, color="#636e72"),
    ))

    # Rolling mean
    if show_rolling:
        roll = result_df[sensor_choice].rolling(60, min_periods=1).mean()
        fig3.add_trace(go.Scattergl(
            x=result_df.index, y=roll,
            mode="lines", name="60-min rolling mean",
            line=dict(width=1.5, color="#f39c12", dash="dot"),
        ))

    # Anomaly overlays by type
    for atype, color in TYPE_COLORS.items():
        if atype in ("normal", "unknown"):
            continue
        mask = anomalies["anomaly_type"].str.contains(atype, na=False)
        sub  = anomalies[mask]
        if sub.empty:
            continue
        fig3.add_trace(go.Scattergl(
            x=sub.index, y=sub[sensor_choice],
            mode="markers",
            marker=dict(size=8, color=color, symbol="circle-open", line=dict(width=2)),
            name=atype,
        ))

    fig3.update_layout(
        height=450, template="plotly_white",
        xaxis_title="Time",
        yaxis_title=f"{SENSOR_LABELS[sensor_choice]} ({SENSOR_UNITS[sensor_choice]})",
        hovermode="x unified",
        margin=dict(l=60, r=20, t=30, b=60),
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Stats table
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Normal statistics**")
        norm_vals = result_df.loc[~result_df["final_anomaly"], sensor_choice]
        st.dataframe(norm_vals.describe().rename("value").to_frame(), use_container_width=True)
    with c2:
        st.markdown("**Anomaly statistics**")
        anom_vals = result_df.loc[result_df["final_anomaly"], sensor_choice]
        if not anom_vals.empty:
            st.dataframe(anom_vals.describe().rename("value").to_frame(), use_container_width=True)
        else:
            st.info("No anomalies with current settings.")


# ════════════════════════════════════════════════════════════════════════════
#  TAB 4 — Detector Comparison
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Detector Agreement")

    bool_cols = [
        "vib_zscore", "vib_iqr", "pressure_drop", "rpm_flatline",
        "temp_zscore", "if_anomaly", "lof_anomaly", "pca_anomaly",
        "roll_anomaly", "collective_anomaly",
    ]
    bool_cols = [c for c in bool_cols if c in result_df.columns]

    col_a, col_b = st.columns([3, 1])

    with col_b:
        resample_period = st.radio(
            "Bin size", options=["15min", "30min", "1H"], index=0
        )

    with col_a:
        bin_df = result_df[bool_cols].resample(resample_period).mean()

        fig4 = px.imshow(
            bin_df.T,
            color_continuous_scale="YlOrRd",
            zmin=0, zmax=1,
            labels=dict(color="Anomaly fraction"),
            aspect="auto",
        )
        fig4.update_layout(
            height=380, template="plotly_white",
            title=f"Detector Heatmap ({resample_period} bins)",
            xaxis_title="Time",
            yaxis_title="Detector",
            margin=dict(l=140, r=20, t=50, b=60),
            coloraxis_colorbar=dict(title="Fraction"),
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Vote distribution bar chart
    st.markdown("#### Vote Distribution")
    st.caption("How many detectors agreed at each anomalous timestamp.")
    vote_counts = result_df["anomaly_votes"].value_counts().sort_index().reset_index()
    vote_counts.columns = ["votes", "count"]
    fig5 = px.bar(
        vote_counts, x="votes", y="count",
        color="votes",
        color_continuous_scale="Blues",
        labels={"votes": "Detector votes", "count": "Timestamp count"},
        text="count",
    )
    fig5.add_vline(
        x=vote_threshold - 0.5,
        line_dash="dash", line_color="red",
        annotation_text=f"Threshold = {vote_threshold}",
        annotation_position="top right",
    )
    fig5.update_layout(
        height=320, template="plotly_white",
        showlegend=False,
        margin=dict(l=60, r=20, t=30, b=60),
    )
    st.plotly_chart(fig5, use_container_width=True)

    # Per-detector counts table
    st.markdown("#### Per-Detector Counts")
    det_counts = {c: int(result_df[c].sum()) for c in bool_cols}
    det_df = pd.DataFrame(
        {"Detector": list(det_counts.keys()), "Flagged": list(det_counts.values())}
    ).sort_values("Flagged", ascending=False)
    det_df["% of total"] = (det_df["Flagged"] / total * 100).round(2)
    st.dataframe(det_df.set_index("Detector"), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 5 — Analysis
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Contextual Analysis")

    # ── Power vs RPM scatter ──────────────────────────────────────────────
    st.markdown("#### Power vs Rotation — Contextual Anomalies")
    st.caption(
        "Normal operation follows a tight linear band. "
        "Points outside it indicate the pump is consuming more power than expected "
        "for its current rotation speed."
    )

    fig6 = go.Figure()
    fig6.add_trace(go.Scattergl(
        x=normals["rotation_rpm"], y=normals["power_kw"],
        mode="markers",
        marker=dict(size=3, color="#bdc3c7", opacity=0.3),
        name="normal",
    ))
    for atype, color in TYPE_COLORS.items():
        if atype in ("normal", "unknown"):
            continue
        mask = anomalies["anomaly_type"].str.contains(atype, na=False)
        sub  = anomalies[mask]
        if sub.empty:
            continue
        fig6.add_trace(go.Scattergl(
            x=sub["rotation_rpm"], y=sub["power_kw"],
            mode="markers",
            marker=dict(size=8, color=color, opacity=0.85),
            name=atype,
        ))
    fig6.update_layout(
        height=420, template="plotly_white",
        xaxis_title="Rotation (RPM)",
        yaxis_title="Power (kW)",
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=60, r=20, t=30, b=80),
    )
    st.plotly_chart(fig6, use_container_width=True)

    # ── Collective drift zoom ─────────────────────────────────────────────
    st.markdown("#### Collective Drift — Bearing Wear Detail")
    st.caption(
        "Shaded regions show sustained periods where BOTH temperature and vibration "
        "rise together — classic bearing wear signature."
    )

    bearing_mask = result_df["anomaly_type"].str.contains("collective_drift", na=False)

    fig7 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                         subplot_titles=["Temperature (°C)", "Vibration (mm/s)"])
    fig7.add_trace(go.Scattergl(
        x=result_df.index, y=result_df["temperature_c"],
        mode="lines", line=dict(width=0.8, color="#e74c3c"), name="temp",
    ), row=1, col=1)
    fig7.add_trace(go.Scattergl(
        x=result_df.index, y=result_df["vibration_mm_s"],
        mode="lines", line=dict(width=0.8, color="#2980b9"), name="vibration",
    ), row=2, col=1)

    # Shade bearing-wear regions
    in_region, start = False, None
    shapes = []
    for ts, flag in bearing_mask.items():
        if flag and not in_region:
            start = ts; in_region = True
        elif not flag and in_region:
            shapes.append(dict(
                type="rect", xref="x", yref="paper",
                x0=str(start), x1=str(ts), y0=0, y1=1,
                fillcolor="#f39c12", opacity=0.15, line_width=0,
            ))
            in_region = False
    if in_region:
        shapes.append(dict(
            type="rect", xref="x", yref="paper",
            x0=str(start), x1=str(result_df.index[-1]), y0=0, y1=1,
            fillcolor="#f39c12", opacity=0.15, line_width=0,
        ))

    fig7.update_layout(
        shapes=shapes,
        height=460, template="plotly_white",
        margin=dict(l=60, r=20, t=40, b=60),
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", y=-0.1),
    )
    st.plotly_chart(fig7, use_container_width=True)

    # ── Correlation heatmap ───────────────────────────────────────────────
    st.markdown("#### Sensor Correlation")
    corr = result_df[FEATURE_COLS].corr()
    fig8 = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=".2f",
        aspect="auto",
    )
    fig8.update_layout(
        height=380, template="plotly_white",
        title="Pearson Correlation between Sensors",
        margin=dict(l=100, r=20, t=50, b=60),
    )
    st.plotly_chart(fig8, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  TAB 6 — Data Table
# ════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("Anomaly Records")

    show_only_anomalies = st.checkbox("Show only anomalies", value=True)
    show_cols = FEATURE_COLS + ["anomaly_votes", "final_anomaly", "anomaly_type"]

    view_df = result_df[show_cols].copy()
    if show_only_anomalies:
        view_df = view_df[view_df["final_anomaly"]]

    st.markdown(f"**{len(view_df):,} rows shown**")

    # Colour the anomaly_type column
    def highlight_type(val):
        color = TYPE_COLORS.get(val.split(",")[0].strip(), "#ffffff")
        return f"background-color: {color}22; color: {color}; font-weight: bold"

    styled = view_df.style.map(highlight_type, subset=["anomaly_type"])
    st.dataframe(styled, use_container_width=True, height=500)

    # Download button
    csv_bytes = view_df.to_csv().encode("utf-8")
    st.download_button(
        label="⬇️ Download as CSV",
        data=csv_bytes,
        file_name="anomaly_predictions.csv",
        mime="text/csv",
    )
