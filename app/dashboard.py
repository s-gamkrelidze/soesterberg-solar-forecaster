# app/dashboard.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="LTD Soli — Solar Dashboard",
    page_icon="☀️",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ System Configuration")
    st.markdown("---")
    system_kwp  = st.number_input("System Capacity (kWp)",       value=15.0,    step=0.5)
    tilt        = st.number_input("Panel Tilt (°)",               value=35,      step=1)
    azimuth     = st.number_input("Panel Azimuth (°)",            value=180,     step=1)
    consumption = st.number_input("Annual Consumption (kWh)",     value=25000,   step=500)
    tariff_self = st.number_input("Self-use Tariff (GEL/kWh)",   value=0.33,    step=0.01)
    tariff_surp = st.number_input("Surplus Tariff (GEL/kWh)",    value=0.16,    step=0.01)
    st.markdown("---")
    st.markdown("🔋 **Battery storage** *(set in `optimizer.py`)*")
    st.markdown(
        "- Capacity: **10 kWh**\n"
        "- Power limit: **5 kW**\n"
        "- Round-trip eff: **~90%**\n"
        "- Reserve floor: **10%**"
    )
    st.caption("Greedy self-consumption dispatch — optimal under the flat "
               "import tariff. Edit constants in optimizer.py and re-run the "
               "pipeline to change.")
    st.markdown("---")
    st.markdown("📍 **Location:** Tbilisi, Georgia")
    st.markdown("🏭 **SME:** LTD Soli")
    st.markdown("📅 **Forecast horizon:** 14 days")
    st.markdown("📅 **Model trained:** 2024–2025")

# ── LOAD DATA ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    files = {
        "monthly":     "data/outputs/report_monthly.csv",
        "forecast":    "data/outputs/forecast_14day_15min.csv",
        "alerts":      "data/outputs/alerts_14day.csv",
        "optim":       "data/outputs/optimization_14day_summary.csv",
        "optim_slots": "data/outputs/optimization_14day.csv",
        "optim_best":  "data/outputs/optimization_best_hours.csv",
        "report14":    "data/outputs/report_14day.csv",
    }
    data = {}
    for key, path in files.items():
        full = PROJECT_ROOT / path
        data[key] = pd.read_csv(full) if full.exists() else pd.DataFrame()
    return data

data = load_data()

# ── ALERT BAR ─────────────────────────────────────────────
if not data["alerts"].empty:
    high   = data["alerts"][data["alerts"]["severity"] == "HIGH"]
    medium = data["alerts"][data["alerts"]["severity"] == "MEDIUM"]
    if not high.empty:
        for _, a in high.iterrows():
            st.error(f"{a['emoji']} **{a['date']} — {a['type']}:** {a['message']}")
    elif not medium.empty:
        for _, a in medium.iterrows():
            st.warning(f"{a['emoji']} **{a['date']} — {a['type']}:** {a['message']}")
    else:
        st.success("✅ No active alerts — all systems normal")
else:
    st.success("✅ No active alerts — all systems normal")

# ── TABS ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🌤 14-Day Forecast",
    "⚡ Optimization",
    "📋 Reports"
])

# ══════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════
with tab1:
    st.header("Annual Yield & Financial Overview")
    st.markdown(f"**System:** {system_kwp} kWp · Tbilisi · {tilt}° tilt · {azimuth}° azimuth")

    annual_row = data["monthly"][data["monthly"]["Month"] == "ANNUAL"]
    if not annual_row.empty:
        annual_gen  = annual_row["Generated(kWh)"].values[0]
        annual_save = annual_row["Saving(GEL)"].values[0]
        self_used   = annual_row["Self-Used(kWh)"].values[0]
        coverage    = round(self_used / consumption * 100, 1)
        payback     = round((system_kwp * 1000 * 0.55) / annual_save, 1)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Annual Generation", f"{annual_gen:,.0f} kWh")
        col2.metric("Annual Saving",     f"{annual_save:,.0f} GEL")
        col3.metric("Self-Consumption",  f"{self_used:,.0f} kWh")
        col4.metric("Coverage Ratio",    f"{coverage}%")
        col5.metric("Est. Payback",      f"{payback} yrs")

    st.markdown("---")
    monthly = data["monthly"][data["monthly"]["Month"] != "ANNUAL"].copy()

    if not monthly.empty:
        # Chart 1: Monthly generation + savings (dual axis)
        st.subheader("📊 Monthly Generation vs. Savings")
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=monthly["Month"], y=monthly["Generated(kWh)"],
            name="Generated (kWh)", marker_color="#FFA500", yaxis="y1"
        ))
        fig1.add_trace(go.Scatter(
            x=monthly["Month"], y=monthly["Saving(GEL)"],
            name="Saving (GEL)", mode="lines+markers",
            marker=dict(size=10, color="#2E8B57"),
            line=dict(width=3, color="#2E8B57"), yaxis="y2"
        ))
        fig1.update_layout(
            xaxis=dict(title="Month"),
            yaxis=dict(title="Generation (kWh)", side="left"),
            yaxis2=dict(title="Saving (GEL)", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", y=-0.2),
            height=400, margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Chart 2: Stacked self-use vs surplus
        st.subheader("🔋 Monthly Self-Use vs Surplus Export")
        fig_stack = go.Figure()
        fig_stack.add_trace(go.Bar(
            x=monthly["Month"], y=monthly["Self-Used(kWh)"],
            name="Self-Used (0.33 GEL/kWh)", marker_color="#2E8B57"
        ))
        fig_stack.add_trace(go.Bar(
            x=monthly["Month"], y=monthly["Surplus(kWh)"],
            name="Surplus Export (0.16 GEL/kWh)", marker_color="#87CEEB"
        ))
        fig_stack.update_layout(
            barmode="stack",
            xaxis=dict(title="Month"),
            yaxis=dict(title="Energy (kWh)"),
            legend=dict(orientation="h", y=-0.2),
            height=350, margin=dict(l=20, r=20, t=20, b=20),
        )
        fig_stack.add_annotation(
            text="Self-use saves 2× more per kWh than export",
            xref="paper", yref="paper", x=0.5, y=1.02,
            showarrow=False, font=dict(size=11, color="#555")
        )
        st.plotly_chart(fig_stack, use_container_width=True)

        st.dataframe(monthly, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Chart 3: ML Feature importance
    st.subheader("🧠 Top ML Features — What Drives Solar Production")
    try:
        import joblib
        model = joblib.load(PROJECT_ROOT / "data" / "models" / "rf_model.pkl")
        FEATURES = [
            "solar_radiation", "effective_radiation", "adjusted_radiation",
            "clouds", "temp", "humidity", "wind_speed",
            "solar_elevation", "solar_elevation_sin", "solar_elevation_cos",
            "cos_incidence", "thermal_factor", "temp_delta_25",
            "wind_cooling_effect", "clouds_x_elevation", "humidity_x_elevation",
            "humidity_radiation", "hours_since_rain", "rain_last_24h",
            "dew_spread", "solar_lag_1h", "solar_lag_24h", "cloud_lag_1h",
            "clouds_3h_mean", "radiation_3h_mean", "cloud_trend_3h",
            "hour", "month", "dayofyear",
        ]
        imp = pd.DataFrame({
            "feature": FEATURES,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=True).tail(10)
        fig2 = px.bar(imp, x="importance", y="feature", orientation="h",
                      color="importance", color_continuous_scale="YlOrRd",
                      labels={"importance": "Importance", "feature": "Feature"})
        fig2.update_layout(height=400, showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Solar geometry (`solar_elevation_*`, `hour`) dominates — "
                   "weather features (clouds, humidity) refine on top.")
    except Exception:
        st.info("Feature importance chart unavailable.")

# ══════════════════════════════════════════════════════════
# TAB 2 — 14-DAY FORECAST
# ══════════════════════════════════════════════════════════
with tab2:
    st.header("14-Day Solar Production Forecast")

    if st.button("🔄 Refresh Forecast"):
        st.cache_data.clear()
        st.rerun()

    df_fc = data["forecast"]

    if not df_fc.empty:
        df_fc = df_fc.copy()
        df_fc["dt"] = pd.to_datetime(df_fc["date"].astype(str) + " " + df_fc["time"].astype(str))
        daily_fc = df_fc.groupby(df_fc["dt"].dt.date)["soli_kwh_pred"].sum().reset_index()
        daily_fc.columns = ["Date", "Predicted kWh"]
        daily_fc["Predicted kWh"] = daily_fc["Predicted kWh"].round(2)

        total_fc  = daily_fc["Predicted kWh"].sum()
        avg_daily = daily_fc["Predicted kWh"].mean()
        best_day  = daily_fc.loc[daily_fc["Predicted kWh"].idxmax()]

        col1, col2, col3 = st.columns(3)
        col1.metric("14-Day Total",  f"{total_fc:.1f} kWh")
        col2.metric("Daily Average", f"{avg_daily:.1f} kWh")
        col3.metric("Best Day",      f"{best_day['Predicted kWh']:.1f} kWh", str(best_day["Date"]))

        st.markdown("---")

        # Hourly production area chart
        st.subheader("⚡ 14-Day Hourly Production")
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(
            x=df_fc["dt"], y=df_fc["soli_kwh_pred"],
            mode="lines", name="Predicted kWh (15-min)",
            line=dict(color="#FFA500", width=2),
            fill="tozeroy", fillcolor="rgba(255,165,0,0.2)"
        ))
        fig_h.update_layout(
            xaxis=dict(title="Date / Time"),
            yaxis=dict(title="Production (kWh per 15-min slot)"),
            height=400, margin=dict(l=20, r=20, t=20, b=20),
            hovermode="x unified",
        )
        st.plotly_chart(fig_h, use_container_width=True)

        # Daily totals bar
        st.subheader("📅 Daily Totals")
        fig_d = px.bar(daily_fc, x="Date", y="Predicted kWh",
                       color="Predicted kWh", color_continuous_scale="YlOrRd",
                       text="Predicted kWh")
        fig_d.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_d.update_layout(height=350, showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_d, use_container_width=True)

        with st.expander("📋 Show hourly forecast table"):
            display_cols = [c for c in [
                "date", "time", "soli_kwh_pred",
                "solar_radiation", "clouds", "temp",
                "precipitation", "wind_speed", "weather"
            ] if c in df_fc.columns]
            st.dataframe(df_fc[display_cols], use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════
# TAB 3 — OPTIMIZATION
# ══════════════════════════════════════════════════════════
with tab3:
    st.header("14-Day Energy Optimization Recommendations")

    df_opt = data["optim"]

    if not df_opt.empty:
        total_saving = df_opt["net_saving_gel"].sum()
        total_gen    = df_opt["generation_kwh"].sum()
        avg_saving   = df_opt["saving_pct"].mean()

        col1, col2, col3 = st.columns(3)
        col1.metric("14-Day Generation", f"{total_gen:.1f} kWh")
        col2.metric("14-Day Saving",     f"{total_saving:.2f} GEL")
        col3.metric("Avg Cost Reduction", f"{avg_saving:.1f}%")

        st.markdown("---")

        st.subheader("📊 Daily Generation vs Consumption vs Savings")
        fig_opt = go.Figure()
        fig_opt.add_trace(go.Bar(
            x=df_opt["date"], y=df_opt["generation_kwh"],
            name="Generation (kWh)", marker_color="#FFA500"
        ))
        fig_opt.add_trace(go.Bar(
            x=df_opt["date"], y=df_opt["consumption_kwh"],
            name="Consumption (kWh)", marker_color="#4A90D9"
        ))
        fig_opt.add_trace(go.Scatter(
            x=df_opt["date"], y=df_opt["net_saving_gel"],
            name="Net Saving (GEL)", mode="lines+markers",
            marker=dict(size=10, color="#2E8B57"),
            line=dict(width=3, color="#2E8B57"), yaxis="y2"
        ))
        fig_opt.update_layout(
            barmode="group",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Energy (kWh)", side="left"),
            yaxis2=dict(title="Saving (GEL)", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", y=-0.2),
            height=400, margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_opt, use_container_width=True)

        st.markdown("---")
        st.subheader("📋 Daily Optimization Summary")
        st.dataframe(df_opt, use_container_width=True, hide_index=True)

        # ── BATTERY STORAGE ───────────────────────────────────
        if "net_saving_batt_gel" in df_opt.columns:
            st.markdown("---")
            st.subheader("🔋 Battery Storage — Charge / Discharge Schedule")

            batt_saving = df_opt["net_saving_batt_gel"].sum()
            batt_extra  = df_opt["battery_extra_gel"].sum()
            batt_cycled = df_opt["batt_discharge_kwh"].sum()
            self_cons   = (df_opt["self_use_kwh"].sum()
                           / df_opt["consumption_kwh"].sum() * 100)

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Saving (Solar + Battery)", f"{batt_saving:.2f} GEL")
            b2.metric("Battery Adds", f"+{batt_extra:.2f} GEL",
                      help="Extra saving on top of solar-only self-use, from "
                           "time-shifting surplus into evening deficit slots.")
            b3.metric("Energy Cycled", f"{batt_cycled:.1f} kWh",
                      help="Total energy discharged from the battery to the load "
                           "over 14 days.")
            b4.metric("Self-Consumption", f"{self_cons:.1f}%",
                      help="Share of consumption covered by PV directly + via the "
                           "battery, instead of the grid.")

            df_slots = data["optim_slots"]
            if not df_slots.empty and "batt_soc_pct" in df_slots.columns:
                df_slots = df_slots.copy()
                df_slots["dt"] = pd.to_datetime(
                    df_slots["date"].astype(str) + " " + df_slots["time"].astype(str))

                st.markdown("**State of Charge & Battery Power (per 15-min slot)**")
                fig_batt = go.Figure()
                fig_batt.add_trace(go.Bar(
                    x=df_slots["dt"], y=df_slots["batt_charge_kwh"],
                    name="Charge (kWh)", marker_color="#2E8B57"
                ))
                fig_batt.add_trace(go.Bar(
                    x=df_slots["dt"], y=-df_slots["batt_discharge_kwh"],
                    name="Discharge (kWh)", marker_color="#E8743B"
                ))
                fig_batt.add_trace(go.Scatter(
                    x=df_slots["dt"], y=df_slots["batt_soc_pct"],
                    name="State of Charge (%)", mode="lines",
                    line=dict(color="#4A90D9", width=2), yaxis="y2"
                ))
                fig_batt.update_layout(
                    barmode="relative",
                    xaxis=dict(title="Date / Time"),
                    yaxis=dict(title="Battery power (kWh per slot)", side="left"),
                    yaxis2=dict(title="SoC (%)", overlaying="y", side="right",
                                range=[0, 100], showgrid=False),
                    legend=dict(orientation="h", y=-0.2),
                    height=400, margin=dict(l=20, r=20, t=20, b=20),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_batt, use_container_width=True)
                st.caption("Green bars charge the battery from PV surplus; orange "
                           "bars discharge it to cover load. The blue line is the "
                           "battery's state of charge (10% reserve floor → 100% full).")

                with st.expander("📋 Show per-slot battery & load schedule"):
                    sched_cols = [c for c in [
                        "date", "time", "pv_adjusted", "consumption_kwh",
                        "batt_charge_kwh", "batt_discharge_kwh", "batt_soc_pct",
                        "grid_import_kwh", "grid_export_kwh", "action", "batt_action"
                    ] if c in df_slots.columns]
                    st.dataframe(df_slots[sched_cols], use_container_width=True,
                                 hide_index=True)

        # ── BEST LOAD HOURS ───────────────────────────────────
        df_best = data["optim_best"]
        if not df_best.empty:
            st.markdown("---")
            st.subheader("⏰ Best Hours to Run Heavy Loads (top 3 per day)")
            st.caption("Highest risk-adjusted PV slots — schedule ovens, "
                       "compressors and other discretionary loads here.")
            st.dataframe(df_best, use_container_width=True, hide_index=True)

        st.info("Run heavy equipment (ovens, compressors) during ⚡ and ✅ hours. "
                "Shift discretionary loads away from 🔌 hours. "
                "The battery stores midday surplus and releases it into evening "
                "deficit slots — every kWh shifted from grid to solar saves 0.17 GEL.")

# ══════════════════════════════════════════════════════════
# TAB 4 — REPORTS
# ══════════════════════════════════════════════════════════
with tab4:
    st.header("Reports & Export")

    st.subheader("📅 14-Day Forecast Report")
    if not data["report14"].empty:
        st.dataframe(data["report14"], use_container_width=True, hide_index=True)
        csv = data["report14"].to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download 14-Day Report", csv, "report_14day.csv", "text/csv")

    st.markdown("---")
    st.subheader("📆 Monthly Projection (full year)")
    if not data["monthly"].empty:
        st.dataframe(data["monthly"], use_container_width=True, hide_index=True)
        csv2 = data["monthly"].to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Monthly Projection", csv2, "monthly_projection.csv", "text/csv")

    st.markdown("---")
    st.subheader("🚨 Active Alerts (14-day)")
    if not data["alerts"].empty:
        st.dataframe(data["alerts"], use_container_width=True, hide_index=True)
    else:
        st.success("No alerts for this period.")

    st.markdown("---")
    st.caption("LTD Soli Solar Forecasting System · MBA Thesis Prototype · "
               "Tbilisi, Georgia · Model: Random Forest R²=0.87 · Forecast: 14 days")
