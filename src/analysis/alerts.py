# src/analysis/alerts.py
import sys
import pandas as pd
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

THRESHOLDS = {
    # 20 mm/day = a genuinely heavy NL rain day. (10 mm flagged ordinary wet
    # days as HIGH.)
    "heavy_rain":      {"precip_mm": 20,  "severity": "HIGH",   "emoji": "🌧"},
    "storm":           {"wind_ms":   15,  "severity": "HIGH",   "emoji": "⛈"},
    "full_overcast":   {"clouds_pct": 85, "severity": "MEDIUM", "emoji": "☁"},
    "production_drop": {"drop_pct":   40, "severity": "MEDIUM", "emoji": "📉"},
    "heat_stress":     {"temp_c":     35, "severity": "LOW",    "emoji": "🌡"},
    "frost_risk":      {"temp_c":      2, "severity": "LOW",    "emoji": "❄"},
}

def load_forecast():
    path = PROJECT_ROOT / "data" / "outputs" / "forecast_14day_15min.csv"
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    return df

def generate_alerts(df):
    alerts = []
    df["date"] = pd.to_datetime(df["datetime"]).dt.date
    daily = df.groupby("date").agg(
        total_precip = ("precipitation", "sum"),
        max_wind     = ("wind_speed",    "max"),
        avg_clouds   = ("clouds",        "mean"),
        max_temp     = ("temp",          "max"),
        min_temp     = ("temp",          "min"),
        total_kwh    = ("soli_kwh_pred", "sum"),
    ).reset_index()

    avg_production = daily["total_kwh"].mean()

    for _, row in daily.iterrows():
        date_str = str(row["date"])
        if row["total_precip"] >= THRESHOLDS["heavy_rain"]["precip_mm"]:
            alerts.append({"date": date_str, "severity": "HIGH", "emoji": "🌧",
                "type": "Heavy Rain",
                "message": f"{row['total_precip']:.1f}mm expected — panel soiling risk, check drainage"})
        if row["max_wind"] >= THRESHOLDS["storm"]["wind_ms"]:
            alerts.append({"date": date_str, "severity": "HIGH", "emoji": "⛈",
                "type": "High Wind",
                "message": f"Wind up to {row['max_wind']:.1f} m/s — check panel mounting and inverter"})
        if row["avg_clouds"] >= THRESHOLDS["full_overcast"]["clouds_pct"]:
            alerts.append({"date": date_str, "severity": "MEDIUM", "emoji": "☁",
                "type": "Overcast Day",
                "message": f"{row['avg_clouds']:.0f}% avg cloud cover — shift heavy loads to grid"})
        if avg_production > 0:
            drop_pct = (avg_production - row["total_kwh"]) / avg_production * 100
            if drop_pct >= THRESHOLDS["production_drop"]["drop_pct"]:
                alerts.append({"date": date_str, "severity": "MEDIUM", "emoji": "📉",
                    "type": "Production Drop",
                    "message": f"{drop_pct:.0f}% below 14-day average — verify no system fault"})
        if row["max_temp"] >= THRESHOLDS["heat_stress"]["temp_c"]:
            alerts.append({"date": date_str, "severity": "LOW", "emoji": "🌡",
                "type": "Heat Stress",
                "message": f"Max temp {row['max_temp']:.1f}°C — panel efficiency reduced ~{(row['max_temp']-25)*0.4:.1f}%"})
        if row["min_temp"] <= THRESHOLDS["frost_risk"]["temp_c"]:
            alerts.append({"date": date_str, "severity": "LOW", "emoji": "❄",
                "type": "Frost Risk",
                "message": f"Min temp {row['min_temp']:.1f}°C — check inverter cold-start behaviour"})

    return pd.DataFrame(alerts) if alerts else pd.DataFrame(
        columns=["date", "severity", "emoji", "type", "message"])

def run_alerts():
    df        = load_forecast()
    alerts_df = generate_alerts(df)

    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    if not alerts_df.empty:
        alerts_df["_order"] = alerts_df["severity"].map(order)
        alerts_df = alerts_df.sort_values(["_order", "date"]).drop("_order", axis=1)

    high   = len(alerts_df[alerts_df["severity"] == "HIGH"])   if not alerts_df.empty else 0
    medium = len(alerts_df[alerts_df["severity"] == "MEDIUM"]) if not alerts_df.empty else 0
    low    = len(alerts_df[alerts_df["severity"] == "LOW"])    if not alerts_df.empty else 0
    print(f"Alerts — HIGH: {high}  MEDIUM: {medium}  LOW: {low}")

    out = PROJECT_ROOT / "data" / "outputs" / "alerts_14day.csv"
    alerts_df.to_csv(out, index=False)
    print(f"Saved → data/outputs/alerts_14day.csv")
    return alerts_df

if __name__ == "__main__":
    run_alerts()
