# src/analysis/report.py
import sys
import pandas as pd
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

TARIFF_SELF        = 0.33
TARIFF_SURPLUS     = 0.16
TARIFF_GRID        = 0.33
SYSTEM_KWP         = 2.25   # ID003 real scale (AC rating). Was 15 kWp (SME-pipeline parity).
ANNUAL_CONSUMPTION = 25_000

def forecast_report():
    """14-day forecast report from optimizer output."""
    path = PROJECT_ROOT / "data" / "outputs" / "optimization_14day_summary.csv"
    df   = pd.read_csv(path, parse_dates=["date"])

    report = df[[
        "date", "generation_kwh", "consumption_kwh",
        "surplus_kwh", "deficit_kwh", "net_saving_gel", "saving_pct"
    ]].copy()
    report.columns = [
        "Date", "Generated(kWh)", "Consumed(kWh)",
        "Surplus(kWh)", "Deficit(kWh)", "Net Saving(GEL)", "Saving%"
    ]

    totals = pd.DataFrame([{
        "Date":            "TOTAL",
        "Generated(kWh)":  report["Generated(kWh)"].sum().round(2),
        "Consumed(kWh)":   report["Consumed(kWh)"].sum().round(2),
        "Surplus(kWh)":    report["Surplus(kWh)"].sum().round(2),
        "Deficit(kWh)":    report["Deficit(kWh)"].sum().round(2),
        "Net Saving(GEL)": report["Net Saving(GEL)"].sum().round(2),
        "Saving%":         report["Saving%"].mean().round(1),
    }])
    return pd.concat([report, totals], ignore_index=True)

def monthly_projection():
    """Full 12-month projection using PVGIS monthly yield."""
    pvgis = pd.read_csv(PROJECT_ROOT / "data" / "outputs" / "monthly_yield.csv")
    monthly_consumption = ANNUAL_CONSUMPTION / 12
    rows = []
    for _, r in pvgis.iterrows():
        gen          = r["pvgis_kwh"]
        cons         = monthly_consumption
        self_rate    = 0.45
        self_consumed = min(gen * self_rate, cons)
        surplus       = max(gen - self_consumed, 0)
        deficit       = max(cons - self_consumed, 0)
        saving_self    = self_consumed * TARIFF_SELF
        saving_surplus = surplus * TARIFF_SURPLUS
        grid_cost      = deficit * TARIFF_GRID
        net_saving     = saving_self + saving_surplus
        baseline_cost  = cons * TARIFF_GRID
        saving_pct     = (net_saving / baseline_cost * 100) if baseline_cost > 0 else 0
        rows.append({
            "Month":          r["month_name"],
            "Generated(kWh)": round(gen, 0),
            "Consumed(kWh)":  round(cons, 0),
            "Self-Used(kWh)": round(self_consumed, 0),
            "Surplus(kWh)":   round(surplus, 0),
            "Deficit(kWh)":   round(deficit, 0),
            "Saving(GEL)":    round(net_saving, 2),
            "Grid Cost(GEL)": round(grid_cost, 2),
            "Saving%":        round(saving_pct, 1),
        })
    df = pd.DataFrame(rows)
    totals = {
        "Month":          "ANNUAL",
        "Generated(kWh)": df["Generated(kWh)"].sum(),
        "Consumed(kWh)":  df["Consumed(kWh)"].sum(),
        "Self-Used(kWh)": df["Self-Used(kWh)"].sum(),
        "Surplus(kWh)":   df["Surplus(kWh)"].sum(),
        "Deficit(kWh)":   df["Deficit(kWh)"].sum(),
        "Saving(GEL)":    df["Saving(GEL)"].sum().round(2),
        "Grid Cost(GEL)": df["Grid Cost(GEL)"].sum().round(2),
        "Saving%":        df["Saving%"].mean().round(1),
    }
    return pd.concat([df, pd.DataFrame([totals])], ignore_index=True)

def run_reports():
    print("=" * 70)
    fc = forecast_report()
    print(fc.to_string(index=False))
    fc.to_csv(PROJECT_ROOT / "data" / "outputs" / "report_14day.csv", index=False)
    print(f"\nSaved → data/outputs/report_14day.csv")

    print("\n" + "=" * 70)
    monthly = monthly_projection()
    print(monthly.to_string(index=False))
    monthly.to_csv(PROJECT_ROOT / "data" / "outputs" / "report_monthly.csv", index=False)
    print(f"Saved → data/outputs/report_monthly.csv")
    return fc, monthly

if __name__ == "__main__":
    run_reports()
