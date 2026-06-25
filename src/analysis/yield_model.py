# src/analysis/yield_model.py
import sys
import requests
import pandas as pd
import numpy as np
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SYSTEM_KWP             = 2.25   # ID003 real scale (AC rating). Was 15 kWp (SME-pipeline parity).
TILT                   = 40
AZIMUTH                = 203
PERF_RATIO             = 0.80
PANEL_EFF              = 0.20
PANEL_AREA_M2          = SYSTEM_KWP * 1000 / (PANEL_EFF * 1000)
LAT_GE, LON_GE         = 52.1088, 5.1253   # was 41.71, 44.78 (Tbilisi) — now Soesterberg NL
TARIFF_SELF            = 0.33
TARIFF_SURPLUS         = 0.16
ANNUAL_CONSUMPTION_KWH = 25_000

def pvgis_annual_yield(lat, lon, system_kwp, tilt, azimuth, perf_ratio):
    print("Calling PVGIS API for Soesterberg NL ...")
    url = "https://re.jrc.ec.europa.eu/api/v5_2/PVcalc"
    params = {
        "lat": lat, "lon": lon, "peakpower": system_kwp,
        "angle": tilt, "aspect": azimuth - 180,
        "loss": (1 - perf_ratio) * 100,
        "outputformat": "json", "browser": 0
    }
    r = requests.get(url, params=params, timeout=30)
    data = r.json()
    if "outputs" not in data:
        raise ValueError(f"PVGIS error: {data}")
    totals     = data["outputs"]["totals"]["fixed"]
    annual_kwh = totals["E_y"]
    h_in_plane = totals["H(i)_y"]           # yearly in-plane irradiation, kWh/m²/yr
    monthly    = data["outputs"]["monthly"]["fixed"]
    df_monthly = pd.DataFrame(monthly)[["month", "E_m"]].rename(columns={"E_m": "pvgis_kwh"})
    df_monthly["month_name"] = pd.to_datetime(df_monthly["month"], format="%m").dt.strftime("%B")
    print(f"  PVGIS annual yield:    {annual_kwh:,.0f} kWh/year")
    print(f"  PVGIS in-plane irrad.: {h_in_plane:,.0f} kWh/m²/yr")
    return annual_kwh, df_monthly, h_in_plane

def physics_annual_yield(system_kwp, panel_area, panel_eff, h_tilted, perf_ratio=0.80):
    # h_tilted MUST be the site's in-plane irradiation. Do not hardcode it: the
    # old default (1850) was a southern-latitude value left over from the Tbilisi
    # project and overstated NL yield ~1.6×. We now pass PVGIS's H(i)_y for the site.
    kwh = panel_area * panel_eff * h_tilted * perf_ratio
    print(f"  Physics cross-check:   {kwh:,.0f} kWh/year  (h_tilted={h_tilted:,.0f})")
    return kwh

def compute_financials(annual_gen_kwh, annual_consumption_kwh, tariff_self, tariff_surplus):
    self_consume_rate = 0.45
    self_consumed_kwh = min(annual_gen_kwh * self_consume_rate, annual_consumption_kwh)
    surplus_kwh       = max(annual_gen_kwh - self_consumed_kwh, 0)
    grid_kwh          = max(annual_consumption_kwh - self_consumed_kwh, 0)
    savings_self      = self_consumed_kwh * tariff_self
    savings_surplus   = surplus_kwh * tariff_surplus
    total_savings     = savings_self + savings_surplus
    coverage_pct      = (self_consumed_kwh / annual_consumption_kwh) * 100
    return {
        "annual_generation_kwh":    round(annual_gen_kwh, 0),
        "self_consumed_kwh":        round(self_consumed_kwh, 0),
        "surplus_kwh":              round(surplus_kwh, 0),
        "grid_kwh_remaining":       round(grid_kwh, 0),
        "savings_self_gel":         round(savings_self, 2),
        "savings_surplus_gel":      round(savings_surplus, 2),
        "total_annual_savings_gel": round(total_savings, 2),
        "coverage_pct":             round(coverage_pct, 1),
    }

def run_yield_model():
    pvgis_kwh, df_monthly, h_in_plane = pvgis_annual_yield(
        LAT_GE, LON_GE, SYSTEM_KWP, TILT, AZIMUTH, PERF_RATIO)
    # Physics is a consistency check (it uses the SAME site irradiation PVGIS
    # computed), not an independent estimate — so we report PVGIS as the yield
    # rather than averaging two numbers that should agree by construction.
    physics_kwh = physics_annual_yield(
        SYSTEM_KWP, PANEL_AREA_M2, PANEL_EFF, h_tilted=h_in_plane, perf_ratio=PERF_RATIO)
    agreement = physics_kwh / pvgis_kwh * 100 if pvgis_kwh else float("nan")
    print(f"  Physics/PVGIS agreement: {agreement:.0f}%")
    if abs(agreement - 100) > 15:
        print("  ⚠ Physics and PVGIS disagree >15% — check SYSTEM_KWP / PERF_RATIO / panel area.")

    final_kwh = pvgis_kwh
    print(f"\n  Annual yield estimate (PVGIS, authoritative): {final_kwh:,.0f} kWh/year")
    fin = compute_financials(final_kwh, ANNUAL_CONSUMPTION_KWH, TARIFF_SELF, TARIFF_SURPLUS)
    out = PROJECT_ROOT / "data" / "outputs" / "monthly_yield.csv"
    df_monthly.to_csv(out, index=False)
    print(f"Saved → data/outputs/monthly_yield.csv")
    return fin, df_monthly

if __name__ == "__main__":
    run_yield_model()
