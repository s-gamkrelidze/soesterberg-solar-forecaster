# src/inference/forecast_climate.py
# Weekly solar generation forecast: week 2 → week 52 ahead.
#
# Three-phase output:
#   Week 1  (days 1–7)   : covered by forecast_pv.py (14-day model) — not duplicated here
#   Weeks 2–26 (days 8–182) : ECMWF SEAS5 via Open-Meteo Seasonal API → physics → kWh
#   Weeks 27–52 (days 183–365): historical climatology (5-year percentiles, fallback)
#
# SEAS5 path:
#   clearsky_index = seas5_radiation / pvlib_clearsky
#   kwh_day = clearsky_index × peak_sun_hours × SYSTEM_KWP × PVGIS_CALIB
#   Then scaled by Model 2 monthly correction (two_stage_model2.pkl) if available.
#
# Reads:  Open-Meteo Archive API (cached), Open-Meteo Seasonal API (SEAS5)
# Writes: data/outputs/forecast_365day_weekly.csv
#         data/raw/site_historical_weather.csv
#
# Dependencies: pip install requests pvlib pandas numpy joblib

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import requests
import numpy as np
import pandas as pd
import pvlib
import joblib
from pathlib import Path
from datetime import timedelta, date

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
try:
    from ingestion.fetch_seas5_om import fetch_seas5_weekly
    _SEAS5_AVAILABLE = True
except ImportError:
    _SEAS5_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── SITE CONFIG ───────────────────────────────────────────
SITE_LAT       = 52.1088
SITE_LON       = 5.1253
SITE_ALT_M     = 15            # Soesterberg altitude (m above sea level, approx)
SYSTEM_KWP     = 2.25   # ID003 real scale (AC rating). Was 15 kWp (SME-pipeline parity).
PVGIS_CALIB    = 1.0     # was 0.6281 (NL→Tbilisi). NL-only mode: set to 1.0.
SITE_TZ_STR    = "UTC"
HISTORY_YEARS  = 5      # years of ERA5-equivalent data to base the normals on
CACHE_MAX_DAYS = 30     # refresh cache if older than this many days


# ── HISTORICAL WEATHER FETCH ──────────────────────────────
def fetch_site_history(years: int = HISTORY_YEARS) -> pd.DataFrame:
    """
    Fetch daily historical weather for the configured site from Open-Meteo Archive API.
    Results are cached to data/raw/site_historical_weather.csv.
    Cache is reused if it's less than CACHE_MAX_DAYS old.

    Variables returned:
        date, radiation_sum_wh, temp_mean, cloud_cover, precip_mm
    """
    cache_path = PROJECT_ROOT / "data" / "raw" / "site_historical_weather.csv"
    end_date   = date.today()
    start_date = end_date.replace(year=end_date.year - years)

    # Use cache if it exists and is fresh
    if cache_path.exists():
        cached    = pd.read_csv(cache_path, parse_dates=["date"])
        cache_end = cached["date"].max().date()
        age_days  = (end_date - cache_end).days
        if age_days < CACHE_MAX_DAYS:
            print(f"  Using cached historical data ({age_days} days old) — {cache_path.name}")
            return cached
        print(f"  Cache is {age_days} days old — refreshing...")

    print(f"Fetching {years} years of site weather from Open-Meteo Archive API...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   SITE_LAT,
        "longitude":  SITE_LON,
        "start_date": start_date.isoformat(),
        "end_date":   end_date.isoformat(),
        "daily": [
            "shortwave_radiation_sum",
            "temperature_2m_mean",
            "cloud_cover_mean",
            "precipitation_sum",
        ],
        "timezone": SITE_TZ_STR,
    }

    try:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
    except requests.RequestException as e:
        if cache_path.exists():
            print(f"  API request failed ({e}). Using stale cache.")
            return pd.read_csv(cache_path, parse_dates=["date"])
        raise RuntimeError(f"Open-Meteo Archive API request failed and no cache exists: {e}")

    data  = r.json()
    daily = data["daily"]

    df = pd.DataFrame({
        "date":             pd.to_datetime(daily["time"]),
        "radiation_sum_mj": daily.get("shortwave_radiation_sum", []),
        "temp_mean":        daily.get("temperature_2m_mean", []),
        "cloud_cover":      daily.get("cloud_cover_mean", []),
        "precip_mm":        daily.get("precipitation_sum", []),
    })

    # MJ/m²/day → Wh/m²/day  (1 MJ = 277.78 Wh)
    df["radiation_sum_wh"] = (pd.to_numeric(df["radiation_sum_mj"], errors="coerce") * 277.78).clip(lower=0)
    df = df.drop(columns=["radiation_sum_mj"])
    df = df.dropna(subset=["radiation_sum_wh"]).reset_index(drop=True)

    print(f"  Retrieved {len(df)} days ({df['date'].min().date()} → {df['date'].max().date()})")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"  Cached → {cache_path.relative_to(PROJECT_ROOT)}")
    return df


# ── CLEAR-SKY DAILY ───────────────────────────────────────
def compute_daily_clearsky_wh(dates: pd.DatetimeIndex) -> pd.Series:
    """
    Daily clear-sky GHI (Wh/m²) for the site using pvlib Ineichen model.
    Returns Series indexed by timezone-naive date.
    """
    location = pvlib.location.Location(
        latitude=SITE_LAT, longitude=SITE_LON,
        altitude=SITE_ALT_M, tz=SITE_TZ_STR,
    )
    start = pd.Timestamp(dates.min()).normalize()
    end   = pd.Timestamp(dates.max()).normalize() + timedelta(days=1)
    hours = pd.date_range(start=start, end=end, freq="1h", tz=SITE_TZ_STR)

    cs       = location.get_clearsky(hours, model="ineichen")
    cs_daily = cs["ghi"].resample("D").sum()
    cs_daily.index = cs_daily.index.tz_localize(None).normalize()
    return cs_daily


# ── DAILY kWh ESTIMATION ──────────────────────────────────
def estimate_daily_kwh(hist: pd.DataFrame) -> pd.DataFrame:
    """
    Convert historical daily radiation → estimated kWh.
    Consistent formula with forecast_medium.py:
        kWh = clearsky_index × peak_sun_hours × SYSTEM_KWP × PVGIS_CALIB
    """
    print("  Estimating daily kWh from historical radiation (pvlib clearsky)...")
    hist      = hist.copy()
    dates_idx = pd.DatetimeIndex(hist["date"])
    cs_daily  = compute_daily_clearsky_wh(dates_idx)

    hist["clearsky_wh"]  = hist["date"].map(lambda d: cs_daily.get(d.normalize(), np.nan))
    hist["clearsky_index"] = (
        hist["radiation_sum_wh"] / hist["clearsky_wh"].clip(lower=1)
    ).clip(0, 1.2)
    hist["peak_sun_hours"] = hist["clearsky_wh"] / 1000.0
    hist["kwh_est"]        = (
        hist["clearsky_index"] * hist["peak_sun_hours"] * SYSTEM_KWP * PVGIS_CALIB
    ).clip(lower=0)

    print(f"  Mean daily kWh (all years): {hist['kwh_est'].mean():.2f}")
    return hist


# ── WEEKLY NORMALS ────────────────────────────────────────
def build_weekly_normals(hist: pd.DataFrame) -> pd.DataFrame:
    """
    Compute week-of-year statistics from 5 years of historical daily data.
    Uses ISO week number (1–53).

    Output columns per week:
        kwh_p10/p25/p50/p75/p90  — daily kWh percentiles
        cloud_mean                — mean cloud cover (%)
        temp_mean                 — mean temperature (°C)
        n_days                    — number of historical days in this week
    """
    hist = hist.copy()
    hist["week_of_year"] = hist["date"].dt.isocalendar().week.astype(int)

    def pct(n):
        return lambda x: float(np.nanpercentile(x.dropna(), n)) if len(x.dropna()) > 0 else np.nan

    normals = hist.groupby("week_of_year").agg(
        kwh_p10    = ("kwh_est",    pct(10)),
        kwh_p25    = ("kwh_est",    pct(25)),
        kwh_p50    = ("kwh_est",    pct(50)),
        kwh_p75    = ("kwh_est",    pct(75)),
        kwh_p90    = ("kwh_est",    pct(90)),
        cloud_mean = ("cloud_cover", "mean"),
        temp_mean  = ("temp_mean",   "mean"),
        n_days     = ("kwh_est",     "count"),
    ).reset_index()

    return normals


# ── FORECAST TABLE ────────────────────────────────────────
def build_forecast_table(normals: pd.DataFrame) -> pd.DataFrame:
    """
    Map the next 90–365 days onto historical week-of-year normals.
    Each row = one calendar week in the future (Monday → Sunday).

    Columns:
        week_start, week_end, week_of_year, days_ahead, season
        kwh_day_p10/p25/p50/p75/p90   — expected daily kWh
        kwh_week_p50                   — weekly total (p50 × 7)
        cloud_mean_pct, temp_mean_c
        n_years, note
    """
    today  = pd.Timestamp.today().normalize()
    day8   = today + timedelta(days=8)   # week 1 belongs to short-range model
    day365 = today + timedelta(days=365)

    season_map = {
        12: "Winter", 1: "Winter",  2: "Winter",
        3:  "Spring", 4: "Spring",  5: "Spring",
        6:  "Summer", 7: "Summer",  8: "Summer",
        9:  "Autumn", 10: "Autumn", 11: "Autumn",
    }

    future_mondays = pd.date_range(start=day8, end=day365, freq="W-MON")
    rows = []

    for week_start in future_mondays:
        iso_week = int(week_start.isocalendar()[1])
        match    = normals[normals["week_of_year"] == iso_week]

        if len(match) == 0:
            continue

        m = match.iloc[0]
        rows.append({
            "week_start":     week_start.date().isoformat(),
            "week_end":       (week_start + timedelta(days=6)).date().isoformat(),
            "week_of_year":   iso_week,
            "days_ahead":     int((week_start - today).days),
            "season":         season_map.get(week_start.month, ""),
            # Daily kWh percentiles
            "kwh_day_p10":    round(float(m["kwh_p10"]), 2),
            "kwh_day_p25":    round(float(m["kwh_p25"]), 2),
            "kwh_day_p50":    round(float(m["kwh_p50"]), 2),
            "kwh_day_p75":    round(float(m["kwh_p75"]), 2),
            "kwh_day_p90":    round(float(m["kwh_p90"]), 2),
            # Weekly total (p50 × 7 days)
            "kwh_week_p50":   round(float(m["kwh_p50"]) * 7, 1),
            "cloud_mean_pct": round(float(m["cloud_mean"]), 1),
            "temp_mean_c":    round(float(m["temp_mean"]), 1),
            "n_years":        int(m["n_days"]),
            "note":           f"historical average — {HISTORY_YEARS} years",
            "seas5_source":   False,
        })

    return pd.DataFrame(rows)


# ── SEAS5 OVERRIDE ────────────────────────────────────────
def apply_seas5_override(forecast: pd.DataFrame, seas5: pd.DataFrame) -> pd.DataFrame:
    """
    Replace historical rows with SEAS5-derived kWh for weeks present in seas5.

    Physics path:
        clearsky_index = seas5_radiation_wh / clearsky_wh_for_that_week
        kwh_day_p50    = clearsky_index × peak_sun_hours × SYSTEM_KWP × PVGIS_CALIB

    Uncertainty proxy: ±20% around p50 (SEAS5 ensemble spread not yet downscaled).
    """
    if seas5.empty:
        return forecast

    forecast    = forecast.copy()
    seas5_index = pd.to_datetime(seas5["week_start"]).dt.normalize()

    # Pre-compute weekly clearsky Wh for each SEAS5 week
    all_dates = pd.date_range(
        start=seas5_index.min(),
        end=seas5_index.max() + timedelta(days=6),
        freq="D",
    )
    cs_daily = compute_daily_clearsky_wh(pd.DatetimeIndex(all_dates))

    def _week_clearsky_wh(week_start):
        days = pd.date_range(week_start, periods=7, freq="D")
        return float(np.mean([cs_daily.get(pd.Timestamp(d).normalize(), np.nan) for d in days]))

    # Load Model 2 seasonal scale if available
    m2_scale = {}
    m2_path  = PROJECT_ROOT / "data" / "models" / "two_stage_model2.pkl"
    if m2_path.exists():
        try:
            m2       = joblib.load(m2_path)
            m2_scale = m2.get("scale", {})
        except Exception:
            pass

    seas5_lookup = seas5.set_index(seas5_index)

    for idx, row in forecast.iterrows():
        ws = pd.Timestamp(row["week_start"]).normalize()
        if ws not in seas5_lookup.index:
            continue

        s           = seas5_lookup.loc[ws]
        cs_wh       = _week_clearsky_wh(ws)
        if cs_wh <= 0 or np.isnan(cs_wh):
            continue

        ci          = min(float(s["radiation_sum_wh"]) / cs_wh, 1.2)
        psh         = cs_wh / 1000.0                         # peak sun hours
        kwh_raw     = ci * psh * SYSTEM_KWP * PVGIS_CALIB
        month_scale = m2_scale.get(ws.month, 1.0)
        kwh_p50     = max(kwh_raw * month_scale, 0.0)

        forecast.at[idx, "kwh_day_p50"]   = round(kwh_p50, 2)
        forecast.at[idx, "kwh_day_p10"]   = round(kwh_p50 * 0.80, 2)
        forecast.at[idx, "kwh_day_p25"]   = round(kwh_p50 * 0.90, 2)
        forecast.at[idx, "kwh_day_p75"]   = round(kwh_p50 * 1.10, 2)
        forecast.at[idx, "kwh_day_p90"]   = round(kwh_p50 * 1.20, 2)
        forecast.at[idx, "kwh_week_p50"]  = round(kwh_p50 * 7, 1)
        forecast.at[idx, "cloud_mean_pct"] = round(float(s["cloud_mean"]), 1)
        forecast.at[idx, "temp_mean_c"]    = round(float(s["temp_mean"]), 1)
        forecast.at[idx, "seas5_source"]   = True
        forecast.at[idx, "note"]           = "ECMWF SEAS5 via Open-Meteo"

    n_seas5 = forecast["seas5_source"].sum()
    print(f"  SEAS5 override applied to {n_seas5} weeks  "
          f"({'Model 2 scale applied' if m2_scale else 'no Model 2 scale'})")
    return forecast


# ── DAILY EXPANSION (mirrors 14-day file style) ───────────
def expand_to_daily(forecast: pd.DataFrame) -> pd.DataFrame:
    """
    Fan the weekly seasonal forecast out to one row per day (day 8 → day 365).
    Each day in a week inherits that week's p50/p10/p90 — SEAS5/climatology has
    no sub-weekly skill, so a flat allocation is the honest representation.

    Output columns are deliberately parallel to forecast_14day_15min.csv:
        date, datetime, soli_kwh_pred, soli_kwh_q05, soli_kwh_q95,
        clouds, temp, weather, source, season, week_of_year, days_ahead

    `weather` is a simple cloud-cover label, matching the convention in
    forecast_pv.py::weather_label().
    """
    def _weather(c):
        if c is None or pd.isna(c): return "n/a"
        if c > 70: return "☁ Overcast"
        if c > 30: return "⛅ Partly cloudy"
        return "☀ Clear"

    rows = []
    for _, w in forecast.iterrows():
        week_start = pd.Timestamp(w["week_start"])
        for d in range(7):
            day = week_start + timedelta(days=d)
            rows.append({
                "date":           day.date().isoformat(),
                "datetime":       day.to_pydatetime().isoformat(sep=" "),
                "soli_kwh_pred":  round(float(w["kwh_day_p50"]), 3),
                "soli_kwh_q05":   round(float(w["kwh_day_p10"]), 3),  # p10 as low band
                "soli_kwh_q95":   round(float(w["kwh_day_p90"]), 3),  # p90 as high band
                "clouds":         w["cloud_mean_pct"],
                "temp":           w["temp_mean_c"],
                "weather":        _weather(w["cloud_mean_pct"]),
                "source":         "SEAS5" if bool(w.get("seas5_source", False)) else "Climatology-5yr",
                "season":         w["season"],
                "week_of_year":   int(w["week_of_year"]),
                "days_ahead":     int(w["days_ahead"]) + d,
            })
    return pd.DataFrame(rows)


# ── PRINT SUMMARY ─────────────────────────────────────────
def print_monthly_summary(forecast: pd.DataFrame):
    print(f"\n── Monthly kWh/day averages (p50), based on {HISTORY_YEARS}-year history ──")
    forecast  = forecast.copy()
    forecast["month"] = pd.to_datetime(forecast["week_start"]).dt.month
    monthly   = forecast.groupby("month")["kwh_day_p50"].mean()
    max_val   = monthly.max()
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    for m, val in monthly.items():
        bar = "█" * int(val / max_val * 24)
        print(f"  {month_names.get(m, m):>3}: {val:5.1f} kWh/day  {bar}")


# ── MAIN ──────────────────────────────────────────────────
def run_forecast_climate() -> pd.DataFrame:
    print("\n── Weekly solar forecast (week 2 → week 52) ──")

    # 1. Historical weather (used for weeks 27–52 and as baseline)
    hist = fetch_site_history(years=HISTORY_YEARS)

    # 2. Estimate kWh from historical radiation
    hist = estimate_daily_kwh(hist)

    # 3. Weekly normals from history
    print("  Computing week-of-year statistics...")
    normals = build_weekly_normals(hist)
    print(f"  {len(normals)} week profiles computed")

    # 4. Base forecast table (weeks 2–52, all historical by default)
    forecast = build_forecast_table(normals)
    print(f"  {len(forecast)} forecast weeks (day 8 → day 365)")

    # 5. SEAS5 override for weeks 2–26 (days 8–182)
    if _SEAS5_AVAILABLE:
        try:
            print("Fetching ECMWF SEAS5 forecast (Open-Meteo Seasonal API)...")
            seas5 = fetch_seas5_weekly(SITE_LAT, SITE_LON, months=6, skip_days=7)
            print(f"  SEAS5: {len(seas5)} weeks  "
                  f"({seas5['week_start'].min().date()} → {seas5['week_start'].max().date()})  "
                  f"members: {seas5['n_members'].iloc[0] if len(seas5) else 'n/a'}")
            forecast = apply_seas5_override(forecast, seas5)
        except Exception as e:
            print(f"  SEAS5 fetch failed ({e}) — using historical climatology for all weeks")
    else:
        print("  SEAS5 module not available — using historical climatology for all weeks")

    # 6. Save weekly
    out_path = PROJECT_ROOT / "data" / "outputs" / "forecast_365day_weekly.csv"
    forecast.to_csv(out_path, index=False)
    n_seas5 = int(forecast["seas5_source"].sum()) if "seas5_source" in forecast.columns else 0
    n_hist  = len(forecast) - n_seas5
    print(f"Saved → {out_path.relative_to(PROJECT_ROOT)}  "
          f"({len(forecast)} rows: {n_seas5} SEAS5 + {n_hist} historical)")

    # 7. Save daily (mirrors the 14-day file's style — one row per day)
    daily = expand_to_daily(forecast)
    daily_path = PROJECT_ROOT / "data" / "outputs" / "forecast_seasonal_daily.csv"
    daily.to_csv(daily_path, index=False)
    n_seas5_d = int((daily["source"] == "SEAS5").sum())
    n_hist_d  = len(daily) - n_seas5_d
    print(f"Saved → {daily_path.relative_to(PROJECT_ROOT)}  "
          f"({len(daily)} rows: {n_seas5_d} SEAS5 + {n_hist_d} historical, "
          f"{daily['date'].iloc[0]} → {daily['date'].iloc[-1]})")

    print_monthly_summary(forecast)

    return forecast


if __name__ == "__main__":
    run_forecast_climate()
