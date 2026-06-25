# src/inference/forecast_pv.py
# Inference pipeline — matches train_pv.py exactly.
# Changes from the previous XGBoost version:
#   ─ Loads lgbm_model_*.pkl instead of xgb_model_*.pkl
#   ─ Computes clearsky_ghi & clearsky_index via pvlib (LOCAL — Soesterberg)
#   ─ Adds Fourier time features (hour/month/doy × sin/cos)
#   ─ Applies Mondrian conformal correction (per-row, by extreme flag)
#
# Dependency:  pip install lightgbm pvlib

import sys
import requests
import numpy as np
import pandas as pd
import pvlib
from pathlib import Path
import joblib
from datetime import datetime, timedelta
import pytz

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── SITE CONFIG (solar-sme-actual = NL plan B) ────────────
# Site is the SAME Soesterberg location as Model B's training data (ID003).
# PVGIS_CALIB was a NL→Tbilisi bias correction; here we predict for NL on a
# model trained on NL measurements, so no cross-country adjustment is needed.
SITE_LAT, SITE_LON = 52.1088, 5.1253
SITE_ALT_M       = 15           # Soesterberg altitude (m above sea level, approx)
SYSTEM_KWP     = 2.25           # ID003 real scale (AC rating, 2250 W) — capacity_factor
                                # is normalized by this same value, so it is the only
                                # consistent CF→kWh base. Was 15 kWp for SME-pipeline parity.
PANEL_TILT     = 40
PANEL_AZIMUTH  = 203
TEMP_COEFF     = 0.004
STC_TEMP       = 25
PVGIS_CALIB    = 1.0            # was 0.6281 (NL→Tbilisi); not applicable for NL→NL

SITE_TZ_STR = "UTC"
SITE_TZ     = pytz.timezone(SITE_TZ_STR)

# ── FEATURES (must match train_pv.py) ─────────────────────
FEATURES = [
    # Physics
    "solar_radiation", "effective_radiation", "adjusted_radiation",
    "solar_elevation", "solar_elevation_sin", "solar_elevation_cos",
    "solar_azimuth", "air_mass",
    "cos_incidence", "poa_estimate",
    "cell_temp_est", "cell_temp_derating",  # NOCT thermal model
    "poa_temp_corrected",                   # poa × derating interaction
    # Clear-sky decomposition  ── NEW ──
    "clearsky_ghi", "clearsky_index",
    "clearsky_poa", "clearsky_poa_index",
    # Weather
    "clouds", "temp", "humidity", "wind_speed",
    "thermal_factor", "temp_delta_25", "wind_cooling_effect",
    # Interaction
    "clouds_x_elevation", "humidity_x_elevation", "humidity_radiation",
    # Rain / soiling
    "hours_since_rain", "rain_last_24h", "dew_spread",
    # Lag & rolling
    "solar_lag_1h", "solar_lag_24h", "cloud_lag_1h",
    "clouds_3h_mean", "radiation_3h_mean", "cloud_trend_3h",
    # Time (integer + Fourier)
    "hour", "month", "dayofyear",
    "hour_sin", "hour_cos",            # ── NEW ──
    "month_sin", "month_cos",          # ── NEW ──
    "doy_sin", "doy_cos",              # ── NEW ──
    # Cloud motion & extreme events
    "radiation_change_1h", "cloud_change_1h",
    "cloud_burst_flag", "irradiance_drop_flag",
]

# ── TIME HELPERS ──────────────────────────────────────────
def get_next_15min():
    now = datetime.now(SITE_TZ).replace(tzinfo=None)
    minutes = (now.minute // 15 + 1) * 15
    if minutes == 60:
        return now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    return now.replace(minute=minutes, second=0, microsecond=0)

# ── OPEN-METEO FETCH ──────────────────────────────────────
def fetch_forecast(lat, lon, days=14):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "cloud_cover",
            "shortwave_radiation", "wind_speed_10m", "wind_direction_10m",
            "pressure_msl", "precipitation", "dew_point_2m"
        ],
        # Open-Meteo defaults wind to km/h. The model was trained on KNMI wind in
        # m/s, and the alert storm threshold is in m/s — request m/s so both agree.
        "wind_speed_unit": "ms",
        "forecast_days": days,
        "timezone": SITE_TZ_STR,
    }
    r = requests.get(url, params=params, timeout=30)
    data = r.json()
    df = pd.DataFrame({
        "datetime":        pd.to_datetime(data["hourly"]["time"]),
        "temp":            data["hourly"]["temperature_2m"],
        "humidity":        data["hourly"]["relative_humidity_2m"],
        "clouds":          data["hourly"]["cloud_cover"],
        "solar_radiation": data["hourly"]["shortwave_radiation"],
        "wind_speed":      data["hourly"]["wind_speed_10m"],
        "wind_direction":  data["hourly"]["wind_direction_10m"],
        "pressure":        data["hourly"]["pressure_msl"],
        "precipitation":   data["hourly"]["precipitation"],
        "dew_point":       data["hourly"]["dew_point_2m"],
    })
    if df["datetime"].dt.tz is not None:
        df["datetime"] = df["datetime"].dt.tz_localize(None)
    return df

# ── POA ESTIMATE (must match build_soesterberg_training.py) ──
def add_poa_estimate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate plane-of-array irradiance from horizontal GHI using pvlib's
    Hay-Davies transposition. Works at inference too — Open-Meteo gives GHI,
    we compute POA the same way.
    """
    times = pd.DatetimeIndex(df["datetime"])
    if times.tz is None:
        times = times.tz_localize(SITE_TZ_STR)
    location = pvlib.location.Location(
        latitude=SITE_LAT, longitude=SITE_LON,
        altitude=SITE_ALT_M, tz=SITE_TZ_STR,
    )
    sol = location.get_solarposition(times)
    # Split GHI into DNI + DHI using Erbs decomposition
    dni_extra = pvlib.irradiance.get_extra_radiation(times).values
    erbs = pvlib.irradiance.erbs(df["solar_radiation"].values,
                                  sol["zenith"].values,
                                  times.dayofyear)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=PANEL_TILT, surface_azimuth=PANEL_AZIMUTH,
        solar_zenith=sol["zenith"].values,
        solar_azimuth=sol["azimuth"].values,
        dni=erbs["dni"], ghi=df["solar_radiation"].values, dhi=erbs["dhi"],
        model="haydavies", dni_extra=dni_extra,
    )
    df["poa_estimate"] = np.nan_to_num(poa["poa_global"], nan=0.0)
    return df


# ── CLEAR-SKY + FOURIER (must match train_pv.py) ──────────
def add_clearsky_features(df):
    """
    Computes Soesterberg's clear-sky GHI for each timestamp and the clear-sky
    index = actual / clearsky.
    """
    times = pd.DatetimeIndex(df["datetime"])
    if times.tz is None:
        times = times.tz_localize(SITE_TZ_STR)
    location = pvlib.location.Location(
        latitude=SITE_LAT, longitude=SITE_LON,
        altitude=SITE_ALT_M, tz=SITE_TZ_STR,
    )
    cs  = location.get_clearsky(times, model="ineichen")
    df["clearsky_ghi"]   = cs["ghi"].values
    df["clearsky_index"] = (
        df["solar_radiation"] / df["clearsky_ghi"].clip(lower=1)
    ).clip(0, 1.5)
    # Clearsky POA via Hay-Davies — needs poa_estimate already in df
    sol       = location.get_solarposition(times)
    dni_extra = pvlib.irradiance.get_extra_radiation(times).values
    cs_poa    = pvlib.irradiance.get_total_irradiance(
        surface_tilt=PANEL_TILT, surface_azimuth=PANEL_AZIMUTH,
        solar_zenith=sol["zenith"].values,
        solar_azimuth=sol["azimuth"].values,
        dni=cs["dni"].values, ghi=cs["ghi"].values, dhi=cs["dhi"].values,
        model="haydavies", dni_extra=dni_extra,
    )
    df["clearsky_poa"]       = np.nan_to_num(cs_poa["poa_global"], nan=0.0)
    df["clearsky_poa_index"] = (
        df["poa_estimate"] / df["clearsky_poa"].clip(lower=1)
    ).clip(0, 1.5)
    return df

def add_fourier_features(df):
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]      / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]      / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"]     / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]     / 12)
    df["doy_sin"]   = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"]   = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    return df

# ── FEATURE ENGINEERING ───────────────────────────────────
def add_features(df, lat):
    # ── Solar geometry (existing) ─────────────────────────
    n      = df["datetime"].dt.dayofyear
    hour   = df["datetime"].dt.hour + df["datetime"].dt.minute / 60
    decl   = 23.44 * np.cos(np.radians((360 / 365) * (n - 172)))
    ha     = 15 * (hour - 12)
    lat_r  = np.radians(lat)
    decl_r = np.radians(decl)
    ha_r   = np.radians(ha)

    df["solar_elevation"] = np.degrees(np.arcsin(
        np.sin(lat_r) * np.sin(decl_r) +
        np.cos(lat_r) * np.cos(decl_r) * np.cos(ha_r)
    ))
    df["solar_elevation_sin"] = np.sin(np.radians(df["solar_elevation"]))
    df["solar_elevation_cos"] = np.cos(np.radians(df["solar_elevation"]))

    elev_r  = np.radians(df["solar_elevation"])
    cos_az  = (np.sin(decl_r) * np.cos(lat_r) -
               np.cos(decl_r) * np.sin(lat_r) * np.cos(ha_r)) / np.cos(elev_r)
    azimuth = np.degrees(np.arccos(cos_az.clip(-1, 1)))
    df["solar_azimuth"] = np.where(ha > 0, 360 - azimuth, azimuth)
    zenith = 90 - df["solar_elevation"]
    df["air_mass"] = pvlib.atmosphere.get_relative_airmass(zenith.values)

    # ── POA estimate (Hay-Davies) ─────────────────────────
    df = add_poa_estimate(df)

    df["cell_temp_est"]      = df["temp"] + (df["poa_estimate"] / 800) * (45 - 20)
    df["cell_temp_derating"] = (1 - 0.004 * (df["cell_temp_est"] - 25)).clip(0.7, 1.0)
    df["poa_temp_corrected"] = df["poa_estimate"] * df["cell_temp_derating"]

    tilt_r  = np.radians(PANEL_TILT)
    az_diff = np.radians(df["solar_azimuth"] - PANEL_AZIMUTH)
    cos_inc = (np.sin(elev_r) * np.cos(tilt_r) +
               np.cos(elev_r) * np.sin(tilt_r) * np.cos(az_diff)).clip(0, 1)

    # ── Existing derived features ─────────────────────────
    df["cos_incidence"]        = cos_inc
    df["effective_radiation"]  = df["solar_radiation"] * cos_inc
    df["temp_delta_25"]        = df["temp"] - STC_TEMP
    df["thermal_factor"]       = 1 - (df["temp_delta_25"] * TEMP_COEFF)
    df["wind_cooling_effect"]  = df["wind_speed"] * np.maximum(0, df["temp_delta_25"])
    df["adjusted_radiation"]   = df["effective_radiation"] * df["thermal_factor"]
    df["had_rain"]             = (df["precipitation"] > 0).astype(int)
    df["hours_since_rain"]     = df.groupby(df["had_rain"].cumsum()).cumcount()
    df["rain_last_24h"]        = df["precipitation"].rolling(24, min_periods=1).sum().gt(0).astype(int)
    df["dew_spread"]           = df["temp"] - df["dew_point"]
    df["humidity_radiation"]   = df["humidity"] * df["solar_radiation"] / 100
    elev_pos                   = np.maximum(0, df["solar_elevation"])
    df["clouds_x_elevation"]   = df["clouds"]   * elev_pos / 100
    df["humidity_x_elevation"] = df["humidity"] * elev_pos / 100
    df["hour"]                 = df["datetime"].dt.hour
    df["month"]                = df["datetime"].dt.month
    df["dayofyear"]            = df["datetime"].dt.dayofyear
    df["solar_lag_1h"]         = df["solar_radiation"].shift(1)
    df["solar_lag_24h"]        = df["solar_radiation"].shift(24)
    df["cloud_lag_1h"]         = df["clouds"].shift(1)
    df["clouds_3h_mean"]       = df["clouds"].rolling(3, min_periods=1).mean()
    df["radiation_3h_mean"]    = df["solar_radiation"].rolling(3, min_periods=1).mean()
    df["cloud_trend_3h"]       = df["clouds"] - df["clouds_3h_mean"]

    # Cloud motion & extreme events
    df["radiation_change_1h"]  = df["solar_radiation"] - df["solar_lag_1h"]
    df["cloud_change_1h"]      = df["clouds"]          - df["cloud_lag_1h"]
    df["cloud_burst_flag"]     = (df["cloud_change_1h"]      > 25).astype(int)
    df["irradiance_drop_flag"] = (df["radiation_change_1h"] < -100).astype(int)

    # ── NEW: clear-sky + Fourier ──────────────────────────
    df = add_clearsky_features(df)
    df = add_fourier_features(df)

    return df

# ── CORRECTION LOADING (handles legacy + Mondrian) ────────
def _normalize_correction(correction):
    """
    Accepts both correction formats:
      Legacy (2 keys):    {lower, upper}
      Mondrian (4 keys):  {normal_lower, normal_upper, extreme_lower, extreme_upper}

    Returns Mondrian format. Legacy is upgraded by reusing the same correction
    for both groups (i.e. behaves identically to the old global correction).
    """
    if "normal_lower" in correction:
        return correction
    return {
        "normal_lower":  correction["lower"],
        "normal_upper":  correction["upper"],
        "extreme_lower": correction["lower"],
        "extreme_upper": correction["upper"],
    }

def apply_mondrian_correction(q05_raw, q95_raw, extreme_mask, correction):
    """Per-row pick of (lower, upper) corrections based on extreme flag."""
    lower = np.where(extreme_mask, correction["extreme_lower"], correction["normal_lower"])
    upper = np.where(extreme_mask, correction["extreme_upper"], correction["normal_upper"])
    q05_adj = np.maximum(q05_raw - lower, 0)
    q95_adj = np.minimum(q95_raw + upper, 1)
    return q05_adj, q95_adj

# ── PREDICT ───────────────────────────────────────────────
def predict(df, models, correction):
    df = df.sort_values("datetime").reset_index(drop=True)

    df["soli_kwh_pred"] = 0.0
    df["soli_kwh_q05"]  = 0.0
    df["soli_kwh_q95"]  = 0.0

    for i in range(len(df)):
        if df.at[i, "solar_elevation"] <= 0:
            continue

        x = df.loc[[i], FEATURES].fillna(0)
        q50_cf = float(models["q50"].predict(x).clip(0, 1)[0])
        q05_cf = float(models["q05"].predict(x).clip(0, 1)[0])
        q95_cf = float(models["q95"].predict(x).clip(0, 1)[0])

        is_extreme = (
            (df.at[i, "cloud_burst_flag"]     == 1) or
            (df.at[i, "irradiance_drop_flag"] == 1)
        )
        q05_adj, q95_adj = apply_mondrian_correction(
            np.array([q05_cf]), np.array([q95_cf]),
            np.array([is_extreme]), correction,
        )

        df.at[i, "soli_kwh_pred"] = q50_cf     * PVGIS_CALIB * SYSTEM_KWP
        df.at[i, "soli_kwh_q05"]  = q05_adj[0] * PVGIS_CALIB * SYSTEM_KWP
        df.at[i, "soli_kwh_q95"]  = q95_adj[0] * PVGIS_CALIB * SYSTEM_KWP

    return df

# ── SUMMARIES ─────────────────────────────────────────────
def weather_label(clouds, precip):
    if precip > 2:    return "🌧 Rainy"
    if precip > 0.5:  return "🌦 Showers"
    if clouds > 70:   return "☁ Overcast"
    if clouds > 30:   return "⛅ Partly cloudy"
    return "☀ Clear"

def daily_summary(df):
    df["date"] = df["datetime"].dt.date
    daily = df.groupby("date").agg(
        predicted_kwh = ("soli_kwh_pred", "sum"),
        avg_clouds    = ("clouds", "mean"),
        total_precip  = ("precipitation", "sum"),
        avg_temp      = ("temp", "mean"),
    ).reset_index()
    daily["weather"]       = daily.apply(lambda r: weather_label(r["avg_clouds"], r["total_precip"]), axis=1)
    daily["predicted_kwh"] = daily["predicted_kwh"].round(2)
    daily["avg_temp"]      = daily["avg_temp"].round(1)
    daily["avg_clouds"]    = daily["avg_clouds"].round(0).astype(int)
    return daily

def hourly_summary(df):
    df["date"]    = df["datetime"].dt.date
    df["time"]    = df["datetime"].dt.strftime("%H:%M")
    df["weather"] = df.apply(lambda r: weather_label(r["clouds"], r["precipitation"]), axis=1)
    out = df[df["soli_kwh_pred"] > 0.01].copy()
    out = out[[
        "date", "time", "datetime",
        "soli_kwh_pred", "soli_kwh_q05", "soli_kwh_q95",
        "solar_radiation", "clouds", "temp",
        "precipitation", "wind_speed", "dew_point", "humidity", "weather"
    ]].copy()
    out["soli_kwh_pred"]   = out["soli_kwh_pred"].round(3)
    out["soli_kwh_q05"]    = out["soli_kwh_q05"].round(3)
    out["soli_kwh_q95"]    = out["soli_kwh_q95"].round(3)
    out["solar_radiation"] = out["solar_radiation"].round(0).astype(int)
    out["clouds"]          = out["clouds"].round(0).astype(int)
    out["temp"]            = out["temp"].round(1)
    out["precipitation"]   = out["precipitation"].round(2)
    out["wind_speed"]      = out["wind_speed"].round(1)
    return out.reset_index(drop=True)

# ── MAIN ──────────────────────────────────────────────────
def run_forecast(days=14):
    models_dir = PROJECT_ROOT / "data" / "models"

    # ── solar-sme-actual: Model B (trained on REAL ID003 measured generation) ──
    # Prefers lgbm_real_*.pkl. Falls back to lgbm_model_* / xgb_model_* only if
    # the real-data model is missing, so this script still runs in fallback mode.
    def _load(stem):
        for prefix in ("lgbm_real_", "lgbm_model_", "xgb_model_"):
            path = models_dir / f"{prefix}{stem}.pkl"
            if path.exists():
                print(f"Loaded {path.name}")
                return joblib.load(path)
        raise FileNotFoundError(f"No model file found for {stem} in {models_dir}")

    models = {"q50": _load("q50"), "q05": _load("q05"), "q95": _load("q95")}

    # Conformal correction: prefer real-data version
    correction = None
    for cname in ("conformal_correction_real.pkl", "conformal_correction.pkl"):
        cpath = models_dir / cname
        if cpath.exists():
            print(f"Loaded {cname}")
            correction = _normalize_correction(joblib.load(cpath))
            break
    if correction is None:
        raise FileNotFoundError(f"No conformal correction file found in {models_dir}")

    df = fetch_forecast(SITE_LAT, SITE_LON, days=days)
    df = add_features(df, SITE_LAT)
    df = predict(df, models, correction)

    # Save raw hourly feature matrix for XAI (before 15-min interpolation)
    # analysis/xai.py reads this file — do not remove.
    day_mask_feat = df["solar_elevation"] > 0
    feat_out = df[day_mask_feat][["datetime"] + FEATURES].copy()
    feat_path = PROJECT_ROOT / "data" / "outputs" / "features_14day.csv"
    feat_out.to_csv(feat_path, index=False)
    print(f"Saved → {feat_path.relative_to(PROJECT_ROOT)}  ({len(feat_out)} rows, feature matrix for XAI)")

    # Resample hourly → 15-min via linear interpolation
    df = df.set_index("datetime")
    numeric_cols = df.select_dtypes(include="number").columns
    df_15 = df[numeric_cols].resample("15min").interpolate(method="linear")

    # Precipitation is an HOURLY ACCUMULATION (mm/hour), not an instantaneous
    # state. Linear interpolation + summing the four 15-min slots would inflate
    # daily totals ~4×. Spread each hourly total evenly across its slots so the
    # daily sum is conserved.
    if "precipitation" in df.columns:
        df_15["precipitation"] = df["precipitation"].resample("15min").ffill() / 4

    df    = df_15.reset_index()

    start_time = get_next_15min()
    print(f"Forecast from: {start_time}")
    df = df[df["datetime"] >= start_time].reset_index(drop=True)

    daily  = daily_summary(df)
    hourly = hourly_summary(df)

    out_path = PROJECT_ROOT / "data" / "outputs" / "forecast_14day_15min.csv"
    hourly.to_csv(out_path, index=False)
    print(f"Saved → {out_path.relative_to(PROJECT_ROOT)}  ({len(hourly)} rows)")
    return daily, hourly, df

if __name__ == "__main__":
    run_forecast(days=14)