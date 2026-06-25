"""
features/engineering.py — CANONICAL feature engineering (remediation C3).

Single source of truth imported by BOTH the training builder and the inference
path, so train and serve compute byte-identical features. Do not fork this
logic into the inference script again.

Public API:
    engineer_features(df)  -> df with all physics/weather/clear-sky/time features
    add_cf_lags(df)        -> df with cf_lag_1h / cf_lag_2h / cf_roll3h (PV-lag;
                              used ONLY by the nowcast/legacy single-model path,
                              NOT by the two-stage fast model)

Expected input columns on df:
    datetime, solar_radiation, temp, dew_point, wind_speed, precipitation,
    clouds, humidity
"""

import numpy as np
import pandas as pd
import pvlib

# ── Site geometry (ID003 Soesterberg) ──────────────────────────────────────────
SITE_LAT      = 52.1088
SITE_LON      = 5.1253
SITE_ALT_M    = 15
PANEL_TILT    = 40      # degrees
PANEL_AZIMUTH = 203     # degrees (SSW)


def _pvlib_location():
    return pvlib.location.Location(
        latitude=SITE_LAT, longitude=SITE_LON,
        altitude=SITE_ALT_M, tz="UTC",
    )


def add_solar_position(df: pd.DataFrame) -> pd.DataFrame:
    times = pd.DatetimeIndex(df["datetime"])
    if times.tz is None:
        times = times.tz_localize("UTC")
    pos = _pvlib_location().get_solarposition(times)
    df["solar_elevation"] = pos["elevation"].values
    df["solar_azimuth"]   = pos["azimuth"].values

    elev_rad = np.radians(df["solar_elevation"].clip(lower=0))
    df["solar_elevation_sin"] = np.sin(elev_rad)
    df["solar_elevation_cos"] = np.cos(elev_rad)

    aoi = pvlib.irradiance.aoi(
        PANEL_TILT, PANEL_AZIMUTH,
        pos["zenith"].values, pos["azimuth"].values,
    )
    df["cos_incidence"] = np.cos(np.radians(aoi)).clip(0)
    df["air_mass"]      = pvlib.atmosphere.get_relative_airmass(pos["zenith"].values)
    return df


def add_poa_estimate(df: pd.DataFrame) -> pd.DataFrame:
    times = pd.DatetimeIndex(df["datetime"])
    if times.tz is None:
        times = times.tz_localize("UTC")
    sol       = _pvlib_location().get_solarposition(times)
    dni_extra = pvlib.irradiance.get_extra_radiation(times).values
    erbs = pvlib.irradiance.erbs(
        df["solar_radiation"].values, sol["zenith"].values, times.dayofyear
    )
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=PANEL_TILT, surface_azimuth=PANEL_AZIMUTH,
        solar_zenith=sol["zenith"].values, solar_azimuth=sol["azimuth"].values,
        dni=erbs["dni"], ghi=df["solar_radiation"].values, dhi=erbs["dhi"],
        model="haydavies", dni_extra=dni_extra,
    )
    df["poa_estimate"] = np.nan_to_num(poa["poa_global"], nan=0.0)
    return df


def add_clearsky(df: pd.DataFrame) -> pd.DataFrame:
    times = pd.DatetimeIndex(df["datetime"])
    if times.tz is None:
        times = times.tz_localize("UTC")
    cs  = _pvlib_location().get_clearsky(times, model="ineichen")
    df["clearsky_ghi"]   = cs["ghi"].values
    df["clearsky_index"] = (
        df["solar_radiation"] / df["clearsky_ghi"].clip(lower=1)
    ).clip(0, 1.5)

    sol       = _pvlib_location().get_solarposition(times)
    dni_extra = pvlib.irradiance.get_extra_radiation(times).values
    cs_poa    = pvlib.irradiance.get_total_irradiance(
        surface_tilt=PANEL_TILT, surface_azimuth=PANEL_AZIMUTH,
        solar_zenith=sol["zenith"].values, solar_azimuth=sol["azimuth"].values,
        dni=cs["dni"].values, ghi=cs["ghi"].values, dhi=cs["dhi"].values,
        model="haydavies", dni_extra=dni_extra,
    )
    df["clearsky_poa"]       = np.nan_to_num(cs_poa["poa_global"], nan=0.0)
    df["clearsky_poa_index"] = (
        df["poa_estimate"] / df["clearsky_poa"].clip(lower=1)
    ).clip(0, 1.5)
    return df


def add_fourier(df: pd.DataFrame) -> pd.DataFrame:
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]      / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]      / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"]     / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]     / 12)
    df["doy_sin"]   = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"]   = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("datetime").reset_index(drop=True)

    df["hour"]      = df["datetime"].dt.hour
    df["month"]     = df["datetime"].dt.month
    df["dayofyear"] = df["datetime"].dt.dayofyear

    df = add_solar_position(df)

    df["effective_radiation"] = df["solar_radiation"] * df["solar_elevation_sin"].clip(lower=0)
    df["thermal_factor"]      = (1.0 - 0.005 * (df["temp"] - 25)).clip(0.5, 1.5)
    df["adjusted_radiation"]  = df["effective_radiation"] * df["thermal_factor"]

    df["temp_delta_25"]       = df["temp"] - 25
    df["wind_cooling_effect"] = df["wind_speed"] * df["temp_delta_25"].clip(upper=0).abs()
    df["dew_spread"]          = df["temp"] - df["dew_point"]

    df["clouds_x_elevation"]   = df["clouds"]   * df["solar_elevation"].clip(lower=0)
    df["humidity_x_elevation"] = df["humidity"] * df["solar_elevation"].clip(lower=0)
    df["humidity_radiation"]   = df["humidity"] * df["solar_radiation"]

    df["rain_last_24h"]    = df["precipitation"].rolling(24, min_periods=1).sum()
    rain_times_ffill       = df["datetime"].where(df["precipitation"] > 0.01).ffill()
    df["hours_since_rain"] = (
        (df["datetime"] - rain_times_ffill).dt.total_seconds() / 3600
    ).fillna(999).clip(upper=999)

    df["solar_lag_1h"]  = df["solar_radiation"].shift(1)
    df["solar_lag_24h"] = df["solar_radiation"].shift(24)
    df["cloud_lag_1h"]  = df["clouds"].shift(1)

    df["clouds_3h_mean"]    = df["clouds"].rolling(3, min_periods=1).mean()
    df["radiation_3h_mean"] = df["solar_radiation"].rolling(3, min_periods=1).mean()
    df["cloud_trend_3h"]    = df["clouds"] - df["clouds_3h_mean"]

    df["radiation_change_1h"]  = df["solar_radiation"] - df["solar_lag_1h"]
    df["cloud_change_1h"]      = df["clouds"]          - df["cloud_lag_1h"]
    df["cloud_burst_flag"]     = (df["cloud_change_1h"]      > 25).astype(int)
    df["irradiance_drop_flag"] = (df["radiation_change_1h"] < -100).astype(int)

    df = add_poa_estimate(df)

    df["cell_temp_est"]      = df["temp"] + (df["poa_estimate"] / 800) * (45 - 20)
    df["cell_temp_derating"] = (1 - 0.004 * (df["cell_temp_est"] - 25)).clip(0.7, 1.0)
    df["poa_temp_corrected"] = df["poa_estimate"] * df["cell_temp_derating"]

    df = add_clearsky(df)
    df = add_fourier(df)
    return df


def add_cf_lags(df: pd.DataFrame) -> pd.DataFrame:
    """PV-output lag features. Target column must be named 'capacity_factor'.
    Used only by the nowcast / legacy single-model path — the two-stage FAST
    model must NOT use these (no dependence on lagged PV output)."""
    df = df.sort_values("datetime").reset_index(drop=True)
    df["cf_lag_1h"] = df["capacity_factor"].shift(1)
    df["cf_lag_2h"] = df["capacity_factor"].shift(2)
    df["cf_roll3h"] = df["capacity_factor"].shift(1).rolling(3, min_periods=1).mean()
    return df
