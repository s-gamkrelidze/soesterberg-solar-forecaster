# src/ingestion/fetch_era5.py
# Downloads ERA5 hourly reanalysis data for Soesterberg, Utrecht, Netherlands from Copernicus CDS.
# ERA5 is the same underlying dataset PVGIS uses — ensures consistency
# between training weather inputs and the PVGIS PV output target.
#
# Site: former Soesterberg Air Base, ~10 km SE of Utrecht, Netherlands
#   Coordinates: 52.1088°N, 5.1253°E
#   Tilt: 40°  |  Azimuth: 203° (SSW)  |  Annual yield: 1015 kWh/kWp
#
# Variables downloaded (surface level, hourly):
#   ssrd  → surface solar radiation downwards (J/m²) → convert to W/m²
#   t2m   → 2m temperature (K) → convert to °C
#   u10   → 10m U-wind component (m/s)
#   v10   → 10m V-wind component (m/s) → combine with u10 for speed
#   tp    → total precipitation (m) → convert to mm
#   tcc   → total cloud cover (0–1) → convert to %
#   d2m   → 2m dew point (K) → convert to °C
#   sp    → surface pressure (Pa) → convert to hPa
#
# Coverage: 2014–2017 (4 years, matching available target data)
# Area: tight bounding box around Soesterberg solar farm (52.1088°N, 5.1253°E)

import cdsapi
import xarray as xr
import pandas as pd
import numpy as np
import zipfile
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHUNKS_DIR   = PROJECT_ROOT / "data" / "raw"
OUTPUT_CSV   = PROJECT_ROOT / "data" / "raw" / "Soesterberg_NL_solar.csv"

LAT, LON = 52.1088, 5.1253

AREA = [LAT + 0.25, LON - 0.25, LAT - 0.25, LON + 0.25]  # [N, W, S, E] — 0.5° box around the plant

VARIABLES = [
    "surface_solar_radiation_downwards",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
    "total_cloud_cover",
    "2m_dewpoint_temperature",
    "surface_pressure",
]

YEARS   = [str(y) for y in range(2014, 2018)]   # 2014–2017
HALVES  = [["01", "02", "03"], ["04", "05", "06"], ["07", "08", "09"], ["10", "11", "12"]]
DAYS    = [f"{d:02d}" for d in range(1, 32)]
TIMES   = [f"{h:02d}:00" for h in range(24)]


def download_era5() -> xr.Dataset:
    print("Downloading ERA5 for Soesterberg NL solar plant (3-month chunks) ...")
    c = None
    parts = []
    for year in YEARS:
        for i, months in enumerate(HALVES, start=1):
            part_path = CHUNKS_DIR / f"Soesterberg_NL_solar_raw_{year}_Q{i}.nc"
            if part_path.exists():
                print(f"  {year} Q{i}: already exists, skipping.")
            else:
                if c is None:
                    c = cdsapi.Client()
                print(f"  {year} Q{i}: requesting from CDS ...")
                c.retrieve(
                    "reanalysis-era5-single-levels",
                    {
                        "product_type": "reanalysis",
                        "variable":     VARIABLES,
                        "year":         [year],
                        "month":        months,
                        "day":          DAYS,
                        "time":         TIMES,
                        "area":         AREA,
                        "data_format":  "netcdf",
                    },
                    str(part_path),
                )
                print(f"  {year} Q{i}: saved → {part_path}")
            parts.append(part_path)

    print("Merging chunks in memory ...")
    quarter_datasets = []
    tmp_dirs = []
    for p in parts:
        with open(str(p), "rb") as f:
            magic = f.read(2)
        if magic == b"PK":
            # New CDS API wraps NetCDF files in a ZIP archive
            tmp = p.parent / (p.stem + "_unzipped")
            tmp.mkdir(exist_ok=True)
            tmp_dirs.append(tmp)
            with zipfile.ZipFile(str(p)) as z:
                z.extractall(str(tmp))
            inner = [xr.open_dataset(str(f), engine="netcdf4") for f in sorted(tmp.glob("*.nc"))]
            merged = xr.merge(inner, compat="override").load()
            for d in inner:
                d.close()
            quarter_datasets.append(merged)
        else:
            quarter_datasets.append(xr.open_dataset(str(p), engine="netcdf4").load())

    ds = xr.concat(quarter_datasets, dim="valid_time").sortby("valid_time")

    for p in parts:
        p.unlink()
    for tmp in tmp_dirs:
        shutil.rmtree(str(tmp))
    print(f"  Merged {len(parts)} chunks — {ds.sizes['valid_time']:,} timesteps")
    return ds


def process_era5(ds: xr.Dataset) -> pd.DataFrame:
    print("Processing ERA5 data ...")

    ds = ds.sel(latitude=LAT, longitude=LON, method="nearest")

    df = pd.DataFrame({"datetime": pd.to_datetime(ds["valid_time"].values)})

    # Solar radiation: ERA5 gives accumulated J/m² per hour → divide by 3600 → W/m²
    ssrd = ds["ssrd"].values
    ssrd_hourly        = np.diff(ssrd, prepend=0)
    ssrd_hourly        = np.where(ssrd_hourly < 0, ssrd, ssrd_hourly)  # reset at midnight
    df["solar_radiation"] = np.maximum(ssrd_hourly / 3600, 0)

    # Temperature: K → °C
    df["temp"]      = ds["t2m"].values - 273.15
    df["dew_point"] = ds["d2m"].values - 273.15

    # Wind: components → speed
    u10 = ds["u10"].values
    v10 = ds["v10"].values
    df["wind_speed"]     = np.sqrt(u10**2 + v10**2)
    df["wind_direction"] = (np.degrees(np.arctan2(u10, v10)) + 360) % 360

    # Precipitation: m per hour → mm
    tp = ds["tp"].values
    tp_hourly     = np.diff(tp, prepend=0)
    tp_hourly     = np.where(tp_hourly < 0, tp, tp_hourly)
    df["precipitation"] = np.maximum(tp_hourly * 1000, 0)

    # Cloud cover: 0–1 → %
    df["clouds"] = ds["tcc"].values * 100

    # Humidity: derive from temp + dew_point (Magnus formula)
    e  = 6.112 * np.exp(17.67 * df["dew_point"] / (df["dew_point"] + 243.5))
    es = 6.112 * np.exp(17.67 * df["temp"]      / (df["temp"]      + 243.5))
    df["humidity"] = (e / es * 100).clip(0, 100)

    # Pressure: Pa → hPa
    df["pressure"] = ds["sp"].values / 100

    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)

    print(f"  Rows:  {len(df):,}")
    print(f"  Range: {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"  Solar radiation — mean: {df['solar_radiation'].mean():.1f} W/m², "
          f"max: {df['solar_radiation'].max():.1f} W/m²")
    print(f"  Temperature    — mean: {df['temp'].mean():.1f}°C")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved → {OUTPUT_CSV}")
    return df


if __name__ == "__main__":
    if OUTPUT_CSV.exists():
        print(f"CSV already exists: {OUTPUT_CSV} — skipping download.")
        print("Delete it and re-run to refresh.")
        df = pd.read_csv(OUTPUT_CSV)
    else:
        ds = download_era5()
        df = process_era5(ds)

    print("\nSample:")
    print(df[["datetime", "solar_radiation", "temp", "clouds", "humidity", "precipitation"]].head(10))
