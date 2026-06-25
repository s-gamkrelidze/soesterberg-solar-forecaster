

import sys
import pandas as pd
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# CANONICAL feature engineering — single source of truth shared by training and
# inference (remediation C3). The physics/weather/clear-sky/time logic and the
# PV-lag helper live in features/engineering.py; do not re-fork them here.
from engineering import engineer_features


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    weather_path = PROJECT_ROOT / "data" / "cleaned" / "Soesterberg_KNMI_10min_cleaned.csv"
    id003_path   = PROJECT_ROOT / "data" / "cleaned" / "id003_production_10min.csv"
    out_path     = PROJECT_ROOT / "data" / "models" / "training_dataset_real-weather-generation.csv"

    for p in (weather_path, id003_path):
        if not p.exists():
            sys.exit(f"Missing input file: {p}")

    # ── Load weather ───────────────────────────────────────────────────────────
    print(f"Loading {weather_path.name} ...")
    weather = pd.read_csv(weather_path, parse_dates=["datetime"])
    print(f"  Rows: {len(weather):,}  "
          f"({weather['datetime'].min().date()} → {weather['datetime'].max().date()})")
    print(f"  solar_radiation  mean={weather['solar_radiation'].mean():.1f} W/m²  "
          f"max={weather['solar_radiation'].max():.1f} W/m²")

    rad_mean = weather["solar_radiation"].mean()
    if rad_mean > 350:
        sys.exit(
            f"solar_radiation mean {rad_mean:.1f} W/m² is too high for horizontal GHI — "
            f"check that the file contains ERA5 GHI, not POA."
        )

    # ── Load real generation ───────────────────────────────────────────────────
    print(f"Loading {id003_path.name} ...")
    measured = pd.read_csv(id003_path, parse_dates=["datetime"])
    print(f"  Rows: {len(measured):,}  "
          f"({measured['datetime'].min().date()} → {measured['datetime'].max().date()})")
    daylight_cf = measured.loc[measured["capacity_factor"] > 0.01, "capacity_factor"]
    print(f"  Mean daylight CF: {daylight_cf.mean():.4f}")

    # ── Merge (inner — only hours present in both) ─────────────────────────────
    df = weather.merge(
        measured[["datetime", "capacity_factor", "valid_minutes", "minute_max_cf", "q_clipping"]],
        on="datetime",
        how="inner",
    )
    print(f"  After merge: {len(df):,} rows  "
          f"(weather {len(weather):,} ∩ measured {len(measured):,})")
    if len(df) < 25_000:
        print("  WARNING: fewer rows than expected — check timestamp alignment")

    # ── Feature engineering ────────────────────────────────────────────────────
    print("Engineering features ...")
    df = engineer_features(df)

    # ── Save ───────────────────────────────────────────────────────────────────
    df.to_csv(out_path, index=False)

    daylight   = (df["solar_elevation"] > 0).sum()
    cf_day_mean = df.loc[df["solar_elevation"] > 0, "capacity_factor"].mean()
    clipping_n  = int(df["q_clipping"].sum())

    print(f"\nSaved → {out_path.relative_to(PROJECT_ROOT)}")
    print(f"  Total rows:     {len(df):,}")
    print(f"  Daylight rows:  {daylight:,}")
    print(f"  Columns:        {len(df.columns)}")
    print(f"  CF range:       {df['capacity_factor'].min():.4f} – {df['capacity_factor'].max():.4f}")
    print(f"  CF mean (day):  {cf_day_mean:.4f}")
    print(f"  Clipping hours: {clipping_n:,}  ({clipping_n / len(df) * 100:.2f}%)")
    print(f"\n  Weather source: {weather_path.name}")
    print(f"  Target source:  {id003_path.name}  (real inverter, ID003)")


if __name__ == "__main__":
    main()
