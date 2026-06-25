"""
Aggregate ID003 1-minute production to 10-minute capacity factor, ON THE SAME GRID
as the KNMI 10-minute weather (data/raw/Soesterberg_KNMI_10min.csv) so the two join
row-for-row on `datetime`.

Source: data/raw/id003_production_raw_extracted.csv
        2,103,839 rows, 1-min, 2014-01-01 → 2017-12-31, UTC.
        Column "production_kw" is MISNAMED — values are WATTS (peak ~2268 W matches
        the 2250 W AC rating). Do NOT divide by 1000.

Output: data/raw/id003_production_10min.csv  (~210k rows, 10-min)
    datetime          — naive UTC, END-labelled 10-min grid (matches KNMI)
    production_w       — mean AC power over the 10-min interval (W)
    capacity_factor    — mean (production_w / 2250), clipped [0, 1.1]
    valid_minutes      — how many of the 10 one-minute readings were present (0–10)
    minute_max_cf      — peak 1-min CF inside the interval (for ramp analysis)
    q_clipping         — 1 if capacity_factor ≥ 0.95 (inverter likely saturating)

GRID ALIGNMENT (important):
    KNMI 10-min observations are END-labelled — the 12:00 record covers 11:50–12:00.
    So we bin production as [T-10min, T) labelled T (resample label='right',
    closed='left'): bin "12:00" = minutes 11:50…11:59. Production timestamps then
    coincide with the KNMI weather timestamps for a clean merge.

Cleanup rules (mirror aggregate_id003_to_hourly.py):
    • Negative production clipped to 0 (sensor noise dips slightly below 0 at night)
    • Missing 1-min readings are skipped in the mean and counted in valid_minutes
    • All 10-min bins kept (complete grid); filter on valid_minutes downstream
"""

import sys
import pandas as pd
from pathlib import Path

# Force UTF-8 stdout so arrow/box chars work on any terminal (cp1252 etc.)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── CONSTANTS ─────────────────────────────────────────────
# production_kw column is actually WATTS (verified: peak ~2268 W ≈ 2250 W AC rating).
AC_CAPACITY_W = 2250                    # from metadata.csv → estimated_ac_capacity
AC_CAPACITY_KW = AC_CAPACITY_W / 1000.0 # = 2.25 kWp (for kWh totals only)

CLIP_THRESHOLD = 0.95                   # 10-min CF ≥ this → flag as inverter clipping
CF_MAX = 1.1                            # absolute upper clip (allows minor over-rating)

INPUT_PATH  = PROJECT_ROOT / "data" / "raw" / "id003_production_raw_extracted.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "cleaned" / "id003_production_10min.csv"


def main():
    print(f"Loading {INPUT_PATH.name} ...")
    df = pd.read_csv(INPUT_PATH, parse_dates=["datetime"])
    print(f"  Raw rows:    {len(df):,}")
    print(f"  Date range:  {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"  Non-null:    {df['production_kw'].notna().sum():,}  "
          f"({df['production_kw'].notna().mean()*100:.1f}%)")

    # Strip timezone to naive UTC (KNMI 10-min file is naive UTC)
    if df["datetime"].dt.tz is not None:
        df["datetime"] = df["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)

    # Rename to reflect true units (column was misnamed in source data)
    df = df.rename(columns={"production_kw": "production_w"})

    # Clip negatives (sensor noise) — keeps NaN as NaN so valid_minutes stays honest
    df["production_w"] = df["production_w"].clip(lower=0)

    # Per-minute CF (W / W, dimensionless)
    df["cf_minute"] = df["production_w"] / AC_CAPACITY_W

    # Resample to the END-labelled 10-min grid that matches KNMI weather.
    print("Resampling 1-min → 10-min (END-labelled, KNMI-aligned) ...")
    df = df.set_index("datetime")
    grp = df["cf_minute"].resample("10min", label="right", closed="left")
    cf_mean = grp.mean()

    result = pd.DataFrame({
        "datetime":        cf_mean.index,
        "production_w":    (cf_mean * AC_CAPACITY_W).values,   # mean power over the bin
        "capacity_factor": cf_mean.values,
        "valid_minutes":   grp.count().values,                 # 0–10
        "minute_max_cf":   grp.max().values,
    })

    # Flag clipping bins (kept, just labelled) on the UNCLIPPED mean, then clamp.
    result["q_clipping"] = (result["capacity_factor"] >= CLIP_THRESHOLD).astype(int)
    result["capacity_factor"] = result["capacity_factor"].clip(0, CF_MAX)
    result["production_w"] = result["production_w"].clip(0, CF_MAX * AC_CAPACITY_W)

    result = result[["datetime", "production_w", "capacity_factor",
                     "valid_minutes", "minute_max_cf", "q_clipping"]]
    result.to_csv(OUTPUT_PATH, index=False)

    # ── Sanity checks ────────────────────────────────────
    daylight   = (result["capacity_factor"] > 0.01).sum()
    full_bins  = (result["valid_minutes"] == 10).sum()
    empty_bins = (result["valid_minutes"] == 0).sum()
    years      = (result["datetime"].max() - result["datetime"].min()).days / 365.25
    # Energy: mean power (W) per 10-min × (10/60) h, summed → Wh → kWh.
    total_kwh  = (result["capacity_factor"].fillna(0) * AC_CAPACITY_KW * (10/60)).sum()
    annual_per_kwp = (total_kwh / years) / AC_CAPACITY_KW

    print(f"\nSaved → {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  10-min rows:         {len(result):,}")
    print(f"  Range:               {result['datetime'].min()} → {result['datetime'].max()}")
    step = result["datetime"].diff().mode()
    print(f"  Step:                {step.iloc[0] if not step.empty else 'n/a'}  (expected 0 days 00:10:00)")
    print(f"  Full bins (10/10):   {full_bins:,}  ({full_bins/len(result)*100:.1f}%)")
    print(f"  Empty bins (0/10):   {empty_bins:,}  ({empty_bins/len(result)*100:.1f}%)")
    print(f"  Daylight bins:       {daylight:,}")
    print(f"  Clipping flagged:    {result['q_clipping'].sum():,} "
          f"({result['q_clipping'].mean()*100:.2f}%)")
    print(f"  CF range:            {result['capacity_factor'].min():.4f} – "
          f"{result['capacity_factor'].max():.4f}")
    print(f"  Annual kWh/kWp:      {annual_per_kwp:.0f}  (reference ~1015)")


if __name__ == "__main__":
    main()
