"""
Clean the KNMI De Bilt 10-minute weather record for the Soesterberg PV project.

Input : data/raw/Soesterberg_KNMI_10min.csv     (EDR pull, already unit-converted)
Output: data/cleaned/Soesterberg_KNMI_10min_cleaned.csv

Philosophy (PhD-grade, reproducible, no fabricated data):
  • Every rule only ever turns a bad value into NaN — it never invents data.
  • Gaps are LEFT as gaps (Option A): the time index is made regular and complete,
    missing observations are explicit NaN. No interpolation / forward-fill.
  • Every action is counted and printed as a QC report so the cleaning is auditable.

NOTE on KNMI special codes. The raw KNMI codes (N=9 sky-invisible, RH=-1 humidity
missing, DD=990 variable wind) were already UNIT-CONVERTED by the fetcher, so in this
file they appear as:
      N=9   → clouds = 9 × 12.5 = 112.5   (the only code that survived; ~2,280 rows)
      RH=-1 → humidity = -1               (EDR already nulled these → 0 present)
      DD=990→ wind_direction = 990        (EDR already nulled these → 0 present)
The physical-range bounds below catch all three regardless (clouds>100, humidity<0,
wind_direction>360 → NaN); we also report the named codes explicitly for transparency.

Pipeline:
  1. Timestamps  — parse robustly (file may be Excel-mangled day-first), drop exact
                   duplicates, sort, REINDEX onto the full regular 10-min grid.
  2. Special codes + physical sanity — out-of-range values → NaN (see BOUNDS).
  3. Consistency — dew_point ≤ temp; humidity 0–100; radiation ≥ 0 (via BOUNDS).
  4. Spikes      — conservative, ONLY on smooth variables (temp, dew_point, pressure,
                   humidity). Radiation / wind / precipitation are EXEMPT because their
                   real 10-min variability is large and would be destroyed by despiking.
  5. Save        — clean ISO datetimes, complete grid, gaps as NaN, + QC report.

Wind direction is circular (0° ≡ 360°): we only NaN the 990 "variable" code here. Do
NOT take a plain arithmetic mean of it downstream — decompose to sin/cos first.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATH  = PROJECT_ROOT / "data" / "raw" / "Soesterberg_KNMI_10min.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "cleaned" / "Soesterberg_KNMI_10min_cleaned.csv"

FREQ = "10min"
VALUE_COLS = ["solar_radiation", "temp", "dew_point", "wind_speed", "wind_direction",
              "precipitation", "clouds", "humidity", "pressure"]

# Physically valid ranges; anything outside → NaN. Bounds chosen for a NL lowland
# station (and to satisfy the stated sanity rules). precipitation is mm per 10-min:
# the all-time ~10-min rainfall record is ~50 mm, so >50 is treated as a sensor spike.
BOUNDS = {
    "temp":            (-60.0, 60.0),
    "dew_point":       (-60.0, 50.0),
    "pressure":        (870.0, 1085.0),
    "wind_speed":      (0.0, 60.0),
    "wind_direction":  (0.0, 360.0),     # 990 "variable" → NaN
    "precipitation":   (0.0, 50.0),
    "clouds":          (0.0, 100.0),     # 112.5 (N=9) → NaN
    "humidity":        (0.0, 100.0),     # -1 (RH missing) → NaN
    "solar_radiation": (0.0, 1400.0),    # > solar constant is impossible
}

# Conservative spike thresholds: a point is a spike if it deviates from the local
# rolling median by MORE than this (physically implausible for a 10-min change).
# Only applied to SMOOTH variables. Window is centred so it judges by both neighbours.
SPIKE_MAX_DEV = {"temp": 8.0, "dew_point": 8.0, "pressure": 6.0, "humidity": 40.0}
SPIKE_WINDOW = 5            # centred ~ ±20 min


class Report:
    """Accumulates (stage, column, n) so the whole cleaning is auditable."""
    def __init__(self):
        self.rows = []
    def add(self, stage, column, n):
        if n:
            self.rows.append((stage, column, int(n)))
    def show(self):
        if not self.rows:
            print("  (no values changed)")
            return
        w = max(len(r[0]) for r in self.rows)
        for stage, col, n in self.rows:
            print(f"    {stage:<{w}}  {col:<18} {n:>8,}")


def load_raw() -> pd.DataFrame:
    print(f"Loading {INPUT_PATH.name} ...")
    df = pd.read_csv(INPUT_PATH)
    # The fetcher writes ISO 8601 (YYYY-MM-DD HH:MM:SS). Parse it as such.
    # DO NOT use dayfirst=True here: on ISO dates it makes pandas read the day
    # field as the month, so days 13–31 become "month 13" → NaT (dropped, ~60%
    # of all rows) and days 1–12 silently swap month/day. Use ISO8601 explicitly.
    df["datetime"] = pd.to_datetime(df["datetime"], format="ISO8601", errors="coerce")
    n_bad = df["datetime"].isna().sum()
    if n_bad:
        print(f"  WARNING: dropped {n_bad} unparseable timestamps")
        df = df.dropna(subset=["datetime"])
    for c in VALUE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"  Rows: {len(df):,}")
    return df


def fix_timestamps(df: pd.DataFrame, rep: Report) -> pd.DataFrame:
    n_dup = df["datetime"].duplicated().sum()
    rep.add("timestamp", "duplicates dropped", n_dup)
    df = df.drop_duplicates("datetime", keep="first")

    if not df["datetime"].is_monotonic_increasing:
        rep.add("timestamp", "re-sorted", 1)
    df = df.sort_values("datetime")

    # Reindex onto the complete regular grid — missing steps become explicit NaN rows.
    full = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq=FREQ)
    n_missing = len(full) - df["datetime"].nunique()
    rep.add("timestamp", "missing steps inserted", n_missing)
    df = df.set_index("datetime").reindex(full).rename_axis("datetime").reset_index()
    return df


def clean_values(df: pd.DataFrame, rep: Report) -> pd.DataFrame:
    # Named special codes — report explicitly (the bounds below also catch them).
    rep.add("special code", "clouds N=9 (112.5)", (df["clouds"] == 112.5).sum())
    rep.add("special code", "wind_dir DD=990",    (df["wind_direction"] == 990).sum())
    rep.add("special code", "humidity RH=-1",     (df["humidity"] == -1).sum())

    # Physical range gate: anything outside [lo, hi] → NaN.
    for col, (lo, hi) in BOUNDS.items():
        bad = (df[col] < lo) | (df[col] > hi)
        rep.add("out-of-range", col, bad.sum())
        df.loc[bad, col] = np.nan

    # Consistency: dew point cannot exceed air temperature. NaN the suspect dew value
    # (small float slack to avoid flagging equality / rounding).
    bad_dew = df["dew_point"] > df["temp"] + 0.1
    rep.add("consistency", "dew_point > temp", bad_dew.sum())
    df.loc[bad_dew, "dew_point"] = np.nan
    return df


def remove_spikes(df: pd.DataFrame, rep: Report) -> pd.DataFrame:
    # Only smooth variables — radiation/wind/precip have real high-frequency swings.
    for col, max_dev in SPIKE_MAX_DEV.items():
        med = df[col].rolling(SPIKE_WINDOW, center=True, min_periods=3).median()
        spike = (df[col] - med).abs() > max_dev
        rep.add("spike", col, spike.sum())
        df.loc[spike, col] = np.nan
    return df


def main():
    rep = Report()
    df = load_raw()

    print("Fixing timestamps (dedup / sort / complete grid) ...")
    df = fix_timestamps(df, rep)

    print("Applying special codes + physical sanity + consistency ...")
    df = clean_values(df, rep)

    print("Removing spikes (smooth variables only) ...")
    df = remove_spikes(df, rep)

    df = df[["datetime", *VALUE_COLS]]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)   # ISO datetimes

    # ── QC report ────────────────────────────────────────
    print("\n── Cleaning actions ──────────────────────────")
    rep.show()

    print("\n── Cleaned dataset ───────────────────────────")
    print(f"  Rows:          {len(df):,}")
    print(f"  Range:         {df['datetime'].min()} → {df['datetime'].max()}")
    step = df["datetime"].diff().mode()
    print(f"  Step:          {step.iloc[0] if not step.empty else 'n/a'}  (regular grid)")
    print("  Missing (NaN) per column:")
    for c in VALUE_COLS:
        pct = df[c].isna().mean() * 100
        print(f"    {c:<16} {df[c].isna().sum():>8,}  ({pct:4.1f}%)")
    # Consistency checks must ignore NaN (NaN comparisons are False and would
    # otherwise raise a spurious VIOLATION on the gap rows).
    both = df["dew_point"].notna() & df["temp"].notna()
    dew_ok = (df.loc[both, "dew_point"] <= df.loc[both, "temp"] + 0.1).all()
    print(f"\n  solar_radiation: {df['solar_radiation'].min():.0f} – {df['solar_radiation'].max():.0f} W/m²")
    print(f"  temp:            {df['temp'].min():.1f} – {df['temp'].max():.1f} °C")
    print(f"  dew ≤ temp:      {'OK' if dew_ok else 'VIOLATION'}")
    print(f"  clouds ≤ 100:    {'OK' if (df['clouds'].dropna() <= 100).all() else 'VIOLATION'}")
    print(f"\nSaved → {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
