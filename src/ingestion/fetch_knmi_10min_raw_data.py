
#
# Collection: 10-minute-in-situ-meteorological-observations  (archive since 2012)
# Location:   0-20000-0-06260  (De Bilt; WIGOS id)
# Endpoint:   /edr/v1/collections/{coll}/locations/{loc}?datetime=START/END&parameter-name=...
# Response:   CoverageJSON — coverages[0].domain.axes.t.values (UTC timestamps) zipped
#             with coverages[0].ranges[param].values.
#
# KNMI param → output schema (native units, verified against the file API for De Bilt):
#   qg  (W/m², 10-min mean)  → solar_radiation   (MEASURED irradiance — the key field)
#   ta  (°C)                 → temp
#   td  (°C)                 → dew_point          (real, measured)
#   rh  (%)                  → humidity
#   ff  (m/s, 10-min mean)   → wind_speed
#   dd  (°)                  → wind_direction
#   rg  (mm/h, rain gauge)   → precipitation      (× 10/60 → mm per 10-min interval)
#   n   (okta 0–8)           → clouds             (× 12.5 → 0–100 %)
#   pp  (hPa, MSL)           → pressure           (sea-level)
#
# Output: data/raw/Soesterberg_KNMI_10min.csv
#   datetime, solar_radiation, temp, dew_point, wind_speed, wind_direction,
#   precipitation, clouds, humidity, pressure

import os
import sys
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_CSV   = PROJECT_ROOT / "data" / "raw" / "Soesterberg_KNMI_10min.csv"
KEY_FILE     = PROJECT_ROOT / ".knmi_key"

BASE       = "API KEY"
COLLECTION = "10-minute-in-situ-meteorological-observations"
LOCATION   = "0-20000-0-06260"        # De Bilt (06260), WIGOS id

START_DATE = os.environ.get("START_DATE", "2014-01-01")
END_DATE   = os.environ.get("END_DATE",   "2017-12-31")

MAX_DATAPOINTS = 300_000              # EDR per-request cap (timesteps × params)
RETRIES = 4

# KNMI EDR param → (output column, scale).
VAR_MAP = {
    "qg": ("solar_radiation", 1.0),
    "ta": ("temp", 1.0),
    "td": ("dew_point", 1.0),
    "rh": ("humidity", 1.0),
    "ff": ("wind_speed", 1.0),
    "dd": ("wind_direction", 1.0),
    "rg": ("precipitation", 10.0 / 60.0),   # mm/h → mm per 10-min interval
    "n":  ("clouds", 12.5),                 # okta (0–8) → %
    "pp": ("pressure", 1.0),
}
PARAM_NAMES = list(VAR_MAP.keys())
OUT_COLS = ["solar_radiation", "temp", "dew_point", "wind_speed", "wind_direction",
            "precipitation", "clouds", "humidity", "pressure"]


def load_key() -> str:
    key = os.environ.get("KNMI_API_KEY")
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text(encoding="utf-8").strip()
    if not key:
        raise SystemExit(f"No KNMI EDR key. Put it in {KEY_FILE} or env KNMI_API_KEY.")
    return key


def quarter_chunks(start: str, end: str):
    """Yield (start_iso, end_iso) per calendar quarter within [start, end] — each well
    under the 300k-datapoint cap (~13,200 steps × 9 params ≈ 119k)."""
    q_starts = pd.date_range(pd.Timestamp(start).normalize().to_period("Q").start_time,
                             pd.Timestamp(end), freq="QS")
    for qs in q_starts:
        cs = max(qs, pd.Timestamp(start))
        ce = min(qs + pd.offsets.QuarterEnd(0), pd.Timestamp(end)) \
            .replace(hour=23, minute=50)
        yield cs.strftime("%Y-%m-%dT%H:%M:%SZ"), ce.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_chunk(session: requests.Session, key: str, start_iso: str, end_iso: str) -> pd.DataFrame:
    url = f"{BASE}/collections/{COLLECTION}/locations/{LOCATION}"
    params = {"datetime": f"{start_iso}/{end_iso}", "parameter-name": ",".join(PARAM_NAMES)}
    for attempt in range(1, RETRIES + 1):
        r = session.get(url, headers={"Authorization": key}, params=params, timeout=180)
        if r.status_code == 200:
            break
        if r.status_code == 429 and attempt < RETRIES:        # rate limited → back off
            wait = 5 * attempt
            print(f"    HTTP 429 — backing off {wait}s ...")
            time.sleep(wait); continue
        raise RuntimeError(f"EDR request failed (HTTP {r.status_code}): {r.text[:200]}")

    cov = r.json()["coverages"][0]
    t = cov["domain"]["axes"]["t"]["values"]
    df = pd.DataFrame({"datetime": pd.to_datetime(t, utc=True).tz_localize(None)})  # naive UTC
    ranges = cov["ranges"]
    for p, (col, scale) in VAR_MAP.items():
        if p in ranges:
            df[col] = pd.to_numeric(pd.Series(ranges[p]["values"]), errors="coerce") * scale
        else:
            df[col] = np.nan
    return df


def main():
    key = load_key()
    chunks = list(quarter_chunks(START_DATE, END_DATE))
    print(f"KNMI native 10-min via EDR — De Bilt {LOCATION} — {START_DATE} → {END_DATE}")
    print(f"  {len(chunks)} quarterly request(s), 9 parameters each\n")

    parts = []
    with requests.Session() as session:
        for cs, ce in chunks:
            t0 = time.time()
            part = fetch_chunk(session, key, cs, ce)
            parts.append(part)
            print(f"  {cs[:10]} → {ce[:10]}: {len(part):,} steps  ({time.time()-t0:.1f}s)")

    df = pd.concat(parts, ignore_index=True)
    df = df[["datetime", *OUT_COLS]].sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n  Rows:  {len(df):,}")
    print(f"  Range: {df['datetime'].min()} → {df['datetime'].max()}")
    step = df["datetime"].diff().mode()
    print(f"  Step:  {step.iloc[0] if not step.empty else 'n/a'}  (expected 0 days 00:10:00)")
    miss = df[OUT_COLS].isna().sum()
    print(f"  NaN per column: {miss[miss > 0].to_dict() or 'none'}")
    dayl = df["solar_radiation"] > 1
    print(f"  Solar radiation (MEASURED) — mean {df['solar_radiation'].mean():.1f}, "
          f"max {df['solar_radiation'].max():.1f} W/m²  (daylight steps {dayl.sum():,})")
    print(f"Saved → {OUTPUT_CSV}")
    print("\nSample (around solar noon 2017-06-21):")
    noon = df[(df.datetime >= "2017-06-21 11:30") & (df.datetime <= "2017-06-21 12:30")]
    print((noon if not noon.empty else df.head(8))
          [["datetime", "solar_radiation", "temp", "clouds", "humidity", "wind_speed"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
