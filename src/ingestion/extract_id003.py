import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE_CSV   = PROJECT_ROOT / "data" / "raw" / "filtered_pv_power_measurements_ac.csv"
OUTPUT_CSV   = PROJECT_ROOT / "data" / "raw" / "id003_production_raw_extracted.csv"

CHUNKSIZE = 100_000


def extract_id003():
    print(f"Reading: {SOURCE_CSV.name}")

    # Peek at header to find exact column name (may have trailing space)
    header = pd.read_csv(SOURCE_CSV, nrows=0)
    id003_col = next((c for c in header.columns if c.strip() == "ID003"), None)
    if id003_col is None:
        raise ValueError(f"ID003 column not found. Columns: {list(header.columns)}")
    print(f"  Found column: {repr(id003_col)}")

    chunks = []
    for chunk in pd.read_csv(SOURCE_CSV, usecols=["DateTime", id003_col],
                              chunksize=CHUNKSIZE):
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)

    # Normalise column name and datetime
    df = df.rename(columns={id003_col: "production_kw", "DateTime": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)

    df.to_csv(OUTPUT_CSV, index=False)

    print(f"  Rows:  {len(df):,}")
    print(f"  Range: {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"  Non-zero readings: {(df['production_kw'] > 0).sum():,}")
    print(f"Saved → {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")
    return df


if __name__ == "__main__":
    extract_id003()