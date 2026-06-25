# src/analysis/xai.py
# XAI module — SHAP-based explainability for quantile solar forecast models.
#
# Architecture (fits existing CSV-communication pattern):
#   Reads:  data/outputs/features_14day.csv      ← raw hourly feature matrix (from forecast_pv.py)
#           data/models/lgbm_model_q05/q50/q95.pkl
#   Writes: data/outputs/shap_14day.csv          ← per-timestep SHAP values (wide format)
#           data/outputs/shap_summary.csv         ← mean |SHAP| per feature × season
#
# Thesis contributions enabled:
#   1. Quantile SHAP divergence (q95 - q05 SHAP) → what drives UNCERTAINTY, not just prediction
#   2. Seasonal SHAP shift → which features matter in summer vs winter
#   3. Uncertainty driver ranking → top features that widen/narrow the prediction interval
#
# Run standalone:  python -m analysis.xai
# Or via pipeline: imported by run_pipeline.py (Step 6)
#
# Install dep:  pip install shap

import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.train_pv_real import FEATURES

SEASON_MAP = {
    12: "Winter", 1: "Winter",  2: "Winter",
    3:  "Spring", 4: "Spring",  5: "Spring",
    6:  "Summer", 7: "Summer",  8: "Summer",
    9:  "Autumn", 10: "Autumn", 11: "Autumn",
}


# ── MODEL LOADING ─────────────────────────────────────────
def load_models():
    models_dir = PROJECT_ROOT / "data" / "models"
    models = {}
    # solar-sme-actual: prefer Model B (lgbm_real_*) for SHAP
    for q in ["q05", "q50", "q95"]:
        for prefix in ("lgbm_real_", "lgbm_model_", "xgb_model_"):
            path = models_dir / f"{prefix}{q}.pkl"
            if path.exists():
                models[q] = joblib.load(path)
                print(f"  Loaded {path.name}")
                break
        if q not in models:
            raise FileNotFoundError(f"No model file found for {q} in {models_dir}")
    return models


# ── SHAP COMPUTATION ──────────────────────────────────────
def compute_shap(models: dict, X: pd.DataFrame) -> dict:
    """
    Compute TreeSHAP values for all 3 quantile models.

    Returns:
        dict {q05, q50, q95} → np.ndarray of shape (n_samples, n_features)

    TreeSHAP is exact (not approximate) and fast for tree-based models.
    Works natively with both LightGBM and XGBoost.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("Install shap:  pip install shap")

    shap_values = {}
    for q, model in models.items():
        print(f"  Computing SHAP for {q} ({len(X)} rows)...")
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        # LightGBM may return a list for multi-class; for regression it's a 2D array
        if isinstance(sv, list):
            sv = sv[0]
        shap_values[q] = np.array(sv)
        print(f"    Done. Shape: {shap_values[q].shape}")
    return shap_values


# ── BUILD OUTPUT DATAFRAMES ───────────────────────────────
def build_shap_df(shap_values: dict, datetimes: pd.Series) -> pd.DataFrame:
    """
    Wide-format output:
      datetime | {feat}_q50 | {feat}_q05 | {feat}_q95 | {feat}_uncertainty

    uncertainty = shap_q95 - shap_q05
      → positive: feature WIDENS the interval at this timestep
      → negative: feature NARROWS the interval
    """
    cols = {"datetime": datetimes.values}
    for i, feat in enumerate(FEATURES):
        cols[f"{feat}_q50"]        = shap_values["q50"][:, i]
        cols[f"{feat}_q05"]        = shap_values["q05"][:, i]
        cols[f"{feat}_q95"]        = shap_values["q95"][:, i]
        cols[f"{feat}_uncertainty"] = shap_values["q95"][:, i] - shap_values["q05"][:, i]

    return pd.DataFrame(cols)


def build_shap_summary(shap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary table: mean |SHAP| per feature × season.
    Columns: feature | importance_global | importance_{season} ×4 | uncertainty_driver

    This is the main thesis table — shows seasonal SHAP shift.
    A feature that ranks high in summer but low in winter tells a story.
    """
    shap_df = shap_df.copy()
    shap_df["season"] = pd.to_datetime(shap_df["datetime"]).dt.month.map(SEASON_MAP)

    rows = []
    for feat in FEATURES:
        row = {"feature": feat}

        # Global: mean |SHAP| for the median model (q50)
        row["importance_global"] = shap_df[f"{feat}_q50"].abs().mean()

        # Per season. Absent seasons get NaN (NOT 0.0): a season with zero rows
        # in the data window has UNKNOWN importance, not zero importance. Writing
        # 0.0 here is what made a forward 14-day forecast (all one season) look
        # like every feature mattered only in summer.
        for season in ["Winter", "Spring", "Summer", "Autumn"]:
            mask = shap_df["season"] == season
            row[f"importance_{season.lower()}"] = (
                shap_df.loc[mask, f"{feat}_q50"].abs().mean() if mask.sum() > 0 else np.nan
            )

        # Uncertainty driver: mean |SHAP_q95 - SHAP_q05|
        # High value → this feature is responsible for wide prediction bands
        row["uncertainty_driver"] = shap_df[f"{feat}_uncertainty"].abs().mean()

        rows.append(row)

    summary = (
        pd.DataFrame(rows)
        .sort_values("importance_global", ascending=False)
        .reset_index(drop=True)
    )
    return summary


# ── PRINT HELPERS ─────────────────────────────────────────
def _bar(value, max_value, width=30):
    filled = int(value / max_value * width) if max_value > 0 else 0
    return "█" * filled


def print_report(summary: pd.DataFrame, seasons_present=None):
    print("\n── Top 10 features by global importance (q50 model) ──")
    top10 = summary.head(10)
    max_g = summary["importance_global"].max()
    for _, row in top10.iterrows():
        print(f"  {row['feature']:<30} {row['importance_global']:.4f}  "
              f"{_bar(row['importance_global'], max_g)}")

    print("\n── Top 8 UNCERTAINTY DRIVERS (|SHAP_q95 − SHAP_q05|) ──")
    unc_top = summary.sort_values("uncertainty_driver", ascending=False).head(8)
    max_u = summary["uncertainty_driver"].max()
    for _, row in unc_top.iterrows():
        print(f"  {row['feature']:<30} {row['uncertainty_driver']:.4f}  "
              f"{_bar(row['uncertainty_driver'], max_u)}")

    # Seasonal shift is only meaningful when the data window spans ≥2 seasons.
    # A forward 14-day forecast lives in a single season, so the cross-season
    # comparison is degenerate — suppress it rather than print misleading zeros.
    seasons = sorted(seasons_present) if seasons_present is not None else []
    if len(seasons) < 2:
        only = seasons[0] if seasons else "n/a"
        print("\n── Seasonal importance shift — SKIPPED ──")
        print(f"  Data window covers a single season ({only}); a cross-season")
        print(f"  comparison needs ≥2 seasons. Run XAI over the full training set")
        print(f"  (Soesterberg_KNMI_10min_cleaned.csv) for the retrospective table.")
        return

    print("\n── Seasonal importance shift (mean |SHAP q50|) ──")
    print(f"  {'Feature':<30} {'Global':>7} {'Winter':>7} {'Spring':>7} {'Summer':>7} {'Autumn':>7}")
    print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    def _cell(v):
        return f"{v:>7.4f}" if pd.notna(v) else f"{'—':>7}"

    for _, row in summary.head(12).iterrows():
        print(
            f"  {row['feature']:<30} "
            f"{_cell(row['importance_global'])} "
            f"{_cell(row['importance_winter'])} "
            f"{_cell(row['importance_spring'])} "
            f"{_cell(row['importance_summer'])} "
            f"{_cell(row['importance_autumn'])}"
        )


# ── SHAP CORE (shared by forward & retrospective modes) ───
def _explain_and_save(models: dict, feat_day: pd.DataFrame,
                      shap_name: str, summary_name: str):
    """Compute SHAP on an already-daytime-filtered feature frame and save.

    feat_day must contain `datetime` + all FEATURES columns.
    """
    X         = feat_day[FEATURES].fillna(0)
    datetimes = feat_day["datetime"].reset_index(drop=True)
    print(f"  Daytime rows for SHAP: {len(X)}")

    print("\n── Computing SHAP values ──")
    shap_values = compute_shap(models, X)

    shap_df      = build_shap_df(shap_values, datetimes)
    shap_summary = build_shap_summary(shap_df)

    out_dir      = PROJECT_ROOT / "data" / "outputs"
    shap_path    = out_dir / shap_name
    summary_path = out_dir / summary_name

    shap_df.to_csv(shap_path, index=False)
    shap_summary.to_csv(summary_path, index=False)

    print(f"\nSaved → {shap_path.relative_to(PROJECT_ROOT)}"
          f"  ({len(shap_df)} rows, {len(shap_df.columns)} cols)")
    print(f"Saved → {summary_path.relative_to(PROJECT_ROOT)}"
          f"  ({len(shap_summary)} features)")

    seasons_present = set(datetimes.dt.month.map(SEASON_MAP).dropna().unique())
    print_report(shap_summary, seasons_present)

    return shap_df, shap_summary


# ── FORWARD MODE (14-day forecast window) ─────────────────
def run_xai():
    print("\n── Loading models ──")
    models = load_models()

    # Load raw hourly feature matrix saved by forecast_pv.py (Step 1)
    features_path = PROJECT_ROOT / "data" / "outputs" / "features_14day.csv"
    if not features_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at: {features_path}\n"
            "Run Step 1 (forecast_pv.py / run_pipeline.py) first — it writes this file."
        )

    print(f"\n── Loading feature matrix ({features_path.name}) ──")
    feat_df = pd.read_csv(features_path, parse_dates=["datetime"])
    print(f"  Rows: {len(feat_df)}  |  Cols: {len(feat_df.columns)}")

    # Day-time only — matches training (solar_elevation > 0)
    feat_day = feat_df[feat_df["solar_elevation"] > 0].reset_index(drop=True)
    return _explain_and_save(models, feat_day, "shap_14day.csv", "shap_summary.csv")


# ── RETROSPECTIVE MODE (full cleaned-weather history) ─────
def run_xai_full(sample: int | None = 20_000):
    """Retrospective SHAP over the full cleaned KNMI weather history.

    Reads data/cleaned/Soesterberg_KNMI_10min_cleaned.csv, engineers the SAME
    canonical features used in training, and explains the trained model across
    ALL seasons — this is the source for the seasonal-shift thesis table.

    Outputs go to shap_full.csv / shap_summary_full.csv so the forward 14-day
    outputs are never overwritten.

    sample : cap on daytime rows (random, seed=42) for tractable TreeSHAP.
             Set None to use every row. 20k rows give stable mean|SHAP| and
             still span all four seasons across the 2014–2017 record.
    """
    print("\n── Loading models ──")
    models = load_models()

    weather_path = PROJECT_ROOT / "data" / "cleaned" / "Soesterberg_KNMI_10min_cleaned.csv"
    if not weather_path.exists():
        raise FileNotFoundError(f"Cleaned weather not found at: {weather_path}")

    print(f"\n── Loading cleaned weather ({weather_path.name}) ──")
    weather = pd.read_csv(weather_path, parse_dates=["datetime"])
    print(f"  Rows: {len(weather):,}  "
          f"({weather['datetime'].min().date()} → {weather['datetime'].max().date()})")

    # CANONICAL feature engineering — identical to the training builder so the
    # model sees the same feature distribution it was fit on. engenering/ is not
    # a package, so add it to the path before importing.
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "engenering"))
    from engineering import engineer_features

    print("── Engineering canonical features ──")
    feat_df = engineer_features(weather)

    # Daytime only + drop rows with any missing model input (cleaning leaves
    # real gaps as NaN; we drop rather than fabricate zeros for a retrospective
    # importance study).
    feat_day = feat_df[feat_df["solar_elevation"] > 0].dropna(subset=FEATURES)
    print(f"  Daytime rows with complete features: {len(feat_day):,}")

    if sample is not None and len(feat_day) > sample:
        feat_day = feat_day.sample(sample, random_state=42)
        print(f"  Sampled {sample:,} rows (seed=42) for tractable TreeSHAP")

    feat_day = feat_day.sort_values("datetime").reset_index(drop=True)
    return _explain_and_save(models, feat_day, "shap_full.csv", "shap_summary_full.csv")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="SHAP explainability for the PV quantile models.")
    ap.add_argument("--source", choices=["forecast", "full"], default="forecast",
                    help="forecast = 14-day window (default); full = retrospective over "
                         "cleaned KNMI history (all seasons).")
    ap.add_argument("--sample", type=int, default=20_000,
                    help="[full only] cap on daytime rows for TreeSHAP. 0 = use all.")
    args = ap.parse_args()

    if args.source == "full":
        run_xai_full(sample=None if args.sample == 0 else args.sample)
    else:
        run_xai()
