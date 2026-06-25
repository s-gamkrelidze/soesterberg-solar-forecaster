
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

TARGET    = "capacity_factor"
QUANTILES = {"q05": 0.05, "q50": 0.5, "q95": 0.95}

FEATURES = [
    # Physics
    "solar_radiation", "effective_radiation", "adjusted_radiation",
    "solar_elevation", "solar_elevation_sin", "solar_elevation_cos",
    "solar_azimuth", "air_mass",
    "cos_incidence", "poa_estimate",
    "cell_temp_est", "cell_temp_derating",
    "poa_temp_corrected",
    # Clear-sky decomposition
    "clearsky_ghi", "clearsky_index",
    "clearsky_poa", "clearsky_poa_index",
    # Weather
    "clouds", "temp", "humidity", "wind_speed",
    "thermal_factor", "temp_delta_25", "wind_cooling_effect",
    # Interaction
    "clouds_x_elevation", "humidity_x_elevation", "humidity_radiation",
    # Rain / soiling
    "hours_since_rain",
    "rain_last_24h", "dew_spread",
    # Lag & rolling — weather inputs
    "solar_lag_1h", "solar_lag_24h", "cloud_lag_1h",
    "clouds_3h_mean", "radiation_3h_mean", "cloud_trend_3h",
    # Time (integer + Fourier)
    "hour", "month", "dayofyear",
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "doy_sin", "doy_cos",
    # Cloud motion & extreme events
    "radiation_change_1h", "cloud_change_1h",
    "cloud_burst_flag", "irradiance_drop_flag",
]

REAL_TRAINING_CSV = PROJECT_ROOT / "data" / "models" / "training_dataset_real-weather-generation.csv"
MODEL_PREFIX      = "lgbm_real"
CONFORMAL_NAME    = "conformal_correction_real.pkl"
METRICS_NAME      = "model_metrics_real.csv"


# ── DATA LOADING ───────────────────────────────────────────
def load_training_data_real():
    '''
    if not REAL_TRAINING_CSV.exists():
        sys.exit(
            f"Missing {REAL_TRAINING_CSV.name}. Build it first:\n"
            f"    python src/pipeline/build_soesterberg_training_real.py"
        )
    '''
    df = pd.read_csv(REAL_TRAINING_CSV, parse_dates=["datetime"])
    df = df[df["solar_elevation"] > 0].copy()
    df = df.dropna(subset=FEATURES + [TARGET])
    print(f"\n[Model B — REAL measured target]")
    print(f"Training rows (daylight only): {len(df):,}")
    print(f"Date range: {df['datetime'].min().date()} → {df['datetime'].max().date()}")
    print(f"Cloud burst events:            {df['cloud_burst_flag'].sum():,}")
    print(f"Irradiance drop events:        {df['irradiance_drop_flag'].sum():,}")
    print(f"Mean clear-sky index:          {df['clearsky_index'].mean():.3f}")
    print(f"Target source:                 ID003 measured (real inverter)")
    return df


# ── SPLIT ─────────────────────────────────────────────────
def split_data(df: pd.DataFrame):
    """Chronological split snapped to month boundaries (~70 / 15 / 15 %)."""
    df = df.sort_values("datetime").reset_index(drop=True)
    n  = len(df)

    def _next_month_start(ts):
        return (ts.replace(day=1) + pd.DateOffset(months=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    train_cutoff = _next_month_start(df["datetime"].iloc[int(n * 0.70)])
    calib_cutoff = _next_month_start(df["datetime"].iloc[int(n * 0.85)])

    train = df[df["datetime"] <  train_cutoff]
    calib = df[(df["datetime"] >= train_cutoff) & (df["datetime"] < calib_cutoff)]
    test  = df[df["datetime"] >= calib_cutoff]

    if len(calib) < 200:
        raise ValueError(
            f"Calib set has only {len(calib)} rows — too small for Mondrian calibration."
        )

    print(f"Train: {len(train):,}  |  Calib: {len(calib):,}  |  Test: {len(test):,}")
    print(f"  {train['datetime'].min().date()} → {train['datetime'].max().date()}"
          f"  |  {calib['datetime'].min().date()} → {calib['datetime'].max().date()}"
          f"  |  {test['datetime'].min().date()} → {test['datetime'].max().date()}")
    return train, calib, test


# ── TRAIN ─────────────────────────────────────────────────
def train_quantile_models(train_df: pd.DataFrame, calib_df: pd.DataFrame) -> dict:
    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_calib, y_calib = calib_df[FEATURES], calib_df[TARGET]
    models = {}
    for name, q in QUANTILES.items():
        print(f"Training LightGBM {name} (α={q}) ...")
        model = LGBMRegressor(
            objective         = "quantile",
            alpha             = q,
            n_estimators      = 2000,
            max_depth         = 6,
            num_leaves        = 31,
            learning_rate     = 0.05,
            subsample         = 0.8,
            subsample_freq    = 1,
            colsample_bytree  = 0.8,
            min_child_samples = 20,
            n_jobs            = -1,
            random_state      = 42,
            verbose           = -1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_calib, y_calib)],
            callbacks=[
                early_stopping(stopping_rounds=50, verbose=False),
                log_evaluation(period=0),
            ],
        )
        print(f"  Best iteration: {model.best_iteration_}")
        models[name] = model
    return models


# ── MONDRIAN CONFORMAL CALIBRATION ────────────────────────
MAX_CORRECTION = 0.10

def fit_conformal_mondrian(models: dict, calib_df: pd.DataFrame,
                           target_coverage: float = 0.80) -> dict:
    X_calib = calib_df[FEATURES]
    y_calib = calib_df[TARGET].values
    q05 = models["q05"].predict(X_calib).clip(0, 1)
    q95 = models["q95"].predict(X_calib).clip(0, 1)

    extreme_mask = (
        (calib_df["cloud_burst_flag"].values     == 1) |
        (calib_df["irradiance_drop_flag"].values == 1)
    )
    normal_mask = ~extreme_mask
    tail_q = (1 + target_coverage) / 2

    def _quantile_correction(y, lo, hi, mask):
        if mask.sum() < 20:
            return None, None
        lower_scores = np.maximum(lo[mask] - y[mask], 0)
        upper_scores = np.maximum(y[mask] - hi[mask], 0)
        cl = min(float(np.quantile(lower_scores, tail_q)), MAX_CORRECTION)
        cu = min(float(np.quantile(upper_scores, tail_q)), MAX_CORRECTION)
        return cl, cu

    nl, nu = _quantile_correction(y_calib, q05, q95, normal_mask)
    el, eu = _quantile_correction(y_calib, q05, q95, extreme_mask)
    if el is None:
        el, eu = nl, nu

    correction = {
        "normal_lower":  nl, "normal_upper":  nu,
        "extreme_lower": el, "extreme_upper": eu,
    }

    def _coverage(mask, cl, cu):
        if mask.sum() == 0:
            return float("nan"), float("nan")
        pre  = np.mean((y_calib[mask] >= q05[mask]) & (y_calib[mask] <= q95[mask])) * 100
        post = np.mean(
            (y_calib[mask] >= (q05[mask] - cl).clip(0, 1)) &
            (y_calib[mask] <= (q95[mask] + cu).clip(0, 1))
        ) * 100
        return pre, post

    n_pre, n_post = _coverage(normal_mask,  nl, nu)
    e_pre, e_post = _coverage(extreme_mask, el, eu)
    print(f"\n── Mondrian Conformal (target {target_coverage*100:.0f}%, cap {MAX_CORRECTION}) ─")
    print(f"  Normal  ({normal_mask.sum():>5} rows): "
          f"corr=(-{nl:.4f}, +{nu:.4f})  coverage {n_pre:5.1f}% → {n_post:5.1f}%")
    print(f"  Extreme ({extreme_mask.sum():>5} rows): "
          f"corr=(-{el:.4f}, +{eu:.4f})  coverage {e_pre:5.1f}% → {e_post:5.1f}%")
    return correction


def apply_mondrian_correction(q05_raw, q95_raw, extreme_mask, correction):
    lower   = np.where(extreme_mask, correction["extreme_lower"], correction["normal_lower"])
    upper   = np.where(extreme_mask, correction["extreme_upper"], correction["normal_upper"])
    q05_adj = (q05_raw - lower).clip(0, 1)
    q95_adj = (q95_raw + upper).clip(0, 1)
    return q05_adj, q95_adj


# ── EVALUATE ──────────────────────────────────────────────
def evaluate(models: dict, test_df: pd.DataFrame, correction: dict,
             system_kwp: float = 2.25) -> dict:
    # ID003 AC rating = 2250 W; capacity_factor is normalized by this same value
    # (see id003_1_minutes_to_10_minutes.py), so 2.25 is the only consistent
    # CF→kWh base. (DC rating is 2.503 kWp — not the normalization base.)
    X_test  = test_df[FEATURES].reset_index(drop=True)
    y_true  = test_df[TARGET].reset_index(drop=True)
    y_pred  = models["q50"].predict(X_test).clip(0, 1)
    q05_raw = models["q05"].predict(X_test).clip(0, 1)
    q95_raw = models["q95"].predict(X_test).clip(0, 1)

    extreme_mask = (
        (test_df["cloud_burst_flag"].values     == 1) |
        (test_df["irradiance_drop_flag"].values == 1)
    )
    q05_adj, q95_adj = apply_mondrian_correction(q05_raw, q95_raw, extreme_mask, correction)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    peak_mask     = y_true >= y_true.quantile(0.9)
    peak_error    = mean_absolute_error(y_true[peak_mask], y_pred[peak_mask])
    ramp_error    = mean_absolute_error(y_true.diff().dropna(), pd.Series(y_pred).diff().dropna())
    extreme_error = (mean_absolute_error(y_true[extreme_mask], y_pred[extreme_mask])
                     if extreme_mask.sum() > 0 else float("nan"))

    cov_raw    = np.mean((y_true >= q05_raw) & (y_true <= q95_raw)) * 100
    cov_adj    = np.mean((y_true >= q05_adj) & (y_true <= q95_adj)) * 100
    band_w     = (q95_adj - q05_adj).mean()

    print(f"\n── LightGBM + Mondrian Conformal — Final Test Metrics ──")
    print(f"  R²:                          {r2:.4f}")
    print(f"  MAE:                         {mae:.4f} CF  →  ±{mae*system_kwp:.3f} kWh/h @ {system_kwp:g} kW AC")
    print(f"  RMSE:                        {rmse:.4f}")
    print(f"  Peak error (top 10%):        {peak_error:.4f} CF")
    print(f"  Ramp error:                  {ramp_error:.4f} CF")
    print(f"  Extreme event error:         {extreme_error:.4f} CF")
    print(f"  Band coverage (raw):         {cov_raw:.1f}%")
    print(f"  Band coverage (conformal):   {cov_adj:.1f}%")
    print(f"  Avg band width:              {band_w:.4f} CF")

    return {
        "model":                        "LightGBM_real_q05_q50_q95_mondrian",
        "r2":                           round(r2, 4),
        "mae":                          round(mae, 4),
        "rmse":                         round(rmse, 4),
        "peak_error":                   round(peak_error, 4),
        "ramp_error":                   round(ramp_error, 4),
        "extreme_event_error":          round(extreme_error, 4),
        "band_coverage_raw_pct":        round(cov_raw, 1),
        "band_coverage_conformal_pct":  round(cov_adj, 1),
        "conformal_normal_lower":       round(correction["normal_lower"],  4),
        "conformal_normal_upper":       round(correction["normal_upper"],  4),
        "conformal_extreme_lower":      round(correction["extreme_lower"], 4),
        "conformal_extreme_upper":      round(correction["extreme_upper"], 4),
    }


# ── FEATURE IMPORTANCE ────────────────────────────────────
def print_top_features(models: dict, n: int = 15) -> None:
    imp     = pd.Series(models["q50"].feature_importances_, index=FEATURES)
    imp_max = imp.max() if imp.max() > 0 else 1
    print(f"\nTop {n} features (q50 model):")
    for feat, score in imp.sort_values(ascending=False).head(n).items():
        bar = "█" * int(score / imp_max * 40)
        print(f"  {feat:<28} {int(score):>6}  {bar}")


# ── SAVE ──────────────────────────────────────────────────
def save_models_real(models: dict, metrics: dict, correction: dict) -> None:
    models_dir  = PROJECT_ROOT / "data" / "models"
    outputs_dir = PROJECT_ROOT / "data" / "outputs"
    for name, model in models.items():
        path = models_dir / f"{MODEL_PREFIX}_{name}.pkl"
        joblib.dump(model, path)
        print(f"Saved → data/models/{path.name}")

    cpath = models_dir / CONFORMAL_NAME
    joblib.dump(correction, cpath)
    print(f"Saved → data/models/{CONFORMAL_NAME}  (Mondrian — 4 corrections)")

    mpath = outputs_dir / METRICS_NAME
    pd.DataFrame([metrics]).to_csv(mpath, index=False)
    print(f"Saved → data/outputs/{METRICS_NAME}")


if __name__ == "__main__":
    df                 = load_training_data_real()
    train, calib, test = split_data(df)
    models             = train_quantile_models(train, calib)
    correction         = fit_conformal_mondrian(models, calib, target_coverage=0.80)
    metrics            = evaluate(models, test, correction)
    print_top_features(models)
    save_models_real(models, metrics, correction)
