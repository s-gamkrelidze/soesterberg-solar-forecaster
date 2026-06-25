"""
study1_stats.py — Plan-B PV thesis: inferential statistics for Study 1.

Wired to the ACTUAL project (B - Soesterbeg-actual). Reuses the project's own
feature list, chronological split and PVWatts physics chain as the single source
of truth (imported from src/models/compare_pvlib_vs_lgbm.py).

Three tests:
  1) Diebold-Mariano  — LightGBM q50 vs PVWatts physics on the IDENTICAL test
                        split  → p-value for "is ML significantly more accurate?"
  2) Walk-forward CV   — R²/MAE/RMSE distribution + 95% CI over expanding folds
                        → evidence of generalisation, not a single lucky split.
  3) Monte Carlo       — 14-day net-saving and balancing-penalty distributions,
                        using the real forecast_14day_15min.csv + project tariffs.

Run:
    cd "B - Soesterbeg-actual"
    python research/study1_stats.py

Deps (already used by this project):
    numpy pandas scipy scikit-learn lightgbm pvlib joblib
Note: walk-forward retrains 8 LightGBM models → expect ~1-3 min.
"""
from __future__ import annotations
import sys
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as tdist
from scipy.special import ndtr                      # standard-normal CDF (copula)
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from lightgbm import LGBMRegressor

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]          # project root
sys.path.insert(0, str(ROOT / "src"))

# Reuse the project's exact physics + split + feature list (single source of truth)
from models.compare_pvlib_vs_lgbm import (          # noqa: E402
    pvlib_cf, split_data, FEATURES, TARGET, CF_MAX,
)

DATA      = ROOT / "data"
TRAIN_CSV = DATA / "models" / "training_dataset_real-weather-generation.csv"
Q50_PKL   = DATA / "models" / "lgbm_real_q50.pkl"
FCST_CSV  = DATA / "outputs" / "forecast_14day_15min.csv"
OUT_DIR   = DATA / "outputs"

# Financial constants — mirror optimizer.py / yield_model.py
TARIFF_GRID, TARIFF_SURPLUS = 0.33, 0.16
ANNUAL_CONSUMPTION = 25_000
OP_START, OP_END   = 7, 20

# LightGBM q50 config — identical to train_pv_real.py
LGB_Q50 = dict(
    objective="quantile", alpha=0.5, n_estimators=2000, max_depth=6,
    num_leaves=31, learning_rate=0.05, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, min_child_samples=20, n_jobs=-1,
    random_state=42, verbose=-1,
)


def load_daylight() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_CSV, parse_dates=["datetime"])
    df = df[df["solar_elevation"] > 0].copy()
    df = df.dropna(subset=FEATURES + [TARGET])
    return df


# ============================================================
# 1) DIEBOLD-MARIANO  (HLN small-sample corrected)
# ============================================================
def diebold_mariano(actual, pred1, pred2, h: int = 1, loss: str = "mse"):
    """H0: equal forecast accuracy.  Negative DM => pred1 lower loss (better)."""
    actual = np.asarray(actual, float)
    e1 = actual - np.asarray(pred1, float)
    e2 = actual - np.asarray(pred2, float)
    d = (e1**2 - e2**2) if loss == "mse" else (np.abs(e1) - np.abs(e2))
    T = len(d)
    dbar = d.mean()
    gamma0 = np.var(d, ddof=0)
    acov = [np.cov(d[k:], d[:-k], ddof=0)[0, 1] for k in range(1, h)]
    var_d = (gamma0 + 2 * sum(acov)) / T
    dm = dbar / np.sqrt(var_d)
    k = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)   # Harvey-Leybourne-Newbold
    dm_hln = dm * k
    p = 2 * tdist.cdf(-abs(dm_hln), df=T - 1)
    return dm_hln, p


def test_dm(df: pd.DataFrame) -> pd.DataFrame:
    _, _, test = split_data(df)
    y  = test[TARGET].to_numpy()
    q50 = joblib.load(Q50_PKL)
    ml = np.clip(q50.predict(test[FEATURES]), 0, CF_MAX)
    pv = pvlib_cf(test)                       # PVWatts physics, zero fitting
    rows = []
    for loss in ("mse", "mae"):
        stat, p = diebold_mariano(y, ml, pv, loss=loss)
        rows.append({
            "comparison": "LightGBM_q50 vs PVWatts",
            "loss": loss.upper(),
            "DM_stat": round(stat, 3),
            "p_value": f"{p:.2e}",
            "sig_5pct": bool(p < 0.05),
            "more_accurate": "LightGBM" if stat < 0 else "PVWatts",
        })
    out = pd.DataFrame(rows)
    print("\n=== 1) DIEBOLD-MARIANO (held-out test split) ===")
    print(f"test rows (daylight): {len(test):,}  "
          f"[{test.datetime.min().date()} -> {test.datetime.max().date()}]")
    print(out.to_string(index=False))
    out.to_csv(OUT_DIR / "research_diebold_mariano.csv", index=False)
    return out


# ============================================================
# 2) WALK-FORWARD / ROLLING-ORIGIN CROSS-VALIDATION
# ============================================================
def walk_forward_cv(df: pd.DataFrame, n_splits: int = 8) -> pd.DataFrame:
    df = df.sort_values("datetime").reset_index(drop=True)
    blocks = np.array_split(np.arange(len(df)), n_splits + 1)
    rows = []
    for i in range(1, n_splits + 1):
        tr = np.concatenate(blocks[:i])
        te = blocks[i]
        model = LGBMRegressor(**LGB_Q50)
        model.fit(df.loc[tr, FEATURES], df.loc[tr, TARGET])
        pred = np.clip(model.predict(df.loc[te, FEATURES]), 0, CF_MAX)
        yt = df.loc[te, TARGET].to_numpy()
        rows.append({"fold": i, "n_test": len(te),
                     "R2": r2_score(yt, pred),
                     "MAE": mean_absolute_error(yt, pred),
                     "RMSE": root_mean_squared_error(yt, pred)})
        print(f"  fold {i}/{n_splits}  R2={rows[-1]['R2']:.3f}  "
              f"MAE={rows[-1]['MAE']:.4f}")

    res = pd.DataFrame(rows).set_index("fold")
    cols = ["R2", "MAE", "RMSE"]
    n = len(res)
    tcrit = tdist.ppf(0.975, df=n - 1)
    mean = res[cols].mean()
    sd = res[cols].std(ddof=1)
    summary = pd.DataFrame({
        "mean": mean, "sd": sd,
        "ci95_low":  mean - tcrit * sd / np.sqrt(n),
        "ci95_high": mean + tcrit * sd / np.sqrt(n),
    })
    print("\n=== 2) WALK-FORWARD CV (expanding window, "
          f"{n_splits} folds) ===")
    print(res.round(4).to_string())
    print("\nmean +/- 95% CI:")
    print(summary.round(4).to_string())
    res.to_csv(OUT_DIR / "research_walkforward_folds.csv")
    summary.to_csv(OUT_DIR / "research_walkforward_summary.csv")
    return summary


# ============================================================
# 3) MONTE CARLO — financial outcome distribution
#    two error models: (a) independent per-slot   (b) day-correlated copula
# ============================================================
def _inverse_cdf(u, q05, q50, q95):
    """Map uniform u -> generation via piecewise-linear inverse CDF through
    (.05,q05)(.5,q50)(.95,q95). Preserves each slot's marginal; respects skew."""
    lo = (q50 - q05) / 0.45        # slope on prob-interval [0.05, 0.50]
    hi = (q95 - q50) / 0.45        # slope on prob-interval [0.50, 0.95]
    val = np.where(u < 0.5, q50 - (0.5 - u) * lo, q50 + (u - 0.5) * hi)
    return np.clip(val, 0.0, None)


def _sample_independent(q05, q50, q95, day_id, n_sims, rng, rho=0.0):
    """Per-slot uncertainty drawn INDEPENDENTLY (the naive assumption)."""
    u = rng.random((n_sims, len(q50)))
    return _inverse_cdf(u, q05, q50, q95)


def _sample_correlated(q05, q50, q95, day_id, n_sims, rng, rho=0.7):
    """One-factor Gaussian copula: every slot in a day shares a common daily
    shock (weight sqrt(rho)) plus an idiosyncratic part (sqrt(1-rho)).
    Marginals (q05/q50/q95) are preserved exactly — Phi(z) is uniform — but
    within-day slots become correlated by rho. This reflects that a cloudy day
    pushes ALL its slots down together, so the 14-day aggregate uncertainty no
    longer averages out to ~0 (the flaw of the independent version)."""
    n_days = int(day_id.max()) + 1
    z_day = rng.standard_normal((n_sims, n_days))[:, day_id]   # shared per day
    z_eps = rng.standard_normal((n_sims, len(q50)))            # idiosyncratic
    z = np.sqrt(rho) * z_day + np.sqrt(1.0 - rho) * z_eps
    u = ndtr(z)                                                # Phi(z) ~ Uniform(0,1)
    return _inverse_cdf(u, q05, q50, q95)


def _financial(gen, cons, q50, penalty_rate):
    """Vectorised financial outcome per simulation (mirrors optimizer.py)."""
    self_use = np.minimum(gen, cons)
    surplus  = np.maximum(gen - cons, 0.0)
    saving  = (self_use * TARIFF_GRID + surplus * TARIFF_SURPLUS).sum(axis=1)
    penalty = (np.abs(gen - q50) * penalty_rate).sum(axis=1)   # imbalance cost
    return saving, penalty


def monte_carlo(n_sims: int = 10_000, penalty_rate: float = 0.05,
                rho: float = 0.7, seed: int = 42):
    rng = np.random.default_rng(seed)
    f = pd.read_csv(FCST_CSV, parse_dates=["datetime"])
    # raw columns are hourly-rate kWh; optimizer.py divides by 4 -> per 15-min slot
    q05 = f["soli_kwh_q05"].to_numpy() / 4
    q50 = f["soli_kwh_pred"].to_numpy() / 4
    q95 = f["soli_kwh_q95"].to_numpy() / 4

    # per-slot consumption (mirror optimizer.py)
    op_hours = OP_END - OP_START
    daily_cons = (ANNUAL_CONSUMPTION / 365) * (op_hours / 24)
    slot_base = daily_cons / (op_hours * 4)
    hour = f["datetime"].dt.hour.to_numpy()
    cons = np.where((hour >= OP_START) & (hour < OP_END), slot_base, 0.0)
    day_id = pd.factorize(f["datetime"].dt.normalize())[0]     # 0..D-1 per slot

    pct = lambda a, q: float(np.percentile(a, q))
    rows = []
    print(f"\n=== 3) MONTE CARLO (14-day, n_sims={n_sims:,}) ===")
    for label, sampler, kw in [
        ("independent",            _sample_independent, {}),
        (f"correlated(rho={rho})", _sample_correlated,  {"rho": rho}),
    ]:
        gen = sampler(q05, q50, q95, day_id, n_sims, rng, **kw)
        saving, penalty = _financial(gen, cons, q50, penalty_rate)
        band = pct(saving, 90) - pct(saving, 10)
        rows.append({
            "model": label,
            "save_P10": round(pct(saving, 10), 1),
            "save_P50": round(pct(saving, 50), 1),
            "save_P90": round(pct(saving, 90), 1),
            "save_sd":  round(float(saving.std()), 2),
            "save_band_P10_P90": round(band, 1),
            "save_band_pct": round(100 * band / pct(saving, 50), 1),
            "penalty_P50": round(pct(penalty, 50), 2),
            "penalty_P90": round(pct(penalty, 90), 2),
        })
        print(f"  [{label:>20}]  saving P10/P50/P90 = "
              f"{pct(saving,10):.1f} / {pct(saving,50):.1f} / {pct(saving,90):.1f} GEL  "
              f"(band {band:.1f} = {100*band/pct(saving,50):.1f}% of P50)  "
              f"penalty P50={pct(penalty,50):.2f}")
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "research_monte_carlo.csv", index=False)
    return out


def main():
    df = load_daylight()
    print(f"Loaded {len(df):,} daylight rows "
          f"[{df.datetime.min().date()} -> {df.datetime.max().date()}]")
    test_dm(df)
    walk_forward_cv(df, n_splits=8)
    monte_carlo()
    print("\nSaved -> data/outputs/research_*.csv")


if __name__ == "__main__":
    main()
