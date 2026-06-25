# src/pipeline/run_results_pipeline.py
# Daily results pipeline: forecast → optimize → alerts → reports → XAI → briefing.
# Requires trained models in data/models/. Run run_training_pipeline.py first if models
# are missing.
import sys
import traceback
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PASS = "  ✅ PASS"
FAIL = "  ❌ FAIL"


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_output(path, label):
    p = PROJECT_ROOT / path
    if not p.exists():
        print(f"     → {label}: FILE NOT FOUND")
        return False
    if p.suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(p)
        print(f"     → {label}: {len(df)} rows, {len(df.columns)} cols")
    else:
        print(f"     → {label}: {p.stat().st_size:,} bytes")
    return True


# ── STEP 1: SEAS5 SEASONAL FORECAST ──────────────────────
section("STEP 1 — SEAS5 Seasonal Forecast (weeks 2–26)")
try:
    from ingestion.fetch_seas5_om import fetch_seas5_weekly
    seas5_df = fetch_seas5_weekly(52.1088, 5.1253)
    out = PROJECT_ROOT / "data" / "outputs" / "forecast_seasonal_seas5.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    seas5_df.to_csv(out, index=False)
    check_output("data/outputs/forecast_seasonal_seas5.csv", "forecast_seasonal_seas5.csv")
    print(PASS)
except Exception:
    traceback.print_exc()
    print(FAIL)


# ── STEP 1b: SEASONAL PV FORECAST ─────────────────────────
# Converts the raw SEAS5 weather (Step 1) into a year-ahead weekly PV forecast.
# Weeks 2–26: SEAS5-driven physics. Weeks 27–52: 5-year historical climatology.
# Produces forecast_365day_weekly.csv (otherwise SEAS5 data is fetched but unused).
section("STEP 1b — Seasonal PV Forecast (weeks 2–52)")
try:
    from inference.forecast_climate import run_forecast_climate
    run_forecast_climate()
    check_output("data/outputs/forecast_365day_weekly.csv",  "forecast_365day_weekly.csv")
    check_output("data/outputs/forecast_seasonal_daily.csv", "forecast_seasonal_daily.csv")
    print(PASS)
except Exception:
    traceback.print_exc()
    print(FAIL)


# ── STEP 2: FORECAST ──────────────────────────────────────
section("STEP 2 — PV Forecast (14 days)")
try:
    from inference.forecast_pv import run_forecast
    daily, hourly, df_raw = run_forecast(days=14)
    check_output("data/outputs/forecast_14day_15min.csv", "forecast_14day_15min.csv")
    print(PASS)
except Exception:
    traceback.print_exc()
    print(FAIL)
    sys.exit(1)


# ── STEP 3: OPTIMIZER ─────────────────────────────────────
section("STEP 3 — Optimizer")
try:
    from analysis.optimizer import run_optimizer
    df_opt, daily_opt, best = run_optimizer()
    check_output("data/outputs/optimization_14day_summary.csv", "optimization_14day_summary.csv")
    check_output("data/outputs/optimization_14day.csv",         "optimization_14day.csv")
    check_output("data/outputs/optimization_best_hours.csv",    "optimization_best_hours.csv")
    print(PASS)
except Exception:
    traceback.print_exc()
    print(FAIL)
    sys.exit(1)


# ── STEP 4: ALERTS ────────────────────────────────────────
section("STEP 4 — Alerts")
try:
    from analysis.alerts import run_alerts
    alerts_df = run_alerts()
    check_output("data/outputs/alerts_14day.csv", "alerts_14day.csv")
    print(PASS)
except Exception:
    traceback.print_exc()
    print(FAIL)
    sys.exit(1)


# ── STEP 5: MONTHLY YIELD (PVGIS) ─────────────────────────
# Produces monthly_yield.csv, which the Reports step (12-month projection) reads.
section("STEP 5 — Monthly Yield (PVGIS)")
try:
    from analysis.yield_model import run_yield_model
    fin, df_monthly = run_yield_model()
    check_output("data/outputs/monthly_yield.csv", "monthly_yield.csv")
    print(PASS)
except Exception:
    traceback.print_exc()
    print(FAIL)
    sys.exit(1)


# ── STEP 6: REPORTS ───────────────────────────────────────
section("STEP 6 — Reports")
try:
    from analysis.report import run_reports
    fc, monthly = run_reports()
    check_output("data/outputs/report_14day.csv",   "report_14day.csv")
    check_output("data/outputs/report_monthly.csv", "report_monthly.csv")
    print(PASS)
except Exception:
    traceback.print_exc()
    print(FAIL)
    sys.exit(1)


# ── STEP 7: XAI (SHAP) ────────────────────────────────────
# Forward pass explains the 14-day window (single season → no seasonal table).
# Retrospective pass over the full cleaned-weather history produces the real
# cross-season seasonal-shift table (shap_summary_full.csv).
section("STEP 7 — XAI / SHAP Explainability")
try:
    from analysis.xai import run_xai, run_xai_full
    shap_df, shap_summary = run_xai()
    check_output("data/outputs/shap_14day.csv",   "shap_14day.csv")
    check_output("data/outputs/shap_summary.csv", "shap_summary.csv")

    run_xai_full()
    check_output("data/outputs/shap_full.csv",         "shap_full.csv")
    check_output("data/outputs/shap_summary_full.csv", "shap_summary_full.csv")
    print(PASS)
except Exception:
    traceback.print_exc()
    print(FAIL)


# ── STEP 8: AI BRIEFING (Claude API) ──────────────────────
section("STEP 8 — AI Daily Briefing (Claude API)")
try:
    from analysis.briefing import run_briefing
    run_briefing()
    check_output("data/outputs/daily_briefing.txt",             "daily_briefing.txt")
    check_output("data/outputs/daily_briefing_validation.json", "daily_briefing_validation.json")
    print(PASS)
except Exception:
    traceback.print_exc()
    print(FAIL)


# ── SUMMARY ───────────────────────────────────────────────
section("RESULTS PIPELINE COMPLETE — output files")
outputs = [
    "data/outputs/forecast_seasonal_seas5.csv",
    "data/outputs/forecast_365day_weekly.csv",
    "data/outputs/forecast_seasonal_daily.csv",
    "data/outputs/forecast_14day_15min.csv",
    "data/outputs/features_14day.csv",
    "data/outputs/optimization_14day_summary.csv",
    "data/outputs/optimization_14day.csv",
    "data/outputs/optimization_best_hours.csv",
    "data/outputs/alerts_14day.csv",
    "data/outputs/monthly_yield.csv",
    "data/outputs/report_14day.csv",
    "data/outputs/report_monthly.csv",
    "data/outputs/shap_14day.csv",
    "data/outputs/shap_summary.csv",
    "data/outputs/shap_full.csv",
    "data/outputs/shap_summary_full.csv",
    "data/outputs/daily_briefing.txt",
    "data/outputs/daily_briefing_validation.json",
]
for f in outputs:
    p = PROJECT_ROOT / f
    status = "✅" if p.exists() else "❌ MISSING"
    print(f"  {status}  {f}")
