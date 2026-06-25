# src/analysis/optimizer.py
import sys
import pandas as pd
import numpy as np
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SYSTEM_KWP         = 2.25   # ID003 real scale (AC rating). Was 15 kWp (SME-pipeline parity).
TARIFF_SELF        = 0.33
TARIFF_GRID        = 0.33
TARIFF_SURPLUS     = 0.16
ANNUAL_CONSUMPTION = 25_000
OPERATING_START    = 7
OPERATING_END      = 20
OPERATING_HOURS    = OPERATING_END - OPERATING_START
DAILY_CONSUMPTION  = (ANNUAL_CONSUMPTION / 365) * (OPERATING_HOURS / 24)

# ── Risk-aware optimizer parameters ──────────────────────────────────────────
# Scenario weights: S1 worst-case, S2 expected, S3 best-case
W_WORST         = 0.30
W_MEDIAN        = 0.60
W_BEST          = 0.10

# Conformal safety buffer: PV_adj = q50 - K*(q50-q05), K in [0.3, 0.5]
# Higher K → more conservative scheduling
RISK_K          = 0.35

# Consumption uncertainty band for scenario spread
CONS_SPREAD     = 0.20   # ±20% around baseline

# Penalty weights (GEL per kWh)
DEFICIT_PENALTY = 0.50   # extra risk cost per kWh of grid dependency in S1
LAMBDA_RAMP     = 0.02   # penalise rapid PV ramps → prevents unstable scheduling
# ─────────────────────────────────────────────────────────────────────────────

# ── Battery storage parameters ───────────────────────────────────────────────
SLOT_HOURS            = 0.25   # each forecast slot is 15 minutes
BATTERY_CAPACITY_KWH  = 10.0   # usable battery capacity
BATTERY_MAX_POWER_KW  = 5.0    # inverter charge/discharge power limit (kW)
BATTERY_EFF_CHARGE    = 0.95   # one-way charging efficiency
BATTERY_EFF_DISCHARGE = 0.95   # one-way discharging efficiency (round-trip ≈ 0.90)
BATTERY_SOC_MIN       = 0.10   # reserve floor — never discharge below 10%
BATTERY_SOC_INIT      = 0.50   # state of charge at the start of the horizon
# ─────────────────────────────────────────────────────────────────────────────

LOADS = {
    "heavy":  {"kwh_per_hour": 5.0, "label": "Heavy equipment (ovens, compressors)"},
    "medium": {"kwh_per_hour": 2.5, "label": "Medium equipment (refrigeration, lighting)"},
    "light":  {"kwh_per_hour": 1.0, "label": "Light equipment (office, small appliances)"},
}


def load_forecast():
    path = PROJECT_ROOT / "data" / "outputs" / "forecast_14day_15min.csv"
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    for col in ["soli_kwh_pred", "soli_kwh_q05", "soli_kwh_q95"]:
        if col in df.columns:
            df[col] = df[col] / 4
    return df


def _slot_cost(pv, cons):
    """Net economic cost for one 15-min slot (works on scalars or Series).
    Returns negative values when solar surplus earns export revenue."""
    grid_import = np.maximum(0, cons - pv)
    surplus     = np.maximum(0, pv - cons)
    return grid_import * TARIFF_GRID - surplus * TARIFF_SURPLUS


def classify_hours(df):
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["date"] = df["datetime"].dt.date

    slot_base_load = DAILY_CONSUMPTION / (OPERATING_HOURS * 4)

    df["is_operating"]    = df["hour"].between(OPERATING_START, OPERATING_END - 1)
    df["consumption_kwh"] = np.where(df["is_operating"], slot_base_load, 0)

    has_bands = "soli_kwh_q05" in df.columns and "soli_kwh_q95" in df.columns

    if has_bands:
        q05 = df["soli_kwh_q05"]
        q50 = df["soli_kwh_pred"]
        q95 = df["soli_kwh_q95"]

        # Conformal safety buffer: shift planning assumption toward pessimistic
        # Guarantees optimizer does not overcommit on uncertain hours
        df["pv_adjusted"] = (q50 - RISK_K * (q50 - q05)).clip(lower=0)

        # Consumption scenarios (±CONS_SPREAD around baseline)
        cons_low  = df["consumption_kwh"] * (1 - CONS_SPREAD)
        cons_mid  = df["consumption_kwh"]
        cons_high = df["consumption_kwh"] * (1 + CONS_SPREAD)

        # Ramp risk: |ΔPV_adjusted| between consecutive slots
        # High ramp → unreliable production → penalise aggressive scheduling
        df["ramp_pv"] = df["pv_adjusted"].diff().abs().fillna(0)

        # Scenario costs
        cost_s1 = _slot_cost(q05, cons_high)   # S1: worst  — low PV, high load
        cost_s2 = _slot_cost(q50, cons_mid)    # S2: expected
        cost_s3 = _slot_cost(q95, cons_low)    # S3: best   — high PV, low load

        # Extra risk penalty for grid dependency under worst case
        deficit_s1 = np.maximum(0, cons_high - q05)

        # Risk-aware objective per slot:
        # minimize: 0.6*E[cost] + 0.3*worst_cost + 0.1*best_cost + ramp_penalty
        df["risk_objective"] = (
            W_MEDIAN * cost_s2
            + W_WORST  * (cost_s1 + DEFICIT_PENALTY * deficit_s1)
            + W_BEST   * cost_s3
            + LAMBDA_RAMP * df["ramp_pv"]
        )

        # Balance and savings use risk-adjusted PV for conservative decisions
        pv_ref = df["pv_adjusted"]
    else:
        # Fallback: deterministic mode (no quantile bands available)
        df["pv_adjusted"]    = df["soli_kwh_pred"]
        df["ramp_pv"]        = df["soli_kwh_pred"].diff().abs().fillna(0)
        df["risk_objective"] = _slot_cost(df["soli_kwh_pred"], df["consumption_kwh"])
        pv_ref = df["soli_kwh_pred"]

    df["balance_kwh"] = pv_ref - df["consumption_kwh"]
    df["surplus_kwh"] = df["balance_kwh"].clip(lower=0)
    df["deficit_kwh"] = (-df["balance_kwh"]).clip(lower=0)
    df["saving_gel"]  = (
        df["surplus_kwh"] * TARIFF_SURPLUS
        + np.minimum(pv_ref, df["consumption_kwh"]) * TARIFF_SELF
    )

    def recommend(row):
        if not row["is_operating"]:
            return "🌙 Outside operating hours"
        if row["pv_adjusted"] <= 0.01:
            return "🔌 Use grid — no solar available"
        ratio = (row["pv_adjusted"] / row["consumption_kwh"]
                 if row["consumption_kwh"] > 0 else 0)
        if ratio >= 1.5:  return "⚡ RUN HEAVY LOADS — strong surplus"
        if ratio >= 1.0:  return "✅ Run all loads — solar covers consumption"
        if ratio >= 0.7:  return "✅ Run medium loads — good solar"
        if ratio >= 0.4:  return "💡 Run light loads — partial solar"
        if ratio >= 0.2:  return "⚠️  Minimal solar — reduce where possible"
        return "🔌 Low solar — shift heavy loads to peak hours"

    df["action"] = df.apply(recommend, axis=1)
    return df


def simulate_battery(df):
    """Greedy self-consumption battery dispatch on the risk-adjusted PV series.

    Charges the battery from every PV surplus slot and discharges it into every
    deficit slot, capped by power, capacity and the reserve floor. Under the flat
    import tariff used here (self-use and avoided grid import are both worth
    TARIFF_GRID, above the export price TARIFF_SURPLUS) there is no price
    arbitrage, so this greedy pass is the cost-optimal policy. If time-of-use
    tariffs are ever introduced, this is the function to replace with an
    arbitrage-aware schedule.
    """
    df = df.copy()
    cap        = BATTERY_CAPACITY_KWH
    soc_floor  = cap * BATTERY_SOC_MIN
    slot_limit = BATTERY_MAX_POWER_KW * SLOT_HOURS   # max kWh moved per 15-min slot

    pv   = df["pv_adjusted"].to_numpy(dtype=float)
    cons = df["consumption_kwh"].to_numpy(dtype=float)
    n    = len(df)

    soc       = np.empty(n)        # state of charge at END of each slot (kWh)
    charge    = np.zeros(n)        # kWh drawn from PV surplus into the battery
    discharge = np.zeros(n)        # kWh delivered from the battery to the load
    g_import  = np.zeros(n)        # residual grid import (kWh)
    g_export  = np.zeros(n)        # residual surplus exported (kWh)

    state = cap * BATTERY_SOC_INIT
    for i in range(n):
        net = pv[i] - cons[i]
        if net >= 0:
            # Surplus → charge. Limited by power and remaining headroom (the
            # headroom is divided by efficiency: more PV is drawn than stored).
            headroom_in = (cap - state) / BATTERY_EFF_CHARGE
            c = min(net, slot_limit, headroom_in)
            state += c * BATTERY_EFF_CHARGE
            charge[i]   = c
            g_export[i] = net - c
        else:
            # Deficit → discharge. Limited by power and usable energy above floor
            # (delivered energy is what the load receives after discharge losses).
            deficit    = -net
            usable_out = max(0.0, state - soc_floor) * BATTERY_EFF_DISCHARGE
            d = min(deficit, slot_limit, usable_out)
            state -= d / BATTERY_EFF_DISCHARGE
            discharge[i] = d
            g_import[i]  = deficit - d
        soc[i] = state

    df["batt_charge_kwh"]    = charge
    df["batt_discharge_kwh"] = discharge
    df["batt_soc_kwh"]       = soc.round(3)
    df["batt_soc_pct"]       = (soc / cap * 100).round(1)
    df["grid_import_kwh"]    = g_import
    df["grid_export_kwh"]    = g_export
    # Self-consumption = PV used directly + PV time-shifted through the battery
    df["self_use_kwh"]       = np.minimum(pv, cons) + discharge
    df["cost_with_batt_gel"] = g_import * TARIFF_GRID - g_export * TARIFF_SURPLUS

    def batt_action(row):
        if not row["is_operating"] and row["batt_charge_kwh"] <= 0.01:
            return "🌙 Outside operating hours"
        if row["batt_charge_kwh"] > 0.01:
            return "🔋 Charge — store PV surplus"
        if row["batt_discharge_kwh"] > 0.01:
            return "🔋 Discharge — cover load from battery"
        if row["grid_import_kwh"] > 0.01:
            return "🔌 Grid import — battery at reserve floor"
        return "— Idle"

    df["batt_action"] = df.apply(batt_action, axis=1)
    return df


def daily_optimization(df):
    has_bands = "soli_kwh_q05" in df.columns
    has_batt  = "batt_charge_kwh" in df.columns

    agg = {
        "generation_kwh":  ("soli_kwh_pred",  "sum"),
        "consumption_kwh": ("consumption_kwh", "sum"),
        "pv_adjusted_kwh": ("pv_adjusted",     "sum"),
        "surplus_kwh":     ("surplus_kwh",     "sum"),
        "deficit_kwh":     ("deficit_kwh",     "sum"),
        "saving_gel":      ("saving_gel",      "sum"),
        "risk_objective":  ("risk_objective",  "sum"),
        "ramp_penalty_gel": ("ramp_pv",        lambda x: (x * LAMBDA_RAMP).sum()),
    }
    if has_bands:
        agg["soli_kwh_q05"] = ("soli_kwh_q05", "sum")
        agg["soli_kwh_q95"] = ("soli_kwh_q95", "sum")
    if has_batt:
        agg["batt_charge_kwh"]    = ("batt_charge_kwh",    "sum")
        agg["batt_discharge_kwh"] = ("batt_discharge_kwh", "sum")
        agg["grid_import_kwh"]    = ("grid_import_kwh",    "sum")
        agg["grid_export_kwh"]    = ("grid_export_kwh",    "sum")
        agg["self_use_kwh"]       = ("self_use_kwh",       "sum")
        agg["cost_with_batt_gel"] = ("cost_with_batt_gel", "sum")
        agg["batt_soc_end_pct"]   = ("batt_soc_pct",       "last")

    daily = df.groupby("date").agg(**agg).reset_index()

    daily["baseline_cost_gel"] = daily["consumption_kwh"] * TARIFF_GRID
    daily["actual_cost_gel"]   = (daily["deficit_kwh"] * TARIFF_GRID
                                   - daily["surplus_kwh"] * TARIFF_SURPLUS)
    daily["net_saving_gel"]    = daily["baseline_cost_gel"] - daily["actual_cost_gel"]
    daily["saving_pct"]        = (daily["net_saving_gel"]
                                   / daily["baseline_cost_gel"] * 100).round(1)

    if has_batt:
        # Savings once the battery time-shifts surplus into deficit slots.
        daily["net_saving_batt_gel"]  = (daily["baseline_cost_gel"]
                                          - daily["cost_with_batt_gel"])
        # Incremental value the battery adds on top of solar-only self-use.
        daily["battery_extra_gel"]    = (daily["net_saving_batt_gel"]
                                          - daily["net_saving_gel"])
        daily["saving_batt_pct"]      = (daily["net_saving_batt_gel"]
                                          / daily["baseline_cost_gel"] * 100).round(1)
        daily["self_consumption_pct"] = (daily["self_use_kwh"]
                                          / daily["consumption_kwh"] * 100).round(1)

    if has_bands:
        cons_low  = daily["consumption_kwh"] * (1 - CONS_SPREAD)
        cons_high = daily["consumption_kwh"] * (1 + CONS_SPREAD)

        daily["cost_worst_gel"]    = (
            _slot_cost(daily["soli_kwh_q05"], cons_high)
            + DEFICIT_PENALTY * np.maximum(0, cons_high - daily["soli_kwh_q05"])
        ).round(2)
        daily["cost_expected_gel"] = _slot_cost(
            daily["generation_kwh"], daily["consumption_kwh"]
        ).round(2)
        daily["cost_best_gel"]     = _slot_cost(
            daily["soli_kwh_q95"], cons_low
        ).round(2)

        daily["saving_conservative_gel"] = (
            np.minimum(daily["soli_kwh_q05"], daily["consumption_kwh"]) * TARIFF_SELF
            + np.maximum(daily["soli_kwh_q05"] - daily["consumption_kwh"], 0) * TARIFF_SURPLUS
        ).round(2)
        daily["saving_optimistic_gel"] = (
            np.minimum(daily["soli_kwh_q95"], daily["consumption_kwh"]) * TARIFF_SELF
            + np.maximum(daily["soli_kwh_q95"] - daily["consumption_kwh"], 0) * TARIFF_SURPLUS
        ).round(2)

    for col in daily.select_dtypes(include="number").columns:
        daily[col] = daily[col].round(2)
    return daily


def best_load_hours(df, top_n=3):
    df_op = df[df["is_operating"] & (df["pv_adjusted"] > 0.1)].copy()
    best = (df_op.groupby("date")
                 .apply(lambda x: x.nlargest(top_n, "pv_adjusted"))
                 .reset_index(drop=True))
    if "date" not in best.columns:
        best["date"] = best["datetime"].dt.date
    return best[["date", "datetime", "hour", "pv_adjusted",
                 "balance_kwh", "risk_objective", "action"]]


def run_optimizer():
    df    = load_forecast()
    df    = classify_hours(df)
    df    = simulate_battery(df)
    daily = daily_optimization(df)
    best  = best_load_hours(df, top_n=3)

    total_gen     = daily["generation_kwh"].sum()
    total_saving  = daily["net_saving_gel"].sum()
    total_risk    = daily["risk_objective"].sum()
    total_batt    = daily["net_saving_batt_gel"].sum()
    batt_extra    = daily["battery_extra_gel"].sum()
    batt_cycled   = daily["batt_discharge_kwh"].sum()
    print(
        f"14-Day Generation: {total_gen:.1f} kWh  |  "
        f"Net Saving (solar): {total_saving:.2f} GEL  |  "
        f"Risk Objective: {total_risk:.2f} GEL"
    )
    print(
        f"Battery: {batt_cycled:.1f} kWh discharged  |  "
        f"Net Saving (solar+battery): {total_batt:.2f} GEL  |  "
        f"Battery adds: +{batt_extra:.2f} GEL"
    )

    out = PROJECT_ROOT / "data" / "outputs"
    df.to_csv(out / "optimization_14day.csv", index=False)
    daily.to_csv(out / "optimization_14day_summary.csv", index=False)
    best.to_csv(out / "optimization_best_hours.csv", index=False)
    print("Saved → optimization_14day.csv + _summary.csv + _best_hours.csv")
    return df, daily, best


if __name__ == "__main__":
    run_optimizer()
