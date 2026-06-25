# src/analysis/briefing.py
# AI-generated daily briefing for the SME owner.
#
# Architecture (3 layers of data containment):
#   Layer 1 — Strict system prompt forbidding external knowledge
#   Layer 2 — RAG: all data injected in <data> tags from local CSVs only
#   Layer 3 — Post-validation: checks numbers, speculation, external references
#
# Reads:  data/outputs/forecast_14day_15min.csv
#         data/outputs/optimization_14day_summary.csv
#         data/outputs/alerts_14day.csv
# Writes: data/outputs/daily_briefing.txt
#         data/outputs/daily_briefing_validation.json
#
# Setup:
#   pip install anthropic
#   export ANTHROPIC_API_KEY=sk-ant-...
#
# Run: python -m analysis.briefing

import re
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── LAYER 1: STRICT SYSTEM PROMPT ─────────────────────────
SYSTEM_PROMPT = """You are a solar energy advisor for a Dutch SME with a 2.25 kWp rooftop solar installation in Soesterberg, Netherlands. All monetary values are in euros (EUR, €).

CRITICAL RULES — ANY VIOLATION IS A COMPLETE FAILURE:

1. You may ONLY use information from the <data> blocks in the user message.
2. You MUST NOT use general knowledge about:
   - The Netherlands, Soesterberg, Utrecht, or the European weather climate
   - Solar energy physics, panel efficiency, weather patterns
   - Electricity prices, market conditions, regulations
   - Any external facts, statistics, or studies
3. If something is not in <data>, respond exactly: "Not available in your system data."
4. NEVER estimate, guess, or extrapolate beyond the data shown. You MAY state a
   low–high range ONLY when both bound values appear in <data>.
5. NEVER mention numbers that are not present in <data>.
6. Quote exact values from <data> — do not round, average, or approximate.
7. Currency is euros (€). NEVER write GEL, dollars, or any other currency.
8. Forbidden phrases: "typically", "usually", "in general", "studies show",
   "research indicates", "on average for the region", "tend to", "experts say".
9. Respond in English only.
10. Be specific — name exact dates, weekdays, hours, kWh, and € amounts.
11. Keep the briefing under 250 words."""


# ── DATA LOADING ──────────────────────────────────────────
# GEL columns are a legacy of the original Tbilisi project. The values are the
# real per-day savings; only the label was wrong for this NL site → relabel EUR.
def load_pipeline_data() -> dict:
    """
    Load the pipeline CSVs and build compact, clearly-labelled summaries:
      - daily_outlook : one merged table (generation + low/high range + weather +
                        net saving in €), with each day tagged today/tomorrow/weekday
      - period_context: totals and best/lowest day across the forecast period
      - active_alerts : current warnings
    Returns dict of strings (not DataFrames — keep the context window small).
    """
    out = PROJECT_ROOT / "data" / "outputs"
    summaries = {}

    # ── Spine: optimization summary (generation, savings, uncertainty) ──
    daily = None
    opt_path = out / "optimization_14day_summary.csv"
    if opt_path.exists():
        opt = pd.read_csv(opt_path, parse_dates=["date"])
        rename = {
            "generation_kwh":          "gen_kwh",
            "soli_kwh_q05":            "gen_low_kwh",
            "soli_kwh_q95":            "gen_high_kwh",
            "net_saving_gel":          "net_saving_eur",
            "saving_conservative_gel": "saving_low_eur",
            "saving_optimistic_gel":   "saving_high_eur",
            "saving_pct":              "saving_pct",
        }
        keep = ["date"] + [c for c in rename if c in opt.columns]
        daily = opt[keep].rename(columns=rename)

    # ── Merge in daily weather from the 15-min forecast ──
    fc_path = out / "forecast_14day_15min.csv"
    if fc_path.exists():
        fc = pd.read_csv(fc_path, parse_dates=["date"])
        if "soli_kwh_pred" in fc.columns:
            w = (fc.groupby(fc["date"].dt.normalize())
                   .agg(avg_clouds=("clouds", "mean"),
                        avg_temp=("temp", "mean"),
                        total_precip=("precipitation", "sum"))
                   .reset_index())
            daily = w if daily is None else daily.merge(w, on="date", how="left")

    if daily is not None and len(daily):
        daily = daily.sort_values("date").reset_index(drop=True)
        # Anchor "today" to the forecast's own first day (robust to clock skew).
        today = pd.to_datetime(daily["date"]).min().normalize()

        def _label(d):
            delta = (pd.Timestamp(d).normalize() - today).days
            if delta <= 0:
                return "today (partial)"
            if delta == 1:
                return "tomorrow"
            return pd.Timestamp(d).day_name()

        labels    = daily["date"].map(_label)
        date_strs = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")
        for c in daily.select_dtypes("number").columns:
            daily[c] = daily[c].round(2)
        daily.insert(0, "day", labels)
        daily["date"] = date_strs

        front = ["date", "day"] + [c for c in
            ["gen_kwh", "gen_low_kwh", "gen_high_kwh", "avg_clouds", "avg_temp",
             "total_precip", "net_saving_eur", "saving_low_eur", "saving_high_eur",
             "saving_pct"] if c in daily.columns]
        summaries["daily_outlook"] = daily[front].head(4).to_string(index=False)

        ctx = []
        if "gen_kwh" in daily:
            best  = daily.loc[daily["gen_kwh"].idxmax()]
            worst = daily.loc[daily["gen_kwh"].idxmin()]
            ctx.append(f"Total expected generation: {daily['gen_kwh'].sum():.1f} kWh")
            ctx.append(f"Best day: {best['date']} ({best['gen_kwh']:.1f} kWh)")
            ctx.append(f"Lowest day: {worst['date']} ({worst['gen_kwh']:.1f} kWh)")
        if "net_saving_eur" in daily:
            ctx.append(f"Total net saving: €{daily['net_saving_eur'].sum():.2f}")
        summaries["period_context"] = "\n".join(ctx) if ctx else "Not available."
    else:
        summaries["daily_outlook"]   = "File not found."
        summaries["period_context"]  = "Not available."

    # ── Alerts ──
    al_path = out / "alerts_14day.csv"
    if al_path.exists():
        al = pd.read_csv(al_path)
        summaries["active_alerts"] = (
            al.head(8).to_string(index=False) if not al.empty else "No active alerts."
        )
    else:
        summaries["active_alerts"] = "File not found."

    return summaries


# ── LAYER 2: DATA INJECTION ───────────────────────────────
def build_user_prompt(data_summaries: dict) -> str:
    """Wrap each CSV summary in its own <data> tag for clear boundary."""
    blocks = []
    for source, content in data_summaries.items():
        blocks.append(f'<data source="{source}">\n{content}\n</data>')

    data_block = "\n\n".join(blocks)

    instructions = """Write the SME owner's daily briefing. Use ONLY numbers from the <data> blocks; all money is in euros (€). Note the "day" column labels each row today/tomorrow/weekday — the first row may be a partial day.

Structure it EXACTLY as these sections:

**Headline** — one sentence: the day labelled "tomorrow", its expected generation with the low–high range (gen_low_kwh–gen_high_kwh), and whether any warning is active.

**Tomorrow's outlook** — for the "tomorrow" row: expected gen_kwh and its low–high range, avg_clouds, avg_temp.

**Load scheduling** — name the 1–2 best upcoming full days to run heavy loads, citing their date, weekday, gen_kwh, and net_saving_eur in €.

**Warnings** — only items in active_alerts, each with its date and the exact action. If none, write "No active warnings."

**Period outlook** — one line from period_context: total expected generation and total net saving (€), and the best day.

Cite exact dates, weekdays, and values. Respond in English only."""

    return f"{data_block}\n\n{instructions}"


# ── LAYER 3: POST-VALIDATION ──────────────────────────────
SPECULATION_PHRASES = [
    "typically", "usually", "in general", "generally speaking",
    "studies show", "research indicates", "on average for",
    "tend to", "experts say", "according to", "it is known that",
    "as a rule", "historically", "based on industry",
]

EXTERNAL_REFERENCES = [
    "EU directive", "PVGIS database", "NASA", "World Bank",
    "IEA report", "Bloomberg", "IPCC", "Wikipedia",
    "according to research", "scientific literature",
]

# Matches either a euro-prefixed amount (€1.69) or a number followed by a unit
# (6.28 kWh, 55.2 %, 1.69 EUR). GEL is still matched so a relapse into the old
# currency is flagged as a hallucinated number.
UNIT_PATTERN = re.compile(
    r'€\s*(\d+(?:[.,]\d+)?)|(\d+(?:[.,]\d+)?)\s*(kWh|EUR|GEL|%|°C|m/s|mm|kWp)',
    re.IGNORECASE,
)


def _number_in_source(value: float, source_text: str, tolerance: float = 0.05) -> bool:
    """
    Check if a number (within tolerance) exists anywhere in the source text.
    Tolerance is fractional — 0.05 = ±5%.
    """
    source_numbers = re.findall(r'\d+(?:[.,]\d+)?', source_text)
    target = value
    for s in source_numbers:
        try:
            n = float(s.replace(",", "."))
        except ValueError:
            continue
        if n == 0:
            if target == 0:
                return True
            continue
        if abs(n - target) / max(abs(n), abs(target), 1e-9) <= tolerance:
            return True
    return False


def validate_briefing(response_text: str, data_summaries: dict) -> dict:
    """
    Multi-check validation of the briefing against source data.
    Returns dict with passed flag, score (0-100), and list of issues.
    """
    issues = []
    source_text = " ".join(str(v) for v in data_summaries.values()).lower()
    response_lower = response_text.lower()

    # 1. Speculation phrases
    for phrase in SPECULATION_PHRASES:
        if phrase in response_lower:
            issues.append({"type": "speculation", "evidence": phrase})

    # 2. External references
    for ref in EXTERNAL_REFERENCES:
        if ref.lower() in response_lower:
            issues.append({"type": "external_reference", "evidence": ref})

    # 3. Number fidelity (with unit). Group 1 = euro-prefixed (€1.69);
    #    groups 2+3 = number + trailing unit (6.28 kWh).
    for match in UNIT_PATTERN.finditer(response_text):
        if match.group(1) is not None:
            raw_val, unit = match.group(1), "€"
        else:
            raw_val, unit = match.group(2), match.group(3)
        try:
            value = float(raw_val.replace(",", "."))
        except (ValueError, AttributeError):
            continue
        if not _number_in_source(value, source_text):
            issues.append({
                "type": "hallucinated_number",
                "evidence": f"{raw_val} {unit}",
            })

    score = max(0, 100 - len(issues) * 15)
    return {
        "passed": len(issues) == 0,
        "score": score,
        "issue_count": len(issues),
        "issues": issues,
    }


# ── OFFLINE FALLBACK ──────────────────────────────────────
def _fallback_briefing(data_summaries: dict) -> str:
    """Deterministic, data-only briefing used when the language model is
    unreachable. Numbers come straight from the summaries, so it always passes
    validation and never leaves the owner without a briefing."""
    return (
        "Daily briefing (offline mode — generated without the language model).\n\n"
        "**Upcoming days**\n"
        f"{data_summaries.get('daily_outlook', 'Not available.')}\n\n"
        "**Period outlook**\n"
        f"{data_summaries.get('period_context', 'Not available.')}\n\n"
        "**Active warnings**\n"
        f"{data_summaries.get('active_alerts', 'Not available.')}\n"
    )


# ── MAIN ──────────────────────────────────────────────────
def generate_briefing(verbose: bool = True) -> dict:
    """
    Generate a daily briefing and validate it against source data.
    Returns dict: {briefing, validation, model, generated_at}
    """
    try:
        import httpx
    except ImportError:
        raise ImportError("Install:  pip install httpx")

    if verbose:
        print("\n── Loading pipeline data ──")
    data_summaries = load_pipeline_data()
    if verbose:
        for k in data_summaries:
            preview = data_summaries[k][:80].replace("\n", " ")
            print(f"  {k}: {preview}...")

    if verbose:
        print("\n── Calling Ollama (gemma3:4b) ──")
    user_prompt = build_user_prompt(data_summaries)

    model_name = "gemma3:4b"
    try:
        resp = httpx.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        briefing_text = resp.json()["message"]["content"]
        model_used    = model_name
        if verbose:
            print(f"  Response received: {len(briefing_text)} chars")
    except Exception as e:
        # Ollama not running / timed out / bad response → never fail the step.
        model_used    = "offline-fallback"
        briefing_text = _fallback_briefing(data_summaries)
        if verbose:
            print(f"  ⚠ LLM unavailable ({e.__class__.__name__}) — using offline fallback.")

    if verbose:
        print("\n── Validating (Layer 3) ──")
    validation = validate_briefing(briefing_text, data_summaries)

    if verbose:
        status = "✅ PASSED" if validation["passed"] else "⚠️ ISSUES FOUND"
        print(f"  {status}  |  Score: {validation['score']}/100")
        if validation["issues"]:
            for issue in validation["issues"][:5]:
                print(f"    - {issue['type']}: {issue['evidence']!r}")

    # Save outputs
    out_dir         = PROJECT_ROOT / "data" / "outputs"
    briefing_path   = out_dir / "daily_briefing.txt"
    validation_path = out_dir / "daily_briefing_validation.json"

    header = (
        f"# Daily Briefing — Generated {datetime.now().isoformat(timespec='seconds')}\n"
        f"# Model: {model_used}  |  Validation score: {validation['score']}/100\n\n"
    )
    briefing_path.write_text(header + briefing_text, encoding="utf-8")
    validation_path.write_text(
        json.dumps({
            "passed":       validation["passed"],
            "score":        validation["score"],
            "issue_count":  validation["issue_count"],
            "issues":       validation["issues"],
            "model":        model_used,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }, indent=2),
        encoding="utf-8",
    )

    if verbose:
        print(f"\nSaved → {briefing_path.relative_to(PROJECT_ROOT)}")
        print(f"Saved → {validation_path.relative_to(PROJECT_ROOT)}")
        print("\n── BRIEFING ──\n")
        print(briefing_text)

    return {
        "briefing":     briefing_text,
        "validation":   validation,
        "model":        model_used,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def run_briefing():
    """Pipeline-compatible entry point."""
    return generate_briefing(verbose=True)


if __name__ == "__main__":
    generate_briefing()
