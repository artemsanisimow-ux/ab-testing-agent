"""
A/B Testing Agent
=================
PM-layer agent for experiment lifecycle management:
1. Hypothesis Generator  — structures insights into testable hypotheses
2. Experiment Designer   — sample size, duration, metrics, traffic split
3. Statistical Analyzer  — p-value, CI, effect size, practical significance
4. Decision Recommender  — ship / iterate / kill with reasoning
5. Learning Extractor    — persists findings for future hypotheses

Reads from: Discovery agent reports
Feeds into: Planning agent (velocity), Retro agent (learnings)

Setup:
    pip install langgraph langchain-anthropic langgraph-checkpoint-sqlite python-dotenv

Usage:
    python3 ab_agent.py --lang en
    python3 ab_agent.py --lang ru
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

load_dotenv()
from i18n import get_language, get_language_instruction
from i18n import t as tr

# ─────────────────────────────────────────────
# CONSTANTS & MODEL
# ─────────────────────────────────────────────

MODEL = ChatAnthropic(model="claude-opus-4-5", max_tokens=2048)
DB_PATH = "ab_experiments.db"

SIGNIFICANCE_LEVEL = 0.05   # α
DEFAULT_POWER = 0.80         # 1 - β
MIN_DETECTABLE_EFFECT = 0.05 # 5% relative change


# ─────────────────────────────────────────────
# PERSISTENCE
# ─────────────────────────────────────────────

def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS experiments (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            hypothesis  TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'draft',
            result      TEXT,
            decision    TEXT,
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS learnings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            hypothesis  TEXT NOT NULL,
            decision    TEXT NOT NULL,
            key_insight TEXT NOT NULL,
            effect_size REAL,
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()


def save_experiment(session_id: str, hypothesis: str, status: str,
                    result: dict = None, decision: str = None) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO experiments (session_id, hypothesis, status, result, decision) "
        "VALUES (?,?,?,?,?)",
        (session_id, hypothesis, status,
         json.dumps(result, ensure_ascii=False) if result else None, decision)
    )
    conn.commit()
    conn.close()


def save_learning(session_id: str, hypothesis: str, decision: str,
                  insight: str, effect_size: float = None) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO learnings (session_id, hypothesis, decision, key_insight, effect_size) "
        "VALUES (?,?,?,?,?)",
        (session_id, hypothesis, decision, insight, effect_size)
    )
    conn.commit()
    conn.close()


def get_past_learnings(limit: int = 5) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT hypothesis, decision, key_insight, effect_size, created_at "
        "FROM learnings ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────
# STATISTICS (no external deps)
# ─────────────────────────────────────────────

def _norm_ppf(p: float) -> float:
    """Approximation of normal PPF (percent point function) — Beasley-Springer-Moro."""
    a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
    b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
    c = [0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
         0.0276438810333863, 0.0038405729373609, 0.0003951896511349,
         0.0000321767881768, 0.0000002888167364, 0.0000003960315187]
    y = p - 0.5
    if abs(y) < 0.42:
        r = y * y
        return y * (((a[3]*r + a[2])*r + a[1])*r + a[0]) / ((((b[3]*r + b[2])*r + b[1])*r + b[0])*r + 1)
    r = math.log(-math.log(p if y < 0 else 1 - p))
    x = c[0] + r*(c[1] + r*(c[2] + r*(c[3] + r*(c[4] + r*(c[5] + r*(c[6] + r*(c[7] + r*c[8])))))))
    return -x if y < 0 else x


def sample_size_per_variant(baseline_rate: float, mde: float,
                             alpha: float = SIGNIFICANCE_LEVEL,
                             power: float = DEFAULT_POWER) -> int:
    """Two-proportion z-test sample size (Evans formula)."""
    if not (0 < baseline_rate < 1):
        return 0
    p1 = baseline_rate
    p2 = baseline_rate * (1 + mde)
    p2 = min(max(p2, 0.001), 0.999)
    z_alpha = _norm_ppf(1 - alpha / 2)
    z_beta  = _norm_ppf(power)
    p_avg   = (p1 + p2) / 2
    n = (z_alpha * math.sqrt(2 * p_avg * (1 - p_avg)) +
         z_beta  * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (p2 - p1) ** 2
    return math.ceil(n)


def test_duration_days(n_per_variant: int, daily_users: int,
                        traffic_split: float = 0.5) -> int:
    """Minimum test duration in days."""
    if daily_users <= 0:
        return 0
    exposed_per_day = daily_users * traffic_split
    return math.ceil(n_per_variant / max(exposed_per_day, 1))


def compute_p_value(control_conversions: int, control_n: int,
                    variant_conversions: int, variant_n: int) -> float:
    """Two-proportion z-test p-value (two-tailed)."""
    if control_n == 0 or variant_n == 0:
        return 1.0
    p1 = control_conversions / control_n
    p2 = variant_conversions / variant_n
    p_pool = (control_conversions + variant_conversions) / (control_n + variant_n)
    if p_pool in (0, 1):
        return 1.0
    se = math.sqrt(p_pool * (1 - p_pool) * (1/control_n + 1/variant_n))
    if se == 0:
        return 1.0
    z = (p2 - p1) / se
    # Two-tailed p-value via normal CDF approximation
    return 2 * (1 - _normal_cdf(abs(z)))


def _normal_cdf(x: float) -> float:
    """Approximation of standard normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def confidence_interval(conversions: int, n: int,
                         confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = _norm_ppf(1 - (1 - confidence) / 2)
    p = conversions / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = z * math.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return (max(0, center - margin), min(1, center + margin))


def relative_lift(control_rate: float, variant_rate: float) -> float:
    """Relative lift of variant over control."""
    if control_rate == 0:
        return 0.0
    return (variant_rate - control_rate) / control_rate


# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────

class ABState(TypedDict):
    # Input
    feature_name: str
    discovery_insights: str   # text from Discovery agent
    primary_metric: str       # e.g. "conversion rate"
    baseline_rate: float      # e.g. 0.12 = 12%
    daily_users: int
    mde: float                # minimum detectable effect (relative)
    traffic_split: float      # fraction exposed to experiment

    # Generated
    hypothesis: dict          # {statement, rationale, prediction, guardrails}
    experiment_design: dict   # {sample_size, duration, variants, metrics}
    experiment_results: dict  # raw results entered by PM
    statistical_analysis: dict
    decision: dict            # {verdict, reasoning, next_steps}
    learnings: dict

    # Control
    has_results: bool
    needs_human: bool
    human_approved: bool
    iteration: int
    session_id: str
    audit: list


# ─────────────────────────────────────────────
# PROMPTS — concise, structured output only
# ─────────────────────────────────────────────

HYPOTHESIS_PROMPT = """{lang}You are a senior PM writing a structured A/B test hypothesis.

Feature: {feature}
Discovery insights: {insights}
Primary metric: {metric}
Baseline rate: {baseline:.1%}

Past experiment learnings:
{learnings}

Write a rigorous hypothesis following the format:
"If [specific change] then [primary metric] will [increase/decrease] by [X%] because [causal mechanism]"

Return JSON only:
{{"statement": "If ... then ... will ... because ...",
  "rationale": "evidence from discovery that supports this",
  "prediction": {{"direction": "increase|decrease", "magnitude_pct": 5.0}},
  "guardrail_metrics": ["metric that must not degrade"],
  "confidence": "high|medium|low",
  "assumptions": ["assumption 1"]}}"""

DESIGN_PROMPT = """{lang}Design an A/B experiment for this hypothesis.

Hypothesis: {hypothesis}
Primary metric: {metric}
Sample size per variant: {n} users
Test duration: {days} days
Traffic split: {split:.0%} exposed ({exposed_pct:.0%} to each variant)
Daily users: {daily}

Return JSON only:
{{"variants": [{{"name": "Control", "description": "current experience"}},
               {{"name": "Variant A", "description": "what changes"}}],
  "primary_metric": {{"name": str, "measurement": "how to measure"}},
  "secondary_metrics": [{{"name": str, "measurement": str}}],
  "guardrail_metrics": [{{"name": str, "threshold": "must stay above/below X"}}],
  "exclusion_criteria": ["who to exclude from test"],
  "targeting": "who to include",
  "pre_test_checklist": ["item 1"],
  "risks": [{{"risk": str, "mitigation": str}}]}}"""

ANALYSIS_PROMPT = """{lang}Analyze A/B test results with statistical rigor.

Hypothesis: {hypothesis}
Primary metric: {metric}

Results:
- Control: {cc} conversions / {cn} users (rate: {cr:.1%})
- Variant: {vc} conversions / {vn} users (rate: {vr:.1%})

Computed statistics:
- p-value: {pval:.4f}
- Relative lift: {lift:+.1%}
- Control CI 95%%: [{ci_lo:.1%}, {ci_hi:.1%}]
- Statistically significant: {sig}
- Sample size adequate: {adequate}

Assess:
1. Statistical significance (α=0.05)
2. Practical significance — is the effect size meaningful?
3. Guardrail metrics impact
4. Novelty effect risk
5. Segment heterogeneity to investigate

Return JSON only:
{{"statistical_significance": bool,
  "practical_significance": bool,
  "effect_size_assessment": "large|medium|small|negligible",
  "confidence_level": "high|medium|low",
  "risks": ["risk 1"],
  "segments_to_investigate": ["segment 1"],
  "interpretation": "plain-language summary"}}"""

DECISION_PROMPT = """{lang}Make a ship/iterate/kill decision for this A/B test.

Hypothesis: {hypothesis}
Statistical significance: {sig}
Practical significance: {practical}
Relative lift: {lift:+.1%}
Confidence: {confidence}
Risks identified: {risks}

Decision criteria:
- SHIP: statistically + practically significant, guardrails OK, no major risks
- ITERATE: promising signal but needs refinement (extend, segment, redesign)
- KILL: no signal, negative impact, or guardrail violation

Return JSON only:
{{"verdict": "ship|iterate|kill",
  "reasoning": "2-3 sentences explaining the decision",
  "next_steps": ["concrete action 1"],
  "key_learning": "one sentence insight for future experiments",
  "planning_impact": {{"velocity_note": str, "risk_note": str}}}}"""


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def L(en: str, ru: str) -> str:
    return en if get_language() == "en" else ru


def call_model(prompt: str) -> dict:
    """Call Claude and parse JSON response."""
    resp = MODEL.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=tr("prompt_finalize")),
    ])
    raw = resp.content.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return {"error": "parse_failed", "raw": raw[:200]}


def log(state: ABState, step: str, summary: str) -> list:
    entry = {"ts": datetime.now().isoformat(), "step": step, "summary": summary}
    print(f"[{step.upper()}] {summary}")
    return state.get("audit", []) + [entry]


def collect_input(prompt_en: str, prompt_ru: str, default: str = "") -> str:
    text = input(L(prompt_en, prompt_ru)).strip()
    return text or default


# ─────────────────────────────────────────────
# NODES
# ─────────────────────────────────────────────

def gather_input(state: ABState) -> ABState:
    print(f"\n{'=' * 60}")
    print(f"🧪 {L('A/B Testing Agent', 'A/B Testing агент')} | "
          f"{L('Session', 'Сессия')}: {state['session_id']}")

    feature = state.get("feature_name") or collect_input(
        "Feature / experiment name: ",
        "Название фичи / эксперимента: ")

    discovery = state.get("discovery_insights") or collect_input(
        "Paste Discovery insights (or Enter to skip): ",
        "Вставь инсайты из Discovery агента (или Enter): ")

    metric = state.get("primary_metric") or collect_input(
        "Primary metric (e.g. 'conversion rate', 'activation rate'): ",
        "Основная метрика (например 'конверсия', 'активация'): ",
        "conversion rate")

    baseline_str = collect_input(
        f"Baseline {metric} in % (e.g. 12 for 12%): ",
        f"Текущее значение {metric} в % (например 12 для 12%): ",
        "10")
    baseline = float(baseline_str.replace("%", "").replace(",", ".")) / 100

    daily_str = collect_input(
        "Daily unique users in experiment scope: ",
        "Уникальных пользователей в день в скоупе эксперимента: ",
        "1000")
    daily = int(daily_str.replace(",", "").replace(" ", ""))

    mde_str = collect_input(
        "Minimum detectable effect in % (e.g. 5 for 5% relative change): ",
        "Минимально значимый эффект в % (например 5 для 5% относительного изменения): ",
        "5")
    mde = float(mde_str.replace("%", "").replace(",", ".")) / 100

    split_str = collect_input(
        "Traffic split — % of users in experiment (e.g. 50 for 50%): ",
        "Доля трафика в эксперименте — % (например 50 для 50%): ",
        "50")
    traffic_split = float(split_str.replace("%", "").replace(",", ".")) / 100

    audit = log(state, "gather_input", f"{feature} | {metric} | baseline={baseline:.1%} | daily={daily}")

    return {**state,
            "feature_name": feature,
            "discovery_insights": discovery,
            "primary_metric": metric,
            "baseline_rate": baseline,
            "daily_users": daily,
            "mde": mde,
            "traffic_split": traffic_split,
            "audit": audit}


def generate_hypothesis(state: ABState) -> ABState:
    print(f"\n💡 {L('Generating hypothesis...', 'Генерирую гипотезу...')}")

    past = get_past_learnings()
    learnings_text = (
        "\n".join(f"- [{r['decision'].upper()}] {r['hypothesis'][:80]} → {r['key_insight']}"
                  for r in past)
        or L("No past experiments", "Нет прошлых экспериментов")
    )

    hypothesis = call_model(HYPOTHESIS_PROMPT.format(
        lang=get_language_instruction(),
        feature=state["feature_name"],
        insights=state["discovery_insights"] or L("No discovery data provided", "Данные Discovery не предоставлены"),
        metric=state["primary_metric"],
        baseline=state["baseline_rate"],
        learnings=learnings_text,
    ))

    conf = hypothesis.get("confidence", "medium")
    stmt = hypothesis.get("statement", "")[:80]
    print(f"   ✅ {L('Confidence', 'Уверенность')}: {conf} | {stmt}...")

    audit = log(state, "hypothesis", f"confidence={conf}")
    return {**state, "hypothesis": hypothesis, "audit": audit}


def design_experiment(state: ABState) -> ABState:
    print(f"\n📐 {L('Designing experiment...', 'Проектирую эксперимент...')}")

    n = sample_size_per_variant(state["baseline_rate"], state["mde"])
    days = test_duration_days(n, state["daily_users"], state["traffic_split"] / 2)
    exposed_pct = state["traffic_split"] / 2  # each variant gets half

    design = call_model(DESIGN_PROMPT.format(
        lang=get_language_instruction(),
        hypothesis=state["hypothesis"].get("statement", ""),
        metric=state["primary_metric"],
        n=n,
        days=days,
        split=state["traffic_split"],
        exposed_pct=exposed_pct,
        daily=state["daily_users"],
    ))

    design["_computed"] = {
        "sample_size_per_variant": n,
        "total_sample": n * 2,
        "min_duration_days": days,
        "traffic_split": state["traffic_split"],
    }

    print(f"   ✅ n={n:,} per variant | {days}d | "
          f"{state['traffic_split']:.0%} {L('traffic', 'трафика')}")

    audit = log(state, "design", f"n={n}, days={days}")
    return {**state, "experiment_design": design, "needs_human": True, "audit": audit}


def human_checkpoint_design(state: ABState) -> ABState:
    """Show experiment design and ask for approval before collecting results."""
    hyp = state["hypothesis"]
    design = state["experiment_design"]
    comp = design.get("_computed", {})

    print(f"\n{'=' * 60}")
    print(f"✋ {L('EXPERIMENT DESIGN — REVIEW REQUIRED', 'ДИЗАЙН ЭКСПЕРИМЕНТА — ТРЕБУЕТ ПРОВЕРКИ')}")
    print(f"{'=' * 60}")

    print(f"\n📋 {L('Hypothesis', 'Гипотеза')}:")
    print(f"   {hyp.get('statement', '')}")
    print(f"   {L('Confidence', 'Уверенность')}: {hyp.get('confidence', '?')} | "
          f"{L('Direction', 'Направление')}: {hyp.get('prediction', {}).get('direction', '?')} "
          f"+{hyp.get('prediction', {}).get('magnitude_pct', 0):.0f}%")

    print(f"\n📊 {L('Experiment parameters', 'Параметры эксперимента')}:")
    print(f"   {L('Sample size', 'Размер выборки')}: {comp.get('sample_size_per_variant', 0):,} "
          f"{L('per variant', 'на вариант')} ({comp.get('total_sample', 0):,} {L('total', 'всего')})")
    print(f"   {L('Duration', 'Длительность')}: {comp.get('min_duration_days', 0)} "
          f"{L('days minimum', 'дней минимум')}")
    print(f"   {L('Traffic', 'Трафик')}: {comp.get('traffic_split', 0):.0%} "
          f"{L('exposed', 'в эксперименте')}")

    variants = design.get("variants", [])
    if variants:
        print(f"\n🔀 {L('Variants', 'Варианты')}:")
        for v in variants:
            print(f"   • {v.get('name', '')}: {v.get('description', '')}")

    guardrails = design.get("guardrail_metrics", [])
    if guardrails:
        print(f"\n🛡️  {L('Guardrail metrics', 'Защитные метрики')}:")
        for g in guardrails:
            print(f"   • {g.get('name', '')}: {g.get('threshold', '')}")

    risks = design.get("risks", [])
    if risks:
        print(f"\n⚠️  {L('Risks', 'Риски')}:")
        for r in risks[:3]:
            print(f"   • {r.get('risk', '')}")

    print(f"\n{'=' * 60}")
    print(L("Options: y — approve design | n — revise | r — enter results (if experiment done)",
            "Варианты: y — утвердить дизайн | n — доработать | r — ввести результаты (если эксперимент завершён)"))
    choice = input(f"\n{L('Choice', 'Выбор')}: ").strip().lower()

    audit = log(state, "checkpoint_design", f"choice={choice}")

    if choice == "r":
        return {**state, "has_results": True, "needs_human": False,
                "human_approved": True, "audit": audit}
    elif choice == "y":
        save_experiment(state["session_id"], hyp.get("statement", ""), "running")
        return {**state, "needs_human": False, "human_approved": True, "audit": audit}
    else:
        feedback = input(f"{L('What to change?', 'Что изменить?')} → ").strip()
        return {**state,
                "iteration": state.get("iteration", 0) + 1,
                "discovery_insights": state["discovery_insights"] + (f"\nFeedback: {feedback}" if feedback else ""),
                "needs_human": False, "human_approved": False, "audit": audit}


def collect_results(state: ABState) -> ABState:
    """PM enters experiment results."""
    print(f"\n📥 {L('Enter experiment results', 'Введи результаты эксперимента')}")
    print(f"   {L('Primary metric', 'Основная метрика')}: {state['primary_metric']}")

    def get_int(prompt_en: str, prompt_ru: str) -> int:
        val = input(L(prompt_en, prompt_ru)).strip().replace(",", "").replace(" ", "")
        return int(val) if val.isdigit() else 0

    print(f"\n{L('Control group', 'Контрольная группа')}:")
    cc = get_int("  Conversions: ", "  Конверсии: ")
    cn = get_int("  Total users: ", "  Всего пользователей: ")

    print(f"\n{L('Variant group', 'Группа варианта')}:")
    vc = get_int("  Conversions: ", "  Конверсии: ")
    vn = get_int("  Total users: ", "  Всего пользователей: ")

    guardrail_input = input(
        L("Guardrail metrics OK? (yes/no + notes): ",
          "Защитные метрики в норме? (да/нет + комментарий): ")
    ).strip()

    results = {
        "control": {"conversions": cc, "n": cn, "rate": cc/cn if cn else 0},
        "variant": {"conversions": vc, "n": vn, "rate": vc/vn if vn else 0},
        "guardrails_ok": guardrail_input.lower().startswith(("y", "yes", "да")),
        "guardrails_notes": guardrail_input,
        "duration_days": get_int(
            "Actual test duration (days): ",
            "Фактическая длительность теста (дней): "),
    }

    audit = log(state, "collect_results",
                f"control={cc}/{cn} variant={vc}/{vn}")
    return {**state, "experiment_results": results, "audit": audit}


def analyze_statistics(state: ABState) -> ABState:
    print(f"\n📊 {L('Computing statistics...', 'Считаю статистику...')}")

    res = state["experiment_results"]
    cc, cn = res["control"]["conversions"], res["control"]["n"]
    vc, vn = res["variant"]["conversions"], res["variant"]["n"]
    cr = cc / cn if cn else 0
    vr = vc / vn if vn else 0

    pval = compute_p_value(cc, cn, vc, vn)
    lift = relative_lift(cr, vr)
    ci_lo, ci_hi = confidence_interval(cc, cn)
    sig = pval < SIGNIFICANCE_LEVEL
    n_required = sample_size_per_variant(state["baseline_rate"], state["mde"])
    adequate = cn >= n_required and vn >= n_required

    print(f"   p={pval:.4f} | lift={lift:+.1%} | sig={'✅' if sig else '❌'} | "
          f"adequate={'✅' if adequate else '⚠️'}")

    # LLM interpretation
    analysis = call_model(ANALYSIS_PROMPT.format(
        lang=get_language_instruction(),
        hypothesis=state["hypothesis"].get("statement", ""),
        metric=state["primary_metric"],
        cc=cc, cn=cn, cr=cr, vc=vc, vn=vn, vr=vr,
        pval=pval, lift=lift, ci_lo=ci_lo, ci_hi=ci_hi,
        sig=sig, adequate=adequate,
    ))

    analysis["_stats"] = {
        "p_value": round(pval, 4),
        "relative_lift": round(lift, 4),
        "control_rate": round(cr, 4),
        "variant_rate": round(vr, 4),
        "ci_95_control": [round(ci_lo, 4), round(ci_hi, 4)],
        "statistically_significant": sig,
        "sample_adequate": adequate,
    }

    audit = log(state, "statistics", f"p={pval:.4f} lift={lift:+.1%}")
    return {**state, "statistical_analysis": analysis, "audit": audit}


def make_decision(state: ABState) -> ABState:
    print(f"\n⚖️  {L('Making decision...', 'Принимаю решение...')}")

    analysis = state["statistical_analysis"]
    stats = analysis.get("_stats", {})

    decision = call_model(DECISION_PROMPT.format(
        lang=get_language_instruction(),
        hypothesis=state["hypothesis"].get("statement", ""),
        sig=analysis.get("statistical_significance", False),
        practical=analysis.get("practical_significance", False),
        lift=stats.get("relative_lift", 0),
        confidence=analysis.get("confidence_level", "medium"),
        risks=analysis.get("risks", []),
    ))

    verdict = decision.get("verdict", "iterate")
    emoji = {"ship": "🚀", "iterate": "🔄", "kill": "❌"}.get(verdict, "❓")
    print(f"   {emoji} {L('Verdict', 'Вердикт')}: {verdict.upper()}")
    print(f"   {decision.get('reasoning', '')[:80]}...")

    audit = log(state, "decision", f"verdict={verdict}")
    return {**state, "decision": decision, "needs_human": True, "audit": audit}


def human_checkpoint_decision(state: ABState) -> ABState:
    """Show full analysis and ask for final approval."""
    analysis = state["statistical_analysis"]
    decision = state["decision"]
    stats = analysis.get("_stats", {})
    verdict = decision.get("verdict", "iterate")
    emoji = {"ship": "🚀", "iterate": "🔄", "kill": "❌"}.get(verdict, "❓")

    print(f"\n{'=' * 60}")
    print(f"✋ {L('EXPERIMENT RESULTS — DECISION REQUIRED', 'РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА — НУЖНО РЕШЕНИЕ')}")
    print(f"{'=' * 60}")

    print(f"\n📊 {L('Statistics', 'Статистика')}:")
    print(f"   {L('Control', 'Контроль')}: {stats.get('control_rate', 0):.1%} | "
          f"{L('Variant', 'Вариант')}: {stats.get('variant_rate', 0):.1%}")
    print(f"   {L('Relative lift', 'Относительный прирост')}: {stats.get('relative_lift', 0):+.1%}")
    print(f"   p-value: {stats.get('p_value', 1):.4f} "
          f"({'✅ significant' if stats.get('statistically_significant') else '❌ not significant'})")
    print(f"   {L('Effect size', 'Размер эффекта')}: {analysis.get('effect_size_assessment', '?')}")

    print(f"\n{emoji} {L('Recommended verdict', 'Рекомендованный вердикт')}: {verdict.upper()}")
    print(f"   {decision.get('reasoning', '')}")

    steps = decision.get("next_steps", [])
    if steps:
        print(f"\n📋 {L('Next steps', 'Следующие шаги')}:")
        for s in steps[:4]:
            print(f"   • {s}")

    insight = decision.get("key_learning", "")
    if insight:
        print(f"\n💡 {L('Key learning', 'Ключевой вывод')}: {insight}")

    print(f"\n{'=' * 60}")
    print(L("y — confirm & save | n — revise analysis",
            "y — подтвердить и сохранить | n — пересмотреть анализ"))
    choice = input(f"\n{L('Choice', 'Выбор')}: ").strip().lower()

    audit = log(state, "checkpoint_decision", f"choice={choice}")

    if choice == "y":
        return {**state, "human_approved": True, "needs_human": False, "audit": audit}
    else:
        feedback = input(f"{L('What to reconsider?', 'Что пересмотреть?')} → ").strip()
        return {**state,
                "human_approved": False, "needs_human": False,
                "iteration": state.get("iteration", 0) + 1,
                "experiment_results": {
                    **state["experiment_results"],
                    "pm_feedback": feedback
                },
                "audit": audit}


def extract_learnings(state: ABState) -> ABState:
    """Persist learnings and produce final report."""
    decision = state["decision"]
    verdict = decision.get("verdict", "iterate")
    insight = decision.get("key_learning", "")
    stats = state["statistical_analysis"].get("_stats", {})

    save_experiment(state["session_id"],
                    state["hypothesis"].get("statement", ""),
                    "completed", state["experiment_results"], verdict)
    save_learning(state["session_id"],
                  state["hypothesis"].get("statement", ""),
                  verdict, insight,
                  stats.get("relative_lift"))

    # Build report
    lang = get_language()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sid = state["session_id"]

    if lang == "en":
        report = f"""# A/B Test Report: {state['feature_name']}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Session: {sid}

## Hypothesis
{state['hypothesis'].get('statement', '—')}

**Confidence:** {state['hypothesis'].get('confidence', '?')} | **Direction:** {state['hypothesis'].get('prediction', {}).get('direction', '?')}

## Results

| Metric | Control | Variant | Lift |
|--------|---------|---------|------|
| {state['primary_metric']} | {stats.get('control_rate', 0):.2%} | {stats.get('variant_rate', 0):.2%} | {stats.get('relative_lift', 0):+.1%} |

**p-value:** {stats.get('p_value', 1):.4f} ({'significant ✅' if stats.get('statistically_significant') else 'not significant ❌'})
**Effect size:** {state['statistical_analysis'].get('effect_size_assessment', '?')}
**Sample adequate:** {'Yes ✅' if stats.get('sample_adequate') else 'No ⚠️'}

## Verdict: {verdict.upper()} {'🚀' if verdict == 'ship' else '🔄' if verdict == 'iterate' else '❌'}

{decision.get('reasoning', '')}

## Next Steps
{chr(10).join(f'- {s}' for s in decision.get('next_steps', []))}

## Key Learning
{insight}

## Planning Impact
- Velocity: {decision.get('planning_impact', {}).get('velocity_note', '—')}
- Risk: {decision.get('planning_impact', {}).get('risk_note', '—')}
"""
    else:
        report = f"""# Отчёт A/B теста: {state['feature_name']}
Дата: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Сессия: {sid}

## Гипотеза
{state['hypothesis'].get('statement', '—')}

**Уверенность:** {state['hypothesis'].get('confidence', '?')} | **Направление:** {state['hypothesis'].get('prediction', {}).get('direction', '?')}

## Результаты

| Метрика | Контроль | Вариант | Прирост |
|---------|----------|---------|---------|
| {state['primary_metric']} | {stats.get('control_rate', 0):.2%} | {stats.get('variant_rate', 0):.2%} | {stats.get('relative_lift', 0):+.1%} |

**p-value:** {stats.get('p_value', 1):.4f} ({'значимо ✅' if stats.get('statistically_significant') else 'незначимо ❌'})
**Размер эффекта:** {state['statistical_analysis'].get('effect_size_assessment', '?')}
**Выборка достаточна:** {'Да ✅' if stats.get('sample_adequate') else 'Нет ⚠️'}

## Вердикт: {verdict.upper()} {'🚀' if verdict == 'ship' else '🔄' if verdict == 'iterate' else '❌'}

{decision.get('reasoning', '')}

## Следующие шаги
{chr(10).join(f'- {s}' for s in decision.get('next_steps', []))}

## Ключевой вывод
{insight}

## Влияние на планирование
- Velocity: {decision.get('planning_impact', {}).get('velocity_note', '—')}
- Риски: {decision.get('planning_impact', {}).get('risk_note', '—')}
"""

    report_file = f"ab_report_{sid}_{ts}.md"
    Path(report_file).write_text(report, encoding="utf-8")

    # Audit log
    audit_file = f"ab_audit_{sid}_{ts}.json"
    Path(audit_file).write_text(
        json.dumps(state.get("audit", []), ensure_ascii=False, indent=2),
        encoding="utf-8")

    print(f"\n{'=' * 60}")
    if lang == "en":
        print(f"✅ Experiment complete!")
        print(f"   Verdict: {verdict.upper()} | Lift: {stats.get('relative_lift', 0):+.1%}")
        print(f"📄 Report: {report_file}")
        print(f"💡 Learning saved to DB for future experiments")
    else:
        print(f"✅ Эксперимент завершён!")
        print(f"   Вердикт: {verdict.upper()} | Прирост: {stats.get('relative_lift', 0):+.1%}")
        print(f"📄 Отчёт: {report_file}")
        print(f"💡 Вывод сохранён в БД для будущих экспериментов")

    learnings_out = {
        "verdict": verdict,
        "key_learning": insight,
        "planning_impact": decision.get("planning_impact", {}),
        "effect_size": stats.get("relative_lift"),
        "report_file": report_file,
    }
    audit = log(state, "learnings", f"verdict={verdict} report={report_file}")
    return {**state, "learnings": learnings_out, "human_approved": True, "audit": audit}


# ─────────────────────────────────────────────
# ROUTERS
# ─────────────────────────────────────────────

def after_design_checkpoint(state: ABState) -> Literal[
        "collect_results", "generate_hypothesis", "extract_learnings"]:
    if not state.get("human_approved"):
        return "generate_hypothesis"
    if state.get("has_results"):
        return "collect_results"
    return "extract_learnings"


def after_decision_checkpoint(state: ABState) -> Literal[
        "extract_learnings", "analyze_statistics"]:
    return "extract_learnings" if state.get("human_approved") else "analyze_statistics"


# ─────────────────────────────────────────────
# GRAPH
# ─────────────────────────────────────────────

def build_graph(db_path: str = "ab_checkpoints.db") -> object:
    g = StateGraph(ABState)

    for name, fn in [
        ("gather_input",              gather_input),
        ("generate_hypothesis",       generate_hypothesis),
        ("design_experiment",         design_experiment),
        ("human_checkpoint_design",   human_checkpoint_design),
        ("collect_results",           collect_results),
        ("analyze_statistics",        analyze_statistics),
        ("make_decision",             make_decision),
        ("human_checkpoint_decision", human_checkpoint_decision),
        ("extract_learnings",         extract_learnings),
    ]:
        g.add_node(name, fn)

    g.set_entry_point("gather_input")
    g.add_edge("gather_input", "generate_hypothesis")
    g.add_edge("generate_hypothesis", "design_experiment")
    g.add_edge("design_experiment", "human_checkpoint_design")
    g.add_conditional_edges("human_checkpoint_design", after_design_checkpoint, {
        "collect_results": "collect_results",
        "generate_hypothesis": "generate_hypothesis",
        "extract_learnings": "extract_learnings",
    })
    g.add_edge("collect_results", "analyze_statistics")
    g.add_edge("analyze_statistics", "make_decision")
    g.add_edge("make_decision", "human_checkpoint_decision")
    g.add_conditional_edges("human_checkpoint_decision", after_decision_checkpoint, {
        "extract_learnings": "extract_learnings",
        "analyze_statistics": "analyze_statistics",
    })
    g.add_edge("extract_learnings", END)

    conn = sqlite3.connect(db_path, check_same_thread=False)
    return g.compile(checkpointer=SqliteSaver(conn))


# ─────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────

def run_ab_test(
    feature_name: str = "",
    discovery_insights: str = "",
    primary_metric: str = "",
    baseline_rate: float = 0.0,
    daily_users: int = 0,
    mde: float = MIN_DETECTABLE_EFFECT,
    traffic_split: float = 0.5,
    session_id: str = None,
    resume: bool = False,
) -> tuple[dict, str]:
    init_db()
    sid = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    app = build_graph()
    config = {"configurable": {"thread_id": sid}}

    if resume:
        print(f"\n▶️  {L('Resuming session', 'Продолжаю сессию')} {sid}...")
        final = app.invoke(None, config=config)
    else:
        initial = ABState(
            feature_name=feature_name,
            discovery_insights=discovery_insights,
            primary_metric=primary_metric,
            baseline_rate=baseline_rate,
            daily_users=daily_users,
            mde=mde,
            traffic_split=traffic_split,
            hypothesis={}, experiment_design={},
            experiment_results={}, statistical_analysis={},
            decision={}, learnings={},
            has_results=False, needs_human=False,
            human_approved=False, iteration=0,
            session_id=sid, audit=[],
        )
        final = app.invoke(initial, config=config)

    print(f"\n💾 Session ID: {sid}")
    return final, sid


if __name__ == "__main__":
    run_ab_test()
