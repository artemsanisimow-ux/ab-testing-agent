# 🧪 A/B Testing Agent

AI-powered experiment lifecycle agent built with LangGraph + Claude. Covers the full PM workflow from hypothesis generation to ship/iterate/kill decision — with real statistical rigor.

## What it does

1. **Hypothesis Generator** — structures Discovery insights into testable hypotheses using the format `If [change] then [metric] will [direction] by [X%] because [mechanism]`. Reads past experiment learnings from DB to avoid repeating failed approaches.
2. **Experiment Designer** — computes sample size (two-proportion z-test, power analysis), test duration, traffic split, variants, guardrail metrics, and pre-launch checklist.
3. **Human checkpoint** — review design before the experiment runs. Options: approve, revise, or enter results if the experiment is already done.
4. **Statistical Analyzer** — p-value, 95% Wilson CI, relative lift, effect size assessment, practical vs statistical significance. All math implemented from scratch — no scipy, no statsmodels.
5. **Decision Recommender** — `ship / iterate / kill` with reasoning, next steps, and planning impact (velocity note, risk note).
6. **Learning Extractor** — persists every experiment outcome to SQLite. Future hypotheses are generated with awareness of past results.

## How it works

```
Discovery insights (file or paste)
        ↓
Generate hypothesis (reads past learnings from DB)
        ↓
Design experiment (sample size, duration, guardrails)
        ↓
Human checkpoint — approve or revise
        ↓
Enter results (conversions + N per variant)
        ↓
Statistical analysis (p-value, CI, lift, effect size)
        ↓
Decision: ship / iterate / kill
        ↓
Human checkpoint — confirm decision
        ↓
Save report + persist learning to DB
```

## Quick start

```bash
git clone https://github.com/artemsanisimow-ux/ab-agent.git
cd ab-agent
python3 -m venv venv
source venv/bin/activate
pip install langgraph langchain-anthropic langgraph-checkpoint-sqlite python-dotenv
```

Add to `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
LANGUAGE=en
```

```bash
python3 ab_agent.py --lang en
```

## Language support

```bash
python3 ab_agent.py --lang en  # English
python3 ab_agent.py --lang ru  # Russian
```

## Input parameters

| Parameter | Example | Notes |
|-----------|---------|-------|
| Feature name | "Checkout button redesign" | What you're testing |
| Discovery report | path/to/report.md | Optional — loads insights from Discovery agent |
| Primary metric | "conversion rate" | What you're optimizing |
| Baseline rate | 12 | Current metric value in % |
| Daily users | 5000 | Users in experiment scope per day |
| MDE | 20 | Minimum detectable effect in % (relative) |
| Traffic split | 50 | % of users in the experiment |

## Statistics

All statistical methods implemented from scratch (no scipy/statsmodels):

- **Sample size** — two-proportion z-test with power analysis (α=0.05, power=80%)
- **p-value** — two-tailed two-proportion z-test
- **Confidence interval** — Wilson score (95%)
- **Relative lift** — (variant − control) / control
- **Normal distribution** — Beasley-Springer-Moro approximation

## Output

- `ab_report_SESSION_TIMESTAMP.md` — full report with hypothesis, results table, statistics, verdict, next steps, key learning
- `ab_audit_SESSION_TIMESTAMP.json` — step-by-step log
- `ab_experiments.db` — SQLite with all experiments and learnings (feeds future hypothesis generation)

## Running tests

```bash
pytest test_ab_agent.py -v   # 58 tests
```

Tests cover: statistics math, DB persistence, state transitions, router logic, i18n, graph structure.

## Part of a larger system

| Agent | Repo | Description |
|-------|------|-------------|
| Discovery | [discovery-agent](https://github.com/artemsanisimow-ux/discovery-agent) | Raw data → insights → hypotheses |
| Grooming | [grooming-agent](https://github.com/artemsanisimow-ux/grooming-agent) | Jira + Linear → estimate → acceptance criteria |
| Planning | [planning-agent](https://github.com/artemsanisimow-ux/planning-agent) | Monte Carlo → sprint plan → publish |
| Retrospective | [retro-agent](https://github.com/artemsanisimow-ux/retro-agent) | Sprint analysis → action items → planning insights |
| PRD | [prd-agent](https://github.com/artemsanisimow-ux/prd-agent) | Feature → full PRD → tasks in Jira + Linear |
| Stakeholder | [stakeholder-agent](https://github.com/artemsanisimow-ux/stakeholder-agent) | Sprint data → tailored comms for 5 audiences |
| A/B Testing | this repo | Hypothesis → experiment design → statistical analysis → decision |
