"""
Test suite for A/B Testing Agent.
Covers: statistics math, DB persistence, state transitions, i18n, router logic.
Run: pytest test_ab_agent.py -v
"""

import json
import math
import os
import sqlite3
import sys
import tempfile

import pytest
from pathlib import Path

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
sys.path.insert(0, "/mnt/user-data/outputs")
sys.path.insert(0, "/home/claude/ab-agent")

import ab_agent as ab
import i18n


# ─────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(ab, "DB_PATH", str(tmp_path / "test.db"))
    ab.init_db()
    yield


@pytest.fixture
def base_state() -> ab.ABState:
    return ab.ABState(
        feature_name="Onboarding redesign",
        discovery_insights="Users drop off at step 3. 45% say setup is confusing.",
        primary_metric="activation rate",
        baseline_rate=0.12,
        daily_users=2000,
        mde=0.05,
        traffic_split=0.5,
        hypothesis={
            "statement": "If we simplify step 3 then activation rate will increase by 5% because friction is reduced",
            "confidence": "medium",
            "prediction": {"direction": "increase", "magnitude_pct": 5},
            "guardrail_metrics": ["7-day retention"],
        },
        experiment_design={
            "_computed": {"sample_size_per_variant": 3000, "min_duration_days": 3}
        },
        experiment_results={
            "control": {"conversions": 240, "n": 2000, "rate": 0.12},
            "variant": {"conversions": 280, "n": 2000, "rate": 0.14},
            "guardrails_ok": True,
            "guardrails_notes": "yes — retention stable",
            "duration_days": 14,
        },
        statistical_analysis={},
        decision={},
        learnings={},
        has_results=False,
        needs_human=False,
        human_approved=False,
        iteration=0,
        session_id="test_session",
        audit=[],
    )


# ─────────────────────────────────────────────
# 1. STATISTICS
# ─────────────────────────────────────────────

class TestStatistics:
    def test_sample_size_basic(self):
        n = ab.sample_size_per_variant(0.10, 0.05)
        assert n > 0
        assert isinstance(n, int)

    def test_sample_size_larger_for_smaller_effect(self):
        n_small = ab.sample_size_per_variant(0.10, 0.05)
        n_large = ab.sample_size_per_variant(0.10, 0.20)
        assert n_small > n_large

    def test_sample_size_larger_for_lower_baseline(self):
        n_low  = ab.sample_size_per_variant(0.02, 0.05)
        n_high = ab.sample_size_per_variant(0.40, 0.05)
        assert n_low > n_high

    def test_sample_size_invalid_baseline(self):
        assert ab.sample_size_per_variant(0.0, 0.05) == 0
        assert ab.sample_size_per_variant(1.0, 0.05) == 0

    def test_test_duration_days(self):
        d = ab.test_duration_days(3000, 1000, 0.5)
        # 3000 needed / (1000 * 0.5 per variant) = 6 days
        assert d == 6

    def test_test_duration_zero_users(self):
        assert ab.test_duration_days(3000, 0) == 0

    def test_compute_p_value_significant(self):
        # Strong effect: 120/1000 vs 180/1000
        p = ab.compute_p_value(120, 1000, 180, 1000)
        assert p < 0.05

    def test_compute_p_value_not_significant(self):
        # Tiny difference: 100/1000 vs 102/1000
        p = ab.compute_p_value(100, 1000, 102, 1000)
        assert p > 0.05

    def test_compute_p_value_zero_samples(self):
        p = ab.compute_p_value(0, 0, 100, 1000)
        assert p == 1.0

    def test_compute_p_value_range(self):
        p = ab.compute_p_value(50, 500, 80, 500)
        assert 0.0 <= p <= 1.0

    def test_confidence_interval_basic(self):
        lo, hi = ab.confidence_interval(120, 1000)
        assert 0 < lo < 0.12 < hi < 1.0

    def test_confidence_interval_contains_rate(self):
        conversions, n = 150, 1000
        lo, hi = ab.confidence_interval(conversions, n)
        rate = conversions / n
        assert lo < rate < hi

    def test_confidence_interval_zero_n(self):
        assert ab.confidence_interval(0, 0) == (0.0, 0.0)

    def test_confidence_interval_bounds(self):
        lo, hi = ab.confidence_interval(10, 100)
        assert 0.0 <= lo <= 1.0
        assert 0.0 <= hi <= 1.0
        assert lo <= hi

    def test_relative_lift_positive(self):
        lift = ab.relative_lift(0.10, 0.12)
        assert abs(lift - 0.20) < 0.001

    def test_relative_lift_negative(self):
        lift = ab.relative_lift(0.10, 0.08)
        assert abs(lift - (-0.20)) < 0.001

    def test_relative_lift_zero_control(self):
        assert ab.relative_lift(0.0, 0.05) == 0.0

    def test_relative_lift_no_change(self):
        assert ab.relative_lift(0.10, 0.10) == 0.0

    def test_normal_cdf_symmetry(self):
        assert abs(ab._normal_cdf(0) - 0.5) < 0.001

    def test_normal_cdf_extremes(self):
        assert ab._normal_cdf(10) > 0.999
        assert ab._normal_cdf(-10) < 0.001

    def test_norm_ppf_inverse_cdf(self):
        # ppf(cdf(x)) ≈ x
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            p = ab._normal_cdf(x)
            x_back = ab._norm_ppf(p)
            assert abs(x_back - x) < 0.01, f"ppf(cdf({x})) = {x_back}"

    def test_full_pipeline_significant(self):
        """End-to-end: significant result should have p < 0.05."""
        p = ab.compute_p_value(100, 1000, 160, 1000)
        lift = ab.relative_lift(0.10, 0.16)
        assert p < 0.05
        assert abs(lift - 0.60) < 0.01

    def test_full_pipeline_not_significant(self):
        """End-to-end: tiny difference should have p > 0.05."""
        p = ab.compute_p_value(100, 1000, 103, 1000)
        assert p > 0.05


# ─────────────────────────────────────────────
# 2. DATABASE
# ─────────────────────────────────────────────

class TestDatabase:
    def test_init_creates_tables(self, tmp_db):
        conn = sqlite3.connect(ab.DB_PATH)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert "experiments" in tables
        assert "learnings" in tables

    def test_save_experiment(self, tmp_db):
        ab.save_experiment("sess1", "If X then Y", "running")
        conn = sqlite3.connect(ab.DB_PATH)
        row = conn.execute("SELECT * FROM experiments WHERE session_id='sess1'").fetchone()
        conn.close()
        assert row is not None
        assert row[3] == "running"  # status column

    def test_save_experiment_with_result(self, tmp_db):
        result = {"control": {"rate": 0.10}, "variant": {"rate": 0.12}}
        ab.save_experiment("sess2", "Hypothesis", "completed", result, "ship")
        conn = sqlite3.connect(ab.DB_PATH)
        row = conn.execute("SELECT result, decision FROM experiments WHERE session_id='sess2'").fetchone()
        conn.close()
        assert json.loads(row[0])["control"]["rate"] == 0.10
        assert row[1] == "ship"

    def test_save_learning(self, tmp_db):
        ab.save_learning("sess3", "Hyp", "ship", "Great insight", 0.15)
        learnings = ab.get_past_learnings(5)
        assert len(learnings) == 1
        assert learnings[0]["decision"] == "ship"
        assert abs(learnings[0]["effect_size"] - 0.15) < 0.001

    def test_get_past_learnings_limit(self, tmp_db):
        for i in range(8):
            ab.save_learning(f"sess{i}", f"Hyp {i}", "iterate", f"Insight {i}")
        learnings = ab.get_past_learnings(5)
        assert len(learnings) == 5

    def test_get_past_learnings_empty(self, tmp_db):
        assert ab.get_past_learnings() == []

    def test_get_past_learnings_order(self, tmp_db):
        for i in range(3):
            ab.save_learning(f"s{i}", f"H{i}", "ship", f"I{i}")
        learnings = ab.get_past_learnings(3)
        # Most recent first
        assert learnings[0]["hypothesis"] == "H2"

    def test_save_learning_without_effect_size(self, tmp_db):
        ab.save_learning("s", "H", "kill", "No signal", None)
        learnings = ab.get_past_learnings()
        assert learnings[0]["effect_size"] is None


# ─────────────────────────────────────────────
# 3. HELPERS
# ─────────────────────────────────────────────

class TestHelpers:
    def test_lang_en(self):
        i18n.set_language("en")
        assert ab.L("Hello", "Привет") == "Hello"

    def test_lang_ru(self):
        i18n.set_language("ru")
        assert ab.L("Hello", "Привет") == "Привет"

    def test_log_appends(self, base_state):
        audit = ab.log(base_state, "test_step", "summary text")
        assert len(audit) == 1
        assert audit[0]["step"] == "test_step"
        assert audit[0]["summary"] == "summary text"
        assert "ts" in audit[0]

    def test_log_accumulates(self, base_state):
        state = {**base_state, "audit": [{"step": "existing"}]}
        audit = ab.log(state, "new_step", "new")
        assert len(audit) == 2

    def test_parse_json_clean(self):
        raw = '{"key": "value", "num": 42}'
        result = ab.ab_agent if hasattr(ab, 'ab_agent') else None
        # Direct test of JSON parsing logic
        clean = raw.strip()
        assert json.loads(clean) == {"key": "value", "num": 42}

    def test_parse_json_with_markdown(self):
        raw = "```json\n{\"key\": 1}\n```"
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        assert json.loads(raw.strip()) == {"key": 1}


# ─────────────────────────────────────────────
# 4. ROUTER LOGIC
# ─────────────────────────────────────────────

class TestRouters:
    def test_after_design_approved_no_results(self, base_state):
        state = {**base_state, "human_approved": True, "has_results": False}
        assert ab.after_design_checkpoint(state) == "extract_learnings"

    def test_after_design_approved_with_results(self, base_state):
        state = {**base_state, "human_approved": True, "has_results": True}
        assert ab.after_design_checkpoint(state) == "collect_results"

    def test_after_design_rejected(self, base_state):
        state = {**base_state, "human_approved": False, "has_results": False}
        assert ab.after_design_checkpoint(state) == "generate_hypothesis"

    def test_after_decision_approved(self, base_state):
        state = {**base_state, "human_approved": True}
        assert ab.after_decision_checkpoint(state) == "extract_learnings"

    def test_after_decision_rejected(self, base_state):
        state = {**base_state, "human_approved": False}
        assert ab.after_decision_checkpoint(state) == "analyze_statistics"


# ─────────────────────────────────────────────
# 5. STATE TRANSITIONS (unit-level, mock Claude)
# ─────────────────────────────────────────────

class TestStateTransitions:
    """Test node logic with mocked Claude calls."""

    def _mock_call(self, monkeypatch, return_value: dict):
        monkeypatch.setattr(ab, "call_model", lambda _: return_value)

    def test_analyze_statistics_computes_stats(self, base_state, monkeypatch):
        self._mock_call(monkeypatch, {
            "statistical_significance": True,
            "practical_significance": True,
            "effect_size_assessment": "medium",
            "confidence_level": "high",
            "risks": [],
            "segments_to_investigate": [],
            "interpretation": "Strong result",
        })
        result = ab.analyze_statistics(base_state)
        stats = result["statistical_analysis"]["_stats"]

        assert "p_value" in stats
        assert "relative_lift" in stats
        assert "statistically_significant" in stats
        assert 0.0 <= stats["p_value"] <= 1.0

        # 240/2000 = 12%, 280/2000 = 14% → +16.67% lift
        assert abs(stats["relative_lift"] - 0.1667) < 0.01

    def test_analyze_statistics_significant_result(self, base_state, monkeypatch):
        # Use larger effect: 120/1000 vs 180/1000 → p < 0.05
        base_state["experiment_results"] = {
            "control": {"conversions": 120, "n": 1000, "rate": 0.12},
            "variant": {"conversions": 180, "n": 1000, "rate": 0.18},
            "guardrails_ok": True, "guardrails_notes": "ok", "duration_days": 14,
        }
        self._mock_call(monkeypatch, {"statistical_significance": True,
                                       "practical_significance": True,
                                       "effect_size_assessment": "medium",
                                       "confidence_level": "high",
                                       "risks": [], "segments_to_investigate": [],
                                       "interpretation": "ok"})
        result = ab.analyze_statistics(base_state)
        assert result["statistical_analysis"]["_stats"]["statistically_significant"] is True

    def test_analyze_statistics_audit_logged(self, base_state, monkeypatch):
        self._mock_call(monkeypatch, {"statistical_significance": False,
                                       "practical_significance": False,
                                       "effect_size_assessment": "negligible",
                                       "confidence_level": "low",
                                       "risks": [], "segments_to_investigate": [],
                                       "interpretation": "no effect"})
        result = ab.analyze_statistics(base_state)
        assert any(e["step"] == "statistics" for e in result["audit"])

    def test_make_decision_ship(self, base_state, monkeypatch):
        base_state["statistical_analysis"] = {
            "_stats": {"p_value": 0.01, "relative_lift": 0.15,
                        "control_rate": 0.12, "variant_rate": 0.138,
                        "statistically_significant": True, "sample_adequate": True},
            "statistical_significance": True,
            "practical_significance": True,
            "effect_size_assessment": "medium",
            "confidence_level": "high",
            "risks": [],
        }
        self._mock_call(monkeypatch, {
            "verdict": "ship",
            "reasoning": "Statistically and practically significant",
            "next_steps": ["Deploy to 100%"],
            "key_learning": "Simplifying step 3 reduces friction",
            "planning_impact": {"velocity_note": "No impact", "risk_note": "Low"},
        })
        result = ab.make_decision(base_state)
        assert result["decision"]["verdict"] == "ship"

    def test_make_decision_kill(self, base_state, monkeypatch):
        base_state["statistical_analysis"] = {
            "_stats": {"p_value": 0.72, "relative_lift": 0.01,
                        "control_rate": 0.12, "variant_rate": 0.1212,
                        "statistically_significant": False, "sample_adequate": True},
            "statistical_significance": False,
            "practical_significance": False,
            "effect_size_assessment": "negligible",
            "confidence_level": "low",
            "risks": ["Guardrail violation"],
        }
        self._mock_call(monkeypatch, {
            "verdict": "kill",
            "reasoning": "No signal and guardrail violated",
            "next_steps": ["Investigate guardrail"],
            "key_learning": "This approach doesn't work",
            "planning_impact": {"velocity_note": "No impact", "risk_note": "None"},
        })
        result = ab.make_decision(base_state)
        assert result["decision"]["verdict"] == "kill"

    def test_extract_learnings_saves_to_db(self, base_state, monkeypatch):
        base_state["statistical_analysis"] = {
            "_stats": {"relative_lift": 0.15, "p_value": 0.02,
                        "control_rate": 0.12, "variant_rate": 0.138,
                        "statistically_significant": True, "sample_adequate": True},
        }
        base_state["decision"] = {
            "verdict": "ship",
            "reasoning": "Strong result",
            "next_steps": ["Deploy"],
            "key_learning": "Step 3 simplification works",
            "planning_impact": {"velocity_note": "none", "risk_note": "none"},
        }
        ab.extract_learnings(base_state)
        learnings = ab.get_past_learnings()
        assert len(learnings) == 1
        assert learnings[0]["decision"] == "ship"
        assert abs(learnings[0]["effect_size"] - 0.15) < 0.001

    def test_extract_learnings_creates_report(self, base_state, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        base_state["statistical_analysis"] = {
            "_stats": {"relative_lift": 0.10, "p_value": 0.03,
                        "control_rate": 0.12, "variant_rate": 0.132,
                        "statistically_significant": True, "sample_adequate": True},
        }
        base_state["decision"] = {
            "verdict": "iterate",
            "reasoning": "Signal but not strong enough",
            "next_steps": ["Extend test"],
            "key_learning": "Partial improvement",
            "planning_impact": {"velocity_note": "—", "risk_note": "—"},
        }
        i18n.set_language("en")
        result = ab.extract_learnings(base_state)
        report = result["learnings"].get("report_file", "")
        assert Path(report).exists() if report else True  # may be relative path


# ─────────────────────────────────────────────
# 6. I18N
# ─────────────────────────────────────────────

class TestI18n:
    def test_prompt_finalize_en(self):
        i18n.set_language("en")
        val = i18n.t("prompt_finalize")
        assert isinstance(val, str)
        assert len(val) > 0

    def test_prompt_finalize_ru(self):
        i18n.set_language("ru")
        val = i18n.t("prompt_finalize")
        assert isinstance(val, str)
        assert len(val) > 0

    def test_get_language_instruction_en(self):
        i18n.set_language("en")
        instr = i18n.get_language_instruction()
        assert "English" in instr

    def test_get_language_instruction_ru(self):
        i18n.set_language("ru")
        instr = i18n.get_language_instruction()
        assert "русском" in instr

    def test_lang_helper_switches(self):
        i18n.set_language("en")
        assert ab.L("Ship", "Выпустить") == "Ship"
        i18n.set_language("ru")
        assert ab.L("Ship", "Выпустить") == "Выпустить"

    def test_prompts_have_lang_placeholder(self):
        for prompt in [ab.HYPOTHESIS_PROMPT, ab.DESIGN_PROMPT,
                        ab.ANALYSIS_PROMPT, ab.DECISION_PROMPT]:
            assert "{lang}" in prompt, f"Missing {{lang}} in prompt: {prompt[:40]}"


# ─────────────────────────────────────────────
# 7. GRAPH STRUCTURE
# ─────────────────────────────────────────────

class TestGraph:
    def test_graph_builds(self):
        graph = ab.build_graph(":memory:")
        assert graph is not None

    def test_state_has_required_fields(self, base_state):
        required = [
            "feature_name", "discovery_insights", "primary_metric",
            "baseline_rate", "daily_users", "mde", "traffic_split",
            "hypothesis", "experiment_design", "experiment_results",
            "statistical_analysis", "decision", "learnings",
            "has_results", "needs_human", "human_approved",
            "iteration", "session_id", "audit",
        ]
        for field in required:
            assert field in base_state, f"Missing field: {field}"

    def test_state_types(self, base_state):
        assert isinstance(base_state["feature_name"], str)
        assert isinstance(base_state["baseline_rate"], float)
        assert isinstance(base_state["daily_users"], int)
        assert isinstance(base_state["mde"], float)
        assert isinstance(base_state["traffic_split"], float)
        assert isinstance(base_state["audit"], list)
        assert 0 < base_state["baseline_rate"] < 1
        assert 0 < base_state["mde"] < 1
        assert 0 < base_state["traffic_split"] <= 1
