"""Tests for mega-vision cost tracker + safety gates (plan Task 25)."""

from __future__ import annotations

import os

import pytest

from src.mega_vision.cost_tracker import CostTracker
from src.mega_vision.safety_gates import SafetyGates


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------


class TestCostTracker:
    def test_subscription_mode_ignores_budget(self):
        ct = CostTracker(budget_usd=0.0, subscription_mode=True)
        ct.record("claude-opus-4-6", {"input_tokens": 1000, "output_tokens": 500})
        assert ct.can_afford() is True
        assert ct.total_cost_usd == 0.0

    def test_api_mode_tracks_dollar_cost(self):
        ct = CostTracker(budget_usd=10.0, subscription_mode=False)
        ct.record("claude-opus-4-6", {"input_tokens": 1_000_000, "output_tokens": 1_000_000})
        # Opus: $5/M in + $25/M out = $30 for 1M each
        assert ct.total_cost_usd > 20.0

    def test_api_mode_can_afford_respects_budget(self):
        ct = CostTracker(budget_usd=1.0, subscription_mode=False)
        assert ct.can_afford() is True
        ct.record("claude-opus-4-6", {"input_tokens": 1_000_000, "output_tokens": 1_000_000})
        assert ct.can_afford() is False

    def test_decision_count_increments(self):
        ct = CostTracker(subscription_mode=True)
        ct.record("claude-opus-4-6", {"input_tokens": 100, "output_tokens": 50})
        ct.record("claude-opus-4-6", {"input_tokens": 100, "output_tokens": 50})
        assert ct.decision_count == 2

    def test_per_model_breakdown(self):
        ct = CostTracker(subscription_mode=False)
        ct.record("claude-opus-4-6", {"input_tokens": 1000, "output_tokens": 500})
        ct.record("claude-haiku-4-5", {"input_tokens": 2000, "output_tokens": 1000})
        assert "claude-opus-4-6" in ct.per_model
        assert "claude-haiku-4-5" in ct.per_model
        assert ct.per_model["claude-haiku-4-5"]["calls"] == 1

    def test_to_dict(self):
        ct = CostTracker(subscription_mode=True)
        ct.record("claude-opus-4-6", {"input_tokens": 100, "output_tokens": 50})
        data = ct.to_dict()
        assert data["decision_count"] == 1
        assert data["cost_category"] == "subscription"
        assert data["total_cost_usd"] is None


# ---------------------------------------------------------------------------
# SafetyGates
# ---------------------------------------------------------------------------


class FakeTracker:
    def __init__(self, status: str = "pending"):
        self.status = status


class TestSafetyGates:
    def test_valid_pick_passes(self):
        gates = SafetyGates(
            active_strategies=["sss", "ichimoku"],
            prop_firm_tracker=FakeTracker(),
        )
        ok, reason = gates.validate_pick({"strategy_picks": ["sss"]})
        assert ok is True
        assert reason is None

    def test_kill_switch_blocks(self, monkeypatch):
        monkeypatch.setenv("MEGA_VISION_KILL_SWITCH", "1")
        gates = SafetyGates(
            active_strategies=["sss"],
            prop_firm_tracker=FakeTracker(),
        )
        ok, reason = gates.validate_pick({"strategy_picks": ["sss"]})
        assert ok is False
        assert "kill switch" in reason

    def test_pick_not_in_active_strategies_rejected(self):
        gates = SafetyGates(
            active_strategies=["sss"],
            prop_firm_tracker=FakeTracker(),
        )
        ok, reason = gates.validate_pick({"strategy_picks": ["ichimoku"]})
        assert ok is False
        assert "ichimoku" in reason

    def test_prop_firm_failed_rejects(self):
        gates = SafetyGates(
            active_strategies=["sss"],
            prop_firm_tracker=FakeTracker(status="failed_mll"),
        )
        ok, reason = gates.validate_pick({"strategy_picks": ["sss"]})
        assert ok is False
        assert "failed_mll" in reason

    def test_cost_budget_exhausted_rejects(self):
        ct = CostTracker(budget_usd=0.0, subscription_mode=False)
        # Force over budget
        ct.total_cost_usd = 1.0
        gates = SafetyGates(
            active_strategies=["sss"],
            prop_firm_tracker=FakeTracker(),
            cost_tracker=ct,
        )
        ok, reason = gates.validate_pick({"strategy_picks": ["sss"]})
        assert ok is False
        assert "budget" in reason

    def test_position_cap_exceeded_rejects(self):
        gates = SafetyGates(
            active_strategies=["sss", "ichimoku", "ema_pullback", "asian_breakout"],
            prop_firm_tracker=FakeTracker(),
            max_positions=2,
        )
        ok, reason = gates.validate_pick(
            {"strategy_picks": ["sss", "ichimoku", "ema_pullback"]}
        )
        assert ok is False
        assert "max_positions" in reason

    def test_empty_picks_accepted(self):
        gates = SafetyGates(
            active_strategies=["sss"],
            prop_firm_tracker=FakeTracker(),
        )
        ok, reason = gates.validate_pick({"strategy_picks": []})
        assert ok is True

    def test_non_list_picks_rejected(self):
        gates = SafetyGates(
            active_strategies=["sss"],
            prop_firm_tracker=FakeTracker(),
        )
        ok, reason = gates.validate_pick({"strategy_picks": "sss"})
        assert ok is False
