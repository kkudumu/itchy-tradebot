"""Unit tests for topstep_combine_pass_score (plan Task 12)."""

from __future__ import annotations

import pytest

from src.optimization.objectives import topstep_combine_pass_score


def _fake_result(prop_firm: dict) -> dict:
    return {"prop_firm": prop_firm}


class TestTopstepCombinePassScore:
    def test_passed_run_returns_one(self) -> None:
        result = _fake_result(
            {
                "active_tracker": {
                    "status": "passed",
                    "initial_balance": 50_000.0,
                    "current_balance": 53_000.0,
                    "profit_target_usd": 3_000.0,
                }
            }
        )
        assert topstep_combine_pass_score(result) == 1.0

    def test_failed_mll_applies_penalty(self) -> None:
        result = _fake_result(
            {
                "active_tracker": {
                    "status": "failed_mll",
                    "initial_balance": 50_000.0,
                    "current_balance": 48_000.0,
                    "profit_target_usd": 3_000.0,
                }
            }
        )
        score = topstep_combine_pass_score(result)
        # balance_score = (48000 - 50000) / 3000 = -0.67, penalty = -0.5
        assert score == pytest.approx(-0.667 - 0.5, abs=0.01)

    def test_failed_daily_loss_applies_penalty(self) -> None:
        result = _fake_result(
            {
                "active_tracker": {
                    "status": "failed_daily_loss",
                    "initial_balance": 50_000.0,
                    "current_balance": 49_000.0,
                    "profit_target_usd": 3_000.0,
                }
            }
        )
        score = topstep_combine_pass_score(result)
        # balance_score = -0.33, penalty = -0.3
        assert score == pytest.approx(-0.333 - 0.3, abs=0.01)

    def test_failed_consistency_applies_penalty(self) -> None:
        result = _fake_result(
            {
                "active_tracker": {
                    "status": "failed_consistency",
                    "initial_balance": 50_000.0,
                    "current_balance": 53_500.0,
                    "profit_target_usd": 3_000.0,
                }
            }
        )
        score = topstep_combine_pass_score(result)
        # balance_score = 1.0 (clipped), penalty = -0.2
        assert score == pytest.approx(0.8, abs=0.01)

    def test_missing_prop_firm_returns_negative(self) -> None:
        assert topstep_combine_pass_score({}) == -1.0
        assert topstep_combine_pass_score({"prop_firm": None}) == -1.0

    def test_without_active_tracker_uses_prop_firm_directly(self) -> None:
        # Legacy path where the engine puts topstep fields on prop_firm itself
        result = _fake_result(
            {
                "status": "passed",
                "initial_balance": 50_000.0,
                "current_balance": 53_000.0,
                "profit_target_usd": 3_000.0,
            }
        )
        assert topstep_combine_pass_score(result) == 1.0
