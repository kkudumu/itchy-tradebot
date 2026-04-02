"""Tests for mini-backtest + walk-forward validation of generated filters."""

from unittest.mock import patch, MagicMock
from dataclasses import dataclass

import pytest


@dataclass
class FakeBacktestResult:
    """Minimal stand-in for BacktestResult."""
    trades: list
    metrics: dict


class TestMiniBacktestValidator:
    def test_accepts_filter_that_maintains_win_rate(self):
        from src.discovery.codegen.backtest_validator import validate_with_backtest

        baseline_result = FakeBacktestResult(
            trades=[{"r_multiple": 1.5}] * 37 + [{"r_multiple": -1.0}] * 63,
            metrics={"win_rate": 0.37, "total_trades": 100},
        )
        candidate_result = FakeBacktestResult(
            trades=[{"r_multiple": 1.5}] * 35 + [{"r_multiple": -1.0}] * 65,
            metrics={"win_rate": 0.35, "total_trades": 100},
        )

        with patch(
            "src.discovery.codegen.backtest_validator._run_mini_backtest",
            side_effect=[baseline_result, candidate_result],
        ):
            result = validate_with_backtest(
                filter_code="fake code",
                filter_name="test_filter",
                filter_class_name="TestFilter",
                category="entry",
                max_win_rate_degradation=0.05,
            )
            assert result.passed is True

    def test_rejects_filter_that_degrades_win_rate(self):
        from src.discovery.codegen.backtest_validator import validate_with_backtest

        baseline_result = FakeBacktestResult(
            trades=[{"r_multiple": 1.5}] * 37 + [{"r_multiple": -1.0}] * 63,
            metrics={"win_rate": 0.37, "total_trades": 100},
        )
        # Severe degradation: 37% -> 20%
        candidate_result = FakeBacktestResult(
            trades=[{"r_multiple": 1.5}] * 20 + [{"r_multiple": -1.0}] * 80,
            metrics={"win_rate": 0.20, "total_trades": 100},
        )

        with patch(
            "src.discovery.codegen.backtest_validator._run_mini_backtest",
            side_effect=[baseline_result, candidate_result],
        ):
            result = validate_with_backtest(
                filter_code="fake code",
                filter_name="test_filter",
                filter_class_name="TestFilter",
                category="entry",
                max_win_rate_degradation=0.05,
            )
            assert result.passed is False
            assert "win rate" in result.reason.lower() or "degradation" in result.reason.lower()

    def test_accepts_filter_that_blocks_all_when_no_degradation(self):
        from src.discovery.codegen.backtest_validator import validate_with_backtest

        baseline_result = FakeBacktestResult(
            trades=[{"r_multiple": 1.5}] * 5 + [{"r_multiple": -1.0}] * 10,
            metrics={"win_rate": 0.33, "total_trades": 15},
        )
        # Blocks most trades but remaining ones have higher win rate
        candidate_result = FakeBacktestResult(
            trades=[{"r_multiple": 1.5}] * 4 + [{"r_multiple": -1.0}] * 4,
            metrics={"win_rate": 0.50, "total_trades": 8},
        )

        with patch(
            "src.discovery.codegen.backtest_validator._run_mini_backtest",
            side_effect=[baseline_result, candidate_result],
        ):
            result = validate_with_backtest(
                filter_code="fake code",
                filter_name="test_filter",
                filter_class_name="TestFilter",
                category="entry",
                max_win_rate_degradation=0.05,
            )
            assert result.passed is True


class TestWalkForwardValidation:
    def test_requires_improvement_on_2_oos_windows(self):
        from src.discovery.codegen.backtest_validator import (
            validate_walk_forward,
            WalkForwardWindow,
        )

        windows = [
            WalkForwardWindow(
                window_id="oos_1",
                baseline_win_rate=0.37,
                candidate_win_rate=0.42,
                baseline_avg_r=-0.05,
                candidate_avg_r=0.10,
                n_trades=25,
            ),
            WalkForwardWindow(
                window_id="oos_2",
                baseline_win_rate=0.35,
                candidate_win_rate=0.40,
                baseline_avg_r=-0.03,
                candidate_avg_r=0.08,
                n_trades=30,
            ),
            WalkForwardWindow(
                window_id="oos_3",
                baseline_win_rate=0.40,
                candidate_win_rate=0.38,
                baseline_avg_r=0.02,
                candidate_avg_r=-0.01,
                n_trades=20,
            ),
        ]

        result = validate_walk_forward(windows, min_improved_windows=2)
        assert result.passed is True
        assert result.improved_count >= 2

    def test_rejects_when_fewer_than_2_windows_improve(self):
        from src.discovery.codegen.backtest_validator import (
            validate_walk_forward,
            WalkForwardWindow,
        )

        windows = [
            WalkForwardWindow(
                window_id="oos_1",
                baseline_win_rate=0.37,
                candidate_win_rate=0.42,
                baseline_avg_r=-0.05,
                candidate_avg_r=0.10,
                n_trades=25,
            ),
            WalkForwardWindow(
                window_id="oos_2",
                baseline_win_rate=0.35,
                candidate_win_rate=0.30,
                baseline_avg_r=-0.03,
                candidate_avg_r=-0.10,
                n_trades=30,
            ),
            WalkForwardWindow(
                window_id="oos_3",
                baseline_win_rate=0.40,
                candidate_win_rate=0.35,
                baseline_avg_r=0.02,
                candidate_avg_r=-0.05,
                n_trades=20,
            ),
        ]

        result = validate_walk_forward(windows, min_improved_windows=2)
        assert result.passed is False
        assert result.improved_count < 2
