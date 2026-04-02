# tests/test_walk_forward_gate.py
"""Tests for the walk-forward validation gate."""

import pytest


class TestWalkForwardGate:
    def test_edge_passes_with_sufficient_oos_improvement(self):
        from src.discovery.walk_forward_gate import WalkForwardGate

        gate = WalkForwardGate(min_oos_windows=2, min_improvement_pct=1.0)

        # Simulate 3 OOS windows where the edge improved pass rate
        oos_results = [
            {"window_id": "w_004", "pass_rate_before": 0.20, "pass_rate_after": 0.25},
            {"window_id": "w_005", "pass_rate_before": 0.22, "pass_rate_after": 0.28},
            {"window_id": "w_006", "pass_rate_before": 0.18, "pass_rate_after": 0.24},
        ]
        verdict = gate.evaluate(oos_results)

        assert verdict.passed is True
        assert verdict.windows_improved >= 2

    def test_edge_fails_with_insufficient_oos_improvement(self):
        from src.discovery.walk_forward_gate import WalkForwardGate

        gate = WalkForwardGate(min_oos_windows=2, min_improvement_pct=1.0)

        # Only 1 window improved
        oos_results = [
            {"window_id": "w_004", "pass_rate_before": 0.20, "pass_rate_after": 0.25},
            {"window_id": "w_005", "pass_rate_before": 0.22, "pass_rate_after": 0.21},
            {"window_id": "w_006", "pass_rate_before": 0.18, "pass_rate_after": 0.17},
        ]
        verdict = gate.evaluate(oos_results)

        assert verdict.passed is False
        assert verdict.windows_improved == 1

    def test_edge_fails_when_degradation_detected(self):
        from src.discovery.walk_forward_gate import WalkForwardGate

        gate = WalkForwardGate(
            min_oos_windows=2,
            min_improvement_pct=1.0,
            max_degradation_pct=2.0,
        )

        # Pass rate dropped significantly in one window
        oos_results = [
            {"window_id": "w_004", "pass_rate_before": 0.30, "pass_rate_after": 0.32},
            {"window_id": "w_005", "pass_rate_before": 0.30, "pass_rate_after": 0.35},
            {"window_id": "w_006", "pass_rate_before": 0.30, "pass_rate_after": 0.25},
        ]
        verdict = gate.evaluate(oos_results)

        assert verdict.degraded is True

    def test_empty_oos_results_fails(self):
        from src.discovery.walk_forward_gate import WalkForwardGate

        gate = WalkForwardGate(min_oos_windows=2)
        verdict = gate.evaluate([])

        assert verdict.passed is False

    def test_verdict_contains_summary(self):
        from src.discovery.walk_forward_gate import WalkForwardGate

        gate = WalkForwardGate(min_oos_windows=2, min_improvement_pct=1.0)
        oos_results = [
            {"window_id": "w_004", "pass_rate_before": 0.20, "pass_rate_after": 0.26},
            {"window_id": "w_005", "pass_rate_before": 0.22, "pass_rate_after": 0.28},
        ]
        verdict = gate.evaluate(oos_results)

        assert isinstance(verdict.summary, str)
        assert len(verdict.summary) > 0
        assert isinstance(verdict.avg_improvement_pct, float)


class TestWalkForwardGateWithMetrics:
    def test_evaluate_with_multiple_metrics(self):
        from src.discovery.walk_forward_gate import WalkForwardGate

        gate = WalkForwardGate(min_oos_windows=2, min_improvement_pct=1.0)

        oos_results = [
            {
                "window_id": "w_004",
                "pass_rate_before": 0.20, "pass_rate_after": 0.26,
                "win_rate_before": 0.35, "win_rate_after": 0.40,
                "sharpe_before": 0.8, "sharpe_after": 1.1,
            },
            {
                "window_id": "w_005",
                "pass_rate_before": 0.22, "pass_rate_after": 0.28,
                "win_rate_before": 0.36, "win_rate_after": 0.42,
                "sharpe_before": 0.9, "sharpe_after": 1.2,
            },
        ]
        verdict = gate.evaluate(oos_results)

        assert verdict.passed is True
        assert verdict.avg_improvement_pct > 0

    def test_marginal_improvement_below_threshold_fails(self):
        from src.discovery.walk_forward_gate import WalkForwardGate

        gate = WalkForwardGate(min_oos_windows=2, min_improvement_pct=5.0)

        # Improvements are below 5 percentage points
        oos_results = [
            {"window_id": "w_004", "pass_rate_before": 0.20, "pass_rate_after": 0.22},
            {"window_id": "w_005", "pass_rate_before": 0.22, "pass_rate_after": 0.24},
        ]
        verdict = gate.evaluate(oos_results)

        assert verdict.passed is False
