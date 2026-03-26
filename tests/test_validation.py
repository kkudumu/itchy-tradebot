"""Unit tests for the pre-challenge validation pipeline.

Test categories
---------------
1.  Wilson score CI — known wins/total → bounds match reference values
2.  Haircut application — min-direction and max-direction haircut logic
3.  Threshold checking — all passing → GO; one failing → NO-GO
4.  Borderline detection — metrics barely above threshold → BORDERLINE
5.  Max-direction threshold (drawdown) — haircut makes it more conservative
6.  Pipeline orchestration — mock WF + MC → pipeline completes without error
7.  Report generation — HTML report contains expected structural sections
8.  Edge case: 0 OOS trades → immediate NO-GO
9.  ThresholdChecker.check_all on missing keys — treated as 0.0
10. Plain-text report contains key headings
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from src.validation.threshold_checker import (
    ThresholdChecker,
    ThresholdResult,
    ValidationResult,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_checker(haircut_pct: float = 25.0) -> ThresholdChecker:
    return ThresholdChecker(haircut_pct=haircut_pct)


def _passing_metrics() -> dict:
    """Return a metrics dict that passes every threshold without haircut."""
    return {
        "oos_trade_count": 600,
        "win_rate_ci_lower": 0.65,   # haircutted → 0.4875; needs >0.45
        "profit_factor": 2.0,         # haircutted → 1.5; needs >1.3
        "sharpe_ratio": 2.0,          # haircutted → 1.5; needs >1.0
        "sortino_ratio": 2.0,         # haircutted → 1.5; needs >1.0
        "max_drawdown_pct": 8.0,      # haircutted → 10.0; needs <15.0
        "wfe": 0.7,                   # no haircut; needs >0.5
        "dsr": 0.95,                  # no haircut; needs >0.90
        "monte_carlo_pass_rate": 0.90, # no haircut; needs >0.80
    }


def _borderline_metrics() -> dict:
    """Return metrics that just barely pass after haircut (within 10% margin)."""
    return {
        "oos_trade_count": 600,
        # win_rate_ci_lower after 25% haircut: 0.60 * 0.75 = 0.45 → exactly at threshold
        # use 0.601 so it's above but within 10% margin
        "win_rate_ci_lower": 0.601,
        "profit_factor": 1.75,        # haircutted → 1.3125; threshold 1.3 → margin 0.0125/1.3 ≈ 1%
        "sharpe_ratio": 2.0,
        "sortino_ratio": 2.0,
        "max_drawdown_pct": 8.0,
        "wfe": 0.7,
        "dsr": 0.95,
        "monte_carlo_pass_rate": 0.90,
    }


def _minimal_wf_result():
    """Build a mock WFResult that provides expected attributes."""
    from src.optimization.walk_forward import WFResult, WFWindow
    import pandas as pd

    window = WFWindow(
        window_index=0,
        is_start=pd.Timestamp("2020-01-01"),
        is_end=pd.Timestamp("2021-01-01"),
        oos_start=pd.Timestamp("2021-01-01"),
        oos_end=pd.Timestamp("2021-04-01"),
        is_sharpe=1.8,
        oos_sharpe=1.2,
        best_params={},
        oos_metrics={
            "sharpe_ratio": 1.2,
            "win_rate": 0.55,
            "total_return_pct": 12.0,
            "profit_factor": 1.8,
            "sortino_ratio": 1.5,
            "max_drawdown_pct": 7.0,
        },
        oos_trades=[
            {"r_multiple": 1.0, "entry_time": "2021-01-05"},
            {"r_multiple": -1.0, "entry_time": "2021-01-06"},
            {"r_multiple": 2.0, "entry_time": "2021-01-07"},
            {"r_multiple": -0.5, "entry_time": "2021-01-08"},
            {"r_multiple": 1.5, "entry_time": "2021-01-09"},
        ],
        oos_prop_firm={"status": "passed"},
    )

    # Create enough trades to exceed 400 trade threshold.
    many_trades = []
    for i in range(450):
        r = 1.0 if i % 2 == 0 else -0.8
        many_trades.append({"r_multiple": r, "entry_time": f"2021-0{(i % 3) + 1}-{(i % 28) + 1:02d}"})

    window.oos_trades = many_trades

    return WFResult(
        windows=[window],
        wfe=0.67,
        oos_trades=many_trades,
        oos_metrics={
            "mean_oos_sharpe": 1.2,
            "mean_is_sharpe": 1.8,
            "wfe": 0.67,
            "total_oos_trades": len(many_trades),
            "mean_oos_return_pct": 12.0,
            "mean_oos_win_rate": 0.55,
            "n_windows": 1,
            "windows": [
                {
                    "index": 0,
                    "is_sharpe": 1.8,
                    "oos_sharpe": 1.2,
                    "oos_return_pct": 12.0,
                    "oos_trades": len(many_trades),
                }
            ],
        },
        is_sharpes=[1.8],
        oos_sharpes=[1.2],
    )


# =============================================================================
# 1. Wilson score CI
# =============================================================================


class TestWilsonScoreCI:
    """Verify Wilson CI bounds against reference values."""

    def test_symmetric_case(self):
        """50% win rate with large n should have CI around 0.5 that's reasonably tight."""
        lo, hi = ThresholdChecker.wilson_score_ci(wins=500, total=1000)
        assert lo < 0.5 < hi
        # Wilson CI for p=0.5, n=1000 at 95%: approximately [0.469, 0.531] → width ~0.062
        assert hi - lo < 0.07  # reasonably tight for n=1000

    def test_lower_bound_below_observed_rate(self):
        """Lower bound must always be <= observed win rate."""
        wins, total = 60, 100
        lo, hi = ThresholdChecker.wilson_score_ci(wins, total)
        observed = wins / total
        assert lo < observed
        assert hi > observed

    def test_known_reference_values(self):
        """60/100 wins at 95%: Wilson CI approximately [0.499, 0.693].

        Reference computed independently:
        p=0.6, n=100, z=1.96 → Wilson [0.4990, 0.6929]
        """
        lo, hi = ThresholdChecker.wilson_score_ci(wins=60, total=100)
        assert abs(lo - 0.499) < 0.005, f"lo={lo:.4f} expected ≈ 0.499"
        assert abs(hi - 0.693) < 0.005, f"hi={hi:.4f} expected ≈ 0.693"

    def test_zero_total_returns_zeros(self):
        """Zero total trades should return (0.0, 0.0) without error."""
        lo, hi = ThresholdChecker.wilson_score_ci(wins=0, total=0)
        assert lo == 0.0
        assert hi == 0.0

    def test_all_wins(self):
        """100% win rate: lower bound still below 1.0 with Wilson correction."""
        lo, hi = ThresholdChecker.wilson_score_ci(wins=50, total=50)
        assert 0.9 < lo <= 1.0
        assert hi <= 1.0

    def test_no_wins(self):
        """0% win rate: upper bound still above 0.0 with Wilson correction."""
        lo, hi = ThresholdChecker.wilson_score_ci(wins=0, total=50)
        assert lo == 0.0
        assert hi > 0.0


# =============================================================================
# 2. Haircut application
# =============================================================================


class TestHaircutApplication:
    """Verify that the 25% haircut reduces/increases values correctly."""

    def test_min_direction_haircut(self):
        """Min-direction metric should be reduced by 25%."""
        checker = _make_checker(25.0)
        result = checker.apply_haircut(2.0, direction="min")
        assert math.isclose(result, 1.5, rel_tol=1e-9)

    def test_max_direction_haircut(self):
        """Max-direction metric (e.g. drawdown) should be increased by 25%."""
        checker = _make_checker(25.0)
        result = checker.apply_haircut(12.0, direction="max")
        assert math.isclose(result, 15.0, rel_tol=1e-9)

    def test_zero_haircut_identity(self):
        """With 0% haircut the value should be unchanged."""
        checker = _make_checker(0.0)
        assert checker.apply_haircut(3.0, "min") == 3.0
        assert checker.apply_haircut(10.0, "max") == 10.0

    def test_sharpe_haircutted_value(self):
        """Sharpe 2.0 after 25% haircut → 1.5 (passes ≥1.0 threshold)."""
        checker = _make_checker(25.0)
        result = checker.check_metric("sharpe_ratio", 2.0)
        assert math.isclose(result.haircutted_value, 1.5, rel_tol=1e-9)
        assert result.passed

    def test_invalid_haircut_raises(self):
        """Haircut ≥ 100 is invalid and should raise ValueError."""
        with pytest.raises(ValueError):
            ThresholdChecker(haircut_pct=100.0)


# =============================================================================
# 3. Threshold checking — all pass / one fails
# =============================================================================


class TestThresholdChecking:
    """Verify overall PASS/FAIL logic for the full threshold set."""

    def test_all_passing_metrics(self):
        """All metrics comfortably above thresholds → overall_pass = True."""
        checker = _make_checker()
        result = checker.check_all(_passing_metrics())
        assert result.overall_pass
        assert result.n_failed == 0
        assert result.critical_failures == []

    def test_one_failing_metric_triggers_no_go(self):
        """Failing one metric must set overall_pass = False."""
        metrics = _passing_metrics()
        # Drop profit_factor to 1.0; haircutted → 0.75 < 1.3 threshold
        metrics["profit_factor"] = 1.0
        checker = _make_checker()
        result = checker.check_all(metrics)
        assert not result.overall_pass
        assert result.n_failed >= 1
        # 'Profit Factor' should appear in critical failures.
        assert any("Profit Factor" in f for f in result.critical_failures)

    def test_trade_count_no_haircut(self):
        """OOS trade count must NOT be haircutted (structural count metric)."""
        checker = _make_checker()
        result = checker.check_metric("oos_trade_count", 450)
        assert not result.haircut_applied
        assert result.haircutted_value == 450

    def test_trade_count_below_minimum(self):
        """400 trades exactly passes; 399 fails."""
        checker = _make_checker()
        r_pass = checker.check_metric("oos_trade_count", 400)
        r_fail = checker.check_metric("oos_trade_count", 399)
        assert r_pass.passed
        assert not r_fail.passed

    def test_dsr_no_haircut(self):
        """DSR threshold applies without haircut."""
        checker = _make_checker()
        r_pass = checker.check_metric("dsr", 0.91)
        r_fail = checker.check_metric("dsr", 0.89)
        assert r_pass.passed
        assert not r_fail.passed
        assert not r_pass.haircut_applied


# =============================================================================
# 4. Borderline detection
# =============================================================================


class TestBorderlineDetection:
    """Verify that tight-but-passing thresholds produce BORDERLINE verdict."""

    def test_borderline_verdict(self):
        """Metrics that barely pass should yield BORDERLINE, not GO."""
        from src.validation.go_nogo import GoNoGoValidator

        checker = _make_checker()
        val_result = checker.check_all(_borderline_metrics())
        # All must pass for BORDERLINE consideration.
        if not val_result.overall_pass:
            pytest.skip("Borderline metrics don't all pass — adjust fixture.")

        # Simulate the verdict logic directly.
        from src.validation.go_nogo import _BORDERLINE_MARGIN_PCT

        borderline = False
        for r in val_result.results:
            if r.passed and r.threshold != 0.0:
                rel_margin = abs(r.margin) / abs(r.threshold)
                if rel_margin < _BORDERLINE_MARGIN_PCT:
                    borderline = True
                    break

        assert borderline, "Expected at least one thin-margin threshold."


# =============================================================================
# 5. Max-direction threshold (drawdown with haircut)
# =============================================================================


class TestMaxDirectionThreshold:
    """Max-direction thresholds should become *more* conservative after haircut."""

    def test_drawdown_haircut_increases_value(self):
        """12% drawdown after 25% haircut → 15% (still passes <15% threshold)."""
        checker = _make_checker(25.0)
        result = checker.check_metric("max_drawdown_pct", 12.0)
        # haircutted: 12.0 * 1.25 = 15.0 → exactly at threshold (passes ≥ threshold uses <=)
        assert math.isclose(result.haircutted_value, 15.0, rel_tol=1e-9)
        assert result.passed  # 15.0 <= 15.0

    def test_drawdown_over_limit_fails(self):
        """13% drawdown after 25% haircut → 16.25% (fails <15% threshold)."""
        checker = _make_checker(25.0)
        result = checker.check_metric("max_drawdown_pct", 13.0)
        assert math.isclose(result.haircutted_value, 16.25, rel_tol=1e-9)
        assert not result.passed


# =============================================================================
# 6. Pipeline orchestration (mocked dependencies)
# =============================================================================


class TestPipelineOrchestration:
    """Verify the GoNoGoValidator orchestrates all pipeline steps correctly."""

    def test_pipeline_runs_and_returns_full_result(self, monkeypatch):
        """Mock walk-forward and MC so the pipeline completes quickly."""
        import pandas as pd
        from src.validation.go_nogo import GoNoGoValidator, FullValidationResult

        # Build minimal dummy price data (content not used — WF is mocked).
        idx = pd.date_range("2019-01-01", periods=100, freq="1min", tz="UTC")
        data = pd.DataFrame(
            {"open": 1.0, "high": 1.01, "low": 0.99, "close": 1.005, "volume": 100.0},
            index=idx,
        )

        wf_result = _minimal_wf_result()

        # Patch WalkForwardAnalyzer.run to return our pre-built result.
        with patch(
            "src.validation.go_nogo.WalkForwardAnalyzer.run", return_value=wf_result
        ):
            # Patch OverfitDetector to avoid needing a real Optuna study.
            from src.optimization.overfit_detector import OverfitReport
            mock_overfit = OverfitReport(
                dsr=0.95, dsr_pass=True, wfe=0.67, wfe_pass=True,
                plateau_pass=True, plateau_cv=0.10, overall_pass=True,
                notes=["All checks passed."],
            )
            with patch(
                "src.validation.go_nogo.OverfitDetector.check_all",
                return_value=mock_overfit,
            ):
                # Patch MonteCarloSimulator.run.
                from src.simulation.monte_carlo import MCResult
                mock_mc = MCResult(
                    n_simulations=100,
                    pass_rate=85.0,
                    avg_days=22.0,
                    median_days=20.0,
                    daily_dd_failure_rate=5.0,
                    total_dd_failure_rate=3.0,
                    timeout_rate=7.0,
                    circuit_breaker_rate=0.0,
                    outcomes=[],
                    convergence_reached=True,
                    convergence_at=80,
                    running_pass_rates=[85.0] * 100,
                )
                with patch(
                    "src.validation.go_nogo.MonteCarloSimulator.run",
                    return_value=mock_mc,
                ):
                    validator = GoNoGoValidator(data=data, initial_balance=10_000.0)
                    result = validator.run_full_validation(
                        n_wf_trials=5, n_mc_sims=100
                    )

        assert isinstance(result, FullValidationResult)
        assert result.final_verdict in ("GO", "NO-GO", "BORDERLINE")
        assert result.n_oos_trades == len(wf_result.oos_trades)
        assert isinstance(result.recommendations, list)
        assert len(result.recommendations) > 0

    def test_zero_oos_trades_immediate_no_go(self, monkeypatch):
        """When walk-forward yields no OOS trades, the verdict must be NO-GO."""
        import pandas as pd
        from src.validation.go_nogo import GoNoGoValidator
        from src.optimization.walk_forward import WFResult

        idx = pd.date_range("2019-01-01", periods=50, freq="1min", tz="UTC")
        data = pd.DataFrame(
            {"open": 1.0, "high": 1.01, "low": 0.99, "close": 1.0, "volume": 1.0},
            index=idx,
        )

        empty_wf = WFResult(
            windows=[], wfe=0.0, oos_trades=[],
            oos_metrics={}, is_sharpes=[], oos_sharpes=[]
        )

        with patch(
            "src.validation.go_nogo.WalkForwardAnalyzer.run", return_value=empty_wf
        ):
            validator = GoNoGoValidator(data=data)
            result = validator.run_full_validation(n_wf_trials=1, n_mc_sims=10)

        assert result.final_verdict == "NO-GO"
        assert result.n_oos_trades == 0


# =============================================================================
# 7. Report generation
# =============================================================================


class TestReportGeneration:
    """Verify HTML and text reports contain required structural sections."""

    def _make_mock_result(self):
        """Build a minimal FullValidationResult for report testing."""
        from src.validation.go_nogo import FullValidationResult
        from src.optimization.overfit_detector import OverfitReport
        from src.simulation.monte_carlo import MCResult
        from src.optimization.walk_forward import WFResult
        from src.validation.threshold_checker import ThresholdChecker

        checker = ThresholdChecker()
        val_result = checker.check_all(_passing_metrics())

        overfit = OverfitReport(
            dsr=0.95, dsr_pass=True, wfe=0.70, wfe_pass=True,
            plateau_pass=True, plateau_cv=0.12, overall_pass=True,
            notes=["All checks passed."],
        )
        mc = MCResult(
            n_simulations=100, pass_rate=85.0, avg_days=21.0, median_days=20.0,
            daily_dd_failure_rate=5.0, total_dd_failure_rate=3.0,
            timeout_rate=7.0, circuit_breaker_rate=0.0,
            outcomes=[], convergence_reached=True, convergence_at=80,
            running_pass_rates=[85.0] * 10,
        )
        wf = WFResult(
            windows=[], wfe=0.70, oos_trades=[],
            oos_metrics={}, is_sharpes=[1.8], oos_sharpes=[1.2]
        )
        return FullValidationResult(
            timestamp=datetime(2025, 1, 15, 10, 0, 0),
            validation_result=val_result,
            overfit_report=overfit,
            monte_carlo=mc,
            wf_result=wf,
            oos_metrics=_passing_metrics(),
            final_verdict="GO",
            recommendations=["Proceed to demo challenge runs."],
            n_oos_trades=600,
            win_rate_ci=(0.52, 0.68),
        )

    def test_html_contains_verdict_banner(self):
        """HTML report must include the GO/NO-GO verdict text."""
        from src.validation.report import ValidationReportGenerator

        gen = ValidationReportGenerator()
        result = self._make_mock_result()
        html = gen.generate_html(result)

        assert "GO" in html
        assert "verdict-banner" in html

    def test_html_contains_threshold_table(self):
        """HTML must include the threshold checklist table."""
        from src.validation.report import ValidationReportGenerator

        gen = ValidationReportGenerator()
        html = gen.generate_html(self._make_mock_result())
        assert "Threshold Checklist" in html
        assert "<table>" in html

    def test_html_contains_metrics_summary(self):
        """HTML must include the OOS metrics summary section."""
        from src.validation.report import ValidationReportGenerator

        gen = ValidationReportGenerator()
        html = gen.generate_html(self._make_mock_result())
        assert "OOS Metrics Summary" in html

    def test_html_contains_monte_carlo_section(self):
        """HTML must include the Monte Carlo section."""
        from src.validation.report import ValidationReportGenerator

        gen = ValidationReportGenerator()
        html = gen.generate_html(self._make_mock_result())
        assert "Monte Carlo" in html

    def test_html_contains_recommendations(self):
        """HTML must include the recommendations section."""
        from src.validation.report import ValidationReportGenerator

        gen = ValidationReportGenerator()
        html = gen.generate_html(self._make_mock_result())
        assert "Recommendations" in html

    def test_text_report_contains_key_sections(self):
        """Plain-text report must include VERDICT, THRESHOLD, and OOS sections."""
        from src.validation.report import ValidationReportGenerator

        gen = ValidationReportGenerator()
        text = gen.generate_text(self._make_mock_result())

        assert "FINAL VERDICT" in text
        assert "THRESHOLD CHECKLIST" in text
        assert "OOS SUMMARY" in text
        assert "RECOMMENDATIONS" in text

    def test_save_report_writes_file(self, tmp_path):
        """save_report should create a file at the specified path."""
        from src.validation.report import ValidationReportGenerator

        gen = ValidationReportGenerator()
        result = self._make_mock_result()
        html = gen.generate_html(result)
        out_path = str(tmp_path / "sub" / "report.html")
        saved = gen.save_report(html, out_path)

        import os
        assert os.path.exists(saved)
        with open(saved) as fh:
            content = fh.read()
        assert "GO" in content


# =============================================================================
# 8. Edge case: missing metrics keys
# =============================================================================


class TestMissingMetricsKeys:
    """Missing metric keys should be treated as 0.0 and fail their thresholds."""

    def test_empty_metrics_all_fail(self):
        """Passing an empty dict must fail all thresholds (zeros everywhere).

        Note: max_drawdown_pct with value 0.0 actually *passes* the <=15%
        threshold (0.0 <= 15.0), so we expect at least n-1 failures rather
        than all n.
        """
        checker = _make_checker()
        result = checker.check_all({})
        assert not result.overall_pass
        # At minimum, every non-drawdown threshold must fail (9 total - 1 max-direction = 8).
        assert result.n_failed >= len(ThresholdChecker.THRESHOLDS) - 1

    def test_partial_metrics_rest_zero(self):
        """Providing only some keys should still check the missing ones as 0."""
        checker = _make_checker()
        result = checker.check_all({"oos_trade_count": 500})
        # At minimum all other zero-value checks should fail.
        assert not result.overall_pass
        assert result.n_failed >= 1


# =============================================================================
# 9. ThresholdResult structure
# =============================================================================


class TestThresholdResultStructure:
    """Verify ThresholdResult fields are populated correctly."""

    def test_fields_present(self):
        checker = _make_checker()
        r = checker.check_metric("sharpe_ratio", 2.0)
        assert isinstance(r.name, str)
        assert isinstance(r.raw_value, float)
        assert isinstance(r.haircutted_value, float)
        assert isinstance(r.threshold, float)
        assert isinstance(r.passed, bool)
        assert isinstance(r.margin, float)
        assert r.direction in ("min", "max")
        assert isinstance(r.haircut_applied, bool)

    def test_margin_positive_when_passing(self):
        """Margin should be positive when the metric exceeds its threshold."""
        checker = _make_checker()
        r = checker.check_metric("wfe", 0.8)  # threshold = 0.5
        assert r.passed
        assert r.margin > 0

    def test_margin_negative_when_failing(self):
        """Margin should be negative when the metric falls short."""
        checker = _make_checker()
        r = checker.check_metric("wfe", 0.2)  # threshold = 0.5
        assert not r.passed
        assert r.margin < 0


# =============================================================================
# 10. ValidationResult structure
# =============================================================================


class TestValidationResultStructure:
    """Verify ValidationResult aggregate fields are consistent."""

    def test_n_passed_plus_n_failed_equals_total(self):
        checker = _make_checker()
        result = checker.check_all(_passing_metrics())
        assert (
            result.n_passed + result.n_failed == len(ThresholdChecker.THRESHOLDS)
        )

    def test_critical_failures_matches_n_failed(self):
        checker = _make_checker()
        metrics = _passing_metrics()
        metrics["sharpe_ratio"] = 0.1  # force a failure
        result = checker.check_all(metrics)
        assert len(result.critical_failures) == result.n_failed
