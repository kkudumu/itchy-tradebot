"""Full pre-challenge go/no-go validation pipeline.

Pipeline
--------
1. Walk-forward analysis (12-month IS / 3-month OOS windows).
2. Collect all OOS trades from every window.
3. Calculate performance metrics on the combined OOS trade set.
4. Apply 25% haircut to magnitude-based metrics.
5. Check all go/no-go thresholds → PASS/FAIL per threshold.
6. Compute Wilson score CI for win rate; check lower bound >= 45%.
7. Run Monte Carlo from the OOS trade distribution.
8. Compile final verdict: GO / BORDERLINE / NO-GO.

BORDERLINE verdict
------------------
The system returns BORDERLINE when all thresholds pass but one or more
metrics are within 10% of their threshold (after haircut).  A BORDERLINE
result warrants additional manual review before committing to a live challenge.

Demo challenge validation
-------------------------
As a complementary step (not automated here), run three simultaneous demo
challenges on The5ers demo accounts using the optimised parameters and live
XAU/USD data.  All three must complete successfully before a funded challenge
attempt is justified.  This off-system step is captured in the final report's
recommendations section.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.optimization.overfit_detector import OverfitDetector, OverfitReport
from src.optimization.walk_forward import WalkForwardAnalyzer, WFResult
from src.simulation.monte_carlo import MCResult, MonteCarloSimulator
from src.validation.threshold_checker import ThresholdChecker, ValidationResult

logger = logging.getLogger(__name__)

# Fraction below threshold that still counts as BORDERLINE (not clean GO).
_BORDERLINE_MARGIN_PCT: float = 0.10


# =============================================================================
# Result dataclass
# =============================================================================


@dataclass
class FullValidationResult:
    """Complete output from the pre-challenge validation pipeline.

    Attributes
    ----------
    timestamp:
        UTC time when the validation run was initiated.
    validation_result:
        Per-threshold pass/fail breakdown with haircut details.
    overfit_report:
        DSR, WFE, and plateau check results from OverfitDetector.
    monte_carlo:
        MCResult from the Monte Carlo simulation over OOS trades.
    wf_result:
        Raw WalkForwardAnalyzer output (windows, WFE, aggregated metrics).
    oos_metrics:
        Computed metrics dict for the combined OOS trade set.
    final_verdict:
        One of 'GO', 'NO-GO', or 'BORDERLINE'.
    recommendations:
        Human-readable list of action items based on the verdict.
    n_oos_trades:
        Total OOS trades collected across all walk-forward windows.
    win_rate_ci:
        (lower, upper) Wilson 95% CI tuple for the OOS win rate.
    """

    timestamp: datetime
    validation_result: ValidationResult
    overfit_report: OverfitReport
    monte_carlo: MCResult
    wf_result: WFResult
    oos_metrics: dict
    final_verdict: str
    recommendations: List[str]
    n_oos_trades: int = 0
    win_rate_ci: tuple = field(default_factory=lambda: (0.0, 0.0))


# =============================================================================
# GoNoGoValidator
# =============================================================================


class GoNoGoValidator:
    """Full pre-challenge validation pipeline orchestrator.

    Parameters
    ----------
    data:
        1-minute OHLCV DataFrame with a UTC DatetimeIndex.
    config:
        Optional configuration overrides passed to the walk-forward analyzer
        and Monte Carlo simulator.
    initial_balance:
        Starting account balance for backtests.  Default: 10,000.
    haircut_pct:
        Haircut percentage for performance metrics.  Default: 25.0.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config: Optional[dict] = None,
        initial_balance: float = 10_000.0,
        haircut_pct: float = 25.0,
    ) -> None:
        self._data = data
        self._config = config or {}
        self._initial_balance = initial_balance

        self._walk_forward = WalkForwardAnalyzer(
            is_months=self._config.get("is_months", 12),
            oos_months=self._config.get("oos_months", 3),
            initial_balance=initial_balance,
        )
        self._overfit_detector = OverfitDetector()
        self._monte_carlo = MonteCarloSimulator(
            initial_balance=initial_balance,
            profit_target_pct=self._config.get("profit_target_pct", 8.0),
            max_daily_dd_pct=self._config.get("max_daily_dd_pct", 5.0),
            max_total_dd_pct=self._config.get("max_total_dd_pct", 10.0),
        )
        self._threshold_checker = ThresholdChecker(haircut_pct=haircut_pct)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_full_validation(
        self,
        n_wf_trials: int = 200,
        n_mc_sims: int = 10_000,
        storage: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> FullValidationResult:
        """Execute the complete validation pipeline.

        Parameters
        ----------
        n_wf_trials:
            Optuna trials per walk-forward IS window.  Default: 200.
        n_mc_sims:
            Monte Carlo simulation count.  Default: 10,000.
        storage:
            Optional PostgreSQL URL for Optuna trial persistence.
        seed:
            Random seed for the Monte Carlo simulation.

        Returns
        -------
        FullValidationResult
        """
        timestamp = datetime.utcnow()
        logger.info("Starting pre-challenge validation — %s", timestamp.isoformat())

        # ----------------------------------------------------------------
        # Step 1: Walk-forward analysis
        # ----------------------------------------------------------------
        logger.info("Step 1/7 — Running walk-forward analysis (n_trials=%d).", n_wf_trials)
        wf_result = self._walk_forward.run(
            data=self._data, n_trials=n_wf_trials, storage=storage
        )

        # ----------------------------------------------------------------
        # Step 2: Collect OOS trades
        # ----------------------------------------------------------------
        logger.info("Step 2/7 — Collecting OOS trades from walk-forward windows.")
        oos_trades = self._collect_oos_trades(wf_result)
        n_oos_trades = len(oos_trades)
        logger.info("Total OOS trades collected: %d", n_oos_trades)

        # Early exit: zero OOS trades is an immediate NO-GO.
        if n_oos_trades == 0:
            logger.warning("No OOS trades available — immediate NO-GO.")
            return self._immediate_no_go(
                timestamp=timestamp,
                wf_result=wf_result,
                reason="Zero OOS trades collected from walk-forward windows.",
            )

        # ----------------------------------------------------------------
        # Step 3: Calculate OOS metrics
        # ----------------------------------------------------------------
        logger.info("Step 3/7 — Calculating OOS performance metrics.")
        oos_metrics = self._calculate_oos_metrics(oos_trades, wf_result)

        # ----------------------------------------------------------------
        # Step 4: Wilson CI for win rate
        # ----------------------------------------------------------------
        logger.info("Step 4/7 — Computing Wilson score CI for win rate.")
        wins = sum(1 for t in oos_trades if float(t.get("r_multiple") or 0.0) > 0)
        ci_lower, ci_upper = ThresholdChecker.wilson_score_ci(wins, n_oos_trades)
        win_rate_ci = (ci_lower, ci_upper)
        oos_metrics["win_rate_ci_lower"] = ci_lower
        logger.info(
            "Win rate: %.1f%% — Wilson 95%% CI: [%.4f, %.4f]",
            wins / n_oos_trades * 100.0,
            ci_lower,
            ci_upper,
        )

        # ----------------------------------------------------------------
        # Step 5: Overfitting checks (DSR via placeholder study)
        # ----------------------------------------------------------------
        logger.info("Step 5/7 — Running overfitting checks.")
        overfit_report = self._run_overfit_checks(wf_result, n_wf_trials)
        oos_metrics["dsr"] = overfit_report.dsr
        oos_metrics["wfe"] = wf_result.wfe

        # ----------------------------------------------------------------
        # Step 6: Monte Carlo simulation
        # ----------------------------------------------------------------
        logger.info("Step 6/7 — Running Monte Carlo simulation (n=%d).", n_mc_sims)
        mc_result = self._run_monte_carlo(oos_trades, n_mc_sims, seed)
        # pass_rate is in [0, 100]; normalise to [0, 1] for threshold check
        oos_metrics["monte_carlo_pass_rate"] = mc_result.pass_rate / 100.0
        logger.info("MC pass rate: %.2f%%", mc_result.pass_rate)

        # ----------------------------------------------------------------
        # Step 7: Threshold checking
        # ----------------------------------------------------------------
        logger.info("Step 7/7 — Checking go/no-go thresholds.")
        validation_result = self._threshold_checker.check_all(oos_metrics)

        # ----------------------------------------------------------------
        # Final verdict
        # ----------------------------------------------------------------
        final_verdict = self._determine_verdict(validation_result)
        recommendations = self._build_recommendations(
            validation_result, overfit_report, mc_result, final_verdict
        )

        logger.info(
            "Validation complete — verdict: %s  (%d/%d thresholds passed)",
            final_verdict,
            validation_result.n_passed,
            validation_result.n_passed + validation_result.n_failed,
        )

        return FullValidationResult(
            timestamp=timestamp,
            validation_result=validation_result,
            overfit_report=overfit_report,
            monte_carlo=mc_result,
            wf_result=wf_result,
            oos_metrics=oos_metrics,
            final_verdict=final_verdict,
            recommendations=recommendations,
            n_oos_trades=n_oos_trades,
            win_rate_ci=win_rate_ci,
        )

    # ------------------------------------------------------------------
    # Internal pipeline steps
    # ------------------------------------------------------------------

    def _collect_oos_trades(self, wf_result: WFResult) -> List[dict]:
        """Extract and concatenate OOS trades from all walk-forward windows."""
        return list(wf_result.oos_trades)

    def _calculate_oos_metrics(
        self, oos_trades: List[dict], wf_result: WFResult
    ) -> dict:
        """Calculate performance metrics from the combined OOS trade set.

        Reconstructs a synthetic equity curve from R-multiples and a fixed
        risk-per-trade, then delegates to PerformanceMetrics.calculate().
        """
        from src.backtesting.metrics import PerformanceMetrics

        if not oos_trades:
            return {}

        # Build a synthetic equity curve from R-multiples.
        # Use 1% risk per trade as a neutral normalisation assumption.
        # Use a monotonically increasing synthetic DatetimeIndex at 5-minute
        # intervals starting from an arbitrary epoch.  This avoids duplicate-
        # label errors that arise when multiple trades share the same date string.
        risk_pct = 0.01
        balance = float(self._initial_balance)
        equity_values = [balance]

        for trade in oos_trades:
            r = float(trade.get("r_multiple") or 0.0)
            pnl = r * balance * risk_pct
            balance += pnl
            equity_values.append(balance)

        epoch = pd.Timestamp("2020-01-01", tz="UTC")
        all_timestamps = [
            epoch + pd.Timedelta(minutes=5 * i) for i in range(len(equity_values))
        ]
        equity_series = pd.Series(equity_values, index=all_timestamps)

        calc = PerformanceMetrics()
        metrics = calc.calculate(oos_trades, equity_series, self._initial_balance)

        # Augment with walk-forward aggregate data.
        metrics["oos_trade_count"] = len(oos_trades)
        metrics["wfe"] = wf_result.wfe
        metrics["mean_oos_sharpe"] = wf_result.oos_metrics.get("mean_oos_sharpe", 0.0)

        # Handle profit_factor None (inf case from PerformanceMetrics).
        if metrics.get("profit_factor") is None:
            metrics["profit_factor"] = 9.99  # large but finite; treat as passing

        return metrics

    def _run_overfit_checks(
        self, wf_result: WFResult, n_trials: int
    ) -> OverfitReport:
        """Run DSR and WFE overfitting checks without a live Optuna study.

        When called outside a live optimisation session, we create a minimal
        placeholder study to allow OverfitDetector.check_all() to execute.
        The DSR is computed directly from the walk-forward OOS Sharpe
        distribution rather than from Optuna trial values, providing a
        meaningful signal without requiring a full study object.
        """
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Create a synthetic study populated from the WF OOS Sharpe values.
        study = optuna.create_study(direction="maximize")

        oos_sharpes = wf_result.oos_sharpes or []
        if not oos_sharpes:
            oos_sharpes = [0.0]

        # Add synthetic trials so OverfitDetector has data to work with.
        for sharpe_val in oos_sharpes:
            trial = optuna.trial.create_trial(
                params={},
                distributions={},
                value=sharpe_val if math.isfinite(sharpe_val) else 0.0,
            )
            study.add_trial(trial)

        try:
            report = self._overfit_detector.check_all(
                study=study,
                wf_result=wf_result,
                n_trials=max(n_trials, len(oos_sharpes)),
            )
        except Exception as exc:
            logger.warning("OverfitDetector raised an exception: %s", exc)
            # Return a conservative failing report on error.
            from src.optimization.overfit_detector import OverfitReport
            report = OverfitReport(
                dsr=0.0,
                dsr_pass=False,
                wfe=wf_result.wfe,
                wfe_pass=wf_result.wfe >= 0.5,
                plateau_pass=False,
                plateau_cv=1.0,
                overall_pass=False,
                notes=["OverfitDetector error; conservative values substituted."],
            )

        return report

    def _run_monte_carlo(
        self,
        oos_trades: List[dict],
        n_mc_sims: int,
        seed: Optional[int],
    ) -> MCResult:
        """Run the Monte Carlo simulation from OOS trade results."""
        try:
            return self._monte_carlo.run(
                trade_results=oos_trades,
                n_simulations=n_mc_sims,
                use_fat_tails=True,
                block_bootstrap=True,
                seed=seed,
            )
        except Exception as exc:
            logger.error("Monte Carlo simulation failed: %s", exc)
            # Return a zero-pass-rate result to trigger NO-GO on MC threshold.
            from src.simulation.monte_carlo import MCResult, ChallengeOutcome
            return MCResult(
                n_simulations=0,
                pass_rate=0.0,
                avg_days=0.0,
                median_days=0.0,
                daily_dd_failure_rate=0.0,
                total_dd_failure_rate=0.0,
                timeout_rate=0.0,
                circuit_breaker_rate=0.0,
                outcomes=[],
                convergence_reached=False,
                convergence_at=0,
                running_pass_rates=[],
            )

    # ------------------------------------------------------------------
    # Verdict logic
    # ------------------------------------------------------------------

    def _determine_verdict(self, validation_result: ValidationResult) -> str:
        """Derive the final verdict from threshold results.

        - NO-GO: any threshold fails.
        - BORDERLINE: all pass but one or more have margin < 10% of threshold.
        - GO: all thresholds pass with comfortable margin.
        """
        if not validation_result.overall_pass:
            return "NO-GO"

        # Check if any passing threshold is uncomfortably close to its limit.
        borderline = False
        for r in validation_result.results:
            if not r.passed:
                continue  # already captured by overall_pass
            # margin is absolute distance from threshold.
            # Borderline when within 10% of the threshold value.
            if r.threshold != 0.0:
                relative_margin = abs(r.margin) / abs(r.threshold)
                if relative_margin < _BORDERLINE_MARGIN_PCT:
                    borderline = True
                    break

        return "BORDERLINE" if borderline else "GO"

    def _build_recommendations(
        self,
        validation_result: ValidationResult,
        overfit_report: OverfitReport,
        mc_result: MCResult,
        verdict: str,
    ) -> List[str]:
        """Generate actionable recommendations based on the validation outcome."""
        recs: List[str] = []

        if verdict == "GO":
            recs.append(
                "All thresholds passed with comfortable margin.  "
                "Proceed to live challenge after completing three simultaneous "
                "demo challenge runs with the optimised parameters."
            )
        elif verdict == "BORDERLINE":
            recs.append(
                "All thresholds pass but some margins are thin.  "
                "Consider extending the data window or adding more Optuna trials "
                "before committing to a funded challenge."
            )
            # Identify which thresholds are close.
            for r in validation_result.results:
                if r.passed and r.threshold != 0.0:
                    rel = abs(r.margin) / abs(r.threshold)
                    if rel < _BORDERLINE_MARGIN_PCT:
                        recs.append(
                            f"  - '{r.name}' is within {rel * 100:.1f}% of its "
                            f"threshold ({r.threshold}); raw={r.raw_value:.4f}, "
                            f"haircutted={r.haircutted_value:.4f}."
                        )
        else:  # NO-GO
            recs.append(
                "One or more thresholds failed.  Do not submit a funded challenge."
            )
            for name in validation_result.critical_failures:
                recs.append(f"  - FAILED: {name}")

        # Overfitting-specific notes.
        if not overfit_report.dsr_pass:
            recs.append(
                f"DSR={overfit_report.dsr:.4f} is below 0.90.  "
                "Increase n_trials or reduce the parameter search space to lower "
                "multiple-testing inflation."
            )
        if not overfit_report.wfe_pass:
            recs.append(
                f"WFE={overfit_report.wfe:.4f} is below 0.50.  "
                "Strategy is overfitting in-sample.  Simplify the model or "
                "extend the OOS evaluation period."
            )

        # Monte Carlo notes.
        if mc_result.pass_rate < 80.0:
            recs.append(
                f"Monte Carlo pass rate is {mc_result.pass_rate:.1f}% — below 80%.  "
                f"Failure breakdown: daily DD {mc_result.daily_dd_failure_rate:.1f}%, "
                f"total DD {mc_result.total_dd_failure_rate:.1f}%, "
                f"timeout {mc_result.timeout_rate:.1f}%."
            )

        # Always include the demo challenge reminder.
        recs.append(
            "Regardless of the automated verdict, validate with three simultaneous "
            "demo challenge runs before risking real capital."
        )

        return recs

    # ------------------------------------------------------------------
    # Immediate NO-GO helper
    # ------------------------------------------------------------------

    def _immediate_no_go(
        self,
        timestamp: datetime,
        wf_result: WFResult,
        reason: str,
    ) -> FullValidationResult:
        """Return a FullValidationResult with NO-GO and a single failure note."""
        from src.optimization.overfit_detector import OverfitReport
        from src.simulation.monte_carlo import MCResult

        empty_overfit = OverfitReport(
            dsr=0.0,
            dsr_pass=False,
            wfe=0.0,
            wfe_pass=False,
            plateau_pass=False,
            plateau_cv=0.0,
            overall_pass=False,
            notes=[reason],
        )
        empty_mc = MCResult(
            n_simulations=0,
            pass_rate=0.0,
            avg_days=0.0,
            median_days=0.0,
            daily_dd_failure_rate=0.0,
            total_dd_failure_rate=0.0,
            timeout_rate=0.0,
            circuit_breaker_rate=0.0,
            outcomes=[],
            convergence_reached=False,
            convergence_at=0,
            running_pass_rates=[],
        )
        empty_validation = self._threshold_checker.check_all({})

        return FullValidationResult(
            timestamp=timestamp,
            validation_result=empty_validation,
            overfit_report=empty_overfit,
            monte_carlo=empty_mc,
            wf_result=wf_result,
            oos_metrics={},
            final_verdict="NO-GO",
            recommendations=[
                f"NO-GO — {reason}",
                "Ensure the data window is long enough to produce at least "
                "one complete walk-forward window (12-month IS + 3-month OOS).",
            ],
            n_oos_trades=0,
            win_rate_ci=(0.0, 0.0),
        )
