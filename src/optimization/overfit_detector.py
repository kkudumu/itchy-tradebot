"""Overfitting detection for Optuna optimization studies.

Three complementary tests guard against over-fitted strategy parameters:

Deflated Sharpe Ratio (DSR)
    Bailey & Lopez de Prado (2014) — adjusts the observed Sharpe ratio
    downward to account for multiple testing across *n_trials* parameter
    combinations.  A DSR > 0.95 indicates that the best observed Sharpe
    is unlikely to be a statistical artefact of the search.

Walk-Forward Efficiency (WFE)
    Sourced from ``WalkForwardAnalyzer``.  A WFE > 0.5 indicates that
    the in-sample edge generalises to out-of-sample data.

Plateau Test
    Checks whether the top decile of Optuna trials produces consistently
    high scores or whether performance collapses outside the single best
    parameter set.  A robust strategy should have a broad performance
    plateau near the optimum rather than a narrow spike.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import optuna

logger = logging.getLogger(__name__)

# Thresholds used for the pass/fail flags in OverfitReport.
_DSR_THRESHOLD: float = 0.95
_WFE_THRESHOLD: float = 0.50
# Plateau passes when the coefficient of variation across top trials
# is below this limit (lower CV = flatter, more robust plateau).
_PLATEAU_CV_THRESHOLD: float = 0.30


# =============================================================================
# OverfitReport
# =============================================================================


@dataclass
class OverfitReport:
    """Consolidated overfitting check results.

    Attributes
    ----------
    dsr:
        Deflated Sharpe Ratio value.
    dsr_pass:
        ``True`` when DSR ≥ 0.95.
    wfe:
        Walk-Forward Efficiency value.
    wfe_pass:
        ``True`` when WFE ≥ 0.50.
    plateau_pass:
        ``True`` when the top-decile CV is below the threshold.
    plateau_cv:
        Coefficient of variation across top-decile trial scores.
    overall_pass:
        ``True`` only when all three individual checks pass.
    notes:
        Human-readable summary of each check result.
    """

    dsr: float
    dsr_pass: bool
    wfe: float
    wfe_pass: bool
    plateau_pass: bool
    plateau_cv: float
    overall_pass: bool
    notes: List[str]


# =============================================================================
# OverfitDetector
# =============================================================================


class OverfitDetector:
    """Run statistical overfitting checks on Optuna studies and WF results.

    Usage
    -----
    ::

        detector = OverfitDetector()
        report = detector.check_all(study, wf_result, n_trials=300)
        if not report.overall_pass:
            logger.warning("Overfitting detected: %s", report.notes)
    """

    # ------------------------------------------------------------------
    # DSR
    # ------------------------------------------------------------------

    def deflated_sharpe_ratio(
        self,
        sharpe: float,
        n_trials: int,
        var_sharpe: float,
        skew: float,
        kurtosis: float,
        T: int,
    ) -> float:
        """Compute the Deflated Sharpe Ratio.

        Implements the formula from Bailey & Lopez de Prado (2014),
        "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
        Overfitting and Non-Normality"::

            DSR(SR*) = P[ SR* > max_expected_SR ]

        where ``max_expected_SR`` is approximated via the expected maximum
        Sharpe from *n_trials* independent tests using the Euler-Mascheroni
        constant and the Gumbel distribution.

        Parameters
        ----------
        sharpe:
            Best annualised Sharpe ratio observed across *n_trials*.
        n_trials:
            Total number of parameter combinations evaluated (controls the
            multiple-testing correction).
        var_sharpe:
            Variance of Sharpe ratios across all trials (cross-sectional,
            not time-series variance).
        skew:
            Skewness of the strategy's return distribution.
        kurtosis:
            Excess kurtosis of the return distribution.
        T:
            Number of observations (trading days) in the backtest.

        Returns
        -------
        float
            DSR in [0, 1].  Values ≥ 0.95 are considered acceptable.
        """
        if T <= 1 or n_trials <= 0 or var_sharpe < 0:
            return 0.0

        # Expected maximum Sharpe from Gumbel distribution of order statistics.
        # Euler-Mascheroni constant γ ≈ 0.5772.
        gamma = 0.5772156649
        e_max_sr = math.sqrt(var_sharpe) * (
            (1 - gamma) * _inv_normal(1.0 - 1.0 / n_trials)
            + gamma * _inv_normal(1.0 - 1.0 / (n_trials * math.e))
        )

        # Non-normality correction using moments of the return distribution.
        # The correction reduces the effective Sharpe when returns are
        # fat-tailed or negatively skewed.
        sr_star_num = (sharpe - e_max_sr) * math.sqrt(T - 1)
        radicand = (
            1.0
            - skew * sharpe
            + ((kurtosis - 1) / 4.0) * (sharpe ** 2)
        )
        # Guard against numerical issues: clamp to a small positive value so
        # the square root is always valid.  A near-zero or negative radicand
        # means the moment-correction is degenerate; return 0.0 in that case.
        if radicand <= 0.0:
            return 0.0

        sr_star_den = math.sqrt(radicand)

        if sr_star_den <= 0.0:
            return 0.0

        z = sr_star_num / sr_star_den

        # Convert to probability via the normal CDF.
        dsr = _normal_cdf(z)
        return float(np.clip(dsr, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Plateau test
    # ------------------------------------------------------------------

    def plateau_test(
        self, study: optuna.Study, top_pct: float = 0.1
    ) -> tuple[bool, float]:
        """Check whether near-optimal parameters form a robust plateau.

        A "plateau" means that many parameter combinations in the top
        percentile produce similar performance.  A narrow performance spike
        (only the single best trial performs well) is a sign of overfitting.

        Parameters
        ----------
        study:
            Completed single-objective Optuna study.
        top_pct:
            Fraction of trials considered as "top performers".  Default: 0.1
            (top 10 %).

        Returns
        -------
        (pass_flag, cv)
            ``pass_flag`` is ``True`` when the coefficient of variation of
            top-trial scores is below ``_PLATEAU_CV_THRESHOLD``.
            ``cv`` is the raw coefficient of variation value.
        """
        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.value is not None
            and math.isfinite(t.value)
        ]

        if len(completed) < 5:
            logger.debug("Plateau test skipped — fewer than 5 completed trials.")
            return True, 0.0  # Cannot assess with so few trials; assume OK.

        values = sorted([t.value for t in completed], reverse=True)
        n_top = max(1, round(len(values) * top_pct))
        top_values = values[:n_top]

        mean_val = float(np.mean(top_values))
        std_val = float(np.std(top_values, ddof=1)) if len(top_values) > 1 else 0.0

        if mean_val == 0.0:
            return True, 0.0

        cv = std_val / abs(mean_val)
        passed = cv < _PLATEAU_CV_THRESHOLD

        logger.debug(
            "Plateau test — top %d trials: mean=%.4f, std=%.4f, CV=%.4f, pass=%s",
            n_top,
            mean_val,
            std_val,
            cv,
            passed,
        )
        return passed, cv

    # ------------------------------------------------------------------
    # Composite check
    # ------------------------------------------------------------------

    def check_all(
        self,
        study: optuna.Study,
        wf_result,
        n_trials: int,
    ) -> OverfitReport:
        """Run all three overfitting checks and return a consolidated report.

        Parameters
        ----------
        study:
            Completed single-objective Optuna study.
        wf_result:
            ``WFResult`` from ``WalkForwardAnalyzer.run()``.
        n_trials:
            Number of Optuna trials evaluated (used for DSR calculation).

        Returns
        -------
        OverfitReport
            Consolidated pass/fail flags and diagnostic values.
        """
        notes: List[str] = []

        # ------------------------------------------------------------------
        # 1. Deflated Sharpe Ratio
        # ------------------------------------------------------------------
        completed_values = [
            t.value
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
            and t.value is not None
            and math.isfinite(t.value)
            and t.value > 0
        ]

        dsr = 0.0
        if completed_values and len(completed_values) >= 2:
            best_sharpe = max(completed_values)
            var_sharpe = float(np.var(completed_values, ddof=1))

            # Approximate return distribution moments from the study's best trial.
            # Without raw return series we use conservative assumptions:
            # slight negative skew and mild excess kurtosis common in trading.
            skew = -0.5
            kurt = 1.0
            T = self._estimate_T(study)

            dsr = self.deflated_sharpe_ratio(
                sharpe=best_sharpe,
                n_trials=n_trials,
                var_sharpe=var_sharpe,
                skew=skew,
                kurtosis=kurt,
                T=T,
            )
            notes.append(
                f"DSR={dsr:.4f} (threshold={_DSR_THRESHOLD}) "
                f"— {'PASS' if dsr >= _DSR_THRESHOLD else 'FAIL'}"
            )
        else:
            notes.append("DSR could not be computed — insufficient completed trials.")

        dsr_pass = dsr >= _DSR_THRESHOLD

        # ------------------------------------------------------------------
        # 2. Walk-Forward Efficiency
        # ------------------------------------------------------------------
        wfe = getattr(wf_result, "wfe", 0.0)
        wfe_pass = wfe >= _WFE_THRESHOLD
        notes.append(
            f"WFE={wfe:.4f} (threshold={_WFE_THRESHOLD}) "
            f"— {'PASS' if wfe_pass else 'FAIL'}"
        )

        # ------------------------------------------------------------------
        # 3. Plateau test
        # ------------------------------------------------------------------
        plateau_pass, plateau_cv = self.plateau_test(study)
        notes.append(
            f"Plateau CV={plateau_cv:.4f} (threshold={_PLATEAU_CV_THRESHOLD}) "
            f"— {'PASS' if plateau_pass else 'FAIL'}"
        )

        overall_pass = dsr_pass and wfe_pass and plateau_pass

        return OverfitReport(
            dsr=dsr,
            dsr_pass=dsr_pass,
            wfe=wfe,
            wfe_pass=wfe_pass,
            plateau_pass=plateau_pass,
            plateau_cv=plateau_cv,
            overall_pass=overall_pass,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_T(study: optuna.Study) -> int:
        """Estimate the number of backtest observations from the study.

        Falls back to 252 (one trading year) when no trial metadata is
        available.
        """
        # Use the number of completed trials as a rough proxy when the
        # actual backtest bar count is not stored in trial user attributes.
        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if completed:
            # Try to read from user_attrs if stored by the objective.
            t = completed[0]
            T_val = t.user_attrs.get("n_observations")
            if T_val is not None:
                return int(T_val)
        return 252  # Conservative fallback: one year of daily returns.


# =============================================================================
# Statistical helpers (no external dependency on scipy)
# =============================================================================


def _inv_normal(p: float) -> float:
    """Rational approximation of the inverse normal CDF (Abramowitz & Stegun).

    Valid for p in (0, 1).  Maximum error ≈ 4.5 × 10⁻⁴.
    """
    if p <= 0.0:
        return -float("inf")
    if p >= 1.0:
        return float("inf")

    # Coefficients for rational approximation.
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]

    if p < 0.5:
        t = math.sqrt(-2.0 * math.log(p))
        num = c[0] + c[1] * t + c[2] * t ** 2
        den = 1.0 + d[0] * t + d[1] * t ** 2 + d[2] * t ** 3
        return -(t - num / den)
    else:
        t = math.sqrt(-2.0 * math.log(1.0 - p))
        num = c[0] + c[1] * t + c[2] * t ** 2
        den = 1.0 + d[0] * t + d[1] * t ** 2 + d[2] * t ** 3
        return t - num / den


def _normal_cdf(x: float) -> float:
    """Standard normal CDF using the complementary error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
