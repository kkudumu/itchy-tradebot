"""Go/no-go threshold checking with 25% haircut and Wilson score CI.

All backtest-derived metrics receive a conservative 25% haircut before being
compared against the required thresholds.  Structural counts (OOS trade count,
WFE, DSR, Monte Carlo pass rate) are exempt from the haircut because they are
not magnitude-based performance claims.

Wilson Score Confidence Interval
---------------------------------
The normal approximation (p ± z * sqrt(p*(1-p)/n)) breaks down for small
sample sizes and win rates near 0 or 1.  The Wilson score interval is more
accurate across the full range and is the recommended alternative in
statistical literature (Wilson 1927, Newcombe 1998).  We use the 95%
confidence level and check that the *lower* bound exceeds the 45% threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Result dataclasses
# =============================================================================


@dataclass
class ThresholdResult:
    """Pass/fail verdict for a single metric threshold.

    Attributes
    ----------
    name:
        Human-readable metric name.
    raw_value:
        Metric value before haircut.
    haircutted_value:
        Metric value after haircut (equals raw_value when haircut not applied).
    threshold:
        The required threshold value.
    passed:
        True when the haircutted value satisfies the threshold direction.
    margin:
        Signed distance from the threshold.  Positive means comfortably above
        (for min thresholds) or below (for max thresholds).
    direction:
        'min' → value must be >= threshold; 'max' → value must be <= threshold.
    haircut_applied:
        True when the 25% haircut was applied to this metric.
    """

    name: str
    raw_value: float
    haircutted_value: float
    threshold: float
    passed: bool
    margin: float
    direction: str = "min"
    haircut_applied: bool = False


@dataclass
class ValidationResult:
    """Aggregate pass/fail verdict across all thresholds.

    Attributes
    ----------
    overall_pass:
        True only when every individual threshold passes.
    results:
        Per-threshold results in check order.
    n_passed:
        Number of thresholds that passed.
    n_failed:
        Number of thresholds that failed.
    critical_failures:
        Names of thresholds that did not pass.
    """

    overall_pass: bool
    results: List[ThresholdResult]
    n_passed: int
    n_failed: int
    critical_failures: List[str] = field(default_factory=list)


# =============================================================================
# ThresholdChecker
# =============================================================================


class ThresholdChecker:
    """Check all go/no-go thresholds with an optional 25% haircut.

    The THRESHOLDS class variable defines the full check set.  Each entry
    contains:

    - ``min`` or ``max``: the required threshold value.
    - ``haircut``: whether the 25% haircut applies.
    - ``direction``: 'min' (value must be >= threshold) or 'max' (<=).

    Parameters
    ----------
    haircut_pct:
        Haircut percentage to apply to eligible metrics.  Default: 25.0.
    """

    THRESHOLDS: Dict[str, dict] = {
        "oos_trade_count": {
            "min": 400,
            "haircut": False,
            "direction": "min",
            "label": "OOS Trade Count",
        },
        "win_rate_ci_lower": {
            "min": 0.45,
            "haircut": True,
            "direction": "min",
            "label": "Win Rate CI Lower (95%)",
        },
        "profit_factor": {
            "min": 1.3,
            "haircut": True,
            "direction": "min",
            "label": "Profit Factor",
        },
        "sharpe_ratio": {
            "min": 1.0,
            "haircut": True,
            "direction": "min",
            "label": "Sharpe Ratio",
        },
        "sortino_ratio": {
            "min": 1.0,
            "haircut": True,
            "direction": "min",
            "label": "Sortino Ratio",
        },
        "max_drawdown_pct": {
            "max": 15.0,
            "haircut": True,
            "direction": "max",
            "label": "Max Drawdown %",
        },
        "wfe": {
            "min": 0.5,
            "haircut": False,
            "direction": "min",
            "label": "Walk-Forward Efficiency",
        },
        "dsr": {
            "min": 0.90,
            "haircut": False,
            "direction": "min",
            "label": "Deflated Sharpe Ratio",
        },
        "monte_carlo_pass_rate": {
            "min": 0.80,
            "haircut": False,
            "direction": "min",
            "label": "Monte Carlo Pass Rate",
        },
    }

    def __init__(self, haircut_pct: float = 25.0) -> None:
        if not (0.0 <= haircut_pct < 100.0):
            raise ValueError(
                f"haircut_pct must be in [0, 100), got {haircut_pct}"
            )
        self.haircut = haircut_pct / 100.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_haircut(self, value: float, direction: str = "min") -> float:
        """Apply the configured haircut to a metric value.

        For 'min' thresholds (higher is better) the haircut reduces the
        observed value:  value * (1 - haircut).

        For 'max' thresholds (lower is better, e.g. drawdown) the haircut
        increases the observed value, making the test more conservative:
        value * (1 + haircut).

        Parameters
        ----------
        value:
            Raw metric value.
        direction:
            'min' or 'max'.  Defaults to 'min'.

        Returns
        -------
        float
            Haircutted value.
        """
        if direction == "max":
            return value * (1.0 + self.haircut)
        return value * (1.0 - self.haircut)

    def check_metric(
        self,
        name: str,
        raw_value: float,
        spec: Optional[dict] = None,
    ) -> ThresholdResult:
        """Check a single metric against its threshold specification.

        Parameters
        ----------
        name:
            Metric key matching a THRESHOLDS entry (or provide *spec* directly).
        raw_value:
            Observed metric value before haircut.
        spec:
            Optional override threshold specification dict.  When None, the
            entry from THRESHOLDS[name] is used.

        Returns
        -------
        ThresholdResult
        """
        if spec is None:
            spec = self.THRESHOLDS.get(name, {})

        direction = spec.get("direction", "min")
        apply_hc = spec.get("haircut", False)
        label = spec.get("label", name)

        if apply_hc:
            haircutted = self.apply_haircut(raw_value, direction)
        else:
            haircutted = raw_value

        threshold = float(spec.get("min" if direction == "min" else "max", 0.0))

        if direction == "min":
            passed = haircutted >= threshold
            margin = haircutted - threshold
        else:
            passed = haircutted <= threshold
            margin = threshold - haircutted  # positive = comfortably below limit

        return ThresholdResult(
            name=label,
            raw_value=raw_value,
            haircutted_value=haircutted,
            threshold=threshold,
            passed=passed,
            margin=margin,
            direction=direction,
            haircut_applied=apply_hc,
        )

    def check_all(self, metrics: dict) -> ValidationResult:
        """Check all thresholds against the provided metrics dict.

        Expected keys in *metrics* (any missing key is treated as 0.0):

        - ``oos_trade_count``    – total OOS trades across all WF windows
        - ``win_rate_ci_lower``  – lower bound of the Wilson 95% CI
        - ``profit_factor``      – OOS profit factor
        - ``sharpe_ratio``       – OOS annualised Sharpe
        - ``sortino_ratio``      – OOS Sortino ratio
        - ``max_drawdown_pct``   – OOS peak-to-trough drawdown (positive %)
        - ``wfe``                – walk-forward efficiency
        - ``dsr``                – deflated Sharpe ratio (0–1)
        - ``monte_carlo_pass_rate`` – fraction of MC sims that passed (0–1)

        Parameters
        ----------
        metrics:
            Dict mapping metric keys to observed float values.

        Returns
        -------
        ValidationResult with per-threshold breakdown and overall verdict.
        """
        results: List[ThresholdResult] = []

        for key, spec in self.THRESHOLDS.items():
            raw = float(metrics.get(key) or 0.0)
            result = self.check_metric(key, raw, spec)
            results.append(result)

        n_passed = sum(1 for r in results if r.passed)
        n_failed = len(results) - n_passed
        critical_failures = [r.name for r in results if not r.passed]
        overall = n_failed == 0

        return ValidationResult(
            overall_pass=overall,
            results=results,
            n_passed=n_passed,
            n_failed=n_failed,
            critical_failures=critical_failures,
        )

    # ------------------------------------------------------------------
    # Wilson score confidence interval
    # ------------------------------------------------------------------

    @staticmethod
    def wilson_score_ci(
        wins: int,
        total: int,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """Wilson score confidence interval for a binomial proportion.

        More accurate than the normal approximation, particularly for small
        samples or extreme win rates.  Returns the lower and upper bounds of
        the CI for the true win probability.

        Parameters
        ----------
        wins:
            Number of winning trades.
        total:
            Total number of trades (wins + losses).
        confidence:
            Confidence level.  Default: 0.95 (95%).

        Returns
        -------
        (lower_bound, upper_bound)
            Both values are in [0, 1].

        References
        ----------
        Wilson, E.B. (1927). "Probable inference, the law of succession, and
        statistical inference." Journal of the American Statistical Association.
        """
        if total <= 0:
            return 0.0, 0.0

        # z for two-tailed confidence interval
        alpha = 1.0 - confidence
        z = _inv_normal(1.0 - alpha / 2.0)
        z2 = z * z

        p_hat = wins / total
        n = total

        # Wilson score formula
        centre = (p_hat + z2 / (2 * n)) / (1 + z2 / n)
        margin = (z / (1 + z2 / n)) * math.sqrt(
            p_hat * (1 - p_hat) / n + z2 / (4 * n * n)
        )

        lower = max(0.0, centre - margin)
        upper = min(1.0, centre + margin)
        return lower, upper


# =============================================================================
# Statistical helper (no scipy dependency)
# =============================================================================


def _inv_normal(p: float) -> float:
    """Rational approximation of the inverse normal CDF (Abramowitz & Stegun).

    Valid for p in (0, 1).  Maximum error ≈ 4.5e-4.
    """
    if p <= 0.0:
        return -float("inf")
    if p >= 1.0:
        return float("inf")

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
