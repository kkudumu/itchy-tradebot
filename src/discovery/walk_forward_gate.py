"""Walk-forward validation gate for discovered edges.

New edges must improve metrics on 2+ out-of-sample windows before being
absorbed into config/edges.yaml. Reverts if degradation exceeds threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationVerdict:
    """Result of walk-forward validation for a candidate edge."""

    passed: bool
    """Whether the edge passed the validation gate."""

    windows_improved: int
    """Number of OOS windows where the edge improved pass rate."""

    windows_degraded: int
    """Number of OOS windows where the edge degraded pass rate."""

    degraded: bool
    """True if any window showed degradation beyond the threshold."""

    avg_improvement_pct: float
    """Average pass rate improvement across all OOS windows (percentage points)."""

    summary: str
    """Human-readable summary of the validation result."""

    per_window_results: List[Dict[str, Any]] = field(default_factory=list)
    """Detailed results per OOS window."""


class WalkForwardGate:
    """Validate discovered edges against out-of-sample windows.

    Parameters
    ----------
    min_oos_windows:
        Minimum number of OOS windows where the edge must show improvement.
    min_improvement_pct:
        Minimum pass rate improvement in percentage points for a window
        to count as "improved" (e.g., 1.0 means +1pp).
    max_degradation_pct:
        Maximum tolerable pass rate drop in any single OOS window
        (percentage points). Exceeding this triggers degraded=True.
    """

    def __init__(
        self,
        min_oos_windows: int = 2,
        min_improvement_pct: float = 1.0,
        max_degradation_pct: float = 2.0,
    ) -> None:
        self._min_oos = min_oos_windows
        self._min_improvement = min_improvement_pct
        self._max_degradation = max_degradation_pct

    def evaluate(self, oos_results: List[Dict[str, Any]]) -> ValidationVerdict:
        """Evaluate an edge candidate against OOS window results.

        Parameters
        ----------
        oos_results:
            List of dicts, each with at minimum:
                - window_id: str
                - pass_rate_before: float (0-1)
                - pass_rate_after: float (0-1)
            Optional additional metrics (win_rate_*, sharpe_*) are
            recorded but only pass_rate is used for the gate decision.

        Returns
        -------
        ValidationVerdict with pass/fail and detailed breakdown.
        """
        if not oos_results:
            return ValidationVerdict(
                passed=False,
                windows_improved=0,
                windows_degraded=0,
                degraded=False,
                avg_improvement_pct=0.0,
                summary="No OOS results provided -- cannot validate.",
                per_window_results=[],
            )

        improved = 0
        degraded_count = 0
        any_degraded = False
        improvements: List[float] = []
        per_window: List[Dict[str, Any]] = []

        for r in oos_results:
            before = r["pass_rate_before"]
            after = r["pass_rate_after"]
            delta_pct = (after - before) * 100.0  # percentage points

            window_result = {
                "window_id": r["window_id"],
                "pass_rate_before": before,
                "pass_rate_after": after,
                "delta_pct": round(delta_pct, 2),
            }

            if delta_pct >= self._min_improvement:
                improved += 1
                window_result["verdict"] = "improved"
            elif delta_pct <= -self._max_degradation:
                degraded_count += 1
                any_degraded = True
                window_result["verdict"] = "degraded"
            else:
                window_result["verdict"] = "neutral"

            improvements.append(delta_pct)
            per_window.append(window_result)

        avg_improvement = sum(improvements) / len(improvements) if improvements else 0.0
        passed = improved >= self._min_oos and not any_degraded

        summary_parts = [
            f"OOS validation: {improved}/{len(oos_results)} windows improved",
            f"(need {self._min_oos}),",
            f"avg improvement: {avg_improvement:+.1f}pp.",
        ]
        if any_degraded:
            summary_parts.append(
                f"DEGRADATION detected in {degraded_count} window(s) "
                f"(>{self._max_degradation}pp drop)."
            )
        if passed:
            summary_parts.append("PASSED -- edge approved for absorption.")
        else:
            summary_parts.append("FAILED -- edge rejected.")

        verdict = ValidationVerdict(
            passed=passed,
            windows_improved=improved,
            windows_degraded=degraded_count,
            degraded=any_degraded,
            avg_improvement_pct=round(avg_improvement, 2),
            summary=" ".join(summary_parts),
            per_window_results=per_window,
        )

        logger.info("Walk-forward gate: %s", verdict.summary)
        return verdict
