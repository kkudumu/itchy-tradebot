"""Pattern detection hook for the backtest loop.

Provides a simple interface for the backtester to call on each bar
(or on each new swing point) to detect chart patterns and compute
confluence adjustments. Strategy-agnostic -- works with any strategy
that produces swing points.

Usage in the backtest loop:
    pattern_hook = PatternHook(atr=current_atr)
    # After swing detection:
    patterns = pattern_hook.detect(swing_history)
    adjustment = pattern_hook.get_confluence_adjustment(patterns, "long")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.discovery.chart_patterns import ChartPattern, PatternDetector
from src.discovery.pattern_confluence import PatternConfluenceScorer

logger = logging.getLogger(__name__)


class PatternHook:
    """Lightweight hook for wiring pattern detection into any backtest loop.

    Parameters
    ----------
    atr:
        Current ATR for the instrument. Updated via update_atr().
    tolerance_atr_mult:
        Tolerance multiplier for pattern detection. Default 1.5.
    min_swings_for_scan:
        Minimum swing points before attempting pattern detection. Default 4.
    """

    def __init__(
        self,
        atr: float = 5.0,
        tolerance_atr_mult: float = 1.5,
        min_swings_for_scan: int = 4,
    ) -> None:
        self._atr = max(atr, 0.01)
        self._tolerance_mult = tolerance_atr_mult
        self._min_swings = min_swings_for_scan
        self._detector = PatternDetector(
            atr=self._atr,
            tolerance_atr_mult=self._tolerance_mult,
        )
        self._scorer = PatternConfluenceScorer()
        self._last_patterns: List[ChartPattern] = []

    def update_atr(self, atr: float) -> None:
        """Update ATR and rebuild the detector with new tolerance."""
        self._atr = max(atr, 0.01)
        self._detector = PatternDetector(
            atr=self._atr,
            tolerance_atr_mult=self._tolerance_mult,
        )

    def detect(self, swings: List) -> List[ChartPattern]:
        """Run pattern detection on the current swing history.

        Parameters
        ----------
        swings: List of SwingPoint objects (from BreathingRoomDetector).

        Returns
        -------
        List of detected ChartPattern objects, sorted by confidence.
        """
        if len(swings) < self._min_swings:
            return []

        # Only scan the most recent 20 swings to avoid stale pattern matches
        recent = swings[-20:] if len(swings) > 20 else swings
        self._last_patterns = self._detector.detect_all(recent)
        return self._last_patterns

    def get_confluence_adjustment(
        self,
        patterns: Optional[List[ChartPattern]] = None,
        trade_direction: str = "long",
    ) -> int:
        """Get confluence score adjustment from detected patterns.

        Parameters
        ----------
        patterns: Patterns to score. Defaults to last detect() result.
        trade_direction: 'long' or 'short'.

        Returns
        -------
        Integer confluence adjustment (-3 to +3).
        """
        pts = patterns if patterns is not None else self._last_patterns
        return self._scorer.score(pts, trade_direction=trade_direction)

    @property
    def last_patterns(self) -> List[ChartPattern]:
        """Most recently detected patterns."""
        return self._last_patterns

    def patterns_as_dicts(self) -> List[Dict[str, Any]]:
        """Serialize last detected patterns to dicts for EdgeContext injection."""
        return [p.to_dict() for p in self._last_patterns]
