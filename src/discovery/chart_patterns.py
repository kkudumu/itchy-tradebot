"""Chart pattern detection from swing points.

Detects classical chart patterns (double top/bottom, head & shoulders,
triangles, wedges) from a sequence of SwingPoints produced by the
existing BreathingRoomDetector. Strategy-agnostic -- works with any
OHLCV data that produces swing points.

Pattern detection uses geometric relationships between consecutive swing
highs and lows, with tolerance thresholds scaled to ATR for volatility
adaptation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ChartPattern:
    """A detected chart pattern with metadata.

    Attributes
    ----------
    pattern_type:
        One of: double_top, double_bottom, head_and_shoulders,
        inverse_head_and_shoulders, ascending_triangle,
        descending_triangle, symmetrical_triangle,
        rising_wedge, falling_wedge.
    direction:
        'bullish' or 'bearish' -- the implied directional bias.
    confidence:
        0.0 to 1.0 -- how well the swing points match the ideal pattern.
    key_prices:
        List of significant price levels (e.g. peaks, neckline).
    start_index:
        Bar index where the pattern begins.
    end_index:
        Bar index where the pattern completes.
    description:
        Human-readable explanation.
    swing_indices:
        Bar indices of the swing points forming this pattern.
    """

    pattern_type: str
    direction: str
    confidence: float
    key_prices: List[float]
    start_index: int
    end_index: int
    description: str
    swing_indices: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "pattern_type": self.pattern_type,
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "key_prices": [round(p, 2) for p in self.key_prices],
            "start_index": self.start_index,
            "end_index": self.end_index,
            "description": self.description,
            "swing_indices": self.swing_indices,
        }


# Type alias for external SwingPoint
try:
    from src.strategy.strategies.sss.breathing_room import SwingPoint
except ImportError:
    # Fallback for standalone testing
    SwingPoint = Any  # type: ignore


class PatternDetector:
    """Detects classical chart patterns from a sequence of swing points.

    All distance thresholds are scaled to ATR for volatility adaptation.
    This makes the detector work across different market regimes.

    Parameters
    ----------
    atr:
        Current Average True Range for the instrument/timeframe.
    tolerance_atr_mult:
        How many ATRs apart two peaks/troughs can be and still be
        considered "equal" for pattern matching. Default 1.5.
    min_pattern_bars:
        Minimum bar span for a pattern to be valid. Default 10.
    """

    def __init__(
        self,
        atr: float,
        tolerance_atr_mult: float = 1.5,
        min_pattern_bars: int = 10,
    ) -> None:
        self._atr = max(atr, 0.01)  # guard against zero
        self._tolerance = atr * tolerance_atr_mult
        self._min_bars = min_pattern_bars

    def _prices_equal(self, a: float, b: float) -> bool:
        """Check if two prices are within ATR tolerance."""
        return abs(a - b) <= self._tolerance

    def _confidence_from_diff(self, a: float, b: float) -> float:
        """Higher confidence when peaks/troughs are closer together."""
        diff = abs(a - b)
        if diff <= 0.001:
            return 1.0
        ratio = diff / self._tolerance
        return max(0.0, 1.0 - ratio * 0.5)

    # ------------------------------------------------------------------
    # Double Top
    # ------------------------------------------------------------------

    def detect_double_tops(self, swings: List) -> List[ChartPattern]:
        """Detect double top patterns from swing point sequence.

        A double top requires:
        1. Two swing highs at roughly equal price (within ATR tolerance)
        2. A swing low between them (the neckline)
        3. Minimum bar span

        Returns list of detected ChartPattern objects.
        """
        if len(swings) < 3:
            return []

        patterns: List[ChartPattern] = []
        highs = [(i, s) for i, s in enumerate(swings) if s.swing_type == "high"]

        for a_idx in range(len(highs) - 1):
            for b_idx in range(a_idx + 1, min(a_idx + 4, len(highs))):
                i_a, sw_a = highs[a_idx]
                i_b, sw_b = highs[b_idx]

                if not self._prices_equal(sw_a.price, sw_b.price):
                    continue

                bar_span = abs(sw_b.index - sw_a.index)
                if bar_span < self._min_bars:
                    continue

                # Find the lowest low between the two highs
                lows_between = [
                    s for s in swings[i_a + 1:i_b]
                    if s.swing_type == "low"
                ]
                if not lows_between:
                    continue

                neckline_swing = min(lows_between, key=lambda s: s.price)
                neckline = neckline_swing.price

                # The pullback between peaks must be meaningful (> 0.5 ATR)
                pullback = max(sw_a.price, sw_b.price) - neckline
                if pullback < self._atr * 0.5:
                    continue

                conf = self._confidence_from_diff(sw_a.price, sw_b.price)

                patterns.append(ChartPattern(
                    pattern_type="double_top",
                    direction="bearish",
                    confidence=conf,
                    key_prices=[sw_a.price, neckline, sw_b.price],
                    start_index=sw_a.index,
                    end_index=sw_b.index,
                    description=(
                        f"Double top at {sw_a.price:.1f}/{sw_b.price:.1f}, "
                        f"neckline {neckline:.1f}"
                    ),
                    swing_indices=[sw_a.index, neckline_swing.index, sw_b.index],
                ))

        return patterns

    # ------------------------------------------------------------------
    # Double Bottom
    # ------------------------------------------------------------------

    def detect_double_bottoms(self, swings: List) -> List[ChartPattern]:
        """Detect double bottom patterns from swing point sequence.

        Mirror of double top: two roughly equal lows with a high between.
        """
        if len(swings) < 3:
            return []

        patterns: List[ChartPattern] = []
        lows = [(i, s) for i, s in enumerate(swings) if s.swing_type == "low"]

        for a_idx in range(len(lows) - 1):
            for b_idx in range(a_idx + 1, min(a_idx + 4, len(lows))):
                i_a, sw_a = lows[a_idx]
                i_b, sw_b = lows[b_idx]

                if not self._prices_equal(sw_a.price, sw_b.price):
                    continue

                bar_span = abs(sw_b.index - sw_a.index)
                if bar_span < self._min_bars:
                    continue

                highs_between = [
                    s for s in swings[i_a + 1:i_b]
                    if s.swing_type == "high"
                ]
                if not highs_between:
                    continue

                neckline_swing = max(highs_between, key=lambda s: s.price)
                neckline = neckline_swing.price

                pullback = neckline - min(sw_a.price, sw_b.price)
                if pullback < self._atr * 0.5:
                    continue

                conf = self._confidence_from_diff(sw_a.price, sw_b.price)

                patterns.append(ChartPattern(
                    pattern_type="double_bottom",
                    direction="bullish",
                    confidence=conf,
                    key_prices=[sw_a.price, neckline, sw_b.price],
                    start_index=sw_a.index,
                    end_index=sw_b.index,
                    description=(
                        f"Double bottom at {sw_a.price:.1f}/{sw_b.price:.1f}, "
                        f"neckline {neckline:.1f}"
                    ),
                    swing_indices=[sw_a.index, neckline_swing.index, sw_b.index],
                ))

        return patterns
