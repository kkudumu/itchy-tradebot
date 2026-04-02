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

    # ------------------------------------------------------------------
    # Head and Shoulders
    # ------------------------------------------------------------------

    def detect_head_and_shoulders(self, swings: List) -> List[ChartPattern]:
        """Detect head & shoulders patterns.

        Requires 5 swing points: H-L-H-L-H where the middle H is highest
        and the two outer H's are roughly equal (within tolerance).
        The two lows form the neckline.
        """
        if len(swings) < 5:
            return []

        patterns: List[ChartPattern] = []
        highs = [(i, s) for i, s in enumerate(swings) if s.swing_type == "high"]

        for h_idx in range(len(highs) - 2):
            # Need 3 consecutive highs
            i_ls, ls = highs[h_idx]       # left shoulder
            i_hd, hd = highs[h_idx + 1]   # head
            i_rs, rs = highs[h_idx + 2]   # right shoulder

            # Head must be highest
            if hd.price <= ls.price or hd.price <= rs.price:
                continue

            # Shoulders must be roughly equal
            if not self._prices_equal(ls.price, rs.price):
                continue

            # Head must be meaningfully higher than shoulders (> 0.5 ATR)
            head_prominence = hd.price - max(ls.price, rs.price)
            if head_prominence < self._atr * 0.5:
                continue

            # Find neckline lows between shoulders and head
            lows_left = [s for s in swings[i_ls + 1:i_hd] if s.swing_type == "low"]
            lows_right = [s for s in swings[i_hd + 1:i_rs] if s.swing_type == "low"]

            if not lows_left or not lows_right:
                continue

            nl_left = min(lows_left, key=lambda s: s.price)
            nl_right = min(lows_right, key=lambda s: s.price)
            neckline = (nl_left.price + nl_right.price) / 2

            shoulder_conf = self._confidence_from_diff(ls.price, rs.price)
            neckline_conf = self._confidence_from_diff(nl_left.price, nl_right.price)
            conf = (shoulder_conf + neckline_conf) / 2

            patterns.append(ChartPattern(
                pattern_type="head_and_shoulders",
                direction="bearish",
                confidence=conf,
                key_prices=[ls.price, hd.price, rs.price, neckline],
                start_index=ls.index,
                end_index=rs.index,
                description=(
                    f"H&S: shoulders {ls.price:.1f}/{rs.price:.1f}, "
                    f"head {hd.price:.1f}, neckline {neckline:.1f}"
                ),
                swing_indices=[ls.index, nl_left.index, hd.index, nl_right.index, rs.index],
            ))

        return patterns

    # ------------------------------------------------------------------
    # Inverse Head and Shoulders
    # ------------------------------------------------------------------

    def detect_inverse_head_and_shoulders(self, swings: List) -> List[ChartPattern]:
        """Detect inverse head & shoulders patterns.

        Mirror of H&S: L-H-L-H-L where the middle L is lowest and
        the two outer L's are roughly equal.
        """
        if len(swings) < 5:
            return []

        patterns: List[ChartPattern] = []
        lows = [(i, s) for i, s in enumerate(swings) if s.swing_type == "low"]

        for l_idx in range(len(lows) - 2):
            i_ls, ls = lows[l_idx]         # left shoulder
            i_hd, hd = lows[l_idx + 1]     # head (lowest)
            i_rs, rs = lows[l_idx + 2]     # right shoulder

            # Head must be lowest
            if hd.price >= ls.price or hd.price >= rs.price:
                continue

            # Shoulders must be roughly equal
            if not self._prices_equal(ls.price, rs.price):
                continue

            # Head must be meaningfully lower than shoulders
            head_prominence = min(ls.price, rs.price) - hd.price
            if head_prominence < self._atr * 0.5:
                continue

            # Find neckline highs between shoulders and head
            highs_left = [s for s in swings[i_ls + 1:i_hd] if s.swing_type == "high"]
            highs_right = [s for s in swings[i_hd + 1:i_rs] if s.swing_type == "high"]

            if not highs_left or not highs_right:
                continue

            nl_left = max(highs_left, key=lambda s: s.price)
            nl_right = max(highs_right, key=lambda s: s.price)
            neckline = (nl_left.price + nl_right.price) / 2

            shoulder_conf = self._confidence_from_diff(ls.price, rs.price)
            neckline_conf = self._confidence_from_diff(nl_left.price, nl_right.price)
            conf = (shoulder_conf + neckline_conf) / 2

            patterns.append(ChartPattern(
                pattern_type="inverse_head_and_shoulders",
                direction="bullish",
                confidence=conf,
                key_prices=[ls.price, hd.price, rs.price, neckline],
                start_index=ls.index,
                end_index=rs.index,
                description=(
                    f"Inverse H&S: shoulders {ls.price:.1f}/{rs.price:.1f}, "
                    f"head {hd.price:.1f}, neckline {neckline:.1f}"
                ),
                swing_indices=[ls.index, nl_left.index, hd.index, nl_right.index, rs.index],
            ))

        return patterns

    # ------------------------------------------------------------------
    # Triangles
    # ------------------------------------------------------------------

    def detect_triangles(self, swings: List) -> List[ChartPattern]:
        """Detect ascending, descending, and symmetrical triangles.

        Requires at least 3 highs and 3 lows to establish trendlines.
        """
        if len(swings) < 6:
            return []

        highs = [s for s in swings if s.swing_type == "high"]
        lows = [s for s in swings if s.swing_type == "low"]

        if len(highs) < 3 or len(lows) < 3:
            return []

        patterns: List[ChartPattern] = []

        # Use last 3 highs and last 3 lows
        recent_highs = highs[-3:]
        recent_lows = lows[-3:]

        # Classify high slope: flat, falling, rising
        high_slope = self._classify_slope(
            [h.price for h in recent_highs],
            [h.index for h in recent_highs],
        )

        # Classify low slope: flat, falling, rising
        low_slope = self._classify_slope(
            [l.price for l in recent_lows],
            [l.index for l in recent_lows],
        )

        start_idx = min(recent_highs[0].index, recent_lows[0].index)
        end_idx = max(recent_highs[-1].index, recent_lows[-1].index)
        all_indices = [s.index for s in recent_highs + recent_lows]

        # Ascending triangle: flat highs + rising lows
        if high_slope == "flat" and low_slope == "rising":
            resistance = np.mean([h.price for h in recent_highs])
            conf = self._triangle_confidence(recent_highs, recent_lows)
            patterns.append(ChartPattern(
                pattern_type="ascending_triangle",
                direction="bullish",
                confidence=conf,
                key_prices=[h.price for h in recent_highs] + [l.price for l in recent_lows],
                start_index=start_idx,
                end_index=end_idx,
                description=f"Ascending triangle, resistance ~{resistance:.1f}",
                swing_indices=sorted(all_indices),
            ))

        # Descending triangle: falling highs + flat lows
        elif high_slope == "falling" and low_slope == "flat":
            support = np.mean([l.price for l in recent_lows])
            conf = self._triangle_confidence(recent_highs, recent_lows)
            patterns.append(ChartPattern(
                pattern_type="descending_triangle",
                direction="bearish",
                confidence=conf,
                key_prices=[h.price for h in recent_highs] + [l.price for l in recent_lows],
                start_index=start_idx,
                end_index=end_idx,
                description=f"Descending triangle, support ~{support:.1f}",
                swing_indices=sorted(all_indices),
            ))

        # Symmetrical triangle: falling highs + rising lows
        elif high_slope == "falling" and low_slope == "rising":
            conf = self._triangle_confidence(recent_highs, recent_lows)
            patterns.append(ChartPattern(
                pattern_type="symmetrical_triangle",
                direction="neutral",
                confidence=conf,
                key_prices=[h.price for h in recent_highs] + [l.price for l in recent_lows],
                start_index=start_idx,
                end_index=end_idx,
                description="Symmetrical triangle -- converging trendlines",
                swing_indices=sorted(all_indices),
            ))

        return patterns

    def _classify_slope(self, prices: List[float], indices: List[int]) -> str:
        """Classify a price sequence as flat, rising, or falling.

        'flat' = all prices within ATR tolerance of each other.
        'rising' = each price higher than the previous (within noise).
        'falling' = each price lower than the previous.
        """
        if len(prices) < 2:
            return "flat"

        # Check if all roughly equal (flat)
        price_range = max(prices) - min(prices)
        if price_range <= self._tolerance:
            return "flat"

        # Check monotonic direction
        diffs = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
        if all(d > -self._atr * 0.2 for d in diffs) and sum(d > 0 for d in diffs) >= len(diffs) * 0.5:
            return "rising"
        if all(d < self._atr * 0.2 for d in diffs) and sum(d < 0 for d in diffs) >= len(diffs) * 0.5:
            return "falling"

        return "mixed"

    def _triangle_confidence(self, highs: List, lows: List) -> float:
        """Calculate triangle pattern confidence based on convergence."""
        if len(highs) < 2 or len(lows) < 2:
            return 0.5

        # Range between highs and lows should be narrowing
        first_range = highs[0].price - lows[0].price
        last_range = highs[-1].price - lows[-1].price

        if first_range <= 0:
            return 0.3

        convergence_ratio = last_range / first_range
        # Perfect triangle converges; ratio < 1 is good
        if convergence_ratio < 0.3:
            return 0.9
        elif convergence_ratio < 0.6:
            return 0.75
        elif convergence_ratio < 0.85:
            return 0.6
        else:
            return 0.4

    # ------------------------------------------------------------------
    # Wedges
    # ------------------------------------------------------------------

    def detect_wedges(self, swings: List) -> List[ChartPattern]:
        """Detect rising and falling wedge patterns.

        Rising wedge: higher highs + higher lows, but range narrowing (bearish).
        Falling wedge: lower highs + lower lows, but range narrowing (bullish).
        """
        if len(swings) < 6:
            return []

        highs = [s for s in swings if s.swing_type == "high"]
        lows = [s for s in swings if s.swing_type == "low"]

        if len(highs) < 3 or len(lows) < 3:
            return []

        patterns: List[ChartPattern] = []
        recent_highs = highs[-3:]
        recent_lows = lows[-3:]

        high_slope = self._classify_slope(
            [h.price for h in recent_highs],
            [h.index for h in recent_highs],
        )
        low_slope = self._classify_slope(
            [l.price for l in recent_lows],
            [l.index for l in recent_lows],
        )

        # Check convergence: range narrowing
        first_range = recent_highs[0].price - recent_lows[0].price
        last_range = recent_highs[-1].price - recent_lows[-1].price
        is_converging = last_range < first_range * 0.85 if first_range > 0 else False

        start_idx = min(recent_highs[0].index, recent_lows[0].index)
        end_idx = max(recent_highs[-1].index, recent_lows[-1].index)
        all_indices = [s.index for s in recent_highs + recent_lows]

        # Rising wedge: both rising + converging
        if high_slope == "rising" and low_slope == "rising" and is_converging:
            conf = self._triangle_confidence(recent_highs, recent_lows)
            patterns.append(ChartPattern(
                pattern_type="rising_wedge",
                direction="bearish",
                confidence=conf,
                key_prices=[h.price for h in recent_highs] + [l.price for l in recent_lows],
                start_index=start_idx,
                end_index=end_idx,
                description="Rising wedge -- bearish reversal pattern",
                swing_indices=sorted(all_indices),
            ))

        # Falling wedge: both falling + converging
        elif high_slope == "falling" and low_slope == "falling" and is_converging:
            conf = self._triangle_confidence(recent_highs, recent_lows)
            patterns.append(ChartPattern(
                pattern_type="falling_wedge",
                direction="bullish",
                confidence=conf,
                key_prices=[h.price for h in recent_highs] + [l.price for l in recent_lows],
                start_index=start_idx,
                end_index=end_idx,
                description="Falling wedge -- bullish reversal pattern",
                swing_indices=sorted(all_indices),
            ))

        return patterns

    # ------------------------------------------------------------------
    # Unified scan
    # ------------------------------------------------------------------

    def detect_all(self, swings: List) -> List[ChartPattern]:
        """Run all pattern detectors and return combined results.

        Deduplicates overlapping patterns, keeping the highest confidence.
        """
        all_patterns: List[ChartPattern] = []
        all_patterns.extend(self.detect_double_tops(swings))
        all_patterns.extend(self.detect_double_bottoms(swings))
        all_patterns.extend(self.detect_head_and_shoulders(swings))
        all_patterns.extend(self.detect_inverse_head_and_shoulders(swings))
        all_patterns.extend(self.detect_triangles(swings))
        all_patterns.extend(self.detect_wedges(swings))

        # Sort by confidence descending
        all_patterns.sort(key=lambda p: p.confidence, reverse=True)

        # Deduplicate: if two patterns overlap by >50% of bar range, keep higher confidence
        unique: List[ChartPattern] = []
        for p in all_patterns:
            overlaps = False
            for q in unique:
                overlap_start = max(p.start_index, q.start_index)
                overlap_end = min(p.end_index, q.end_index)
                p_span = max(1, p.end_index - p.start_index)
                overlap_pct = max(0, overlap_end - overlap_start) / p_span
                if overlap_pct > 0.5 and p.direction == q.direction:
                    overlaps = True
                    break
            if not overlaps:
                unique.append(p)

        return unique
