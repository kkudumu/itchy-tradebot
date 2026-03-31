"""FiftyTapCalculator — 50% retracement entry confirmation for SSS.

Jay: "I drag my 50s from the low point which I do NOT want to see swept,
to the highest point low."  A tap of the 50% zone followed by a CBC
forms the highest-probability entry sequence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .breathing_room import SwingPoint

log = logging.getLogger(__name__)


@dataclass
class FiftyTapLevel:
    """One calculated 50% retracement level and its live state."""

    price: float              # The 50% level price
    direction: str            # 'bullish' or 'bearish'
    anchor_low: float         # Protected low (don't want swept) — bullish
    anchor_high: float        # Target high — bullish; protected high — bearish
    is_tapped: bool = False
    tap_bar_index: Optional[int] = None
    is_invalidated: bool = False


class FiftyTapCalculator:
    """Calculate and track 50% Fibonacci retracement levels for SSS entries.
    tap_level=0.5, tolerance_pips=0.5 (pips), pip_value=0.1 (XAUUSD).
    """

    def __init__(
        self,
        tap_level: float = 0.5,
        tolerance_pips: float = 0.5,
        pip_value: float = 0.1,
    ) -> None:
        self._tap_level = tap_level
        self._tolerance = tolerance_pips * pip_value
        self._pip_value = pip_value

    def calculate_level(
        self,
        direction: str,
        anchor_swing: SwingPoint,
        target_swing: SwingPoint,
    ) -> FiftyTapLevel:
        """Calculate the 50% level between two swing points.

        Bullish: anchor_swing=low (don't want swept), target_swing=highest swing low.
                 level = anchor_low + (target_low - anchor_low) * tap_level
        Bearish: mirror — anchor_swing=high, target_swing=lowest swing high.
                 level = anchor_high - (anchor_high - target_high) * tap_level
        """
        if direction == "bullish":
            anchor_low = anchor_swing.price
            anchor_high = target_swing.price
            price = anchor_low + (anchor_high - anchor_low) * self._tap_level
        elif direction == "bearish":
            anchor_high = anchor_swing.price
            anchor_low = target_swing.price
            price = anchor_high - (anchor_high - anchor_low) * self._tap_level
        else:
            raise ValueError(f"direction must be 'bullish' or 'bearish', got {direction!r}")

        log.debug("FiftyTap %s level=%.4f anchor_low=%.4f anchor_high=%.4f",
                  direction, price, anchor_low, anchor_high)
        return FiftyTapLevel(
            price=price,
            direction=direction,
            anchor_low=anchor_low,
            anchor_high=anchor_high,
        )

    def check_tap(
        self,
        level: FiftyTapLevel,
        bar_index: int,
        high: float,
        low: float,
    ) -> FiftyTapLevel:
        """Return updated level; sets is_tapped if bar touches the 50% zone.

        Bullish: tapped when bar low  <= level.price + tolerance
        Bearish: tapped when bar high >= level.price - tolerance
        """
        if level.is_tapped or level.is_invalidated:
            return level

        if level.direction == "bullish":
            tapped = low <= level.price + self._tolerance
        else:
            tapped = high >= level.price - self._tolerance

        if not tapped:
            return level

        log.debug("FiftyTap tapped bar=%d high=%.4f low=%.4f level=%.4f",
                  bar_index, high, low, level.price)
        return FiftyTapLevel(
            price=level.price,
            direction=level.direction,
            anchor_low=level.anchor_low,
            anchor_high=level.anchor_high,
            is_tapped=True,
            tap_bar_index=bar_index,
            is_invalidated=False,
        )

    def check_invalidation(
        self,
        level: FiftyTapLevel,
        high: float,
        low: float,
    ) -> FiftyTapLevel:
        """Return updated level; sets is_invalidated if anchor is swept.

        Bullish: invalidated when low  < anchor_low  (swept downward)
        Bearish: invalidated when high > anchor_high (swept upward)
        """
        if level.is_invalidated:
            return level

        if level.direction == "bullish":
            invalidated = low < level.anchor_low
        else:
            invalidated = high > level.anchor_high

        if not invalidated:
            return level

        log.debug("FiftyTap %s invalidated high=%.4f low=%.4f", level.direction, high, low)
        return FiftyTapLevel(
            price=level.price,
            direction=level.direction,
            anchor_low=level.anchor_low,
            anchor_high=level.anchor_high,
            is_tapped=level.is_tapped,
            tap_bar_index=level.tap_bar_index,
            is_invalidated=True,
        )

    def reset(self) -> None:
        """Clear any tracked state (calculator is stateless — no-op)."""
        log.debug("FiftyTapCalculator reset")
