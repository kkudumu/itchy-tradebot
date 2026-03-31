"""SequenceExitMode — trails stops behind breathing-room swing points for SSS trades.

No partial exits in v1. Stop is trailed behind confirmed swing points in the trade
direction. Initial stop is placed at the sequence invalidation price; if the distance
would be less than the min-stop floor, calculate_initial_stop returns None and the
trade is skipped (never widened).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from src.risk.exit_manager import ActiveTrade, ExitDecision

logger = logging.getLogger(__name__)


@dataclass
class SwingPoint:
    """Confirmed swing high or low (breathing room boundary).

    Canonical definition lives in breathing_room.py once that module merges.
    """

    index: int
    timestamp: datetime
    price: float
    swing_type: str          # 'high' or 'low'
    bar_count_since_prev: int


class SequenceExitMode:
    """Exit mode that trails stops behind confirmed swing points for SSS trades.

    Parameters
    ----------
    spread_multiplier:
        Stop distance must be >= spread * this value.
    min_stop_pips:
        Absolute minimum stop distance in pips.
    pip_value:
        Price units per pip (0.1 for XAUUSD).
    """

    def __init__(
        self,
        spread_multiplier: float = 2.0,
        min_stop_pips: float = 10.0,
        pip_value: float = 0.1,
    ) -> None:
        self._spread_multiplier = spread_multiplier
        self._min_stop_pips = min_stop_pips
        self._pip_value = pip_value

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def check_exit(
        self,
        trade: ActiveTrade,
        current_price: float,
        recent_swings: List[SwingPoint],
        spread: float = 0.0,
        bar_high: Optional[float] = None,
        bar_low: Optional[float] = None,
    ) -> ExitDecision:
        """Evaluate the current bar and return an exit decision.

        Priority: (1) full_exit if stop hit, (2) trail_update if new swing
        improves the stop, (3) no_action.

        recent_swings is expected newest-last (append-ordered from detector).
        bar_high/bar_low enable accurate intrabar stop-hit detection.
        """
        worst_price = self._worst_price(trade, current_price, bar_high, bar_low)
        r = self._calc_r(trade, current_price)
        trade.current_r = r

        if self._stop_hit(trade, worst_price):
            stop_r = self._calc_r(trade, trade.stop_loss)
            return ExitDecision(
                action="full_exit",
                close_pct=trade.remaining_pct,
                new_stop=None,
                reason=f"Stop hit at {trade.stop_loss:.5f} (R={stop_r:.2f})",
                r_multiple=stop_r,
            )

        new_stop = self._get_trail_stop(trade.direction, recent_swings)
        if new_stop is not None and self._stop_improves(trade, new_stop):
            trade.stop_loss = new_stop
            return ExitDecision(
                action="trail_update",
                close_pct=0.0,
                new_stop=new_stop,
                reason=f"Trailing stop moved to {new_stop:.5f} (R={r:.2f})",
                r_multiple=r,
            )

        return ExitDecision(
            action="no_action",
            close_pct=0.0,
            new_stop=None,
            reason=f"Holding (R={r:.2f})",
            r_multiple=r,
        )

    def calculate_initial_stop(
        self,
        direction: str,
        invalidation_price: float,
        spread: float,
        entry_price: float,
    ) -> Optional[float]:
        """Return the initial stop at the sequence invalidation price.

        Returns None if stop_distance < min_stop_distance — trade should be skipped,
        not widened.
        """
        if direction == "long":
            stop_distance = entry_price - invalidation_price
        else:
            stop_distance = invalidation_price - entry_price

        min_dist = self._min_stop_distance(spread)
        if stop_distance < min_dist:
            logger.debug(
                "Initial stop too tight: %.5f < %.5f — trade skipped",
                stop_distance,
                min_dist,
            )
            return None

        return invalidation_price

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _get_trail_stop(
        self, direction: str, recent_swings: List[SwingPoint]
    ) -> Optional[float]:
        """Most recent swing LOW for longs, swing HIGH for shorts (newest-last scan)."""
        target_type = "low" if direction == "long" else "high"
        for swing in reversed(recent_swings):
            if swing.swing_type == target_type:
                return swing.price
        return None

    def _min_stop_distance(self, spread: float) -> float:
        spread_floor = spread * self._spread_multiplier
        pip_floor = self._min_stop_pips * self._pip_value
        return max(spread_floor, pip_floor)

    def _worst_price(
        self,
        trade: ActiveTrade,
        current_price: float,
        bar_high: Optional[float],
        bar_low: Optional[float],
    ) -> float:
        if trade.direction == "long":
            return bar_low if bar_low is not None else current_price
        return bar_high if bar_high is not None else current_price

    def _stop_hit(self, trade: ActiveTrade, price: float) -> bool:
        if trade.direction == "long":
            return price <= trade.stop_loss
        return price >= trade.stop_loss

    def _stop_improves(self, trade: ActiveTrade, new_stop: float) -> bool:
        """Longs: stop moves UP. Shorts: stop moves DOWN."""
        if trade.direction == "long":
            return new_stop > trade.stop_loss
        return new_stop < trade.stop_loss

    def _calc_r(self, trade: ActiveTrade, price: float) -> float:
        """R-multiple of price vs entry, relative to initial risk."""
        initial_risk = trade.initial_risk
        if initial_risk <= 0:
            return 0.0
        if trade.direction == "long":
            return (price - trade.entry_price) / initial_risk
        return (trade.entry_price - price) / initial_risk
