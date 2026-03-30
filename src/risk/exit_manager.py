"""Hybrid 50/50 exit strategy with Kijun-Sen trailing.

Exit logic (HARD-CODED, immutable):
- Phase 1: Close 50% of the position at 2R (fixed take-profit).
- Phase 2: Trail the remaining 50% using the Kijun-Sen level.

Trailing rules:
- NO breakeven move before 1R — gold retraces 0.3–0.7R within valid trends.
  Moving to BE prematurely gets stop-hunted. This rule is critical.
- Kijun trail begins at 1.5R (signal timeframe Kijun-Sen).
- Higher-TF Kijun trail activates at 3R+ to give room on extended moves.

All R-multiple thresholds are HARD-CODED — the learning loop cannot override them.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ActiveTrade:
    """State of an open trade."""

    entry_price: float
    """Price at which the position was opened."""

    stop_loss: float
    """Current stop-loss level (may be updated by trailing)."""

    take_profit: float
    """Fixed take-profit level for the first 50% partial exit."""

    direction: str
    """Trade direction: 'long' or 'short'."""

    lot_size: float
    """Original lot size at entry."""

    entry_time: datetime.datetime
    """Timestamp when the trade was opened."""

    remaining_pct: float = 1.0
    """Fraction of the original position still open (1.0 = full, 0.5 = half closed)."""

    current_r: float = 0.0
    """Last calculated R-multiple for this trade."""

    partial_exits: List[dict] = field(default_factory=list)
    """Log of partial exit events, each as a dict with price, pct, reason."""

    original_stop_loss: float = field(init=False)
    """Stop-loss at entry — preserved for BE-trap prevention checks."""

    def __post_init__(self) -> None:
        self.original_stop_loss = self.stop_loss

    @property
    def initial_risk(self) -> float:
        """Initial risk in price units (entry to original stop)."""
        if self.direction == "long":
            return self.entry_price - self.original_stop_loss
        return self.original_stop_loss - self.entry_price

    @property
    def is_partial(self) -> bool:
        """True if the first partial exit has already been executed."""
        return self.remaining_pct < 1.0


@dataclass
class ExitDecision:
    """Decision returned by the exit manager for a given price update."""

    action: str
    """One of: 'no_action', 'partial_exit', 'trail_update', 'full_exit'."""

    close_pct: float
    """Fraction of the open position to close (0.0, 0.5, or 1.0)."""

    new_stop: Optional[float]
    """Updated stop-loss level, or None if unchanged."""

    reason: str
    """Human-readable explanation of the decision."""

    r_multiple: float
    """Current R-multiple at the time of the decision."""


class HybridExitManager:
    """Hybrid 50/50 exit with Kijun-Sen trailing stop.

    Parameters
    ----------
    tp_r_multiple:
        R-multiple at which the first 50% is closed. Default: 2.0R.
    breakeven_threshold_r:
        R-multiple below which stop is NEVER moved to breakeven. Default: 1.0R.
        This is a critical anti-trap rule for gold.
    kijun_trail_start_r:
        R-multiple at which Kijun-Sen trailing begins. Default: 1.5R.
    higher_tf_kijun_start_r:
        R-multiple at which the higher-TF Kijun takes over trailing. Default: 3.0R.
    """

    # ------------------------------------------------------------------ #
    # Hard-coded R-multiple thresholds — immutable                         #
    # ------------------------------------------------------------------ #
    _ABSOLUTE_TP_R_MAX: float = 5.0    # partial exit cannot be delayed beyond this
    _ABSOLUTE_KIJUN_START_MIN: float = 1.0  # Kijun trail cannot begin before 1R

    def __init__(
        self,
        tp_r_multiple: float = 2.0,
        breakeven_threshold_r: float = 1.0,
        kijun_trail_start_r: float = 1.5,
        higher_tf_kijun_start_r: float = 3.0,
    ) -> None:
        # Enforce minimum safety constraints
        self._tp_r_multiple = min(tp_r_multiple, self._ABSOLUTE_TP_R_MAX)
        self._breakeven_threshold_r = breakeven_threshold_r
        self._kijun_trail_start_r = max(kijun_trail_start_r, self._ABSOLUTE_KIJUN_START_MIN)
        self._higher_tf_kijun_start_r = higher_tf_kijun_start_r

    # ------------------------------------------------------------------ #
    # Core exit decision logic                                              #
    # ------------------------------------------------------------------ #

    def check_exit(
        self,
        trade: ActiveTrade,
        current_price: float,
        kijun_value: float,
        higher_tf_kijun: Optional[float] = None,
        bar_high: Optional[float] = None,
        bar_low: Optional[float] = None,
    ) -> ExitDecision:
        """Evaluate the current bar and return an exit decision.

        When bar_high/bar_low are provided, SL is checked against the
        worst intrabar price (low for longs, high for shorts) and TP is
        checked against the best intrabar price. This prevents the
        close-only bias that misses stops and targets hit within the bar.

        Decision priority (evaluated in order):
        1. Full exit — stop hit by worst intrabar price.
        2. Partial exit — TP hit by best intrabar price.
        3. Trail update — Kijun level has moved favourably.
        4. No action — nothing to do.
        """
        # Use intrabar extremes when available
        if trade.direction == "long":
            worst_price = bar_low if bar_low is not None else current_price
            best_price = bar_high if bar_high is not None else current_price
        else:
            worst_price = bar_high if bar_high is not None else current_price
            best_price = bar_low if bar_low is not None else current_price

        r = self.calculate_r_multiple(
            trade.entry_price, current_price, trade.stop_loss, trade.direction
        )
        trade.current_r = r

        # --- Check if the trailing stop has been hit (use worst price) ---
        if self._stop_hit(trade, worst_price):
            # Exit at the stop level, not at the worst price
            stop_r = self.calculate_r_multiple(
                trade.entry_price, trade.stop_loss, trade.original_stop_loss, trade.direction
            )
            return ExitDecision(
                action="full_exit",
                close_pct=trade.remaining_pct,
                new_stop=None,
                reason=f"Stop hit at {trade.stop_loss:.5f} (R={stop_r:.2f})",
                r_multiple=stop_r,
            )

        # --- Partial exit: check TP against best intrabar price ---
        best_r = self.calculate_r_multiple(
            trade.entry_price, best_price, trade.stop_loss, trade.direction
        )
        if not trade.is_partial and best_r >= self._tp_r_multiple:
            return ExitDecision(
                action="partial_exit",
                close_pct=0.5,
                new_stop=trade.stop_loss,  # stop unchanged at partial exit
                reason=f"2R target reached (R={r:.2f}), closing 50%",
                r_multiple=r,
            )

        # --- Trail update (only after first partial exit) ---
        if trade.is_partial:
            new_stop = self.get_trailing_stop(trade, kijun_value, higher_tf_kijun)
            if new_stop is not None and self._stop_improves(trade, new_stop):
                return ExitDecision(
                    action="trail_update",
                    close_pct=0.0,
                    new_stop=new_stop,
                    reason=self._trail_reason(r, new_stop, higher_tf_kijun),
                    r_multiple=r,
                )

        return ExitDecision(
            action="no_action",
            close_pct=0.0,
            new_stop=None,
            reason=f"Holding (R={r:.2f})",
            r_multiple=r,
        )

    # ------------------------------------------------------------------ #
    # R-multiple calculation                                                #
    # ------------------------------------------------------------------ #

    def calculate_r_multiple(
        self,
        entry_price: float,
        current_price: float,
        stop_loss: float,
        direction: str,
    ) -> float:
        """Calculate the current R-multiple relative to the original risk.

        For longs:  R = (current_price - entry_price) / (entry_price - stop_loss)
        For shorts: R = (entry_price - current_price) / (stop_loss - entry_price)

        A negative R-multiple means the trade is currently in drawdown.
        """
        if direction == "long":
            risk = entry_price - stop_loss
        else:
            risk = stop_loss - entry_price

        if risk <= 0:
            # Degenerate case: stop is beyond entry (already in loss territory)
            return 0.0

        if direction == "long":
            return (current_price - entry_price) / risk
        else:
            return (entry_price - current_price) / risk

    # ------------------------------------------------------------------ #
    # Trailing stop logic                                                   #
    # ------------------------------------------------------------------ #

    def get_trailing_stop(
        self,
        trade: ActiveTrade,
        kijun_value: float,
        higher_tf_kijun: Optional[float] = None,
    ) -> Optional[float]:
        """Return the trailing stop level based on the current R-multiple.

        Trailing schedule (HARD-CODED):
        - R < 1.0R  : No change (original stop preserved; NO breakeven moves).
        - 1.0R–1.5R : No change (still below Kijun trail activation).
        - 1.5R–3.0R : Trail to signal-TF Kijun-Sen.
        - 3.0R+     : Trail to higher-TF Kijun-Sen (if provided; otherwise signal-TF).

        Returns None if no trail update is warranted.
        """
        r = trade.current_r

        # Below Kijun trail activation — leave stop alone
        if r < self._kijun_trail_start_r:
            return None

        # At 3R+ switch to higher-TF Kijun if available
        if r >= self._higher_tf_kijun_start_r and higher_tf_kijun is not None:
            return higher_tf_kijun

        # 1.5R–3.0R: use signal-TF Kijun
        return kijun_value

    # ------------------------------------------------------------------ #
    # Private helpers                                                       #
    # ------------------------------------------------------------------ #

    def _stop_hit(self, trade: ActiveTrade, current_price: float) -> bool:
        """Return True if the current price has reached or passed the stop-loss."""
        if trade.direction == "long":
            return current_price <= trade.stop_loss
        return current_price >= trade.stop_loss

    def _stop_improves(self, trade: ActiveTrade, new_stop: float) -> bool:
        """Return True if the proposed stop level moves favourably (tighter risk).

        For longs, the stop must move UP (closer to price).
        For shorts, the stop must move DOWN (closer to price).

        Critically: this prevents the manager from accidentally moving the stop
        back to a worse level if the Kijun retraces.
        """
        if trade.direction == "long":
            return new_stop > trade.stop_loss
        return new_stop < trade.stop_loss

    def _trail_reason(
        self, r: float, new_stop: float, higher_tf_kijun: Optional[float]
    ) -> str:
        using_higher = (
            higher_tf_kijun is not None
            and abs(new_stop - higher_tf_kijun) < 1e-8
        )
        source = "higher-TF Kijun" if using_higher else "signal-TF Kijun"
        return f"Trailing stop updated to {new_stop:.5f} via {source} (R={r:.2f})"
