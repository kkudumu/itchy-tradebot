"""Default ATR-based trailing exit mode — no Ichimoku dependency.

Exit logic (HARD-CODED, immutable):
- Phase 1: Close 50% of the position at 2R (fixed take-profit).
- Phase 2: Trail the remaining 50% using a fixed ATR multiple.

Trailing rules:
- NO breakeven move before 1R (same anti-trap rule as KijunExitMode).
- ATR trail begins at 1.5R.
- Trail distance: current_price ± 2 * ATR.

All R-multiple thresholds are HARD-CODED — the learning loop cannot override them.
"""

from __future__ import annotations

from typing import Optional

from src.strategy.base import TradingMode, ExitDecision, EvalMatrix


class DefaultExitMode(TradingMode):
    """Simplified ATR-based trailing exit mode.

    Hard-coded thresholds (immutable):
    - 50% partial exit at 2R.
    - ATR trail begins at 1.5R.
    - Trail distance: 2 * ATR.
    - NO breakeven move before 1R.
    """

    # ------------------------------------------------------------------ #
    # Hard-coded R-multiple thresholds — immutable                         #
    # ------------------------------------------------------------------ #
    _TP_R_MULTIPLE: float = 2.0
    _TRAIL_START_R: float = 1.5
    _ATR_MULTIPLIER: float = 2.0

    def check_exit(
        self,
        trade: object,
        current_data: dict,
        eval_results: EvalMatrix,
    ) -> ExitDecision:
        """Evaluate whether the open trade should be exited or modified.

        Parameters
        ----------
        trade:
            An ActiveTrade-compatible object with attributes: entry_price,
            stop_loss, direction, remaining_pct, current_r, is_partial,
            initial_risk, original_stop_loss.
        current_data:
            Dict with at minimum ``'close'`` (current price) and optionally
            ``'atr'`` (Average True Range for trailing distance).
        eval_results:
            EvalMatrix snapshot (not used by this mode; accepted for
            interface compatibility).

        Returns
        -------
        ExitDecision
        """
        current_price: float = current_data['close']
        atr: float = current_data.get('atr', 0.0)

        # ---------------------------------------------------------------- #
        # 1. Calculate R-multiple against original stop                     #
        # ---------------------------------------------------------------- #
        r = self._calculate_r(trade, current_price)
        trade.current_r = r

        # ---------------------------------------------------------------- #
        # 2. Check if stop has been hit                                     #
        # ---------------------------------------------------------------- #
        if self._stop_hit(trade, current_price):
            return ExitDecision(
                action='full_exit',
                close_pct=trade.remaining_pct,
                new_stop=None,
                reason=f"Stop hit at {trade.stop_loss:.5f} (R={r:.2f})",
            )

        # ---------------------------------------------------------------- #
        # 3. Partial exit at 2R (first hit only)                           #
        # ---------------------------------------------------------------- #
        if not trade.is_partial and r >= self._TP_R_MULTIPLE:
            return ExitDecision(
                action='partial_exit',
                close_pct=0.5,
                new_stop=trade.stop_loss,
                reason=f"2R target reached (R={r:.2f}), closing 50%",
            )

        # ---------------------------------------------------------------- #
        # 4. ATR trail (only after first partial exit and R >= 1.5)        #
        # ---------------------------------------------------------------- #
        if trade.is_partial and r >= self._TRAIL_START_R:
            new_stop = self._get_atr_trail(trade, current_price, atr)
            if new_stop is not None and self._stop_improves(trade, new_stop):
                return ExitDecision(
                    action='trail_update',
                    close_pct=0.0,
                    new_stop=new_stop,
                    reason=(
                        f"ATR trail updated to {new_stop:.5f} "
                        f"({self._ATR_MULTIPLIER}xATR={atr:.5f}, R={r:.2f})"
                    ),
                )

        # ---------------------------------------------------------------- #
        # 5. No action                                                      #
        # ---------------------------------------------------------------- #
        return ExitDecision(
            action='hold',
            close_pct=0.0,
            new_stop=None,
            reason=f"Holding (R={r:.2f})",
        )

    # ------------------------------------------------------------------ #
    # Private helpers                                                       #
    # ------------------------------------------------------------------ #

    def _calculate_r(self, trade: object, current_price: float) -> float:
        """Calculate current R-multiple relative to original stop."""
        if trade.direction == 'long':
            risk = trade.entry_price - trade.original_stop_loss
        else:
            risk = trade.original_stop_loss - trade.entry_price

        if risk <= 0:
            return 0.0

        if trade.direction == 'long':
            return (current_price - trade.entry_price) / risk
        else:
            return (trade.entry_price - current_price) / risk

    def _stop_hit(self, trade: object, current_price: float) -> bool:
        """Return True if current price has reached or passed the stop-loss."""
        if trade.direction == 'long':
            return current_price <= trade.stop_loss
        return current_price >= trade.stop_loss

    def _stop_improves(self, trade: object, new_stop: float) -> bool:
        """Return True if the proposed stop moves favourably (tighter risk).

        Long: stop must move UP (closer to price).
        Short: stop must move DOWN (closer to price).
        """
        if trade.direction == 'long':
            return new_stop > trade.stop_loss
        return new_stop < trade.stop_loss

    def _get_atr_trail(
        self,
        trade: object,
        current_price: float,
        atr: float,
    ) -> Optional[float]:
        """Compute the ATR-based trailing stop level.

        Long:  current_price - ATR_MULTIPLIER * atr
        Short: current_price + ATR_MULTIPLIER * atr

        Returns None if ATR is zero or negative.
        """
        if atr <= 0:
            return None

        trail_distance = self._ATR_MULTIPLIER * atr
        if trade.direction == 'long':
            return current_price - trail_distance
        else:
            return current_price + trail_distance
