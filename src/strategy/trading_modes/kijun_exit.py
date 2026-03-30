"""Kijun-Sen trailing exit mode — exact replica of HybridExitManager logic.

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

from typing import Optional

from src.strategy.base import TradingMode, ExitDecision, EvalMatrix


class KijunExitMode(TradingMode):
    """Kijun-Sen trailing exit mode — replica of HybridExitManager.

    Hard-coded R-multiple thresholds (immutable):
    - 50% partial exit at 2R.
    - Kijun trail begins at 1.5R (signal-TF Kijun-Sen).
    - Higher-TF Kijun trail activates at 3R+.
    - NO breakeven move before 1R.
    """

    # ------------------------------------------------------------------ #
    # Hard-coded R-multiple thresholds — immutable                         #
    # ------------------------------------------------------------------ #
    _TP_R_MULTIPLE: float = 2.0
    _KIJUN_TRAIL_START_R: float = 1.5
    _HIGHER_TF_KIJUN_START_R: float = 3.0
    _BREAKEVEN_THRESHOLD_R: float = 1.0

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
            Dict with at minimum ``'close'`` (current price).
        eval_results:
            EvalMatrix snapshot; kijun values are read from:
            - Signal-TF kijun: ichimoku_1H or ichimoku_15M metadata.
            - Higher-TF kijun: ichimoku_4H metadata.

        Returns
        -------
        ExitDecision
        """
        current_price: float = current_data['close']

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
        # 4. Trail update (only after first partial exit)                  #
        # ---------------------------------------------------------------- #
        if trade.is_partial:
            kijun_value, higher_tf_kijun = self._extract_kijun(eval_results)
            new_stop = self._get_trailing_stop(trade, r, kijun_value, higher_tf_kijun)
            if new_stop is not None and self._stop_improves(trade, new_stop):
                return ExitDecision(
                    action='trail_update',
                    close_pct=0.0,
                    new_stop=new_stop,
                    reason=self._trail_reason(r, new_stop, higher_tf_kijun),
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

    def _get_trailing_stop(
        self,
        trade: object,
        r: float,
        kijun_value: Optional[float],
        higher_tf_kijun: Optional[float],
    ) -> Optional[float]:
        """Return the trailing stop level based on current R-multiple.

        Trailing schedule (HARD-CODED):
        - R < 1.5R  : No change (original stop preserved).
        - 1.5R–3.0R : Trail to signal-TF Kijun-Sen.
        - 3.0R+     : Trail to higher-TF Kijun-Sen (if available; else signal-TF).

        Returns None if no trail update is warranted.
        """
        if r < self._KIJUN_TRAIL_START_R:
            return None

        if r >= self._HIGHER_TF_KIJUN_START_R and higher_tf_kijun is not None:
            return higher_tf_kijun

        return kijun_value

    def _extract_kijun(
        self,
        eval_results: EvalMatrix,
    ) -> tuple[Optional[float], Optional[float]]:
        """Extract signal-TF and higher-TF kijun values from EvalMatrix."""
        # Try to get signal-TF kijun (1H preferred, fall back to 15M)
        kijun_value: Optional[float] = None
        for tf in ['1H', '15M']:
            r = eval_results.get(f'ichimoku_{tf}')
            if r is not None and 'kijun' in r.metadata:
                kijun_value = r.metadata['kijun']
                break

        # Try to get higher-TF kijun (4H)
        higher_tf_kijun: Optional[float] = None
        r = eval_results.get('ichimoku_4H')
        if r is not None and 'kijun' in r.metadata:
            higher_tf_kijun = r.metadata['kijun']

        return kijun_value, higher_tf_kijun

    def _trail_reason(
        self,
        r: float,
        new_stop: float,
        higher_tf_kijun: Optional[float],
    ) -> str:
        using_higher = (
            higher_tf_kijun is not None
            and abs(new_stop - higher_tf_kijun) < 1e-8
        )
        source = 'higher-TF Kijun' if using_higher else 'signal-TF Kijun'
        return f"Trailing stop updated to {new_stop:.5f} via {source} (R={r:.2f})"
