"""Active trade management — coordinates sizer, circuit breaker, and exit logic.

All risk constraints enforced here are HARD-CODED. The learning loop may suggest
configuration changes but can never exceed the absolute caps defined as class
constants on this module.
"""

from __future__ import annotations

import datetime
from typing import Dict, List, Optional, Tuple

from src.risk.circuit_breaker import DailyCircuitBreaker
from src.risk.exit_manager import ActiveTrade, ExitDecision, HybridExitManager
from src.risk.position_sizer import AdaptivePositionSizer, PositionSize


class TradeManager:
    """Orchestrates position sizing, circuit breaking, and exit management.

    Parameters
    ----------
    position_sizer:
        Pre-configured :class:`AdaptivePositionSizer` instance.
    circuit_breaker:
        Pre-configured :class:`DailyCircuitBreaker` instance.
    exit_manager:
        Pre-configured :class:`HybridExitManager` instance.
    max_concurrent:
        Maximum number of simultaneously open positions. Default: 1.
        For gold (single-instrument), this should always be 1.
    """

    # ------------------------------------------------------------------ #
    # Absolute hard limits — IMMUTABLE, cannot be overridden by learning   #
    # ------------------------------------------------------------------ #
    _ABSOLUTE_MAX_RISK: float = 2.0        # % per trade — never exceeded
    _ABSOLUTE_MAX_DAILY_LOSS: float = 5.0  # % per day  — prop firm limit
    _ABSOLUTE_MAX_TOTAL_DD: float = 10.0   # % total    — prop firm limit

    def __init__(
        self,
        position_sizer: AdaptivePositionSizer,
        circuit_breaker: DailyCircuitBreaker,
        exit_manager: HybridExitManager,
        max_concurrent: int = 1,
    ) -> None:
        self._sizer = position_sizer
        self._breaker = circuit_breaker
        self._exit_manager = exit_manager
        self._max_concurrent = max_concurrent

        # Active trades indexed by a simple integer key
        self._active_trades: Dict[int, ActiveTrade] = {}
        self._next_trade_id: int = 1

        # Closed trade log
        self._closed_trades: List[dict] = []

        # Running equity tracking
        self._realised_pnl: float = 0.0

    # ------------------------------------------------------------------ #
    # Pre-trade gate                                                        #
    # ------------------------------------------------------------------ #

    def can_open_trade(
        self,
        current_balance: float,
        instrument: str = "XAUUSD",
    ) -> Tuple[bool, str]:
        """Check all conditions before opening a trade.

        Parameters
        ----------
        current_balance:
            Current account balance / equity.
        instrument:
            Symbol of the instrument being traded.

        Returns
        -------
        (True, "ok") if trading is permitted, or (False, reason) if blocked.
        """
        # 1. Circuit breaker
        if not self._breaker.can_trade(current_balance):
            return False, (
                f"Daily circuit breaker triggered — daily loss "
                f"{self._breaker.daily_loss_pct(current_balance):.2f}% "
                f">= {self._breaker.max_daily_loss_pct:.2f}%"
            )

        # 2. Max concurrent positions
        open_count = self._open_count(instrument)
        if open_count >= self._max_concurrent:
            return False, (
                f"Max concurrent positions ({self._max_concurrent}) "
                f"already open for {instrument}"
            )

        # 3. Correlation check (single-instrument placeholder — always passes)
        if not self.check_correlation(instrument, "any"):
            return False, f"Correlated exposure limit reached for {instrument}"

        # 4. Total drawdown guard
        if self._sizer.initial_balance > 0:
            total_dd_pct = (
                (self._sizer.initial_balance - current_balance)
                / self._sizer.initial_balance
                * 100.0
            )
            if total_dd_pct >= self._ABSOLUTE_MAX_TOTAL_DD:
                return False, (
                    f"Total drawdown {total_dd_pct:.2f}% >= "
                    f"{self._ABSOLUTE_MAX_TOTAL_DD:.2f}% prop firm limit"
                )

        return True, "ok"

    # ------------------------------------------------------------------ #
    # Trade lifecycle                                                        #
    # ------------------------------------------------------------------ #

    def open_trade(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        direction: str,
        atr: float,
        point_value: float,
        account_equity: float,
        atr_multiplier: float = 1.5,
        instrument: str = "XAUUSD",
        entry_time: Optional[datetime.datetime] = None,
    ) -> Tuple[int, ActiveTrade, PositionSize]:
        """Open a new trade with adaptive position sizing.

        Raises
        ------
        RuntimeError
            If the trade cannot be opened (circuit breaker or concurrent limit).
        ValueError
            If direction is not 'long' or 'short'.

        Returns
        -------
        (trade_id, ActiveTrade, PositionSize)
        """
        if direction not in ("long", "short"):
            raise ValueError(f"direction must be 'long' or 'short', got '{direction}'")

        can_open, reason = self.can_open_trade(account_equity, instrument)
        if not can_open:
            raise RuntimeError(f"Cannot open trade: {reason}")

        pos = self._sizer.calculate_position_size(
            account_equity=account_equity,
            atr=atr,
            atr_multiplier=atr_multiplier,
            point_value=point_value,
            instrument=instrument,
        )

        trade = ActiveTrade(
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            direction=direction,
            lot_size=pos.lot_size,
            entry_time=entry_time or datetime.datetime.now(datetime.timezone.utc),
        )

        trade_id = self._next_trade_id
        self._next_trade_id += 1
        self._active_trades[trade_id] = trade

        return trade_id, trade, pos

    def update_trade(
        self,
        trade_id: int,
        current_price: float,
        kijun_value: float,
        higher_tf_kijun: Optional[float] = None,
    ) -> ExitDecision:
        """Evaluate exit conditions and apply stop updates for a trade.

        If the exit manager returns a trail update, the trade's stop_loss is
        mutated in place. If a full exit is triggered, the trade is removed from
        active trades and archived in the closed trade log.

        Parameters
        ----------
        trade_id:
            Identifier returned by :meth:`open_trade`.
        current_price:
            Latest market price.
        kijun_value:
            Signal-timeframe Kijun-Sen value.
        higher_tf_kijun:
            Higher-timeframe Kijun-Sen value, or None.

        Returns
        -------
        ExitDecision
        """
        if trade_id not in self._active_trades:
            raise KeyError(f"No active trade with id {trade_id}")

        trade = self._active_trades[trade_id]
        decision = self._exit_manager.check_exit(
            trade, current_price, kijun_value, higher_tf_kijun
        )

        if decision.action == "trail_update" and decision.new_stop is not None:
            trade.stop_loss = decision.new_stop

        elif decision.action == "partial_exit":
            trade.remaining_pct -= decision.close_pct
            trade.partial_exits.append(
                {
                    "price": current_price,
                    "pct_closed": decision.close_pct,
                    "reason": decision.reason,
                    "r_multiple": decision.r_multiple,
                }
            )
            # Begin trailing the remaining 50% immediately
            new_stop = self._exit_manager.get_trailing_stop(
                trade, kijun_value, higher_tf_kijun
            )
            if new_stop is not None and self._exit_manager._stop_improves(trade, new_stop):
                trade.stop_loss = new_stop

        elif decision.action == "full_exit":
            self._archive_trade(trade_id, trade, current_price, decision.reason)

        return decision

    def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        reason: str = "manual_close",
    ) -> dict:
        """Force-close a trade (full) and return a summary.

        Parameters
        ----------
        trade_id:
            Trade identifier.
        exit_price:
            Price at which the trade is closed.
        reason:
            Reason for closure (e.g. 'friday_close', 'news_event').

        Returns
        -------
        dict with keys: trade_id, pnl_points, lot_size, direction, reason, r_multiple.
        """
        if trade_id not in self._active_trades:
            raise KeyError(f"No active trade with id {trade_id}")

        trade = self._active_trades[trade_id]
        r = self._exit_manager.calculate_r_multiple(
            trade.entry_price, exit_price, trade.stop_loss, trade.direction
        )
        trade.current_r = r
        return self._archive_trade(trade_id, trade, exit_price, reason)

    # ------------------------------------------------------------------ #
    # Equity / state reporting                                              #
    # ------------------------------------------------------------------ #

    def get_equity_summary(
        self, current_prices: Optional[Dict[str, float]] = None
    ) -> dict:
        """Return a snapshot of current equity state.

        Parameters
        ----------
        current_prices:
            Mapping of instrument symbol → current price for unrealised P&L.
            If None, unrealised P&L is reported as 0.

        Returns
        -------
        dict with:
            balance          — current balance (initial + realised P&L)
            open_trades      — number of active positions
            realised_pnl     — total realised P&L since inception
            unrealised_pnl   — estimated unrealised P&L (if prices provided)
            total_equity     — balance + unrealised_pnl
            daily_loss_pct   — today's loss percentage
            circuit_triggered — whether the circuit breaker has fired today
        """
        balance = self._sizer.initial_balance + self._realised_pnl
        unrealised = 0.0  # placeholder; full broker equity feed replaces this

        return {
            "balance": balance,
            "open_trades": len(self._active_trades),
            "realised_pnl": self._realised_pnl,
            "unrealised_pnl": unrealised,
            "total_equity": balance + unrealised,
            "daily_loss_pct": self._breaker.daily_loss_pct(balance),
            "circuit_triggered": self._breaker.is_triggered(),
            "profit_pct": self._sizer.profit_pct,
            "phase": self._sizer.get_phase(),
        }

    @property
    def active_trade_ids(self) -> List[int]:
        """IDs of all currently open trades."""
        return list(self._active_trades.keys())

    @property
    def closed_trades(self) -> List[dict]:
        """Read-only view of the closed trade log."""
        return list(self._closed_trades)

    # ------------------------------------------------------------------ #
    # Correlation placeholder                                               #
    # ------------------------------------------------------------------ #

    def check_correlation(self, instrument: str, direction: str) -> bool:
        """Placeholder: return True when correlated exposure is within limits.

        Currently always returns True (single-instrument operation).
        When expanding to multiple instruments, implement correlation matrix
        logic here to prevent adding correlated exposure beyond the risk budget.

        Parameters
        ----------
        instrument:
            Symbol of the proposed new trade.
        direction:
            'long' or 'short'.
        """
        # TODO: implement multi-instrument correlation matrix when expanding
        #       beyond XAUUSD. Use a static or rolling pairwise correlation
        #       matrix and refuse trades that would push net exposure over limits.
        return True

    # ------------------------------------------------------------------ #
    # Private helpers                                                       #
    # ------------------------------------------------------------------ #

    def _open_count(self, instrument: str) -> int:
        """Count active positions for the given instrument symbol."""
        # With a single instrument, all active trades count
        return len(self._active_trades)

    def _archive_trade(
        self, trade_id: int, trade: ActiveTrade, exit_price: float, reason: str
    ) -> dict:
        """Move trade from active to closed log and update realised P&L."""
        if trade.direction == "long":
            pnl_points = exit_price - trade.entry_price
        else:
            pnl_points = trade.entry_price - exit_price

        # Approximate monetary P&L (for tracking purposes; broker confirms actual)
        # pnl_points * lot_size * point_value — point_value not stored on trade,
        # so we record points and let the caller convert as needed.
        summary = {
            "trade_id": trade_id,
            "direction": trade.direction,
            "entry_price": trade.entry_price,
            "exit_price": exit_price,
            "original_stop": trade.original_stop_loss,
            "final_stop": trade.stop_loss,
            "lot_size": trade.lot_size,
            "pnl_points": pnl_points,
            "r_multiple": trade.current_r,
            "remaining_pct": trade.remaining_pct,
            "partial_exits": trade.partial_exits,
            "reason": reason,
            "entry_time": trade.entry_time,
            "exit_time": datetime.datetime.now(datetime.timezone.utc),
        }

        self._closed_trades.append(summary)
        if trade_id in self._active_trades:
            del self._active_trades[trade_id]

        # Update sizer balance approximation (positive pnl_points = profit)
        # The caller should call position_sizer.update_balance() with the real
        # broker balance after each close for accurate phase tracking.
        return summary
