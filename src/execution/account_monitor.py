"""Real-time account and position monitoring for circuit-breaker integration.

Provides equity, margin, and position data at the frequency required by the
risk management layer without duplicating MT5 connection logic.

The circuit breaker in :mod:`src.risk.circuit_breaker` calls
:meth:`AccountMonitor.get_account_info` on every bar to decide whether trading
should be halted for the day.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AccountInfo:
    """Snapshot of the MT5 account state."""

    balance: float
    """Settled account balance (closed trades only)."""

    equity: float
    """Balance plus unrealised P&L on all open positions."""

    margin: float
    """Margin currently in use by open positions."""

    free_margin: float
    """Equity minus used margin — the amount available for new positions."""

    unrealized_pnl: float
    """Floating P&L across all open positions (equity - balance)."""

    leverage: int
    """Account leverage (e.g. 100 for 1:100)."""


@dataclass
class PositionInfo:
    """State of a single open MT5 position."""

    ticket: int
    """MT5 position ticket."""

    instrument: str
    """Symbol string (e.g. 'XAUUSD')."""

    direction: str
    """'long' or 'short'."""

    volume: float
    """Current open volume in lots."""

    entry_price: float
    """Price at which the position was opened."""

    current_price: float
    """Current market price (bid for long, ask for short)."""

    stop_loss: float
    """Current stop-loss level (0.0 if not set)."""

    take_profit: float
    """Current take-profit level (0.0 if not set)."""

    profit: float
    """Floating P&L in account currency."""

    time: datetime
    """UTC timestamp when the position was opened."""


# ---------------------------------------------------------------------------
# AccountMonitor
# ---------------------------------------------------------------------------

class AccountMonitor:
    """Real-time account and position monitoring via MT5Bridge.

    Parameters
    ----------
    bridge:
        Connected :class:`~src.execution.mt5_bridge.MT5Bridge` instance.
    """

    def __init__(self, bridge) -> None:
        self.bridge = bridge

    # ------------------------------------------------------------------
    # Account state
    # ------------------------------------------------------------------

    def get_account_info(self) -> Optional[AccountInfo]:
        """Fetch current account equity, balance, and margin from MT5.

        Returns None if the terminal is not connected or returns invalid data.
        """
        mt5 = self.bridge.mt5
        if mt5 is None:
            logger.warning("get_account_info: MT5 not connected")
            return None

        try:
            account = mt5.account_info()
            if account is None:
                error = mt5.last_error()
                logger.error("account_info() returned None: %s", error)
                return None

            balance = float(account.balance)
            equity = float(account.equity)
            margin = float(account.margin)
            free_margin = float(account.margin_free)
            leverage = int(account.leverage)

            return AccountInfo(
                balance=balance,
                equity=equity,
                margin=margin,
                free_margin=free_margin,
                unrealized_pnl=equity - balance,
                leverage=leverage,
            )

        except Exception as exc:  # noqa: BLE001
            logger.exception("Error fetching account info: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Open positions
    # ------------------------------------------------------------------

    def get_positions(self, instrument: Optional[str] = None) -> List[PositionInfo]:
        """Get all open positions, optionally filtered by instrument.

        Parameters
        ----------
        instrument:
            If provided, only positions for this symbol are returned.

        Returns
        -------
        List of :class:`PositionInfo` (empty list on error or no positions).
        """
        mt5 = self.bridge.mt5
        if mt5 is None:
            return []

        try:
            if instrument:
                raw = mt5.positions_get(symbol=instrument)
            else:
                raw = mt5.positions_get()

            if raw is None:
                return []

            result: List[PositionInfo] = []
            for pos in raw:
                direction = "long" if int(pos.type) == mt5.ORDER_TYPE_BUY else "short"
                result.append(
                    PositionInfo(
                        ticket=int(pos.ticket),
                        instrument=str(pos.symbol),
                        direction=direction,
                        volume=float(pos.volume),
                        entry_price=float(pos.price_open),
                        current_price=float(pos.price_current),
                        stop_loss=float(pos.sl),
                        take_profit=float(pos.tp),
                        profit=float(pos.profit),
                        time=datetime.fromtimestamp(int(pos.time), tz=timezone.utc),
                    )
                )

            return result

        except Exception as exc:  # noqa: BLE001
            logger.exception("Error fetching positions: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Pending orders
    # ------------------------------------------------------------------

    def get_open_orders(self, instrument: Optional[str] = None) -> List[dict]:
        """Get all pending (not yet filled) orders.

        Parameters
        ----------
        instrument:
            If provided, only orders for this symbol are returned.

        Returns
        -------
        List of dicts with keys: ticket, symbol, type, volume, price, sl, tp, time.
        """
        mt5 = self.bridge.mt5
        if mt5 is None:
            return []

        try:
            if instrument:
                raw = mt5.orders_get(symbol=instrument)
            else:
                raw = mt5.orders_get()

            if raw is None:
                return []

            result = []
            for order in raw:
                result.append(
                    {
                        "ticket": int(order.ticket),
                        "symbol": str(order.symbol),
                        "type": int(order.type),
                        "volume": float(order.volume_current),
                        "price": float(order.price_open),
                        "sl": float(order.sl),
                        "tp": float(order.tp),
                        "time": datetime.fromtimestamp(int(order.time_setup), tz=timezone.utc),
                    }
                )

            return result

        except Exception as exc:  # noqa: BLE001
            logger.exception("Error fetching open orders: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Daily P&L
    # ------------------------------------------------------------------

    def get_daily_pnl(self) -> float:
        """Calculate today's realised P&L from the MT5 deal history.

        Queries deals from the start of today (UTC) to now.

        Returns
        -------
        Net P&L in account currency, or 0.0 on error.
        """
        mt5 = self.bridge.mt5
        if mt5 is None:
            return 0.0

        try:
            import time as _time

            today = date.today()
            start_dt = datetime(today.year, today.month, today.day, tzinfo=timezone.utc)
            end_dt = datetime.now(timezone.utc)

            # MT5 expects Unix timestamps
            start_ts = int(start_dt.timestamp())
            end_ts = int(end_dt.timestamp())

            deals = mt5.history_deals_get(start_ts, end_ts)
            if deals is None:
                return 0.0

            # Filter to trade deals only (entry/exit, not commission or deposit)
            # DEAL_ENTRY_OUT = 1 (closing deals carry the realised P&L)
            total_pnl = 0.0
            for deal in deals:
                # Include all deal types that carry P&L (avoid double-counting entries)
                profit = float(getattr(deal, "profit", 0.0))
                total_pnl += profit

            return total_pnl

        except Exception as exc:  # noqa: BLE001
            logger.exception("Error calculating daily P&L: %s", exc)
            return 0.0

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def equity(self) -> float:
        """Current account equity, or 0.0 on error.

        Convenience shortcut for circuit-breaker polling.
        """
        info = self.get_account_info()
        return info.equity if info is not None else 0.0

    @property
    def balance(self) -> float:
        """Current account balance, or 0.0 on error."""
        info = self.get_account_info()
        return info.balance if info is not None else 0.0
