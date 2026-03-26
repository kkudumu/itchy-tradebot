"""Order execution and management for MT5.

Handles all order types used by the Ichimoku strategy:
- Market orders for immediate entry
- Limit orders for Kijun pullback entries
- Stop-limit orders for cloud breakout entries
- Position modification (trailing stops)
- Full and partial position closure (hybrid 50/50 exit)

All interaction with MT5 goes through the MT5Bridge so tests can mock everything
without importing the MetaTrader5 package.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filling mode bit flags (mirrors MetaTrader5 ORDER_FILLING_* constants)
# ---------------------------------------------------------------------------

# These values match the mt5 filling_mode bitmask on symbol_info:
#   bit 0 → FOK supported
#   bit 1 → IOC supported
# When filling_mode == 0 the broker uses RETURN (exchange-mode) execution.
_FILLING_FOK    = 1   # ORDER_FILLING_FOK
_FILLING_IOC    = 2   # ORDER_FILLING_IOC
_FILLING_RETURN = 4   # ORDER_FILLING_RETURN (exchange execution)

# Common MT5 return codes
_RETCODE_DONE        = 10009
_RETCODE_MARKET_CLOSED = 10030  # market is closed
_RETCODE_REJECTED    = 10006   # request rejected by server
_RETCODE_NO_MONEY    = 10019   # not enough money
_RETCODE_PRICE_OFF   = 10021   # price has changed, retry
_RETCODE_REQUOTE     = 10004   # requote

# MT5 magic number identifying orders placed by this agent
_MAGIC_NUMBER = 234567


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class OrderResult:
    """Outcome of an order_send() request."""

    success: bool
    """True if the order was accepted and filled / pending."""

    ticket: int
    """Position or order ticket assigned by MT5. 0 on failure."""

    price: float
    """Actual fill price. 0.0 on failure."""

    volume: float
    """Filled or pending volume (lots)."""

    retcode: int
    """MT5 return code (10009 = DONE)."""

    comment: str
    """Server comment for the order."""

    slippage: float
    """Slippage in points relative to the requested price (positive = adverse)."""

    error_message: str = ""
    """Human-readable error description when success is False."""


# ---------------------------------------------------------------------------
# OrderManager
# ---------------------------------------------------------------------------

class OrderManager:
    """Execute and manage orders via MT5Bridge.

    Parameters
    ----------
    bridge:
        Connected :class:`~src.execution.mt5_bridge.MT5Bridge` instance.
    deviation:
        Maximum allowed slippage in points for market orders. Default: 20.
    """

    def __init__(self, bridge, deviation: int = 20) -> None:
        self.bridge = bridge
        self._deviation = deviation
        self._slippage_log: List[dict] = []

    # ------------------------------------------------------------------
    # Market orders
    # ------------------------------------------------------------------

    def market_order(
        self,
        instrument: str,
        direction: str,
        lot_size: float,
        stop_loss: float,
        take_profit: float,
        comment: str = "",
    ) -> OrderResult:
        """Execute a market (instant-fill) order.

        Parameters
        ----------
        instrument:
            Symbol string (e.g. 'XAUUSD').
        direction:
            'long' or 'short'.
        lot_size:
            Volume in lots.
        stop_loss:
            Stop-loss price level.
        take_profit:
            Take-profit price level.
        comment:
            Order comment (max 31 chars in MT5).

        Returns
        -------
        :class:`OrderResult`
        """
        mt5 = self.bridge.mt5
        if mt5 is None:
            return self._error_result("MT5 not connected")

        try:
            lot_size = self._normalize_volume(lot_size, instrument)
            if lot_size <= 0:
                return self._error_result(f"Invalid lot size after normalization: {lot_size}")

            order_type = mt5.ORDER_TYPE_BUY if direction == "long" else mt5.ORDER_TYPE_SELL

            tick = self.bridge.get_tick(instrument)
            if not tick:
                return self._error_result(f"Could not get tick for {instrument}")

            requested_price = float(tick["ask"]) if direction == "long" else float(tick["bid"])
            filling = self._detect_filling_mode(instrument)

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": instrument,
                "volume": float(lot_size),
                "type": order_type,
                "price": requested_price,
                "sl": float(stop_loss),
                "tp": float(take_profit),
                "deviation": self._deviation,
                "magic": _MAGIC_NUMBER,
                "comment": comment[:31],
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling,
            }

            result = mt5.order_send(request)
            return self._parse_result(result, requested_price, mt5)

        except Exception as exc:  # noqa: BLE001
            logger.exception("market_order failed for %s: %s", instrument, exc)
            return self._error_result(str(exc))

    # ------------------------------------------------------------------
    # Limit orders (Kijun pullback entries)
    # ------------------------------------------------------------------

    def limit_order(
        self,
        instrument: str,
        direction: str,
        lot_size: float,
        price: float,
        stop_loss: float,
        take_profit: float,
        comment: str = "",
    ) -> OrderResult:
        """Place a pending limit order for Kijun pullback entries.

        A buy limit sits below the current price; a sell limit sits above.
        The order rests until price retraces to the Kijun level.

        Parameters
        ----------
        price:
            Limit price (typically the current Kijun-Sen value).
        """
        mt5 = self.bridge.mt5
        if mt5 is None:
            return self._error_result("MT5 not connected")

        try:
            lot_size = self._normalize_volume(lot_size, instrument)
            if lot_size <= 0:
                return self._error_result(f"Invalid lot size after normalization: {lot_size}")

            order_type = (
                mt5.ORDER_TYPE_BUY_LIMIT
                if direction == "long"
                else mt5.ORDER_TYPE_SELL_LIMIT
            )
            filling = self._detect_filling_mode(instrument)

            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": instrument,
                "volume": float(lot_size),
                "type": order_type,
                "price": float(price),
                "sl": float(stop_loss),
                "tp": float(take_profit),
                "deviation": self._deviation,
                "magic": _MAGIC_NUMBER,
                "comment": comment[:31],
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling,
            }

            result = mt5.order_send(request)
            return self._parse_result(result, price, mt5)

        except Exception as exc:  # noqa: BLE001
            logger.exception("limit_order failed for %s: %s", instrument, exc)
            return self._error_result(str(exc))

    # ------------------------------------------------------------------
    # Stop-limit orders (cloud breakout entries)
    # ------------------------------------------------------------------

    def stop_limit_order(
        self,
        instrument: str,
        direction: str,
        lot_size: float,
        stop_price: float,
        limit_price: float,
        stop_loss: float,
        take_profit: float,
        comment: str = "",
    ) -> OrderResult:
        """Place a stop-limit order for cloud breakout entries.

        The stop price triggers the order; the limit price is the worst
        acceptable fill.  This avoids chasing a breakout at a bad price.

        Parameters
        ----------
        stop_price:
            Price at which the order activates.
        limit_price:
            Maximum (buy) or minimum (sell) acceptable fill price.
        """
        mt5 = self.bridge.mt5
        if mt5 is None:
            return self._error_result("MT5 not connected")

        try:
            lot_size = self._normalize_volume(lot_size, instrument)
            if lot_size <= 0:
                return self._error_result(f"Invalid lot size after normalization: {lot_size}")

            order_type = (
                mt5.ORDER_TYPE_BUY_STOP_LIMIT
                if direction == "long"
                else mt5.ORDER_TYPE_SELL_STOP_LIMIT
            )
            filling = self._detect_filling_mode(instrument)

            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": instrument,
                "volume": float(lot_size),
                "type": order_type,
                "price": float(stop_price),     # trigger price
                "stoplimit": float(limit_price),
                "sl": float(stop_loss),
                "tp": float(take_profit),
                "deviation": self._deviation,
                "magic": _MAGIC_NUMBER,
                "comment": comment[:31],
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling,
            }

            result = mt5.order_send(request)
            return self._parse_result(result, stop_price, mt5)

        except Exception as exc:  # noqa: BLE001
            logger.exception("stop_limit_order failed for %s: %s", instrument, exc)
            return self._error_result(str(exc))

    # ------------------------------------------------------------------
    # Position modification (trailing stop updates)
    # ------------------------------------------------------------------

    def modify_position(
        self,
        ticket: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> bool:
        """Modify the stop-loss and/or take-profit of an open position.

        Used for trailing-stop updates driven by the Kijun-Sen level.

        Parameters
        ----------
        ticket:
            MT5 position ticket.
        stop_loss:
            New stop-loss price, or None to leave unchanged.
        take_profit:
            New take-profit price, or None to leave unchanged.

        Returns
        -------
        True if the modification was accepted.
        """
        mt5 = self.bridge.mt5
        if mt5 is None:
            logger.error("modify_position called while disconnected")
            return False

        try:
            # Fetch current position to preserve unchanged fields
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                logger.error("No position found with ticket %s", ticket)
                return False

            pos = positions[0]

            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": pos.symbol,
                "position": ticket,
                "sl": float(stop_loss) if stop_loss is not None else float(pos.sl),
                "tp": float(take_profit) if take_profit is not None else float(pos.tp),
                "magic": _MAGIC_NUMBER,
            }

            result = mt5.order_send(request)
            if result is None:
                logger.error("modify_position: order_send returned None for ticket %s", ticket)
                return False

            success = int(result.retcode) == _RETCODE_DONE
            if not success:
                logger.warning(
                    "modify_position failed for ticket %s: retcode=%s comment=%s",
                    ticket,
                    result.retcode,
                    result.comment,
                )
            return success

        except Exception as exc:  # noqa: BLE001
            logger.exception("modify_position error for ticket %s: %s", ticket, exc)
            return False

    # ------------------------------------------------------------------
    # Position closure
    # ------------------------------------------------------------------

    def close_position(
        self,
        ticket: int,
        lot_size: Optional[float] = None,
    ) -> OrderResult:
        """Close a position fully or partially.

        For a full close, pass lot_size=None (the full position volume is used).
        For a partial close, specify the volume to reduce.

        MT5 closes by sending an opposite market order against the position.
        """
        mt5 = self.bridge.mt5
        if mt5 is None:
            return self._error_result("MT5 not connected")

        try:
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return self._error_result(f"No position found with ticket {ticket}")

            pos = positions[0]
            close_volume = float(lot_size) if lot_size is not None else float(pos.volume)
            close_volume = self._normalize_volume(close_volume, pos.symbol)

            # Opposite direction to close
            if pos.type == mt5.ORDER_TYPE_BUY:
                close_type = mt5.ORDER_TYPE_SELL
                tick = self.bridge.get_tick(pos.symbol)
                price = float(tick.get("bid", 0.0))
            else:
                close_type = mt5.ORDER_TYPE_BUY
                tick = self.bridge.get_tick(pos.symbol)
                price = float(tick.get("ask", 0.0))

            filling = self._detect_filling_mode(pos.symbol)

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": close_volume,
                "type": close_type,
                "position": ticket,
                "price": price,
                "deviation": self._deviation,
                "magic": _MAGIC_NUMBER,
                "comment": "close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling,
            }

            result = mt5.order_send(request)
            return self._parse_result(result, price, mt5)

        except Exception as exc:  # noqa: BLE001
            logger.exception("close_position failed for ticket %s: %s", ticket, exc)
            return self._error_result(str(exc))

    def close_partial(self, ticket: int, close_pct: float = 0.5) -> OrderResult:
        """Close a percentage of an open position.

        Used by the hybrid 50/50 exit strategy to close 50% at 2R.

        Parameters
        ----------
        ticket:
            MT5 position ticket.
        close_pct:
            Fraction to close (0 < close_pct <= 1.0). Default: 0.5.
        """
        if not 0.0 < close_pct <= 1.0:
            return self._error_result(f"close_pct must be in (0, 1], got {close_pct}")

        mt5 = self.bridge.mt5
        if mt5 is None:
            return self._error_result("MT5 not connected")

        try:
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return self._error_result(f"No position found with ticket {ticket}")

            pos = positions[0]
            partial_volume = float(pos.volume) * close_pct
            return self.close_position(ticket, lot_size=partial_volume)

        except Exception as exc:  # noqa: BLE001
            logger.exception("close_partial failed for ticket %s: %s", ticket, exc)
            return self._error_result(str(exc))

    # ------------------------------------------------------------------
    # Slippage log
    # ------------------------------------------------------------------

    @property
    def slippage_log(self) -> List[dict]:
        """Read-only copy of the slippage log for post-trade analysis."""
        return list(self._slippage_log)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_filling_mode(self, instrument: str) -> int:
        """Auto-detect the supported filling mode for a symbol.

        MT5 symbol_info.filling_mode is a bitmask:
        - bit 0 set → FOK supported
        - bit 1 set → IOC supported
        - neither set → RETURN (exchange execution)

        Returns the MT5 ORDER_FILLING_* constant appropriate for this symbol.
        """
        mt5 = self.bridge.mt5
        if mt5 is None:
            return 0  # fallback; will cause order error but prevents crash

        try:
            info = self.bridge.get_symbol_info(instrument)
            filling_mode_bits = int(info.get("filling_mode", 0))

            if filling_mode_bits & _FILLING_FOK:
                return mt5.ORDER_FILLING_FOK
            if filling_mode_bits & _FILLING_IOC:
                return mt5.ORDER_FILLING_IOC
            # Exchange/RETURN mode — no partial fills, no requotes
            return mt5.ORDER_FILLING_RETURN

        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not detect filling mode for %s: %s", instrument, exc)
            # Default to FOK as it is the most common for CFD brokers
            return mt5.ORDER_FILLING_FOK

    def _normalize_volume(self, volume: float, instrument: str) -> float:
        """Round a lot size to the nearest valid lot step.

        MT5 rejects orders with volumes not aligned to volume_step.
        For most CFD brokers this is 0.01.
        """
        try:
            info = self.bridge.get_symbol_info(instrument)
            step = float(info.get("volume_step", 0.01))
            vol_min = float(info.get("volume_min", 0.01))
            vol_max = float(info.get("volume_max", 500.0))

            if step <= 0:
                step = 0.01

            # Round down to nearest step (floor to avoid exceeding risk budget)
            steps = math.floor(volume / step)
            normalized = round(steps * step, 8)  # avoid float representation drift

            # Clamp to broker limits
            normalized = max(vol_min, min(normalized, vol_max))
            return normalized

        except Exception as exc:  # noqa: BLE001
            logger.warning("Volume normalization failed for %s: %s", instrument, exc)
            return round(volume, 2)

    def _parse_result(self, result, requested_price: float, mt5) -> OrderResult:
        """Parse an mt5.order_send() result into an :class:`OrderResult`.

        Also logs slippage unconditionally — even on success — for later analysis.
        """
        if result is None:
            error = mt5.last_error()
            logger.error("order_send returned None: %s", error)
            return self._error_result(f"order_send returned None: {error}")

        retcode = int(result.retcode)
        success = retcode == _RETCODE_DONE

        fill_price = float(getattr(result, "price", 0.0))
        volume = float(getattr(result, "volume", 0.0))
        comment = str(getattr(result, "comment", ""))
        ticket = int(getattr(result, "order", 0))

        # Calculate slippage in points (positive = price moved against us)
        slippage = 0.0
        if fill_price > 0.0 and requested_price > 0.0:
            slippage = fill_price - requested_price

        # Log slippage regardless of success — 0.0 for failed orders
        self._log_slippage(
            requested_price=requested_price,
            filled_price=fill_price,
            direction="n/a",
            slippage_points=slippage,
            retcode=retcode,
            success=success,
        )

        if not success:
            error_msg = self._retcode_description(retcode)
            logger.warning(
                "order_send failed: retcode=%s (%s) comment=%s",
                retcode,
                error_msg,
                comment,
            )
            return OrderResult(
                success=False,
                ticket=ticket,
                price=fill_price,
                volume=volume,
                retcode=retcode,
                comment=comment,
                slippage=slippage,
                error_message=error_msg,
            )

        logger.info(
            "Order filled: ticket=%s price=%.5f volume=%.2f slippage=%.5f",
            ticket,
            fill_price,
            volume,
            slippage,
        )
        return OrderResult(
            success=True,
            ticket=ticket,
            price=fill_price,
            volume=volume,
            retcode=retcode,
            comment=comment,
            slippage=slippage,
        )

    def _log_slippage(
        self,
        requested_price: float,
        filled_price: float,
        direction: str,
        slippage_points: float,
        retcode: int,
        success: bool,
    ) -> None:
        """Append a slippage entry to the internal log."""
        self._slippage_log.append(
            {
                "requested_price": requested_price,
                "filled_price": filled_price,
                "direction": direction,
                "slippage_points": slippage_points,
                "retcode": retcode,
                "success": success,
            }
        )

    @staticmethod
    def _error_result(message: str) -> OrderResult:
        """Construct a failed OrderResult with the given error message."""
        logger.error("OrderResult error: %s", message)
        return OrderResult(
            success=False,
            ticket=0,
            price=0.0,
            volume=0.0,
            retcode=0,
            comment="",
            slippage=0.0,
            error_message=message,
        )

    @staticmethod
    def _retcode_description(retcode: int) -> str:
        """Map common MT5 return codes to human-readable descriptions."""
        descriptions = {
            _RETCODE_DONE:          "Request executed successfully",
            _RETCODE_MARKET_CLOSED: "Market is closed (retcode 10030)",
            _RETCODE_REJECTED:      "Request rejected by server (retcode 10006)",
            _RETCODE_NO_MONEY:      "Not enough money (retcode 10019)",
            _RETCODE_PRICE_OFF:     "Price has changed, retry (retcode 10021)",
            _RETCODE_REQUOTE:       "Requote (retcode 10004)",
        }
        return descriptions.get(retcode, f"Unknown retcode {retcode}")
