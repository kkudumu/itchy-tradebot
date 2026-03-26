"""MT5 execution bridge — order management, monitoring, and screenshots.

This package wraps all MetaTrader5 interactions behind mockable interfaces,
enabling full test coverage on Linux where the MetaTrader5 package is unavailable.

Modules
-------
mt5_bridge      — MT5 terminal initialisation, data retrieval, connection management
order_manager   — Order execution, modification, partial closes, slippage logging
screenshot      — Chart screenshot capture with matplotlib fallback
account_monitor — Real-time equity and position monitoring for circuit-breaker integration
"""

from .mt5_bridge import MT5Bridge
from .order_manager import OrderManager, OrderResult
from .screenshot import ScreenshotCapture
from .account_monitor import AccountMonitor, AccountInfo, PositionInfo

__all__ = [
    "MT5Bridge",
    "OrderManager",
    "OrderResult",
    "ScreenshotCapture",
    "AccountMonitor",
    "AccountInfo",
    "PositionInfo",
]
