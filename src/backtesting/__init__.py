"""
Vectorbt backtesting integration for the XAU/USD Ichimoku trading system.

Provides:
- BacktestDataPreparer: multi-TF data preparation from 1M source
- IchimokuBacktester: main engine wrapping vectorbt Portfolio.from_orders()
- PerformanceMetrics: Sharpe, Sortino, Calmar, profit factor, win rate, etc.
- PropFirmTracker: The5ers-style challenge constraint tracking
- TradeLogger: PostgreSQL insertion with pgvector embeddings

Usage example::

    from src.backtesting import IchimokuBacktester
    result = IchimokuBacktester(initial_balance=10000.0).run(candles_1m)
    print(result.metrics)
"""

from .vectorbt_engine import IchimokuBacktester, BacktestResult
from .multi_tf import BacktestDataPreparer
from .metrics import PerformanceMetrics, PropFirmTracker, PropFirmStatus
from .trade_logger import TradeLogger

__all__ = [
    "IchimokuBacktester",
    "BacktestResult",
    "BacktestDataPreparer",
    "PerformanceMetrics",
    "PropFirmTracker",
    "PropFirmStatus",
    "TradeLogger",
]
