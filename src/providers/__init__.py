"""Provider abstractions and concrete adapters."""

from .base import (
    AccountProvider,
    AccountSnapshot,
    ContractSpec,
    ExecutionProvider,
    ExecutionResult,
    MarketDataProvider,
    OrderSnapshot,
    PositionSnapshot,
    TickSnapshot,
)
from .mt5_adapters import MT5AccountProvider, MT5ExecutionProvider, MT5MarketDataProvider
from .projectx import (
    ProjectXAccountProvider,
    ProjectXApiError,
    ProjectXClient,
    ProjectXExecutionProvider,
    ProjectXHistoricalDataLoader,
    ProjectXMarketDataProvider,
    build_projectx_stack,
)

__all__ = [
    "AccountProvider",
    "AccountSnapshot",
    "ContractSpec",
    "ExecutionProvider",
    "ExecutionResult",
    "MarketDataProvider",
    "OrderSnapshot",
    "PositionSnapshot",
    "TickSnapshot",
    "MT5AccountProvider",
    "MT5ExecutionProvider",
    "MT5MarketDataProvider",
    "ProjectXAccountProvider",
    "ProjectXApiError",
    "ProjectXClient",
    "ProjectXExecutionProvider",
    "ProjectXHistoricalDataLoader",
    "ProjectXMarketDataProvider",
    "build_projectx_stack",
]
