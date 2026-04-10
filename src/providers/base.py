"""Provider abstractions for market data, account state, and execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Protocol

import pandas as pd


@dataclass
class TickSnapshot:
    bid: float
    ask: float
    spread: float
    time: Optional[datetime] = None


@dataclass
class ContractSpec:
    contract_id: str
    symbol_id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    tick_size: Optional[float] = None
    tick_value: Optional[float] = None
    point_value: Optional[float] = None
    default_quantity: Optional[int] = None
    provider: str = "unknown"
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccountSnapshot:
    account_id: int
    balance: float
    equity: float
    margin: float = 0.0
    free_margin: float = 0.0
    unrealized_pnl: float = 0.0
    leverage: int = 0
    can_trade: bool = True
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionSnapshot:
    position_id: int | str
    account_id: int | None
    contract_id: str
    instrument: str
    direction: str
    quantity: float
    entry_price: float
    current_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    profit: float | None = None
    time: Optional[datetime] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderSnapshot:
    order_id: int | str
    account_id: int | None
    contract_id: str
    instrument: str
    side: str
    order_type: str
    quantity: float
    status: str | int | None = None
    limit_price: float | None = None
    stop_price: float | None = None
    filled_price: float | None = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    success: bool
    order_id: int | str | None = None
    fill_price: float | None = None
    quantity: float | None = None
    error_code: int | None = None
    error_message: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


class MarketDataProvider(Protocol):
    def get_multi_tf_data(
        self,
        instrument: str,
        count: int = 500,
        include_partial_bar: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        ...

    def fetch_bars(
        self,
        instrument: str,
        timeframe: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None,
        include_partial_bar: bool = False,
    ) -> pd.DataFrame:
        ...

    def get_tick(self, instrument: str) -> Dict[str, Any]:
        ...

    def get_contract_spec(self, instrument: str) -> ContractSpec:
        ...


class ExecutionProvider(Protocol):
    def place_market_order(
        self,
        instrument: str,
        direction: str,
        quantity: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        comment: str = "",
    ) -> ExecutionResult:
        ...

    def place_limit_order(
        self,
        instrument: str,
        direction: str,
        quantity: float,
        price: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        comment: str = "",
    ) -> ExecutionResult:
        ...

    def modify_order(
        self,
        order_id: int | str,
        quantity: float | None = None,
        limit_price: float | None = None,
        stop_price: float | None = None,
        trail_price: float | None = None,
    ) -> bool:
        ...

    def cancel_order(self, order_id: int | str) -> bool:
        ...

    def close_position(self, instrument: str, quantity: float | None = None) -> bool:
        ...

    def partial_close_position(self, instrument: str, quantity: float) -> bool:
        ...


class AccountProvider(Protocol):
    def get_account_info(self) -> AccountSnapshot | None:
        ...

    def get_positions(self, instrument: str | None = None) -> list[PositionSnapshot]:
        ...

    def get_open_orders(self, instrument: str | None = None) -> list[OrderSnapshot]:
        ...
