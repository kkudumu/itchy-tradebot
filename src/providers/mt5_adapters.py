"""Compatibility adapters that expose the existing MT5 stack as generic providers."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from .base import (
    AccountProvider,
    AccountSnapshot,
    ContractSpec,
    ExecutionProvider,
    ExecutionResult,
    MarketDataProvider,
    OrderSnapshot,
    PositionSnapshot,
)


_TF_TO_MT5_LABEL = {
    "1M": "M1",
    "5M": "M5",
    "15M": "M15",
    "1H": "H1",
    "4H": "H4",
}


class MT5MarketDataProvider(MarketDataProvider):
    def __init__(self, bridge) -> None:
        self.bridge = bridge

    def get_multi_tf_data(
        self,
        instrument: str,
        count: int = 500,
        include_partial_bar: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        result: Dict[str, pd.DataFrame] = {}
        for label in ("1M", "5M", "15M", "1H", "4H"):
            result[label] = self.fetch_bars(
                instrument=instrument,
                timeframe=label,
                limit=count,
                include_partial_bar=include_partial_bar,
            )
        return result

    def fetch_bars(
        self,
        instrument: str,
        timeframe: str,
        start_time=None,
        end_time=None,
        limit: int | None = None,
        include_partial_bar: bool = False,
    ) -> pd.DataFrame:
        tf_label = _TF_TO_MT5_LABEL.get(timeframe.upper(), timeframe.upper())
        tf_const = self.bridge.timeframe_constant(tf_label)
        if tf_const is None:
            return pd.DataFrame()
        return self.bridge.get_rates(instrument, tf_const, count=limit or 500)

    def get_tick(self, instrument: str) -> Dict[str, Any]:
        return self.bridge.get_tick(instrument)

    def get_contract_spec(self, instrument: str) -> ContractSpec:
        info = self.bridge.get_symbol_info(instrument)
        point = float(info.get("point", 0.0) or 0.0)
        tick_value = float(info.get("trade_tick_value", 0.0) or 0.0)
        point_value = tick_value / point if point > 0 and tick_value > 0 else None
        return ContractSpec(
            contract_id=instrument,
            symbol_id=instrument,
            name=instrument,
            tick_size=point or None,
            tick_value=tick_value or None,
            point_value=point_value,
            provider="mt5",
            raw=info,
        )


class MT5ExecutionProvider(ExecutionProvider):
    def __init__(self, order_manager) -> None:
        self.order_manager = order_manager

    def place_market_order(
        self,
        instrument: str,
        direction: str,
        quantity: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        comment: str = "",
    ) -> ExecutionResult:
        result = self.order_manager.market_order(
            instrument=instrument,
            direction=direction,
            lot_size=quantity,
            stop_loss=float(stop_loss or 0.0),
            take_profit=float(take_profit or 0.0),
            comment=comment,
        )
        return ExecutionResult(
            success=result.success,
            order_id=result.ticket,
            fill_price=result.price,
            quantity=result.volume,
            error_code=result.retcode,
            error_message=result.error_message,
            raw={"comment": result.comment, "slippage": result.slippage},
        )

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
        result = self.order_manager.limit_order(
            instrument=instrument,
            direction=direction,
            lot_size=quantity,
            price=price,
            stop_loss=float(stop_loss or 0.0),
            take_profit=float(take_profit or 0.0),
            comment=comment,
        )
        return ExecutionResult(
            success=result.success,
            order_id=result.ticket,
            fill_price=result.price,
            quantity=result.volume,
            error_code=result.retcode,
            error_message=result.error_message,
            raw={"comment": result.comment, "slippage": result.slippage},
        )

    def modify_order(
        self,
        order_id: int | str,
        quantity: float | None = None,
        limit_price: float | None = None,
        stop_price: float | None = None,
        trail_price: float | None = None,
    ) -> bool:
        stop = stop_price if stop_price is not None else limit_price
        return self.order_manager.modify_position(int(order_id), stop_loss=stop)

    def cancel_order(self, order_id: int | str) -> bool:
        return False

    def close_position(self, instrument: str, quantity: float | None = None) -> bool:
        ticket = int(instrument) if str(instrument).isdigit() else None
        if ticket is None:
            return False
        result = self.order_manager.close_position(ticket=ticket, lot_size=quantity)
        return result.success

    def partial_close_position(self, instrument: str, quantity: float) -> bool:
        return self.close_position(instrument, quantity)


class MT5AccountProvider(AccountProvider):
    def __init__(self, account_monitor) -> None:
        self.account_monitor = account_monitor

    def get_account_info(self) -> AccountSnapshot | None:
        info = self.account_monitor.get_account_info()
        if info is None:
            return None
        return AccountSnapshot(
            account_id=0,
            balance=info.balance,
            equity=info.equity,
            margin=info.margin,
            free_margin=info.free_margin,
            unrealized_pnl=info.unrealized_pnl,
            leverage=info.leverage,
            raw={},
        )

    def get_positions(self, instrument: str | None = None) -> list[PositionSnapshot]:
        result: list[PositionSnapshot] = []
        for pos in self.account_monitor.get_positions(instrument=instrument):
            result.append(
                PositionSnapshot(
                    position_id=pos.ticket,
                    account_id=None,
                    contract_id=pos.instrument,
                    instrument=pos.instrument,
                    direction=pos.direction,
                    quantity=pos.volume,
                    entry_price=pos.entry_price,
                    current_price=pos.current_price,
                    stop_loss=pos.stop_loss,
                    take_profit=pos.take_profit,
                    profit=pos.profit,
                    time=pos.time,
                )
            )
        return result

    def get_open_orders(self, instrument: str | None = None) -> list[OrderSnapshot]:
        result: list[OrderSnapshot] = []
        for order in self.account_monitor.get_open_orders(instrument=instrument):
            result.append(
                OrderSnapshot(
                    order_id=order["ticket"],
                    account_id=None,
                    contract_id=order["symbol"],
                    instrument=order["symbol"],
                    side=str(order["type"]),
                    order_type=str(order["type"]),
                    quantity=order["volume"],
                    limit_price=order["price"],
                    raw=order,
                )
            )
        return result
