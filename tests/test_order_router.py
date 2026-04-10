"""Tests for OrderRouter (plan Task 18)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pytest

from src.live.live_runner import PaperExecutionProvider
from src.live.order_router import OrderRouter, RouterRejection, RouterResult
from src.risk.instrument_sizer import FuturesContractSizer


@dataclass
class FakeSignal:
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    strategy_name: str = "test"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FakeTracker:
    def __init__(self, status: str = "pending", distance_to_mll: float = 10_000.0):
        self.status = status
        self._distance = distance_to_mll

    def to_dict(self):
        return {"status": self.status, "distance_to_mll": self._distance}


@pytest.fixture
def paper_exec():
    return PaperExecutionProvider()


@pytest.fixture
def sizer():
    return FuturesContractSizer(tick_size=0.10, tick_value_usd=1.0, max_contracts=50)


class TestOrderRouter:
    def test_happy_path_places_order(self, paper_exec, sizer):
        router = OrderRouter(
            execution_provider=paper_exec,
            instrument_sizer=sizer,
            prop_firm_tracker=FakeTracker(),
        )
        signal = FakeSignal(direction="long", entry_price=2000.0, stop_loss=1995.0, take_profit=2010.0)
        result = router.route(signal, "MGC", account_equity=50_000.0, risk_pct=1.0)
        assert isinstance(result, RouterResult)
        assert result.executed is True
        assert result.quantity == 10  # $500 risk / $50 per contract = 10 contracts

    def test_prop_firm_failed_rejects(self, paper_exec, sizer):
        router = OrderRouter(
            execution_provider=paper_exec,
            instrument_sizer=sizer,
            prop_firm_tracker=FakeTracker(status="failed_mll"),
        )
        signal = FakeSignal(direction="long", entry_price=2000.0, stop_loss=1995.0, take_profit=2010.0)
        result = router.route(signal, "MGC", account_equity=50_000.0, risk_pct=1.0)
        assert isinstance(result, RouterRejection)
        assert result.stage == "prop_firm"

    def test_zero_stop_distance_rejects(self, paper_exec, sizer):
        router = OrderRouter(
            execution_provider=paper_exec,
            instrument_sizer=sizer,
            prop_firm_tracker=FakeTracker(),
        )
        signal = FakeSignal(direction="long", entry_price=2000.0, stop_loss=2000.0, take_profit=2010.0)
        result = router.route(signal, "MGC", account_equity=50_000.0, risk_pct=1.0)
        assert isinstance(result, RouterRejection)
        assert result.stage == "size"

    def test_zero_qty_rejects(self, paper_exec, sizer):
        router = OrderRouter(
            execution_provider=paper_exec,
            instrument_sizer=sizer,
            prop_firm_tracker=FakeTracker(),
        )
        # Huge stop → 0 contracts
        signal = FakeSignal(direction="long", entry_price=2000.0, stop_loss=1000.0, take_profit=2010.0)
        result = router.route(signal, "MGC", account_equity=50_000.0, risk_pct=0.01)
        assert isinstance(result, RouterRejection)
        assert result.stage == "size"

    def test_contract_cap_enforced(self, paper_exec):
        small_cap_sizer = FuturesContractSizer(
            tick_size=0.10, tick_value_usd=1.0, max_contracts=3
        )
        router = OrderRouter(
            execution_provider=paper_exec,
            instrument_sizer=small_cap_sizer,
            prop_firm_tracker=FakeTracker(),
            max_contracts=3,
        )
        # Would naturally size to 10 contracts but cap clamps to 3
        signal = FakeSignal(direction="long", entry_price=2000.0, stop_loss=1995.0, take_profit=2010.0)
        result = router.route(signal, "MGC", account_equity=50_000.0, risk_pct=1.0)
        assert isinstance(result, RouterResult)
        assert result.quantity == 3

    def test_mll_headroom_rejects_when_risk_exceeds(self, paper_exec, sizer):
        # Only $100 distance to MLL, $500 risk trade → reject
        router = OrderRouter(
            execution_provider=paper_exec,
            instrument_sizer=sizer,
            prop_firm_tracker=FakeTracker(distance_to_mll=100.0),
        )
        signal = FakeSignal(direction="long", entry_price=2000.0, stop_loss=1995.0, take_profit=2010.0)
        result = router.route(signal, "MGC", account_equity=50_000.0, risk_pct=1.0)
        assert isinstance(result, RouterRejection)
        assert result.stage == "mll_headroom"

    def test_kill_switch_rejects(self, paper_exec, sizer):
        router = OrderRouter(
            execution_provider=paper_exec,
            instrument_sizer=sizer,
            prop_firm_tracker=FakeTracker(),
            kill_switch_fn=lambda: True,
        )
        signal = FakeSignal(direction="long", entry_price=2000.0, stop_loss=1995.0, take_profit=2010.0)
        result = router.route(signal, "MGC", account_equity=50_000.0, risk_pct=1.0)
        assert isinstance(result, RouterRejection)
        assert result.stage == "kill_switch"

    def test_placed_order_shows_up_in_paper_ledger(self, paper_exec, sizer):
        router = OrderRouter(
            execution_provider=paper_exec,
            instrument_sizer=sizer,
            prop_firm_tracker=FakeTracker(),
        )
        signal = FakeSignal(direction="long", entry_price=2000.0, stop_loss=1995.0, take_profit=2010.0)
        router.route(signal, "MGC", account_equity=50_000.0, risk_pct=1.0)
        assert len(paper_exec.open_positions()) == 1

    def test_emits_telemetry_on_rejection(self, paper_exec, sizer):
        from src.backtesting.strategy_telemetry import StrategyTelemetryCollector

        telemetry = StrategyTelemetryCollector(run_id="router_test")
        router = OrderRouter(
            execution_provider=paper_exec,
            instrument_sizer=sizer,
            prop_firm_tracker=FakeTracker(status="failed_daily_loss"),
            telemetry=telemetry,
        )
        signal = FakeSignal(direction="long", entry_price=2000.0, stop_loss=1995.0, take_profit=2010.0)
        router.route(signal, "MGC", account_equity=50_000.0, risk_pct=1.0)
        events = telemetry.events()
        assert len(events) == 1
        assert events[0].event_type == "signal_rejected_risk"
        assert events[0].filter_stage == "router.prop_firm"
