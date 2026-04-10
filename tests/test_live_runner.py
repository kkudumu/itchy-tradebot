"""Tests for LiveRunner + PaperExecutionProvider (plan Task 18)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

from src.live.live_runner import LiveRunner, LiveRunnerConfig, PaperExecutionProvider
from src.providers.base import ContractSpec, PositionSnapshot, OrderSnapshot, AccountSnapshot


UTC = timezone.utc


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeMarketData:
    """Returns a fixed 5M DataFrame that grows by one bar on each call."""

    def __init__(self) -> None:
        base_ts = datetime(2026, 4, 9, 10, 0, tzinfo=UTC)
        self._bars: list[tuple[datetime, float, float, float, float]] = [
            (base_ts, 2000.0, 2001.0, 1999.5, 2000.5),
            (base_ts + timedelta(minutes=5), 2000.5, 2002.0, 2000.0, 2001.5),
        ]
        self._call_count = 0

    def _make_df(self) -> pd.DataFrame:
        rows = [
            {
                "time": ts,
                "open": o,
                "high": h,
                "low": lo,
                "close": c,
                "volume": 100,
            }
            for ts, o, h, lo, c in self._bars
        ]
        df = pd.DataFrame(rows).set_index("time")
        return df

    def get_multi_tf_data(self, instrument: str, count: int = 500, include_partial_bar: bool = False):
        self._call_count += 1
        # Add a new bar on the third call so LiveRunner sees a new bar
        if self._call_count == 3:
            last_ts, *_ = self._bars[-1]
            self._bars.append(
                (last_ts + timedelta(minutes=5), 2001.5, 2003.0, 2001.0, 2002.5)
            )
        return {"5M": self._make_df(), "15M": self._make_df()}

    def fetch_bars(self, *args, **kwargs):
        return self._make_df()

    def get_tick(self, instrument: str):
        return {}

    def get_contract_spec(self, instrument: str) -> ContractSpec:
        return ContractSpec(contract_id="MGC", provider="fake")


class FakeAccount:
    def get_account_info(self) -> AccountSnapshot | None:
        return AccountSnapshot(
            account_id=1, balance=50_000.0, equity=50_000.0
        )

    def get_positions(self, instrument: str | None = None) -> List[PositionSnapshot]:
        return []

    def get_open_orders(self, instrument: str | None = None) -> List[OrderSnapshot]:
        return []


# ---------------------------------------------------------------------------
# PaperExecutionProvider
# ---------------------------------------------------------------------------


class TestPaperExecutionProvider:
    def test_place_market_order_returns_success(self):
        p = PaperExecutionProvider()
        result = p.place_market_order(
            "MGC", "long", 1.0, stop_loss=1995.0, take_profit=2010.0
        )
        assert result.success is True
        assert result.order_id.startswith("paper-")
        assert len(p.open_positions()) == 1

    def test_close_position_removes_from_ledger(self):
        p = PaperExecutionProvider()
        p.place_market_order("MGC", "long", 1.0, stop_loss=1995.0, take_profit=2010.0)
        assert len(p.open_positions()) == 1
        assert p.close_position("MGC") is True
        assert len(p.open_positions()) == 0


# ---------------------------------------------------------------------------
# LiveRunner
# ---------------------------------------------------------------------------


class TestLiveRunner:
    def test_runs_with_duration_limit_and_exits_cleanly(self):
        runner = LiveRunner(
            config=LiveRunnerConfig(
                instrument="MGC",
                poll_interval_seconds=0.05,
                duration_seconds=0.3,
                paper=True,
            ),
            market_data=FakeMarketData(),
            execution=PaperExecutionProvider(),
            account=FakeAccount(),
            strategy_fn=None,
        )
        summary = runner.run()
        assert summary["instrument"] == "MGC"
        assert summary["paper"] is True
        assert summary["bars_seen"] >= 1

    def test_request_stop_exits_loop(self):
        runner = LiveRunner(
            config=LiveRunnerConfig(
                instrument="MGC",
                poll_interval_seconds=0.05,
                duration_seconds=10.0,  # long duration
                paper=True,
            ),
            market_data=FakeMarketData(),
            execution=PaperExecutionProvider(),
            account=FakeAccount(),
        )
        # Stop before run() — exits the loop before even the first poll
        runner.request_stop()
        summary = runner.run()
        # The loop exits at the top of the while check, so no bars are
        # seen regardless of how fast the fake market data responds.
        assert summary["bars_seen"] == 0

    def test_strategy_fn_gets_called_on_new_bar(self):
        calls = []

        def fake_strategy_fn(mtf):
            calls.append(1)
            return None  # don't actually place any orders

        runner = LiveRunner(
            config=LiveRunnerConfig(
                instrument="MGC",
                poll_interval_seconds=0.05,
                duration_seconds=0.5,
                paper=True,
            ),
            market_data=FakeMarketData(),
            execution=PaperExecutionProvider(),
            account=FakeAccount(),
            strategy_fn=fake_strategy_fn,
        )
        runner.run()
        assert len(calls) >= 1
