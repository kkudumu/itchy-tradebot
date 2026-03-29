"""Integration tests for StrategyHealthMonitor wiring into backtester, decision engine, and dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from src.strategy.signal_engine import Signal, ScanResult, SignalEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candles(n: int = 1000) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=n, freq="1min")
    close = 2000.0 + np.cumsum(np.random.default_rng(42).normal(0, 0.5, n))
    return pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 100,
        },
        index=dates,
    )


def _make_signal(ts=None) -> Signal:
    return Signal(
        timestamp=ts or datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc),
        instrument="XAUUSD",
        direction="long",
        entry_price=2000.0,
        stop_loss=1990.0,
        take_profit=2020.0,
        confluence_score=5,
        quality_tier="B",
        atr=10.0,
    )


# ===========================================================================
# Part 1: ScanResult tests
# ===========================================================================


class TestScanResultBackwardCompatibility:
    """scan() still returns Optional[Signal] by default."""

    def test_scan_default_returns_none_or_signal(self):
        engine = SignalEngine()
        result = engine.scan(data_1m=_make_candles(500))
        assert result is None or isinstance(result, Signal)

    def test_scan_default_does_not_return_scan_result(self):
        engine = SignalEngine()
        result = engine.scan(data_1m=_make_candles(500))
        assert not isinstance(result, ScanResult)


class TestScanResultWithFlag:
    """scan(return_scan_result=True) returns ScanResult."""

    def test_returns_scan_result_type(self):
        engine = SignalEngine()
        result = engine.scan(data_1m=_make_candles(500), return_scan_result=True)
        assert isinstance(result, ScanResult)

    def test_scan_result_has_filters(self):
        engine = SignalEngine()
        result = engine.scan(data_1m=_make_candles(500), return_scan_result=True)
        assert isinstance(result.filters, dict)
        assert len(result.filters) >= 1  # at least 4h_cloud

    def test_scan_result_4h_cloud_always_present(self):
        engine = SignalEngine()
        result = engine.scan(data_1m=_make_candles(500), return_scan_result=True)
        assert "4h_cloud" in result.filters

    def test_scan_result_filter_has_pass_and_reason(self):
        engine = SignalEngine()
        result = engine.scan(data_1m=_make_candles(500), return_scan_result=True)
        for name, filt in result.filters.items():
            assert "pass" in filt, f"Filter '{name}' missing 'pass' key"
            assert "reason" in filt, f"Filter '{name}' missing 'reason' key"

    def test_scan_result_passed_all_false_when_no_signal(self):
        engine = SignalEngine()
        result = engine.scan(data_1m=_make_candles(500), return_scan_result=True)
        if result.signal is None:
            assert result.passed_all is False

    def test_scan_result_passed_all_true_when_signal(self):
        engine = SignalEngine()
        result = engine.scan(data_1m=_make_candles(500), return_scan_result=True)
        if result.signal is not None:
            assert result.passed_all is True
            assert isinstance(result.signal, Signal)

    def test_scan_result_signal_matches_standalone(self):
        """ScanResult.signal should be the same Signal as standalone scan."""
        candles = _make_candles(500)
        engine = SignalEngine()
        plain = engine.scan(data_1m=candles)
        rich = engine.scan(data_1m=candles, return_scan_result=True)
        if plain is not None:
            assert rich.signal is not None
            assert rich.signal.direction == plain.direction
            assert rich.signal.entry_price == plain.entry_price


# ===========================================================================
# Part 2: Backtester health monitor integration
# ===========================================================================


class TestBacktesterHealthMonitor:
    """IchimokuBacktester creates and wires the health monitor."""

    def test_backtester_has_health_monitor_attr(self):
        from src.backtesting.vectorbt_engine import IchimokuBacktester
        bt = IchimokuBacktester()
        assert hasattr(bt, "health_monitor")

    def test_health_monitor_type(self):
        from src.backtesting.vectorbt_engine import IchimokuBacktester
        from src.monitoring.health_monitor import StrategyHealthMonitor
        bt = IchimokuBacktester()
        assert isinstance(bt.health_monitor, StrategyHealthMonitor)

    def test_health_monitor_mode_is_backtest(self):
        from src.backtesting.vectorbt_engine import IchimokuBacktester
        bt = IchimokuBacktester()
        assert bt.health_monitor._mode == "backtest"

    def test_health_monitor_shares_signal_engine(self):
        from src.backtesting.vectorbt_engine import IchimokuBacktester
        bt = IchimokuBacktester()
        assert bt.health_monitor._signal_engine is bt.signal_engine

    def test_health_monitor_shares_edge_manager(self):
        from src.backtesting.vectorbt_engine import IchimokuBacktester
        bt = IchimokuBacktester()
        assert bt.health_monitor._edge_manager is bt.edge_manager


# ===========================================================================
# Part 3: Dashboard health fields
# ===========================================================================


class TestDashboardHealthFields:
    """LiveDashboardServer state dict includes health monitor fields."""

    def test_health_state_in_initial_state(self):
        from src.backtesting.live_dashboard import LiveDashboardServer
        server = LiveDashboardServer(port=0, auto_open=False)
        state = server._state.get()
        assert "health_state" in state
        assert state["health_state"] == "normal"

    def test_health_drought_in_initial_state(self):
        from src.backtesting.live_dashboard import LiveDashboardServer
        server = LiveDashboardServer(port=0, auto_open=False)
        state = server._state.get()
        assert "health_drought" in state
        assert state["health_drought"] is False

    def test_health_relaxation_tier_in_initial_state(self):
        from src.backtesting.live_dashboard import LiveDashboardServer
        server = LiveDashboardServer(port=0, auto_open=False)
        state = server._state.get()
        assert "health_relaxation_tier" in state
        assert state["health_relaxation_tier"] == 0

    def test_health_regime_in_initial_state(self):
        from src.backtesting.live_dashboard import LiveDashboardServer
        server = LiveDashboardServer(port=0, auto_open=False)
        state = server._state.get()
        assert "health_regime" in state

    def test_health_bottleneck_in_initial_state(self):
        from src.backtesting.live_dashboard import LiveDashboardServer
        server = LiveDashboardServer(port=0, auto_open=False)
        state = server._state.get()
        assert "health_bottleneck" in state

    def test_health_message_in_initial_state(self):
        from src.backtesting.live_dashboard import LiveDashboardServer
        server = LiveDashboardServer(port=0, auto_open=False)
        state = server._state.get()
        assert "health_message" in state


# ===========================================================================
# Part 4: Decision engine health monitor
# ===========================================================================


class TestDecisionEngineHealthMonitor:
    """DecisionEngine creates health monitor when dependencies are provided."""

    def test_has_health_monitor_attr(self):
        from src.engine.decision_engine import DecisionEngine
        engine = DecisionEngine(config={})
        assert hasattr(engine, "health_monitor")

    def test_health_monitor_none_without_deps(self):
        from src.engine.decision_engine import DecisionEngine
        engine = DecisionEngine(config={})
        assert engine.health_monitor is None

    def test_health_monitor_created_with_deps(self):
        from src.engine.decision_engine import DecisionEngine
        from src.monitoring.health_monitor import StrategyHealthMonitor
        se = SignalEngine()
        from src.edges.manager import EdgeManager
        em = EdgeManager(edge_configs={})
        engine = DecisionEngine(
            config={},
            signal_engine=se,
            edge_manager=em,
        )
        assert isinstance(engine.health_monitor, StrategyHealthMonitor)


# ===========================================================================
# Part 5: Health monitor state machine integration
# ===========================================================================


class TestHealthMonitorStateMachine:
    """Test state machine transitions via the orchestrator API."""

    def test_starts_in_normal_state(self):
        from src.monitoring.health_monitor import StrategyHealthMonitor, HealthState
        se = MagicMock()
        em = MagicMock()
        # Prevent AdaptiveRelaxer from calling real edge methods
        em.get_edge = MagicMock(side_effect=KeyError("mock"))
        monitor = StrategyHealthMonitor(se, em)
        assert monitor.state == HealthState.NORMAL

    def test_is_halted_false_initially(self):
        from src.monitoring.health_monitor import StrategyHealthMonitor
        se = MagicMock()
        em = MagicMock()
        em.get_edge = MagicMock(side_effect=KeyError("mock"))
        monitor = StrategyHealthMonitor(se, em)
        assert monitor.is_halted is False

    def test_get_status_returns_health_status(self):
        from src.monitoring.health_monitor import StrategyHealthMonitor, HealthStatus
        se = MagicMock()
        em = MagicMock()
        em.get_edge = MagicMock(side_effect=KeyError("mock"))
        monitor = StrategyHealthMonitor(se, em)
        status = monitor.get_status()
        assert isinstance(status, HealthStatus)
        assert status.bars_processed == 0

    def test_on_bar_increments_bar_count(self):
        from src.monitoring.health_monitor import StrategyHealthMonitor
        se = MagicMock()
        em = MagicMock()
        em.get_edge = MagicMock(side_effect=KeyError("mock"))
        monitor = StrategyHealthMonitor(se, em)
        monitor.on_bar(0)
        monitor.on_bar(1)
        monitor.on_bar(2)
        assert monitor.get_status().bars_processed == 3

    def test_on_trade_closed_increments_count(self):
        from src.monitoring.health_monitor import StrategyHealthMonitor
        se = MagicMock()
        em = MagicMock()
        em.get_edge = MagicMock(side_effect=KeyError("mock"))
        monitor = StrategyHealthMonitor(se, em)
        monitor.on_trade_closed(won=True)
        monitor.on_trade_closed(won=False)
        assert monitor.get_status().trades_closed == 2
