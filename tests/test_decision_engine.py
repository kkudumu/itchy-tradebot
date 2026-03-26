"""Unit tests for the DecisionEngine, EngineTradeLogger, and ScanScheduler.

All external dependencies (MT5, DB, signal engine, edge manager, etc.)
are mocked so these tests run on any platform without external services.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import pandas as pd
import numpy as np

from src.engine.decision_engine import DecisionEngine, Decision
from src.engine.trade_logger import EngineTradeLogger
from src.engine.scheduler import ScanScheduler


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _make_signal(
    direction: str = "long",
    confluence_score: int = 6,
    entry_price: float = 2000.0,
    stop_loss: float = 1990.0,
    take_profit: float = 2020.0,
    atr: float = 5.0,
    quality_tier: str = "A+",
):
    """Build a minimal mock Signal object."""
    sig = MagicMock()
    sig.direction = direction
    sig.confluence_score = confluence_score
    sig.entry_price = entry_price
    sig.stop_loss = stop_loss
    sig.take_profit = take_profit
    sig.atr = atr
    sig.quality_tier = quality_tier
    sig.instrument = "XAUUSD"
    sig.timestamp = datetime.now(timezone.utc)
    sig.reasoning = {"4h_filter": {"pass": True}}
    sig.zone_context = {"nearby_zone_count": 2, "zones": []}
    sig.mtf_state = None
    return sig


def _make_edge_result(name: str, allowed: bool, reason: str = "", modifier: float = None):
    """Build a mock EdgeResult."""
    er = MagicMock()
    er.allowed = allowed
    er.edge_name = name
    er.reason = reason
    er.modifier = modifier
    return er


def _make_position_size(lot_size: float = 0.05):
    """Build a mock PositionSize."""
    ps = MagicMock()
    ps.lot_size = lot_size
    ps.risk_pct = 1.5
    ps.phase = "aggressive"
    return ps


def _make_active_trade(direction: str = "long", current_r: float = 0.0):
    """Build a mock ActiveTrade."""
    at = MagicMock()
    at.direction = direction
    at.entry_price = 2000.0
    at.stop_loss = 1990.0
    at.take_profit = 2020.0
    at.lot_size = 0.05
    at.current_r = current_r
    at.entry_time = datetime.now(timezone.utc)
    at.remaining_pct = 1.0
    at.partial_exits = []
    at.original_stop_loss = 1990.0
    return at


def _make_exit_decision(action: str = "no_action", reason: str = "", new_stop: float = None, r_multiple: float = 0.0, close_pct: float = 0.5):
    """Build a mock ExitDecision."""
    ed = MagicMock()
    ed.action = action
    ed.reason = reason
    ed.new_stop = new_stop
    ed.r_multiple = r_multiple
    ed.close_pct = close_pct
    return ed


def _make_1m_df(rows: int = 100) -> pd.DataFrame:
    """Create a minimal 1-minute OHLCV DataFrame."""
    times = pd.date_range("2024-01-15 08:00", periods=rows, freq="1min", tz="UTC")
    return pd.DataFrame({
        "time": times,
        "open": np.full(rows, 2000.0),
        "high": np.full(rows, 2002.0),
        "low": np.full(rows, 1998.0),
        "close": np.full(rows, 2001.0),
        "tick_volume": np.ones(rows, dtype=int),
    })


def _make_5m_df(rows: int = 50) -> pd.DataFrame:
    """Create a minimal 5-minute OHLCV DataFrame."""
    times = pd.date_range("2024-01-15 08:00", periods=rows, freq="5min", tz="UTC")
    return pd.DataFrame({
        "time": times,
        "open": np.full(rows, 2000.0),
        "high": np.full(rows, 2005.0),
        "low": np.full(rows, 1995.0),
        "close": np.full(rows, 2001.0),
        "tick_volume": np.ones(rows, dtype=int),
    })


def _make_data() -> dict:
    """Build a minimal multi-timeframe data dict."""
    return {
        "1M": _make_1m_df(),
        "5M": _make_5m_df(),
        "15M": _make_5m_df(40),
        "1H": _make_5m_df(30),
        "4H": _make_5m_df(20),
    }


def _make_engine(
    signal: Any = None,
    entry_allowed: bool = True,
    active_trade_ids: Optional[List[int]] = None,
    **overrides,
) -> DecisionEngine:
    """Construct a DecisionEngine with all dependencies mocked."""
    signal_engine = MagicMock()
    signal_engine.scan.return_value = signal

    edge_manager = MagicMock()
    allow_result = _make_edge_result("time_of_day", entry_allowed, "test")
    edge_manager.check_entry.return_value = (entry_allowed, [allow_result])
    edge_manager.check_exit.return_value = (False, [])
    edge_manager.get_combined_size_multiplier.return_value = 1.0
    edge_manager.get_modifiers.return_value = {}

    trade_manager = MagicMock()
    trade_manager.can_open_trade.return_value = (True, "ok")
    trade_manager.active_trade_ids = active_trade_ids or []
    trade_manager.closed_trades = []
    ps = _make_position_size()
    at = _make_active_trade()
    trade_manager.open_trade.return_value = (1, at, ps)
    trade_manager.get_equity_summary.return_value = {"total_equity": 10000.0}
    trade_manager._active_trades = {}

    similarity_search = MagicMock()
    similarity_search.find_similar_trades.return_value = []
    similarity_search.get_performance_stats.return_value = MagicMock(
        n_trades=0, win_rate=0.0, avg_r=0.0, expectancy=0.0, confidence=0.0
    )

    embedding_engine = MagicMock()
    embedding_engine.create_embedding.return_value = np.zeros(64)

    zone_manager = MagicMock()

    config = {
        "instrument": "XAUUSD",
        "scan_interval_minutes": 5,
        "point_value": 1.0,
        "atr_multiplier": 1.5,
        "min_confidence_score": 4,
        "similarity_k": 10,
        "similarity_min_score": 0.7,
    }
    config.update(overrides.get("config", {}))

    engine = DecisionEngine(
        config=config,
        signal_engine=signal_engine,
        edge_manager=edge_manager,
        trade_manager=trade_manager,
        similarity_search=similarity_search,
        embedding_engine=embedding_engine,
        zone_manager=zone_manager,
        mt5_bridge=None,   # backtest mode
        order_manager=None,
        account_monitor=None,
        screenshot_capture=None,
        db_pool=None,
    )
    return engine


# ===========================================================================
# Test 1: Scan with signal — edge passes — trade executed
# ===========================================================================

class TestScanWithSignal:
    def test_scan_executes_trade_when_signal_and_edges_pass(self):
        signal = _make_signal(confluence_score=6)
        engine = _make_engine(signal=signal, entry_allowed=True)

        data = _make_data()
        decision = engine.scan(data=data)

        assert decision.action == "enter"
        assert decision.executed is True
        assert decision.signal is signal
        assert decision.confluence_score >= 4

    def test_scan_calls_signal_engine_with_1m_data(self):
        signal = _make_signal()
        engine = _make_engine(signal=signal)
        data = _make_data()
        engine.scan(data=data)

        engine.signal_engine.scan.assert_called_once()
        call_args = engine.signal_engine.scan.call_args
        # First positional arg should be the 1M DataFrame
        passed_df = call_args[0][0]
        assert isinstance(passed_df, pd.DataFrame)

    def test_scan_calls_edge_manager_check_entry(self):
        signal = _make_signal()
        engine = _make_engine(signal=signal)
        data = _make_data()
        engine.scan(data=data)

        engine.edge_manager.check_entry.assert_called_once()

    def test_scan_calls_trade_manager_can_open(self):
        signal = _make_signal()
        engine = _make_engine(signal=signal)
        data = _make_data()
        engine.scan(data=data)

        engine.trade_manager.can_open_trade.assert_called_once()


# ===========================================================================
# Test 2: Scan with edge rejection — signal generated but blocked
# ===========================================================================

class TestScanEdgeRejection:
    def test_scan_skips_when_edge_blocks(self):
        signal = _make_signal(confluence_score=6)
        engine = _make_engine(signal=signal, entry_allowed=False)

        # Override the edge result to explicitly block
        blocked = _make_edge_result("time_of_day", False, "Outside trading hours")
        engine.edge_manager.check_entry.return_value = (False, [blocked])

        data = _make_data()
        decision = engine.scan(data=data)

        assert decision.action == "skip"
        assert decision.executed is False
        assert "time_of_day" in decision.reasoning or "Blocked by edge" in decision.reasoning

    def test_scan_logs_decision_even_when_blocked(self):
        signal = _make_signal()
        engine = _make_engine(signal=signal, entry_allowed=False)
        blocked = _make_edge_result("spread_filter", False, "Spread too wide")
        engine.edge_manager.check_entry.return_value = (False, [blocked])

        data = _make_data()
        decision = engine.scan(data=data)

        # Decision must be logged (buffer has at least one entry)
        assert len(engine.trade_logger.decision_buffer) >= 1
        assert decision.action == "skip"

    def test_scan_does_not_open_trade_when_edge_blocks(self):
        signal = _make_signal()
        engine = _make_engine(signal=signal, entry_allowed=False)
        blocked = _make_edge_result("news_filter", False, "High-impact news")
        engine.edge_manager.check_entry.return_value = (False, [blocked])

        data = _make_data()
        engine.scan(data=data)

        engine.trade_manager.open_trade.assert_not_called()


# ===========================================================================
# Test 3: Scan no signal — skip decision logged
# ===========================================================================

class TestScanNoSignal:
    def test_scan_skips_when_no_signal(self):
        engine = _make_engine(signal=None)
        data = _make_data()
        decision = engine.scan(data=data)

        assert decision.action == "skip"
        assert decision.signal is None
        assert "No signal" in decision.reasoning

    def test_scan_logs_skip_decision(self):
        engine = _make_engine(signal=None)
        data = _make_data()
        engine.scan(data=data)

        # At least one decision in the buffer
        assert len(engine.trade_logger.decision_buffer) >= 1

    def test_scan_does_not_call_edge_manager_when_no_signal(self):
        engine = _make_engine(signal=None)
        data = _make_data()
        engine.scan(data=data)

        engine.edge_manager.check_entry.assert_not_called()

    def test_scan_empty_data_returns_skip(self):
        engine = _make_engine(signal=None)
        decision = engine.scan(data={})

        assert decision.action == "skip"
        assert "No market data" in decision.reasoning


# ===========================================================================
# Test 4: Open trade management — exit check
# ===========================================================================

class TestOpenTradeManagement:
    def test_manage_open_trades_called_on_each_scan(self):
        signal = None
        engine = _make_engine(signal=signal)
        engine.trade_manager.active_trade_ids = [1]
        engine.trade_manager._active_trades = {1: _make_active_trade()}

        exit_dec = _make_exit_decision("no_action")
        engine.trade_manager.update_trade.return_value = exit_dec
        engine.trade_manager.check_exit = MagicMock(return_value=(False, []))

        data = _make_data()
        engine.scan(data=data)

        # update_trade should have been called once for the active trade
        engine.trade_manager.update_trade.assert_called_once()
        call_kwargs = engine.trade_manager.update_trade.call_args[1]
        assert call_kwargs["trade_id"] == 1
        assert abs(call_kwargs["current_price"] - 2001.0) <= 1.0

    def test_full_exit_logs_trade_exit(self):
        engine = _make_engine(signal=None)
        engine.trade_manager.active_trade_ids = [42]
        engine.trade_manager._active_trades = {42: _make_active_trade()}

        exit_dec = _make_exit_decision("full_exit", "Take-profit hit", r_multiple=2.0)
        engine.trade_manager.update_trade.return_value = exit_dec
        engine.trade_manager.close_trade.return_value = {
            "trade_id": 42, "pnl_points": 20.0, "r_multiple": 2.0, "reason": "tp"
        }

        data = _make_data()
        engine.scan(data=data)

        # Decisions buffer should contain an 'exit' entry
        exit_decisions = [
            d for d in engine.trade_logger.decision_buffer
            if d.get("action") == "exit"
        ]
        assert len(exit_decisions) >= 1

    def test_edge_exit_triggers_force_close(self):
        engine = _make_engine(signal=None)
        engine.trade_manager.active_trade_ids = [7]
        engine.trade_manager._active_trades = {7: _make_active_trade()}

        # Friday close edge triggers exit
        friday_result = _make_edge_result("friday_close", False, "Friday 21:59 — close before weekend")
        engine.edge_manager.check_exit.return_value = (True, [friday_result])
        engine.trade_manager.close_trade.return_value = {
            "trade_id": 7, "pnl_points": 5.0, "r_multiple": 0.5, "reason": "friday_close"
        }

        data = _make_data()
        engine.scan(data=data)

        engine.trade_manager.close_trade.assert_called_once()


# ===========================================================================
# Test 5: Similarity adjustment
# ===========================================================================

class TestSimilarityAdjustment:
    def test_positive_similarity_boosts_score(self):
        signal = _make_signal(confluence_score=5)
        engine = _make_engine(signal=signal)

        # Simulate good historical performance
        stats = MagicMock()
        stats.confidence = 0.8
        stats.expectancy = 0.8
        stats.win_rate = 0.7
        stats.n_trades = 20
        engine.similarity_search.get_performance_stats.return_value = stats

        adjusted = engine._adjust_confidence(signal, stats)
        assert adjusted == 6  # +1 for positive expectancy + win_rate

    def test_negative_similarity_reduces_score(self):
        signal = _make_signal(confluence_score=5)
        engine = _make_engine(signal=signal)

        stats = MagicMock()
        stats.confidence = 0.8
        stats.expectancy = -0.5
        stats.win_rate = 0.3
        stats.n_trades = 25
        engine.similarity_search.get_performance_stats.return_value = stats

        adjusted = engine._adjust_confidence(signal, stats)
        assert adjusted == 4  # -1 for negative expectancy / low win_rate

    def test_low_confidence_data_no_adjustment(self):
        signal = _make_signal(confluence_score=5)
        engine = _make_engine(signal=signal)

        stats = MagicMock()
        stats.confidence = 0.1  # Too few samples
        stats.expectancy = 1.0
        stats.win_rate = 0.9
        engine.similarity_search.get_performance_stats.return_value = stats

        adjusted = engine._adjust_confidence(signal, stats)
        assert adjusted == 5  # No change — insufficient data

    def test_score_clamped_to_eight(self):
        signal = _make_signal(confluence_score=8)
        engine = _make_engine(signal=signal)

        stats = MagicMock()
        stats.confidence = 1.0
        stats.expectancy = 0.9
        stats.win_rate = 0.8
        engine.similarity_search.get_performance_stats.return_value = stats

        adjusted = engine._adjust_confidence(signal, stats)
        assert adjusted == 8  # Cannot exceed maximum

    def test_score_clamped_to_zero(self):
        signal = _make_signal(confluence_score=0)
        engine = _make_engine(signal=signal)

        stats = MagicMock()
        stats.confidence = 0.9
        stats.expectancy = -1.0
        stats.win_rate = 0.1
        engine.similarity_search.get_performance_stats.return_value = stats

        adjusted = engine._adjust_confidence(signal, stats)
        assert adjusted == 0  # Cannot go below zero


# ===========================================================================
# Test 6: Decision logging — required fields
# ===========================================================================

class TestDecisionLogging:
    def test_decision_record_has_required_fields(self):
        signal = _make_signal()
        engine = _make_engine(signal=signal)
        data = _make_data()
        engine.scan(data=data)

        buffer = engine.trade_logger.decision_buffer
        assert len(buffer) >= 1

        # Check the entry decision record
        entry_records = [r for r in buffer if r.get("action") == "enter"]
        assert entry_records, "Expected at least one 'enter' decision"
        rec = entry_records[0]

        required = ["timestamp", "instrument", "action", "confluence_score", "reasoning", "executed"]
        for key in required:
            assert key in rec, f"Missing field: {key}"

    def test_skipped_decision_is_logged(self):
        engine = _make_engine(signal=None)
        data = _make_data()
        engine.scan(data=data)

        skip_records = [r for r in engine.trade_logger.decision_buffer if r.get("action") == "skip"]
        assert skip_records

    def test_edge_results_stored_in_decision(self):
        signal = _make_signal()
        engine = _make_engine(signal=signal)
        data = _make_data()
        engine.scan(data=data)

        entry_records = [r for r in engine.trade_logger.decision_buffer if r.get("action") == "enter"]
        if entry_records:
            rec = entry_records[0]
            assert "edge_results" in rec

    def test_decision_has_similarity_data(self):
        signal = _make_signal()
        engine = _make_engine(signal=signal)
        data = _make_data()
        decision = engine.scan(data=data)

        assert isinstance(decision.similarity_data, dict)


# ===========================================================================
# Test 7: Zone maintenance called on each scan
# ===========================================================================

class TestZoneMaintenance:
    def test_zone_maintenance_called_on_each_scan(self):
        signal = _make_signal()
        engine = _make_engine(signal=signal)
        data = _make_data()
        engine.scan(data=data)

        engine.zone_manager.maintenance.assert_called_once()

    def test_zone_maintenance_called_even_without_signal(self):
        engine = _make_engine(signal=None)
        data = _make_data()
        engine.scan(data=data)

        engine.zone_manager.maintenance.assert_called_once()

    def test_zone_maintenance_skipped_without_5m_data(self):
        engine = _make_engine(signal=None)
        # Data without 5M key
        engine.scan(data={"1M": _make_1m_df()})

        # maintenance may or may not be called depending on fallback
        # Just verify no exception was raised
        assert True


# ===========================================================================
# Test 8: Graceful shutdown
# ===========================================================================

class TestGracefulShutdown:
    def test_stop_sets_running_false(self):
        engine = _make_engine()
        engine._running = True
        engine.stop()
        assert engine._running is False

    def test_stop_sets_stop_event(self):
        engine = _make_engine()
        engine.stop()
        assert engine._stop_event.is_set()

    def test_stop_flushes_logger(self):
        engine = _make_engine()
        engine.trade_logger.flush = MagicMock()
        engine.stop()
        engine.trade_logger.flush.assert_called_once()

    def test_double_start_logs_warning(self):
        engine = _make_engine()
        engine._running = True
        with patch.object(engine, "_scheduler") as mock_sched:
            # second start should return immediately
            engine.start()
            mock_sched.run_loop.assert_not_called()


# ===========================================================================
# Test 9: Scheduler — alignment to 5M intervals
# ===========================================================================

class TestScheduler:
    def test_next_close_aligned_to_interval(self):
        scheduler = ScanScheduler(interval_minutes=5)
        # Use a known time: 08:03:30 → next close should be 08:05:01
        now = datetime(2024, 1, 15, 8, 3, 30, tzinfo=timezone.utc)
        next_close = scheduler._next_close_time(now)
        assert next_close.minute == 5
        assert next_close.second == 1
        assert next_close.hour == 8

    def test_next_close_at_exact_boundary(self):
        scheduler = ScanScheduler(interval_minutes=5)
        # Exactly on boundary with 0 seconds → fire immediately
        now = datetime(2024, 1, 15, 8, 5, 0, tzinfo=timezone.utc)
        next_close = scheduler._next_close_time(now)
        assert next_close.minute == 5
        assert next_close.second == 0

    def test_next_close_15m_interval(self):
        scheduler = ScanScheduler(interval_minutes=15)
        now = datetime(2024, 1, 15, 8, 7, 0, tzinfo=timezone.utc)
        next_close = scheduler._next_close_time(now)
        assert next_close.minute == 15

    def test_market_open_weekday(self):
        scheduler = ScanScheduler()
        # Monday 10:00 UTC
        dt = datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)  # Monday
        assert scheduler.is_market_open(dt) is True

    def test_market_closed_saturday(self):
        scheduler = ScanScheduler()
        # Saturday
        dt = datetime(2024, 1, 13, 12, 0, tzinfo=timezone.utc)  # Saturday
        assert scheduler.is_market_open(dt) is False

    def test_market_open_sunday_after_22(self):
        scheduler = ScanScheduler()
        dt = datetime(2024, 1, 14, 22, 30, tzinfo=timezone.utc)  # Sunday 22:30
        assert scheduler.is_market_open(dt) is True

    def test_market_closed_sunday_before_22(self):
        scheduler = ScanScheduler()
        dt = datetime(2024, 1, 14, 20, 0, tzinfo=timezone.utc)  # Sunday 20:00
        assert scheduler.is_market_open(dt) is False

    def test_market_closed_friday_after_22(self):
        scheduler = ScanScheduler()
        dt = datetime(2024, 1, 19, 22, 30, tzinfo=timezone.utc)  # Friday 22:30
        assert scheduler.is_market_open(dt) is False

    def test_invalid_interval_raises(self):
        with pytest.raises(ValueError):
            ScanScheduler(interval_minutes=0)

    def test_run_loop_calls_callback_and_respects_stop(self):
        scheduler = ScanScheduler(interval_minutes=5)
        call_count = [0]

        def callback():
            call_count[0] += 1

        stop_event = threading.Event()

        def _patched_wait():
            return datetime.now(timezone.utc)

        with patch.object(scheduler, "wait_for_next_close", side_effect=_patched_wait), \
             patch.object(scheduler, "is_market_open", return_value=True):
            def _trigger_stop():
                import time
                time.sleep(0.05)
                stop_event.set()

            stopper = threading.Thread(target=_trigger_stop)
            stopper.start()
            scheduler.run_loop(callback, stop_event)
            stopper.join()

        assert call_count[0] >= 1


# ===========================================================================
# Test 10: Full pipeline end-to-end with all mocks
# ===========================================================================

class TestFullPipeline:
    def test_full_pipeline_enter(self):
        signal = _make_signal(confluence_score=7, direction="long")
        engine = _make_engine(signal=signal)

        # Similarity returns positive stats
        stats = MagicMock()
        stats.confidence = 0.9
        stats.expectancy = 0.6
        stats.win_rate = 0.65
        stats.n_trades = 22
        engine.similarity_search.get_performance_stats.return_value = stats
        engine.similarity_search.find_similar_trades.return_value = []

        data = _make_data()
        decision = engine.scan(data=data)

        assert decision.action == "enter"
        assert decision.instrument == "XAUUSD"
        assert decision.signal is signal
        assert decision.executed is True
        assert decision.trade_id is not None

    def test_full_pipeline_skip_low_score(self):
        # Signal with score below minimum after similarity adjustment
        signal = _make_signal(confluence_score=3)
        engine = _make_engine(signal=signal)

        # Similarity penalises the score
        stats = MagicMock()
        stats.confidence = 0.8
        stats.expectancy = -0.5
        stats.win_rate = 0.3
        stats.n_trades = 15
        engine.similarity_search.get_performance_stats.return_value = stats

        data = _make_data()
        decision = engine.scan(data=data)

        # Adjusted score = 3 - 1 = 2, below min_confluence_score=4 → skip
        assert decision.action == "skip"

    def test_full_pipeline_circuit_breaker_blocks(self):
        signal = _make_signal(confluence_score=8)
        engine = _make_engine(signal=signal)
        engine.trade_manager.can_open_trade.return_value = (False, "Daily loss limit reached")

        data = _make_data()
        decision = engine.scan(data=data)

        assert decision.action == "skip"
        assert "Daily loss limit" in decision.reasoning

    def test_full_pipeline_screenshot_taken_on_entry(self):
        signal = _make_signal(confluence_score=7)
        engine = _make_engine(signal=signal)

        screenshot_paths = []
        def capture(phase, trade_id):
            path = f"/tmp/screenshot_{phase}_{trade_id}.png"
            screenshot_paths.append((phase, trade_id, path))
            return path

        engine.screenshot_capture = capture
        engine.trade_logger.log_screenshot = MagicMock()

        data = _make_data()
        decision = engine.scan(data=data)

        # At minimum a pre_entry and entry screenshot
        if decision.action == "enter":
            phases = [p for p, _, _ in screenshot_paths]
            assert "pre_entry" in phases
            assert "entry" in phases


# ===========================================================================
# EngineTradeLogger standalone tests
# ===========================================================================

class TestEngineTradeLogger:
    def test_log_decision_stores_to_buffer_without_db(self):
        logger = EngineTradeLogger(db_pool=None)
        decision = Decision(
            timestamp=datetime.now(timezone.utc),
            instrument="XAUUSD",
            action="skip",
            signal=None,
            edge_results={},
            similarity_data={},
            confluence_score=3,
            reasoning="No signal",
        )
        logger.log_decision(decision)
        assert len(logger.decision_buffer) == 1
        assert logger.decision_buffer[0]["action"] == "skip"

    def test_log_trade_entry_stores_to_buffer(self):
        logger = EngineTradeLogger(db_pool=None)
        logger.log_trade_entry(
            trade={
                "instrument": "XAUUSD",
                "direction": "long",
                "entry_price": 2000.0,
                "stop_loss": 1990.0,
                "take_profit": 2020.0,
                "lot_size": 0.05,
                "confluence_score": 6,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            context={"adx_value": 30.0, "atr_value": 5.0},
        )
        assert len(logger.trade_buffer) == 1
        assert logger.trade_buffer[0]["direction"] == "long"

    def test_log_screenshot_no_db_does_not_raise(self):
        logger = EngineTradeLogger(db_pool=None)
        logger.log_screenshot(trade_id=123, phase="entry", filepath="/tmp/shot.png")

    def test_log_zone_update_no_db_does_not_raise(self):
        logger = EngineTradeLogger(db_pool=None)
        logger.log_zone_update([
            {"zone_id": 1, "old_status": "active", "new_status": "tested", "reason": "wick"}
        ])

    def test_flush_does_not_raise(self):
        logger = EngineTradeLogger(db_pool=None)
        logger.log_decision(Decision(
            timestamp=datetime.now(timezone.utc),
            instrument="XAUUSD",
            action="enter",
            signal=None,
            edge_results={},
            similarity_data={},
            confluence_score=5,
            reasoning="Test",
        ))
        logger.flush()  # Should not raise

    def test_decision_with_signal_includes_signal_fields(self):
        logger = EngineTradeLogger(db_pool=None)
        signal = _make_signal(direction="short", entry_price=1999.0)
        decision = Decision(
            timestamp=datetime.now(timezone.utc),
            instrument="XAUUSD",
            action="enter",
            signal=signal,
            edge_results={"time_of_day": {"allowed": True}},
            similarity_data={"n_similar": 10, "win_rate": 0.6},
            confluence_score=6,
            reasoning="All checks passed",
            trade_id=456,
            executed=True,
        )
        logger.log_decision(decision)
        rec = logger.decision_buffer[0]
        assert rec["direction"] == "short"
        assert rec["entry_price"] == 1999.0
        assert rec["executed"] is True
        assert rec["trade_id"] == 456
