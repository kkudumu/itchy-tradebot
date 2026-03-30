"""
Tests for PreFlightDiagnostic.

Covers:
1. Basic functionality — run() returns PreFlightResult with all fields
2. Sampling — _sample_data() takes middle slice, handles small datasets
3. Signal found on first scan — no relaxation, passed=True
4. No signal, relaxation fixes it — signals appear after tier 1 or 2
5. No signal, all tiers exhausted — aborted=True
6. Funnel report — contains meaningful diagnostic data
7. Small/edge-case datasets — < sample_size, empty, single bar
8. Edge cases — warmup-only data, very short windows
9. Message field — correct human-readable summary per outcome
10. Relaxation tier tracking
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from unittest.mock import MagicMock, patch, call
import pytest
import pandas as pd

from src.edges.manager import EdgeManager
from src.monitoring.pre_flight import PreFlightDiagnostic, PreFlightResult
from src.strategy.signal_engine import Signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_candles(n: int = 6000) -> pd.DataFrame:
    """Build a minimal 1-minute OHLCV DataFrame of length n."""
    dates = pd.date_range("2024-01-01", periods=n, freq="1min")
    return pd.DataFrame(
        {
            "open": 2000.0,
            "high": 2001.0,
            "low": 1999.0,
            "close": 2000.0,
            "volume": 100,
        },
        index=dates,
    )


def make_signal(ts: Optional[datetime] = None) -> Signal:
    """Return a minimal valid Signal."""
    return Signal(
        timestamp=ts or datetime(2024, 1, 2, tzinfo=timezone.utc),
        instrument="XAUUSD",
        direction="long",
        entry_price=2000.0,
        stop_loss=1990.0,
        take_profit=2020.0,
        confluence_score=5,
        quality_tier="B",
        atr=10.0,
    )


def minimal_edge_config() -> dict:
    """EdgeManager config that enables the edges AdaptiveRelaxer needs."""
    return {
        "time_of_day": {
            "enabled": True,
            "params": {"start_utc": "08:00", "end_utc": "17:00"},
        },
        "day_of_week": {
            "enabled": True,
            "params": {"allowed_days": [1, 2, 3]},
        },
        "london_open_delay": {
            "enabled": True,
            "params": {"london_open_utc": "08:00", "delay_minutes": 30},
        },
        "regime_filter": {
            "enabled": True,
            "params": {"adx_min": 28, "cloud_thickness_percentile": 5.0},
        },
        "confluence_scoring": {
            "enabled": True,
            "params": {"min_score": 4},
        },
        "candle_close_confirmation": {"enabled": False, "params": {}},
        "spread_filter": {"enabled": False, "params": {}},
        "news_filter": {"enabled": False, "params": {}},
        "friday_close": {"enabled": False, "params": {}},
        "time_stop": {"enabled": False, "params": {}},
        "bb_squeeze": {"enabled": False, "params": {}},
        "equity_curve": {"enabled": False, "params": {}},
    }


class MockSignalEngine:
    """Controllable signal engine for testing.

    Parameters
    ----------
    signal_at_bars:
        Set of scan_count values at which to return a Signal.
        scan_count increments on every call to scan().
    """

    def __init__(self, signal_at_bars: Optional[set] = None) -> None:
        self.signal_at_bars: set = signal_at_bars or set()
        self.scan_count: int = 0

    def scan(self, data_1m: pd.DataFrame, current_bar: int = -1) -> Optional[Signal]:
        self.scan_count += 1
        if self.scan_count in self.signal_at_bars:
            return make_signal(ts=data_1m.index[-1])
        return None


class AlwaysSignalEngine:
    """Engine that always returns a Signal (after warmup)."""

    def scan(self, data_1m: pd.DataFrame, current_bar: int = -1) -> Optional[Signal]:
        return make_signal()


class NeverSignalEngine:
    """Engine that never returns a Signal."""

    def scan(self, data_1m: pd.DataFrame, current_bar: int = -1) -> Optional[Signal]:
        return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def edge_manager() -> EdgeManager:
    return EdgeManager(minimal_edge_config())


@pytest.fixture
def candles_6k() -> pd.DataFrame:
    return make_candles(6000)


@pytest.fixture
def candles_small() -> pd.DataFrame:
    return make_candles(1000)


# ---------------------------------------------------------------------------
# 1. Basic functionality
# ---------------------------------------------------------------------------


class TestPreFlightResultFields:
    """PreFlightResult has all required fields with correct types."""

    def test_result_has_passed_field(self, edge_manager, candles_6k):
        engine = AlwaysSignalEngine()
        pf = PreFlightDiagnostic(engine, edge_manager)
        result = pf.run(candles_6k)
        assert isinstance(result.passed, bool)

    def test_result_has_signals_found(self, edge_manager, candles_6k):
        engine = AlwaysSignalEngine()
        pf = PreFlightDiagnostic(engine, edge_manager)
        result = pf.run(candles_6k)
        assert isinstance(result.signals_found, int)
        assert result.signals_found >= 0

    def test_result_has_sample_size(self, edge_manager, candles_6k):
        engine = AlwaysSignalEngine()
        pf = PreFlightDiagnostic(engine, edge_manager)
        result = pf.run(candles_6k)
        assert isinstance(result.sample_size, int)
        assert result.sample_size > 0

    def test_result_has_relaxation_applied(self, edge_manager, candles_6k):
        engine = AlwaysSignalEngine()
        pf = PreFlightDiagnostic(engine, edge_manager)
        result = pf.run(candles_6k)
        assert isinstance(result.relaxation_applied, bool)

    def test_result_has_relaxation_tier(self, edge_manager, candles_6k):
        engine = AlwaysSignalEngine()
        pf = PreFlightDiagnostic(engine, edge_manager)
        result = pf.run(candles_6k)
        assert isinstance(result.relaxation_tier, int)

    def test_result_has_funnel_report(self, edge_manager, candles_6k):
        engine = AlwaysSignalEngine()
        pf = PreFlightDiagnostic(engine, edge_manager)
        result = pf.run(candles_6k)
        assert isinstance(result.funnel_report, dict)

    def test_result_has_aborted(self, edge_manager, candles_6k):
        engine = AlwaysSignalEngine()
        pf = PreFlightDiagnostic(engine, edge_manager)
        result = pf.run(candles_6k)
        assert isinstance(result.aborted, bool)

    def test_result_has_message(self, edge_manager, candles_6k):
        engine = AlwaysSignalEngine()
        pf = PreFlightDiagnostic(engine, edge_manager)
        result = pf.run(candles_6k)
        assert isinstance(result.message, str)
        assert len(result.message) > 0


# ---------------------------------------------------------------------------
# 2. Sampling
# ---------------------------------------------------------------------------


class TestSampleData:
    """_sample_data() extracts the right slice."""

    def test_large_dataset_returns_sample_size_bars(self, edge_manager):
        pf = PreFlightDiagnostic(NeverSignalEngine(), edge_manager, sample_size=1000)
        candles = make_candles(10000)
        sample = pf._sample_data(candles)
        assert len(sample) == 1000

    def test_sample_starts_at_25_percent(self, edge_manager):
        pf = PreFlightDiagnostic(NeverSignalEngine(), edge_manager, sample_size=1000)
        candles = make_candles(10000)
        sample = pf._sample_data(candles)
        # Should start at index 2500 (25% of 10000)
        assert sample.index[0] == candles.index[2500]

    def test_small_dataset_returns_everything(self, edge_manager):
        pf = PreFlightDiagnostic(NeverSignalEngine(), edge_manager, sample_size=5000)
        candles = make_candles(2000)
        sample = pf._sample_data(candles)
        assert len(sample) == 2000

    def test_dataset_equal_to_sample_size_returns_everything(self, edge_manager):
        pf = PreFlightDiagnostic(NeverSignalEngine(), edge_manager, sample_size=5000)
        candles = make_candles(5000)
        sample = pf._sample_data(candles)
        assert len(sample) == 5000

    def test_sample_end_does_not_exceed_total(self, edge_manager):
        """When 25%+sample_size > total, slice is capped at the end."""
        pf = PreFlightDiagnostic(NeverSignalEngine(), edge_manager, sample_size=5000)
        candles = make_candles(6000)
        sample = pf._sample_data(candles)
        # start = 6000//4 = 1500, end = min(1500+5000, 6000) = 6000 → 4500 bars
        assert len(sample) == min(5000, 6000 - 6000 // 4)

    def test_sample_is_contiguous_slice(self, edge_manager):
        pf = PreFlightDiagnostic(NeverSignalEngine(), edge_manager, sample_size=500)
        candles = make_candles(4000)
        sample = pf._sample_data(candles)
        start_idx = 4000 // 4
        assert sample.index[0] == candles.index[start_idx]
        assert sample.index[-1] == candles.index[start_idx + 499]


# ---------------------------------------------------------------------------
# 3. Signal found on first scan
# ---------------------------------------------------------------------------


class TestSignalFoundOnFirstScan:
    """When signals exist on the initial scan, no relaxation is needed."""

    def test_passed_is_true(self, edge_manager, candles_6k):
        engine = AlwaysSignalEngine()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        assert result.passed is True

    def test_relaxation_applied_is_false(self, edge_manager, candles_6k):
        engine = AlwaysSignalEngine()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        assert result.relaxation_applied is False

    def test_relaxation_tier_is_zero(self, edge_manager, candles_6k):
        engine = AlwaysSignalEngine()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        assert result.relaxation_tier == 0

    def test_aborted_is_false(self, edge_manager, candles_6k):
        engine = AlwaysSignalEngine()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        assert result.aborted is False

    def test_signals_found_positive(self, edge_manager, candles_6k):
        engine = AlwaysSignalEngine()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        assert result.signals_found > 0

    def test_message_mentions_no_relaxation(self, edge_manager, candles_6k):
        engine = AlwaysSignalEngine()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        assert "No relaxation" in result.message

    def test_sample_size_set_correctly(self, edge_manager, candles_6k):
        engine = AlwaysSignalEngine()
        pf = PreFlightDiagnostic(engine, edge_manager, sample_size=5000)
        result = pf.run(candles_6k)
        assert result.sample_size > 0

    def test_signal_found_on_specific_bar(self, edge_manager, candles_6k):
        """MockSignalEngine generates a signal at scan_count=5."""
        engine = MockSignalEngine(signal_at_bars={5})
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        assert result.passed is True
        assert result.signals_found >= 1
        assert result.relaxation_applied is False


# ---------------------------------------------------------------------------
# 4. No signal, relaxation fixes it
# ---------------------------------------------------------------------------


class TestRelaxationFixesZeroSignals:
    """When the initial scan returns nothing, auto-relax should find signals."""

    def _make_engine_that_signals_after_relax(self):
        """Engine that returns None for all initial-scan calls, then returns Signal
        during the first relaxation re-scan.

        With sample_size=5000 and 6k candles: sample = 4500 bars, step=5,
        initial scan runs at most 4500//5 = 900 calls (minus warmup ~60).
        Setting threshold at 1000 ensures the signal only appears in the
        re-scan that happens after the first tier is relaxed.
        """

        class _Engine:
            def __init__(self):
                self.calls = 0

            def scan(self, data_1m, current_bar=-1):
                self.calls += 1
                if self.calls > 1000:
                    return make_signal()
                return None

        return _Engine()

    def test_relaxation_applied_is_true(self, edge_manager, candles_6k):
        engine = self._make_engine_that_signals_after_relax()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        assert result.relaxation_applied is True

    def test_passed_is_true_after_relaxation(self, edge_manager, candles_6k):
        engine = self._make_engine_that_signals_after_relax()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        assert result.passed is True

    def test_aborted_is_false_after_relaxation(self, edge_manager, candles_6k):
        engine = self._make_engine_that_signals_after_relax()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        assert result.aborted is False

    def test_relaxation_tier_positive(self, edge_manager, candles_6k):
        engine = self._make_engine_that_signals_after_relax()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        assert result.relaxation_tier >= 1

    def test_signals_found_positive_after_relax(self, edge_manager, candles_6k):
        engine = self._make_engine_that_signals_after_relax()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        assert result.signals_found > 0

    def test_message_mentions_relaxation(self, edge_manager, candles_6k):
        engine = self._make_engine_that_signals_after_relax()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        assert "relaxation" in result.message.lower()


# ---------------------------------------------------------------------------
# 5. No signal, all tiers exhausted → aborted
# ---------------------------------------------------------------------------


class TestAllTiersExhausted:
    """When even max relaxation finds nothing, the result is aborted."""

    def test_aborted_is_true(self, edge_manager, candles_6k):
        engine = NeverSignalEngine()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        assert result.aborted is True

    def test_passed_is_false(self, edge_manager, candles_6k):
        engine = NeverSignalEngine()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        assert result.passed is False

    def test_signals_found_is_zero(self, edge_manager, candles_6k):
        engine = NeverSignalEngine()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        assert result.signals_found == 0

    def test_relaxation_applied_is_true(self, edge_manager, candles_6k):
        engine = NeverSignalEngine()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        assert result.relaxation_applied is True

    def test_message_mentions_aborted(self, edge_manager, candles_6k):
        engine = NeverSignalEngine()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        assert "ABORTED" in result.message or "aborted" in result.message.lower()

    def test_message_mentions_bottleneck(self, edge_manager, candles_6k):
        engine = NeverSignalEngine()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        # NeverSignalEngine always fails full_pipeline → bottleneck = 'full_pipeline'
        assert "bottleneck" in result.message.lower() or "filter" in result.message.lower()


# ---------------------------------------------------------------------------
# 6. Funnel report
# ---------------------------------------------------------------------------


class TestFunnelReport:
    """Funnel report contains meaningful diagnostic data."""

    def test_funnel_report_has_total_bars(self, edge_manager, candles_6k):
        engine = AlwaysSignalEngine()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        assert "total_bars" in result.funnel_report
        assert result.funnel_report["total_bars"] > 0

    def test_funnel_report_has_total_signals(self, edge_manager, candles_6k):
        engine = AlwaysSignalEngine()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        assert "total_signals" in result.funnel_report
        assert result.funnel_report["total_signals"] >= 0

    def test_funnel_report_has_filters(self, edge_manager, candles_6k):
        engine = AlwaysSignalEngine()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        assert "filters" in result.funnel_report
        assert isinstance(result.funnel_report["filters"], dict)

    def test_funnel_report_full_pipeline_present(self, edge_manager, candles_6k):
        engine = AlwaysSignalEngine()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        filters = result.funnel_report["filters"]
        assert "full_pipeline" in filters

    def test_funnel_report_pass_rate_between_0_and_1(self, edge_manager, candles_6k):
        engine = AlwaysSignalEngine()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        fp = result.funnel_report["filters"].get("full_pipeline", {})
        if fp:
            assert 0.0 <= fp["pass_rate"] <= 1.0

    def test_funnel_report_has_is_drought_500(self, edge_manager, candles_6k):
        engine = AlwaysSignalEngine()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        assert "is_drought_500" in result.funnel_report

    def test_funnel_report_not_drought_when_signals_exist(self, edge_manager, candles_6k):
        engine = AlwaysSignalEngine()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        assert result.funnel_report["is_drought_500"] is False

    def test_funnel_report_has_bottleneck_key(self, edge_manager, candles_6k):
        engine = NeverSignalEngine()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        assert "bottleneck" in result.funnel_report

    def test_funnel_report_bottleneck_is_full_pipeline_on_never_engine(
        self, edge_manager, candles_6k
    ):
        engine = NeverSignalEngine()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        # Only filter recorded is 'full_pipeline', so it must be the bottleneck
        assert result.funnel_report["bottleneck"] == "full_pipeline"


# ---------------------------------------------------------------------------
# 7. Small / edge-case datasets
# ---------------------------------------------------------------------------


class TestEdgeCaseDatasets:
    """Handles unusual dataset sizes gracefully."""

    def test_empty_dataframe_returns_aborted_result(self, edge_manager):
        engine = AlwaysSignalEngine()
        result = PreFlightDiagnostic(engine, edge_manager).run(pd.DataFrame())
        assert result.aborted is True
        assert result.passed is False
        assert result.sample_size == 0

    def test_empty_dataframe_message_mentions_empty(self, edge_manager):
        engine = AlwaysSignalEngine()
        result = PreFlightDiagnostic(engine, edge_manager).run(pd.DataFrame())
        assert "empty" in result.message.lower()

    def test_dataset_smaller_than_sample_size(self, edge_manager):
        """Dataset < sample_size → use entire dataset."""
        engine = AlwaysSignalEngine()
        candles = make_candles(800)
        result = PreFlightDiagnostic(engine, edge_manager, sample_size=5000).run(candles)
        assert result.sample_size == 800

    def test_warmup_only_dataset_no_signals(self, edge_manager):
        """Dataset too small to pass the 300-bar warmup → zero signals."""
        engine = AlwaysSignalEngine()
        candles = make_candles(200)  # Every window will be < 300 bars
        result = PreFlightDiagnostic(engine, edge_manager).run(candles)
        # Scan loop skips everything below warmup → zero signals → relaxation path
        assert result.signals_found == 0

    def test_single_bar_dataset(self, edge_manager):
        engine = AlwaysSignalEngine()
        candles = make_candles(1)
        result = PreFlightDiagnostic(engine, edge_manager).run(candles)
        assert result.signals_found == 0
        assert result.passed is False

    def test_exactly_warmup_bars(self, edge_manager):
        """Exactly 300-bar dataset: first window passes warmup check."""
        engine = AlwaysSignalEngine()
        candles = make_candles(300)
        pf = PreFlightDiagnostic(engine, edge_manager, sample_size=300)
        result = pf.run(candles)
        # Window of 300 bars at bar_idx=0 (step=5) → window[:1] = 1 bar → below 300
        # Actually the scan skips bar_idx 0..295 due to warmup, hits bar_idx 299
        # With step=5, bar_idx goes 0,5,...295 → windows 1..296 bars → all <300
        assert isinstance(result, PreFlightResult)


# ---------------------------------------------------------------------------
# 8. Scan step logic
# ---------------------------------------------------------------------------


class TestScanSampleLogic:
    """Verify _scan_sample() advances in 5-bar steps and skips warmup."""

    def test_scan_skips_below_warmup_bars(self, edge_manager):
        """Scan counts should be zero for tiny windows."""
        engine = MockSignalEngine(signal_at_bars=set(range(1, 100)))
        candles = make_candles(200)
        pf = PreFlightDiagnostic(engine, edge_manager)
        sample = pf._sample_data(candles)
        from src.monitoring.funnel_tracker import SignalFunnelTracker
        tracker = SignalFunnelTracker()
        count = pf._scan_sample(sample, tracker)
        # No window of 200 bars (step=5) will reach 300, so scan_count stays 0
        assert engine.scan_count == 0
        assert count == 0

    def test_scan_calls_engine_for_each_5m_bar(self, edge_manager):
        """scan() is called once per 5-bar step (minus warmup bars)."""
        engine = NeverSignalEngine()
        candles = make_candles(2000)
        pf = PreFlightDiagnostic(engine, edge_manager, sample_size=2000)
        sample = pf._sample_data(candles)

        original_scan = engine.scan
        call_count = [0]

        def counting_scan(data_1m, current_bar=-1):
            call_count[0] += 1
            return original_scan(data_1m, current_bar)

        engine.scan = counting_scan

        from src.monitoring.funnel_tracker import SignalFunnelTracker
        tracker = SignalFunnelTracker()
        pf._scan_sample(sample, tracker)

        # Expected: (2000 // 5) steps, minus those where window < 300
        # step=5: bar_idx = 0, 5, ..., 1995 → 400 iterations
        # first call where len(window)>=300: bar_idx=299 → but step is 5 so bar_idx=300 (bar 300+1)
        # Actually bar_idx=295 → window[:296] = 296 bars < 300 → skip
        # bar_idx=300 → window[:301] = 301 bars ≥ 300 → call
        warmup = 300
        expected_calls = sum(
            1
            for bar_idx in range(0, 2000, 5)
            if bar_idx + 1 >= warmup
        )
        assert call_count[0] == expected_calls


# ---------------------------------------------------------------------------
# 9. Auto-relaxation loop internals
# ---------------------------------------------------------------------------


class TestAutoRelaxLoop:
    """_auto_relax_loop returns correct (signals, tier, aborted, relaxed_config) tuples."""

    def test_returns_tuple_of_three(self, edge_manager, candles_6k):
        engine = NeverSignalEngine()
        pf = PreFlightDiagnostic(engine, edge_manager)
        sample = pf._sample_data(candles_6k)
        from src.monitoring.funnel_tracker import SignalFunnelTracker
        tracker = SignalFunnelTracker()
        result = pf._auto_relax_loop(sample, tracker)
        assert len(result) == 4

    def test_aborted_when_never_engine(self, edge_manager, candles_6k):
        engine = NeverSignalEngine()
        pf = PreFlightDiagnostic(engine, edge_manager)
        sample = pf._sample_data(candles_6k)
        from src.monitoring.funnel_tracker import SignalFunnelTracker
        tracker = SignalFunnelTracker()
        signals, tier, aborted, relaxed_cfg = pf._auto_relax_loop(sample, tracker)
        assert signals == 0
        assert aborted is True
        assert relaxed_cfg is None

    def test_not_aborted_when_engine_signals_eventually(self, edge_manager, candles_6k):
        """Engine that starts signalling after 200 calls should succeed."""

        class _DelayedEngine:
            def __init__(self):
                self.calls = 0

            def scan(self, data_1m, current_bar=-1):
                self.calls += 1
                if self.calls > 200:
                    return make_signal()
                return None

        pf = PreFlightDiagnostic(_DelayedEngine(), edge_manager)
        sample = pf._sample_data(candles_6k)
        from src.monitoring.funnel_tracker import SignalFunnelTracker
        tracker = SignalFunnelTracker()
        signals, tier, aborted, relaxed_cfg = pf._auto_relax_loop(sample, tracker)
        assert aborted is False
        assert signals > 0
        assert tier >= 1
        assert relaxed_cfg is not None


# ---------------------------------------------------------------------------
# 10. run() integration — consistency checks
# ---------------------------------------------------------------------------


class TestRunIntegration:
    """End-to-end consistency checks on run() output."""

    def test_passed_true_implies_signals_positive(self, edge_manager, candles_6k):
        engine = AlwaysSignalEngine()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        if result.passed:
            assert result.signals_found > 0

    def test_aborted_true_implies_passed_false(self, edge_manager, candles_6k):
        engine = NeverSignalEngine()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        if result.aborted:
            assert result.passed is False

    def test_no_relaxation_implies_tier_zero(self, edge_manager, candles_6k):
        engine = AlwaysSignalEngine()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        if not result.relaxation_applied:
            assert result.relaxation_tier == 0

    def test_sample_size_respects_constructor_param(self, edge_manager, candles_6k):
        engine = NeverSignalEngine()
        pf = PreFlightDiagnostic(engine, edge_manager, sample_size=1000)
        result = pf.run(candles_6k)
        # sample_size 1000 from 6000 bars: start=1500, end=2500 → 1000 bars
        assert result.sample_size == 1000

    def test_funnel_report_is_populated_even_when_aborted(self, edge_manager, candles_6k):
        engine = NeverSignalEngine()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        assert result.funnel_report != {}
        assert "total_bars" in result.funnel_report

    def test_config_none_does_not_raise(self, edge_manager, candles_6k):
        engine = NeverSignalEngine()
        pf = PreFlightDiagnostic(engine, edge_manager, config=None)
        result = pf.run(candles_6k)
        assert isinstance(result, PreFlightResult)

    def test_custom_sample_size_smaller_than_default(self, edge_manager, candles_6k):
        engine = AlwaysSignalEngine()
        pf = PreFlightDiagnostic(engine, edge_manager, sample_size=500)
        result = pf.run(candles_6k)
        assert result.sample_size == 500

    def test_run_returns_preflight_result_instance(self, edge_manager, candles_6k):
        engine = AlwaysSignalEngine()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        assert isinstance(result, PreFlightResult)


# ---------------------------------------------------------------------------
# 11. relaxed_config field
# ---------------------------------------------------------------------------


class TestRelaxedConfig:
    """Tests for the PreFlightResult.relaxed_config field."""

    def test_no_relaxation_returns_none(self, edge_manager, candles_6k):
        """When signals are found on first scan, relaxed_config is None."""
        engine = AlwaysSignalEngine()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        assert result.relaxed_config is None

    def test_aborted_returns_none(self, edge_manager, candles_6k):
        """When all tiers exhausted, relaxed_config is None."""
        engine = NeverSignalEngine()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)
        assert result.relaxed_config is None

    def test_empty_candles_returns_none(self, edge_manager):
        """Empty candle input → aborted, relaxed_config is None."""
        engine = AlwaysSignalEngine()
        result = PreFlightDiagnostic(engine, edge_manager).run(pd.DataFrame())
        assert result.relaxed_config is None

    def test_relaxation_success_returns_config_dict(self, edge_manager, candles_6k):
        """When relaxation fixes the drought, relaxed_config is a non-empty dict."""

        class _SignalAfterRelax:
            """Signals only after the first full scan fails (simulates relaxation fix)."""

            def __init__(self):
                self._scan_count = 0
                self._threshold = 200  # signals after this many scans

            def scan(self, data_1m, current_bar=-1):
                self._scan_count += 1
                if self._scan_count > self._threshold:
                    return make_signal()
                return None

        engine = _SignalAfterRelax()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)

        if result.relaxation_applied and result.passed:
            assert result.relaxed_config is not None
            assert isinstance(result.relaxed_config, dict)
            assert len(result.relaxed_config) > 0

    def test_relaxed_config_has_expected_keys(self, edge_manager, candles_6k):
        """relaxed_config dict should contain edge parameter keys."""

        class _SignalAfterRelax:
            def __init__(self):
                self._scan_count = 0

            def scan(self, data_1m, current_bar=-1):
                self._scan_count += 1
                if self._scan_count > 200:
                    return make_signal()
                return None

        engine = _SignalAfterRelax()
        result = PreFlightDiagnostic(engine, edge_manager).run(candles_6k)

        if result.relaxed_config is not None:
            # Should contain at least some of the known edge parameter keys
            known_prefixes = [
                "day_of_week__", "time_of_day__", "london_open_delay__",
                "regime_filter__", "confluence_scoring__",
            ]
            has_known_key = any(
                any(k.startswith(p) for p in known_prefixes)
                for k in result.relaxed_config
            )
            assert has_known_key
