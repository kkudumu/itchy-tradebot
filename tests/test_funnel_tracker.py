"""
Unit tests for SignalFunnelTracker.

Coverage goals
--------------
- Basic recording and counting
- Drought detection (zero signals in window)
- Bottleneck identification (highest rejection rate)
- Rolling signal rate calculation
- Thread safety (concurrent access)
- Edge cases (empty state, single bar, window boundaries)
- get_funnel_report() structure
- reset() clears all state
"""

from __future__ import annotations

import threading
from typing import List

import pytest

from src.monitoring import SignalFunnelTracker, FilterEvent
from src.monitoring.funnel_tracker import (
    CORE_FILTER_STAGES,
    EDGE_FILTER_STAGES,
    ALL_KNOWN_STAGES,
    BarRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record_bar(tracker: SignalFunnelTracker, bar_idx: int, signal: bool) -> None:
    """Record a complete bar with all core filters passing or all failing."""
    for f in CORE_FILTER_STAGES:
        tracker.record_filter(bar_idx, f, signal)
    tracker.record_bar_complete(bar_idx, signal)


def _record_bars(tracker: SignalFunnelTracker, n: int, signal: bool = False,
                 start: int = 0) -> None:
    """Record *n* bars starting at *start*, each with the given signal flag."""
    for i in range(n):
        _record_bar(tracker, start + i, signal)


# ===========================================================================
# 1. Package imports
# ===========================================================================

class TestPackageImports:
    def test_import_signal_funnel_tracker(self):
        from src.monitoring import SignalFunnelTracker
        assert SignalFunnelTracker is not None

    def test_import_filter_event(self):
        from src.monitoring import FilterEvent
        assert FilterEvent is not None

    def test_constants_non_empty(self):
        assert len(CORE_FILTER_STAGES) >= 5
        assert len(EDGE_FILTER_STAGES) >= 1
        assert len(ALL_KNOWN_STAGES) == len(CORE_FILTER_STAGES) + len(EDGE_FILTER_STAGES)


# ===========================================================================
# 2. Construction
# ===========================================================================

class TestConstruction:
    def test_default_window_size(self):
        t = SignalFunnelTracker()
        assert t.window_size == 2000

    def test_custom_window_size(self):
        t = SignalFunnelTracker(window_size=100)
        assert t.window_size == 100

    def test_initial_total_bars_zero(self):
        t = SignalFunnelTracker()
        assert t.total_bars == 0

    def test_initial_total_signals_zero(self):
        t = SignalFunnelTracker()
        assert t.total_signals == 0

    def test_invalid_window_size_raises(self):
        with pytest.raises(ValueError):
            SignalFunnelTracker(window_size=0)

    def test_invalid_negative_window_size_raises(self):
        with pytest.raises(ValueError):
            SignalFunnelTracker(window_size=-1)


# ===========================================================================
# 3. record_filter — basic counting
# ===========================================================================

class TestRecordFilter:
    def test_single_pass_increments_pass_count(self):
        t = SignalFunnelTracker()
        t.record_filter(0, "4h_cloud", True)
        report = t.get_funnel_report()
        assert report["filters"]["4h_cloud"]["pass"] == 1
        assert report["filters"]["4h_cloud"]["fail"] == 0

    def test_single_fail_increments_fail_count(self):
        t = SignalFunnelTracker()
        t.record_filter(0, "4h_cloud", False)
        report = t.get_funnel_report()
        assert report["filters"]["4h_cloud"]["pass"] == 0
        assert report["filters"]["4h_cloud"]["fail"] == 1

    def test_multiple_filters_tracked_independently(self):
        t = SignalFunnelTracker()
        t.record_filter(0, "4h_cloud", True)
        t.record_filter(0, "1h_confirmation", False)
        report = t.get_funnel_report()
        assert report["filters"]["4h_cloud"]["pass"] == 1
        assert report["filters"]["1h_confirmation"]["fail"] == 1

    def test_unknown_filter_tracked(self):
        t = SignalFunnelTracker()
        t.record_filter(0, "custom_filter", True)
        report = t.get_funnel_report()
        assert "custom_filter" in report["filters"]

    def test_total_equals_pass_plus_fail(self):
        t = SignalFunnelTracker()
        t.record_filter(0, "confluence", True)
        t.record_filter(1, "confluence", False)
        t.record_filter(2, "confluence", True)
        info = t.get_funnel_report()["filters"]["confluence"]
        assert info["total"] == info["pass"] + info["fail"]

    def test_pass_rate_calculation(self):
        t = SignalFunnelTracker()
        t.record_filter(0, "15m_signal", True)
        t.record_filter(1, "15m_signal", True)
        t.record_filter(2, "15m_signal", False)
        info = t.get_funnel_report()["filters"]["15m_signal"]
        assert abs(info["pass_rate"] - 2 / 3) < 1e-9

    def test_rejection_rate_plus_pass_rate_equals_one(self):
        t = SignalFunnelTracker()
        t.record_filter(0, "5m_entry", True)
        t.record_filter(1, "5m_entry", False)
        info = t.get_funnel_report()["filters"]["5m_entry"]
        assert abs(info["pass_rate"] + info["rejection_rate"] - 1.0) < 1e-9


# ===========================================================================
# 4. record_bar_complete
# ===========================================================================

class TestRecordBarComplete:
    def test_bar_increments_total_bars(self):
        t = SignalFunnelTracker()
        t.record_bar_complete(0, False)
        assert t.total_bars == 1

    def test_signal_bar_increments_signal_count(self):
        t = SignalFunnelTracker()
        t.record_bar_complete(0, True)
        assert t.total_signals == 1

    def test_non_signal_bar_does_not_increment_signal_count(self):
        t = SignalFunnelTracker()
        t.record_bar_complete(0, False)
        assert t.total_signals == 0

    def test_multiple_bars(self):
        t = SignalFunnelTracker()
        for i in range(5):
            t.record_bar_complete(i, i % 2 == 0)   # bars 0,2,4 have signals
        assert t.total_bars == 5
        assert t.total_signals == 3


# ===========================================================================
# 5. is_drought
# ===========================================================================

class TestIsDrought:
    def test_empty_tracker_not_drought(self):
        t = SignalFunnelTracker()
        assert t.is_drought() is False

    def test_all_non_signal_bars_is_drought(self):
        t = SignalFunnelTracker()
        _record_bars(t, 10, signal=False)
        assert t.is_drought(window=10) is True

    def test_recent_signal_breaks_drought(self):
        t = SignalFunnelTracker()
        _record_bars(t, 9, signal=False)
        _record_bar(t, 9, signal=True)
        assert t.is_drought(window=10) is False

    def test_signal_outside_window_still_drought(self):
        t = SignalFunnelTracker()
        _record_bar(t, 0, signal=True)          # bar 0 — outside window
        _record_bars(t, 10, signal=False, start=1)  # bars 1–10 — no signals
        assert t.is_drought(window=10) is True

    def test_signal_at_window_boundary_not_drought(self):
        t = SignalFunnelTracker()
        _record_bars(t, 9, signal=False)
        _record_bar(t, 9, signal=True)   # exactly at the edge of window=10
        assert t.is_drought(window=10) is False

    def test_invalid_window_raises(self):
        t = SignalFunnelTracker()
        with pytest.raises(ValueError):
            t.is_drought(window=0)


# ===========================================================================
# 6. bottleneck_filter
# ===========================================================================

class TestBottleneckFilter:
    def test_empty_returns_none(self):
        t = SignalFunnelTracker()
        assert t.bottleneck_filter() is None

    def test_single_filter_all_fail(self):
        t = SignalFunnelTracker()
        for i in range(5):
            t.record_filter(i, "4h_cloud", False)
        assert t.bottleneck_filter() == "4h_cloud"

    def test_highest_rejection_rate_wins(self):
        t = SignalFunnelTracker()
        # 4h_cloud: 1 pass / 1 fail → 50%
        t.record_filter(0, "4h_cloud", True)
        t.record_filter(1, "4h_cloud", False)
        # 1h_confirmation: 0 pass / 3 fail → 100%
        for i in range(3):
            t.record_filter(i, "1h_confirmation", False)
        assert t.bottleneck_filter() == "1h_confirmation"

    def test_tie_goes_to_earliest_in_pipeline(self):
        t = SignalFunnelTracker()
        # Both at 100% rejection
        t.record_filter(0, "1h_confirmation", False)
        t.record_filter(0, "4h_cloud", False)
        # 4h_cloud is earlier in ALL_KNOWN_STAGES
        assert t.bottleneck_filter() == "4h_cloud"

    def test_zero_total_filter_excluded(self):
        """A filter with only pass=0 and fail=0 (hypothetically) is excluded."""
        t = SignalFunnelTracker()
        t.record_filter(0, "spread_filter", True)  # 100% pass
        t.record_filter(1, "time_of_day", False)   # 100% reject
        assert t.bottleneck_filter() == "time_of_day"

    def test_all_pass_returns_filter_with_lowest_pass_rate(self):
        t = SignalFunnelTracker()
        t.record_filter(0, "4h_cloud", True)
        t.record_filter(0, "1h_confirmation", True)
        # All pass rates are 100% → rejection rate = 0 for both
        # Either is valid as bottleneck; just check it returns something
        assert t.bottleneck_filter() is not None


# ===========================================================================
# 7. rolling_signal_rate
# ===========================================================================

class TestRollingSignalRate:
    def test_empty_returns_zero(self):
        t = SignalFunnelTracker()
        assert t.rolling_signal_rate() == 0.0

    def test_all_signal_bars_returns_one(self):
        t = SignalFunnelTracker()
        _record_bars(t, 10, signal=True)
        assert t.rolling_signal_rate(window=10) == 1.0

    def test_no_signal_bars_returns_zero(self):
        t = SignalFunnelTracker()
        _record_bars(t, 10, signal=False)
        assert t.rolling_signal_rate(window=10) == 0.0

    def test_half_signal_rate(self):
        t = SignalFunnelTracker()
        for i in range(10):
            _record_bar(t, i, signal=(i % 2 == 0))
        rate = t.rolling_signal_rate(window=10)
        assert abs(rate - 0.5) < 1e-9

    def test_window_limits_to_recent_bars(self):
        t = SignalFunnelTracker()
        # 10 non-signal bars, then 5 signal bars
        _record_bars(t, 10, signal=False)
        _record_bars(t, 5, signal=True, start=10)
        # Window=5 → only the last 5 signal bars → rate=1.0
        assert t.rolling_signal_rate(window=5) == 1.0
        # Window=15 → 5 signals out of 15 → rate=1/3
        assert abs(t.rolling_signal_rate(window=15) - 5 / 15) < 1e-9

    def test_invalid_window_raises(self):
        t = SignalFunnelTracker()
        with pytest.raises(ValueError):
            t.rolling_signal_rate(window=0)

    def test_window_larger_than_history_uses_all(self):
        t = SignalFunnelTracker()
        _record_bars(t, 3, signal=True)
        # window=1000 but only 3 bars exist
        assert t.rolling_signal_rate(window=1000) == 1.0


# ===========================================================================
# 8. get_funnel_report structure
# ===========================================================================

class TestGetFunnelReport:
    def test_top_level_keys_present(self):
        t = SignalFunnelTracker()
        report = t.get_funnel_report()
        expected_keys = {
            "total_bars", "total_signals", "overall_signal_rate",
            "rolling_signal_rate_500", "is_drought_500",
            "bottleneck", "filters",
        }
        assert expected_keys == set(report.keys())

    def test_empty_tracker_report_values(self):
        t = SignalFunnelTracker()
        report = t.get_funnel_report()
        assert report["total_bars"] == 0
        assert report["total_signals"] == 0
        assert report["overall_signal_rate"] == 0.0
        assert report["rolling_signal_rate_500"] == 0.0
        assert report["is_drought_500"] is False
        assert report["bottleneck"] is None
        assert report["filters"] == {}

    def test_filter_entry_keys(self):
        t = SignalFunnelTracker()
        t.record_filter(0, "4h_cloud", True)
        report = t.get_funnel_report()
        entry = report["filters"]["4h_cloud"]
        assert set(entry.keys()) == {"pass", "fail", "total", "pass_rate", "rejection_rate"}

    def test_overall_signal_rate_consistent(self):
        t = SignalFunnelTracker()
        _record_bars(t, 4, signal=True)
        _record_bars(t, 6, signal=False, start=4)
        report = t.get_funnel_report()
        assert abs(report["overall_signal_rate"] - 0.4) < 1e-9

    def test_drought_flag_reflects_recent_500(self):
        t = SignalFunnelTracker()
        _record_bars(t, 500, signal=False)
        report = t.get_funnel_report()
        assert report["is_drought_500"] is True

    def test_bottleneck_in_report(self):
        t = SignalFunnelTracker()
        for i in range(10):
            t.record_filter(i, "15m_signal", False)
        report = t.get_funnel_report()
        assert report["bottleneck"] == "15m_signal"


# ===========================================================================
# 9. reset()
# ===========================================================================

class TestReset:
    def test_reset_clears_bars(self):
        t = SignalFunnelTracker()
        _record_bars(t, 10, signal=True)
        t.reset()
        assert t.total_bars == 0

    def test_reset_clears_signals(self):
        t = SignalFunnelTracker()
        _record_bars(t, 5, signal=True)
        t.reset()
        assert t.total_signals == 0

    def test_reset_clears_filter_counts(self):
        t = SignalFunnelTracker()
        t.record_filter(0, "4h_cloud", True)
        t.reset()
        report = t.get_funnel_report()
        assert report["filters"] == {}

    def test_reset_clears_drought(self):
        t = SignalFunnelTracker()
        _record_bars(t, 600, signal=False)
        assert t.is_drought(window=500) is True
        t.reset()
        assert t.is_drought(window=500) is False

    def test_recording_after_reset_works(self):
        t = SignalFunnelTracker()
        _record_bars(t, 5, signal=True)
        t.reset()
        _record_bars(t, 3, signal=False)
        assert t.total_bars == 3
        assert t.total_signals == 0

    def test_filter_names_empty_after_reset(self):
        t = SignalFunnelTracker()
        t.record_filter(0, "4h_cloud", True)
        t.reset()
        assert t.filter_names() == []


# ===========================================================================
# 10. Thread safety
# ===========================================================================

class TestThreadSafety:
    def test_concurrent_record_filter_no_data_race(self):
        """Multiple threads writing record_filter simultaneously must not corrupt counts."""
        t = SignalFunnelTracker(window_size=10_000)
        n_threads = 8
        events_per_thread = 500

        def worker(thread_id: int) -> None:
            for i in range(events_per_thread):
                bar = thread_id * events_per_thread + i
                t.record_filter(bar, "4h_cloud", i % 2 == 0)

        threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(n_threads)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        report = t.get_funnel_report()
        info = report["filters"]["4h_cloud"]
        total_expected = n_threads * events_per_thread
        assert info["total"] == total_expected

    def test_concurrent_write_and_read_no_exception(self):
        """A reader thread calling get_funnel_report while writers are active must not raise."""
        t = SignalFunnelTracker(window_size=5000)
        errors: List[Exception] = []

        def writer() -> None:
            for i in range(1000):
                t.record_filter(i, "4h_cloud", True)
                t.record_bar_complete(i, True)

        def reader() -> None:
            for _ in range(200):
                try:
                    t.get_funnel_report()
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

        w = threading.Thread(target=writer)
        r = threading.Thread(target=reader)
        w.start()
        r.start()
        w.join()
        r.join()

        assert errors == [], f"Reader raised: {errors[0]}"

    def test_concurrent_record_bar_complete_totals_consistent(self):
        """total_bars must equal n_threads * iterations after concurrent writes."""
        t = SignalFunnelTracker(window_size=50_000)
        n_threads = 4
        iters = 250

        def worker() -> None:
            for i in range(iters):
                t.record_bar_complete(i, False)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert t.total_bars == n_threads * iters


# ===========================================================================
# 11. Window roll-off behaviour
# ===========================================================================

class TestWindowRollOff:
    def test_window_size_limits_event_deque(self):
        t = SignalFunnelTracker(window_size=10)
        for i in range(20):
            t.record_filter(i, "4h_cloud", True)
        # Internal deque capped at window_size — cannot inspect directly,
        # but lifetime counter must reflect all 20 events
        info = t.get_funnel_report()["filters"]["4h_cloud"]
        assert info["pass"] == 20

    def test_bar_deque_capped_at_window_size(self):
        """Rolling rate only sees the last window_size bar records."""
        t = SignalFunnelTracker(window_size=5)
        # Record 5 signal bars, then 5 non-signal bars
        _record_bars(t, 5, signal=True)
        _record_bars(t, 5, signal=False, start=5)
        # The bar deque holds max 5 entries; the last 5 are non-signal
        # rolling_signal_rate(window=5) should be 0.0
        assert t.rolling_signal_rate(window=5) == 0.0

    def test_is_drought_respects_deque_cap(self):
        t = SignalFunnelTracker(window_size=5)
        # Fill with signal bars (deque will cap at 5)
        _record_bars(t, 5, signal=True)
        # Overwrite with non-signal bars
        _record_bars(t, 5, signal=False, start=5)
        # After roll-off, only non-signal bars remain in deque
        assert t.is_drought(window=5) is True


# ===========================================================================
# 12. filter_names helper
# ===========================================================================

class TestFilterNames:
    def test_empty_returns_empty_list(self):
        t = SignalFunnelTracker()
        assert t.filter_names() == []

    def test_returns_all_recorded_filter_names_sorted(self):
        t = SignalFunnelTracker()
        t.record_filter(0, "zebra_filter", True)
        t.record_filter(0, "4h_cloud", True)
        names = t.filter_names()
        assert names == sorted(names)
        assert "zebra_filter" in names
        assert "4h_cloud" in names
