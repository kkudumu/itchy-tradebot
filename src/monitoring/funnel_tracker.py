"""
SignalFunnelTracker — per-filter pass/fail/rate metrics for the signal cascade.

This module is *pure observation only* — it records what happened, never
triggers any side effects.  It is designed to be shared between the backtest
loop (writes) and the live dashboard (reads) via a threading.Lock.

Filter stages (in pipeline order)
----------------------------------
1. 4h_cloud           — 4H cloud direction
2. 1h_confirmation    — 1H TK alignment
3. 15m_signal         — 15M TK cross + cloud position + Chikou
4. 5m_entry           — 5M pullback to Kijun
5. confluence         — Minimum confluence score
   then per-edge filters:
6. time_of_day
7. day_of_week
8. london_open_delay
9. regime_filter
10. confluence_scoring
11. spread_filter
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical ordered list of core filter stages.
CORE_FILTER_STAGES: list[str] = [
    "4h_cloud",
    "1h_confirmation",
    "15m_signal",
    "5m_entry",
    "confluence",
]

#: Additional per-edge filter stages that may be recorded.
EDGE_FILTER_STAGES: list[str] = [
    "time_of_day",
    "day_of_week",
    "london_open_delay",
    "regime_filter",
    "confluence_scoring",
    "spread_filter",
]

ALL_KNOWN_STAGES: list[str] = CORE_FILTER_STAGES + EDGE_FILTER_STAGES


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FilterEvent:
    """A single filter evaluation recorded for one bar."""
    bar_idx: int
    filter_name: str
    passed: bool


@dataclass(frozen=True)
class BarRecord:
    """Summary record written once per bar after all filters are evaluated."""
    bar_idx: int
    signal_generated: bool


# ---------------------------------------------------------------------------
# Main tracker
# ---------------------------------------------------------------------------

class SignalFunnelTracker:
    """
    Tracks per-filter pass/fail counts and rolling signal rates across the
    signal cascade.

    Parameters
    ----------
    window_size:
        Maximum number of *filter events* retained in the internal deque.
        Older events are dropped automatically as the deque reaches capacity.
        Bar records use a separate deque of the same size.
    """

    def __init__(self, window_size: int = 2000) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self._window_size = window_size
        self._lock = Lock()

        # Rolling window of individual filter events
        self._events: deque[FilterEvent] = deque(maxlen=window_size)

        # Rolling window of per-bar summaries (signal_generated flag)
        self._bars: deque[BarRecord] = deque(maxlen=window_size)

        # Lifetime counters (never shrink — they survive window roll-off)
        self._filter_pass: dict[str, int] = {}
        self._filter_fail: dict[str, int] = {}

        # Lifetime totals
        self._total_bars: int = 0
        self._signal_count: int = 0  # bars where a signal was generated

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def record_filter(
        self,
        bar_idx: int,
        filter_name: str,
        passed: bool,
    ) -> None:
        """
        Record one filter evaluation for *bar_idx*.

        This must be called *before* ``record_bar_complete`` for the same bar.

        Parameters
        ----------
        bar_idx:
            Monotonically increasing bar index (0-based).
        filter_name:
            Identifier for the filter stage (e.g. ``"4h_cloud"``).
        passed:
            Whether the candidate passed this filter.
        """
        event = FilterEvent(bar_idx=bar_idx, filter_name=filter_name, passed=passed)
        with self._lock:
            self._events.append(event)
            if passed:
                self._filter_pass[filter_name] = (
                    self._filter_pass.get(filter_name, 0) + 1
                )
            else:
                self._filter_fail[filter_name] = (
                    self._filter_fail.get(filter_name, 0) + 1
                )

    def record_bar_complete(self, bar_idx: int, signal_generated: bool) -> None:
        """
        Called once after all filters have been evaluated for *bar_idx*.

        Parameters
        ----------
        bar_idx:
            Same bar index used in the preceding ``record_filter`` calls.
        signal_generated:
            True when all core filters passed and a trade signal was emitted.
        """
        record = BarRecord(bar_idx=bar_idx, signal_generated=signal_generated)
        with self._lock:
            self._bars.append(record)
            self._total_bars += 1
            if signal_generated:
                self._signal_count += 1

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def is_drought(self, window: int = 500) -> bool:
        """
        Return True when zero signals have been generated in the last *window*
        bars.

        If fewer than *window* bars have been recorded, the check spans all
        recorded bars (it does not wait for a full window before reporting a
        drought).

        Parameters
        ----------
        window:
            Number of most-recent bars to inspect.
        """
        if window < 1:
            raise ValueError("window must be >= 1")
        with self._lock:
            if not self._bars:
                return False
            recent = list(self._bars)[-window:]
            return not any(r.signal_generated for r in recent)

    def bottleneck_filter(self) -> Optional[str]:
        """
        Return the name of the filter with the highest *rejection rate*.

        Rejection rate = failures / (passes + failures).

        Returns None when no filter events have been recorded yet.
        If multiple filters share the same maximum rejection rate, the one
        that appears earliest in ``ALL_KNOWN_STAGES`` is returned; unknown
        filters come after known ones.
        """
        with self._lock:
            all_filters = set(self._filter_pass) | set(self._filter_fail)
            if not all_filters:
                return None

            best_name: Optional[str] = None
            best_rate: float = -1.0

            # Iterate in a stable, deterministic order
            def _sort_key(name: str) -> tuple[int, str]:
                try:
                    return (ALL_KNOWN_STAGES.index(name), name)
                except ValueError:
                    return (len(ALL_KNOWN_STAGES), name)

            for name in sorted(all_filters, key=_sort_key):
                passes = self._filter_pass.get(name, 0)
                fails = self._filter_fail.get(name, 0)
                total = passes + fails
                if total == 0:
                    continue
                rate = fails / total
                if rate > best_rate:
                    best_rate = rate
                    best_name = name

            return best_name

    def rolling_signal_rate(self, window: int = 500) -> float:
        """
        Return the fraction of bars that generated a signal over the last
        *window* bars.

        Returns 0.0 when no bars have been recorded.

        Parameters
        ----------
        window:
            Number of most-recent bars to inspect.
        """
        if window < 1:
            raise ValueError("window must be >= 1")
        with self._lock:
            if not self._bars:
                return 0.0
            recent = list(self._bars)[-window:]
            n = len(recent)
            if n == 0:
                return 0.0
            signals = sum(1 for r in recent if r.signal_generated)
            return signals / n

    def get_funnel_report(self) -> dict:
        """
        Return a comprehensive diagnostic snapshot.

        Schema
        ------
        {
            "total_bars": int,
            "total_signals": int,
            "overall_signal_rate": float,          # signals / total_bars
            "rolling_signal_rate_500": float,
            "is_drought_500": bool,
            "bottleneck": str | None,
            "filters": {
                "<filter_name>": {
                    "pass": int,
                    "fail": int,
                    "total": int,
                    "pass_rate": float,            # 0.0–1.0
                    "rejection_rate": float,       # 0.0–1.0
                },
                ...
            },
        }
        """
        with self._lock:
            total_bars = self._total_bars
            total_signals = self._signal_count
            overall_rate = (
                total_signals / total_bars if total_bars > 0 else 0.0
            )

            # Per-filter stats
            all_filters = set(self._filter_pass) | set(self._filter_fail)
            filters: dict = {}
            for name in all_filters:
                passes = self._filter_pass.get(name, 0)
                fails = self._filter_fail.get(name, 0)
                total = passes + fails
                filters[name] = {
                    "pass": passes,
                    "fail": fails,
                    "total": total,
                    "pass_rate": (passes / total) if total > 0 else 0.0,
                    "rejection_rate": (fails / total) if total > 0 else 0.0,
                }

            # Rolling stats — computed inside the lock using already-snapshotted data
            recent_500 = list(self._bars)[-500:]
            n500 = len(recent_500)
            rolling_rate_500 = (
                sum(1 for r in recent_500 if r.signal_generated) / n500
                if n500 > 0
                else 0.0
            )
            drought_500 = (
                not any(r.signal_generated for r in recent_500)
                if recent_500
                else False
            )

            # Bottleneck (duplicate logic to stay inside lock)
            best_name: Optional[str] = None
            best_rate: float = -1.0

            def _sort_key(name: str) -> tuple[int, str]:
                try:
                    return (ALL_KNOWN_STAGES.index(name), name)
                except ValueError:
                    return (len(ALL_KNOWN_STAGES), name)

            for name in sorted(all_filters, key=_sort_key):
                passes = self._filter_pass.get(name, 0)
                fails = self._filter_fail.get(name, 0)
                total = passes + fails
                if total == 0:
                    continue
                rate = fails / total
                if rate > best_rate:
                    best_rate = rate
                    best_name = name

        return {
            "total_bars": total_bars,
            "total_signals": total_signals,
            "overall_signal_rate": overall_rate,
            "rolling_signal_rate_500": rolling_rate_500,
            "is_drought_500": drought_500,
            "bottleneck": best_name,
            "filters": filters,
        }

    def reset(self) -> None:
        """Clear all recorded state, resetting the tracker to its initial condition."""
        with self._lock:
            self._events.clear()
            self._bars.clear()
            self._filter_pass.clear()
            self._filter_fail.clear()
            self._total_bars = 0
            self._signal_count = 0

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def total_bars(self) -> int:
        """Total number of bars recorded (lifetime, not rolling)."""
        with self._lock:
            return self._total_bars

    @property
    def total_signals(self) -> int:
        """Total number of signal-generating bars (lifetime)."""
        with self._lock:
            return self._signal_count

    @property
    def window_size(self) -> int:
        """Maximum rolling window size (as set at construction)."""
        return self._window_size

    def filter_names(self) -> list[str]:
        """Return sorted list of all filter names seen so far."""
        with self._lock:
            return sorted(set(self._filter_pass) | set(self._filter_fail))
