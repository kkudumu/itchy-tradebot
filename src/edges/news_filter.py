"""
News event filter.

High-impact news releases cause sharp, unpredictable price spikes that
invalidate technical setups and widen spreads significantly. This filter
blocks new entries within a configurable window before and after any
high-impact event.

The ``NewsCalendar`` interface is deliberately minimal so it can be
backed by a live economic calendar feed, a pre-loaded YAML file, or a
mock for testing. The only requirement is that it exposes
``get_events(date)`` returning a list of ``NewsEvent`` objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable

from .base import EdgeContext, EdgeFilter, EdgeResult


# ---------------------------------------------------------------------------
# News calendar protocol & data types
# ---------------------------------------------------------------------------


@dataclass
class NewsEvent:
    """A single economic news event.

    Attributes
    ----------
    timestamp:
        Scheduled release time in UTC.
    title:
        Short description of the event (e.g. "US CPI").
    impact:
        Impact level string: 'red' (high), 'orange' (medium), 'yellow' (low).
    """

    timestamp: datetime
    title: str
    impact: str = "red"


@runtime_checkable
class NewsCalendar(Protocol):
    """Minimal interface for economic calendar data sources."""

    def get_events(self, date: datetime) -> list[NewsEvent]:
        """Return all news events scheduled on the given calendar date (UTC)."""
        ...


class EmptyCalendar:
    """Placeholder calendar that reports no events.

    Used when no real calendar is configured. Keeps the filter enabled
    but effectively a no-op until a real calendar is injected.
    """

    def get_events(self, date: datetime) -> list[NewsEvent]:  # noqa: ARG002
        return []


# ---------------------------------------------------------------------------
# Filter implementation
# ---------------------------------------------------------------------------


class NewsFilter(EdgeFilter):
    """Block entries within N minutes of high-impact news events.

    Config keys (via ``params``):
        block_minutes_before: int          — Blackout before event. Default 30.
        block_minutes_after:  int          — Blackout after event.  Default 30.
        impact_levels:        list[str]    — Which impact levels trigger the
                                             blackout. Default ["red"].

    Parameters
    ----------
    config:
        Edge config dict.
    calendar:
        Implementation of the NewsCalendar protocol. Defaults to
        EmptyCalendar (no events — filter is a pass-through).
    """

    def __init__(
        self,
        config: dict,
        calendar: NewsCalendar | None = None,
    ) -> None:
        super().__init__("news_filter", config)
        params = config.get("params", {})
        self._before: int = int(params.get("block_minutes_before", 30))
        self._after: int = int(params.get("block_minutes_after", 30))
        self._impact_levels: set[str] = set(params.get("impact_levels", ["red"]))
        self._calendar: NewsCalendar = calendar if calendar is not None else EmptyCalendar()

    def should_allow(self, context: EdgeContext) -> EdgeResult:
        if not self.enabled:
            return self._disabled_result()

        bar_time = context.timestamp
        events = self._calendar.get_events(bar_time)

        for event in events:
            if event.impact not in self._impact_levels:
                continue

            window_start = event.timestamp - timedelta(minutes=self._before)
            window_end = event.timestamp + timedelta(minutes=self._after)

            if window_start <= bar_time <= window_end:
                return EdgeResult(
                    allowed=False,
                    edge_name=self.name,
                    reason=(
                        f"Bar at {bar_time.strftime('%H:%M')} UTC is within "
                        f"{self._before}m pre / {self._after}m post blackout for "
                        f"'{event.title}' ({event.impact.upper()} impact at "
                        f"{event.timestamp.strftime('%H:%M')} UTC)"
                    ),
                )

        return EdgeResult(
            allowed=True,
            edge_name=self.name,
            reason="No high-impact news events within blackout window",
        )

    def set_calendar(self, calendar: NewsCalendar) -> None:
        """Replace the calendar implementation at runtime."""
        self._calendar = calendar
