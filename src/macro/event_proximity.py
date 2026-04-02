"""Event proximity edge filter.

Blocks entries within N hours of high-impact economic events (NFP,
FOMC, CPI) using the deterministic EconCalendar.

Unlike the existing news_filter (which requires an injected calendar
implementation), this filter is self-contained: it generates event
dates deterministically and requires no external data source.

Config keys (via params):
    hours_before: int -- Blackout window before event. Default 4.
    hours_after:  int -- Blackout window after event. Default 2.
"""

from __future__ import annotations

import logging

from src.edges.base import EdgeContext, EdgeFilter, EdgeResult
from src.macro.econ_calendar import EconCalendar

logger = logging.getLogger(__name__)


class EventProximityFilter(EdgeFilter):
    """Block entries within configurable hours of high-impact US events.

    Parameters
    ----------
    config:
        Edge config dict with 'enabled' and 'params' keys.
    """

    def __init__(self, config: dict) -> None:
        super().__init__("event_proximity", config)
        params = config.get("params", {})
        self._hours_before: int = int(params.get("hours_before", 4))
        self._hours_after: int = int(params.get("hours_after", 2))
        self._calendar = EconCalendar()

    def should_allow(self, context: EdgeContext) -> EdgeResult:
        if not self.enabled:
            return self._disabled_result()

        is_near, event = self._calendar.is_near_event(
            context.timestamp,
            hours_before=self._hours_before,
            hours_after=self._hours_after,
        )

        if is_near and event is not None:
            return EdgeResult(
                allowed=False,
                edge_name=self.name,
                reason=(
                    f"Bar at {context.timestamp.strftime('%Y-%m-%d %H:%M')} UTC "
                    f"is within {self._hours_before}h pre / {self._hours_after}h post "
                    f"blackout for '{event.title}' at "
                    f"{event.timestamp.strftime('%H:%M')} UTC"
                ),
            )

        return EdgeResult(
            allowed=True,
            edge_name=self.name,
            reason="No high-impact events within blackout window",
        )
