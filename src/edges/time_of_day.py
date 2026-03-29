"""
Time-of-day entry filter.

Only allows entries during configured UTC hours. Gold exhibits poor
liquidity and directional follow-through outside the London/NY window,
so restricting entries to 08:00–17:00 UTC reduces low-quality setups.
"""

from __future__ import annotations

from .base import EdgeContext, EdgeFilter, EdgeResult


class TimeOfDayFilter(EdgeFilter):
    """Allow entries only within a configured UTC time window.

    Default window: 08:00–17:00 UTC (London open through mid-NY session).

    Config keys (via ``params``):
        start_utc: str  — "HH:MM" format, inclusive start. Default "08:00".
        end_utc:   str  — "HH:MM" format, exclusive end.   Default "17:00".
    """

    def __init__(self, config: dict) -> None:
        super().__init__("time_of_day", config)
        params = config.get("params", {})
        self._start_minutes = self._parse_time(params.get("start_utc", "08:00"))
        self._end_minutes = self._parse_time(params.get("end_utc", "17:00"))

    # ------------------------------------------------------------------
    # Runtime setters (for adaptive relaxation)
    # ------------------------------------------------------------------

    def set_window(self, start_minutes: int, end_minutes: int) -> None:
        """Set trading window at runtime (for adaptive relaxation)."""
        self._start_minutes = int(start_minutes)
        self._end_minutes = int(end_minutes)

    def get_window(self) -> tuple[int, int]:
        """Get current trading window as (start_minutes, end_minutes)."""
        return (self._start_minutes, self._end_minutes)

    @staticmethod
    def _parse_time(hhmm: str) -> int:
        """Convert 'HH:MM' string to minutes since midnight."""
        h, m = hhmm.split(":")
        return int(h) * 60 + int(m)

    def should_allow(self, context: EdgeContext) -> EdgeResult:
        if not self.enabled:
            return self._disabled_result()

        bar_minutes = context.timestamp.hour * 60 + context.timestamp.minute

        if self._start_minutes <= bar_minutes < self._end_minutes:
            return EdgeResult(
                allowed=True,
                edge_name=self.name,
                reason=(
                    f"Bar time {context.timestamp.strftime('%H:%M')} UTC is within "
                    f"allowed window {self._format(self._start_minutes)}–"
                    f"{self._format(self._end_minutes)} UTC"
                ),
            )

        return EdgeResult(
            allowed=False,
            edge_name=self.name,
            reason=(
                f"Bar time {context.timestamp.strftime('%H:%M')} UTC is outside "
                f"allowed window {self._format(self._start_minutes)}–"
                f"{self._format(self._end_minutes)} UTC"
            ),
        )

    @staticmethod
    def _format(minutes: int) -> str:
        return f"{minutes // 60:02d}:{minutes % 60:02d}"
