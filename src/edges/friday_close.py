"""
Friday close (exit edge).

Holding positions over the weekend exposes the account to gap risk from
geopolitical events, economic announcements, and weekend illiquidity.
This edge signals that any open position should be closed on Friday at
or after the configured time.

This is an EXIT edge — it operates on active trades, not on entry signals.
"""

from __future__ import annotations

from .base import EdgeContext, EdgeFilter, EdgeResult

_FRIDAY = 4  # datetime.weekday() == 4


class FridayCloseFilter(EdgeFilter):
    """Signal that open positions should be closed on Friday.

    Returns ``allowed=False`` (exit triggered) when it is Friday and the
    current bar time is at or after ``close_time_utc``.

    Config keys (via ``params``):
        close_time_utc: str  — "HH:MM" UTC. Default "20:00".
        day:            int  — Weekday index for the close day. Default 4 (Friday).
    """

    def __init__(self, config: dict) -> None:
        super().__init__("friday_close", config)
        params = config.get("params", {})
        self._close_minutes = self._parse_time(params.get("close_time_utc", "20:00"))
        self._day: int = int(params.get("day", _FRIDAY))

    @staticmethod
    def _parse_time(hhmm: str) -> int:
        h, m = hhmm.split(":")
        return int(h) * 60 + int(m)

    def should_allow(self, context: EdgeContext) -> EdgeResult:
        """Return allowed=False when the position must be closed (exit triggered)."""
        if not self.enabled:
            return self._disabled_result()

        if context.day_of_week != self._day:
            return EdgeResult(
                allowed=True,
                edge_name=self.name,
                reason=f"Not Friday — no weekend close required",
            )

        bar_minutes = context.timestamp.hour * 60 + context.timestamp.minute
        close_hhmm = f"{self._close_minutes // 60:02d}:{self._close_minutes % 60:02d}"

        if bar_minutes >= self._close_minutes:
            return EdgeResult(
                allowed=False,
                edge_name=self.name,
                reason=(
                    f"Friday {context.timestamp.strftime('%H:%M')} UTC is at or after "
                    f"weekend close time {close_hhmm} UTC — exit to avoid gap risk"
                ),
            )

        return EdgeResult(
            allowed=True,
            edge_name=self.name,
            reason=(
                f"Friday {context.timestamp.strftime('%H:%M')} UTC is before "
                f"weekend close time {close_hhmm} UTC"
            ),
        )
