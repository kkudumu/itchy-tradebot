"""
London open delay filter.

The first 30 minutes after London opens (08:00 UTC) are characterised by
erratic spread widening, stop-hunting, and false breakouts as liquidity
is being established. Suppressing entries during this window avoids being
caught in the initial volatility spike.
"""

from __future__ import annotations

from .base import EdgeContext, EdgeFilter, EdgeResult


class LondonOpenDelayFilter(EdgeFilter):
    """Block entries within N minutes of the London session open.

    Config keys (via ``params``):
        london_open_utc: str  — "HH:MM" UTC. Default "08:00".
        delay_minutes:   int  — Blackout window length. Default 30.
    """

    def __init__(self, config: dict) -> None:
        super().__init__("london_open_delay", config)
        params = config.get("params", {})
        self._open_minutes = self._parse_time(params.get("london_open_utc", "08:00"))
        self._delay: int = int(params.get("delay_minutes", 30))

    @staticmethod
    def _parse_time(hhmm: str) -> int:
        h, m = hhmm.split(":")
        return int(h) * 60 + int(m)

    def should_allow(self, context: EdgeContext) -> EdgeResult:
        if not self.enabled:
            return self._disabled_result()

        bar_minutes = context.timestamp.hour * 60 + context.timestamp.minute
        blackout_end = self._open_minutes + self._delay

        if self._open_minutes <= bar_minutes < blackout_end:
            open_hhmm = f"{self._open_minutes // 60:02d}:{self._open_minutes % 60:02d}"
            end_hhmm = f"{blackout_end // 60:02d}:{blackout_end % 60:02d}"
            return EdgeResult(
                allowed=False,
                edge_name=self.name,
                reason=(
                    f"Bar time {context.timestamp.strftime('%H:%M')} UTC is within "
                    f"London open delay window {open_hhmm}–{end_hhmm} UTC "
                    f"({self._delay} min blackout)"
                ),
            )

        return EdgeResult(
            allowed=True,
            edge_name=self.name,
            reason=(
                f"Bar time {context.timestamp.strftime('%H:%M')} UTC is outside "
                f"London open delay window"
            ),
        )
