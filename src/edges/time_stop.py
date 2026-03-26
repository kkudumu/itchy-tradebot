"""
Time stop (exit edge).

A trade that stalls without reaching the minimum profit target after a
set number of candles is unlikely to reach the full target. Exiting at
breakeven preserves capital and prevents a small winner from becoming a
loser. This is an EXIT edge, not an entry filter.

Exit is triggered when BOTH conditions are true:
- The trade has been open for at least ``candle_limit`` bars.
- The current unrealised profit is below ``breakeven_r_threshold`` R.
"""

from __future__ import annotations

from .base import EdgeContext, EdgeFilter, EdgeResult


class TimeStopFilter(EdgeFilter):
    """Exit at breakeven when the trade has stalled past the candle limit.

    Config keys (via ``params``):
        candle_limit:           int   — Bars to wait before applying time
                                        stop. Default 12.
        breakeven_r_threshold:  float — Minimum R required by ``candle_limit``
                                        to avoid the time stop. Default 0.5.
    """

    def __init__(self, config: dict) -> None:
        super().__init__("time_stop", config)
        params = config.get("params", {})
        self._candle_limit: int = int(params.get("candle_limit", 12))
        self._r_threshold: float = float(params.get("breakeven_r_threshold", 0.5))

    def should_allow(self, context: EdgeContext) -> EdgeResult:
        """Return allowed=False when the time-stop exit condition is met.

        ``allowed=True`` means the trade should be kept open.
        ``allowed=False`` means exit at breakeven.
        """
        if not self.enabled:
            return self._disabled_result()

        candles = context.candles_since_entry
        current_r = context.current_r

        # If trade state fields are absent, this edge cannot evaluate — pass
        if candles is None or current_r is None:
            return EdgeResult(
                allowed=True,
                edge_name=self.name,
                reason="No active trade context — time stop not applicable",
            )

        if candles < self._candle_limit:
            return EdgeResult(
                allowed=True,
                edge_name=self.name,
                reason=(
                    f"Trade is {candles} candles old — time stop activates at "
                    f"{self._candle_limit} candles"
                ),
            )

        # Time limit reached — check if the trade has sufficient profit
        if current_r >= self._r_threshold:
            return EdgeResult(
                allowed=True,
                edge_name=self.name,
                reason=(
                    f"Trade at {current_r:.2f}R after {candles} candles — "
                    f"above {self._r_threshold}R threshold, no time stop"
                ),
            )

        return EdgeResult(
            allowed=False,
            edge_name=self.name,
            reason=(
                f"Trade at {current_r:.2f}R after {candles} candles — "
                f"below {self._r_threshold}R threshold at {self._candle_limit}-candle "
                f"limit: exit at breakeven"
            ),
        )
