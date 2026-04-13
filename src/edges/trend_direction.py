"""
Trend direction alignment filter.

Blocks counter-trend entries by comparing the signal direction against the
higher-timeframe trend determined by EMA slope. When the 1H (or 4H) EMA
is falling, long entries are blocked; when rising, short entries are
blocked.

The trend is measured by comparing the current HTF close to an N-bar
simple moving average of HTF closes. If current close > SMA → bullish;
if below → bearish. This is robust across price scales (works equally
for gold at $5000 and oil at $70) because it's relative.

Config keys (via ``params``):
    htf_lookback:    int   — Number of 1H bars to compute SMA over.
                             Default 50 (≈2 trading days).
    require_both_tf: bool  — When True, both 1H and 4H must agree.
                             When False, 1H is sufficient. Default False.
    tolerance_pct:   float — Deadband: if price is within this % of the
                             SMA, trend is "neutral" and all signals pass.
                             Default 0.5 (0.5%).
"""

from __future__ import annotations

from .base import EdgeContext, EdgeFilter, EdgeResult


class TrendDirectionFilter(EdgeFilter):
    """Block entries that fight the higher-timeframe trend."""

    def __init__(self, config: dict) -> None:
        super().__init__("trend_direction", config)
        params = config.get("params", {})
        self._htf_lookback = int(params.get("htf_lookback", 50))
        self._require_both_tf = bool(params.get("require_both_tf", False))
        self._tolerance_pct = float(params.get("tolerance_pct", 0.5))
        self._block_without_data = bool(params.get("block_without_data", False))

    def should_allow(self, context: EdgeContext) -> EdgeResult:
        if not self.enabled:
            return self._disabled_result()

        # Get signal direction
        direction = None
        if context.signal is not None:
            direction = getattr(context.signal, "direction", None)
        if direction is None:
            return EdgeResult(
                allowed=True,
                edge_name=self.name,
                reason="no signal direction — filter skipped",
            )

        # Determine HTF trend from indicator_values populated by the engine
        trend_1h = context.indicator_values.get("htf_trend_1h")  # "bullish", "bearish", "neutral"
        trend_4h = context.indicator_values.get("htf_trend_4h")

        if trend_1h is None and trend_4h is None:
            return EdgeResult(
                allowed=True,
                edge_name=self.name,
                reason="no HTF trend data — filter skipped",
            )

        # Use 1H as primary, 4H as secondary
        primary_trend = trend_1h or trend_4h or "neutral"

        if self._require_both_tf and trend_1h and trend_4h:
            # Both must agree for a strong trend call
            if trend_1h != trend_4h:
                primary_trend = "neutral"

        # Block counter-trend entries
        if direction == "long" and primary_trend == "bearish":
            return EdgeResult(
                allowed=False,
                edge_name=self.name,
                reason=f"long blocked: HTF trend is bearish (1H={trend_1h}, 4H={trend_4h})",
            )

        if direction == "short" and primary_trend == "bullish":
            return EdgeResult(
                allowed=False,
                edge_name=self.name,
                reason=f"short blocked: HTF trend is bullish (1H={trend_1h}, 4H={trend_4h})",
            )

        return EdgeResult(
            allowed=True,
            edge_name=self.name,
            reason=f"trend aligned: {direction} with {primary_trend} (1H={trend_1h}, 4H={trend_4h})",
        )
