"""
Equity curve filter (modifier edge).

A sustained sequence of losing trades suggests the strategy is misaligned
with current market conditions. When the recent equity curve falls below
its moving average, reducing position size limits drawdown during
deteriorating conditions while keeping the strategy active.

This is a MODIFIER edge. It does not block entries; it reduces the
position size multiplier when the equity curve is below its MA.
"""

from __future__ import annotations

from .base import EdgeContext, EdgeFilter, EdgeResult


class EquityCurveFilter(EdgeFilter):
    """Reduce position size when the equity curve is below its moving average.

    The equity curve is represented as a list of closed trade P&L values
    in R multiples (e.g. [1.0, -1.0, 2.0, -1.0, ...]). The MA is computed
    over the most recent ``lookback_trades`` values.

    Config keys (via ``params``):
        lookback_trades:         int   — MA period. Default 20.
        reduced_size_multiplier: float — Size multiplier when below MA. Default 0.5.

    Returns
    -------
    EdgeResult
        ``allowed`` is always True.
        ``modifier`` is 1.0 (full size) when equity curve is healthy, or
        ``reduced_size_multiplier`` when below the MA.
    """

    def __init__(self, config: dict) -> None:
        super().__init__("equity_curve", config)
        params = config.get("params", {})
        self._lookback: int = int(params.get("lookback_trades", 20))
        self._reduced: float = float(params.get("reduced_size_multiplier", 0.5))

    def should_allow(self, context: EdgeContext) -> EdgeResult:
        if not self.enabled:
            return EdgeResult(
                allowed=True,
                edge_name=self.name,
                reason="equity_curve filter disabled — full size applied",
                modifier=1.0,
            )

        curve = context.equity_curve

        # Need at least 2 points to compute a meaningful MA
        if not curve or len(curve) < 2:
            return EdgeResult(
                allowed=True,
                edge_name=self.name,
                reason=(
                    f"Insufficient equity curve history "
                    f"({len(curve)} trades) — full size applied"
                ),
                modifier=1.0,
            )

        # Use up to ``lookback`` most recent trades
        window = curve[-self._lookback:]
        ma = sum(window) / len(window)
        current = window[-1]

        if current >= ma:
            return EdgeResult(
                allowed=True,
                edge_name=self.name,
                reason=(
                    f"Equity curve current {current:.2f}R ≥ MA {ma:.2f}R "
                    f"over {len(window)} trades — full size"
                ),
                modifier=1.0,
            )

        return EdgeResult(
            allowed=True,
            edge_name=self.name,
            reason=(
                f"Equity curve current {current:.2f}R < MA {ma:.2f}R "
                f"over {len(window)} trades — size reduced to "
                f"{self._reduced:.0%}"
            ),
            modifier=self._reduced,
        )
