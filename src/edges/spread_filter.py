"""
Spread filter.

A wide spread indicates thin liquidity, high volatility, or an imminent
news event. Entering when the spread exceeds the threshold ensures that
the fill cost does not erode the edge of the setup.

For XAU/USD (gold), a spread of 30 points is approximately 0.30 USD/oz
per lot — acceptable under normal conditions. Spreads above this level
suggest abnormal market conditions.
"""

from __future__ import annotations

from .base import EdgeContext, EdgeFilter, EdgeResult


class SpreadFilter(EdgeFilter):
    """Block entries when the current spread exceeds ``max_spread_points``.

    Config keys (via ``params``):
        max_spread_points: float  — Upper spread limit in broker points.
                                    Default 30.
    """

    def __init__(self, config: dict) -> None:
        super().__init__("spread_filter", config)
        params = config.get("params", {})
        self._max_spread: float = float(params.get("max_spread_points", 30))

    def should_allow(self, context: EdgeContext) -> EdgeResult:
        if not self.enabled:
            return self._disabled_result()

        spread = context.spread

        if spread <= self._max_spread:
            return EdgeResult(
                allowed=True,
                edge_name=self.name,
                reason=(
                    f"Spread {spread:.1f} pts is within acceptable limit "
                    f"of {self._max_spread:.0f} pts"
                ),
            )

        return EdgeResult(
            allowed=False,
            edge_name=self.name,
            reason=(
                f"Spread {spread:.1f} pts exceeds maximum {self._max_spread:.0f} pts — "
                f"fill cost too high"
            ),
        )
