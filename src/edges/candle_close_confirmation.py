"""
Candle close confirmation filter.

For cloud breakout setups, entry is only permitted after the H1 candle
body has closed beyond the cloud boundary. This prevents entering on
intrabar wicks that retrace back into the cloud, which is a common
trap in breakout trading.

The context provides ``close_price`` (the confirmed H1 close) and
``kijun_value`` as a proxy for the near edge of the cloud structure.
Callers should populate ``kijun_value`` with the cloud boundary price
most relevant to the breakout direction.
"""

from __future__ import annotations

from .base import EdgeContext, EdgeFilter, EdgeResult


class CandleCloseConfirmationFilter(EdgeFilter):
    """Require the H1 candle body to close beyond the cloud boundary.

    The filter evaluates the signal direction to determine which side
    constitutes "beyond". For a long breakout the close must be above
    ``kijun_value``; for a short breakout it must be below.

    If no signal is attached to the context, or the direction cannot
    be determined, the filter defaults to allowing the trade (pass-through).
    """

    def __init__(self, config: dict) -> None:
        super().__init__("candle_close_confirmation", config)

    def should_allow(self, context: EdgeContext) -> EdgeResult:
        if not self.enabled:
            return self._disabled_result()

        # Resolve trade direction from attached signal
        direction = self._get_direction(context)

        if direction is None:
            # No signal context available — cannot evaluate; allow through
            return EdgeResult(
                allowed=True,
                edge_name=self.name,
                reason="No signal direction available — candle close check skipped",
            )

        close = context.close_price
        boundary = context.kijun_value

        if direction == "long":
            if close > boundary:
                return EdgeResult(
                    allowed=True,
                    edge_name=self.name,
                    reason=(
                        f"H1 close {close:.4f} confirmed above cloud boundary "
                        f"{boundary:.4f} for long breakout"
                    ),
                )
            return EdgeResult(
                allowed=False,
                edge_name=self.name,
                reason=(
                    f"H1 close {close:.4f} did not close above cloud boundary "
                    f"{boundary:.4f} — long breakout unconfirmed"
                ),
            )

        # direction == "short"
        if close < boundary:
            return EdgeResult(
                allowed=True,
                edge_name=self.name,
                reason=(
                    f"H1 close {close:.4f} confirmed below cloud boundary "
                    f"{boundary:.4f} for short breakout"
                ),
            )
        return EdgeResult(
            allowed=False,
            edge_name=self.name,
            reason=(
                f"H1 close {close:.4f} did not close below cloud boundary "
                f"{boundary:.4f} — short breakout unconfirmed"
            ),
        )

    @staticmethod
    def _get_direction(context: EdgeContext) -> str | None:
        """Extract trade direction from the attached signal, if present."""
        if context.signal is None:
            return None
        return getattr(context.signal, "direction", None)
