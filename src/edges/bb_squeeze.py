"""
Bollinger Band squeeze amplifier (modifier edge).

A Bollinger Band squeeze (bands compressing to unusually narrow width)
signals compressed volatility that often precedes a directional breakout.
When the bands are expanding out of a squeeze, the signal is amplified —
this module boosts the confidence/score rather than acting as a binary
filter.

This is a MODIFIER edge. It does not block entries; it adjusts the
confidence score upward when the squeeze condition is active.
"""

from __future__ import annotations

from .base import EdgeContext, EdgeFilter, EdgeResult


class BBSqueezeAmplifier(EdgeFilter):
    """Boost signal confidence when expanding out of a Bollinger Band squeeze.

    The ``bb_squeeze`` field in EdgeContext (bool) indicates whether the
    Bollinger Bands are currently in a squeeze state as computed by
    ``BollingerBandCalculator``. This amplifier adds ``confidence_boost``
    to the modifier value when the squeeze is active.

    Config keys (via ``params``):
        confidence_boost: int  — Amount added to confluence score when
                                 squeeze is active. Default 1.

    Returns
    -------
    EdgeResult
        ``allowed`` is always True.
        ``modifier`` carries the integer boost (0 when no squeeze).
    """

    def __init__(self, config: dict) -> None:
        super().__init__("bb_squeeze", config)
        params = config.get("params", {})
        self._boost: int = int(params.get("confidence_boost", 1))

    def should_allow(self, context: EdgeContext) -> EdgeResult:
        if not self.enabled:
            return EdgeResult(
                allowed=True,
                edge_name=self.name,
                reason="bb_squeeze amplifier disabled — no boost applied",
                modifier=0.0,
            )

        if context.bb_squeeze:
            return EdgeResult(
                allowed=True,
                edge_name=self.name,
                reason=(
                    f"Bollinger Band squeeze active — confidence boosted by "
                    f"+{self._boost}"
                ),
                modifier=float(self._boost),
            )

        return EdgeResult(
            allowed=True,
            edge_name=self.name,
            reason="No Bollinger Band squeeze — no confidence boost",
            modifier=0.0,
        )
