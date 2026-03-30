"""
Market regime filter.

Ichimoku signals perform best when the market is in a genuine trending
regime. This filter combines two independent conditions:

1. ADX must exceed a threshold — confirms directional momentum.
2. Cloud thickness must be above a percentile — confirms the cloud provides
   meaningful support/resistance rather than being paper-thin.

Both conditions must be satisfied for the filter to allow entry.
"""

from __future__ import annotations

from .base import EdgeContext, EdgeFilter, EdgeResult


class RegimeFilter(EdgeFilter):
    """Require ADX above threshold AND cloud thickness above median.

    Config keys (via ``params``):
        adx_min:                    float — ADX floor. Default 28.
        cloud_thickness_percentile: float — Cloud must be above this
                                            percentile of recent history
                                            (0–100). Default 50 (median).
                                            The context field
                                            ``cloud_thickness`` should be
                                            pre-normalised by the caller to
                                            a 0–100 percentile rank if this
                                            parameter is used as a percentile.
                                            Alternatively, treat it as an
                                            absolute minimum ATR count.

    Note: ``cloud_thickness`` in EdgeContext is the raw absolute price
    distance. The percentile parameter is treated as an absolute minimum
    unless the caller provides a pre-ranked value. For simple backtests,
    set ``cloud_thickness_percentile`` to a raw point threshold (e.g. 5.0
    for "at least 5 USD/oz cloud thickness").
    """

    def __init__(self, config: dict) -> None:
        super().__init__("regime_filter", config)
        params = config.get("params", {})
        self._adx_min: float = float(params.get("adx_min", 28))
        # Used as a raw minimum thickness threshold for practical simplicity.
        # Callers may pre-normalise context.cloud_thickness to a percentile rank
        # if they prefer a relative comparison.
        self._cloud_min: float = float(params.get("cloud_thickness_percentile", 50))

    # ------------------------------------------------------------------
    # Runtime setters (for adaptive relaxation)
    # ------------------------------------------------------------------

    def set_adx_min(self, value: float) -> None:
        """Set ADX minimum threshold at runtime (for adaptive relaxation)."""
        self._adx_min = float(value)

    def get_adx_min(self) -> float:
        """Get current ADX minimum threshold."""
        return self._adx_min

    def should_allow(self, context: EdgeContext) -> EdgeResult:
        if not self.enabled:
            return self._disabled_result()

        adx_ok = context.adx >= self._adx_min

        cloud_thickness = context.indicator_values.get('cloud_thickness')
        if cloud_thickness is None:
            return EdgeResult(allowed=True, edge_name=self.name, reason="no cloud_thickness — filter skipped")

        cloud_ok = cloud_thickness >= self._cloud_min

        reasons: list[str] = []

        if adx_ok:
            reasons.append(f"ADX {context.adx:.1f} ≥ {self._adx_min:.0f}")
        else:
            reasons.append(f"ADX {context.adx:.1f} < {self._adx_min:.0f} (weak trend)")

        if cloud_ok:
            reasons.append(
                f"cloud thickness {cloud_thickness:.4f} ≥ {self._cloud_min:.4f}"
            )
        else:
            reasons.append(
                f"cloud thickness {cloud_thickness:.4f} < {self._cloud_min:.4f} "
                f"(thin cloud)"
            )

        return EdgeResult(
            allowed=adx_ok and cloud_ok,
            edge_name=self.name,
            reason="; ".join(reasons),
        )
