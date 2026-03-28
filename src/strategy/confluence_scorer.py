"""
Confluence scoring system for multi-timeframe Ichimoku signals.

Scores range from 0 to 9 across two categories:
  - Ichimoku alignment (0–5): cloud direction, TK alignment, TK cross, Chikou, entry
  - Confluence bonuses (0–4):  ADX trend strength, active session, zone proximity, divergence

Quality tiers derived from total score:
  - A+  (7–9): All conditions aligned, high confidence
  - B   (5–6): Most conditions met, moderate confidence
  - C   (4):   Minimum viable signal, low confidence
  - No trade: below 4
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .mtf_analyzer import MTFState


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ConfluenceResult:
    """Output of :class:`ConfluenceScorer`."""

    total_score: int          # 0–9
    tier: str                 # 'A+', 'B', 'C', or 'no_trade'
    ichimoku_score: int       # 0–5
    adx_bonus: bool
    session_bonus: bool
    zone_bonus: bool
    divergence_bonus: bool = False
    breakdown: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ConfluenceScorer
# ---------------------------------------------------------------------------

class ConfluenceScorer:
    """Score signal confluence on a 0–9 scale.

    Ichimoku component (0–5)
    ~~~~~~~~~~~~~~~~~~~~~~~~
    +1  4H cloud direction aligns with the trade direction
    +1  1H TK alignment (tenkan vs kijun relationship) aligns
    +1  15M TK cross in the signal direction
    +1  15M Chikou confirmation in the signal direction
    +1  5M price near Kijun (pullback entry timing)

    Confluence bonuses (0–4)
    ~~~~~~~~~~~~~~~~~~~~~~~~
    +1  ADX on 15M above ``adx_threshold`` (trending market)
    +1  Current session is London, New York, or their overlap
    +1  Price is near a significant supply/demand or S/R zone
    +1  RSI divergence aligns with trade direction

    Parameters
    ----------
    adx_threshold:
        Minimum ADX value considered trending.  Default: 28.
    min_score:
        Minimum total score required to generate a tradeable signal.  Default: 4.
    kijun_proximity_atr:
        5M entry is valid when price is within this many ATR of the 5M Kijun.
        Default: 0.5.
    """

    _ACTIVE_SESSIONS = {"london", "new_york", "overlap"}

    def __init__(
        self,
        adx_threshold: float = 28.0,
        min_score: int = 4,
        kijun_proximity_atr: float = 0.5,
    ) -> None:
        self.adx_threshold = adx_threshold
        self.min_score = min_score
        self.kijun_proximity_atr = kijun_proximity_atr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        mtf_state: MTFState,
        direction: str,
        zone_confluence: int = 0,
        divergence_signal: bool = False,
    ) -> ConfluenceResult:
        """Calculate the confluence score and quality tier.

        Parameters
        ----------
        mtf_state:
            Current multi-timeframe state snapshot.
        direction:
            Trade direction: 'long' or 'short'.
        zone_confluence:
            Number of nearby zones returned by the zone manager.  A value
            ≥ 1 awards the zone bonus point.
        divergence_signal:
            True when RSI divergence aligns with the trade direction
            (regular bullish for longs, regular bearish for shorts).
            Awards +1 confluence bonus.

        Returns
        -------
        ConfluenceResult
        """
        breakdown: dict = {}
        sign = 1 if direction == "long" else -1

        # ------------------------------------------------------------------
        # Ichimoku scoring (0–5)
        # ------------------------------------------------------------------

        # 1. 4H cloud direction aligns with trade direction
        cloud_4h_aligned = (mtf_state.state_4h.cloud_direction == sign)
        breakdown["4h_cloud_aligned"] = cloud_4h_aligned

        # 2. 1H TK alignment (tenkan > kijun for long, tenkan < kijun for short)
        # tk_cross is stored as alignment direction in the MTF state
        tk_1h_aligned = (mtf_state.state_1h.tk_cross == sign)
        breakdown["1h_tk_aligned"] = tk_1h_aligned

        # 3. 15M TK cross in the signal direction (actual cross event)
        tk_15m_cross = (mtf_state.state_15m.tk_cross == sign)
        breakdown["15m_tk_cross"] = tk_15m_cross

        # 4. 15M Chikou confirmation
        chikou_15m = (mtf_state.state_15m.chikou_confirmed == sign)
        breakdown["15m_chikou_confirmed"] = chikou_15m

        # 5. 5M pullback to Kijun (entry timing)
        near_kijun = self._near_kijun(
            close=mtf_state.close_5m,
            kijun=mtf_state.kijun_5m,
            atr=mtf_state.atr_15m,
        )
        breakdown["5m_near_kijun"] = near_kijun

        ichimoku_score = sum([
            cloud_4h_aligned,
            tk_1h_aligned,
            tk_15m_cross,
            chikou_15m,
            near_kijun,
        ])

        # ------------------------------------------------------------------
        # Confluence bonuses (0–3)
        # ------------------------------------------------------------------

        adx_bonus = (
            mtf_state.adx_15m >= self.adx_threshold
            and not _is_nan(mtf_state.adx_15m)
        )
        breakdown["adx_trending"] = adx_bonus
        breakdown["adx_value"] = mtf_state.adx_15m

        session_bonus = mtf_state.session in self._ACTIVE_SESSIONS
        breakdown["active_session"] = session_bonus
        breakdown["session"] = mtf_state.session

        zone_bonus = zone_confluence >= 1
        breakdown["zone_nearby"] = zone_bonus
        breakdown["zone_count"] = zone_confluence

        divergence_bonus = bool(divergence_signal)
        breakdown["divergence_aligned"] = divergence_bonus

        # ------------------------------------------------------------------
        # Total and tier
        # ------------------------------------------------------------------

        total_score = (
            ichimoku_score
            + int(adx_bonus)
            + int(session_bonus)
            + int(zone_bonus)
            + int(divergence_bonus)
        )
        tier = self._tier(total_score)

        breakdown["ichimoku_score"] = ichimoku_score
        breakdown["total_score"] = total_score
        breakdown["tier"] = tier
        breakdown["direction"] = direction

        return ConfluenceResult(
            total_score=total_score,
            tier=tier,
            ichimoku_score=ichimoku_score,
            adx_bonus=adx_bonus,
            session_bonus=session_bonus,
            zone_bonus=zone_bonus,
            divergence_bonus=divergence_bonus,
            breakdown=breakdown,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _near_kijun(self, close: float, kijun: float, atr: float) -> bool:
        """Return True when close is within kijun_proximity_atr × ATR of Kijun."""
        if _is_nan(close) or _is_nan(kijun) or _is_nan(atr) or atr <= 0:
            return False
        return abs(close - kijun) <= self.kijun_proximity_atr * atr

    def _tier(self, score: int) -> str:
        if score >= 7:
            return "A+"
        if score >= 5:
            return "B"
        if score >= self.min_score:
            return "C"
        return "no_trade"


def _is_nan(v: float) -> bool:
    import math
    try:
        return math.isnan(v)
    except (TypeError, ValueError):
        return True
