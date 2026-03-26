"""
64-dimensional feature vector builder for trade context.

Feature layout
--------------
Ichimoku State      dims  0–14   (15 dims, 3 reserved)
Trend/Momentum      dims 15–24   (10 dims, 2 reserved)
Zone Context        dims 25–34   (10 dims, 3 reserved)
Session/Time        dims 35–44   (10 dims, 2 reserved)
Signal Quality      dims 45–54   (10 dims, 5 reserved)
Market Regime       dims 55–63   (9 dims, 5 reserved)

All continuous features are normalised to [0, 1].
Cyclical features (hour, day-of-week) use sin/cos pairs.
Reserved dimensions are set to 0.0 for forward compatibility.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Ordinal lookup tables
# ---------------------------------------------------------------------------

# cloud_direction and tk_cross use -1/0/1
_DIRECTION_MAP: Dict[int, float] = {-1: 0.0, 0: 0.5, 1: 1.0}

# cloud_position (price vs cloud): below=-1, inside=0, above=1
_POSITION_MAP: Dict[int, float] = {-1: 0.0, 0: 0.5, 1: 1.0}

# zone freshness: "untested", "tested", "invalidated"
_FRESHNESS_MAP: Dict[str, float] = {
    "active":      1.0,
    "untested":    1.0,
    "tested":      0.5,
    "invalidated": 0.0,
}

# signal_tier: no_trade, C, B, A+
_TIER_MAP: Dict[str, float] = {
    "no_trade": 0.0,
    "C":        0.33,
    "B":        0.66,
    "A+":       1.0,
    "A":        1.0,    # treat "A" as "A+" when encountered
}

# volatility regime
_VOL_REGIME_MAP: Dict[str, float] = {
    "low":    0.0,
    "normal": 0.5,
    "high":   1.0,
}

# session names → boolean flags (london, ny, asian, overlap)
_SESSION_LONDON  = {"london"}
_SESSION_NY      = {"new_york", "ny", "new york", "newyork"}
_SESSION_ASIAN   = {"asian", "asia", "tokyo"}
_SESSION_OVERLAP = {"london_ny", "london/ny", "overlap", "london_new_york"}


class FeatureVectorBuilder:
    """Build 64-dimensional feature vectors from trade context dictionaries.

    The context dictionary should contain keys matching MarketContext and Signal
    field names.  Missing keys are silently defaulted to 0.0 so that partial
    contexts (e.g. live snapshots with incomplete indicator data) still produce
    a well-formed vector.

    All output values lie in [0, 1].
    """

    VECTOR_DIM: int = 64

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, context: Dict[str, Any]) -> np.ndarray:
        """Build a 64-dim feature vector from *context*.

        Parameters
        ----------
        context:
            Dictionary with trade / market context fields.  Keys may be absent;
            missing values default to 0.0.

        Returns
        -------
        np.ndarray of shape (64,) with dtype float64, all values in [0, 1].
        """
        vec = np.zeros(self.VECTOR_DIM, dtype=np.float64)

        atr = float(context.get("atr_value") or context.get("atr") or 1.0)
        if atr <= 0.0:
            atr = 1.0

        # ------------------------------------------------------------------
        # Ichimoku State  (dims 0–14)
        # ------------------------------------------------------------------

        # dim 0: cloud_direction_4h  ordinal -1/0/1
        vec[0] = self._encode_ordinal(
            context.get("cloud_direction_4h"), _DIRECTION_MAP
        )
        # dim 1: cloud_direction_1h  ordinal
        vec[1] = self._encode_ordinal(
            context.get("cloud_direction_1h"), _DIRECTION_MAP
        )
        # dim 2: tk_cross_15m  ordinal
        vec[2] = self._encode_ordinal(
            context.get("tk_cross_15m"), _DIRECTION_MAP
        )
        # dim 3: chikou_confirmed  boolean
        vec[3] = self._encode_bool(
            context.get("chikou_confirmation") or context.get("chikou_confirmed")
        )
        # dim 4: cloud_thickness_4h  normalised by ATR (clamp 0–1)
        raw_thick_4h = context.get("cloud_thickness_4h") or 0.0
        vec[4] = self._normalize_continuous(float(raw_thick_4h) / atr, 0.0, 3.0)

        # dim 5: cloud_thickness_1h  normalised by ATR
        raw_thick_1h = context.get("cloud_thickness_1h") or 0.0
        vec[5] = self._normalize_continuous(float(raw_thick_1h) / atr, 0.0, 3.0)

        # dim 6: cloud_position_15m  ordinal
        vec[6] = self._encode_ordinal(
            context.get("cloud_position_15m"), _POSITION_MAP
        )
        # dim 7: cloud_position_5m  ordinal
        vec[7] = self._encode_ordinal(
            context.get("cloud_position_5m"), _POSITION_MAP
        )
        # dim 8: kijun_distance_5m  (price – kijun) / ATR, normalised
        kijun_dist = context.get("kijun_distance_5m") or 0.0
        vec[8] = self._normalize_continuous(float(kijun_dist) / atr, -3.0, 3.0)

        # dim 9: tenkan_kijun_spread  (tenkan – kijun) / ATR
        tk_spread = context.get("tenkan_kijun_spread") or 0.0
        vec[9] = self._normalize_continuous(float(tk_spread) / atr, -3.0, 3.0)

        # dim 10: cloud_twist_15m  boolean
        vec[10] = self._encode_bool(context.get("cloud_twist_15m"))

        # dim 11: cloud_breakout_15m  boolean
        vec[11] = self._encode_bool(context.get("cloud_breakout_15m"))

        # dims 12–14: reserved → 0.0

        # ------------------------------------------------------------------
        # Trend / Momentum  (dims 15–24)
        # ------------------------------------------------------------------

        # dim 15: adx_value  normalised (adx / 100)
        adx = float(context.get("adx_value") or 0.0)
        vec[15] = self._normalize_continuous(adx / 100.0, 0.0, 1.0)

        # dim 16: adx_trending  boolean (ADX > 25)
        adx_threshold = float(context.get("adx_threshold") or 25.0)
        vec[16] = 1.0 if adx >= adx_threshold else 0.0

        # dim 17: rsi_value  normalised (rsi / 100)
        rsi = float(context.get("rsi_value") or 50.0)
        vec[17] = self._normalize_continuous(rsi / 100.0, 0.0, 1.0)

        # dim 18: rsi_overbought  boolean
        vec[18] = 1.0 if rsi >= 70.0 else 0.0

        # dim 19: rsi_oversold  boolean
        vec[19] = 1.0 if rsi <= 30.0 else 0.0

        # dim 20: bb_width_percentile  continuous [0, 1]
        bb_pct = float(context.get("bb_width_percentile") or 0.0)
        vec[20] = self._normalize_continuous(bb_pct, 0.0, 1.0)

        # dim 21: bb_squeeze  boolean
        vec[21] = self._encode_bool(context.get("bb_squeeze"))

        # dim 22: atr_normalised  current ATR / 20-period MA of ATR
        atr_ma = float(context.get("atr_ma") or context.get("atr_20_ma") or atr)
        if atr_ma <= 0.0:
            atr_ma = atr
        vec[22] = self._normalize_continuous(atr / atr_ma, 0.0, 3.0)

        # dims 23–24: reserved → 0.0

        # ------------------------------------------------------------------
        # Zone Context  (dims 25–34)
        # ------------------------------------------------------------------

        # dim 25: nearest_sr_distance  normalised by ATR, clamped 0–1
        sr_dist = float(context.get("nearest_sr_distance") or 0.0)
        vec[25] = self._normalize_continuous(sr_dist / atr, 0.0, 5.0)

        # dim 26: zone_confluence_count  ordinal 0–5+ → 0–1
        zone_count = float(
            context.get("zone_confluence_count")
            or context.get("zone_confluence_score")
            or 0
        )
        vec[26] = self._normalize_continuous(zone_count, 0.0, 5.0)

        # dim 27: in_supply_zone  boolean
        vec[27] = self._encode_bool(context.get("in_supply_zone"))

        # dim 28: in_demand_zone  boolean
        vec[28] = self._encode_bool(context.get("in_demand_zone"))

        # dim 29: at_pivot_level  boolean
        vec[29] = self._encode_bool(context.get("at_pivot_level"))

        # dim 30: zone_strength  continuous [0, 1]
        zone_strength = float(context.get("zone_strength") or 0.0)
        vec[30] = self._normalize_continuous(zone_strength, 0.0, 1.0)

        # dim 31: zone_freshness  ordinal
        freshness_raw = context.get("zone_freshness") or context.get("zone_status")
        vec[31] = _FRESHNESS_MAP.get(str(freshness_raw).lower(), 0.5) if freshness_raw else 0.5

        # dims 32–34: reserved → 0.0

        # ------------------------------------------------------------------
        # Session / Time  (dims 35–44)
        # ------------------------------------------------------------------

        session_str = str(context.get("session") or "").lower().strip()

        # dim 35: session_london
        vec[35] = 1.0 if session_str in _SESSION_LONDON else 0.0
        # dim 36: session_ny
        vec[36] = 1.0 if session_str in _SESSION_NY else 0.0
        # dim 37: session_asian
        vec[37] = 1.0 if session_str in _SESSION_ASIAN else 0.0
        # dim 38: session_overlap
        vec[38] = 1.0 if session_str in _SESSION_OVERLAP else 0.0

        # dim 39–40: hour_of_day (circular)
        hour = float(context.get("hour") or context.get("hour_of_day") or 0.0)
        sin_h, cos_h = self._encode_circular(hour, 24.0)
        vec[39] = (sin_h + 1.0) / 2.0   # shift sin/cos from [-1,1] to [0,1]
        vec[40] = (cos_h + 1.0) / 2.0

        # dim 41–42: day_of_week (circular) — 0=Monday … 4=Friday
        dow = float(context.get("day_of_week") or 0.0)
        sin_d, cos_d = self._encode_circular(dow, 5.0)
        vec[41] = (sin_d + 1.0) / 2.0
        vec[42] = (cos_d + 1.0) / 2.0

        # dims 43–44: reserved → 0.0

        # ------------------------------------------------------------------
        # Signal Quality  (dims 45–54)
        # ------------------------------------------------------------------

        # dim 45: confluence_score  normalised (score / 8)
        conf_score = float(context.get("confluence_score") or 0.0)
        vec[45] = self._normalize_continuous(conf_score / 8.0, 0.0, 1.0)

        # dim 46: signal_tier  ordinal
        tier_str = str(context.get("signal_tier") or context.get("quality_tier") or "no_trade")
        vec[46] = _TIER_MAP.get(tier_str, 0.0)

        # dim 47: direction  binary long=1, short=0
        direction_str = str(context.get("direction") or "").lower()
        vec[47] = 1.0 if direction_str == "long" else 0.0

        # dim 48: risk_pct  normalised — typical range [0, 2 %]
        risk_pct = float(context.get("risk_pct") or 0.0)
        vec[48] = self._normalize_continuous(risk_pct, 0.0, 2.0)

        # dim 49: atr_stop_distance  stop distance normalised by ATR
        stop_dist = float(context.get("atr_stop_distance") or 0.0)
        vec[49] = self._normalize_continuous(stop_dist / atr, 0.0, 5.0)

        # dims 50–54: reserved → 0.0

        # ------------------------------------------------------------------
        # Market Regime  (dims 55–63)
        # ------------------------------------------------------------------

        # dim 55: trend_strength  ADX-based, normalised [0, 1]
        vec[55] = self._normalize_continuous(adx / 100.0, 0.0, 1.0)

        # dim 56: volatility_regime  ordinal
        vol_regime = str(context.get("volatility_regime") or "normal").lower()
        vec[56] = _VOL_REGIME_MAP.get(vol_regime, 0.5)

        # dim 57: spread_normalised  spread / ATR
        spread = float(context.get("spread") or 0.0)
        vec[57] = self._normalize_continuous(spread / atr, 0.0, 0.5)

        # dim 58: daily_range_vs_atr  daily_range / ATR
        daily_range = float(context.get("daily_range") or 0.0)
        vec[58] = self._normalize_continuous(daily_range / atr, 0.0, 5.0)

        # dims 59–63: reserved → 0.0

        # Guarantee all values are finite and within [0, 1]
        np.clip(vec, 0.0, 1.0, out=vec)
        vec = np.where(np.isfinite(vec), vec, 0.0)

        return vec

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_continuous(value: float, min_val: float, max_val: float) -> float:
        """Clamp *value* to [min_val, max_val] then scale to [0, 1].

        When min_val == max_val the function returns 0.0 rather than
        raising a division-by-zero error.
        """
        if max_val == min_val:
            return 0.0
        clamped = max(min_val, min(max_val, float(value)))
        return (clamped - min_val) / (max_val - min_val)

    @staticmethod
    def _encode_ordinal(value: Any, mapping: Dict) -> float:
        """Map an ordinal value to [0, 1] using *mapping*.

        Attempts integer coercion first (to handle string representations
        such as "1" or "-1").  Falls back to 0.0 when the value is absent
        or not found in the mapping.
        """
        if value is None:
            return 0.0
        # Try direct lookup first
        if value in mapping:
            return mapping[value]
        # Try integer coercion (handles "1", "-1", 1.0, etc.)
        try:
            return mapping[int(float(value))]
        except (KeyError, TypeError, ValueError):
            return 0.0

    @staticmethod
    def _encode_bool(value: Any) -> float:
        """Encode a boolean-ish value as 1.0 (truthy) or 0.0 (falsy)."""
        if value is None:
            return 0.0
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return 1.0 if value else 0.0
        # String representations
        return 1.0 if str(value).lower() in ("true", "1", "yes") else 0.0

    @staticmethod
    def _encode_circular(value: float, period: float) -> Tuple[float, float]:
        """Encode a cyclical feature as a (sin, cos) pair.

        Both values lie in [-1, 1]; callers should shift to [0, 1] when
        storing in the feature vector.

        Parameters
        ----------
        value:
            The raw value (e.g. hour 0–23, day-of-week 0–4).
        period:
            The full cycle length (e.g. 24 for hours, 5 for weekdays).
        """
        angle = 2.0 * math.pi * float(value) / float(period)
        return math.sin(angle), math.cos(angle)
