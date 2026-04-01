"""ML-enhanced sequence detection for the SSS strategy.

SSSSequenceMLLayer builds 64-dim feature vectors for SSS setups,
queries similar past setups via cosine similarity, and returns a
confidence adjustment in [-0.3, +0.3].  The layer is dormant until
``min_samples`` trades have been recorded (default 100).

Feature layout
--------------
[0-7]   Sequence state one-hot (5 dims) + direction + 2 reserved
[8-10]  Layer one-hot (SS / ISS / CBC)
[11-15] Candle counts (normalised)
[16-19] F2 history
[20-24] SS level features
[25-29] Entry confirmation
[30-35] Confluence flags
[36-41] Time (circular sin/cos pairs)
[42-47] Volatility
[48-53] Price action
[54-59] Historical performance
[60-63] Reserved (zeros)
"""

from __future__ import annotations

import logging
import math
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

_STATE_ORDER = [
    "scanning", "p2_detected", "two_confirmed",
    "three_active", "four_five_resolved",
]
_MAX_ADJ: float = 0.3
_MIN_ADJ: float = -0.3
_POS_R_THRESH: float = 0.5
_NEG_R_THRESH: float = -0.5
_TOP_K: int = 5


def _norm(value: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.0
    return (max(lo, min(hi, float(value))) - lo) / (hi - lo)


def _circ(value: float, period: float) -> tuple[float, float]:
    angle = 2.0 * math.pi * float(value) / float(period)
    return (math.sin(angle) + 1.0) / 2.0, (math.cos(angle) + 1.0) / 2.0


class SSSSequenceMLLayer:
    """ML-enhanced sequence detection using pgvector embeddings.

    Parameters
    ----------
    similarity_store:
        InMemorySimilarityStore instance.  None disables the layer entirely.
    min_samples:
        Minimum completed trades before ML activates.
    """

    def __init__(self, similarity_store=None, min_samples: int = 100) -> None:
        self._store = similarity_store
        self._min_samples = min_samples
        self._trade_count: int = 0

    @property
    def is_active(self) -> bool:
        """True when enough samples have been collected for ML to contribute."""
        return self._store is not None and self._trade_count >= self._min_samples

    def create_features(self, setup_context: dict) -> np.ndarray:
        """Create a 64-dim feature vector for an SSS trade setup.

        All values normalised to [0, 1].  Missing keys default to 0.
        """
        vec = np.zeros(64, dtype=np.float64)
        ctx = setup_context

        # [0-7] Sequence state + direction
        state = str(ctx.get("sequence_state") or "scanning").lower()
        if state in _STATE_ORDER:
            vec[_STATE_ORDER.index(state)] = 1.0
        vec[5] = 1.0 if str(ctx.get("direction") or "").lower() == "bullish" else 0.0

        # [8-10] Layer one-hot
        layer = str(ctx.get("layer") or "").upper()
        if layer == "SS":
            vec[8] = 1.0
        elif layer == "ISS":
            vec[9] = 1.0
        elif layer == "CBC":
            vec[10] = 1.0

        # [11-15] Candle counts
        vec[11] = _norm(ctx.get("ss_candle_count") or 0, 0, 100)
        vec[12] = _norm(ctx.get("iss_candle_count") or 0, 0, 20)
        vec[13] = _norm(ctx.get("swing_count") or 0, 0, 20)
        vec[14] = _norm(ctx.get("bars_in_state") or 0, 0, 50)
        vec[15] = _norm(ctx.get("warmup_ratio") or 0, 0, 1)

        # [16-19] F2 history
        vec[16] = _norm(ctx.get("f2_count_recent") or 0, 0, 10)
        vec[17] = _norm(ctx.get("f2_age_normalized") or 0, 0, 1)
        vec[18] = 1.0 if bool(ctx.get("f2_direction_match")) else 0.0
        vec[19] = 1.0 if bool(ctx.get("has_active_f2")) else 0.0

        # [20-24] SS level features
        vec[20] = _norm(ctx.get("nearest_target_distance_norm") or 0, 0, 1)
        vec[21] = _norm(ctx.get("target_rr") or 0, 0, 10)
        vec[22] = _norm(ctx.get("active_levels_count_norm") or 0, 0, 1)
        vec[23] = _norm(ctx.get("level_quality") or 0, 0, 1)
        vec[24] = _norm(ctx.get("level_age_norm") or 0, 0, 1)

        # [25-29] Entry confirmation
        cbc_map = {"none": 0.0, "pin": 0.33, "engulf": 0.67, "inside": 1.0}
        vec[25] = cbc_map.get(str(ctx.get("cbc_type") or "none").lower(), 0.0)
        vec[26] = 1.0 if bool(ctx.get("fifty_tap_status")) else 0.0
        vec[27] = _norm(ctx.get("tap_distance_norm") or 0, 0, 1)
        vec[28] = _norm(ctx.get("cbc_confidence") or 0, 0, 1)
        entry_map = {"standard": 0.0, "fifty_tap": 0.5, "aggressive": 1.0}
        vec[29] = entry_map.get(str(ctx.get("entry_mode") or "standard").lower(), 0.0)

        # [30-35] Confluence
        vec[30] = _norm(ctx.get("total_score_norm") or 0, 0, 1)
        vec[31] = 1.0 if bool(ctx.get("ss_aligned")) else 0.0
        vec[32] = 1.0 if bool(ctx.get("iss_aligned")) else 0.0
        vec[33] = 1.0 if bool(ctx.get("cbc_confirmed")) else 0.0
        vec[34] = 1.0 if bool(ctx.get("tap_confirmed")) else 0.0
        vec[35] = 1.0 if bool(ctx.get("no_f2_conflict")) else 0.0

        # [36-41] Time (circular)
        vec[36], vec[37] = _circ(float(ctx.get("hour") or 0), 24.0)
        vec[38], vec[39] = _circ(float(ctx.get("day_of_week") or 0), 5.0)
        vec[40], vec[41] = _circ(float(ctx.get("minute") or 0), 60.0)

        # [42-47] Volatility
        atr = float(ctx.get("atr_value") or ctx.get("atr") or 1.0) or 1.0
        atr_ref = float(ctx.get("atr_reference") or atr) or atr
        vec[42] = _norm(atr / atr_ref, 0, 3)
        vec[43] = _norm(float(ctx.get("spread") or 0) / atr, 0, 0.5)
        vec[44] = _norm(ctx.get("atr_percentile") or 0, 0, 1)
        vec[45] = _norm(ctx.get("spread_percentile") or 0, 0, 1)
        vec[46] = _norm(float(ctx.get("daily_range") or 0) / atr, 0, 5)
        vec[47] = _norm(float(ctx.get("body_size") or 0) / atr, 0, 1)

        # [48-53] Price action
        vec[48] = _norm(ctx.get("trend_strength") or 0, 0, 1)
        vec[49] = _norm(ctx.get("momentum") or 0, -1, 1)
        vec[50] = _norm(ctx.get("swing_amplitude") or 0, 0, atr * 10)
        vec[51] = _norm(ctx.get("price_vs_ss_level") or 0, -1, 1)
        vec[52] = _norm(ctx.get("retrace_depth") or 0, 0, 1)
        vec[53] = _norm(ctx.get("extension_ratio") or 0, 0, 3)

        # [54-59] Historical performance
        vec[54] = _norm(ctx.get("win_streak") or 0, 0, 10)
        vec[55] = _norm(ctx.get("loss_streak") or 0, 0, 10)
        vec[56] = _norm(ctx.get("rolling_win_rate") or 0, 0, 1)
        vec[57] = _norm(ctx.get("time_since_last_trade") or 0, 0, 1440)
        vec[58] = _norm(ctx.get("avg_r_recent") or 0, -3, 3)
        vec[59] = _norm(ctx.get("max_drawdown_recent") or 0, 0, 1)

        # [60-63]: reserved → 0.0

        np.clip(vec, 0.0, 1.0, out=vec)
        vec = np.where(np.isfinite(vec), vec, 0.0)
        return vec

    def get_confidence_adjustment(self, setup_context: dict) -> float:
        """Query similar past setups and return confidence adjustment.

        Returns 0.0 when the ML layer is not yet active (< min_samples).
        Returns float in [-0.3, +0.3].
        """
        if not self.is_active:
            return 0.0

        query_vec = self.create_features(setup_context)
        try:
            similar: List = self._store.find_similar_trades(
                context_embedding=query_vec, k=_TOP_K, min_similarity=0.70,
            )
        except Exception as exc:
            logger.warning("SSSSequenceMLLayer similarity query failed: %s", exc)
            return 0.0

        if not similar:
            return 0.0

        total_weight = sum(float(getattr(t, "similarity", 0.0)) for t in similar)
        if total_weight == 0.0:
            return 0.0

        avg_r = sum(
            float(getattr(t, "similarity", 0.0)) * float(getattr(t, "r_multiple", 0.0))
            for t in similar
        ) / total_weight

        if avg_r > _POS_R_THRESH:
            return round(min(_MAX_ADJ, (avg_r - _POS_R_THRESH) * _MAX_ADJ), 4)
        if avg_r < _NEG_R_THRESH:
            return round(max(_MIN_ADJ, (avg_r - _NEG_R_THRESH) * _MAX_ADJ), 4)
        return 0.0

    def record_outcome(
        self, setup_context: dict, r_multiple: float, won: bool
    ) -> None:
        """Record a trade outcome for future similarity lookups."""
        if self._store is None:
            return
        embedding = self.create_features(setup_context)
        try:
            self._store.record_trade(
                embedding=embedding,
                r_multiple=float(r_multiple),
                context={"won": won, **setup_context},
            )
            self._trade_count += 1
            logger.debug(
                "SSSSequenceMLLayer: recorded trade #%d r=%.3f won=%s",
                self._trade_count, r_multiple, won,
            )
        except Exception as exc:
            logger.warning("SSSSequenceMLLayer.record_outcome failed: %s", exc)
