"""IchimokuStrategy — wraps the existing Ichimoku MTF cascade as a Strategy subclass."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from ..base import (
    Strategy,
    EvalMatrix,
    EvalRequirement,
    ConfluenceResult as BaseConfluenceResult,
    EvaluatorResult,
)
from ..signal_engine import Signal
from ..trading_modes.kijun_exit import KijunExitMode
from ...zones.manager import ZoneManager


def _is_nan(v: float) -> bool:
    try:
        return math.isnan(v)
    except (TypeError, ValueError):
        return True


class IchimokuStrategy(Strategy, key='ichimoku'):
    """MTF Ichimoku strategy: 4H->1H->15M->5M cascade (default) or 15M->5M mode.

    Replicates the exact signal generation logic from SignalEngine.scan()
    and confluence scoring from ConfluenceScorer.score().

    When ``config["timeframes"]`` is ``["15M", "5M"]`` the 4H and 1H gates
    are dropped and the 15M cloud direction is used as the primary trend
    filter, producing significantly more signals for prop-firm trading.
    """

    required_evaluators = [
        EvalRequirement('ichimoku', ['4H', '1H', '15M', '5M']),
        EvalRequirement('adx', ['15M']),
        EvalRequirement('atr', ['15M']),
        EvalRequirement('session', ['5M']),
    ]

    config_model = None  # Will be set to IchimokuConfig after import
    trading_mode = None  # Set in __init__
    warmup_bars = 0  # Will be calculated

    # Confluence constants
    _ACTIVE_SESSIONS = {"london", "new_york", "overlap"}

    def __init__(self, config=None, instrument: str = "XAUUSD") -> None:
        # Default config values matching _DEFAULT_CONFIG in signal_engine.py
        self._cfg = {
            "tenkan_period": 9,
            "kijun_period": 26,
            "senkou_b_period": 52,
            "adx_threshold": 28,
            "adx_period": 14,
            "atr_period": 14,
            "atr_stop_multiplier": 1.5,
            "min_confluence_score": 4,
            "rr_ratio": 2.0,
            "kijun_proximity_atr": 0.5,
        }

        if config is not None:
            # Config can be a Pydantic model or a dict
            if hasattr(config, 'model_dump'):
                config_dict = config.model_dump()
            elif isinstance(config, dict):
                config_dict = config
            else:
                config_dict = {}
            self._cfg.update(config_dict)

        self.instrument = instrument
        self.trading_mode = KijunExitMode()
        self.zone_manager = ZoneManager()

        # Determine mode from configured timeframes
        timeframes = self._cfg.get("timeframes", ["4H", "1H", "15M", "5M"])
        self._15m_only = ("4H" not in timeframes and "1H" not in timeframes)

        # Dynamic required_evaluators based on mode
        if self._15m_only:
            self.required_evaluators = [
                EvalRequirement('ichimoku', ['15M', '5M']),
                EvalRequirement('adx', ['15M']),
                EvalRequirement('atr', ['15M']),
                EvalRequirement('session', ['5M']),
            ]
            # 15M = 60 1M bars, kijun needs 26 bars: 26 * 60 = 1560
            self.warmup_bars = self._cfg["kijun_period"] * 60
        else:
            self.required_evaluators = [
                EvalRequirement('ichimoku', ['4H', '1H', '15M', '5M']),
                EvalRequirement('adx', ['15M']),
                EvalRequirement('atr', ['15M']),
                EvalRequirement('session', ['5M']),
            ]
            # 4H = 240 1M bars, kijun needs 26 bars: 26 * 240 = 6240
            self.warmup_bars = self._cfg["kijun_period"] * 240

    # ------------------------------------------------------------------
    # Strategy ABC: decide()
    # ------------------------------------------------------------------

    def decide(self, eval_matrix: EvalMatrix) -> Optional[Signal]:
        """4H->1H->15M->5M cascade (default) or 15M->5M mode. Returns Signal or None.

        This replicates the exact logic from SignalEngine.scan().
        In 15M-only mode, the 4H and 1H gates are skipped and the 15M cloud
        direction is used as the primary trend filter.
        """
        reasoning: dict = {}

        if self._15m_only:
            direction, reasoning = self._decide_15m_trend(eval_matrix, reasoning)
        else:
            direction, reasoning = self._decide_4h_cascade(eval_matrix, reasoning)

        if direction is None:
            return None

        # --- Step 4: 5M entry timing (pullback to Kijun) ---
        ichi_5m = eval_matrix.get('ichimoku_5M')
        if ichi_5m is None:
            return None

        close_5m = ichi_5m.metadata.get('close', 0.0)
        kijun_5m = ichi_5m.metadata.get('kijun', float('nan'))

        atr_result = eval_matrix.get('atr_15M')
        atr = atr_result.metadata.get('atr', 0.0) if atr_result else 0.0

        if _is_nan(close_5m) or _is_nan(kijun_5m) or _is_nan(atr) or atr <= 0:
            return None

        proximity = self._cfg["kijun_proximity_atr"] * atr
        if abs(close_5m - kijun_5m) > proximity:
            return None  # Too far from Kijun

        reasoning["5m_entry"] = {"pass": True, "close": close_5m, "kijun": kijun_5m}

        # --- Step 5: Zone context ---
        nearby_zones = self.zone_manager.get_nearby_zones(
            price=close_5m,
            atr=atr if atr > 0 else 1.0,
            max_distance_atr=2.0,
        )
        zone_count = len(nearby_zones)
        zone_context = {
            "nearby_zone_count": zone_count,
            "zones": [
                {
                    "type": getattr(z, "zone_type", "unknown"),
                    "high": float(getattr(z, "price_high", 0.0)),
                    "low": float(getattr(z, "price_low", 0.0)),
                }
                for z in nearby_zones[:5]
            ],
        }

        # --- Step 6: Confluence scoring ---
        confluence = self.score_confluence(eval_matrix, direction=direction, zone_count=zone_count)

        if confluence.quality_tier == "no_trade":
            return None

        reasoning["confluence"] = confluence.breakdown

        # --- Step 7: Calculate levels ---
        entry_price = close_5m
        sl_distance = atr * self._cfg["atr_stop_multiplier"]
        tp_distance = sl_distance * self._cfg["rr_ratio"]

        if direction == "long":
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance

        # Get session from eval_matrix
        session_result = eval_matrix.get('session_5M')
        session = session_result.metadata.get('session', 'unknown') if session_result else 'unknown'

        # Get timestamp from 5M data metadata
        timestamp = ichi_5m.metadata.get('timestamp', datetime.utcnow())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        reasoning["session"] = session
        reasoning["levels"] = {
            "stop_loss": round(stop_loss, 5),
            "take_profit": round(take_profit, 5),
            "risk_pips": round(sl_distance, 5),
            "reward_pips": round(tp_distance, 5),
        }

        return Signal(
            timestamp=timestamp if isinstance(timestamp, datetime) else datetime.utcnow(),
            instrument=self.instrument,
            direction=direction,
            entry_price=entry_price,
            stop_loss=round(stop_loss, 5),
            take_profit=round(take_profit, 5),
            confluence_score=confluence.score,
            quality_tier=confluence.quality_tier,
            atr=atr,
            reasoning=reasoning,
            zone_context=zone_context,
        )

    # ------------------------------------------------------------------
    # Private: 4H->1H->15M cascade (original mode)
    # ------------------------------------------------------------------

    def _decide_4h_cascade(
        self, eval_matrix: EvalMatrix, reasoning: dict
    ) -> tuple[Optional[str], dict]:
        """Original 4H->1H->15M cascade filter. Returns (direction, reasoning)."""

        # --- Step 1: 4H cloud direction (hard filter) ---
        ichi_4h = eval_matrix.get('ichimoku_4H')
        if ichi_4h is None:
            return None, reasoning

        cloud_dir_4h = ichi_4h.metadata.get('cloud_direction', 0)
        if cloud_dir_4h == 1:
            direction = "long"
        elif cloud_dir_4h == -1:
            direction = "short"
        else:
            return None, reasoning  # Flat/neutral -- no trade

        reasoning["4h_filter"] = {
            "pass": True,
            "direction": direction,
            "reason": f"4H cloud is {'bullish' if direction == 'long' else 'bearish'}",
        }

        # --- Step 2: 1H TK confirmation ---
        ichi_1h = eval_matrix.get('ichimoku_1H')
        if ichi_1h is None:
            return None, reasoning

        sign = 1 if direction == "long" else -1
        tk_1h = ichi_1h.metadata.get('tk_cross', 0)
        if tk_1h != sign:
            return None, reasoning  # 1H TK not confirming

        reasoning["1h_confirmation"] = {"pass": True, "direction": direction}

        # --- Step 3: 15M signal (TK cross + cloud position + chikou) ---
        ichi_15m = eval_matrix.get('ichimoku_15M')
        if ichi_15m is None:
            return None, reasoning

        tk_15m = ichi_15m.metadata.get('tk_cross', 0)
        cloud_pos_15m = ichi_15m.metadata.get('cloud_position', 0)
        chikou_15m = ichi_15m.metadata.get('chikou_confirmed', 0)

        if tk_15m != sign or cloud_pos_15m != sign or chikou_15m != sign:
            return None, reasoning  # 15M conditions not met

        reasoning["15m_signal"] = {"pass": True}

        return direction, reasoning

    # ------------------------------------------------------------------
    # Private: 15M-only trend filter (no 4H/1H gates)
    # ------------------------------------------------------------------

    def _decide_15m_trend(
        self, eval_matrix: EvalMatrix, reasoning: dict
    ) -> tuple[Optional[str], dict]:
        """15M cloud direction as sole trend filter. Returns (direction, reasoning)."""

        ichi_15m = eval_matrix.get('ichimoku_15M')
        if ichi_15m is None:
            return None, reasoning

        cloud_dir_15m = ichi_15m.metadata.get('cloud_direction', 0)
        if cloud_dir_15m == 1:
            direction = "long"
        elif cloud_dir_15m == -1:
            direction = "short"
        else:
            return None, reasoning  # Flat/neutral -- no trade

        reasoning["15m_trend_filter"] = {
            "pass": True,
            "direction": direction,
            "reason": f"15M cloud is {'bullish' if direction == 'long' else 'bearish'}",
        }
        reasoning["15m_signal"] = {"pass": True}

        return direction, reasoning

    # ------------------------------------------------------------------
    # Strategy ABC: score_confluence()
    # ------------------------------------------------------------------

    def score_confluence(
        self,
        eval_matrix: EvalMatrix,
        direction: str = "long",
        zone_count: int = 0,
        divergence_signal: bool = False,
    ) -> BaseConfluenceResult:
        """Replicate ConfluenceScorer.score() logic using EvalMatrix data.

        Default mode (0-9 scale):
          Ichimoku (0-5): 4H cloud, 1H TK, 15M TK, 15M chikou, 5M near kijun
          Bonuses (0-4): ADX trending, active session, zone nearby, divergence

        15M-only mode (0-7 scale):
          Ichimoku (0-5): 15M cloud, 15M TK, 15M chikou, 5M cloud, 5M near kijun
          Bonuses (0-2): ADX trending, active session
        """
        breakdown: dict = {}
        sign = 1 if direction == "long" else -1

        ichi_15m = eval_matrix.get('ichimoku_15M')
        ichi_5m = eval_matrix.get('ichimoku_5M')

        if self._15m_only:
            # 15M-only scoring (0-5): 15M cloud + 15M TK + 15M chikou
            #                         + 5M cloud + 5M near kijun
            cloud_15m_aligned = (
                ichi_15m.metadata.get('cloud_direction', 0) == sign
            ) if ichi_15m else False
            breakdown["15m_cloud_aligned"] = cloud_15m_aligned

            tk_15m_cross = (
                ichi_15m.metadata.get('tk_cross', 0) == sign
            ) if ichi_15m else False
            breakdown["15m_tk_cross"] = tk_15m_cross

            chikou_15m = (
                ichi_15m.metadata.get('chikou_confirmed', 0) == sign
            ) if ichi_15m else False
            breakdown["15m_chikou_confirmed"] = chikou_15m

            cloud_5m_aligned = (
                ichi_5m.metadata.get('cloud_direction', 0) == sign
            ) if ichi_5m else False
            breakdown["5m_cloud_aligned"] = cloud_5m_aligned

        else:
            # Original scoring: 4H cloud + 1H TK + 15M TK + 15M chikou
            ichi_4h = eval_matrix.get('ichimoku_4H')
            cloud_4h_aligned = (
                ichi_4h.metadata.get('cloud_direction', 0) == sign
            ) if ichi_4h else False
            breakdown["4h_cloud_aligned"] = cloud_4h_aligned

            ichi_1h = eval_matrix.get('ichimoku_1H')
            tk_1h_aligned = (
                ichi_1h.metadata.get('tk_cross', 0) == sign
            ) if ichi_1h else False
            breakdown["1h_tk_aligned"] = tk_1h_aligned

            tk_15m_cross = (
                ichi_15m.metadata.get('tk_cross', 0) == sign
            ) if ichi_15m else False
            breakdown["15m_tk_cross"] = tk_15m_cross

            chikou_15m = (
                ichi_15m.metadata.get('chikou_confirmed', 0) == sign
            ) if ichi_15m else False
            breakdown["15m_chikou_confirmed"] = chikou_15m

        # 5M near kijun (shared by both modes)
        close_5m = ichi_5m.metadata.get('close', 0.0) if ichi_5m else 0.0
        kijun_5m = ichi_5m.metadata.get('kijun', float('nan')) if ichi_5m else float('nan')

        atr_result = eval_matrix.get('atr_15M')
        atr_val = atr_result.metadata.get('atr', 0.0) if atr_result else 0.0

        near_kijun = False
        if not _is_nan(close_5m) and not _is_nan(kijun_5m) and not _is_nan(atr_val) and atr_val > 0:
            near_kijun = abs(close_5m - kijun_5m) <= self._cfg["kijun_proximity_atr"] * atr_val
        breakdown["5m_near_kijun"] = near_kijun

        if self._15m_only:
            ichimoku_score = sum([
                cloud_15m_aligned,
                tk_15m_cross,
                chikou_15m,
                cloud_5m_aligned,
                near_kijun,
            ])
        else:
            ichimoku_score = sum([
                cloud_4h_aligned,
                tk_1h_aligned,
                tk_15m_cross,
                chikou_15m,
                near_kijun,
            ])

        # Bonuses (0-4)
        adx_result = eval_matrix.get('adx_15M')
        adx_val = adx_result.metadata.get('adx', 0.0) if adx_result else 0.0
        adx_bonus = (adx_val >= self._cfg["adx_threshold"] and not _is_nan(adx_val))
        breakdown["adx_trending"] = adx_bonus
        breakdown["adx_value"] = adx_val

        session_result = eval_matrix.get('session_5M')
        session = session_result.metadata.get('session', '') if session_result else ''
        session_bonus = session in self._ACTIVE_SESSIONS
        breakdown["active_session"] = session_bonus
        breakdown["session"] = session

        zone_bonus = zone_count >= 1
        breakdown["zone_nearby"] = zone_bonus

        divergence_bonus = bool(divergence_signal)
        breakdown["divergence_aligned"] = divergence_bonus

        total_score = (
            ichimoku_score
            + int(adx_bonus)
            + int(session_bonus)
            + int(zone_bonus)
            + int(divergence_bonus)
        )

        # Tiers
        if total_score >= 7:
            tier = "A+"
        elif total_score >= 5:
            tier = "B"
        elif total_score >= self._cfg["min_confluence_score"]:
            tier = "C"
        else:
            tier = "no_trade"

        breakdown["ichimoku_score"] = ichimoku_score
        breakdown["total_score"] = total_score
        breakdown["tier"] = tier
        breakdown["direction"] = direction

        return BaseConfluenceResult(
            score=total_score,
            quality_tier=tier,
            breakdown=breakdown,
        )

    # ------------------------------------------------------------------
    # Strategy ABC: suggest_params()
    # ------------------------------------------------------------------

    def suggest_params(self, trial) -> dict:
        """Optuna parameter space for Ichimoku strategy."""
        _BASE_TENKAN = 9
        _BASE_KIJUN = 26
        _BASE_SENKOU_B = 52

        scale = trial.suggest_float("ichimoku_scale", 0.7, 1.3)
        tenkan = max(3, round(_BASE_TENKAN * scale))
        kijun = max(9, round(_BASE_KIJUN * scale))
        senkou_b = max(18, round(_BASE_SENKOU_B * scale))

        return {
            "tenkan_period": tenkan,
            "kijun_period": kijun,
            "senkou_b_period": senkou_b,
            "adx_threshold": trial.suggest_int("adx_threshold", 20, 40),
            "atr_stop_multiplier": trial.suggest_float("atr_stop_mult", 1.0, 2.5),
            "tp_r_multiple": trial.suggest_float("tp_r_multiple", 1.5, 3.0),
            "kijun_trail_start_r": trial.suggest_float("kijun_trail_start_r", 1.0, 2.5),
            "min_confluence_score": trial.suggest_int("min_confluence", 3, 6),
            "initial_risk_pct": trial.suggest_float("risk_initial", 0.5, 2.0),
            "reduced_risk_pct": trial.suggest_float("risk_reduced", 0.25, 1.5),
        }

    # ------------------------------------------------------------------
    # Strategy ABC: populate_edge_context()
    # ------------------------------------------------------------------

    def populate_edge_context(self, eval_matrix: EvalMatrix) -> dict:
        """Return kijun and cloud_thickness for EdgeContext.indicator_values."""
        result = {}

        # Kijun from 5M
        ichi_5m = eval_matrix.get('ichimoku_5M')
        if ichi_5m and 'kijun' in ichi_5m.metadata:
            result['kijun'] = ichi_5m.metadata['kijun']

        # Cloud thickness from 15M
        ichi_15m = eval_matrix.get('ichimoku_15M')
        if ichi_15m and 'cloud_thickness' in ichi_15m.metadata:
            result['cloud_thickness'] = ichi_15m.metadata['cloud_thickness']

        return result
