"""
Core signal generation engine — 4H→1H→15M→5M hierarchy.

Signal flow
~~~~~~~~~~~
1.  Align 1-minute data into 5M / 15M / 1H / 4H timeframes with lookahead
    prevention (all higher-TF indicator columns shifted +1 bar).
2.  Apply 4H hard filter: never trade against the 4H cloud direction.
3.  Confirm with 1H TK alignment.
4.  Look for 15M signal: TK cross in direction + price above/below cloud
    + Chikou confirmation.
5.  Time entry on 5M pullback to Kijun.
6.  Score confluence (0–8 scale).
7.  Check minimum score threshold (default 4).
8.  Compute entry, stop-loss, and take-profit levels.
9.  Return a Signal with a full reasoning trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Union

import numpy as np
import pandas as pd

from .mtf_analyzer import MTFAnalyzer, MTFState
from .confluence_scorer import ConfluenceScorer, ConfluenceResult
from ..indicators.signals import IchimokuSignalState
from ..zones.manager import ZoneManager


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: dict = {
    # Ichimoku periods
    "tenkan_period": 9,
    "kijun_period": 26,
    "senkou_b_period": 52,
    # ADX
    "adx_threshold": 28,
    "adx_period": 14,
    # ATR
    "atr_period": 14,
    "atr_stop_multiplier": 1.5,
    # Signal quality
    "min_confluence_score": 4,
    "rr_ratio": 2.0,
    # 5M Kijun proximity (in ATR)
    "kijun_proximity_atr": 0.5,
}


# ---------------------------------------------------------------------------
# Signal dataclass
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    """A fully-qualified trade signal with entry levels and reasoning."""

    timestamp: datetime
    instrument: str
    direction: str          # 'long' or 'short'
    entry_price: float
    stop_loss: float
    take_profit: float
    confluence_score: int
    quality_tier: str       # 'A+', 'B', 'C'
    atr: float
    reasoning: dict = field(default_factory=dict)
    mtf_state: Optional[MTFState] = None
    zone_context: dict = field(default_factory=dict)


@dataclass
class ScanResult:
    """Result of a signal scan with per-filter diagnostics."""

    signal: Optional[Signal]
    filters: dict
    passed_all: bool


# ---------------------------------------------------------------------------
# SignalEngine
# ---------------------------------------------------------------------------

class SignalEngine:
    """XAU/USD multi-timeframe Ichimoku signal engine.

    Parameters
    ----------
    config:
        Optional override dict.  Missing keys fall back to ``_DEFAULT_CONFIG``.
    instrument:
        Instrument label included in emitted signals.  Default: 'XAUUSD'.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        instrument: str = "XAUUSD",
    ) -> None:
        self._cfg: dict = {**_DEFAULT_CONFIG, **(config or {})}
        self.instrument = instrument

        from ..indicators.ichimoku import IchimokuCalculator
        from ..indicators.confluence import ADXCalculator, ATRCalculator

        ichi = IchimokuCalculator(
            tenkan_period=self._cfg["tenkan_period"],
            kijun_period=self._cfg["kijun_period"],
            senkou_b_period=self._cfg["senkou_b_period"],
        )
        adx = ADXCalculator(
            period=self._cfg["adx_period"],
            threshold=self._cfg["adx_threshold"],
        )
        atr = ATRCalculator(period=self._cfg["atr_period"])

        self.mtf_analyzer = MTFAnalyzer(
            ichimoku_calc=ichi,
            adx_calc=adx,
            atr_calc=atr,
        )
        self.confluence_scorer = ConfluenceScorer(
            adx_threshold=self._cfg["adx_threshold"],
            min_score=self._cfg["min_confluence_score"],
            kijun_proximity_atr=self._cfg["kijun_proximity_atr"],
        )
        self.zone_manager = ZoneManager()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(
        self,
        data_1m: pd.DataFrame,
        current_bar: int = -1,
        return_scan_result: bool = False,
    ) -> Union[Optional[Signal], ScanResult]:
        """Scan for a trade signal at the specified 1M bar.

        Parameters
        ----------
        data_1m:
            1-minute OHLCV DataFrame with a UTC DatetimeIndex.
        current_bar:
            Bar index to evaluate.  -1 = latest available bar.
        return_scan_result:
            When True, return a ScanResult with per-filter diagnostics.
            When False (default), return Optional[Signal] for backward
            compatibility.

        Returns
        -------
        Signal | None when return_scan_result is False.
        ScanResult when return_scan_result is True.
        """
        reasoning: dict = {}
        filters: dict = {}

        # Step 1: align timeframes (applies .shift(1) on indicator cols)
        tf_data = self.mtf_analyzer.align_timeframes(data_1m)
        mtf_state = self.mtf_analyzer.get_current_state(tf_data, current_bar)

        reasoning["timestamp"] = str(mtf_state.timestamp)
        reasoning["session"] = mtf_state.session

        # Step 2: 4H cloud direction (hard filter)
        passes_4h, direction, reason_4h = self._check_4h_filter(mtf_state.state_4h)
        reasoning["4h_filter"] = {"pass": passes_4h, "direction": direction, "reason": reason_4h}
        filters["4h_cloud"] = {"pass": passes_4h, "reason": reason_4h}

        if not passes_4h:
            if return_scan_result:
                return ScanResult(signal=None, filters=filters, passed_all=False)
            return None

        # Step 3: 1H TK alignment
        passes_1h, reason_1h = self._check_1h_confirmation(mtf_state.state_1h, direction)
        reasoning["1h_confirmation"] = {"pass": passes_1h, "reason": reason_1h}
        filters["1h_confirmation"] = {"pass": passes_1h, "reason": reason_1h}

        if not passes_1h:
            if return_scan_result:
                return ScanResult(signal=None, filters=filters, passed_all=False)
            return None

        # Step 4: 15M signal
        passes_15m, reason_15m = self._check_15m_signal(mtf_state.state_15m, direction)
        reasoning["15m_signal"] = {"pass": passes_15m, "reason": reason_15m}
        filters["15m_signal"] = {"pass": passes_15m, "reason": reason_15m}

        if not passes_15m:
            if return_scan_result:
                return ScanResult(signal=None, filters=filters, passed_all=False)
            return None

        # Step 5: 5M entry timing
        passes_5m, reason_5m = self._check_5m_entry(
            state_5m=mtf_state.state_5m,
            close_5m=mtf_state.close_5m,
            kijun_5m=mtf_state.kijun_5m,
            atr=mtf_state.atr_15m,
        )
        reasoning["5m_entry"] = {"pass": passes_5m, "reason": reason_5m}
        filters["5m_entry"] = {"pass": passes_5m, "reason": reason_5m}

        if not passes_5m:
            if return_scan_result:
                return ScanResult(signal=None, filters=filters, passed_all=False)
            return None

        # Step 6: zone context
        atr = mtf_state.atr_15m if mtf_state.atr_15m > 0 else 1.0
        nearby_zones = self.zone_manager.get_nearby_zones(
            price=mtf_state.close_5m,
            atr=atr,
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
                for z in nearby_zones[:5]  # cap detail list at 5
            ],
        }

        # Step 7: confluence scoring
        confluence: ConfluenceResult = self.confluence_scorer.score(
            mtf_state=mtf_state,
            direction=direction,
            zone_confluence=zone_count,
        )
        reasoning["confluence"] = confluence.breakdown
        filters["confluence"] = {"pass": confluence.tier != "no_trade", "reason": f"score={confluence.total_score}, tier={confluence.tier}"}

        if confluence.tier == "no_trade":
            if return_scan_result:
                return ScanResult(signal=None, filters=filters, passed_all=False)
            return None

        # Step 8: trade levels
        entry_price = mtf_state.close_5m
        levels = self._calculate_levels(
            entry_price=entry_price,
            direction=direction,
            atr=atr,
            atr_multiplier=self._cfg["atr_stop_multiplier"],
            rr_ratio=self._cfg["rr_ratio"],
        )
        reasoning["levels"] = levels

        signal = Signal(
            timestamp=mtf_state.timestamp,
            instrument=self.instrument,
            direction=direction,
            entry_price=entry_price,
            stop_loss=levels["stop_loss"],
            take_profit=levels["take_profit"],
            confluence_score=confluence.total_score,
            quality_tier=confluence.tier,
            atr=atr,
            reasoning=reasoning,
            mtf_state=mtf_state,
            zone_context=zone_context,
        )

        if return_scan_result:
            return ScanResult(signal=signal, filters=filters, passed_all=True)
        return signal

    # ------------------------------------------------------------------
    # Filter checks — each returns (pass, reason) or (pass, direction, reason)
    # ------------------------------------------------------------------

    def _check_4h_filter(
        self,
        state_4h: IchimokuSignalState,
    ) -> tuple[bool, str, str]:
        """Check the 4H cloud direction to determine trade bias.

        Returns
        -------
        (passes_filter, direction, reason)
            direction is 'long', 'short', or '' when filter fails.
        """
        cd = state_4h.cloud_direction

        if cd == 1:
            return True, "long", "4H cloud is bullish (senkou_a > senkou_b)"
        if cd == -1:
            return True, "short", "4H cloud is bearish (senkou_a < senkou_b)"
        return False, "", "4H cloud is flat/neutral — no directional bias"

    def _check_1h_confirmation(
        self,
        state_1h: IchimokuSignalState,
        direction: str,
    ) -> tuple[bool, str]:
        """Confirm that 1H TK alignment matches the trade direction.

        For a long signal the 1H Tenkan must be above the Kijun (tk_cross == 1).
        For a short signal the 1H Tenkan must be below the Kijun (tk_cross == -1).
        """
        sign = 1 if direction == "long" else -1

        if state_1h.tk_cross == sign:
            label = "above" if sign == 1 else "below"
            return True, f"1H Tenkan is {label} Kijun — confirms {direction} bias"

        if state_1h.tk_cross == 0:
            return False, "1H Tenkan equals Kijun — no directional confirmation"

        opposite = "above" if sign == -1 else "below"
        return False, f"1H Tenkan is {opposite} Kijun — contradicts {direction} bias"

    def _check_15m_signal(
        self,
        state_15m: IchimokuSignalState,
        direction: str,
    ) -> tuple[bool, str]:
        """Check for a valid 15M signal: TK cross + cloud position + Chikou.

        All three sub-conditions are evaluated; the first failure short-circuits
        the check and its reason is returned to the reasoning trace.
        """
        sign = 1 if direction == "long" else -1
        reasons: list[str] = []
        all_pass = True

        # TK cross
        if state_15m.tk_cross == sign:
            reasons.append("15M TK cross confirmed")
        else:
            all_pass = False
            cross_label = "bullish" if sign == 1 else "bearish"
            reasons.append(f"15M TK cross not {cross_label} (value={state_15m.tk_cross})")

        # Cloud position
        if state_15m.cloud_position == sign:
            pos_label = "above" if sign == 1 else "below"
            reasons.append(f"15M price {pos_label} cloud")
        else:
            all_pass = False
            expected = "above" if sign == 1 else "below"
            reasons.append(
                f"15M price not {expected} cloud (position={state_15m.cloud_position})"
            )

        # Chikou confirmation (bonus, not required — chikou span uses
        # forward-displaced data that is unreliable in shifted backtest context)
        if state_15m.chikou_confirmed == sign:
            reasons.append("15M Chikou confirms direction")
        elif state_15m.chikou_confirmed == 0:
            reasons.append("15M Chikou neutral (ignored)")
        else:
            reasons.append(
                f"15M Chikou not confirming {direction} (value={state_15m.chikou_confirmed})"
            )

        return all_pass, "; ".join(reasons)

    def _check_5m_entry(
        self,
        state_5m: IchimokuSignalState,
        close_5m: float,
        kijun_5m: float,
        atr: float,
    ) -> tuple[bool, str]:
        """Check that 5M price has pulled back to within 0.5 ATR of the Kijun.

        This filters out chasing entries and ensures the risk/reward remains
        favourable by entering on the pullback rather than the breakout candle.
        """
        if _is_nan(close_5m) or _is_nan(kijun_5m) or _is_nan(atr) or atr <= 0:
            return False, "Insufficient 5M data for entry check"

        proximity = self._cfg["kijun_proximity_atr"] * atr
        dist = abs(close_5m - kijun_5m)

        if dist <= proximity:
            return (
                True,
                f"5M close ({close_5m:.4f}) within {self._cfg['kijun_proximity_atr']} ATR "
                f"of Kijun ({kijun_5m:.4f}); distance={dist:.4f}",
            )

        return (
            False,
            f"5M close ({close_5m:.4f}) too far from Kijun ({kijun_5m:.4f}); "
            f"distance={dist:.4f} > {proximity:.4f} ({self._cfg['kijun_proximity_atr']} ATR)",
        )

    # ------------------------------------------------------------------
    # Level calculation
    # ------------------------------------------------------------------

    def _calculate_levels(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        atr_multiplier: float = 1.5,
        rr_ratio: float = 2.0,
    ) -> dict:
        """Compute stop-loss and take-profit from ATR-based risk sizing.

        Stop-loss distance: atr × atr_multiplier
        Take-profit distance: stop_distance × rr_ratio

        Parameters
        ----------
        entry_price:
            Proposed entry price (typically the current 5M close).
        direction:
            'long' or 'short'.
        atr:
            Current ATR value used for sizing.
        atr_multiplier:
            ATR multiplier for the stop-loss distance.  Default: 1.5.
        rr_ratio:
            Risk:Reward ratio for target placement.  Default: 2.0.

        Returns
        -------
        dict with keys 'stop_loss', 'take_profit', 'risk_pips', 'reward_pips'.
        """
        sl_distance = atr * atr_multiplier
        tp_distance = sl_distance * rr_ratio

        if direction == "long":
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance

        return {
            "stop_loss": round(stop_loss, 5),
            "take_profit": round(take_profit, 5),
            "risk_pips": round(sl_distance, 5),
            "reward_pips": round(tp_distance, 5),
            "rr_ratio": rr_ratio,
            "atr_multiplier": atr_multiplier,
        }


# ---------------------------------------------------------------------------
# Module-level nan helper
# ---------------------------------------------------------------------------

def _is_nan(v: float) -> bool:
    import math
    try:
        return math.isnan(v)
    except (TypeError, ValueError):
        return True
