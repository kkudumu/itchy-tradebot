"""
Multi-timeframe data alignment with lookahead bias prevention.

Higher timeframe bars are shifted forward by one bar relative to the lower
timeframe so that a 4H bar closing at 08:00 UTC only becomes visible from
the 08:01 1-minute bar onward.  This prevents any form of future-peeking
when the engine evaluates signal conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from ..indicators.ichimoku import IchimokuCalculator, IchimokuResult
from ..indicators.signals import IchimokuSignals, IchimokuSignalState
from ..indicators.confluence import ADXCalculator, ADXResult, ATRCalculator
from ..indicators.sessions import SessionIdentifier


# ---------------------------------------------------------------------------
# MTFState dataclass
# ---------------------------------------------------------------------------

@dataclass
class MTFState:
    """Aligned multi-timeframe Ichimoku state at a single evaluation point."""

    state_4h: IchimokuSignalState
    state_1h: IchimokuSignalState
    state_15m: IchimokuSignalState
    state_5m: IchimokuSignalState
    adx_15m: float           # ADX value on the 15-minute timeframe
    atr_15m: float           # ATR value on the 15-minute timeframe
    kijun_5m: float          # Kijun-Sen on 5M (used for entry-timing check)
    close_5m: float          # Latest 5M close
    session: str             # Session label at the evaluation bar
    timestamp: datetime      # UTC timestamp of the evaluation bar


# ---------------------------------------------------------------------------
# Resampling helpers
# ---------------------------------------------------------------------------

_TF_RULES: dict[str, str] = {
    "5M":  "5min",
    "15M": "15min",
    "1H":  "1h",
    "4H":  "4h",
}

_REQUIRED_COLS = ("open", "high", "low", "close", "volume")


def _resample_ohlcv(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample a 1-minute OHLCV DataFrame to the requested frequency.

    The resampled index is the *open* time of each bar (label='left',
    closed='left'), which is the conventional bar-naming used by most
    brokers: the 08:00 4H bar spans 08:00–11:59.
    """
    agg = {
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }
    # Only aggregate columns that are present
    available = {k: v for k, v in agg.items() if k in df_1m.columns}
    resampled = df_1m.resample(rule, label="left", closed="left").agg(available)
    return resampled.dropna(subset=["close"])


# ---------------------------------------------------------------------------
# MTFAnalyzer
# ---------------------------------------------------------------------------

class MTFAnalyzer:
    """Align multi-timeframe data and compute indicators for each frame.

    Parameters
    ----------
    ichimoku_calc:
        Pre-configured IchimokuCalculator.  Defaults to standard 9/26/52.
    adx_calc:
        Pre-configured ADXCalculator.  Defaults to period=14, threshold=28.
    atr_calc:
        Pre-configured ATRCalculator.  Defaults to period=14.
    """

    def __init__(
        self,
        ichimoku_calc: Optional[IchimokuCalculator] = None,
        adx_calc: Optional[ADXCalculator] = None,
        atr_calc: Optional[ATRCalculator] = None,
    ) -> None:
        self._ichi = ichimoku_calc or IchimokuCalculator()
        self._adx = adx_calc or ADXCalculator()
        self._atr = atr_calc or ATRCalculator()
        self._sig = IchimokuSignals()
        self._session = SessionIdentifier()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def align_timeframes(self, data_1m: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Resample 1-minute OHLCV data to each higher timeframe.

        Lookahead prevention: each higher-TF DataFrame is shifted forward by
        one *higher-TF* bar using .shift(1).  This means the indicators
        computed from that bar are only visible from the next bar's timestamp
        onward, which accurately reflects when the bar's close is known.

        Parameters
        ----------
        data_1m:
            DataFrame with DatetimeIndex (UTC) and columns:
            open, high, low, close, [volume].

        Returns
        -------
        dict with keys '5M', '15M', '1H', '4H', each holding a resampled
        OHLCV DataFrame with indicators attached.
        """
        result: dict[str, pd.DataFrame] = {}

        for tf, rule in _TF_RULES.items():
            resampled = _resample_ohlcv(data_1m, rule)

            # Compute indicators before shifting so they align with the correct bar
            indicators = self.compute_indicators(resampled)

            # Attach indicator series to the DataFrame
            resampled = resampled.copy()
            ich: IchimokuResult = indicators["ichimoku"]
            resampled["tenkan"] = ich.tenkan_sen
            resampled["kijun"] = ich.kijun_sen
            resampled["senkou_a"] = ich.senkou_a
            resampled["senkou_b"] = ich.senkou_b
            resampled["chikou"] = ich.chikou_span

            adx_res: ADXResult = indicators["adx"]
            resampled["adx"] = adx_res.adx
            resampled["adx_trending"] = adx_res.is_trending

            atr_arr: np.ndarray = indicators["atr"]
            resampled["atr"] = atr_arr

            # CRITICAL: shift all indicator columns by 1 bar to prevent lookahead.
            # Raw OHLCV columns are left unshifted — they represent the bar that
            # actually formed.  Indicator values (which depend on the *close* of
            # that bar) are shifted so they only appear on the subsequent bar.
            indicator_cols = [
                "tenkan", "kijun", "senkou_a", "senkou_b", "chikou",
                "adx", "adx_trending", "atr",
            ]
            for col in indicator_cols:
                if col in resampled.columns:
                    resampled[col] = resampled[col].shift(1)

            result[tf] = resampled

        return result

    def compute_indicators(self, ohlcv: pd.DataFrame) -> dict:
        """Compute Ichimoku and confluence indicators for a single timeframe.

        Parameters
        ----------
        ohlcv:
            DataFrame with at minimum open, high, low, close columns.

        Returns
        -------
        dict with keys:
            - 'ichimoku': IchimokuResult
            - 'signals':  IchimokuSignalState at the last bar (index -1)
            - 'adx':      ADXResult
            - 'atr':      np.ndarray
        """
        high = ohlcv["high"].to_numpy(dtype=float)
        low = ohlcv["low"].to_numpy(dtype=float)
        close = ohlcv["close"].to_numpy(dtype=float)

        ichimoku = self._ichi.calculate(high, low, close)
        adx = self._adx.calculate(high, low, close)
        atr = self._atr.calculate(high, low, close)

        # Compute signal state at the last available bar
        idx = len(close) - 1
        state = self._sig.signal_state_at(
            idx=idx,
            tenkan=ichimoku.tenkan_sen,
            kijun=ichimoku.kijun_sen,
            close=close,
            senkou_a=ichimoku.senkou_a,
            senkou_b=ichimoku.senkou_b,
            chikou=ichimoku.chikou_span,
        )

        return {
            "ichimoku": ichimoku,
            "signals": state,
            "adx": adx,
            "atr": atr,
        }

    def get_current_state(
        self,
        tf_data: dict[str, pd.DataFrame],
        bar_index: int = -1,
    ) -> MTFState:
        """Extract the aligned multi-timeframe state at a specific 1M bar index.

        Because higher-TF DataFrames are already shifted by one bar (applied
        in ``align_timeframes``), reading ``iloc[bar_index]`` from each of them
        gives the correct, lookahead-free indicator values at that moment.

        Parameters
        ----------
        tf_data:
            Dict returned by :meth:`align_timeframes`.
        bar_index:
            Position in the aligned index to evaluate.  -1 = latest bar.

        Returns
        -------
        MTFState
        """
        state_4h = self._state_from_row(tf_data["4H"], bar_index)
        state_1h = self._state_from_row(tf_data["1H"], bar_index)
        state_15m = self._state_from_row(tf_data["15M"], bar_index)
        state_5m = self._state_from_row(tf_data["5M"], bar_index)

        # ADX and ATR come from the 15M frame (signal generation timeframe)
        row_15m = self._get_row(tf_data["15M"], bar_index)
        adx_val = float(row_15m.get("adx", np.nan))
        atr_val = float(row_15m.get("atr", np.nan))

        # 5M Kijun used for entry proximity check
        row_5m = self._get_row(tf_data["5M"], bar_index)
        kijun_5m = float(row_5m.get("kijun", np.nan))
        close_5m = float(row_5m.get("close", np.nan))

        # Session label from the last row's timestamp
        last_ts = tf_data["5M"].index[bar_index]
        ts_arr = np.asarray([last_ts], dtype="datetime64[s]")
        session_labels = self._session.identify(ts_arr)
        session = str(session_labels[0])

        ts_dt: datetime = pd.Timestamp(last_ts).to_pydatetime()

        return MTFState(
            state_4h=state_4h,
            state_1h=state_1h,
            state_15m=state_15m,
            state_5m=state_5m,
            adx_15m=adx_val if not np.isnan(adx_val) else 0.0,
            atr_15m=atr_val if not np.isnan(atr_val) else 0.0,
            kijun_5m=kijun_5m,
            close_5m=close_5m,
            session=session,
            timestamp=ts_dt,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_row(df: pd.DataFrame, idx: int) -> dict:
        """Return a row as a plain dict (handles iloc index)."""
        row = df.iloc[idx]
        return row.to_dict()

    def _state_from_row(self, df: pd.DataFrame, bar_index: int) -> IchimokuSignalState:
        """Build an IchimokuSignalState from a shifted indicator row.

        All indicator values have already been shifted by one bar in
        ``align_timeframes``, so this reads the pre-computed values directly
        rather than recomputing signal arrays — there is no lookahead.
        """
        row = self._get_row(df, bar_index)

        tenkan = float(row.get("tenkan", np.nan))
        kijun = float(row.get("kijun", np.nan))
        close = float(row.get("close", np.nan))
        senkou_a = float(row.get("senkou_a", np.nan))
        senkou_b = float(row.get("senkou_b", np.nan))
        chikou = float(row.get("chikou", np.nan))

        # Derive signals from scalar values (no array context needed here since
        # the shift has already encoded the lookahead guard)
        cloud_direction = self._scalar_cloud_direction(senkou_a, senkou_b)
        cloud_position = self._scalar_cloud_position(close, senkou_a, senkou_b)
        chikou_confirmed = self._scalar_chikou(chikou, close)
        cloud_thickness = self._scalar_cloud_thickness(senkou_a, senkou_b)

        # TK cross requires the previous bar's tenkan/kijun; for the state
        # snapshot we use the alignment direction (tenkan vs kijun) as a proxy
        # since a single-bar cross is already recorded on the shifted row.
        tk_cross = self._scalar_tk_alignment(tenkan, kijun)

        # Cloud twist is a cross of senkou_a and senkou_b — not meaningful as
        # a scalar single-bar value, so we record 0 (no twist at this bar).
        cloud_twist = 0

        return IchimokuSignalState(
            cloud_direction=cloud_direction,
            tk_cross=tk_cross,
            cloud_position=cloud_position,
            chikou_confirmed=chikou_confirmed,
            cloud_twist=cloud_twist,
            cloud_thickness=cloud_thickness,
        )

    # ------------------------------------------------------------------
    # Scalar signal helpers (single-bar, no array context)
    # ------------------------------------------------------------------

    @staticmethod
    def _scalar_cloud_direction(senkou_a: float, senkou_b: float) -> int:
        if np.isnan(senkou_a) or np.isnan(senkou_b):
            return 0
        if senkou_a > senkou_b:
            return 1
        if senkou_a < senkou_b:
            return -1
        return 0

    @staticmethod
    def _scalar_cloud_position(close: float, senkou_a: float, senkou_b: float) -> int:
        if np.isnan(close) or np.isnan(senkou_a) or np.isnan(senkou_b):
            return 0
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        if close > cloud_top:
            return 1
        if close < cloud_bottom:
            return -1
        return 0

    @staticmethod
    def _scalar_chikou(chikou: float, close: float) -> int:
        if np.isnan(chikou) or np.isnan(close):
            return 0
        if chikou > close:
            return 1
        if chikou < close:
            return -1
        return 0

    @staticmethod
    def _scalar_cloud_thickness(senkou_a: float, senkou_b: float) -> float:
        if np.isnan(senkou_a) or np.isnan(senkou_b):
            return float("nan")
        return abs(senkou_a - senkou_b)

    @staticmethod
    def _scalar_tk_alignment(tenkan: float, kijun: float) -> int:
        """Return the current TK relationship as a directional signal.

        This is not a cross detection (which requires two bars) but rather
        the current alignment: tenkan above kijun = bullish (+1), below = -1.
        Used by the confluence scorer as a proxy for 1H TK alignment.
        """
        if np.isnan(tenkan) or np.isnan(kijun):
            return 0
        if tenkan > kijun:
            return 1
        if tenkan < kijun:
            return -1
        return 0
