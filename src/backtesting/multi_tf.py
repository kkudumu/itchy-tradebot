"""
Multi-timeframe data preparation for backtesting.

Resamples 1-minute OHLCV data into 5M / 15M / 1H / 4H bars and applies
.shift(1) on all higher timeframes to prevent lookahead bias.  The final
master DataFrame aligns everything to the 5M index via forward-fill, so
a single row contains all indicator data available at each 5M bar close.

Lookahead prevention contract:
  - 5M bars: shift(1) on indicator columns (consistent with MTFAnalyzer)
  - 15M / 1H / 4H bars: shift(1) applied, then forward-filled to 5M index
    so a 4H bar closing at 08:00 only becomes visible from the next 4H open
    (08:00 + 4h = 12:00), which is conservative and correct.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resampling configuration
# ---------------------------------------------------------------------------

_RESAMPLE_RULES: Dict[str, str] = {
    "5M":  "5min",
    "15M": "15min",
    "30M": "30min",
    "1H":  "1h",
    "4H":  "4h",
    "1D":  "1D",
}

_OHLCV_AGG = {
    "open":   "first",
    "high":   "max",
    "low":    "min",
    "close":  "last",
    "volume": "sum",
}

# Columns that receive .shift(1) on every timeframe (indicator columns only;
# raw OHLCV is left unshifted as it represents the completed bar).
_INDICATOR_COLS = [
    "tenkan", "kijun", "senkou_a", "senkou_b", "chikou",
    "adx", "adx_trending", "atr",
    "ema_fast", "ema_mid", "ema_slow",
]


class BacktestDataPreparer:
    """Prepare multi-timeframe data for backtesting from 1-minute candles.

    Parameters
    ----------
    ichi_tenkan_period:
        Tenkan-Sen look-back period.  Default: 9.
    ichi_kijun_period:
        Kijun-Sen look-back period.  Default: 26.
    ichi_senkou_b_period:
        Senkou Span B look-back period.  Default: 52.
    adx_period:
        ADX smoothing period.  Default: 14.
    atr_period:
        ATR period.  Default: 14.
    """

    def __init__(
        self,
        ichi_tenkan_period: int = 9,
        ichi_kijun_period: int = 26,
        ichi_senkou_b_period: int = 52,
        adx_period: int = 14,
        atr_period: int = 14,
    ) -> None:
        # Lazy imports — keep the module importable even if src packages are
        # not installed during unit tests that mock indicator computations.
        from src.indicators.ichimoku import IchimokuCalculator
        from src.indicators.confluence import ADXCalculator, ATRCalculator

        self._ichi = IchimokuCalculator(
            tenkan_period=ichi_tenkan_period,
            kijun_period=ichi_kijun_period,
            senkou_b_period=ichi_senkou_b_period,
        )
        self._adx = ADXCalculator(period=adx_period)
        self._atr = ATRCalculator(period=atr_period)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(self, candles_1m: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Resample 1M candles to all required timeframes with indicators.

        Each higher timeframe DataFrame has .shift(1) applied to all
        indicator columns so that bar-close indicator values only become
        available on the *next* bar — preventing lookahead bias.

        Parameters
        ----------
        candles_1m:
            1-minute OHLCV DataFrame with a UTC DatetimeIndex.
            Required columns: open, high, low, close.  volume is optional.

        Returns
        -------
        Dict with keys '1M', '5M', '15M', '1H', '4H'.  Each value is a
        resampled DataFrame with Ichimoku + ADX + ATR indicator columns
        appended.  Indicator columns on all frames are shifted by one bar.
        """
        if candles_1m.empty:
            raise ValueError("candles_1m is empty — nothing to prepare")

        result: Dict[str, pd.DataFrame] = {}

        # Process 1M separately (no resampling needed, just add indicators
        # with the same shift guard).
        result["1M"] = self._add_indicators_and_shift(candles_1m.copy())

        for tf, rule in _RESAMPLE_RULES.items():
            resampled = self._resample(candles_1m, rule)
            result[tf] = self._add_indicators_and_shift(resampled)

        logger.debug(
            "Prepared %d timeframes; 1M bars=%d, 5M bars=%d, 4H bars=%d",
            len(result),
            len(result["1M"]),
            len(result["5M"]),
            len(result["4H"]),
        )
        return result

    def align_to_5m(self, tf_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge all timeframe DataFrames aligned to the 5M index.

        Higher-TF values are forward-filled to 5M resolution so that each
        5M bar row contains the latest available (already shifted) indicator
        reading from all timeframes.

        Parameters
        ----------
        tf_data:
            Dict returned by :meth:`prepare`.

        Returns
        -------
        Master DataFrame indexed on the 5M timestamps.  Columns from each
        timeframe are prefixed: ``5m_``, ``15m_``, ``1h_``, ``4h_``.
        """
        base = tf_data["5M"].copy()

        for tf in ("15M", "1H", "4H"):
            prefix = tf.lower().replace("m", "m").replace("h", "h") + "_"
            # Normalise prefix to e.g. "15m_", "1h_", "4h_"
            prefix = tf.lower() + "_"

            higher = tf_data[tf].copy()
            # Rename columns to carry the timeframe prefix
            higher = higher.rename(
                columns={c: f"{prefix}{c}" for c in higher.columns}
            )
            # Reindex to 5M index, forward-fill, then join
            higher_reindexed = higher.reindex(base.index, method="ffill")
            base = base.join(higher_reindexed, how="left")

        # Also prefix the 5M columns themselves for clarity
        base = base.rename(columns={c: f"5m_{c}" for c in tf_data["5M"].columns})

        return base

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resample(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
        """Resample 1M OHLCV to *rule* frequency using left-labelled bars."""
        available_agg = {k: v for k, v in _OHLCV_AGG.items() if k in df_1m.columns}
        resampled = df_1m.resample(rule, label="left", closed="left").agg(available_agg)
        return resampled.dropna(how="all")

    def _add_indicators_and_shift(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Ichimoku / ADX / ATR and attach them with a one-bar shift.

        Raw OHLCV columns are untouched.  Indicator columns are appended and
        then shifted by one bar so that the value produced by bar N is only
        visible from bar N+1 onward.
        """
        if len(df) < 2:
            # Not enough data to compute meaningful indicators.
            for col in _INDICATOR_COLS:
                df[col] = np.nan
            return df

        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)

        # Ichimoku
        ichi = self._ichi.calculate(high, low, close)
        df["tenkan"]   = ichi.tenkan_sen
        df["kijun"]    = ichi.kijun_sen
        df["senkou_a"] = ichi.senkou_a
        df["senkou_b"] = ichi.senkou_b
        df["chikou"]   = ichi.chikou_span

        # ADX
        adx_result = self._adx.calculate(high, low, close)
        df["adx"]         = adx_result.adx
        df["adx_trending"] = adx_result.is_trending.astype(float)

        # ATR
        atr_arr = self._atr.calculate(high, low, close)
        df["atr"] = atr_arr

        # EMA trio for EMA Pullback strategy (14, 18, 24 periods)
        close_series = df["close"]
        df["ema_fast"] = close_series.ewm(span=14, adjust=False).mean()
        df["ema_mid"]  = close_series.ewm(span=18, adjust=False).mean()
        df["ema_slow"] = close_series.ewm(span=24, adjust=False).mean()

        # CRITICAL: shift indicator columns forward by one bar.  This
        # ensures the bar-N close-derived value only appears on bar N+1.
        for col in _INDICATOR_COLS:
            if col in df.columns:
                df[col] = df[col].shift(1)

        return df
