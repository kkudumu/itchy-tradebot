"""Vectorized Ichimoku Kinko Hyo calculator.

All calculations use pandas rolling operations for efficiency,
returning numpy arrays compatible with Vectorbt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class IchimokuResult:
    """Container for all five Ichimoku components.

    All arrays share the same length as the input price series.
    Displaced components (senkou_a, senkou_b, chikou_span) contain NaN
    at positions where displacement results in out-of-range references.
    """

    tenkan_sen: np.ndarray    # Conversion line — (high_9 + low_9) / 2
    kijun_sen: np.ndarray     # Base line — (high_26 + low_26) / 2
    senkou_a: np.ndarray      # Leading Span A — displaced kijun_period forward
    senkou_b: np.ndarray      # Leading Span B — displaced kijun_period forward
    chikou_span: np.ndarray   # Lagging Span — close displaced kijun_period backward


class IchimokuCalculator:
    """Vectorized Ichimoku Kinko Hyo calculator.

    Periods follow the standard Japanese settings (9/26/52) by default,
    but all values are configurable for alternative markets or timeframes.

    The displacement convention used here:
    - Senkou A and B are plotted kijun_period candles INTO THE FUTURE.
      In array terms this means shifting the series forward (shift > 0),
      so the computed cloud values appear at index i + kijun_period.
    - Chikou Span is plotted kijun_period candles INTO THE PAST.
      In array terms this means shifting close backward (shift < 0),
      so chikou[i] = close[i + kijun_period].

    Example
    -------
    >>> calc = IchimokuCalculator()
    >>> result = calc.calculate(high, low, close)
    """

    def __init__(
        self,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52,
    ) -> None:
        if tenkan_period < 1 or kijun_period < 1 or senkou_b_period < 1:
            raise ValueError("All periods must be positive integers.")
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _midpoint(
        prices: np.ndarray, period: int, use_high_low: bool = True,
        high: np.ndarray | None = None, low: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return (rolling_high + rolling_low) / 2 over *period* bars.

        Parameters
        ----------
        prices : np.ndarray
            Ignored when use_high_low=True; kept for API uniformity.
        period : int
            Look-back window.
        use_high_low : bool
            When True (default) uses separate high/low arrays.
        high, low : np.ndarray
            Required when use_high_low=True.
        """
        h = pd.Series(high)
        l = pd.Series(low)
        roll_high = h.rolling(window=period, min_periods=period).max()
        roll_low = l.rolling(window=period, min_periods=period).min()
        return ((roll_high + roll_low) / 2).to_numpy(dtype=np.float64)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> IchimokuResult:
        """Calculate all five Ichimoku components.

        Parameters
        ----------
        high, low, close : np.ndarray
            OHLC price arrays of equal length.

        Returns
        -------
        IchimokuResult
            Named arrays for each component, all of length ``len(close)``.
        """
        high = np.asarray(high, dtype=np.float64)
        low = np.asarray(low, dtype=np.float64)
        close = np.asarray(close, dtype=np.float64)

        if not (len(high) == len(low) == len(close)):
            raise ValueError("high, low, and close must have the same length.")

        # Tenkan-Sen: (highest high + lowest low) / 2 over tenkan_period
        tenkan_sen = self._midpoint(None, self.tenkan_period, high=high, low=low)

        # Kijun-Sen: same formula over kijun_period
        kijun_sen = self._midpoint(None, self.kijun_period, high=high, low=low)

        # Senkou Span A: average of Tenkan and Kijun, displaced forward
        senkou_a_raw = (tenkan_sen + kijun_sen) / 2
        senkou_a = pd.Series(senkou_a_raw).shift(self.kijun_period).to_numpy(dtype=np.float64)

        # Senkou Span B: midpoint of senkou_b_period range, displaced forward
        senkou_b_raw = self._midpoint(None, self.senkou_b_period, high=high, low=low)
        senkou_b = pd.Series(senkou_b_raw).shift(self.kijun_period).to_numpy(dtype=np.float64)

        # Chikou Span: current close placed kijun_period bars in the past
        # shift(-kijun_period) means chikou[i] = close[i + kijun_period]
        chikou_span = pd.Series(close).shift(-self.kijun_period).to_numpy(dtype=np.float64)

        return IchimokuResult(
            tenkan_sen=tenkan_sen,
            kijun_sen=kijun_sen,
            senkou_a=senkou_a,
            senkou_b=senkou_b,
            chikou_span=chikou_span,
        )

    def cloud_thickness(
        self,
        senkou_a: np.ndarray,
        senkou_b: np.ndarray,
    ) -> np.ndarray:
        """Return the absolute width of the cloud at each bar.

        Parameters
        ----------
        senkou_a, senkou_b : np.ndarray
            Leading Span arrays (may contain NaN).

        Returns
        -------
        np.ndarray
            ``|senkou_a - senkou_b|``; NaN propagated from inputs.
        """
        return np.abs(
            np.asarray(senkou_a, dtype=np.float64)
            - np.asarray(senkou_b, dtype=np.float64)
        )

    def cloud_direction(
        self,
        senkou_a: np.ndarray,
        senkou_b: np.ndarray,
    ) -> np.ndarray:
        """Return the polarity of the cloud at each bar.

        Returns
        -------
        np.ndarray of int8
            1  — bullish cloud (Senkou A > Senkou B)
            -1 — bearish cloud (Senkou A < Senkou B)
            0  — neutral / NaN inputs
        """
        a = np.asarray(senkou_a, dtype=np.float64)
        b = np.asarray(senkou_b, dtype=np.float64)
        result = np.zeros(len(a), dtype=np.int8)
        valid = ~(np.isnan(a) | np.isnan(b))
        result[valid & (a > b)] = 1
        result[valid & (a < b)] = -1
        return result
