"""
Confluence indicators for XAU/USD Ichimoku trading agent.

Includes ADX, ATR, RSI, and Bollinger Band width with squeeze detection.
All calculations are vectorized using numpy. Wilder's smoothing uses a
single-pass loop, which is the standard approach for this algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Shared smoothing helper
# ---------------------------------------------------------------------------

def wilders_smooth(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothing (Wilder's Moving Average / RMA).

    First valid value is the simple average of the first ``period`` values.
    Subsequent values follow: result[i] = (result[i-1] * (period - 1) + values[i]) / period

    Parameters
    ----------
    values:
        1-D float array. NaN values at the start are skipped when computing
        the initial SMA seed, but NaNs beyond the warm-up period will
        propagate through the recursion.
    period:
        Smoothing period (number of bars).

    Returns
    -------
    np.ndarray of the same length as ``values``, with NaN for the first
    ``period - 1`` positions.
    """
    n = len(values)
    result = np.full(n, np.nan, dtype=float)

    if n < period:
        return result

    # Seed: simple mean of the first `period` values
    result[period - 1] = np.mean(values[:period])

    # Recursive Wilder step — O(n) loop is unavoidable for this recurrence
    for i in range(period, n):
        result[i] = (result[i - 1] * (period - 1) + values[i]) / period

    return result


# ---------------------------------------------------------------------------
# ADX
# ---------------------------------------------------------------------------

@dataclass
class ADXResult:
    """Output of :class:`ADXCalculator`."""

    adx: np.ndarray
    """Average Directional Index values."""

    plus_di: np.ndarray
    """+DI (positive directional indicator)."""

    minus_di: np.ndarray
    """-DI (negative directional indicator)."""

    is_trending: np.ndarray
    """Boolean mask: ``adx > threshold``."""


class ADXCalculator:
    """Average Directional Index — measures trend strength (not direction).

    Gold-specific default threshold: 28 (higher than the generic 25, accounting
    for gold's tendency to exhibit stronger trending behaviour during sessions).

    Parameters
    ----------
    period:
        Lookback period for Wilder smoothing. Default: 14.
    threshold:
        ADX level above which the market is considered trending. Default: 28.
    """

    def __init__(self, period: int = 14, threshold: float = 28.0) -> None:
        self.period = period
        self.threshold = threshold

    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> ADXResult:
        """Calculate ADX, +DI, and -DI.

        Algorithm (Wilder, 1978):

        1. True Range  = max(H − L, |H − prevC|, |L − prevC|)
        2. +DM         = H − prevH  if (H − prevH) > (prevL − L) and > 0, else 0
        3. −DM         = prevL − L  if (prevL − L) > (H − prevH) and > 0, else 0
        4. Smooth TR, +DM, −DM with Wilder's method
        5. +DI         = 100 × smoothed(+DM) / smoothed(TR)
        6. −DI         = 100 × smoothed(−DM) / smoothed(TR)
        7. DX          = 100 × |+DI − −DI| / (+DI + −DI)
        8. ADX         = Wilder smooth of DX

        Parameters
        ----------
        high, low, close:
            Equal-length 1-D float arrays of OHLC data.

        Returns
        -------
        ADXResult
        """
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)

        n = len(close)

        # True Range (vectorized; index 0 has no previous close so set to H-L)
        prev_close = np.empty(n, dtype=float)
        prev_close[0] = close[0]
        prev_close[1:] = close[:-1]

        tr = np.maximum(
            high - low,
            np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
        )
        # Index 0 is undefined (no prior bar) — zero it so the SMA seed is clean
        tr[0] = high[0] - low[0]

        # Directional movements
        prev_high = np.empty(n, dtype=float)
        prev_high[0] = high[0]
        prev_high[1:] = high[:-1]

        prev_low = np.empty(n, dtype=float)
        prev_low[0] = low[0]
        prev_low[1:] = low[:-1]

        up_move = high - prev_high      # H - prevH
        down_move = prev_low - low      # prevL - L

        # +DM: up_move positive and larger than down_move
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        # -DM: down_move positive and larger than up_move
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        # First bar has no meaningful DM
        plus_dm[0] = 0.0
        minus_dm[0] = 0.0

        # Wilder smoothing
        smooth_tr = wilders_smooth(tr, self.period)
        smooth_plus_dm = wilders_smooth(plus_dm, self.period)
        smooth_minus_dm = wilders_smooth(minus_dm, self.period)

        # Directional indicators (avoid divide-by-zero)
        with np.errstate(invalid="ignore", divide="ignore"):
            plus_di = np.where(smooth_tr > 0, 100.0 * smooth_plus_dm / smooth_tr, 0.0)
            minus_di = np.where(smooth_tr > 0, 100.0 * smooth_minus_dm / smooth_tr, 0.0)

            di_sum = plus_di + minus_di
            dx = np.where(di_sum > 0, 100.0 * np.abs(plus_di - minus_di) / di_sum, 0.0)

        # ADX is Wilder smooth of DX; NaN slots before period-1 propagate correctly
        adx = wilders_smooth(dx, self.period)

        # Propagate NaN from smoothed TR into DI / ADX for consistency
        nan_mask = np.isnan(smooth_tr)
        plus_di = np.where(nan_mask, np.nan, plus_di)
        minus_di = np.where(nan_mask, np.nan, minus_di)

        is_trending = np.where(np.isnan(adx), False, adx > self.threshold).astype(bool)

        return ADXResult(
            adx=adx,
            plus_di=plus_di,
            minus_di=minus_di,
            is_trending=is_trending,
        )


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

class ATRCalculator:
    """Average True Range — normalised volatility measure for stop/target sizing.

    Parameters
    ----------
    period:
        Lookback period for Wilder smoothing. Default: 14.
    """

    def __init__(self, period: int = 14) -> None:
        self.period = period

    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> np.ndarray:
        """Calculate ATR using Wilder's smoothing.

        True Range = max(H − L, |H − prevC|, |L − prevC|)
        ATR        = Wilder smooth of TR

        Parameters
        ----------
        high, low, close:
            Equal-length 1-D float arrays.

        Returns
        -------
        np.ndarray of ATR values (NaN for the first ``period - 1`` bars).
        """
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)
        close = np.asarray(close, dtype=float)

        n = len(close)

        prev_close = np.empty(n, dtype=float)
        prev_close[0] = close[0]
        prev_close[1:] = close[:-1]

        tr = np.maximum(
            high - low,
            np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
        )
        tr[0] = high[0] - low[0]  # First bar: no prior close

        return wilders_smooth(tr, self.period)


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

@dataclass
class RSIResult:
    """Output of :class:`RSICalculator`."""

    rsi: np.ndarray
    """RSI values (0–100)."""

    overbought: np.ndarray
    """Boolean mask: ``rsi > 70``."""

    oversold: np.ndarray
    """Boolean mask: ``rsi < 30``."""


class RSICalculator:
    """Relative Strength Index.

    Parameters
    ----------
    period:
        Lookback period. Default: 14.
    overbought:
        Level above which price is considered overbought. Default: 70.
    oversold:
        Level below which price is considered oversold. Default: 30.
    """

    def __init__(
        self,
        period: int = 14,
        overbought: float = 70.0,
        oversold: float = 30.0,
    ) -> None:
        self.period = period
        self.overbought_level = overbought
        self.oversold_level = oversold

    def calculate(self, close: np.ndarray) -> RSIResult:
        """Calculate RSI using Wilder's smoothing method.

        Steps:
        1. delta[i] = close[i] − close[i-1]
        2. gains    = max(delta, 0)
        3. losses   = max(−delta, 0)
        4. avg_gain = Wilder smooth of gains
        5. avg_loss = Wilder smooth of losses
        6. RS       = avg_gain / avg_loss
        7. RSI      = 100 − 100 / (1 + RS)

        Parameters
        ----------
        close:
            1-D float array of closing prices.

        Returns
        -------
        RSIResult
        """
        close = np.asarray(close, dtype=float)

        n = len(close)
        delta = np.empty(n, dtype=float)
        delta[0] = 0.0
        delta[1:] = np.diff(close)

        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)

        avg_gain = wilders_smooth(gains, self.period)
        avg_loss = wilders_smooth(losses, self.period)

        with np.errstate(invalid="ignore", divide="ignore"):
            rs = np.where(avg_loss > 0, avg_gain / avg_loss, np.inf)
            rsi = np.where(np.isinf(rs), 100.0, 100.0 - 100.0 / (1.0 + rs))

        # Propagate NaN from warm-up period
        nan_mask = np.isnan(avg_gain)
        rsi = np.where(nan_mask, np.nan, rsi)

        valid = ~np.isnan(rsi)
        overbought = np.where(valid, rsi > self.overbought_level, False).astype(bool)
        oversold = np.where(valid, rsi < self.oversold_level, False).astype(bool)

        return RSIResult(rsi=rsi, overbought=overbought, oversold=oversold)


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

@dataclass
class BBResult:
    """Output of :class:`BollingerBandCalculator`."""

    upper: np.ndarray
    """Upper Bollinger Band."""

    middle: np.ndarray
    """Middle band (SMA)."""

    lower: np.ndarray
    """Lower Bollinger Band."""

    width: np.ndarray
    """Normalised band width: (upper − lower) / middle."""

    width_percentile: np.ndarray
    """Rolling percentile rank of width over ``squeeze_lookback`` bars (0–100)."""

    is_squeeze: np.ndarray
    """Boolean: ``width_percentile < squeeze_percentile``."""


class BollingerBandCalculator:
    """Bollinger Bands with normalised width, rolling percentile rank, and squeeze detection.

    A squeeze indicates compressed volatility — a common precursor to a
    directional breakout. Gold-specific default: squeeze threshold at the 20th
    percentile of the width distribution over the lookback window.

    Parameters
    ----------
    period:
        SMA and standard deviation window. Default: 20.
    std_dev:
        Number of standard deviations for band width. Default: 2.0.
    squeeze_lookback:
        Rolling window over which the percentile rank of width is computed. Default: 100.
    squeeze_percentile:
        Width percentile below which a squeeze is declared. Default: 20.0.
    """

    def __init__(
        self,
        period: int = 20,
        std_dev: float = 2.0,
        squeeze_lookback: int = 100,
        squeeze_percentile: float = 20.0,
    ) -> None:
        self.period = period
        self.std_dev = std_dev
        self.squeeze_lookback = squeeze_lookback
        self.squeeze_percentile = squeeze_percentile

    def calculate(self, close: np.ndarray) -> BBResult:
        """Calculate Bollinger Bands, normalised width, percentile rank, and squeeze flag.

        Percentile rank at bar ``i`` answers: "What fraction of the
        ``squeeze_lookback`` prior width values is the current width above?"
        A low rank (< 20th percentile) indicates unusually narrow bands.

        Parameters
        ----------
        close:
            1-D float array of closing prices.

        Returns
        -------
        BBResult
        """
        close = np.asarray(close, dtype=float)
        n = len(close)

        # Rolling SMA and std using stride tricks for full vectorisation
        # pandas rolling is acceptable here (avoids explicit loops)
        import pandas as pd

        s = pd.Series(close)
        middle = s.rolling(self.period).mean().to_numpy()
        std = s.rolling(self.period).std(ddof=0).to_numpy()  # population std, matching Wilder convention

        upper = middle + self.std_dev * std
        lower = middle - self.std_dev * std

        with np.errstate(invalid="ignore", divide="ignore"):
            width = np.where(middle != 0, (upper - lower) / middle, np.nan)

        # Rolling percentile rank of width
        # For each bar i, compute: fraction of width[i-lookback:i] that is <= width[i]
        width_percentile = np.full(n, np.nan, dtype=float)

        # Vectorised approach: build a 2-D window matrix using pandas rolling apply
        width_series = pd.Series(width)

        def _percentile_rank(window: np.ndarray) -> float:
            # Percentile rank of the last element within the window (0–100)
            current = window[-1]
            if np.isnan(current):
                return np.nan
            valid = window[~np.isnan(window)]
            if len(valid) == 0:
                return np.nan
            return float(np.sum(valid <= current) / len(valid) * 100.0)

        # min_periods ensures we only compute once we have a full window
        wp = width_series.rolling(
            window=self.squeeze_lookback, min_periods=self.squeeze_lookback
        ).apply(_percentile_rank, raw=True)
        width_percentile = wp.to_numpy()

        valid = ~np.isnan(width_percentile)
        is_squeeze = np.where(
            valid, width_percentile < self.squeeze_percentile, False
        ).astype(bool)

        return BBResult(
            upper=upper,
            middle=middle,
            lower=lower,
            width=width,
            width_percentile=width_percentile,
            is_squeeze=is_squeeze,
        )
