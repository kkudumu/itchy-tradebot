"""
RSI divergence detection for XAU/USD Ichimoku trading agent.

Detects regular and hidden divergences between price and RSI using
vectorised swing-point identification. A swing point is a local extremum
within a symmetric lookback window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DivergenceResult:
    """Output of :class:`DivergenceDetector`."""

    regular_bullish: np.ndarray
    """Price makes lower low; RSI makes higher low — potential trend reversal up."""

    regular_bearish: np.ndarray
    """Price makes higher high; RSI makes lower high — potential trend reversal down."""

    hidden_bullish: np.ndarray
    """Price makes higher low; RSI makes lower low — bullish trend continuation."""

    hidden_bearish: np.ndarray
    """Price makes lower high; RSI makes higher high — bearish trend continuation."""


def _find_swing_lows(values: np.ndarray, lookback: int) -> np.ndarray:
    """Boolean mask of swing-low pivot bars.

    A bar at index ``i`` is a swing low if ``values[i]`` is strictly less
    than all values in the symmetric window
    ``values[i - lookback : i]`` and ``values[i + 1 : i + lookback + 1]``.

    Parameters
    ----------
    values:
        1-D float array.
    lookback:
        Half-width of the pivot window.

    Returns
    -------
    np.ndarray of bool, same length as ``values``.
    """
    n = len(values)
    mask = np.zeros(n, dtype=bool)

    # Use numpy sliding-window minimum for efficiency
    # A point is a swing low if it equals the rolling min over (2*lookback+1)
    # centred window AND is strictly below its immediate neighbours.
    for i in range(lookback, n - lookback):
        window = values[i - lookback: i + lookback + 1]
        if not np.any(np.isnan(window)):
            if values[i] == np.min(window) and np.sum(window == values[i]) == 1:
                mask[i] = True
    return mask


def _find_swing_highs(values: np.ndarray, lookback: int) -> np.ndarray:
    """Boolean mask of swing-high pivot bars.

    Symmetric counterpart of :func:`_find_swing_lows`.
    """
    n = len(values)
    mask = np.zeros(n, dtype=bool)

    for i in range(lookback, n - lookback):
        window = values[i - lookback: i + lookback + 1]
        if not np.any(np.isnan(window)):
            if values[i] == np.max(window) and np.sum(window == values[i]) == 1:
                mask[i] = True
    return mask


class DivergenceDetector:
    """Detect regular and hidden RSI divergences against price.

    Divergence is evaluated at swing-point pairs separated by at least
    ``min_bars_between`` bars and at most ``max_bars_between`` bars.

    Parameters
    ----------
    lookback:
        Half-window for swing-point detection (bars on each side). Default: 5.
    min_bars_between:
        Minimum separation between two swing points to form a divergence pair. Default: 5.
    max_bars_between:
        Maximum separation between two swing points to form a divergence pair. Default: 50.
    """

    def __init__(
        self,
        lookback: int = 5,
        min_bars_between: int = 5,
        max_bars_between: int = 50,
    ) -> None:
        self.lookback = lookback
        self.min_bars_between = min_bars_between
        self.max_bars_between = max_bars_between

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        close: np.ndarray,
        rsi: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
    ) -> DivergenceResult:
        """Detect regular and hidden divergences between price and RSI.

        Divergence types
        ----------------
        Regular Bullish:
            Price forms a lower low; RSI forms a higher low at the same swing.
            Signals weakening bearish momentum — potential reversal upward.

        Regular Bearish:
            Price forms a higher high; RSI forms a lower high.
            Signals weakening bullish momentum — potential reversal downward.

        Hidden Bullish:
            Price forms a higher low (pullback within uptrend); RSI forms a
            lower low. Confirms trend continuation upward.

        Hidden Bearish:
            Price forms a lower high (rally within downtrend); RSI forms a
            higher high. Confirms trend continuation downward.

        Parameters
        ----------
        close:
            1-D closing prices (used as the general price series for lows/highs
            when ``high``/``low`` swing detection is not differentiated).
        rsi:
            1-D RSI values, same length as ``close``.
        high:
            1-D high prices.
        low:
            1-D low prices.

        Returns
        -------
        DivergenceResult
            Four boolean arrays of the same length as the input.  A ``True``
            value at index ``i`` means a divergence was *confirmed* at bar
            ``i`` (i.e., the second — most recent — pivot of the pair).
        """
        close = np.asarray(close, dtype=float)
        rsi = np.asarray(rsi, dtype=float)
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)

        n = len(close)
        reg_bull = np.zeros(n, dtype=bool)
        reg_bear = np.zeros(n, dtype=bool)
        hid_bull = np.zeros(n, dtype=bool)
        hid_bear = np.zeros(n, dtype=bool)

        # Swing pivot indices
        low_pivots = np.where(_find_swing_lows(low, self.lookback))[0]
        high_pivots = np.where(_find_swing_highs(high, self.lookback))[0]
        rsi_low_pivots = np.where(_find_swing_lows(rsi, self.lookback))[0]
        rsi_high_pivots = np.where(_find_swing_highs(rsi, self.lookback))[0]

        # --- Bullish divergences (compare consecutive swing lows) ---
        reg_bull, hid_bull = self._compare_lows(
            low, rsi, low_pivots, rsi_low_pivots, n, reg_bull, hid_bull
        )

        # --- Bearish divergences (compare consecutive swing highs) ---
        reg_bear, hid_bear = self._compare_highs(
            high, rsi, high_pivots, rsi_high_pivots, n, reg_bear, hid_bear
        )

        return DivergenceResult(
            regular_bullish=reg_bull,
            regular_bearish=reg_bear,
            hidden_bullish=hid_bull,
            hidden_bearish=hid_bear,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compare_lows(
        self,
        price_low: np.ndarray,
        rsi: np.ndarray,
        price_low_pivots: np.ndarray,
        rsi_low_pivots: np.ndarray,
        n: int,
        reg_bull: np.ndarray,
        hid_bull: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Detect bullish divergences from consecutive price and RSI swing lows."""
        for j, p2 in enumerate(price_low_pivots[1:], start=1):
            p1 = price_low_pivots[j - 1]
            gap = p2 - p1
            if gap < self.min_bars_between or gap > self.max_bars_between:
                continue

            # Find RSI swing low closest to p2 within the same window
            candidates = rsi_low_pivots[
                (rsi_low_pivots >= p1) & (rsi_low_pivots <= p2)
            ]
            if len(candidates) < 2:
                continue

            r1_idx = candidates[-2]
            r2_idx = candidates[-1]

            price_lower_low = price_low[p2] < price_low[p1]
            price_higher_low = price_low[p2] > price_low[p1]
            rsi_higher_low = rsi[r2_idx] > rsi[r1_idx]
            rsi_lower_low = rsi[r2_idx] < rsi[r1_idx]

            # Regular bullish: price lower low + RSI higher low
            if price_lower_low and rsi_higher_low:
                reg_bull[p2] = True

            # Hidden bullish: price higher low + RSI lower low
            if price_higher_low and rsi_lower_low:
                hid_bull[p2] = True

        return reg_bull, hid_bull

    def _compare_highs(
        self,
        price_high: np.ndarray,
        rsi: np.ndarray,
        price_high_pivots: np.ndarray,
        rsi_high_pivots: np.ndarray,
        n: int,
        reg_bear: np.ndarray,
        hid_bear: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Detect bearish divergences from consecutive price and RSI swing highs."""
        for j, p2 in enumerate(price_high_pivots[1:], start=1):
            p1 = price_high_pivots[j - 1]
            gap = p2 - p1
            if gap < self.min_bars_between or gap > self.max_bars_between:
                continue

            candidates = rsi_high_pivots[
                (rsi_high_pivots >= p1) & (rsi_high_pivots <= p2)
            ]
            if len(candidates) < 2:
                continue

            r1_idx = candidates[-2]
            r2_idx = candidates[-1]

            price_higher_high = price_high[p2] > price_high[p1]
            price_lower_high = price_high[p2] < price_high[p1]
            rsi_lower_high = rsi[r2_idx] < rsi[r1_idx]
            rsi_higher_high = rsi[r2_idx] > rsi[r1_idx]

            # Regular bearish: price higher high + RSI lower high
            if price_higher_high and rsi_lower_high:
                reg_bear[p2] = True

            # Hidden bearish: price lower high + RSI higher high
            if price_lower_high and rsi_higher_high:
                hid_bear[p2] = True

        return reg_bear, hid_bear
