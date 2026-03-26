"""
Fractal swing point detection for identifying significant highs and lows.

A swing high at bar i is confirmed when high[i] is the highest value within
[i-lookback, i+lookback].  A swing low is the mirror definition on the low array.
Both definitions are inclusive on both sides.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SwingPoints:
    """Container for detected swing highs and swing lows.

    Each entry is a tuple of (bar_index, price, optional_timestamp).
    When timestamps are not provided, the third element is None.
    """

    swing_highs: list[tuple] = field(default_factory=list)
    """List of (index, price) or (index, price, timestamp) for each swing high."""

    swing_lows: list[tuple] = field(default_factory=list)
    """List of (index, price) or (index, price, timestamp) for each swing low."""


class SwingPointDetector:
    """Detect fractal swing highs and lows using a configurable lookback window.

    A swing high at bar ``i`` satisfies:
        high[i] >= max(high[i-lookback : i])   and
        high[i] >= max(high[i+1 : i+lookback+1])

    i.e. the bar is at least as high as the ``lookback`` bars on either side.
    Equality is permitted so that flat-top formations are captured.

    A swing low at bar ``i`` satisfies the mirror condition on the ``low`` array.

    Parameters
    ----------
    lookback:
        Number of bars on each side required to confirm a swing point.
        Must be >= 1.  Typical values: 5 (short-term) to 20 (major structure).
    """

    def __init__(self, lookback: int = 10) -> None:
        if lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {lookback}")
        self.lookback = lookback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        high: np.ndarray,
        low: np.ndarray,
        timestamps: np.ndarray | None = None,
    ) -> SwingPoints:
        """Detect swing highs and lows, optionally attaching timestamps.

        Parameters
        ----------
        high:
            1-D array of bar high prices.
        low:
            1-D array of bar low prices, same length as ``high``.
        timestamps:
            Optional 1-D array of timestamps (any comparable type), same length
            as ``high``.  When supplied, each swing point tuple includes the
            corresponding timestamp as its third element.

        Returns
        -------
        SwingPoints
            Two lists — one for swing highs and one for swing lows — each
            containing tuples of ``(index, price)`` or
            ``(index, price, timestamp)``.
        """
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)

        sh_mask, sl_mask = self.detect_vectorized(high, low)

        swing_highs: list[tuple] = []
        swing_lows: list[tuple] = []

        for i in np.where(sh_mask)[0]:
            if timestamps is not None:
                swing_highs.append((int(i), float(high[i]), timestamps[i]))
            else:
                swing_highs.append((int(i), float(high[i])))

        for i in np.where(sl_mask)[0]:
            if timestamps is not None:
                swing_lows.append((int(i), float(low[i]), timestamps[i]))
            else:
                swing_lows.append((int(i), float(low[i])))

        return SwingPoints(swing_highs=swing_highs, swing_lows=swing_lows)

    def detect_vectorized(
        self,
        high: np.ndarray,
        low: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return boolean masks for swing highs and swing lows.

        Parameters
        ----------
        high, low:
            Equal-length 1-D float arrays.

        Returns
        -------
        swing_high_mask, swing_low_mask:
            Boolean arrays of the same length as the input.  True at bar ``i``
            if that bar is a confirmed swing high / low.
        """
        high = np.asarray(high, dtype=float)
        low = np.asarray(low, dtype=float)

        n = len(high)
        lb = self.lookback

        swing_high_mask = np.zeros(n, dtype=bool)
        swing_low_mask = np.zeros(n, dtype=bool)

        # Only bars with full lookback windows on both sides can be swing points.
        for i in range(lb, n - lb):
            left_window_h = high[i - lb: i]
            right_window_h = high[i + 1: i + lb + 1]
            if high[i] >= np.max(left_window_h) and high[i] >= np.max(right_window_h):
                swing_high_mask[i] = True

            left_window_l = low[i - lb: i]
            right_window_l = low[i + 1: i + lb + 1]
            if low[i] <= np.min(left_window_l) and low[i] <= np.min(right_window_l):
                swing_low_mask[i] = True

        return swing_high_mask, swing_low_mask
