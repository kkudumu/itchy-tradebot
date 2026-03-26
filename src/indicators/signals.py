"""Signal detection built on top of Ichimoku components.

All methods are vectorized — no Python-level loops over bar indices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class IchimokuSignalState:
    """Snapshot of all Ichimoku signal dimensions at a single bar."""

    cloud_direction: int    # 1 bullish, -1 bearish, 0 neutral/flat
    tk_cross: int           # 1 bullish cross, -1 bearish cross, 0 none
    cloud_position: int     # 1 above cloud, -1 below cloud, 0 inside
    chikou_confirmed: int   # 1 bullish, -1 bearish, 0 neutral
    cloud_twist: int        # 1 bullish twist, -1 bearish twist, 0 none
    cloud_thickness: float  # |Senkou A - Senkou B|


class IchimokuSignals:
    """Vectorized signal detection from pre-computed Ichimoku components.

    All public methods accept numpy arrays and return numpy arrays of
    the same length. Signal values follow the convention:

    +1  → bullish / confirming
    -1  → bearish / opposing
     0  → no signal, neutral, or insufficient data (NaN in inputs)
    """

    # ------------------------------------------------------------------
    # TK Cross
    # ------------------------------------------------------------------

    def tk_cross(
        self,
        tenkan: np.ndarray,
        kijun: np.ndarray,
    ) -> np.ndarray:
        """Detect Tenkan/Kijun crossovers bar-by-bar.

        A bullish cross occurs when Tenkan crosses ABOVE Kijun
        (tenkan[i-1] <= kijun[i-1] and tenkan[i] > kijun[i]).
        A bearish cross is the mirror condition.

        Parameters
        ----------
        tenkan, kijun : np.ndarray
            Pre-computed Tenkan-Sen and Kijun-Sen arrays.

        Returns
        -------
        np.ndarray of int8
            1 at each bullish cross bar, -1 at each bearish cross bar, 0 otherwise.
        """
        t = np.asarray(tenkan, dtype=np.float64)
        k = np.asarray(kijun, dtype=np.float64)

        result = np.zeros(len(t), dtype=np.int8)
        if len(t) < 2:
            return result

        # diff > 0 when tenkan is above kijun
        diff_cur = t[1:] - k[1:]
        diff_prev = t[:-1] - k[:-1]

        valid = ~(np.isnan(diff_cur) | np.isnan(diff_prev))

        # Bullish cross: prev tenkan <= kijun, cur tenkan > kijun
        bullish = valid & (diff_prev <= 0) & (diff_cur > 0)
        # Bearish cross: prev tenkan >= kijun, cur tenkan < kijun
        bearish = valid & (diff_prev >= 0) & (diff_cur < 0)

        result[1:][bullish] = 1
        result[1:][bearish] = -1
        return result

    # ------------------------------------------------------------------
    # Cloud position
    # ------------------------------------------------------------------

    def cloud_position(
        self,
        close: np.ndarray,
        senkou_a: np.ndarray,
        senkou_b: np.ndarray,
    ) -> np.ndarray:
        """Classify price position relative to the Ichimoku cloud.

        Parameters
        ----------
        close : np.ndarray
            Closing prices.
        senkou_a, senkou_b : np.ndarray
            Leading Span A and B (may contain NaN at displaced edges).

        Returns
        -------
        np.ndarray of int8
             1 — close is above both spans (bullish cloud position)
            -1 — close is below both spans (bearish cloud position)
             0 — close is inside the cloud, or inputs are NaN
        """
        c = np.asarray(close, dtype=np.float64)
        a = np.asarray(senkou_a, dtype=np.float64)
        b = np.asarray(senkou_b, dtype=np.float64)

        result = np.zeros(len(c), dtype=np.int8)
        valid = ~(np.isnan(c) | np.isnan(a) | np.isnan(b))

        cloud_top = np.maximum(a, b)
        cloud_bottom = np.minimum(a, b)

        above = valid & (c > cloud_top)
        below = valid & (c < cloud_bottom)
        result[above] = 1
        result[below] = -1
        return result

    # ------------------------------------------------------------------
    # Chikou confirmation
    # ------------------------------------------------------------------

    def chikou_confirmation(
        self,
        chikou: np.ndarray,
        close: np.ndarray,
    ) -> np.ndarray:
        """Check whether the Chikou Span confirms directional bias.

        The Chikou Span at bar *i* is close[i] plotted kijun_period bars
        back. To confirm bullish bias it must be ABOVE the price that was
        there kijun_period bars ago — which is simply close[i] itself
        compared to close[i - kijun_period].

        Because ``chikou[i] = close[i + kijun_period]`` by construction
        (see IchimokuCalculator), comparing chikou[i] with close[i] gives
        the same result as comparing close[i + kijun_period] with close[i].

        The comparison made here is element-wise:
            chikou[i] vs close[i]

        This matches standard charting platform behaviour where the
        Chikou value visible at bar *i* is tested against the current bar's
        closing price.

        Returns
        -------
        np.ndarray of int8
             1 — Chikou above close (bullish confirmation)
            -1 — Chikou below close (bearish confirmation)
             0 — equal or NaN
        """
        chi = np.asarray(chikou, dtype=np.float64)
        c = np.asarray(close, dtype=np.float64)

        result = np.zeros(len(chi), dtype=np.int8)
        valid = ~(np.isnan(chi) | np.isnan(c))
        result[valid & (chi > c)] = 1
        result[valid & (chi < c)] = -1
        return result

    # ------------------------------------------------------------------
    # Cloud twist
    # ------------------------------------------------------------------

    def cloud_twist(
        self,
        senkou_a: np.ndarray,
        senkou_b: np.ndarray,
    ) -> np.ndarray:
        """Detect cloud twists — crossovers between Senkou A and Senkou B.

        A bullish twist occurs when Senkou A crosses above Senkou B.
        A bearish twist occurs when Senkou A crosses below Senkou B.

        Because the Senkou spans are already displaced forward in the
        arrays, a twist detected at index *i* in the array corresponds
        to a future cloud event on the chart (kijun_period bars ahead).

        Returns
        -------
        np.ndarray of int8
             1 — bullish twist (A crosses above B)
            -1 — bearish twist (A crosses below B)
             0 — no twist or insufficient data
        """
        a = np.asarray(senkou_a, dtype=np.float64)
        b = np.asarray(senkou_b, dtype=np.float64)

        result = np.zeros(len(a), dtype=np.int8)
        if len(a) < 2:
            return result

        diff_cur = a[1:] - b[1:]
        diff_prev = a[:-1] - b[:-1]

        valid = ~(np.isnan(diff_cur) | np.isnan(diff_prev))

        bullish = valid & (diff_prev <= 0) & (diff_cur > 0)
        bearish = valid & (diff_prev >= 0) & (diff_cur < 0)

        result[1:][bullish] = 1
        result[1:][bearish] = -1
        return result

    # ------------------------------------------------------------------
    # Cloud breakout
    # ------------------------------------------------------------------

    def cloud_breakout(
        self,
        close: np.ndarray,
        senkou_a: np.ndarray,
        senkou_b: np.ndarray,
    ) -> np.ndarray:
        """Detect breakouts through the cloud boundary.

        A bullish breakout occurs when close crosses ABOVE the cloud top
        (i.e. was at or inside cloud, now above cloud top).
        A bearish breakout is the mirror — crosses below cloud bottom.

        Parameters
        ----------
        close : np.ndarray
            Closing prices.
        senkou_a, senkou_b : np.ndarray
            Leading Span A and B arrays.

        Returns
        -------
        np.ndarray of int8
             1 — bullish breakout bar
            -1 — bearish breakout bar
             0 — no breakout
        """
        c = np.asarray(close, dtype=np.float64)
        a = np.asarray(senkou_a, dtype=np.float64)
        b = np.asarray(senkou_b, dtype=np.float64)

        result = np.zeros(len(c), dtype=np.int8)
        if len(c) < 2:
            return result

        cloud_top = np.maximum(a, b)
        cloud_bottom = np.minimum(a, b)

        # Previous bar position relative to cloud
        prev_above = c[:-1] > cloud_top[:-1]
        prev_below = c[:-1] < cloud_bottom[:-1]
        prev_inside = ~prev_above & ~prev_below

        cur_above = c[1:] > cloud_top[1:]
        cur_below = c[1:] < cloud_bottom[1:]

        # Valid: no NaN in either bar
        valid = ~(
            np.isnan(c[:-1]) | np.isnan(c[1:])
            | np.isnan(cloud_top[:-1]) | np.isnan(cloud_top[1:])
            | np.isnan(cloud_bottom[:-1]) | np.isnan(cloud_bottom[1:])
        )

        # Bullish breakout: previous bar was not above cloud, current bar is above
        bullish = valid & ~prev_above & cur_above
        # Bearish breakout: previous bar was not below cloud, current bar is below
        bearish = valid & ~prev_below & cur_below

        result[1:][bullish] = 1
        result[1:][bearish] = -1
        return result

    # ------------------------------------------------------------------
    # Convenience: full signal state snapshot for a single bar
    # ------------------------------------------------------------------

    def signal_state_at(
        self,
        idx: int,
        tenkan: np.ndarray,
        kijun: np.ndarray,
        close: np.ndarray,
        senkou_a: np.ndarray,
        senkou_b: np.ndarray,
        chikou: np.ndarray,
    ) -> IchimokuSignalState:
        """Return all signal dimensions for a single bar index.

        Parameters
        ----------
        idx : int
            Bar index to inspect.
        tenkan, kijun, close, senkou_a, senkou_b, chikou : np.ndarray
            Pre-computed Ichimoku component arrays.

        Returns
        -------
        IchimokuSignalState
        """
        from .ichimoku import IchimokuCalculator

        calc = IchimokuCalculator()

        tk = self.tk_cross(tenkan, kijun)[idx]
        cp = self.cloud_position(close, senkou_a, senkou_b)[idx]
        cc = self.chikou_confirmation(chikou, close)[idx]
        ct = self.cloud_twist(senkou_a, senkou_b)[idx]
        cd = calc.cloud_direction(senkou_a, senkou_b)[idx]
        th = float(calc.cloud_thickness(senkou_a, senkou_b)[idx])

        return IchimokuSignalState(
            cloud_direction=int(cd),
            tk_cross=int(tk),
            cloud_position=int(cp),
            chikou_confirmed=int(cc),
            cloud_twist=int(ct),
            cloud_thickness=th,
        )
