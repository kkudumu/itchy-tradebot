"""
Standard and Fibonacci pivot point calculations.

Pivot points provide objective price levels derived from the prior period's
high, low, and close.  Both standard (arithmetic) and Fibonacci variants are
supported.  Pivot sets can be calculated from pre-aggregated daily, weekly, or
monthly OHLC DataFrames.
"""

from __future__ import annotations

import pandas as pd


class PivotCalculator:
    """Calculate standard and Fibonacci pivot points from OHLC data.

    All pivot formulas operate on a single period's high (H), low (L), and
    close (C).  The resulting levels are forward-projected to the *next*
    period — callers are responsible for aligning pivot levels to future bars.
    """

    # ------------------------------------------------------------------
    # Single-period pivot calculation
    # ------------------------------------------------------------------

    def standard_pivots(self, high: float, low: float, close: float) -> dict[str, float]:
        """Calculate standard (classic) pivot points for a single period.

        Formulas
        --------
        PP  = (H + L + C) / 3
        R1  = 2 × PP − L
        R2  = PP + (H − L)
        R3  = H + 2 × (PP − L)
        S1  = 2 × PP − H
        S2  = PP − (H − L)
        S3  = L − 2 × (H − PP)

        Parameters
        ----------
        high, low, close:
            Scalar OHLC values for the reference period.

        Returns
        -------
        dict with keys: ``PP``, ``R1``, ``R2``, ``R3``, ``S1``, ``S2``, ``S3``.
        """
        pp = (high + low + close) / 3.0
        hl_range = high - low

        return {
            "PP": pp,
            "R1": 2.0 * pp - low,
            "R2": pp + hl_range,
            "R3": high + 2.0 * (pp - low),
            "S1": 2.0 * pp - high,
            "S2": pp - hl_range,
            "S3": low - 2.0 * (high - pp),
        }

    def fibonacci_pivots(self, high: float, low: float, close: float) -> dict[str, float]:
        """Calculate Fibonacci-based pivot points for a single period.

        Formulas
        --------
        PP  = (H + L + C) / 3
        R1  = PP + 0.382 × (H − L)
        R2  = PP + 0.618 × (H − L)
        R3  = PP + 1.000 × (H − L)
        S1  = PP − 0.382 × (H − L)
        S2  = PP − 0.618 × (H − L)
        S3  = PP − 1.000 × (H − L)

        Parameters
        ----------
        high, low, close:
            Scalar OHLC values for the reference period.

        Returns
        -------
        dict with keys: ``PP``, ``R1``, ``R2``, ``R3``, ``S1``, ``S2``, ``S3``.
        """
        pp = (high + low + close) / 3.0
        hl_range = high - low

        return {
            "PP": pp,
            "R1": pp + 0.382 * hl_range,
            "R2": pp + 0.618 * hl_range,
            "R3": pp + 1.000 * hl_range,
            "S1": pp - 0.382 * hl_range,
            "S2": pp - 0.618 * hl_range,
            "S3": pp - 1.000 * hl_range,
        }

    # ------------------------------------------------------------------
    # OHLC DataFrame helpers
    # ------------------------------------------------------------------

    def from_daily_ohlc(self, daily_ohlc: pd.DataFrame) -> pd.DataFrame:
        """Calculate standard pivot points for each day in a daily OHLC DataFrame.

        Each row in the output represents the pivot levels *valid for the next
        trading day*, derived from the current row's H, L, C.

        Parameters
        ----------
        daily_ohlc:
            DataFrame with columns ``high``, ``low``, ``close`` and a
            DatetimeIndex (or ``timestamp`` column).  One row per trading day.

        Returns
        -------
        pd.DataFrame
            Same index as ``daily_ohlc``, columns: PP, R1, R2, R3, S1, S2, S3.
        """
        return self._apply_pivots(daily_ohlc, kind="standard")

    def from_weekly_ohlc(self, weekly_ohlc: pd.DataFrame) -> pd.DataFrame:
        """Calculate standard pivot points for each week in a weekly OHLC DataFrame.

        Parameters
        ----------
        weekly_ohlc:
            DataFrame with columns ``high``, ``low``, ``close``.  One row per
            trading week.

        Returns
        -------
        pd.DataFrame with pivot levels indexed to match ``weekly_ohlc``.
        """
        return self._apply_pivots(weekly_ohlc, kind="standard")

    def from_monthly_ohlc(self, monthly_ohlc: pd.DataFrame) -> pd.DataFrame:
        """Calculate standard pivot points for each month in a monthly OHLC DataFrame.

        Parameters
        ----------
        monthly_ohlc:
            DataFrame with columns ``high``, ``low``, ``close``.  One row per
            calendar month.

        Returns
        -------
        pd.DataFrame with pivot levels indexed to match ``monthly_ohlc``.
        """
        return self._apply_pivots(monthly_ohlc, kind="standard")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_pivots(self, ohlc: pd.DataFrame, kind: str = "standard") -> pd.DataFrame:
        """Apply pivot calculations row-by-row to an OHLC DataFrame.

        Parameters
        ----------
        ohlc:
            DataFrame containing at minimum ``high``, ``low``, ``close``.
        kind:
            ``'standard'`` or ``'fibonacci'``.

        Returns
        -------
        pd.DataFrame with columns PP, R1, R2, R3, S1, S2, S3.
        """
        required = {"high", "low", "close"}
        if not required.issubset(ohlc.columns):
            raise ValueError(f"ohlc DataFrame must contain columns: {required}")

        fn = self.standard_pivots if kind == "standard" else self.fibonacci_pivots

        rows = []
        for _, row in ohlc.iterrows():
            levels = fn(float(row["high"]), float(row["low"]), float(row["close"]))
            rows.append(levels)

        result = pd.DataFrame(rows, index=ohlc.index)
        return result
