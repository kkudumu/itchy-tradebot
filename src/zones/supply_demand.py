"""
Supply and demand zone detection from impulse moves.

A demand zone is the consolidation (base) immediately preceding a strong
bullish impulse move.  A supply zone is the consolidation preceding a strong
bearish impulse move.  The zone represents the base candles, not the impulse
itself — price tends to return to these bases.

Impulse threshold: the total move must exceed ``impulse_threshold × ATR``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SDZone:
    """A supply or demand zone identified from price action.

    Attributes
    ----------
    price_high:
        Upper boundary of the zone (highest high of the base candles).
    price_low:
        Lower boundary of the zone (lowest low of the base candles).
    zone_type:
        ``'demand'`` (base before bullish impulse) or ``'supply'`` (base before
        bearish impulse).
    impulse_size:
        Size of the impulse move expressed as a multiple of ATR at detection
        time.
    base_bars:
        Number of consolidation candles that form the base.
    timestamp:
        Timestamp of the last base candle (i.e. the candle just before the
        impulse starts).
    status:
        Lifecycle state: ``'active'``, ``'tested'``, or ``'invalidated'``.
    """

    price_high: float
    price_low: float
    zone_type: str          # 'supply' or 'demand'
    impulse_size: float     # impulse move size in ATR multiples
    base_bars: int          # number of consolidation candles in the base
    timestamp: datetime
    status: str = "active"  # active, tested, invalidated


class SupplyDemandDetector:
    """Detect supply and demand zones from impulse moves.

    The algorithm scans forward through each bar.  For every position, it
    looks back up to ``consolidation_bars`` candles for a tight consolidation
    (base), then checks whether a large directional move immediately follows.

    Parameters
    ----------
    impulse_threshold:
        The impulse move must exceed this multiple of ATR to qualify as a
        valid supply/demand zone formation.  Default: 2.5.
    consolidation_bars:
        Maximum number of candles allowed in the consolidation base.
        Default: 3.
    """

    def __init__(
        self,
        impulse_threshold: float = 2.5,
        consolidation_bars: int = 3,
    ) -> None:
        if impulse_threshold <= 0:
            raise ValueError(f"impulse_threshold must be > 0, got {impulse_threshold}")
        if consolidation_bars < 1:
            raise ValueError(f"consolidation_bars must be >= 1, got {consolidation_bars}")
        self.impulse_threshold = impulse_threshold
        self.consolidation_bars = consolidation_bars

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        ohlc: pd.DataFrame,
        atr: np.ndarray,
    ) -> list[SDZone]:
        """Detect supply and demand zones in OHLC data.

        Parameters
        ----------
        ohlc:
            DataFrame with columns: ``open``, ``high``, ``low``, ``close``,
            and optionally a DatetimeIndex or a ``timestamp`` column.
        atr:
            1-D array of ATR values aligned with ``ohlc`` rows (same length).
            NaN values at the start (warm-up) are handled gracefully.

        Returns
        -------
        list[SDZone]
            All detected supply and demand zones, sorted chronologically.
        """
        required = {"open", "high", "low", "close"}
        if not required.issubset(ohlc.columns):
            raise ValueError(f"ohlc must contain columns: {required}")

        opens = ohlc["open"].to_numpy(dtype=float)
        highs = ohlc["high"].to_numpy(dtype=float)
        lows = ohlc["low"].to_numpy(dtype=float)
        closes = ohlc["close"].to_numpy(dtype=float)
        atr = np.asarray(atr, dtype=float)

        # Resolve timestamps
        if "timestamp" in ohlc.columns:
            timestamps = ohlc["timestamp"].tolist()
        elif isinstance(ohlc.index, pd.DatetimeIndex):
            timestamps = ohlc.index.to_pydatetime().tolist()
        else:
            # Fallback: integer indices wrapped as datetime sentinels
            timestamps = [datetime(2000, 1, 1) for _ in range(len(ohlc))]

        n = len(ohlc)
        zones: list[SDZone] = []
        # Track which bar indices already contributed to a zone to avoid
        # redundant duplicate zones from overlapping windows.
        used_base_end_indices: set[int] = set()

        for i in range(1, n - 1):
            current_atr = atr[i]
            if np.isnan(current_atr) or current_atr <= 0:
                continue

            threshold = self.impulse_threshold * current_atr

            # --- Check for bullish impulse starting at bar i+1 ---
            # Bullish impulse: close of impulse bar - open of impulse bar > threshold
            # (we look one bar ahead)
            bullish_move = closes[i] - opens[i]
            if bullish_move > threshold:
                # The impulse bar is bar i; look back for the base
                base_end = i - 1
                if base_end >= 0 and base_end not in used_base_end_indices:
                    base_zone = self._extract_base(
                        highs, lows, closes, timestamps, base_end,
                        zone_type="demand",
                        impulse_size=bullish_move / current_atr,
                        atr=current_atr,
                    )
                    if base_zone is not None:
                        zones.append(base_zone)
                        used_base_end_indices.add(base_end)

            # --- Check for bearish impulse starting at bar i ---
            bearish_move = opens[i] - closes[i]
            if bearish_move > threshold:
                base_end = i - 1
                if base_end >= 0 and base_end not in used_base_end_indices:
                    base_zone = self._extract_base(
                        highs, lows, closes, timestamps, base_end,
                        zone_type="supply",
                        impulse_size=bearish_move / current_atr,
                        atr=current_atr,
                    )
                    if base_zone is not None:
                        zones.append(base_zone)
                        used_base_end_indices.add(base_end)

        return zones

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_base(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        timestamps: list,
        base_end: int,
        zone_type: str,
        impulse_size: float,
        atr: float,
    ) -> Optional[SDZone]:
        """Extract consolidation base ending at ``base_end``.

        The base is the longest run of tight (low-range) candles ending at
        ``base_end``, up to ``consolidation_bars`` candles long.  We scan
        backwards until either the range expands beyond a threshold or we
        exhaust ``consolidation_bars``.
        """
        cb = self.consolidation_bars
        start = max(0, base_end - cb + 1)

        base_highs = highs[start: base_end + 1]
        base_lows = lows[start: base_end + 1]

        if len(base_highs) == 0:
            return None

        zone_high = float(np.max(base_highs))
        zone_low = float(np.min(base_lows))

        # A base must be tighter than the impulse — guard against wide bases
        base_range = zone_high - zone_low
        if base_range >= atr * self.impulse_threshold:
            return None

        return SDZone(
            price_high=zone_high,
            price_low=zone_low,
            zone_type=zone_type,
            impulse_size=float(impulse_size),
            base_bars=base_end - start + 1,
            timestamp=timestamps[base_end],
            status="active",
        )
