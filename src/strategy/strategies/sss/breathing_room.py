"""Breathing Room Detector — vectorized swing high/low detection for 1M gold data.

Identifies structural swing points in price data using a rolling window approach.
Guarantees alternating high/low output and filters noise via min_swing_pips.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_LOOKBACK_N: int = 3
_DEFAULT_MIN_SWING_PIPS: float = 1.0
_DEFAULT_PIP_VALUE: float = 0.1  # $0.10 per pip for XAUUSD


# ---------------------------------------------------------------------------
# SwingPoint
# ---------------------------------------------------------------------------

@dataclass
class SwingPoint:
    """A single confirmed swing high or swing low."""

    index: int              # Bar index in the DataFrame
    timestamp: datetime     # Timestamp of the swing bar
    price: float            # Price of the swing (high for swing highs, low for swing lows)
    swing_type: str         # 'high' or 'low'
    bar_count_since_prev: int  # Number of bars since the previous swing point


# ---------------------------------------------------------------------------
# BreathingRoomDetector
# ---------------------------------------------------------------------------

class BreathingRoomDetector:
    """Vectorized swing high/low detector for gold 1M data.

    Uses a rolling window of ``2 * lookback_n + 1`` bars centred on each bar
    to identify local extrema.  Results are filtered by minimum swing size and
    forced to alternate between highs and lows (keeping the more extreme point
    when consecutive same-type swings are found).

    Parameters
    ----------
    lookback_n:
        Number of bars on each side of the candidate bar.  Default 3 gives a
        7-bar window.
    min_swing_pips:
        Minimum distance in pips from the previous swing to qualify.
    pip_value:
        Dollar value per pip (0.1 for XAUUSD so 1 pip = $0.10).
    """

    def __init__(
        self,
        lookback_n: int = _DEFAULT_LOOKBACK_N,
        min_swing_pips: float = _DEFAULT_MIN_SWING_PIPS,
        pip_value: float = _DEFAULT_PIP_VALUE,
    ) -> None:
        self.lookback_n = lookback_n
        self.min_swing_pips = min_swing_pips
        self.pip_value = pip_value
        self._min_move: float = min_swing_pips * pip_value

        # Incremental state — rolling buffer of (bar_index, high, low, timestamp)
        self._window_size: int = 2 * lookback_n + 1
        self._buffer: deque = deque(maxlen=self._window_size)
        self._next_bar_idx: int = 0  # monotonically increasing bar counter

    # ------------------------------------------------------------------
    # Batch detection
    # ------------------------------------------------------------------

    def detect(self, df: pd.DataFrame) -> List[SwingPoint]:
        """Detect all swing highs and lows in *df* using vectorized operations.

        Parameters
        ----------
        df:
            DataFrame with columns ``'high'`` and ``'low'`` and a
            ``DatetimeIndex``.  Rows with NaN values in either column are
            skipped.

        Returns
        -------
        List[SwingPoint]
            Sorted by bar index, guaranteeing alternating high/low order.
        """
        if df.empty:
            return []

        window = 2 * self.lookback_n + 1

        # --- vectorized local-extrema detection --------------------------
        roll_max = df["high"].rolling(window, center=True).max()
        roll_min = df["low"].rolling(window, center=True).min()

        # A bar is a swing high when its high equals the rolling max.
        # NaN comparisons yield False, so NaN bars are automatically excluded.
        swing_high_mask: pd.Series = df["high"] == roll_max
        swing_low_mask: pd.Series = df["low"] == roll_min

        # Exclude NaN rows explicitly
        valid_mask = df["high"].notna() & df["low"].notna()
        swing_high_mask = swing_high_mask & valid_mask
        swing_low_mask = swing_low_mask & valid_mask

        # Vectorised candidate extraction — no Python loops over individual bars.
        # Use numpy positional arrays so timestamp lookup is a numpy slice, not
        # a per-element pandas call.
        ts_arr = df.index.to_numpy()     # numpy datetime64 — O(1) element access
        highs_arr = df["high"].to_numpy()
        lows_arr = df["low"].to_numpy()

        h_pos = swing_high_mask.to_numpy().nonzero()[0]  # shape (H,)
        l_pos = swing_low_mask.to_numpy().nonzero()[0]   # shape (L,)

        # Merge positions, keeping type tag: 1 = high, 0 = low
        all_pos = np.concatenate([h_pos, l_pos])
        type_tag = np.concatenate([
            np.ones(len(h_pos), dtype=np.int8),
            np.zeros(len(l_pos), dtype=np.int8),
        ])

        if len(all_pos) == 0:
            return []

        # Sort merged array by bar position (stable so equal positions keep order)
        order = np.argsort(all_pos, kind="stable")
        all_pos = all_pos[order]
        type_tag = type_tag[order]

        # Build SwingPoint list in one pass over the sorted merged arrays
        raw_points: List[SwingPoint] = []
        for i in range(len(all_pos)):
            pos = int(all_pos[i])
            is_high = bool(type_tag[i])
            raw_points.append(SwingPoint(
                index=pos,
                timestamp=ts_arr[pos],
                price=float(highs_arr[pos]) if is_high else float(lows_arr[pos]),
                swing_type="high" if is_high else "low",
                bar_count_since_prev=0,
            ))

        return self._filter_and_alternate(raw_points)

    # ------------------------------------------------------------------
    # Incremental / bar-by-bar detection
    # ------------------------------------------------------------------

    def detect_incremental(
        self,
        bar_index: int,
        timestamp: datetime,
        high: float,
        low: float,
        history: List[SwingPoint],
    ) -> Optional[SwingPoint]:
        """Process one bar and return a confirmed swing if found.

        A swing is confirmed when the candidate bar (``lookback_n`` bars ago)
        has held as the local extremum for all subsequent bars in the buffer.

        Parameters
        ----------
        bar_index:
            Monotonically increasing integer index of the incoming bar.
        timestamp:
            Timestamp of the incoming bar.
        high, low:
            OHLCV values for the incoming bar.
        history:
            List of previously confirmed SwingPoints (mutated in-place if a
            new point is confirmed).

        Returns
        -------
        SwingPoint or None
        """
        self._buffer.append((bar_index, timestamp, high, low))

        # Need a full window before we can confirm anything
        if len(self._buffer) < self._window_size:
            return None

        # Candidate is the centre of the current window
        mid = self.lookback_n  # 0-indexed position in deque
        c_idx, c_ts, c_high, c_low = self._buffer[mid]

        window_highs = [b[2] for b in self._buffer]
        window_lows = [b[3] for b in self._buffer]

        is_swing_high = (c_high == max(window_highs))
        is_swing_low = (c_low == min(window_lows))

        # Both can't be true simultaneously (unless flat bar) — prefer high
        if is_swing_high and is_swing_low:
            is_swing_low = False

        candidate: Optional[SwingPoint] = None
        if is_swing_high:
            candidate = SwingPoint(
                index=c_idx,
                timestamp=c_ts,
                price=c_high,
                swing_type="high",
                bar_count_since_prev=0,
            )
        elif is_swing_low:
            candidate = SwingPoint(
                index=c_idx,
                timestamp=c_ts,
                price=c_low,
                swing_type="low",
                bar_count_since_prev=0,
            )

        if candidate is None:
            return None

        # Apply min_swing_pips filter against latest confirmed swing
        prev = history[-1] if history else None
        if prev is not None:
            if abs(candidate.price - prev.price) < self._min_move:
                return None

        # Alternation enforcement — replace previous if same type
        if prev is not None and prev.swing_type == candidate.swing_type:
            if candidate.swing_type == "high" and candidate.price > prev.price:
                history[-1] = candidate
            elif candidate.swing_type == "low" and candidate.price < prev.price:
                history[-1] = candidate
            # Recalculate bar_count_since_prev for the updated point
            if len(history) >= 2:
                history[-1].bar_count_since_prev = (
                    history[-1].index - history[-2].index
                )
            return None  # Don't return a new point; history updated in-place

        # New alternating point — calculate bar count
        candidate.bar_count_since_prev = (
            candidate.index - prev.index if prev is not None else 0
        )
        history.append(candidate)
        return candidate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filter_and_alternate(self, raw: List[SwingPoint]) -> List[SwingPoint]:
        """Apply min_swing_pips filter and guarantee alternating high/low."""
        result: List[SwingPoint] = []

        for sp in raw:
            if not result:
                sp.bar_count_since_prev = 0
                result.append(sp)
                continue

            prev = result[-1]

            if prev.swing_type == sp.swing_type:
                # Same type — keep the more extreme, discard the other
                if sp.swing_type == "high" and sp.price > prev.price:
                    result[-1] = sp
                    result[-1].bar_count_since_prev = (
                        result[-1].index - result[-2].index
                        if len(result) >= 2
                        else 0
                    )
                elif sp.swing_type == "low" and sp.price < prev.price:
                    result[-1] = sp
                    result[-1].bar_count_since_prev = (
                        result[-1].index - result[-2].index
                        if len(result) >= 2
                        else 0
                    )
                # Otherwise discard sp — prev is more extreme
                continue

            # Different type — apply min_swing_pips filter
            if abs(sp.price - prev.price) < self._min_move:
                continue

            sp.bar_count_since_prev = sp.index - prev.index
            result.append(sp)

        return result
