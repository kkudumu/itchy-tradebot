"""Asian Range Breakout strategy for London session gold trading.

Marks the Asian session (21:00–06:00 UTC) high/low on gold, then trades the
breakout during the London session (06:00–10:00 UTC).

Session windows (UTC):
    Asian  : 21:00 – 06:00  (next day)
    London : 06:00 – 10:00

Signal rules:
    - Lock Asian range at 06:00 UTC.
    - Validate range: min_range_pips <= range_pips <= max_range_pips.
      (Gold: 1 pip = $0.10, so range_pips = (high - low) * 10)
    - Long  : close > asian_high during London window.
    - Short : close < asian_low  during London window.
    - SL    : opposite side of the range.
    - TP    : entry + risk * rr_ratio.
    - Max 1 signal per calendar day.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

from ..signal_engine import Signal


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: dict = {
    "min_range_pips": 10,    # minimum Asian range in pips
    "max_range_pips": 200,   # maximum Asian range in pips
    "rr_ratio": 2.0,         # reward-to-risk ratio for TP
    "pip_value": 0.1,        # price delta per "pip" (gold: $0.10 = 1 pip; MGC: 1 tick = $0.10)
    "instrument": "XAUUSD",
}


# ---------------------------------------------------------------------------
# AsianBreakoutStrategy
# ---------------------------------------------------------------------------

class AsianBreakoutStrategy:
    """Standalone Asian Range Breakout strategy.

    Unlike the abstract Strategy base, this class uses a simple bar-by-bar
    ``on_bar`` interface (no evaluator matrix required) so it can be used
    both in backtests and in the multi-strategy signal blender.

    Parameters
    ----------
    config:
        Optional config dict with keys: min_range_pips, max_range_pips,
        rr_ratio, instrument.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self._cfg: dict = {**_DEFAULT_CONFIG, **(config or {})}

        # Asian range accumulators (reset each session)
        self._asian_high: Optional[float] = None
        self._asian_low: Optional[float] = None
        self._range_locked: bool = False   # True once we pass 06:00 UTC

        # Day-tracking to trigger a daily reset
        self._session_date: Optional[dt.date] = None  # date of 21:00 candle
        self._signal_fired: bool = False               # one signal per day

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def asian_high(self) -> Optional[float]:
        """Highest high seen during the current Asian session."""
        return self._asian_high

    @property
    def asian_low(self) -> Optional[float]:
        """Lowest low seen during the current Asian session."""
        return self._asian_low

    @property
    def asian_range_pips(self) -> Optional[float]:
        """Range expressed in pips.

        A "pip" here is configurable via ``pip_value`` in the strategy
        config (default 0.1 price units, matching the historical gold
        convention of $0.10 per pip). For MGC futures where tick_size
        is also 0.10, setting pip_value=0.1 yields the same pip count
        as the forex path — no re-tuning needed at the boundary.
        """
        if self._asian_high is None or self._asian_low is None:
            return None
        pip_value = float(self._cfg.get("pip_value", 0.1))
        if pip_value <= 0:
            pip_value = 0.1
        return (self._asian_high - self._asian_low) / pip_value

    @property
    def range_valid(self) -> bool:
        """True if the locked Asian range is within configured pip limits."""
        rp = self.asian_range_pips
        if rp is None:
            return False
        return (
            rp > self._cfg["min_range_pips"]
            and rp < self._cfg["max_range_pips"]
        )

    # ------------------------------------------------------------------
    # Bar processing
    # ------------------------------------------------------------------

    def on_bar(
        self,
        timestamp: dt.datetime,
        *,
        high: float,
        low: float,
        close: float,
    ) -> Optional[Signal]:
        """Process one OHLCV bar and optionally return a Signal.

        Parameters
        ----------
        timestamp:
            Bar open time (timezone-aware UTC datetime).
        high, low, close:
            OHLCV values for this bar.

        Returns
        -------
        Signal or None
        """
        hour = timestamp.hour
        minute = timestamp.minute
        bar_date = timestamp.date()

        # ------------------------------------------------------------------
        # 1. Session reset: when we enter a new Asian session (21:00 UTC)
        #    on a different calendar day than the previous session.
        # ------------------------------------------------------------------
        if hour == 21 and (self._session_date is None or bar_date != self._session_date):
            self._asian_high = None
            self._asian_low = None
            self._range_locked = False
            self._signal_fired = False
            self._session_date = bar_date

        # ------------------------------------------------------------------
        # 2. Accumulate Asian high/low.
        #    Asian session spans 21:00 UTC to 05:59 UTC (next day).
        #    Lock the range at the first bar whose hour >= 6.
        # ------------------------------------------------------------------
        in_asian_window = (hour >= 21) or (hour < 6)

        if in_asian_window and not self._range_locked:
            if self._asian_high is None:
                self._asian_high = float(high)
                self._asian_low = float(low)
            else:
                self._asian_high = max(self._asian_high, float(high))
                self._asian_low = min(self._asian_low, float(low))

        # Lock range as soon as we step out of the Asian window
        if not in_asian_window and not self._range_locked:
            self._range_locked = True

        # ------------------------------------------------------------------
        # 3. London breakout signal check (06:00 – 09:59 UTC).
        # ------------------------------------------------------------------
        in_london_window = 6 <= hour < 10

        if (
            not in_london_window
            or self._signal_fired
            or not self._range_locked
            or not self.range_valid
        ):
            return None

        asian_high = self._asian_high
        asian_low = self._asian_low

        direction: Optional[str] = None
        if close > asian_high:
            direction = "long"
        elif close < asian_low:
            direction = "short"

        if direction is None:
            return None

        # Build signal levels
        entry = float(close)
        if direction == "long":
            stop_loss = asian_low
            risk = entry - stop_loss
            take_profit = entry + risk * self._cfg["rr_ratio"]
        else:
            stop_loss = asian_high
            risk = stop_loss - entry
            take_profit = entry - risk * self._cfg["rr_ratio"]

        self._signal_fired = True

        return Signal(
            timestamp=timestamp,
            instrument=self._cfg["instrument"],
            direction=direction,
            entry_price=entry,
            stop_loss=float(stop_loss),
            take_profit=float(take_profit),
            confluence_score=1,
            quality_tier="B",
            atr=float(asian_high - asian_low),
            reasoning={
                "strategy": "asian_breakout",
                "asian_high": asian_high,
                "asian_low": asian_low,
                "range_pips": self.asian_range_pips,
                "rr_ratio": self._cfg["rr_ratio"],
            },
        )
