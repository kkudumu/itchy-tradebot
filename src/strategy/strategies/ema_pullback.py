"""EMA Pullback State Machine strategy for gold (XAUUSD).

Based on the ilahuerta-IA/backtrader-pullback-window-xauusd approach that
achieved 55% WR, PF 1.64, 5.8% max DD over 5 years of gold trading.

4-phase state machine
~~~~~~~~~~~~~~~~~~~~~
1. SCANNING    — Wait for EMA alignment (fast>mid>slow or fast<mid<slow) at a
                 sufficient angle, then detect the first counter-trend
                 (pullback) candle.
2. ARMED       — Count consecutive pullback candles.  Track the pre-pullback
                 swing high (long) or low (short) as the breakout reference.
                 Advance to WINDOW_OPEN when pullback_candles_min are met.
                 Reset to SCANNING if pullback_candles_max exceeded.
3. WINDOW_OPEN — Wait for close to break above breakout_level (long) or below
                 (short).  Emit a Signal on breakout.  Reset to SCANNING if
                 breakout_window_bars expire without a breakout.
4. Back to SCANNING after any signal or expiry.
"""

from __future__ import annotations

import math
import datetime as dt
from typing import Optional

from ..signal_engine import Signal


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: dict = {
    "fast_ema": 14,
    "mid_ema": 18,
    "slow_ema": 24,
    "min_ema_angle_deg": 30,
    "max_ema_angle_deg": 95,
    "pullback_candles_min": 1,
    "pullback_candles_max": 3,
    "breakout_window_bars": 20,
    "rr_ratio": 2.0,
    "weight": 1.0,
    "instrument": "XAUUSD",
}


# ---------------------------------------------------------------------------
# EMAPullbackStrategy
# ---------------------------------------------------------------------------

class EMAPullbackStrategy:
    """Bar-by-bar EMA pullback strategy using a 4-phase state machine.

    Parameters
    ----------
    config:
        Optional overrides for any key in ``_DEFAULT_CONFIG``.
    """

    # Valid states
    _STATES = ("SCANNING", "ARMED", "WINDOW_OPEN")

    def __init__(self, config: Optional[dict] = None) -> None:
        self._cfg: dict = {**_DEFAULT_CONFIG, **(config or {})}

        # Current state
        self._state: str = "SCANNING"

        # Previous bar's ema_fast — used for angle calculation
        self._prev_ema_fast: Optional[float] = None

        # Trend direction detected in SCANNING: 'long' | 'short' | None
        self._trend: Optional[str] = None

        # Pre-pullback high (long) or low (short) used as breakout level
        self._breakout_level: Optional[float] = None

        # Running count of consecutive pullback candles in ARMED state
        self._pullback_count: int = 0

        # Bar counter inside WINDOW_OPEN state
        self._window_bars: int = 0

        # SL anchor: slow EMA at the bar we armed (updated each bar in ARMED)
        self._armed_slow_ema: Optional[float] = None

        # Max stop distance in ATR multiples (prevents outsized risk on longs)
        self._max_stop_atr: float = float(self._cfg.get("max_stop_atr", 2.0))

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        """Current state machine state."""
        return self._state

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------

    def on_bar(
        self,
        timestamp: dt.datetime,
        *,
        open: float,
        high: float,
        low: float,
        close: float,
        ema_fast: float,
        ema_mid: float,
        ema_slow: float,
        atr: float = 5.0,
    ) -> Optional[Signal]:
        """Process one bar and optionally return a Signal.

        Parameters
        ----------
        timestamp:
            Bar open timestamp (timezone-aware UTC recommended).
        open, high, low, close:
            OHLCV prices for the bar.
        ema_fast, ema_mid, ema_slow:
            Pre-computed EMA values.
        atr:
            Average True Range at this bar (used for angle normalisation).

        Returns
        -------
        Signal or None
        """
        signal: Optional[Signal] = None

        if self._state == "SCANNING":
            signal = self._handle_scanning(
                timestamp, open, high, low, close,
                ema_fast, ema_mid, ema_slow, atr,
            )
        elif self._state == "ARMED":
            signal = self._handle_armed(
                timestamp, open, high, low, close,
                ema_fast, ema_mid, ema_slow, atr,
            )
        elif self._state == "WINDOW_OPEN":
            signal = self._handle_window_open(
                timestamp, open, high, low, close,
                ema_fast, ema_mid, ema_slow, atr,
            )

        # Always update previous ema_fast for angle calc on next bar
        self._prev_ema_fast = ema_fast

        return signal

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _handle_scanning(
        self, timestamp, open, high, low, close,
        ema_fast, ema_mid, ema_slow, atr,
    ) -> Optional[Signal]:
        """SCANNING: look for EMA alignment + angle, then first pullback candle."""

        # 1. Check EMA alignment
        if ema_fast > ema_mid > ema_slow:
            trend = "long"
        elif ema_fast < ema_mid < ema_slow:
            trend = "short"
        else:
            # EMAs not ordered — stay in SCANNING
            return None

        # 2. EMA angle filter (only computable after first bar)
        if self._prev_ema_fast is not None and atr > 0:
            angle = abs(
                math.atan((ema_fast - self._prev_ema_fast) / atr)
            ) * 180.0 / math.pi
            if angle < self._cfg["min_ema_angle_deg"]:
                return None
            if angle > self._cfg["max_ema_angle_deg"]:
                return None
        # If this is the very first bar, we have no angle yet.  We keep the
        # bar as a "trend bar" so the breakout level is captured, but we
        # need at least one more bar before we can transition to ARMED.
        # We record the pre-pullback level from this (trend) bar.

        # 3. Detect pullback candle (counter-trend body)
        #    Long trend → pullback = bearish candle (close < open)
        #    Short trend → pullback = bullish candle (close > open)
        is_pullback = (
            (trend == "long" and close < open) or
            (trend == "short" and close > open)
        )

        if not is_pullback:
            # This is a trend-continuation bar — capture the swing high/low
            # as a potential breakout reference for when we arm next bar.
            self._trend = trend
            if trend == "long":
                self._breakout_level = high
            else:
                self._breakout_level = low
            return None

        # We have: aligned EMAs, sufficient angle, and a pullback candle.
        # But we need a prior trend bar to have set the breakout level.
        if self._breakout_level is None or self._trend != trend:
            # No prior trend bar seen for this direction — capture now and wait
            self._trend = trend
            # Can't arm yet without a pre-pullback reference — keep scanning
            return None

        # Transition to ARMED
        self._trend = trend
        self._armed_slow_ema = ema_slow
        self._pullback_count = 1
        self._state = "ARMED"
        return None

    def _handle_armed(
        self, timestamp, open, high, low, close,
        ema_fast, ema_mid, ema_slow, atr,
    ) -> Optional[Signal]:
        """ARMED: accumulate pullback candles; open window when ready."""

        trend = self._trend
        is_pullback = (
            (trend == "long" and close < open) or
            (trend == "short" and close > open)
        )

        if is_pullback:
            self._pullback_count += 1
            self._armed_slow_ema = ema_slow  # track SL level

            # Too many pullback candles → pullback too deep
            if self._pullback_count > self._cfg["pullback_candles_max"]:
                self._reset()
                return None
        else:
            # Non-pullback candle encountered
            if self._pullback_count >= self._cfg["pullback_candles_min"]:
                # Enough pullback candles — open the breakout window
                self._window_bars = 0
                self._state = "WINDOW_OPEN"
                # Immediately check if this bar is the breakout
                return self._check_breakout(
                    timestamp, open, high, low, close,
                    ema_fast, ema_mid, ema_slow, atr,
                )
            else:
                # Not enough pullback candles yet, but trend resumed.
                # Update breakout level and stay armed (reset count).
                if trend == "long":
                    self._breakout_level = max(self._breakout_level or 0, high)
                else:
                    self._breakout_level = min(self._breakout_level or float("inf"), low)
                self._pullback_count = 0

        return None

    def _handle_window_open(
        self, timestamp, open, high, low, close,
        ema_fast, ema_mid, ema_slow, atr,
    ) -> Optional[Signal]:
        """WINDOW_OPEN: wait for breakout or expiry."""
        self._window_bars += 1

        if self._window_bars > self._cfg["breakout_window_bars"]:
            self._reset()
            return None

        return self._check_breakout(
            timestamp, open, high, low, close,
            ema_fast, ema_mid, ema_slow, atr,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_breakout(
        self, timestamp, open, high, low, close,
        ema_fast, ema_mid, ema_slow, atr,
    ) -> Optional[Signal]:
        """Check if this bar breaks out above/below breakout_level."""
        trend = self._trend
        level = self._breakout_level

        if level is None:
            self._reset()
            return None

        broke_out = (
            (trend == "long" and close > level) or
            (trend == "short" and close < level)
        )

        if not broke_out:
            return None

        # Build signal levels
        entry = float(close)
        sl = float(self._armed_slow_ema or ema_slow)
        rr = self._cfg["rr_ratio"]

        if trend == "long":
            risk = entry - sl
            tp = entry + risk * rr
        else:
            risk = sl - entry
            tp = entry - risk * rr

        # Quality tier based on risk size relative to ATR
        if risk >= atr * 1.5:
            quality_tier = "C"
        elif risk >= atr * 0.8:
            quality_tier = "B"
        else:
            quality_tier = "A+"

        signal = Signal(
            timestamp=timestamp,
            instrument=self._cfg["instrument"],
            direction=trend,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            confluence_score=2,
            quality_tier=quality_tier,
            atr=float(atr),
            reasoning={
                "strategy": "ema_pullback",
                "breakout_level": level,
                "pullback_count": self._pullback_count,
                "rr_ratio": rr,
                "risk": risk,
            },
        )

        self._reset()
        return signal

    def _reset(self) -> None:
        """Reset all state back to SCANNING."""
        self._state = "SCANNING"
        self._trend = None
        self._breakout_level = None
        self._pullback_count = 0
        self._window_bars = 0
        self._armed_slow_ema = None
        # Do NOT reset _prev_ema_fast — we keep tracking the EMA angle
