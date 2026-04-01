"""CBCDetector — 3-candle entry pattern detection for SSS strategy.

CBC (Candle By Candle) identifies turning points using exactly 3 bars on 1M:
  - Candle A establishes an extreme (high or low)
  - Candle B sweeps past that extreme (creates breathing room)
  - Candle C confirms direction: runs back past A (successful) or fails (failed)

Jay's stat: ~33% standalone success — always combine with higher-TF context.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Optional

log = logging.getLogger(__name__)


class CBCType:
    """CBC pattern type string constants."""

    CBC_SUCCESSFUL_TWO = "cbc_successful_two"
    CBC_FAILED_TWO = "cbc_failed_two"
    CBC_CONTINUATION = "cbc_continuation"


@dataclass
class CBCSignal:
    """Emitted when a 3-candle CBC pattern completes."""

    cbc_type: str               # CBCType value
    direction: str              # 'bullish' or 'bearish'
    entry_price: float          # C's close
    invalidation_price: float   # B's sweep extreme
    bar_index: int              # Index of candle C
    timestamp: datetime         # Timestamp of candle C
    candle_a: dict              # {open, high, low, close}
    candle_b: dict
    candle_c: dict


@dataclass
class _Bar:
    index: int
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float

    def to_dict(self) -> dict:
        return {"open": self.open, "high": self.high, "low": self.low, "close": self.close}


class CBCDetector:
    """Detects 3-candle CBC turning patterns on a 1M bar stream.

    Feed bars via ``on_bar()``.  Returns a ``CBCSignal`` when a complete
    A-B-C pattern is recognised and the context gate is satisfied.

    Args:
        require_context: When ``True`` (default), only emits when
            ``context_direction`` matches the detected CBC direction.
    """

    def __init__(self, require_context: bool = True) -> None:
        self._require_context = require_context
        self._buffer: Deque[_Bar] = deque(maxlen=3)

    def on_bar(
        self,
        bar_index: int,
        timestamp: datetime,
        open: float,
        high: float,
        low: float,
        close: float,
        context_direction: Optional[str] = None,
    ) -> Optional[CBCSignal]:
        """Process one new bar.

        Args:
            bar_index: Sequential bar index.
            timestamp: Bar timestamp.
            open/high/low/close: OHLC prices.
            context_direction: ``'bullish'`` or ``'bearish'`` from the larger
                sequence tracker.  Required when ``require_context=True``.

        Returns:
            ``CBCSignal`` if a pattern fires and context gate passes, else ``None``.
        """
        self._buffer.append(_Bar(bar_index, timestamp, open, high, low, close))
        if len(self._buffer) < 3:
            return None
        if self._require_context and context_direction is None:
            return None
        a, b, c = self._buffer[0], self._buffer[1], self._buffer[2]
        return self._classify(a, b, c, context_direction)

    def reset(self) -> None:
        """Clear the rolling bar buffer."""
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify(
        self,
        a: _Bar,
        b: _Bar,
        c: _Bar,
        context_direction: Optional[str],
    ) -> Optional[CBCSignal]:
        b_sweeps_low = b.low < a.low
        b_sweeps_high = b.high > a.high

        # Bullish CBC_SUCCESSFUL_TWO: B sweeps A's low, C closes above A's high
        if b_sweeps_low and c.close > a.high:
            return self._gate(
                CBCSignal(CBCType.CBC_SUCCESSFUL_TWO, "bullish",
                          c.close, b.low, c.index, c.timestamp,
                          a.to_dict(), b.to_dict(), c.to_dict()),
                context_direction,
            )

        # Bearish CBC_SUCCESSFUL_TWO: B sweeps A's high, C closes below A's low
        if b_sweeps_high and c.close < a.low:
            return self._gate(
                CBCSignal(CBCType.CBC_SUCCESSFUL_TWO, "bearish",
                          c.close, b.high, c.index, c.timestamp,
                          a.to_dict(), b.to_dict(), c.to_dict()),
                context_direction,
            )

        # Bullish CBC_FAILED_TWO: B swept A's low but C reverses below B's low
        if b_sweeps_low and c.close < b.low:
            return self._gate(
                CBCSignal(CBCType.CBC_FAILED_TWO, "bullish",
                          c.close, b.low, c.index, c.timestamp,
                          a.to_dict(), b.to_dict(), c.to_dict()),
                context_direction,
            )

        # Bearish CBC_FAILED_TWO: B swept A's high but C reverses above B's high
        if b_sweeps_high and c.close > b.high:
            return self._gate(
                CBCSignal(CBCType.CBC_FAILED_TWO, "bearish",
                          c.close, b.high, c.index, c.timestamp,
                          a.to_dict(), b.to_dict(), c.to_dict()),
                context_direction,
            )

        # CBC_CONTINUATION: all 3 closes trend together, no sweep/reversal
        if b.close > a.close and c.close > b.close:
            return self._gate(
                CBCSignal(CBCType.CBC_CONTINUATION, "bullish",
                          c.close, b.low, c.index, c.timestamp,
                          a.to_dict(), b.to_dict(), c.to_dict()),
                context_direction,
            )

        if b.close < a.close and c.close < b.close:
            return self._gate(
                CBCSignal(CBCType.CBC_CONTINUATION, "bearish",
                          c.close, b.high, c.index, c.timestamp,
                          a.to_dict(), b.to_dict(), c.to_dict()),
                context_direction,
            )

        log.debug("CBC no pattern at bar %d", c.index)
        return None

    def _gate(
        self,
        signal: CBCSignal,
        context_direction: Optional[str],
    ) -> Optional[CBCSignal]:
        """Return signal only when context gate is satisfied."""
        if not self._require_context:
            return signal
        if context_direction == signal.direction:
            return signal
        log.debug("CBC gate blocked: signal=%s context=%s", signal.direction, context_direction)
        return None
