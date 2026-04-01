"""SequenceTracker — core state machine for Jay's SSS sequence model.

Tracks the progression SCANNING -> P2_DETECTED -> TWO_CONFIRMED ->
THREE_ACTIVE -> FOUR_FIVE_RESOLVED -> SCANNING, emitting SequenceEvent
objects at each transition.

Key concepts:
    - #2 is always found FIRST; it sweeps the previous swing of the SAME type.
    - #1 is identified retroactively when #2 sweeps it.
    - F2 (Failed 2): same-type swing sweeps #2 before it's confirmed.
    - Bearish sequence: #1=high, #2=higher_high, #3=low, #4/#5=lower_low.
    - Bullish sequence: #1=low, #2=lower_low, #3=high, #4/#5=higher_high.
    - Inefficient sequences (missing numbers) create SS-level POIs.
    - SS layer : bar_count_since_prev >= ss_candle_min (default 10).
    - ISS layer: iss_candle_min <= bar_count_since_prev <= iss_candle_max.
    - CBC layer: bar_count_since_prev == 3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .breathing_room import SwingPoint

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State constants
# ---------------------------------------------------------------------------

SCANNING = "scanning"
P2_DETECTED = "p2_detected"
TWO_CONFIRMED = "two_confirmed"
THREE_ACTIVE = "three_active"
FOUR_FIVE_RESOLVED = "four_five_resolved"

# ---------------------------------------------------------------------------
# SequenceEvent dataclass
# ---------------------------------------------------------------------------


@dataclass
class SequenceEvent:
    """An event emitted by SequenceTracker when a state transition occurs."""

    event_type: str
    """One of: p2_detected, two_confirmed, f2_detected, one_identified,
    three_active, four_five_resolved, sequence_complete, sequence_timeout."""

    direction: str
    """'bullish' or 'bearish'."""

    level_price: float
    """Price level associated with this event (the relevant swing price)."""

    candle_count: int
    """Number of candles in this swing leg."""

    layer: str
    """'SS', 'ISS', or 'CBC' — derived from candle_count."""

    timestamp: datetime
    """When the event occurred (timestamp of the triggering swing)."""

    sequence_efficient: bool = True
    """False when the sequence has missing numbers, creating a SS-level POI."""


# ---------------------------------------------------------------------------
# Internal sequence context
# ---------------------------------------------------------------------------


@dataclass
class _SequenceContext:
    """Mutable state for one in-progress sequence."""

    state: str = SCANNING
    direction: Optional[str] = None

    # Labelled sequence swing points
    swing_one: Optional[SwingPoint] = None
    swing_two: Optional[SwingPoint] = None
    swing_three: Optional[SwingPoint] = None

    # Most recent confirmed swing of each type (used in SCANNING to spot #2)
    prev_high: Optional[SwingPoint] = None
    prev_low: Optional[SwingPoint] = None

    # How many swings processed while in the current non-scanning state
    swings_in_state: int = 0

    # Whether the sequence is efficient (all numbers present)
    efficient: bool = True


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: dict = {
    "ss_candle_min": 10,
    "iss_candle_min": 4,
    "iss_candle_max": 8,
    "max_bars_in_state": 50,
}


# ---------------------------------------------------------------------------
# SequenceTracker
# ---------------------------------------------------------------------------


class SequenceTracker:
    """Bar-by-bar state machine that tracks Jay's SSS swing sequence.

    Parameters
    ----------
    config:
        Optional dict overriding default settings:
        ss_candle_min, iss_candle_min, iss_candle_max, max_bars_in_state.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self._cfg: dict = {**_DEFAULT_CONFIG, **(config or {})}
        self._ss_min: int = int(self._cfg["ss_candle_min"])
        self._iss_min: int = int(self._cfg["iss_candle_min"])
        self._iss_max: int = int(self._cfg["iss_candle_max"])
        self._max_bars: int = int(self._cfg["max_bars_in_state"])

        self._ctx = _SequenceContext()
        self._history: list[SequenceEvent] = []

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        """Current state of the tracker."""
        return self._ctx.state

    @property
    def direction(self) -> Optional[str]:
        """Current sequence direction, or None if scanning."""
        return self._ctx.direction

    @property
    def active_sequences(self) -> list[dict]:
        """Summary of the currently tracked sequence."""
        if self._ctx.state == SCANNING:
            return []
        return [
            {
                "state": self._ctx.state,
                "direction": self._ctx.direction,
                "swing_one": self._ctx.swing_one,
                "swing_two": self._ctx.swing_two,
                "swing_three": self._ctx.swing_three,
                "efficient": self._ctx.efficient,
            }
        ]

    @property
    def history(self) -> list[SequenceEvent]:
        """All sequence events emitted so far."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def on_swing(self, swing: SwingPoint) -> list[SequenceEvent]:
        """Process a new swing point and return any emitted SequenceEvents.

        Parameters
        ----------
        swing:
            The newly confirmed swing point from BreathingRoomDetector.

        Returns
        -------
        list[SequenceEvent]
            Zero or more events emitted by this transition.
        """
        events: list[SequenceEvent] = []

        # Timeout guard — reset if stuck too long in any active state
        if self._ctx.state != SCANNING:
            self._ctx.swings_in_state += 1
            if self._ctx.swings_in_state > self._max_bars:
                timeout_evt = self._make_event(
                    "sequence_timeout",
                    self._ctx.direction or "bearish",
                    swing.price,
                    swing.bar_count_since_prev,
                    swing.timestamp,
                    efficient=False,
                )
                events.append(timeout_evt)
                self._record(events)
                self._reset_ctx()
                # Fall through to SCANNING with cleared context

        state = self._ctx.state
        if state == SCANNING:
            events.extend(self._handle_scanning(swing))
        elif state == P2_DETECTED:
            events.extend(self._handle_p2_detected(swing))
        elif state == TWO_CONFIRMED:
            events.extend(self._handle_two_confirmed(swing))
        elif state == THREE_ACTIVE:
            events.extend(self._handle_three_active(swing))
        elif state == FOUR_FIVE_RESOLVED:
            self._reset_ctx()
            events.extend(self._handle_scanning(swing))

        self._record(events)
        return events

    def reset(self) -> None:
        """Reset to SCANNING state, clearing all sequence context."""
        self._reset_ctx()
        log.debug("SequenceTracker manually reset to SCANNING")

    # ------------------------------------------------------------------
    # State handlers (private)
    # ------------------------------------------------------------------

    def _handle_scanning(self, swing: SwingPoint) -> list[SequenceEvent]:
        """SCANNING: maintain last high/low; detect potential #2.

        Bearish #2: a swing HIGH that exceeds the previous swing HIGH.
        Bullish #2: a swing LOW that undercuts the previous swing LOW.
        """
        if swing.swing_type == "high":
            prev_high = self._ctx.prev_high
            self._ctx.prev_high = swing  # always advance to latest high

            if prev_high is not None and swing.price > prev_high.price:
                return self._start_sequence("bearish", one=prev_high, two=swing)

        else:  # "low"
            prev_low = self._ctx.prev_low
            self._ctx.prev_low = swing  # always advance to latest low

            if prev_low is not None and swing.price < prev_low.price:
                return self._start_sequence("bullish", one=prev_low, two=swing)

        return []

    def _handle_p2_detected(self, swing: SwingPoint) -> list[SequenceEvent]:
        """P2_DETECTED: confirm #2 or detect F2.

        F2 occurs when a swing of the SAME type as #2 sweeps #2 before
        the sequence is confirmed.  If a swing of opposite type arrives
        without sweeping #2, #2 is confirmed and that swing becomes #3.
        """
        two = self._ctx.swing_two
        direction = self._ctx.direction
        assert two is not None and direction is not None  # invariant

        same_type_as_two = swing.swing_type == two.swing_type

        if same_type_as_two and self._sweeps_same_type(swing, two):
            # F2: sweep of #2 — direction reversal
            return self._handle_f2(swing, two, direction)

        if not same_type_as_two:
            # Opposite type — #2 is confirmed; this swing is #3
            return self._confirm_two_and_set_three(swing)

        # Same type but doesn't sweep #2: update the candidate #2 only if
        # this swing is more extreme (deeper in the direction)
        if self._sweeps_same_type(two, swing):
            # Old #2 is more extreme; keep it, ignore this swing
            pass
        else:
            # New swing is equally/less extreme; treat as noise, keep #2
            pass
        return []

    def _handle_two_confirmed(self, swing: SwingPoint) -> list[SequenceEvent]:
        """TWO_CONFIRMED: transient state; immediately assign #3."""
        direction = self._ctx.direction
        assert direction is not None

        self._ctx.swing_three = swing
        self._ctx.state = THREE_ACTIVE
        self._ctx.swings_in_state = 0
        return [
            self._make_event(
                "three_active",
                direction,
                swing.price,
                swing.bar_count_since_prev,
                swing.timestamp,
            )
        ]

    def _handle_three_active(self, swing: SwingPoint) -> list[SequenceEvent]:
        """THREE_ACTIVE: looking for #4/#5 — a swing that extends past #3.

        In a bearish sequence #3 is a LOW; #4/#5 is another LOW that goes
        below #3.  In a bullish sequence #3 is a HIGH; #4/#5 is another HIGH
        that goes above #3.
        """
        three = self._ctx.swing_three
        direction = self._ctx.direction
        assert three is not None and direction is not None

        # #4/#5 has the same swing type as #3 (both lows in bearish, both highs in bullish)
        # and must extend past #3 in the sequence direction.
        same_type_as_three = swing.swing_type == three.swing_type
        if same_type_as_three and self._sweeps_same_type(swing, three):
            return self._resolve_four_five(swing)

        # Any non-resolving swing — sequence may be inefficient
        self._ctx.efficient = False
        return []

    # ------------------------------------------------------------------
    # Transition helpers (private)
    # ------------------------------------------------------------------

    def _start_sequence(
        self, direction: str, one: SwingPoint, two: SwingPoint
    ) -> list[SequenceEvent]:
        """Initialize a new sequence with direction, #1, and tentative #2."""
        self._ctx.direction = direction
        self._ctx.swing_one = one
        self._ctx.swing_two = two
        self._ctx.state = P2_DETECTED
        self._ctx.swings_in_state = 0
        self._ctx.efficient = True
        log.debug("P2 %s detected: price=%.5f", direction, two.price)
        return [
            self._make_event(
                "p2_detected",
                direction,
                two.price,
                two.bar_count_since_prev,
                two.timestamp,
            )
        ]

    def _handle_f2(
        self,
        swing: SwingPoint,
        old_two: SwingPoint,
        old_direction: str,
    ) -> list[SequenceEvent]:
        """Emit F2 event and set up a new P2 in the opposite direction."""
        log.debug("F2 at price=%.5f (old dir=%s)", swing.price, old_direction)
        f2_evt = self._make_event(
            "f2_detected",
            old_direction,
            swing.price,
            swing.bar_count_since_prev,
            swing.timestamp,
            efficient=False,
        )

        # The F2 swing sweeps #2 => old #2 becomes the new #1,
        # and the current swing is the new tentative #2.
        new_direction = "bullish" if old_direction == "bearish" else "bearish"
        self._ctx.direction = new_direction
        self._ctx.swing_one = old_two
        self._ctx.swing_two = swing
        self._ctx.state = P2_DETECTED
        self._ctx.swings_in_state = 0
        self._ctx.efficient = True

        new_p2_evt = self._make_event(
            "p2_detected",
            new_direction,
            swing.price,
            swing.bar_count_since_prev,
            swing.timestamp,
        )
        return [f2_evt, new_p2_evt]

    def _confirm_two_and_set_three(self, swing: SwingPoint) -> list[SequenceEvent]:
        """Confirm #2, identify #1, and assign the current swing as #3."""
        two = self._ctx.swing_two
        one = self._ctx.swing_one
        direction = self._ctx.direction
        assert two is not None and one is not None and direction is not None

        evt_confirmed = self._make_event(
            "two_confirmed",
            direction,
            two.price,
            two.bar_count_since_prev,
            two.timestamp,
        )
        evt_one = self._make_event(
            "one_identified",
            direction,
            one.price,
            one.bar_count_since_prev,
            one.timestamp,
        )
        self._ctx.swing_three = swing
        self._ctx.state = THREE_ACTIVE
        self._ctx.swings_in_state = 0

        evt_three = self._make_event(
            "three_active",
            direction,
            swing.price,
            swing.bar_count_since_prev,
            swing.timestamp,
        )
        return [evt_confirmed, evt_one, evt_three]

    def _resolve_four_five(self, swing: SwingPoint) -> list[SequenceEvent]:
        """Resolve the sequence at #4/#5 and begin higher-layer detection."""
        direction = self._ctx.direction
        efficient = self._ctx.efficient
        old_two = self._ctx.swing_two
        assert direction is not None

        evt_45 = self._make_event(
            "four_five_resolved",
            direction,
            swing.price,
            swing.bar_count_since_prev,
            swing.timestamp,
            efficient=efficient,
        )
        evt_complete = self._make_event(
            "sequence_complete",
            direction,
            swing.price,
            swing.bar_count_since_prev,
            swing.timestamp,
            efficient=efficient,
        )
        log.debug("Sequence complete: dir=%s efficient=%s", direction, efficient)

        # Every #4/#5 is also a #2 on a higher timeframe (fractal nesting).
        # Transition to P2_DETECTED with old #2 as potential #1.
        self._reset_ctx()
        self._ctx.direction = direction
        self._ctx.swing_one = old_two
        self._ctx.swing_two = swing
        self._ctx.state = P2_DETECTED
        self._ctx.swings_in_state = 0

        higher_p2_evt = self._make_event(
            "p2_detected",
            direction,
            swing.price,
            swing.bar_count_since_prev,
            swing.timestamp,
        )
        return [evt_45, evt_complete, higher_p2_evt]

    # ------------------------------------------------------------------
    # Utility helpers (private)
    # ------------------------------------------------------------------

    def _sweeps_same_type(self, swing: SwingPoint, target: SwingPoint) -> bool:
        """Return True if ``swing`` sweeps ``target`` (assumes same swing_type)."""
        if swing.swing_type == "high":
            return swing.price > target.price
        return swing.price < target.price

    def _classify_layer(self, bar_count: int) -> str:
        """Classify a swing into SS, ISS, or CBC based on candle count."""
        if bar_count == 3:
            return "CBC"
        if self._iss_min <= bar_count <= self._iss_max:
            return "ISS"
        if bar_count >= self._ss_min:
            return "SS"
        return "ISS"  # default fallback for unusual counts

    def _make_event(
        self,
        event_type: str,
        direction: str,
        level_price: float,
        candle_count: int,
        timestamp: datetime,
        efficient: bool = True,
    ) -> SequenceEvent:
        """Construct a SequenceEvent with layer derived from candle_count."""
        return SequenceEvent(
            event_type=event_type,
            direction=direction,
            level_price=level_price,
            candle_count=candle_count,
            layer=self._classify_layer(candle_count),
            timestamp=timestamp,
            sequence_efficient=efficient,
        )

    def _record(self, events: list[SequenceEvent]) -> None:
        """Append events to the history log."""
        self._history.extend(events)

    def _reset_ctx(self) -> None:
        """Reset the sequence context to SCANNING state."""
        self._ctx = _SequenceContext()
