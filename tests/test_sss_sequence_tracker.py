"""Tests for SequenceTracker — SSS state machine."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

import pytest

from src.strategy.strategies.sss import SequenceTracker, SwingPoint, SequenceEvent
from src.strategy.strategies.sss.sequence_tracker import (
    SCANNING, P2_DETECTED, TWO_CONFIRMED, THREE_ACTIVE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_TIME = datetime(2024, 1, 1, 9, 0, 0)
_bar = 0


def _swing(swing_type: str, price: float, bar_count: int = 10) -> SwingPoint:
    global _bar
    _bar += 1
    return SwingPoint(
        index=_bar,
        timestamp=BASE_TIME + timedelta(minutes=_bar),
        price=price,
        swing_type=swing_type,
        bar_count_since_prev=bar_count,
    )


def reset_bar_counter():
    global _bar
    _bar = 0


def make_tracker(**kwargs) -> SequenceTracker:
    return SequenceTracker(config=kwargs or None)


# ---------------------------------------------------------------------------
# Fixture resets global bar counter before each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_bars():
    reset_bar_counter()


# ---------------------------------------------------------------------------
# SCANNING state
# ---------------------------------------------------------------------------


def test_initial_state_is_scanning():
    tracker = make_tracker()
    assert tracker.state == SCANNING
    assert tracker.direction is None


def test_no_event_on_first_high():
    tracker = make_tracker()
    events = tracker.on_swing(_swing("high", 100.0))
    assert events == []
    assert tracker.state == SCANNING


def test_no_event_on_first_low():
    tracker = make_tracker()
    events = tracker.on_swing(_swing("low", 100.0))
    assert events == []
    assert tracker.state == SCANNING


# ---------------------------------------------------------------------------
# P2 detection
# ---------------------------------------------------------------------------


def test_bearish_p2_detected_on_higher_high():
    """Higher high after a high triggers bearish P2."""
    tracker = make_tracker()
    tracker.on_swing(_swing("high", 100.0))
    events = tracker.on_swing(_swing("high", 105.0))
    assert any(e.event_type == "p2_detected" for e in events)
    assert tracker.state == P2_DETECTED
    assert tracker.direction == "bearish"


def test_bullish_p2_detected_on_lower_low():
    """Lower low after a low triggers bullish P2."""
    tracker = make_tracker()
    tracker.on_swing(_swing("low", 100.0))
    events = tracker.on_swing(_swing("low", 95.0))
    assert any(e.event_type == "p2_detected" for e in events)
    assert tracker.state == P2_DETECTED
    assert tracker.direction == "bullish"


def test_no_p2_when_high_does_not_exceed_prev():
    """Equal or lower high should not trigger P2."""
    tracker = make_tracker()
    tracker.on_swing(_swing("high", 100.0))
    events = tracker.on_swing(_swing("high", 99.0))
    assert events == []


# ---------------------------------------------------------------------------
# Full efficient sequence: SCANNING -> P2 -> TWO_CONFIRMED -> THREE -> 4/5
# ---------------------------------------------------------------------------


def _run_full_bearish_sequence(tracker: SequenceTracker):
    """Feed swings to complete a full bearish sequence."""
    # #1 = high at 100
    tracker.on_swing(_swing("high", 100.0, bar_count=12))
    # #2 = higher high at 105 (P2 detected)
    tracker.on_swing(_swing("high", 105.0, bar_count=12))
    # Opposite swing confirms #2 and becomes #3 (low)
    tracker.on_swing(_swing("low", 98.0, bar_count=12))
    # #4/#5 = lower low than #3
    tracker.on_swing(_swing("low", 93.0, bar_count=12))


def test_full_efficient_bearish_sequence():
    tracker = make_tracker()
    _run_full_bearish_sequence(tracker)
    hist = tracker.history
    event_types = [e.event_type for e in hist]
    assert "p2_detected" in event_types
    assert "two_confirmed" in event_types
    assert "one_identified" in event_types
    assert "three_active" in event_types
    assert "four_five_resolved" in event_types
    assert "sequence_complete" in event_types


def test_full_efficient_sequence_directions():
    tracker = make_tracker()
    _run_full_bearish_sequence(tracker)
    for e in tracker.history:
        if e.event_type in ("p2_detected", "two_confirmed", "three_active",
                             "four_five_resolved", "sequence_complete"):
            assert e.direction == "bearish", f"Expected bearish, got {e.direction} for {e.event_type}"


def test_sequence_efficient_flag_true_for_complete_sequence():
    tracker = make_tracker()
    _run_full_bearish_sequence(tracker)
    complete = [e for e in tracker.history if e.event_type == "sequence_complete"]
    assert len(complete) >= 1
    assert complete[0].sequence_efficient is True


# ---------------------------------------------------------------------------
# F2 (failed 2) detection and direction reversal
# ---------------------------------------------------------------------------


def test_f2_detected_on_higher_high_sweep():
    """A higher high while in P2_DETECTED (bearish) should trigger F2."""
    tracker = make_tracker()
    tracker.on_swing(_swing("high", 100.0))
    tracker.on_swing(_swing("high", 105.0))  # P2 bearish
    assert tracker.state == P2_DETECTED
    assert tracker.direction == "bearish"

    events = tracker.on_swing(_swing("high", 110.0))  # F2 sweep
    f2_events = [e for e in events if e.event_type == "f2_detected"]
    assert len(f2_events) == 1
    assert f2_events[0].sequence_efficient is False


def test_f2_causes_direction_reversal():
    """After F2, direction should flip and state back to P2_DETECTED."""
    tracker = make_tracker()
    tracker.on_swing(_swing("high", 100.0))
    tracker.on_swing(_swing("high", 105.0))  # bearish P2
    tracker.on_swing(_swing("high", 110.0))  # F2 sweep
    # After F2, the new direction is bullish (opposite of bearish)
    # because the f2 sweep creates new P2 in opposite direction
    assert tracker.state == P2_DETECTED
    assert tracker.direction == "bullish"


# ---------------------------------------------------------------------------
# Inefficient sequence flag
# ---------------------------------------------------------------------------


def test_inefficient_sequence_flag():
    """Non-resolving intermediate swings should mark sequence as inefficient."""
    tracker = make_tracker()
    # Start bearish sequence
    tracker.on_swing(_swing("high", 100.0, bar_count=12))
    tracker.on_swing(_swing("high", 105.0, bar_count=12))  # P2 bearish
    # Confirm with a low (#3)
    tracker.on_swing(_swing("low", 98.0, bar_count=12))    # THREE_ACTIVE
    # Add an intermediate high (not resolving)
    tracker.on_swing(_swing("high", 99.0, bar_count=12))   # doesn't extend below #3
    # Now resolve with a lower low
    tracker.on_swing(_swing("low", 90.0, bar_count=12))    # four_five_resolved

    complete = [e for e in tracker.history if e.event_type == "sequence_complete"]
    assert len(complete) >= 1
    assert complete[0].sequence_efficient is False


# ---------------------------------------------------------------------------
# Timeout reset
# ---------------------------------------------------------------------------


def test_timeout_resets_to_scanning():
    """If stuck in P2_DETECTED for > max_bars_in_state swings, emit timeout.

    To stay in P2_DETECTED, we feed same-type swings that are less extreme
    than #2 (don't sweep it) — these fall through the pass branches and keep
    the tracker stuck in P2_DETECTED, incrementing swings_in_state each time.
    """
    tracker = make_tracker(max_bars_in_state=3)
    # Establish prev_high at 100, then trigger bearish P2 at 105
    tracker.on_swing(_swing("high", 100.0))
    tracker.on_swing(_swing("high", 105.0))  # P2 bearish; #2.price=105
    assert tracker.state == P2_DETECTED

    # Feed same-type (high) swings that are BELOW 105 (less extreme than #2)
    # These don't sweep #2, so the tracker stays in P2_DETECTED and increments
    # swings_in_state on each call.  After > max_bars_in_state=3 such calls, timeout.
    for price in [103.0, 102.0, 101.0, 100.5]:
        tracker.on_swing(_swing("high", price))

    timeout_evts = [e for e in tracker.history if e.event_type == "sequence_timeout"]
    assert len(timeout_evts) >= 1


# ---------------------------------------------------------------------------
# Layer classification
# ---------------------------------------------------------------------------


def test_layer_ss_for_large_bar_count():
    """bar_count >= ss_candle_min (10) should classify as SS."""
    tracker = make_tracker(ss_candle_min=10)
    tracker.on_swing(_swing("high", 100.0, bar_count=12))
    events = tracker.on_swing(_swing("high", 105.0, bar_count=12))
    p2 = [e for e in events if e.event_type == "p2_detected"]
    assert len(p2) == 1
    assert p2[0].layer == "SS"


def test_layer_iss_for_medium_bar_count():
    """bar_count in [4..8] should classify as ISS."""
    tracker = make_tracker(iss_candle_min=4, iss_candle_max=8)
    tracker.on_swing(_swing("high", 100.0, bar_count=6))
    events = tracker.on_swing(_swing("high", 105.0, bar_count=6))
    p2 = [e for e in events if e.event_type == "p2_detected"]
    assert len(p2) == 1
    assert p2[0].layer == "ISS"


def test_layer_cbc_for_bar_count_3():
    """bar_count == 3 should classify as CBC."""
    tracker = make_tracker()
    tracker.on_swing(_swing("high", 100.0, bar_count=3))
    events = tracker.on_swing(_swing("high", 105.0, bar_count=3))
    p2 = [e for e in events if e.event_type == "p2_detected"]
    assert len(p2) == 1
    assert p2[0].layer == "CBC"


# ---------------------------------------------------------------------------
# History accumulation
# ---------------------------------------------------------------------------


def test_history_accumulates_events():
    tracker = make_tracker()
    assert tracker.history == []
    tracker.on_swing(_swing("high", 100.0))
    tracker.on_swing(_swing("high", 105.0))
    assert len(tracker.history) >= 1


def test_reset_clears_state():
    tracker = make_tracker()
    tracker.on_swing(_swing("high", 100.0))
    tracker.on_swing(_swing("high", 105.0))
    assert tracker.state == P2_DETECTED
    tracker.reset()
    assert tracker.state == SCANNING
    assert tracker.direction is None


# ---------------------------------------------------------------------------
# Fractal nesting: #4/#5 becomes P2 on higher layer
# ---------------------------------------------------------------------------


def test_four_five_becomes_new_p2():
    """After four_five_resolved, state transitions to P2_DETECTED (higher layer)."""
    tracker = make_tracker()
    _run_full_bearish_sequence(tracker)
    # After sequence completes, the #4/#5 should seed a new P2
    assert tracker.state == P2_DETECTED
    assert tracker.direction == "bearish"
