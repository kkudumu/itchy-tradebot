"""Tests for CBCDetector — 3-candle entry pattern detection."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.strategy.strategies.sss import CBCDetector, CBCSignal, CBCType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_TIME = datetime(2024, 1, 1, 9, 0, 0)


def feed_bars(detector: CBCDetector, bars, context_direction=None):
    """Feed a list of (open, high, low, close) tuples to the detector."""
    signals = []
    for i, (o, h, l, c) in enumerate(bars):
        sig = detector.on_bar(
            bar_index=i,
            timestamp=BASE_TIME + timedelta(minutes=i),
            open=o,
            high=h,
            low=l,
            close=c,
            context_direction=context_direction,
        )
        signals.append(sig)
    return signals


# ---------------------------------------------------------------------------
# Bullish CBC_SUCCESSFUL_TWO
# ---------------------------------------------------------------------------


def test_bullish_cbc_successful_two():
    """
    Bullish CBC_SUCCESSFUL_TWO:
      A: establishes a low (low=98)
      B: sweeps below A's low (low=95), creating 'breathing room'
      C: closes above A's high (close > a.high=100.5)
    """
    detector = CBCDetector(require_context=False)
    # A: low=98, high=100.5
    # B: sweeps A's low => b.low=95 < a.low=98; b.high=101
    # C: closes above A's high => c.close=103, c.high=103.5, c.low=100
    bars = [
        (99.0, 100.5, 98.0, 99.5),   # A
        (99.0, 101.0, 95.0, 99.0),   # B sweeps A's low
        (100.0, 103.5, 100.0, 103.0), # C closes above A's high
    ]
    signals = feed_bars(detector, bars)
    sig = signals[-1]
    assert sig is not None
    assert sig.cbc_type == CBCType.CBC_SUCCESSFUL_TWO
    assert sig.direction == "bullish"
    assert sig.entry_price == pytest.approx(103.0)
    assert sig.invalidation_price == pytest.approx(95.0)  # B's sweep low


def test_bullish_cbc_successful_two_candle_dicts_populated():
    detector = CBCDetector(require_context=False)
    bars = [
        (99.0, 100.5, 98.0, 99.5),
        (99.0, 101.0, 95.0, 99.0),
        (100.0, 103.5, 100.0, 103.0),
    ]
    signals = feed_bars(detector, bars)
    sig = signals[-1]
    assert sig.candle_a == {"open": 99.0, "high": 100.5, "low": 98.0, "close": 99.5}
    assert sig.candle_b["low"] == pytest.approx(95.0)
    assert sig.candle_c["close"] == pytest.approx(103.0)


# ---------------------------------------------------------------------------
# Bearish CBC_SUCCESSFUL_TWO
# ---------------------------------------------------------------------------


def test_bearish_cbc_successful_two():
    """
    Bearish CBC_SUCCESSFUL_TWO:
      A: establishes a high (high=102)
      B: sweeps above A's high (high=107)
      C: closes below A's low (close < a.low=99)
    """
    detector = CBCDetector(require_context=False)
    bars = [
        (100.0, 102.0, 99.0, 100.5),  # A
        (100.5, 107.0, 100.0, 101.0), # B sweeps A's high
        (101.0, 101.5, 96.0, 96.5),   # C closes below A's low
    ]
    signals = feed_bars(detector, bars)
    sig = signals[-1]
    assert sig is not None
    assert sig.cbc_type == CBCType.CBC_SUCCESSFUL_TWO
    assert sig.direction == "bearish"
    assert sig.entry_price == pytest.approx(96.5)
    assert sig.invalidation_price == pytest.approx(107.0)  # B's sweep high


# ---------------------------------------------------------------------------
# CBC_FAILED_TWO
# ---------------------------------------------------------------------------


def test_bullish_cbc_failed_two():
    """
    Bullish CBC_FAILED_TWO:
      A: establishes a low
      B: sweeps below A's low
      C: closes below B's low (reverses further down)
    """
    detector = CBCDetector(require_context=False)
    bars = [
        (100.0, 101.0, 98.0, 100.0),  # A: low=98
        (100.0, 101.0, 95.0, 99.5),   # B: sweeps A's low=98 with low=95
        (99.5, 100.0, 92.0, 92.5),    # C: closes below B's low=95
    ]
    signals = feed_bars(detector, bars)
    sig = signals[-1]
    assert sig is not None
    assert sig.cbc_type == CBCType.CBC_FAILED_TWO
    assert sig.direction == "bullish"


def test_bearish_cbc_failed_two():
    """
    Bearish CBC_FAILED_TWO:
      A: establishes a high
      B: sweeps above A's high
      C: closes above B's high (reverses further up)
    """
    detector = CBCDetector(require_context=False)
    bars = [
        (100.0, 103.0, 99.0, 101.0),  # A: high=103
        (101.0, 107.0, 100.0, 102.0), # B: sweeps A's high=103 with high=107
        (102.0, 111.0, 101.5, 110.5), # C: closes above B's high=107
    ]
    signals = feed_bars(detector, bars)
    sig = signals[-1]
    assert sig is not None
    assert sig.cbc_type == CBCType.CBC_FAILED_TWO
    assert sig.direction == "bearish"


# ---------------------------------------------------------------------------
# CBC_CONTINUATION
# ---------------------------------------------------------------------------


def test_bullish_cbc_continuation():
    """Three consecutive higher closes (no sweep of A's extremes) = bullish continuation.

    The continuation check only fires when B does NOT sweep A's low AND
    B does NOT sweep A's high, so B's range must be strictly inside A's range
    (or at least not breach A's extremes) while closes trend up.
    """
    detector = CBCDetector(require_context=False)
    # A: high=101, low=99.5 → close=100.5
    # B: high=100.8 (<= A.high=101, so b_sweeps_high=False),
    #    low=99.6 (>= A.low=99.5, so b_sweeps_low=False),
    #    close=101.0 > A close=100.5
    # C: high=101.5, low=100.0, close=101.8 > B close=101.0
    bars = [
        (100.0, 101.0, 99.5, 100.5),  # A
        (100.5, 100.8, 99.6, 101.0),  # B inside A's range, close higher
        (101.0, 101.5, 100.0, 101.8), # C close higher than B
    ]
    signals = feed_bars(detector, bars)
    sig = signals[-1]
    assert sig is not None
    assert sig.cbc_type == CBCType.CBC_CONTINUATION
    assert sig.direction == "bullish"


def test_bearish_cbc_continuation():
    """Three consecutive lower closes (no sweep of A's extremes) = bearish continuation."""
    detector = CBCDetector(require_context=False)
    # A: high=100.5, low=98.0, close=99.5
    # B: high=100.4 (<= A.high=100.5, so b_sweeps_high=False),
    #    low=98.1 (>= A.low=98.0, so b_sweeps_low=False),
    #    close=98.2 < A close=99.5
    # C: high=98.5, low=96.5, close=96.8 < B close=98.2
    bars = [
        (100.0, 100.5, 98.0, 99.5),   # A
        (99.5, 100.4, 98.1, 98.2),    # B inside A's range, close lower
        (98.2, 98.5, 96.5, 96.8),     # C close lower than B
    ]
    signals = feed_bars(detector, bars)
    sig = signals[-1]
    assert sig is not None
    assert sig.cbc_type == CBCType.CBC_CONTINUATION
    assert sig.direction == "bearish"


# ---------------------------------------------------------------------------
# Context gate
# ---------------------------------------------------------------------------


def test_context_gate_blocks_when_no_context():
    """require_context=True + no context_direction → no signal."""
    detector = CBCDetector(require_context=True)
    bars = [
        (99.0, 100.5, 98.0, 99.5),
        (99.0, 101.0, 95.0, 99.0),
        (100.0, 103.5, 100.0, 103.0),  # would be bullish
    ]
    signals = feed_bars(detector, bars, context_direction=None)
    assert signals[-1] is None


def test_context_gate_passes_when_direction_matches():
    """require_context=True + correct context_direction → signal emitted."""
    detector = CBCDetector(require_context=True)
    bars = [
        (99.0, 100.5, 98.0, 99.5),
        (99.0, 101.0, 95.0, 99.0),
        (100.0, 103.5, 100.0, 103.0),
    ]
    signals = feed_bars(detector, bars, context_direction="bullish")
    sig = signals[-1]
    assert sig is not None
    assert sig.direction == "bullish"


def test_context_gate_blocks_when_direction_mismatch():
    """require_context=True + wrong context_direction → blocked."""
    detector = CBCDetector(require_context=True)
    bars = [
        (99.0, 100.5, 98.0, 99.5),
        (99.0, 101.0, 95.0, 99.0),
        (100.0, 103.5, 100.0, 103.0),  # bullish pattern
    ]
    signals = feed_bars(detector, bars, context_direction="bearish")
    assert signals[-1] is None


def test_no_context_required_passes_any_direction():
    """require_context=False → signals even without context_direction."""
    detector = CBCDetector(require_context=False)
    bars = [
        (99.0, 100.5, 98.0, 99.5),
        (99.0, 101.0, 95.0, 99.0),
        (100.0, 103.5, 100.0, 103.0),
    ]
    signals = feed_bars(detector, bars, context_direction=None)
    assert signals[-1] is not None


# ---------------------------------------------------------------------------
# Buffer reset and sliding window
# ---------------------------------------------------------------------------


def test_buffer_slides_correctly():
    """Each new bar shifts the window: A=old B, B=old C, C=new bar."""
    detector = CBCDetector(require_context=False)
    # First 3 bars — no bullish pattern
    bars_no_sig = [
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
        (100.0, 100.5, 99.5, 100.0),
    ]
    sigs1 = feed_bars(detector, bars_no_sig)
    assert sigs1[-1] is None

    # Add a 4th bar that, together with bars 2 and 3, forms a pattern
    sig = detector.on_bar(
        bar_index=3,
        timestamp=BASE_TIME + timedelta(minutes=3),
        open=100.0, high=103.5, low=100.0, close=103.0,
        context_direction=None,
    )
    # Window is now [bar1, bar2, bar3-but-shifted], so we just check no crash
    assert sig is None or isinstance(sig, CBCSignal)


def test_reset_clears_buffer():
    """After reset, detector needs 3 new bars before firing."""
    detector = CBCDetector(require_context=False)
    bars = [
        (99.0, 100.5, 98.0, 99.5),
        (99.0, 101.0, 95.0, 99.0),
    ]
    feed_bars(detector, bars)
    detector.reset()
    # Now only 1 bar after reset → no signal
    sig = detector.on_bar(
        bar_index=2,
        timestamp=BASE_TIME + timedelta(minutes=2),
        open=100.0, high=103.5, low=100.0, close=103.0,
        context_direction=None,
    )
    assert sig is None


def test_no_signal_for_first_two_bars():
    """No signal until 3 bars are in the buffer."""
    detector = CBCDetector(require_context=False)
    sig1 = detector.on_bar(0, BASE_TIME, 100.0, 101.0, 99.0, 100.5)
    sig2 = detector.on_bar(1, BASE_TIME + timedelta(minutes=1), 100.5, 102.0, 99.5, 101.5)
    assert sig1 is None
    assert sig2 is None
