"""Tests for FiftyTapCalculator — 50% retracement entry confirmation."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.strategy.strategies.sss import FiftyTapCalculator, FiftyTapLevel, SwingPoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_TIME = datetime(2024, 1, 1, 9, 0, 0)


def make_swing(price: float, swing_type: str, idx: int = 0) -> SwingPoint:
    return SwingPoint(
        index=idx,
        timestamp=BASE_TIME + timedelta(minutes=idx),
        price=price,
        swing_type=swing_type,
        bar_count_since_prev=10,
    )


# ---------------------------------------------------------------------------
# Bullish 50% calculation
# ---------------------------------------------------------------------------


def test_bullish_fifty_percent_level():
    """
    Bullish: anchor_low=1900, target_high=1920.
    Level = 1900 + (1920 - 1900) * 0.5 = 1910.
    """
    calc = FiftyTapCalculator(tap_level=0.5)
    anchor = make_swing(1900.0, "low")
    target = make_swing(1920.0, "low")  # 'highest swing low' in Jay's model
    level = calc.calculate_level("bullish", anchor, target)
    assert level.price == pytest.approx(1910.0)
    assert level.direction == "bullish"
    assert level.anchor_low == pytest.approx(1900.0)
    assert level.anchor_high == pytest.approx(1920.0)
    assert level.is_tapped is False
    assert level.is_invalidated is False


def test_bullish_custom_tap_level_618():
    """0.618 Fibonacci retracement."""
    calc = FiftyTapCalculator(tap_level=0.618)
    anchor = make_swing(1900.0, "low")
    target = make_swing(1920.0, "low")
    level = calc.calculate_level("bullish", anchor, target)
    expected = 1900.0 + (1920.0 - 1900.0) * 0.618
    assert level.price == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Bearish 50% calculation
# ---------------------------------------------------------------------------


def test_bearish_fifty_percent_level():
    """
    Bearish: anchor_high=1920, target_low=1900.
    Level = 1920 - (1920 - 1900) * 0.5 = 1910.
    """
    calc = FiftyTapCalculator(tap_level=0.5)
    anchor = make_swing(1920.0, "high")
    target = make_swing(1900.0, "high")  # 'lowest swing high'
    level = calc.calculate_level("bearish", anchor, target)
    assert level.price == pytest.approx(1910.0)
    assert level.direction == "bearish"
    assert level.anchor_high == pytest.approx(1920.0)
    assert level.anchor_low == pytest.approx(1900.0)


def test_invalid_direction_raises():
    calc = FiftyTapCalculator()
    anchor = make_swing(1900.0, "low")
    target = make_swing(1920.0, "low")
    with pytest.raises(ValueError, match="direction must be 'bullish' or 'bearish'"):
        calc.calculate_level("sideways", anchor, target)


# ---------------------------------------------------------------------------
# Tap detection
# ---------------------------------------------------------------------------


def test_bullish_tap_when_bar_low_touches_level():
    """Bullish tap: bar.low <= level.price + tolerance."""
    calc = FiftyTapCalculator(tap_level=0.5, tolerance_pips=0.5, pip_value=0.1)
    anchor = make_swing(1900.0, "low")
    target = make_swing(1920.0, "low")
    level = calc.calculate_level("bullish", anchor, target)
    # level.price = 1910; tolerance = 0.5 * 0.1 = 0.05
    # bar.low = 1910.02 <= 1910 + 0.05 = 1910.05 → should tap
    updated = calc.check_tap(level, bar_index=5, high=1912.0, low=1910.02)
    assert updated.is_tapped is True
    assert updated.tap_bar_index == 5


def test_bullish_tap_exactly_at_level():
    calc = FiftyTapCalculator(tap_level=0.5, tolerance_pips=0.0, pip_value=0.1)
    anchor = make_swing(1900.0, "low")
    target = make_swing(1920.0, "low")
    level = calc.calculate_level("bullish", anchor, target)
    # bar.low == level.price exactly
    updated = calc.check_tap(level, bar_index=3, high=1912.0, low=1910.0)
    assert updated.is_tapped is True


def test_no_tap_when_price_does_not_reach_level():
    """Bar low well above the 50% level → no tap."""
    calc = FiftyTapCalculator(tap_level=0.5, tolerance_pips=0.5, pip_value=0.1)
    anchor = make_swing(1900.0, "low")
    target = make_swing(1920.0, "low")
    level = calc.calculate_level("bullish", anchor, target)
    # level.price=1910; bar.low=1912 > 1910.05
    updated = calc.check_tap(level, bar_index=3, high=1915.0, low=1912.0)
    assert updated.is_tapped is False


def test_bearish_tap_when_bar_high_touches_level():
    """Bearish tap: bar.high >= level.price - tolerance."""
    calc = FiftyTapCalculator(tap_level=0.5, tolerance_pips=0.5, pip_value=0.1)
    anchor = make_swing(1920.0, "high")
    target = make_swing(1900.0, "high")
    level = calc.calculate_level("bearish", anchor, target)
    # level.price=1910; tolerance=0.05; bar.high=1909.98 >= 1910-0.05=1909.95 → tap
    updated = calc.check_tap(level, bar_index=7, high=1909.98, low=1905.0)
    assert updated.is_tapped is True


def test_check_tap_returns_same_level_if_already_tapped():
    calc = FiftyTapCalculator(tap_level=0.5)
    anchor = make_swing(1900.0, "low")
    target = make_swing(1920.0, "low")
    level = calc.calculate_level("bullish", anchor, target)
    level = calc.check_tap(level, bar_index=5, high=1912.0, low=1909.0)
    assert level.is_tapped is True
    # Call again — should return unchanged
    level2 = calc.check_tap(level, bar_index=6, high=1911.0, low=1908.0)
    assert level2.is_tapped is True
    assert level2.tap_bar_index == 5  # original tap bar index unchanged


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------


def test_bullish_invalidated_when_anchor_swept_down():
    """Bullish level: invalidated when low < anchor_low."""
    calc = FiftyTapCalculator()
    anchor = make_swing(1900.0, "low")
    target = make_swing(1920.0, "low")
    level = calc.calculate_level("bullish", anchor, target)
    # low = 1899 < anchor_low = 1900 → invalidated
    updated = calc.check_invalidation(level, high=1905.0, low=1899.0)
    assert updated.is_invalidated is True


def test_bearish_invalidated_when_anchor_swept_up():
    """Bearish level: invalidated when high > anchor_high."""
    calc = FiftyTapCalculator()
    anchor = make_swing(1920.0, "high")
    target = make_swing(1900.0, "high")
    level = calc.calculate_level("bearish", anchor, target)
    # high = 1925 > anchor_high = 1920 → invalidated
    updated = calc.check_invalidation(level, high=1925.0, low=1910.0)
    assert updated.is_invalidated is True


def test_no_invalidation_when_anchor_holds():
    calc = FiftyTapCalculator()
    anchor = make_swing(1900.0, "low")
    target = make_swing(1920.0, "low")
    level = calc.calculate_level("bullish", anchor, target)
    # low stays at 1901 — anchor at 1900 not swept
    updated = calc.check_invalidation(level, high=1915.0, low=1901.0)
    assert updated.is_invalidated is False


def test_already_invalidated_level_unchanged():
    calc = FiftyTapCalculator()
    anchor = make_swing(1900.0, "low")
    target = make_swing(1920.0, "low")
    level = calc.calculate_level("bullish", anchor, target)
    level = calc.check_invalidation(level, high=1905.0, low=1899.0)
    assert level.is_invalidated is True
    # Second call — already invalidated
    level2 = calc.check_invalidation(level, high=1895.0, low=1890.0)
    assert level2.is_invalidated is True


def test_check_tap_skipped_when_invalidated():
    """Invalidated level should not be tapped."""
    calc = FiftyTapCalculator(tap_level=0.5, tolerance_pips=0.5, pip_value=0.1)
    anchor = make_swing(1900.0, "low")
    target = make_swing(1920.0, "low")
    level = calc.calculate_level("bullish", anchor, target)
    level = calc.check_invalidation(level, high=1905.0, low=1899.0)
    updated = calc.check_tap(level, bar_index=5, high=1912.0, low=1909.0)
    assert updated.is_tapped is False


# ---------------------------------------------------------------------------
# Custom tap_level
# ---------------------------------------------------------------------------


def test_custom_tap_level_0_382():
    """0.382 Fibonacci = 38.2% retracement."""
    calc = FiftyTapCalculator(tap_level=0.382)
    anchor = make_swing(1900.0, "low")
    target = make_swing(1920.0, "low")
    level = calc.calculate_level("bullish", anchor, target)
    expected = 1900.0 + (1920.0 - 1900.0) * 0.382
    assert level.price == pytest.approx(expected)
