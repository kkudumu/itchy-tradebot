"""Tests for SSLevelManager — SS-level POI tracking."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.strategy.strategies.sss import SSLevelManager, SSLevel, SequenceEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_TIME = datetime(2024, 1, 1, 9, 0, 0)


def make_event(
    event_type: str = "sequence_complete",
    direction: str = "bearish",
    level_price: float = 1900.0,
    candle_count: int = 12,
    layer: str = "SS",
    sequence_efficient: bool = False,
    timestamp: datetime = BASE_TIME,
) -> SequenceEvent:
    return SequenceEvent(
        event_type=event_type,
        direction=direction,
        level_price=level_price,
        candle_count=candle_count,
        layer=layer,
        timestamp=timestamp,
        sequence_efficient=sequence_efficient,
    )


# ---------------------------------------------------------------------------
# Level creation
# ---------------------------------------------------------------------------


def test_creates_level_for_inefficient_sequence_complete():
    mgr = SSLevelManager()
    evt = make_event(event_type="sequence_complete", sequence_efficient=False)
    lvl = mgr.on_sequence_event(evt)
    assert lvl is not None
    assert lvl.price == pytest.approx(1900.0)
    assert lvl.direction == "bearish"
    assert lvl.is_active is True


def test_creates_level_for_inefficient_sequence_timeout():
    mgr = SSLevelManager()
    evt = make_event(event_type="sequence_timeout", sequence_efficient=False, layer="ISS")
    lvl = mgr.on_sequence_event(evt)
    assert lvl is not None
    assert lvl.layer == "ISS"


def test_no_level_for_efficient_sequence():
    mgr = SSLevelManager()
    evt = make_event(event_type="sequence_complete", sequence_efficient=True)
    lvl = mgr.on_sequence_event(evt)
    assert lvl is None
    assert len(mgr.all_levels) == 0


def test_no_level_for_non_trigger_event_type():
    """Only sequence_complete and sequence_timeout create levels."""
    mgr = SSLevelManager()
    for et in ("p2_detected", "two_confirmed", "three_active", "four_five_resolved"):
        evt = make_event(event_type=et, sequence_efficient=False)
        lvl = mgr.on_sequence_event(evt)
        assert lvl is None


def test_no_level_for_cbc_layer():
    """CBC layer levels are not meaningful POIs."""
    mgr = SSLevelManager()
    evt = make_event(event_type="sequence_complete", sequence_efficient=False, layer="CBC")
    lvl = mgr.on_sequence_event(evt)
    assert lvl is None


def test_level_attributes_populated():
    mgr = SSLevelManager()
    t = datetime(2024, 3, 15, 10, 30, 0)
    evt = make_event(
        level_price=1950.0, direction="bullish", candle_count=15,
        layer="SS", timestamp=t, sequence_efficient=False,
    )
    lvl = mgr.on_sequence_event(evt)
    assert lvl.price == pytest.approx(1950.0)
    assert lvl.direction == "bullish"
    assert lvl.candle_count == 15
    assert lvl.layer == "SS"
    assert lvl.creation_time == t
    assert lvl.swept_time is None


# ---------------------------------------------------------------------------
# Price sweep invalidation
# ---------------------------------------------------------------------------


def test_bullish_level_swept_when_price_goes_below():
    """Bullish level (long target) swept when low < level.price."""
    mgr = SSLevelManager()
    mgr.on_sequence_event(make_event(
        direction="bullish", level_price=1900.0, sequence_efficient=False
    ))
    swept = mgr.update_price(BASE_TIME + timedelta(minutes=1), high=1905.0, low=1895.0)
    assert len(swept) == 1
    assert swept[0].price == pytest.approx(1900.0)
    assert swept[0].is_active is False
    assert swept[0].swept_time is not None


def test_bearish_level_swept_when_price_goes_above():
    """Bearish level (short target) swept when high > level.price."""
    mgr = SSLevelManager()
    mgr.on_sequence_event(make_event(
        direction="bearish", level_price=1900.0, sequence_efficient=False
    ))
    swept = mgr.update_price(BASE_TIME + timedelta(minutes=1), high=1905.0, low=1895.0)
    assert len(swept) == 1
    assert swept[0].direction == "bearish"
    assert swept[0].is_active is False


def test_level_not_swept_when_price_stays_away():
    mgr = SSLevelManager()
    mgr.on_sequence_event(make_event(direction="bullish", level_price=1900.0, sequence_efficient=False))
    swept = mgr.update_price(BASE_TIME + timedelta(minutes=1), high=1910.0, low=1905.0)
    assert swept == []
    assert len(mgr.active_levels) == 1


def test_already_swept_level_not_swept_again():
    mgr = SSLevelManager()
    mgr.on_sequence_event(make_event(direction="bullish", level_price=1900.0, sequence_efficient=False))
    mgr.update_price(BASE_TIME + timedelta(minutes=1), high=1905.0, low=1895.0)
    # Second call — already inactive
    swept2 = mgr.update_price(BASE_TIME + timedelta(minutes=2), high=1910.0, low=1890.0)
    assert swept2 == []


# ---------------------------------------------------------------------------
# Nearest target calculation
# ---------------------------------------------------------------------------


def test_nearest_target_for_bullish_trade():
    """Bullish trade (long) → nearest bearish level ABOVE current price."""
    mgr = SSLevelManager()
    mgr.on_sequence_event(make_event(direction="bearish", level_price=1910.0, sequence_efficient=False))
    mgr.on_sequence_event(make_event(
        direction="bearish", level_price=1920.0, sequence_efficient=False,
        timestamp=BASE_TIME + timedelta(seconds=1)
    ))
    nearest = mgr.get_nearest_target(current_price=1900.0, direction="bullish")
    assert nearest is not None
    assert nearest.price == pytest.approx(1910.0)  # closest above


def test_nearest_target_for_bearish_trade():
    """Bearish trade (short) → nearest bullish level BELOW current price."""
    mgr = SSLevelManager()
    mgr.on_sequence_event(make_event(direction="bullish", level_price=1880.0, sequence_efficient=False))
    mgr.on_sequence_event(make_event(
        direction="bullish", level_price=1870.0, sequence_efficient=False,
        timestamp=BASE_TIME + timedelta(seconds=1)
    ))
    nearest = mgr.get_nearest_target(current_price=1900.0, direction="bearish")
    assert nearest is not None
    assert nearest.price == pytest.approx(1880.0)  # closest below


def test_no_nearest_target_when_none_on_correct_side():
    mgr = SSLevelManager()
    # Add only bearish levels BELOW current price — irrelevant for a bearish trade
    mgr.on_sequence_event(make_event(direction="bearish", level_price=1890.0, sequence_efficient=False))
    nearest = mgr.get_nearest_target(current_price=1900.0, direction="bearish")
    assert nearest is None  # bearish trade needs bullish levels below


def test_swept_levels_excluded_from_nearest_target():
    mgr = SSLevelManager()
    mgr.on_sequence_event(make_event(direction="bearish", level_price=1910.0, sequence_efficient=False))
    # Sweep the only level
    mgr.update_price(BASE_TIME + timedelta(minutes=1), high=1915.0, low=1900.0)
    nearest = mgr.get_nearest_target(current_price=1900.0, direction="bullish")
    assert nearest is None


# ---------------------------------------------------------------------------
# R:R calculation
# ---------------------------------------------------------------------------


def test_rr_calculation():
    mgr = SSLevelManager()
    mgr.on_sequence_event(make_event(direction="bearish", level_price=1920.0, sequence_efficient=False))
    target = mgr.get_nearest_target(1900.0, "bullish")
    rr = mgr.calculate_target_rr(entry_price=1900.0, stop_loss=1890.0, target_level=target)
    assert rr == pytest.approx(2.0)  # reward=20, risk=10


def test_rr_returns_zero_when_risk_is_zero():
    mgr = SSLevelManager()
    mgr.on_sequence_event(make_event(direction="bearish", level_price=1910.0, sequence_efficient=False))
    target = mgr.get_nearest_target(1900.0, "bullish")
    rr = mgr.calculate_target_rr(entry_price=1900.0, stop_loss=1900.0, target_level=target)
    assert rr == 0.0


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------


def test_lru_eviction_when_max_exceeded():
    mgr = SSLevelManager(max_active_levels=3)
    for i in range(4):
        t = BASE_TIME + timedelta(seconds=i)
        mgr.on_sequence_event(make_event(
            direction="bearish", level_price=1900.0 + i, sequence_efficient=False, timestamp=t
        ))
    # After 4 inserts with max=3, the oldest should be evicted
    active = mgr.active_levels
    assert len(active) == 3


def test_lru_evicts_oldest_level():
    mgr = SSLevelManager(max_active_levels=2)
    t0 = BASE_TIME
    t1 = BASE_TIME + timedelta(seconds=1)
    t2 = BASE_TIME + timedelta(seconds=2)
    mgr.on_sequence_event(make_event(direction="bearish", level_price=1900.0, sequence_efficient=False, timestamp=t0))
    mgr.on_sequence_event(make_event(direction="bearish", level_price=1910.0, sequence_efficient=False, timestamp=t1))
    mgr.on_sequence_event(make_event(direction="bearish", level_price=1920.0, sequence_efficient=False, timestamp=t2))
    # oldest (t0, 1900.0) should be evicted
    active_prices = {lvl.price for lvl in mgr.active_levels}
    assert 1900.0 not in active_prices
    assert 1910.0 in active_prices or 1920.0 in active_prices


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def test_reset_clears_all_levels():
    mgr = SSLevelManager()
    mgr.on_sequence_event(make_event(sequence_efficient=False))
    mgr.on_sequence_event(make_event(direction="bullish", level_price=1880.0, sequence_efficient=False))
    assert len(mgr.all_levels) == 2
    mgr.reset()
    assert len(mgr.all_levels) == 0
    assert len(mgr.active_levels) == 0
