"""Tests for SequenceExitMode — breathing room trailing stops."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from src.risk.exit_manager import ActiveTrade, ExitDecision
from src.strategy.strategies.sss import SequenceExitMode, SwingPoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_TIME = datetime(2024, 1, 1, 9, 0, 0)


def make_trade(
    entry_price: float = 1900.0,
    stop_loss: float = 1890.0,
    take_profit: float = 1920.0,
    direction: str = "long",
    lot_size: float = 1.0,
) -> ActiveTrade:
    return ActiveTrade(
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        direction=direction,
        lot_size=lot_size,
        entry_time=BASE_TIME,
    )


def make_exit_mode(**kwargs) -> SequenceExitMode:
    return SequenceExitMode(**kwargs)


def make_swing(price: float, swing_type: str, idx: int = 1) -> SwingPoint:
    return SwingPoint(
        index=idx,
        timestamp=BASE_TIME + timedelta(minutes=idx),
        price=price,
        swing_type=swing_type,
        bar_count_since_prev=5,
    )


# ---------------------------------------------------------------------------
# Initial stop calculation
# ---------------------------------------------------------------------------


def test_initial_stop_long_at_invalidation_price():
    exit_mode = make_exit_mode(min_stop_pips=5.0, pip_value=0.1)
    # entry=1900, invalidation=1890 → distance=10, min=0.5 → OK
    stop = exit_mode.calculate_initial_stop("long", invalidation_price=1890.0, spread=0.0, entry_price=1900.0)
    assert stop == pytest.approx(1890.0)


def test_initial_stop_short_at_invalidation_price():
    exit_mode = make_exit_mode(min_stop_pips=5.0, pip_value=0.1)
    # entry=1900, invalidation=1910 → distance=10 → OK
    stop = exit_mode.calculate_initial_stop("short", invalidation_price=1910.0, spread=0.0, entry_price=1900.0)
    assert stop == pytest.approx(1910.0)


def test_initial_stop_returns_none_when_too_tight():
    """stop_distance < min_stop_distance → return None (skip the trade)."""
    exit_mode = make_exit_mode(min_stop_pips=20.0, pip_value=0.1)
    # distance = 0.5, min = 20 * 0.1 = 2.0 → too tight
    stop = exit_mode.calculate_initial_stop("long", invalidation_price=1899.5, spread=0.0, entry_price=1900.0)
    assert stop is None


def test_initial_stop_considers_spread_multiplier():
    """spread * multiplier = 0.3 * 2 = 0.6 → min_dist = max(0.6, 20*0.1=2.0) = 2.0."""
    exit_mode = make_exit_mode(spread_multiplier=2.0, min_stop_pips=20.0, pip_value=0.1)
    # entry=1900, invalidation=1899, distance=1.0 < min=2.0 → None
    stop = exit_mode.calculate_initial_stop("long", invalidation_price=1899.0, spread=0.3, entry_price=1900.0)
    assert stop is None


def test_initial_stop_uses_spread_floor_when_bigger():
    """spread floor > pip floor → use spread floor."""
    exit_mode = make_exit_mode(spread_multiplier=2.0, min_stop_pips=1.0, pip_value=0.1)
    # spread=5.0, spread_floor=10.0, pip_floor=0.1 → min=10.0
    # entry=1900, invalidation=1895, distance=5.0 < 10.0 → None
    stop = exit_mode.calculate_initial_stop("long", invalidation_price=1895.0, spread=5.0, entry_price=1900.0)
    assert stop is None


# ---------------------------------------------------------------------------
# Trail logic
# ---------------------------------------------------------------------------


def test_trail_stop_moves_up_on_higher_swing_low_for_long():
    """Long trade: stop should trail up to a higher swing low."""
    exit_mode = make_exit_mode()
    trade = make_trade(entry_price=1900.0, stop_loss=1890.0, direction="long")
    # Swing low at 1895 → higher than original stop 1890 → should improve
    swings = [make_swing(1895.0, "low")]
    decision = exit_mode.check_exit(trade, current_price=1910.0, recent_swings=swings)
    assert decision.action == "trail_update"
    assert decision.new_stop == pytest.approx(1895.0)


def test_trail_stop_does_not_move_back_for_long():
    """Long trade: trailing stop should never move BELOW current stop."""
    exit_mode = make_exit_mode()
    trade = make_trade(entry_price=1900.0, stop_loss=1895.0, direction="long")
    # swing low at 1892 < current stop 1895 → should NOT update
    swings = [make_swing(1892.0, "low")]
    decision = exit_mode.check_exit(trade, current_price=1910.0, recent_swings=swings)
    assert decision.action == "no_action"
    assert trade.stop_loss == pytest.approx(1895.0)  # unchanged


def test_trail_stop_moves_down_on_lower_swing_high_for_short():
    """Short trade: stop trails down to a lower swing high."""
    exit_mode = make_exit_mode()
    trade = make_trade(entry_price=1900.0, stop_loss=1910.0, direction="short")
    # swing high at 1905 < current stop 1910 → improvement
    swings = [make_swing(1905.0, "high")]
    decision = exit_mode.check_exit(trade, current_price=1890.0, recent_swings=swings)
    assert decision.action == "trail_update"
    assert decision.new_stop == pytest.approx(1905.0)


def test_trail_stop_does_not_move_up_for_short():
    """Short trade: trailing stop must not move UP (away from price)."""
    exit_mode = make_exit_mode()
    trade = make_trade(entry_price=1900.0, stop_loss=1905.0, direction="short")
    # swing high at 1907 > current stop 1905 → worse, not an improvement
    swings = [make_swing(1907.0, "high")]
    decision = exit_mode.check_exit(trade, current_price=1890.0, recent_swings=swings)
    assert decision.action == "no_action"


def test_trail_uses_most_recent_swing():
    """Most recent swing should be used for trailing (reversed scan)."""
    exit_mode = make_exit_mode()
    trade = make_trade(entry_price=1900.0, stop_loss=1890.0, direction="long")
    # Two swing lows — the last one (1897) is more recent and more favorable
    swings = [make_swing(1892.0, "low", idx=1), make_swing(1897.0, "low", idx=2)]
    decision = exit_mode.check_exit(trade, current_price=1915.0, recent_swings=swings)
    assert decision.action == "trail_update"
    assert decision.new_stop == pytest.approx(1897.0)


def test_no_trail_when_no_swing_lows_for_long():
    """No swing lows in recent_swings → no trail update."""
    exit_mode = make_exit_mode()
    trade = make_trade(entry_price=1900.0, stop_loss=1890.0, direction="long")
    swings = [make_swing(1910.0, "high")]  # only highs, no lows
    decision = exit_mode.check_exit(trade, current_price=1910.0, recent_swings=swings)
    assert decision.action == "no_action"


# ---------------------------------------------------------------------------
# Stop hit detection
# ---------------------------------------------------------------------------


def test_full_exit_when_stop_hit_for_long():
    """Long trade: full exit when bar low <= stop_loss."""
    exit_mode = make_exit_mode()
    trade = make_trade(entry_price=1900.0, stop_loss=1890.0, direction="long")
    decision = exit_mode.check_exit(
        trade, current_price=1888.0, recent_swings=[],
        bar_high=1892.0, bar_low=1888.0,
    )
    assert decision.action == "full_exit"
    assert decision.close_pct == pytest.approx(1.0)


def test_full_exit_when_stop_hit_for_short():
    """Short trade: full exit when bar high >= stop_loss."""
    exit_mode = make_exit_mode()
    trade = make_trade(entry_price=1900.0, stop_loss=1910.0, direction="short")
    decision = exit_mode.check_exit(
        trade, current_price=1912.0, recent_swings=[],
        bar_high=1912.0, bar_low=1900.0,
    )
    assert decision.action == "full_exit"


def test_no_action_when_trade_holding():
    """Trade above stop with no improvement → no action."""
    exit_mode = make_exit_mode()
    trade = make_trade(entry_price=1900.0, stop_loss=1890.0, direction="long")
    decision = exit_mode.check_exit(
        trade, current_price=1905.0, recent_swings=[],
    )
    assert decision.action == "no_action"
    assert "Holding" in decision.reason


def test_stop_hit_priority_over_trail():
    """Stop hit should be returned even when a trail update exists."""
    exit_mode = make_exit_mode()
    trade = make_trade(entry_price=1900.0, stop_loss=1890.0, direction="long")
    # swing low at 1892 > stop 1890 (trail would be valid)
    swings = [make_swing(1892.0, "low")]
    # but bar low = 1889 < stop 1890 → stop hit wins
    decision = exit_mode.check_exit(
        trade, current_price=1889.0, recent_swings=swings,
        bar_high=1900.0, bar_low=1889.0,
    )
    assert decision.action == "full_exit"


# ---------------------------------------------------------------------------
# R-multiple tracking
# ---------------------------------------------------------------------------


def test_r_multiple_positive_for_winning_long():
    exit_mode = make_exit_mode()
    trade = make_trade(entry_price=1900.0, stop_loss=1890.0, direction="long")
    decision = exit_mode.check_exit(trade, current_price=1920.0, recent_swings=[])
    # R = (1920 - 1900) / (1900 - 1890) = 20/10 = 2.0
    assert decision.r_multiple == pytest.approx(2.0)


def test_r_multiple_negative_for_losing_long():
    exit_mode = make_exit_mode()
    trade = make_trade(entry_price=1900.0, stop_loss=1890.0, direction="long")
    decision = exit_mode.check_exit(trade, current_price=1895.0, recent_swings=[])
    assert decision.r_multiple == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# min_stop enforcement
# ---------------------------------------------------------------------------


def test_min_stop_pips_floor():
    """pip_floor = 10 pips * 0.1 = 1.0 → stop distance of 0.5 too tight."""
    exit_mode = make_exit_mode(min_stop_pips=10.0, pip_value=0.1)
    stop = exit_mode.calculate_initial_stop("long", 1899.5, 0.0, 1900.0)
    assert stop is None  # 0.5 < 1.0
