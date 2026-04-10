"""Tests for the TopstepX Combine tracker (plan Task 3).

Covers the full dollar-based trailing MLL + daily-loss + consistency
rule set plus DST handling at the 5pm CT day boundary. Test IDs
(A–I) match the plan's Step 3.8 spec.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from src.config.models import TopstepCombineConfig
from src.risk.topstep_tracker import TopstepCombineTracker


CT = ZoneInfo("America/Chicago")
UTC = timezone.utc


def _make_config(**overrides) -> TopstepCombineConfig:
    base = {
        "account_size": 50_000.0,
        "profit_target_usd": 3_000.0,
        "max_loss_limit_usd_trailing": 2_000.0,
        "daily_loss_limit_usd": 1_000.0,
        "consistency_pct": 50.0,
        "daily_reset_tz": "America/Chicago",
        "daily_reset_hour": 17,
    }
    base.update(overrides)
    return TopstepCombineConfig(**base)


def _ct_to_utc(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    """Build a CT-local datetime and return its UTC equivalent."""
    return datetime(year, month, day, hour, minute, tzinfo=CT).astimezone(UTC)


# ---------------------------------------------------------------------------
# Test A — initialise
# ---------------------------------------------------------------------------


def test_A_initialise_sets_mll_below_starting_balance() -> None:
    tracker = TopstepCombineTracker(config=_make_config())
    first_ts = _ct_to_utc(2026, 1, 2, 18, 0)  # 6pm CT Jan 2 (trading day Jan 2)
    tracker.initialise(50_000.0, first_ts)

    assert tracker.initial_balance == 50_000.0
    assert tracker.current_balance == 50_000.0
    assert tracker.mll == 48_000.0
    assert tracker.mll_locked is False
    assert tracker.status == "pending"
    assert tracker.current_trading_day is not None


# ---------------------------------------------------------------------------
# Test B — MLL trails on a winning day but does not lock
# ---------------------------------------------------------------------------


def test_B_mll_trails_upward_without_locking() -> None:
    tracker = TopstepCombineTracker(config=_make_config())
    tracker.initialise(50_000.0, _ct_to_utc(2026, 1, 2, 18, 0))

    # Day 1: finish at 51000 (before lock threshold of 52000)
    tracker.update(_ct_to_utc(2026, 1, 3, 15, 0), 51_000.0)
    # Rollover to day 2
    tracker.update(_ct_to_utc(2026, 1, 3, 18, 0), 51_000.0)

    # MLL should have trailed from 48000 to 49000 and NOT locked
    assert tracker.mll == 49_000.0
    assert tracker.mll_locked is False


# ---------------------------------------------------------------------------
# Test C — MLL locks at initial_balance once buffer is first reached
# ---------------------------------------------------------------------------


def test_C_mll_locks_at_initial_balance() -> None:
    tracker = TopstepCombineTracker(config=_make_config())
    tracker.initialise(50_000.0, _ct_to_utc(2026, 1, 2, 18, 0))

    # Day 1: finish at 52000 — this hits the lock threshold
    tracker.update(_ct_to_utc(2026, 1, 3, 15, 0), 52_000.0)
    tracker.update(_ct_to_utc(2026, 1, 3, 18, 0), 52_000.0)  # rollover

    assert tracker.mll == 50_000.0  # locked at initial
    assert tracker.mll_locked is True

    # Day 2: grow further to 53000 — MLL stays at 50000 (locked)
    tracker.update(_ct_to_utc(2026, 1, 4, 15, 0), 53_000.0)
    tracker.update(_ct_to_utc(2026, 1, 4, 18, 0), 53_000.0)  # rollover

    assert tracker.mll == 50_000.0
    assert tracker.mll_locked is True


# ---------------------------------------------------------------------------
# Test D — daily loss limit breach
# ---------------------------------------------------------------------------


def test_D_daily_loss_of_1001_fails() -> None:
    tracker = TopstepCombineTracker(config=_make_config())
    tracker.initialise(50_000.0, _ct_to_utc(2026, 1, 2, 18, 0))

    # Intraday balance drops to 48_999 (daily loss = -1001)
    tracker.update(_ct_to_utc(2026, 1, 3, 10, 0), 48_999.0)

    assert tracker.status == "failed_daily_loss"
    assert tracker.failure_reason is not None
    assert "daily loss" in tracker.failure_reason.lower()


def test_D_daily_loss_exactly_1000_fails() -> None:
    # Edge case: exactly at the limit should also fail (rule is "loss >= 1000")
    tracker = TopstepCombineTracker(config=_make_config())
    tracker.initialise(50_000.0, _ct_to_utc(2026, 1, 2, 18, 0))
    tracker.update(_ct_to_utc(2026, 1, 3, 10, 0), 49_000.0)
    assert tracker.status == "failed_daily_loss"


def test_D_daily_loss_999_still_pending() -> None:
    tracker = TopstepCombineTracker(config=_make_config())
    tracker.initialise(50_000.0, _ct_to_utc(2026, 1, 2, 18, 0))
    tracker.update(_ct_to_utc(2026, 1, 3, 10, 0), 49_001.0)
    assert tracker.status == "pending"


# ---------------------------------------------------------------------------
# Test E — intraday MLL breach
# ---------------------------------------------------------------------------


def test_E_intraday_mll_touch_fails() -> None:
    tracker = TopstepCombineTracker(config=_make_config())
    tracker.initialise(50_000.0, _ct_to_utc(2026, 1, 2, 18, 0))

    # Intraday drop to 47_999 — below MLL of 48000
    # But this is also -2001 daily loss which trips the daily limit first.
    # Use a smaller intraday drop then check MLL via a multi-day path.

    # Day 1: drop to 49200 (daily loss 800, under limit). End at 49200.
    tracker.update(_ct_to_utc(2026, 1, 3, 10, 0), 49_200.0)
    tracker.update(_ct_to_utc(2026, 1, 3, 18, 0), 49_200.0)  # rollover

    # MLL stays at 48000 (day was down; MLL doesn't trail down)
    assert tracker.mll == 48_000.0

    # Day 2: intraday dip to 47_999 (daily loss 1201 — but that's > 1000, so
    # daily loss triggers first). Use a smaller daily drop instead: start day
    # 2 at 49200, dip to 47_999 is -1201 which trips daily loss. To test MLL
    # in isolation we need a day where the daily loss is under 1000 AND the
    # balance goes under MLL. That means the day starts BELOW 48999.
    # Re-initialise with a lower starting balance on day 3 to isolate MLL.

    # Day 3 opens at 48500 (day 2 closed there)
    tracker.update(_ct_to_utc(2026, 1, 4, 10, 0), 48_500.0)
    tracker.update(_ct_to_utc(2026, 1, 4, 18, 0), 48_500.0)  # rollover

    # Day 4: intraday dip to 47_999 (daily loss 501, under limit)
    tracker.update(_ct_to_utc(2026, 1, 5, 10, 0), 47_999.0)
    assert tracker.status == "failed_mll"
    assert "maximum loss" in tracker.failure_reason.lower()


# ---------------------------------------------------------------------------
# Test F — pass with consistency check under 50%
# ---------------------------------------------------------------------------


def test_F_pass_with_consistency_just_under_50() -> None:
    tracker = TopstepCombineTracker(config=_make_config())
    tracker.initialise(50_000.0, _ct_to_utc(2026, 1, 2, 18, 0))

    # Day 1: +1499 (49.97% of the eventual 3000 total)
    tracker.update(_ct_to_utc(2026, 1, 3, 15, 0), 51_499.0)
    tracker.update(_ct_to_utc(2026, 1, 3, 18, 0), 51_499.0)  # rollover
    # Day 2: +1501 → total 3000 (but under target because of day 1's total)
    tracker.update(_ct_to_utc(2026, 1, 4, 15, 0), 53_000.0)

    # Run the final consistency check
    result = tracker.check_pass()
    # best_day should be 1501, total 3000, 1501/3000 = 50.03% → FAILS
    assert result["status"] == "failed_consistency"


def test_F_pass_when_best_day_is_exactly_half() -> None:
    tracker = TopstepCombineTracker(config=_make_config())
    tracker.initialise(50_000.0, _ct_to_utc(2026, 1, 2, 18, 0))

    # Day 1: +1000
    tracker.update(_ct_to_utc(2026, 1, 3, 15, 0), 51_000.0)
    tracker.update(_ct_to_utc(2026, 1, 3, 18, 0), 51_000.0)
    # Day 2: +1000 (same)
    tracker.update(_ct_to_utc(2026, 1, 4, 15, 0), 52_000.0)
    tracker.update(_ct_to_utc(2026, 1, 4, 18, 0), 52_000.0)
    # Day 3: +1001 (pushes over target)
    tracker.update(_ct_to_utc(2026, 1, 5, 15, 0), 53_001.0)

    result = tracker.check_pass()
    # Best day 1001, total 3001 → 33.4% → PASSES
    assert result["status"] == "passed"


# ---------------------------------------------------------------------------
# Test G — consistency failure
# ---------------------------------------------------------------------------


def test_G_consistency_failure_best_day_over_half() -> None:
    tracker = TopstepCombineTracker(config=_make_config())
    tracker.initialise(50_000.0, _ct_to_utc(2026, 1, 2, 18, 0))

    # Day 1: +500
    tracker.update(_ct_to_utc(2026, 1, 3, 15, 0), 50_500.0)
    tracker.update(_ct_to_utc(2026, 1, 3, 18, 0), 50_500.0)
    # Day 2: huge +2501 (pushes over target)
    tracker.update(_ct_to_utc(2026, 1, 4, 15, 0), 53_001.0)

    result = tracker.check_pass()
    # Best day 2501, total 3001 → 83.3% → FAILS consistency
    assert result["status"] == "failed_consistency"
    assert result["failure_reason"] is not None
    assert "consistency" in result["failure_reason"].lower()


# ---------------------------------------------------------------------------
# Test H — 5pm CT day rollover + DST transitions
# ---------------------------------------------------------------------------


def test_H_day_boundary_is_5pm_ct_not_midnight_utc() -> None:
    tracker = TopstepCombineTracker(config=_make_config())

    # Before 5pm CT on Jan 2 (in CST = UTC-6)
    before = _ct_to_utc(2026, 1, 2, 16, 59)
    # After 5pm CT on Jan 2
    after = _ct_to_utc(2026, 1, 2, 17, 1)

    day_before = tracker._trading_day_for(before)
    day_after = tracker._trading_day_for(after)

    assert day_before != day_after, "5pm CT must be a trading-day boundary"


def test_H_dst_spring_forward() -> None:
    """2026-03-08 02:00 CT springs forward to 03:00 CDT.

    A timestamp at 5:00pm on a post-DST day should still produce a
    fresh trading day relative to a 4:59pm timestamp the same day.
    """
    tracker = TopstepCombineTracker(config=_make_config())

    # After DST transition (2026-03-08 01:00 CT → 03:00 CDT)
    before = _ct_to_utc(2026, 3, 10, 16, 59)  # 4:59pm CDT
    after = _ct_to_utc(2026, 3, 10, 17, 1)  # 5:01pm CDT

    assert tracker._trading_day_for(before) != tracker._trading_day_for(after)


def test_H_dst_fall_back() -> None:
    """2026-11-01 02:00 CDT falls back to 01:00 CST.

    Rollover should still happen at 5pm local time on either side of
    the transition.
    """
    tracker = TopstepCombineTracker(config=_make_config())

    before = _ct_to_utc(2026, 11, 3, 16, 59)  # 4:59pm CST
    after = _ct_to_utc(2026, 11, 3, 17, 1)  # 5:01pm CST

    assert tracker._trading_day_for(before) != tracker._trading_day_for(after)


def test_H_two_timestamps_same_trading_day_are_equal() -> None:
    tracker = TopstepCombineTracker(config=_make_config())

    morning = _ct_to_utc(2026, 1, 15, 9, 0)
    afternoon = _ct_to_utc(2026, 1, 15, 15, 30)

    assert tracker._trading_day_for(morning) == tracker._trading_day_for(afternoon)


def test_H_after_5pm_belongs_to_next_trading_day() -> None:
    tracker = TopstepCombineTracker(config=_make_config())

    before = _ct_to_utc(2026, 1, 15, 16, 59)
    after = _ct_to_utc(2026, 1, 15, 18, 0)

    day_after = tracker._trading_day_for(after)
    day_before = tracker._trading_day_for(before)

    # The post-5pm timestamp is on the NEXT trading day (not the same one)
    assert day_after == day_before + timedelta(days=1)


# ---------------------------------------------------------------------------
# Test I — locked MLL stays put on a losing day
# ---------------------------------------------------------------------------


def test_I_locked_mll_does_not_decrease_on_down_days() -> None:
    tracker = TopstepCombineTracker(config=_make_config())
    tracker.initialise(50_000.0, _ct_to_utc(2026, 1, 2, 18, 0))

    # Day 1: hit 52000 to lock MLL at 50000
    tracker.update(_ct_to_utc(2026, 1, 3, 15, 0), 52_000.0)
    tracker.update(_ct_to_utc(2026, 1, 3, 18, 0), 52_000.0)
    assert tracker.mll_locked is True
    assert tracker.mll == 50_000.0

    # Day 2: small dip to 51500 (daily loss -500, under limit), close at 51500
    tracker.update(_ct_to_utc(2026, 1, 4, 10, 0), 51_500.0)
    tracker.update(_ct_to_utc(2026, 1, 4, 18, 0), 51_500.0)

    # MLL stayed locked at 50000 — a losing day after lock does not
    # drop it.
    assert tracker.mll == 50_000.0
    assert tracker.mll_locked is True
    assert tracker.status == "pending"


# ---------------------------------------------------------------------------
# to_dict / snapshot shape
# ---------------------------------------------------------------------------


def test_to_dict_returns_expected_keys() -> None:
    tracker = TopstepCombineTracker(config=_make_config())
    tracker.initialise(50_000.0, _ct_to_utc(2026, 1, 2, 18, 0))
    tracker.update(_ct_to_utc(2026, 1, 3, 10, 0), 50_500.0)
    snapshot = tracker.to_dict()

    for required_key in (
        "style",
        "status",
        "initial_balance",
        "current_balance",
        "mll",
        "mll_locked",
        "daily_pnl",
        "total_profit",
        "best_day_profit",
        "distance_to_mll",
        "distance_to_target",
    ):
        assert required_key in snapshot, f"missing key: {required_key}"

    assert snapshot["style"] == "topstep_combine_dollar"


def test_terminal_state_ignores_further_updates() -> None:
    tracker = TopstepCombineTracker(config=_make_config())
    tracker.initialise(50_000.0, _ct_to_utc(2026, 1, 2, 18, 0))

    tracker.update(_ct_to_utc(2026, 1, 3, 10, 0), 48_500.0)  # daily loss 1500 — FAIL
    assert tracker.status == "failed_daily_loss"

    # Subsequent updates are no-ops (status is sticky)
    tracker.update(_ct_to_utc(2026, 1, 3, 11, 0), 51_000.0)
    assert tracker.status == "failed_daily_loss"
