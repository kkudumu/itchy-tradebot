"""Tests for the profile-aware SessionClock (plan Task 4).

Covers forex (midnight UTC) and futures (5pm CT) day boundaries, DST
spring-forward / fall-back, day_open_ts, and is_new_day.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from src.risk.session_clock import SessionClock


CT = ZoneInfo("America/Chicago")
UTC = timezone.utc


def _ct_to_utc(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=CT).astimezone(UTC)


# ---------------------------------------------------------------------------
# Forex profile: midnight UTC rollover
# ---------------------------------------------------------------------------


class TestForexProfile:
    def _clock(self) -> SessionClock:
        return SessionClock(reset_hour_local=0, reset_tz="UTC")

    def test_two_timestamps_same_utc_day_are_same(self) -> None:
        clock = self._clock()
        a = datetime(2026, 1, 15, 0, 30, tzinfo=UTC)
        b = datetime(2026, 1, 15, 23, 30, tzinfo=UTC)
        assert clock.trading_day(a) == clock.trading_day(b)
        assert clock.trading_day(a) == date(2026, 1, 15)

    def test_midnight_rollover(self) -> None:
        clock = self._clock()
        before = datetime(2026, 1, 15, 23, 59, tzinfo=UTC)
        after = datetime(2026, 1, 16, 0, 0, tzinfo=UTC)
        assert clock.trading_day(before) != clock.trading_day(after)
        assert clock.trading_day(after) == date(2026, 1, 16)

    def test_naive_timestamps_treated_as_utc(self) -> None:
        clock = self._clock()
        naive = datetime(2026, 1, 15, 12, 0)
        aware = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
        assert clock.trading_day(naive) == clock.trading_day(aware)


# ---------------------------------------------------------------------------
# Futures profile: 5pm America/Chicago rollover
# ---------------------------------------------------------------------------


class TestFuturesProfile:
    def _clock(self) -> SessionClock:
        return SessionClock(reset_hour_local=17, reset_tz="America/Chicago")

    def test_5pm_ct_is_boundary(self) -> None:
        clock = self._clock()
        before = _ct_to_utc(2026, 1, 15, 16, 59)
        after = _ct_to_utc(2026, 1, 15, 17, 1)
        assert clock.trading_day(before) != clock.trading_day(after)

    def test_two_timestamps_between_rollovers_are_same_day(self) -> None:
        clock = self._clock()
        morning = _ct_to_utc(2026, 1, 15, 9, 0)
        afternoon = _ct_to_utc(2026, 1, 15, 15, 0)
        assert clock.trading_day(morning) == clock.trading_day(afternoon)

    def test_after_5pm_belongs_to_next_day(self) -> None:
        clock = self._clock()
        before = _ct_to_utc(2026, 1, 15, 16, 59)
        after = _ct_to_utc(2026, 1, 15, 18, 0)
        assert clock.trading_day(after) == clock.trading_day(before) + timedelta(days=1)


# ---------------------------------------------------------------------------
# DST transitions
# ---------------------------------------------------------------------------


class TestDST:
    def _clock(self) -> SessionClock:
        return SessionClock(reset_hour_local=17, reset_tz="America/Chicago")

    def test_spring_forward_boundary_still_works(self) -> None:
        """2026-03-08 01:59 CST → 03:00 CDT. The 5pm rollover that day
        must still reliably split into distinct trading days."""
        clock = self._clock()
        before = _ct_to_utc(2026, 3, 10, 16, 59)  # 4:59pm CDT (post-transition)
        after = _ct_to_utc(2026, 3, 10, 17, 1)  # 5:01pm CDT
        assert clock.trading_day(before) != clock.trading_day(after)

    def test_fall_back_boundary_still_works(self) -> None:
        """2026-11-01 01:59 CDT → 01:00 CST. 5pm rollover unaffected."""
        clock = self._clock()
        before = _ct_to_utc(2026, 11, 3, 16, 59)
        after = _ct_to_utc(2026, 11, 3, 17, 1)
        assert clock.trading_day(before) != clock.trading_day(after)


# ---------------------------------------------------------------------------
# is_new_day helper
# ---------------------------------------------------------------------------


class TestIsNewDay:
    def test_none_prev_always_new(self) -> None:
        clock = SessionClock(reset_hour_local=17, reset_tz="America/Chicago")
        assert clock.is_new_day(None, _ct_to_utc(2026, 1, 15, 10, 0)) is True

    def test_same_day_returns_false(self) -> None:
        clock = SessionClock(reset_hour_local=17, reset_tz="America/Chicago")
        assert (
            clock.is_new_day(
                _ct_to_utc(2026, 1, 15, 10, 0),
                _ct_to_utc(2026, 1, 15, 15, 0),
            )
            is False
        )

    def test_cross_boundary_returns_true(self) -> None:
        clock = SessionClock(reset_hour_local=17, reset_tz="America/Chicago")
        assert (
            clock.is_new_day(
                _ct_to_utc(2026, 1, 15, 16, 59),
                _ct_to_utc(2026, 1, 15, 17, 1),
            )
            is True
        )


# ---------------------------------------------------------------------------
# day_open_ts helper
# ---------------------------------------------------------------------------


class TestDayOpenTs:
    def test_forex_day_opens_at_midnight_utc(self) -> None:
        clock = SessionClock(reset_hour_local=0, reset_tz="UTC")
        open_ts = clock.day_open_ts(date(2026, 1, 15))
        assert open_ts == datetime(2026, 1, 15, 0, 0, tzinfo=UTC)

    def test_futures_day_opens_at_5pm_ct_same_calendar_day(self) -> None:
        clock = SessionClock(reset_hour_local=17, reset_tz="America/Chicago")
        # Trading day 2026-01-15 is labelled by the calendar date on which
        # its session begins — 5pm CT on 2026-01-15 — so it opens at
        # that timestamp and runs through 4:59pm CT on 2026-01-16.
        open_ts = clock.day_open_ts(date(2026, 1, 15))
        open_ct = open_ts.astimezone(CT)
        assert open_ct.year == 2026 and open_ct.month == 1 and open_ct.day == 15
        assert open_ct.hour == 17 and open_ct.minute == 0

    def test_futures_day_open_round_trips_through_trading_day(self) -> None:
        """day_open_ts should return a timestamp that immediately
        resolves back to the same trading day via trading_day()."""
        clock = SessionClock(reset_hour_local=17, reset_tz="America/Chicago")
        for d in (date(2026, 1, 15), date(2026, 3, 10), date(2026, 11, 3)):
            open_ts = clock.day_open_ts(d)
            assert clock.trading_day(open_ts) == d


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_valid_hours(self) -> None:
        for h in (0, 12, 17, 23):
            SessionClock(reset_hour_local=h, reset_tz="UTC")

    def test_invalid_hour_rejected(self) -> None:
        with pytest.raises(ValueError):
            SessionClock(reset_hour_local=24, reset_tz="UTC")
        with pytest.raises(ValueError):
            SessionClock(reset_hour_local=-1, reset_tz="UTC")


# ---------------------------------------------------------------------------
# DailyCircuitBreaker integration with SessionClock
# ---------------------------------------------------------------------------


class TestCircuitBreakerWithSessionClock:
    def test_breaker_without_clock_stays_backward_compat(self) -> None:
        from src.risk.circuit_breaker import DailyCircuitBreaker

        breaker = DailyCircuitBreaker(max_daily_loss_pct=2.0)
        breaker.start_day(10_000.0, date=date(2026, 1, 15))
        assert breaker.can_trade(9_900.0) is True  # 1% loss OK
        assert breaker.can_trade(9_795.0) is False  # 2.05% loss trips

    def test_breaker_with_dollar_limit(self) -> None:
        from src.risk.circuit_breaker import DailyCircuitBreaker

        breaker = DailyCircuitBreaker(max_daily_loss_usd=1_000.0)
        breaker.start_day(50_000.0, date=date(2026, 1, 15))
        assert breaker.can_trade(49_500.0) is True  # -500 OK
        assert breaker.can_trade(49_000.0) is False  # -1000 trips exactly

    def test_breaker_uses_session_clock_for_day(self) -> None:
        from src.risk.circuit_breaker import DailyCircuitBreaker

        clock = SessionClock(reset_hour_local=17, reset_tz="America/Chicago")
        breaker = DailyCircuitBreaker(session_clock=clock)

        # 18:00 CT on Jan 15 is AFTER the 17:00 reset, so it belongs to
        # the Jan 15 trading day (which opens at 17:00 Jan 15 and closes
        # at 16:59 Jan 16). Before 17:00 CT the same calendar day would
        # still be in the Jan 14 trading day.
        ts = _ct_to_utc(2026, 1, 15, 18, 0)
        breaker.start_day(50_000.0, ts=ts)
        assert breaker.current_date == date(2026, 1, 15)

        # Verify a pre-17:00 timestamp rolls back to the prior trading day
        breaker2 = DailyCircuitBreaker(session_clock=clock)
        ts_pre = _ct_to_utc(2026, 1, 15, 16, 30)
        breaker2.start_day(50_000.0, ts=ts_pre)
        assert breaker2.current_date == date(2026, 1, 14)
