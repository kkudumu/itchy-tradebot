"""
Tests for the 12 edge optimization modules and EdgeManager.

Each edge is tested in isolation using synthetic EdgeContext instances —
one that satisfies the condition (should pass) and one that violates it
(should fail). This isolation ensures edges can be evaluated independently
without coupling to other system components.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.edges.base import EdgeContext, EdgeResult
from src.edges.time_of_day import TimeOfDayFilter
from src.edges.day_of_week import DayOfWeekFilter
from src.edges.london_open_delay import LondonOpenDelayFilter
from src.edges.candle_close_confirmation import CandleCloseConfirmationFilter
from src.edges.spread_filter import SpreadFilter
from src.edges.news_filter import NewsFilter, NewsEvent, EmptyCalendar
from src.edges.friday_close import FridayCloseFilter
from src.edges.regime_filter import RegimeFilter
from src.edges.time_stop import TimeStopFilter
from src.edges.bb_squeeze import BBSqueezeAmplifier
from src.edges.confluence_scoring import ConfluenceScoringFilter
from src.edges.equity_curve import EquityCurveFilter
from src.edges.manager import EdgeManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

UTC = timezone.utc


def make_ts(hour: int, minute: int = 0, weekday: int = 2) -> datetime:
    """Build a UTC datetime for the given hour/minute on a chosen weekday.

    weekday 0=Mon, 1=Tue, 2=Wed(default), 3=Thu, 4=Fri
    Uses a fixed reference week (2024-01-01 is Monday).
    """
    # 2024-01-01 is Monday; offset by weekday days
    base = datetime(2024, 1, 1, tzinfo=UTC)  # Monday
    return base + timedelta(days=weekday, hours=hour, minutes=minute)


def make_context(
    hour: int = 10,
    minute: int = 0,
    weekday: int = 2,  # Wednesday
    spread: float = 20.0,
    adx: float = 35.0,
    cloud_thickness: float = 100.0,
    bb_squeeze: bool = False,
    confluence_score: int = 6,
    current_r: float | None = None,
    candles_since_entry: int | None = None,
    equity_curve: list | None = None,
    close_price: float = 1900.0,
    kijun_value: float = 1895.0,
    signal=None,
) -> EdgeContext:
    """Factory for synthetic EdgeContext with sensible defaults."""
    ts = make_ts(hour=hour, minute=minute, weekday=weekday)
    return EdgeContext(
        timestamp=ts,
        day_of_week=weekday,
        close_price=close_price,
        high_price=close_price + 5.0,
        low_price=close_price - 5.0,
        spread=spread,
        session="london",
        adx=adx,
        atr=10.0,
        cloud_thickness=cloud_thickness,
        kijun_value=kijun_value,
        bb_squeeze=bb_squeeze,
        confluence_score=confluence_score,
        current_r=current_r,
        candles_since_entry=candles_since_entry,
        equity_curve=equity_curve if equity_curve is not None else [],
        signal=signal,
    )


# ---------------------------------------------------------------------------
# Default edge config factories
# ---------------------------------------------------------------------------

def _cfg(**kwargs) -> dict:
    """Build a minimal edge config dict. kwargs override defaults."""
    base = {"enabled": True, "params": {}}
    base.update(kwargs)
    return base


# ===========================================================================
# 1. TimeOfDayFilter
# ===========================================================================


class TestTimeOfDayFilter:
    def _edge(self, start="08:00", end="17:00"):
        return TimeOfDayFilter(
            {"enabled": True, "params": {"start_utc": start, "end_utc": end}}
        )

    def test_within_window_passes(self):
        edge = self._edge()
        ctx = make_context(hour=10, minute=0)
        result = edge.should_allow(ctx)
        assert result.allowed is True
        assert "10:00" in result.reason

    def test_before_window_fails(self):
        edge = self._edge()
        ctx = make_context(hour=3, minute=0)
        result = edge.should_allow(ctx)
        assert result.allowed is False
        assert "03:00" in result.reason

    def test_after_window_fails(self):
        edge = self._edge()
        ctx = make_context(hour=19, minute=0)
        result = edge.should_allow(ctx)
        assert result.allowed is False

    def test_exact_start_boundary_passes(self):
        edge = self._edge()
        ctx = make_context(hour=8, minute=0)
        result = edge.should_allow(ctx)
        assert result.allowed is True

    def test_exact_end_boundary_fails(self):
        # end is exclusive
        edge = self._edge()
        ctx = make_context(hour=17, minute=0)
        result = edge.should_allow(ctx)
        assert result.allowed is False

    def test_disabled_always_passes(self):
        edge = TimeOfDayFilter(
            {"enabled": False, "params": {"start_utc": "08:00", "end_utc": "17:00"}}
        )
        ctx = make_context(hour=3)  # would normally fail
        result = edge.should_allow(ctx)
        assert result.allowed is True

    def test_result_carries_edge_name(self):
        edge = self._edge()
        result = edge.should_allow(make_context(hour=10))
        assert result.edge_name == "time_of_day"


# ===========================================================================
# 2. DayOfWeekFilter
# ===========================================================================


class TestDayOfWeekFilter:
    def _edge(self, allowed_days=None):
        days = allowed_days if allowed_days is not None else [1, 2, 3]
        return DayOfWeekFilter({"enabled": True, "params": {"allowed_days": days}})

    def test_wednesday_passes(self):
        edge = self._edge()
        ctx = make_context(weekday=2)  # Wednesday
        result = edge.should_allow(ctx)
        assert result.allowed is True
        assert "Wednesday" in result.reason

    def test_monday_fails(self):
        edge = self._edge()
        ctx = make_context(weekday=0)  # Monday
        result = edge.should_allow(ctx)
        assert result.allowed is False
        assert "Monday" in result.reason

    def test_friday_fails_default(self):
        edge = self._edge()
        ctx = make_context(weekday=4)
        result = edge.should_allow(ctx)
        assert result.allowed is False

    def test_custom_allowed_days(self):
        edge = self._edge(allowed_days=[0, 4])  # Mon + Fri only
        assert edge.should_allow(make_context(weekday=0)).allowed is True
        assert edge.should_allow(make_context(weekday=2)).allowed is False  # Wed blocked

    def test_disabled_always_passes(self):
        edge = DayOfWeekFilter({"enabled": False, "params": {"allowed_days": [2]}})
        ctx = make_context(weekday=0)  # Monday — would fail if enabled
        assert edge.should_allow(ctx).allowed is True


# ===========================================================================
# 3. LondonOpenDelayFilter
# ===========================================================================


class TestLondonOpenDelayFilter:
    def _edge(self, delay=30):
        return LondonOpenDelayFilter(
            {"enabled": True, "params": {"london_open_utc": "08:00", "delay_minutes": delay}}
        )

    def test_within_delay_fails(self):
        edge = self._edge()
        ctx = make_context(hour=8, minute=15)  # 08:15 — within 30 min window
        result = edge.should_allow(ctx)
        assert result.allowed is False
        assert "08:15" in result.reason

    def test_after_delay_passes(self):
        edge = self._edge()
        ctx = make_context(hour=9, minute=0)  # 09:00 — after 08:30 blackout
        result = edge.should_allow(ctx)
        assert result.allowed is True

    def test_exact_open_time_fails(self):
        edge = self._edge()
        ctx = make_context(hour=8, minute=0)  # exactly 08:00 — start of blackout
        result = edge.should_allow(ctx)
        assert result.allowed is False

    def test_exact_blackout_end_passes(self):
        edge = self._edge(delay=30)
        ctx = make_context(hour=8, minute=30)  # 08:30 — exactly at end (exclusive)
        result = edge.should_allow(ctx)
        assert result.allowed is True

    def test_before_london_open_passes(self):
        edge = self._edge()
        ctx = make_context(hour=7, minute=0)
        result = edge.should_allow(ctx)
        assert result.allowed is True

    def test_disabled_passes_during_blackout(self):
        edge = LondonOpenDelayFilter(
            {"enabled": False, "params": {"london_open_utc": "08:00", "delay_minutes": 30}}
        )
        ctx = make_context(hour=8, minute=15)
        assert edge.should_allow(ctx).allowed is True


# ===========================================================================
# 4. CandleCloseConfirmationFilter
# ===========================================================================


class _FakeSignal:
    def __init__(self, direction: str):
        self.direction = direction


class TestCandleCloseConfirmationFilter:
    def _edge(self):
        return CandleCloseConfirmationFilter({"enabled": True, "params": {}})

    def test_long_close_above_boundary_passes(self):
        edge = self._edge()
        ctx = make_context(close_price=1910.0, kijun_value=1900.0,
                           signal=_FakeSignal("long"))
        result = edge.should_allow(ctx)
        assert result.allowed is True

    def test_long_close_below_boundary_fails(self):
        edge = self._edge()
        ctx = make_context(close_price=1890.0, kijun_value=1900.0,
                           signal=_FakeSignal("long"))
        result = edge.should_allow(ctx)
        assert result.allowed is False

    def test_short_close_below_boundary_passes(self):
        edge = self._edge()
        ctx = make_context(close_price=1890.0, kijun_value=1900.0,
                           signal=_FakeSignal("short"))
        result = edge.should_allow(ctx)
        assert result.allowed is True

    def test_short_close_above_boundary_fails(self):
        edge = self._edge()
        ctx = make_context(close_price=1910.0, kijun_value=1900.0,
                           signal=_FakeSignal("short"))
        result = edge.should_allow(ctx)
        assert result.allowed is False

    def test_no_signal_passes_through(self):
        edge = self._edge()
        ctx = make_context(close_price=1890.0, kijun_value=1900.0, signal=None)
        result = edge.should_allow(ctx)
        assert result.allowed is True
        assert "skipped" in result.reason.lower()

    def test_disabled_passes(self):
        edge = CandleCloseConfirmationFilter({"enabled": False, "params": {}})
        ctx = make_context(close_price=1890.0, kijun_value=1900.0,
                           signal=_FakeSignal("long"))
        assert edge.should_allow(ctx).allowed is True


# ===========================================================================
# 5. SpreadFilter
# ===========================================================================


class TestSpreadFilter:
    def _edge(self, max_spread=30):
        return SpreadFilter(
            {"enabled": True, "params": {"max_spread_points": max_spread}}
        )

    def test_acceptable_spread_passes(self):
        edge = self._edge()
        ctx = make_context(spread=20.0)
        result = edge.should_allow(ctx)
        assert result.allowed is True
        assert "20.0" in result.reason

    def test_excessive_spread_fails(self):
        edge = self._edge()
        ctx = make_context(spread=50.0)
        result = edge.should_allow(ctx)
        assert result.allowed is False
        assert "50.0" in result.reason

    def test_exact_limit_passes(self):
        edge = self._edge(max_spread=30)
        ctx = make_context(spread=30.0)
        result = edge.should_allow(ctx)
        assert result.allowed is True

    def test_disabled_passes_wide_spread(self):
        edge = SpreadFilter({"enabled": False, "params": {"max_spread_points": 30}})
        ctx = make_context(spread=200.0)
        assert edge.should_allow(ctx).allowed is True


# ===========================================================================
# 6. NewsFilter
# ===========================================================================


class _MockCalendar:
    """Calendar with one red-impact event at 14:00 UTC today."""

    def __init__(self, event_hour: int = 14):
        self._event_hour = event_hour

    def get_events(self, date: datetime) -> list[NewsEvent]:
        event_ts = date.replace(hour=self._event_hour, minute=0, second=0, microsecond=0)
        return [NewsEvent(timestamp=event_ts, title="Test CPI", impact="red")]


class TestNewsFilter:
    def _edge(self, before=30, after=30, calendar=None):
        cfg = {
            "enabled": True,
            "params": {
                "block_minutes_before": before,
                "block_minutes_after": after,
                "impact_levels": ["red"],
            },
        }
        return NewsFilter(cfg, calendar=calendar)

    def test_no_calendar_always_passes(self):
        edge = self._edge()
        ctx = make_context(hour=14, minute=0)
        result = edge.should_allow(ctx)
        assert result.allowed is True

    def test_within_pre_event_window_fails(self):
        cal = _MockCalendar(event_hour=14)
        edge = self._edge(calendar=cal)
        # 13:45 is 15 min before 14:00 — within 30-min pre-event window
        ctx = make_context(hour=13, minute=45)
        result = edge.should_allow(ctx)
        assert result.allowed is False
        assert "Test CPI" in result.reason

    def test_within_post_event_window_fails(self):
        cal = _MockCalendar(event_hour=14)
        edge = self._edge(calendar=cal)
        # 14:15 is 15 min after 14:00 — within 30-min post-event window
        ctx = make_context(hour=14, minute=15)
        result = edge.should_allow(ctx)
        assert result.allowed is False

    def test_outside_event_window_passes(self):
        cal = _MockCalendar(event_hour=14)
        edge = self._edge(calendar=cal)
        # 12:00 is well outside the 13:30–14:30 blackout
        ctx = make_context(hour=12, minute=0)
        result = edge.should_allow(ctx)
        assert result.allowed is True

    def test_orange_event_ignored_when_filter_is_red_only(self):
        cal_events = [
            NewsEvent(
                timestamp=make_ts(hour=14),
                title="Minor Event",
                impact="orange",
            )
        ]

        class OrangeCalendar:
            def get_events(self, date):
                return cal_events

        edge = self._edge(calendar=OrangeCalendar())
        ctx = make_context(hour=14, minute=0)
        assert edge.should_allow(ctx).allowed is True

    def test_disabled_passes_during_event(self):
        cal = _MockCalendar(event_hour=14)
        cfg = {
            "enabled": False,
            "params": {"block_minutes_before": 30, "block_minutes_after": 30,
                       "impact_levels": ["red"]},
        }
        edge = NewsFilter(cfg, calendar=cal)
        ctx = make_context(hour=14, minute=0)
        assert edge.should_allow(ctx).allowed is True

    def test_set_calendar_injects_calendar(self):
        edge = self._edge()  # starts with EmptyCalendar
        ctx = make_context(hour=14, minute=0)
        assert edge.should_allow(ctx).allowed is True  # no events

        edge.set_calendar(_MockCalendar(event_hour=14))
        # Now same context should be blocked
        assert edge.should_allow(ctx).allowed is False


# ===========================================================================
# 7. FridayCloseFilter
# ===========================================================================


class TestFridayCloseFilter:
    def _edge(self, close_time="20:00"):
        return FridayCloseFilter(
            {"enabled": True, "params": {"close_time_utc": close_time, "day": 4}}
        )

    def test_friday_after_close_time_triggers_exit(self):
        edge = self._edge()
        ctx = make_context(hour=21, minute=0, weekday=4)  # Friday 21:00
        result = edge.should_allow(ctx)
        assert result.allowed is False
        assert "exit" in result.reason.lower()

    def test_friday_exactly_at_close_time_triggers_exit(self):
        edge = self._edge(close_time="20:00")
        ctx = make_context(hour=20, minute=0, weekday=4)
        result = edge.should_allow(ctx)
        assert result.allowed is False

    def test_friday_before_close_time_passes(self):
        edge = self._edge()
        ctx = make_context(hour=15, minute=0, weekday=4)  # Friday 15:00
        result = edge.should_allow(ctx)
        assert result.allowed is True

    def test_wednesday_not_affected(self):
        edge = self._edge()
        ctx = make_context(hour=21, minute=0, weekday=2)  # Wednesday 21:00
        result = edge.should_allow(ctx)
        assert result.allowed is True

    def test_thursday_not_affected(self):
        edge = self._edge()
        ctx = make_context(hour=21, minute=0, weekday=3)
        result = edge.should_allow(ctx)
        assert result.allowed is True

    def test_disabled_does_not_trigger_on_friday(self):
        edge = FridayCloseFilter(
            {"enabled": False, "params": {"close_time_utc": "20:00", "day": 4}}
        )
        ctx = make_context(hour=22, weekday=4)
        assert edge.should_allow(ctx).allowed is True


# ===========================================================================
# 8. RegimeFilter
# ===========================================================================


class TestRegimeFilter:
    def _edge(self, adx_min=28, cloud_min=50):
        return RegimeFilter(
            {"enabled": True, "params": {"adx_min": adx_min,
                                         "cloud_thickness_percentile": cloud_min}}
        )

    def test_strong_adx_thick_cloud_passes(self):
        edge = self._edge()
        ctx = make_context(adx=35.0, cloud_thickness=100.0)
        result = edge.should_allow(ctx)
        assert result.allowed is True
        assert "35.0" in result.reason

    def test_weak_adx_fails(self):
        edge = self._edge()
        ctx = make_context(adx=15.0, cloud_thickness=100.0)
        result = edge.should_allow(ctx)
        assert result.allowed is False
        assert "15.0" in result.reason

    def test_thin_cloud_fails(self):
        edge = self._edge()
        ctx = make_context(adx=35.0, cloud_thickness=20.0)
        result = edge.should_allow(ctx)
        assert result.allowed is False

    def test_both_failing_fails(self):
        edge = self._edge()
        ctx = make_context(adx=10.0, cloud_thickness=5.0)
        result = edge.should_allow(ctx)
        assert result.allowed is False

    def test_exact_thresholds_pass(self):
        edge = self._edge(adx_min=28, cloud_min=50)
        ctx = make_context(adx=28.0, cloud_thickness=50.0)
        result = edge.should_allow(ctx)
        assert result.allowed is True

    def test_disabled_passes_weak_regime(self):
        edge = RegimeFilter(
            {"enabled": False, "params": {"adx_min": 28, "cloud_thickness_percentile": 50}}
        )
        ctx = make_context(adx=5.0, cloud_thickness=1.0)
        assert edge.should_allow(ctx).allowed is True


# ===========================================================================
# 9. TimeStopFilter
# ===========================================================================


class TestTimeStopFilter:
    def _edge(self, candle_limit=12, r_threshold=0.5):
        return TimeStopFilter(
            {"enabled": True, "params": {
                "candle_limit": candle_limit,
                "breakeven_r_threshold": r_threshold,
            }}
        )

    def test_trade_below_r_after_limit_triggers_exit(self):
        edge = self._edge()
        ctx = make_context(candles_since_entry=15, current_r=0.3)
        result = edge.should_allow(ctx)
        assert result.allowed is False
        assert "exit" in result.reason.lower() or "breakeven" in result.reason.lower()

    def test_trade_above_r_after_limit_continues(self):
        edge = self._edge()
        ctx = make_context(candles_since_entry=15, current_r=0.8)
        result = edge.should_allow(ctx)
        assert result.allowed is True

    def test_trade_before_candle_limit_continues(self):
        edge = self._edge()
        ctx = make_context(candles_since_entry=5, current_r=0.1)
        result = edge.should_allow(ctx)
        assert result.allowed is True

    def test_exactly_at_limit_with_insufficient_r_triggers(self):
        edge = self._edge(candle_limit=12, r_threshold=0.5)
        ctx = make_context(candles_since_entry=12, current_r=0.3)
        result = edge.should_allow(ctx)
        assert result.allowed is False

    def test_exactly_at_r_threshold_passes(self):
        edge = self._edge(candle_limit=12, r_threshold=0.5)
        ctx = make_context(candles_since_entry=12, current_r=0.5)
        result = edge.should_allow(ctx)
        assert result.allowed is True

    def test_no_trade_context_passes(self):
        edge = self._edge()
        ctx = make_context()  # current_r=None, candles_since_entry=None
        result = edge.should_allow(ctx)
        assert result.allowed is True

    def test_disabled_does_not_exit(self):
        edge = TimeStopFilter(
            {"enabled": False, "params": {"candle_limit": 1, "breakeven_r_threshold": 10.0}}
        )
        ctx = make_context(candles_since_entry=100, current_r=0.0)
        assert edge.should_allow(ctx).allowed is True


# ===========================================================================
# 10. BBSqueezeAmplifier
# ===========================================================================


class TestBBSqueezeAmplifier:
    def _edge(self, boost=1):
        return BBSqueezeAmplifier(
            {"enabled": True, "params": {"confidence_boost": boost}}
        )

    def test_squeeze_returns_boost_modifier(self):
        edge = self._edge(boost=1)
        ctx = make_context(bb_squeeze=True)
        result = edge.should_allow(ctx)
        assert result.allowed is True
        assert result.modifier == 1.0
        assert "boost" in result.reason.lower()

    def test_no_squeeze_returns_zero_modifier(self):
        edge = self._edge()
        ctx = make_context(bb_squeeze=False)
        result = edge.should_allow(ctx)
        assert result.allowed is True
        assert result.modifier == 0.0

    def test_custom_boost_value(self):
        edge = self._edge(boost=2)
        ctx = make_context(bb_squeeze=True)
        result = edge.should_allow(ctx)
        assert result.modifier == 2.0

    def test_disabled_returns_zero_modifier(self):
        edge = BBSqueezeAmplifier(
            {"enabled": False, "params": {"confidence_boost": 1}}
        )
        ctx = make_context(bb_squeeze=True)
        result = edge.should_allow(ctx)
        assert result.allowed is True
        assert result.modifier == 0.0


# ===========================================================================
# 11. ConfluenceScoringFilter
# ===========================================================================


class TestConfluenceScoringFilter:
    def _edge(self, min_score=4, a_plus=7, b=5, c=4):
        return ConfluenceScoringFilter({
            "enabled": True,
            "params": {
                "min_score": min_score,
                "tier_a_plus_threshold": a_plus,
                "tier_b_threshold": b,
                "tier_c_threshold": c,
                "tier_b_size_pct": 0.75,
                "tier_c_size_pct": 0.50,
            }
        })

    def test_score_below_minimum_blocked(self):
        edge = self._edge(min_score=4)
        ctx = make_context(confluence_score=2)
        result = edge.should_allow(ctx)
        assert result.allowed is False
        assert result.modifier == 0.0

    def test_score_at_minimum_c_tier(self):
        edge = self._edge()
        ctx = make_context(confluence_score=4)
        result = edge.should_allow(ctx)
        assert result.allowed is True
        assert result.modifier == 0.50
        assert "C" in result.reason

    def test_score_b_tier(self):
        edge = self._edge()
        ctx = make_context(confluence_score=5)
        result = edge.should_allow(ctx)
        assert result.allowed is True
        assert result.modifier == 0.75
        assert "B" in result.reason

    def test_score_a_plus_tier(self):
        edge = self._edge()
        ctx = make_context(confluence_score=7)
        result = edge.should_allow(ctx)
        assert result.allowed is True
        assert result.modifier == 1.0
        assert "A+" in result.reason

    def test_high_score_full_size(self):
        edge = self._edge()
        ctx = make_context(confluence_score=8)
        result = edge.should_allow(ctx)
        assert result.modifier == 1.0

    def test_disabled_returns_full_size(self):
        edge = ConfluenceScoringFilter({"enabled": False, "params": {}})
        ctx = make_context(confluence_score=1)  # would fail if enabled
        result = edge.should_allow(ctx)
        assert result.allowed is True
        assert result.modifier == 1.0


# ===========================================================================
# 12. EquityCurveFilter
# ===========================================================================


class TestEquityCurveFilter:
    def _edge(self, lookback=20, reduced=0.5):
        return EquityCurveFilter({
            "enabled": True,
            "params": {"lookback_trades": lookback, "reduced_size_multiplier": reduced}
        })

    def test_equity_above_ma_full_size(self):
        edge = self._edge()
        # MA is (1 + 2 + 3) / 3 = 2.0; last trade = 3.0 > 2.0
        ctx = make_context(equity_curve=[1.0, 2.0, 3.0])
        result = edge.should_allow(ctx)
        assert result.allowed is True
        assert result.modifier == 1.0

    def test_equity_below_ma_reduced_size(self):
        edge = self._edge()
        # MA is (3 + 2 + 1) / 3 = 2.0; last trade = 1.0 < 2.0
        ctx = make_context(equity_curve=[3.0, 2.0, 1.0])
        result = edge.should_allow(ctx)
        assert result.allowed is True
        assert result.modifier == 0.5
        assert "reduced" in result.reason.lower() or "%" in result.reason

    def test_empty_curve_full_size(self):
        edge = self._edge()
        ctx = make_context(equity_curve=[])
        result = edge.should_allow(ctx)
        assert result.modifier == 1.0

    def test_single_trade_full_size(self):
        edge = self._edge()
        ctx = make_context(equity_curve=[-5.0])
        result = edge.should_allow(ctx)
        assert result.modifier == 1.0  # cannot compute MA with 1 point

    def test_lookback_window_respected(self):
        # lookback=3 — only last 3 trades used; first trade ignored
        edge = self._edge(lookback=3)
        # Full curve: [10.0, -1.0, -1.0, -1.0]; window = [-1.0, -1.0, -1.0]
        # MA = -1.0; last = -1.0 → equals MA → full size
        ctx = make_context(equity_curve=[10.0, -1.0, -1.0, -1.0])
        result = edge.should_allow(ctx)
        assert result.modifier == 1.0  # current == MA → passes

    def test_custom_reduced_multiplier(self):
        edge = self._edge(reduced=0.25)
        ctx = make_context(equity_curve=[3.0, 2.0, 1.0])  # declining → below MA
        result = edge.should_allow(ctx)
        assert result.modifier == 0.25

    def test_disabled_returns_full_size(self):
        edge = EquityCurveFilter({"enabled": False, "params": {}})
        ctx = make_context(equity_curve=[3.0, 2.0, -5.0])
        result = edge.should_allow(ctx)
        assert result.modifier == 1.0


# ===========================================================================
# 13. EdgeManager
# ===========================================================================


def _default_manager_config() -> dict:
    """Minimal config dict matching all 12 edge keys with default params."""
    return {
        "time_of_day":               {"enabled": True, "params": {"start_utc": "08:00", "end_utc": "17:00"}},
        "day_of_week":               {"enabled": True, "params": {"allowed_days": [1, 2, 3]}},
        "london_open_delay":         {"enabled": True, "params": {"london_open_utc": "08:00", "delay_minutes": 30}},
        "candle_close_confirmation": {"enabled": True, "params": {}},
        "spread_filter":             {"enabled": True, "params": {"max_spread_points": 30}},
        "news_filter":               {"enabled": True, "params": {"block_minutes_before": 30, "block_minutes_after": 30, "impact_levels": ["red"]}},
        "regime_filter":             {"enabled": True, "params": {"adx_min": 28, "cloud_thickness_percentile": 50}},
        "friday_close":              {"enabled": True, "params": {"close_time_utc": "20:00", "day": 4}},
        "time_stop":                 {"enabled": True, "params": {"candle_limit": 12, "breakeven_r_threshold": 0.5}},
        "bb_squeeze":                {"enabled": True, "params": {"confidence_boost": 1}},
        "confluence_scoring":        {"enabled": True, "params": {"min_score": 4, "tier_a_plus_threshold": 7, "tier_b_threshold": 5, "tier_c_threshold": 4, "tier_b_size_pct": 0.75, "tier_c_size_pct": 0.50}},
        "equity_curve":              {"enabled": True, "params": {"lookback_trades": 20, "reduced_size_multiplier": 0.5}},
    }


class TestEdgeManager:
    def _manager(self, overrides: dict | None = None) -> EdgeManager:
        cfg = _default_manager_config()
        if overrides:
            for k, v in overrides.items():
                cfg[k] = v
        return EdgeManager(cfg)

    def test_all_edges_pass_good_context(self):
        manager = self._manager()
        # Wednesday 10:00 UTC, good spread, strong ADX, good score
        ctx = make_context(
            hour=10, weekday=2, spread=20.0, adx=35.0,
            cloud_thickness=100.0, confluence_score=6,
        )
        ok, results = manager.check_entry(ctx)
        assert ok is True, f"Expected entry to pass; failing result: {[r for r in results if not r.allowed]}"

    def test_one_failing_edge_blocks_entry(self):
        manager = self._manager()
        # Monday is blocked by day_of_week
        ctx = make_context(hour=10, weekday=0)
        ok, results = manager.check_entry(ctx)
        assert ok is False
        # The failing result should be present
        assert any(not r.allowed for r in results)

    def test_spread_failure_blocks_entry(self):
        manager = self._manager()
        ctx = make_context(hour=10, weekday=2, spread=100.0)
        ok, results = manager.check_entry(ctx)
        assert ok is False

    def test_time_outside_window_blocks_entry(self):
        manager = self._manager()
        ctx = make_context(hour=3, weekday=2)
        ok, results = manager.check_entry(ctx)
        assert ok is False

    def test_friday_close_triggers_exit(self):
        manager = self._manager()
        ctx = make_context(hour=21, weekday=4)  # Friday 21:00
        triggered, results = manager.check_exit(ctx)
        assert triggered is True

    def test_time_stop_triggers_exit(self):
        manager = self._manager()
        ctx = make_context(candles_since_entry=15, current_r=0.2)
        triggered, results = manager.check_exit(ctx)
        assert triggered is True

    def test_no_exit_conditions_met(self):
        manager = self._manager()
        ctx = make_context(hour=10, weekday=2, candles_since_entry=5, current_r=1.0)
        triggered, results = manager.check_exit(ctx)
        assert triggered is False

    def test_get_modifiers_returns_all_modifier_edges(self):
        manager = self._manager()
        ctx = make_context(bb_squeeze=True, confluence_score=7)
        modifiers = manager.get_modifiers(ctx)
        assert "bb_squeeze" in modifiers
        assert "confluence_scoring" in modifiers
        assert "equity_curve" in modifiers

    def test_get_combined_size_multiplier_full_size(self):
        manager = self._manager()
        # A+ score (7), healthy equity curve → 1.0 × 1.0 = 1.0
        ctx = make_context(confluence_score=7, equity_curve=[1.0, 1.0, 1.0])
        multiplier = manager.get_combined_size_multiplier(ctx)
        assert multiplier == 1.0

    def test_get_combined_size_multiplier_reduced(self):
        manager = self._manager()
        # C tier (score=4 → 0.5) and equity below MA (0.5) → 0.5 × 0.5 = 0.25
        ctx = make_context(
            confluence_score=4,
            equity_curve=[3.0, 2.0, 1.0],  # declining → below MA
        )
        multiplier = manager.get_combined_size_multiplier(ctx)
        assert multiplier == pytest.approx(0.25)

    def test_toggle_edge_disables_filter(self):
        manager = self._manager()
        # Monday normally blocked by day_of_week
        ctx = make_context(hour=10, weekday=0)
        ok_before, _ = manager.check_entry(ctx)
        assert ok_before is False

        manager.toggle_edge("day_of_week", enabled=False)
        ok_after, _ = manager.check_entry(ctx)
        assert ok_after is True  # Now passes since day_of_week is off

    def test_toggle_edge_re_enables_filter(self):
        manager = self._manager()
        manager.toggle_edge("day_of_week", enabled=False)
        manager.toggle_edge("day_of_week", enabled=True)
        ctx = make_context(hour=10, weekday=0)  # Monday — should fail again
        ok, _ = manager.check_entry(ctx)
        assert ok is False

    def test_toggle_unknown_edge_raises(self):
        manager = self._manager()
        with pytest.raises(KeyError, match="Unknown edge"):
            manager.toggle_edge("nonexistent_edge", enabled=False)

    def test_get_enabled_edges_reflects_toggles(self):
        manager = self._manager()
        initial_enabled = manager.get_enabled_edges()
        assert "time_of_day" in initial_enabled

        manager.toggle_edge("time_of_day", enabled=False)
        updated_enabled = manager.get_enabled_edges()
        assert "time_of_day" not in updated_enabled

    def test_manager_loads_from_pydantic_model(self):
        """EdgeManager should accept a Pydantic EdgeConfig model."""
        from src.config.models import EdgeConfig
        cfg = EdgeConfig()
        manager = EdgeManager(cfg)
        assert len(manager.entry_edges) > 0
        assert len(manager.exit_edges) > 0
        assert len(manager.modifier_edges) > 0

    def test_manager_repr(self):
        manager = self._manager()
        r = repr(manager)
        assert "EdgeManager" in r

    def test_entry_edge_count(self):
        # 7 entry edges defined in registry
        manager = self._manager()
        assert len(manager.entry_edges) == 7

    def test_exit_edge_count(self):
        # 2 exit edges: friday_close, time_stop
        manager = self._manager()
        assert len(manager.exit_edges) == 2

    def test_modifier_edge_count(self):
        # 3 modifier edges: bb_squeeze, confluence_scoring, equity_curve
        manager = self._manager()
        assert len(manager.modifier_edges) == 3

    def test_all_edges_disabled_entry_always_passes(self):
        cfg = _default_manager_config()
        for key in cfg:
            cfg[key]["enabled"] = False
        manager = EdgeManager(cfg)
        # Even Monday at 3AM with bad spread should pass
        ctx = make_context(hour=3, weekday=0, spread=500.0, adx=5.0, cloud_thickness=0.0)
        ok, results = manager.check_entry(ctx)
        assert ok is True
        assert results == []  # No enabled edges evaluated

    def test_london_open_delay_blocks_at_8_15_on_wednesday(self):
        manager = self._manager()
        ctx = make_context(hour=8, minute=15, weekday=2)
        ok, results = manager.check_entry(ctx)
        assert ok is False
        assert any(r.edge_name == "london_open_delay" and not r.allowed for r in results)
