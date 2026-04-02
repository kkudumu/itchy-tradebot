# tests/test_econ_calendar.py
"""Tests for deterministic economic calendar date generator."""

import pytest
from datetime import datetime, timezone, timedelta


class TestNFPDates:
    def test_nfp_is_first_friday(self):
        """NFP is released on the first Friday of each month."""
        from src.macro.econ_calendar import EconCalendar

        cal = EconCalendar()
        nfp_dates = cal.get_nfp_dates(2025)

        assert len(nfp_dates) == 12
        for dt in nfp_dates:
            assert dt.weekday() == 4  # Friday
            assert dt.day <= 7  # First 7 days = first week

    def test_nfp_jan_2025(self):
        """NFP for January 2025 should be Friday Jan 3."""
        from src.macro.econ_calendar import EconCalendar

        cal = EconCalendar()
        nfp_dates = cal.get_nfp_dates(2025)
        assert nfp_dates[0].day == 3
        assert nfp_dates[0].month == 1


class TestFOMCDates:
    def test_fomc_count(self):
        """FOMC meets 8 times per year."""
        from src.macro.econ_calendar import EconCalendar

        cal = EconCalendar()
        fomc_dates = cal.get_fomc_dates(2025)
        assert len(fomc_dates) == 8

    def test_fomc_dates_are_wednesdays(self):
        """FOMC decisions are announced on Wednesdays."""
        from src.macro.econ_calendar import EconCalendar

        cal = EconCalendar()
        fomc_dates = cal.get_fomc_dates(2025)
        for dt in fomc_dates:
            assert dt.weekday() == 2  # Wednesday

    def test_fomc_months_match_schedule(self):
        """FOMC meets in Jan, Mar, May, Jun, Jul, Sep, Nov, Dec."""
        from src.macro.econ_calendar import EconCalendar

        cal = EconCalendar()
        fomc_dates = cal.get_fomc_dates(2025)
        months = sorted(set(dt.month for dt in fomc_dates))
        assert months == [1, 3, 5, 6, 7, 9, 11, 12]


class TestCPIDates:
    def test_cpi_count(self):
        """CPI is released monthly -> 12 dates per year."""
        from src.macro.econ_calendar import EconCalendar

        cal = EconCalendar()
        cpi_dates = cal.get_cpi_dates(2025)
        assert len(cpi_dates) == 12

    def test_cpi_falls_around_12th(self):
        """CPI is typically released around the 10th-15th of the month."""
        from src.macro.econ_calendar import EconCalendar

        cal = EconCalendar()
        cpi_dates = cal.get_cpi_dates(2025)
        for dt in cpi_dates:
            assert 8 <= dt.day <= 16  # ~12th +/- a few days for weekends


class TestGetAllEvents:
    def test_returns_sorted_events(self):
        from src.macro.econ_calendar import EconCalendar

        cal = EconCalendar()
        events = cal.get_all_events(2025)

        assert len(events) > 0
        timestamps = [e.timestamp for e in events]
        assert timestamps == sorted(timestamps)

    def test_events_have_correct_impact(self):
        from src.macro.econ_calendar import EconCalendar

        cal = EconCalendar()
        events = cal.get_all_events(2025)

        for event in events:
            assert event.impact == "red"  # All are high-impact

    def test_events_for_date_range(self):
        from src.macro.econ_calendar import EconCalendar

        cal = EconCalendar()
        start = datetime(2025, 3, 1, tzinfo=timezone.utc)
        end = datetime(2025, 3, 31, tzinfo=timezone.utc)
        events = cal.get_events_in_range(start, end)

        # March 2025 should have: 1 NFP + 1 FOMC + 1 CPI = 3 events
        assert len(events) >= 2  # At minimum NFP + CPI


class TestIsNearEvent:
    def test_within_window_returns_true(self):
        from src.macro.econ_calendar import EconCalendar

        cal = EconCalendar()
        # Get NFP date for Jan 2025 (Jan 3)
        nfp = cal.get_nfp_dates(2025)[0]
        nfp_time = nfp.replace(hour=13, minute=30)  # 8:30 AM ET = 13:30 UTC

        check_time = nfp_time - timedelta(hours=1)  # 1 hour before
        is_near, event = cal.is_near_event(check_time, hours_before=4, hours_after=2)
        assert is_near is True
        assert event is not None

    def test_outside_window_returns_false(self):
        from src.macro.econ_calendar import EconCalendar

        cal = EconCalendar()
        # A random Tuesday far from any event
        check_time = datetime(2025, 2, 18, 10, 0, tzinfo=timezone.utc)
        is_near, event = cal.is_near_event(check_time, hours_before=4, hours_after=2)
        assert is_near is False
        assert event is None
