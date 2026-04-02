"""Deterministic economic calendar for high-impact US events.

Generates dates for NFP (Non-Farm Payrolls), FOMC (Federal Reserve
decisions), and CPI (Consumer Price Index) releases using calendar
rules rather than external API calls.

Event timing (all UTC):
    NFP:  First Friday of month, 13:30 UTC (8:30 AM ET)
    FOMC: See schedule below, 19:00 UTC (2:00 PM ET)
    CPI:  ~12th of month (adjusted for weekends), 13:30 UTC (8:30 AM ET)

FOMC schedule: 8 meetings per year. The months are fixed; specific
dates vary by year but follow a pattern of Tue-Wed meetings where
the decision is announced on Wednesday.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Reuse NewsEvent from the edges.news_filter module
from src.edges.news_filter import NewsEvent


# FOMC meeting months (decision day months)
_FOMC_MONTHS = [1, 3, 5, 6, 7, 9, 11, 12]

# FOMC approximate decision day within month (3rd Wednesday as default)
# For precise dates, a hardcoded schedule per year is used.
_FOMC_SCHEDULES: Dict[int, List[Tuple[int, int]]] = {
    # (month, day) for the Wednesday decision announcement
    2024: [(1, 31), (3, 20), (5, 1), (6, 12), (7, 31), (9, 18), (11, 7), (12, 18)],
    2025: [(1, 29), (3, 19), (5, 7), (6, 18), (7, 30), (9, 17), (11, 5), (12, 17)],
    2026: [(1, 28), (3, 18), (5, 6), (6, 17), (7, 29), (9, 16), (11, 4), (12, 16)],
}


def _first_friday(year: int, month: int) -> datetime:
    """Return the first Friday of the given month as a UTC datetime."""
    dt = datetime(year, month, 1, tzinfo=timezone.utc)
    # weekday(): 0=Mon, 4=Fri
    days_until_friday = (4 - dt.weekday()) % 7
    return dt + timedelta(days=days_until_friday)


def _nearest_weekday(year: int, month: int, target_day: int) -> datetime:
    """Return the nearest weekday to target_day in the given month.

    If target_day falls on Saturday, use Friday (target_day - 1).
    If target_day falls on Sunday, use Monday (target_day + 1).
    """
    dt = datetime(year, month, target_day, tzinfo=timezone.utc)
    if dt.weekday() == 5:  # Saturday -> Friday
        dt -= timedelta(days=1)
    elif dt.weekday() == 6:  # Sunday -> Monday
        dt += timedelta(days=1)
    return dt


def _third_wednesday(year: int, month: int) -> datetime:
    """Return the third Wednesday of the given month."""
    dt = datetime(year, month, 1, tzinfo=timezone.utc)
    # Find first Wednesday
    days_until_wed = (2 - dt.weekday()) % 7
    first_wed = dt + timedelta(days=days_until_wed)
    # Third Wednesday = first + 14 days
    return first_wed + timedelta(days=14)


class EconCalendar:
    """Deterministic generator for high-impact US economic event dates.

    Generates NFP, FOMC, and CPI dates for a given year using calendar
    rules. All events are tagged with "red" impact level for use with
    the existing NewsFilter/NewsEvent infrastructure.
    """

    def get_nfp_dates(self, year: int) -> List[datetime]:
        """Get all 12 NFP release dates for a year.

        NFP = first Friday of each month at 13:30 UTC.
        """
        dates = []
        for month in range(1, 13):
            nfp = _first_friday(year, month)
            nfp = nfp.replace(hour=13, minute=30)
            dates.append(nfp)
        return dates

    def get_fomc_dates(self, year: int) -> List[datetime]:
        """Get all 8 FOMC decision dates for a year.

        Uses hardcoded schedules for known years, falls back to
        third-Wednesday heuristic for other years. Time: 19:00 UTC.
        """
        if year in _FOMC_SCHEDULES:
            dates = []
            for month, day in _FOMC_SCHEDULES[year]:
                dt = datetime(year, month, day, 19, 0, tzinfo=timezone.utc)
                dates.append(dt)
            return dates

        # Fallback: third Wednesday of each FOMC month
        dates = []
        for month in _FOMC_MONTHS:
            dt = _third_wednesday(year, month)
            dt = dt.replace(hour=19, minute=0)
            dates.append(dt)
        return dates

    def get_cpi_dates(self, year: int) -> List[datetime]:
        """Get all 12 CPI release dates for a year.

        CPI ~ 12th of month (adjusted for weekends) at 13:30 UTC.
        """
        dates = []
        for month in range(1, 13):
            cpi = _nearest_weekday(year, month, 12)
            cpi = cpi.replace(hour=13, minute=30)
            dates.append(cpi)
        return dates

    def get_all_events(self, year: int) -> List[NewsEvent]:
        """Get all high-impact events for a year, sorted chronologically.

        Returns NewsEvent objects compatible with the existing
        src.edges.news_filter.NewsFilter infrastructure.
        """
        events: List[NewsEvent] = []

        for dt in self.get_nfp_dates(year):
            events.append(NewsEvent(
                timestamp=dt,
                title=f"US Non-Farm Payrolls ({dt.strftime('%b %Y')})",
                impact="red",
            ))

        for dt in self.get_fomc_dates(year):
            events.append(NewsEvent(
                timestamp=dt,
                title=f"FOMC Rate Decision ({dt.strftime('%b %Y')})",
                impact="red",
            ))

        for dt in self.get_cpi_dates(year):
            events.append(NewsEvent(
                timestamp=dt,
                title=f"US CPI ({dt.strftime('%b %Y')})",
                impact="red",
            ))

        events.sort(key=lambda e: e.timestamp)
        logger.info("Generated %d high-impact events for %d", len(events), year)
        return events

    def get_events_in_range(
        self,
        start: datetime,
        end: datetime,
    ) -> List[NewsEvent]:
        """Get events within a date range (may span year boundaries).

        Parameters
        ----------
        start: Range start (inclusive).
        end: Range end (inclusive).
        """
        years = set(range(start.year, end.year + 1))
        all_events = []
        for year in years:
            all_events.extend(self.get_all_events(year))

        return [e for e in all_events if start <= e.timestamp <= end]

    def is_near_event(
        self,
        timestamp: datetime,
        hours_before: int = 4,
        hours_after: int = 2,
    ) -> Tuple[bool, Optional[NewsEvent]]:
        """Check if a timestamp is near any high-impact event.

        Parameters
        ----------
        timestamp: The time to check (UTC).
        hours_before: Blackout window before event.
        hours_after: Blackout window after event.

        Returns
        -------
        (is_near, nearest_event): Whether within window, and which event.
        """
        # Check events for the year of the timestamp
        events = self.get_all_events(timestamp.year)

        for event in events:
            window_start = event.timestamp - timedelta(hours=hours_before)
            window_end = event.timestamp + timedelta(hours=hours_after)
            if window_start <= timestamp <= window_end:
                return True, event

        return False, None
