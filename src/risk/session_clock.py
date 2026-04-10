"""Profile-aware day-boundary detection for prop firm daily-loss rules.

Different brokers / prop firms define "the trading day" differently:

* **The5ers / forex**: midnight UTC rollover
* **TopstepX / CME futures**: 5pm America/Chicago rollover

The ``SessionClock`` wraps this detail so every component that needs to
answer "what trading day is this timestamp on?" (DailyCircuitBreaker,
TopstepCombineTracker, trading-hours filter, telemetry binning) can
agree without each carrying its own timezone math.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

try:
    from zoneinfo import ZoneInfo

    def _astimezone(ts: datetime, tz_name: str) -> datetime:
        return ts.astimezone(ZoneInfo(tz_name))

    def _localize_naive(ts: datetime, tz_name: str) -> datetime:
        return ts.replace(tzinfo=ZoneInfo(tz_name))

except ImportError:  # pragma: no cover
    import pytz  # type: ignore[import-not-found]

    def _astimezone(ts: datetime, tz_name: str) -> datetime:
        tz = pytz.timezone(tz_name)
        if ts.tzinfo is None:
            return tz.localize(ts)
        return ts.astimezone(tz)

    def _localize_naive(ts: datetime, tz_name: str) -> datetime:
        tz = pytz.timezone(tz_name)
        return tz.localize(ts.replace(tzinfo=None))


class SessionClock:
    """Determines trading-day boundaries for a given reset hour + timezone.

    The *trading day* for any timestamp is computed by:

    1. Converting the timestamp (assumed UTC if naive) into the
       configured local timezone (``reset_tz``).
    2. Subtracting ``reset_hour`` hours from that localized time.
    3. Taking the date portion.

    This handles DST transitions correctly because the subtraction
    happens AFTER the timezone conversion — zoneinfo takes care of the
    spring-forward / fall-back shift.

    Example — TopstepX 5pm CT rollover::

        clock = SessionClock(reset_hour_local=17, reset_tz="America/Chicago")
        clock.trading_day(datetime(2026, 1, 2, 22, 0, tzinfo=timezone.utc))
        # -> 4pm CT on Jan 2 → trading day Jan 2

        clock.trading_day(datetime(2026, 1, 2, 23, 30, tzinfo=timezone.utc))
        # -> 5:30pm CT on Jan 2 → trading day Jan 3

    Example — the5ers / forex midnight UTC::

        clock = SessionClock(reset_hour_local=0, reset_tz="UTC")
        clock.trading_day(datetime(2026, 1, 2, 23, 59, tzinfo=timezone.utc))
        # -> Jan 2
    """

    def __init__(self, reset_hour_local: int, reset_tz: str = "UTC") -> None:
        if not (0 <= reset_hour_local < 24):
            raise ValueError(
                f"reset_hour_local must be in [0, 24); got {reset_hour_local}"
            )
        self._reset_hour_local = reset_hour_local
        self._reset_tz = reset_tz

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def reset_hour_local(self) -> int:
        return self._reset_hour_local

    @property
    def reset_tz(self) -> str:
        return self._reset_tz

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def trading_day(self, ts: datetime) -> date:
        """Return the trading day *ts* belongs to.

        Accepts either tz-aware or naive datetimes; naive values are
        treated as UTC to match the rest of the codebase.
        """
        ts = self._ensure_utc(ts)
        local = _astimezone(ts, self._reset_tz)
        shifted = local - timedelta(hours=self._reset_hour_local)
        return shifted.date()

    def is_new_day(
        self,
        prev_ts: datetime | None,
        current_ts: datetime,
    ) -> bool:
        """Return True if *current_ts* is on a different trading day than *prev_ts*.

        A ``None`` previous timestamp always returns True — the first
        call after construction or reset signals "new day" so callers
        can initialise their daily state.
        """
        if prev_ts is None:
            return True
        return self.trading_day(prev_ts) != self.trading_day(current_ts)

    def day_open_ts(self, trading_day: date) -> datetime:
        """Return the UTC timestamp at which *trading_day* opens.

        Trading day X is labelled by the calendar date on which the
        session **begins** in ``reset_tz`` local time:

        * Forex (reset_hour=0, UTC): the 2026-01-15 trading day opens at
          ``2026-01-15 00:00 UTC``.
        * Futures (reset_hour=17, America/Chicago): the 2026-01-15
          trading day opens at ``2026-01-15 17:00 CT`` and runs through
          ``2026-01-16 16:59 CT``.
        """
        naive = datetime.combine(trading_day, time(self._reset_hour_local, 0))
        localized = _localize_naive(naive, self._reset_tz)
        return localized.astimezone(timezone.utc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_utc(ts: datetime) -> datetime:
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts

    def __repr__(self) -> str:
        return (
            f"SessionClock(reset_hour_local={self._reset_hour_local}, "
            f"reset_tz={self._reset_tz!r})"
        )
