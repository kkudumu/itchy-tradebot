"""Scan interval scheduling aligned to candle closes.

The scheduler waits until the next N-minute candle close boundary, then
invokes the callback.  This ensures the engine always evaluates a fully
formed candle rather than an in-progress one.

Market hours:
- Forex (and XAU/USD) trades Sunday 22:00 UTC through Friday 22:00 UTC.
- Saturday and most of Sunday are considered closed.
- The scheduler skips scan cycles when the market is closed.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class ScanScheduler:
    """Schedule scan cycles aligned to candle-close boundaries.

    Parameters
    ----------
    interval_minutes:
        Candle interval in minutes.  Default: 5 (the primary entry TF).
    """

    def __init__(self, interval_minutes: int = 5) -> None:
        if interval_minutes <= 0:
            raise ValueError(f"interval_minutes must be positive, got {interval_minutes}")
        self.interval = interval_minutes

    # ------------------------------------------------------------------
    # Timing helpers
    # ------------------------------------------------------------------

    def wait_for_next_close(self) -> datetime:
        """Block until the next aligned candle-close boundary.

        The boundary is always aligned to the clock:
        - 5M: 08:00, 08:05, 08:10, …
        - 15M: 08:00, 08:15, 08:30, …

        Returns the UTC datetime of the boundary that was waited for.
        """
        now = datetime.now(timezone.utc)
        next_close = self._next_close_time(now)
        wait_seconds = (next_close - now).total_seconds()

        if wait_seconds > 0:
            logger.debug(
                "Waiting %.1fs for next %dM close at %s",
                wait_seconds,
                self.interval,
                next_close.strftime("%H:%M:%S"),
            )
            time.sleep(wait_seconds)

        return next_close

    def _next_close_time(self, now: datetime) -> datetime:
        """Calculate the next aligned close time after ``now``.

        Alignment is to the top of the minute grid (minute % interval == 0).
        A small buffer (1 second) ensures we do not fire just before the close.
        """
        minute = now.minute
        second = now.second

        # Minutes until the next aligned boundary
        minutes_past_boundary = minute % self.interval
        minutes_to_next = self.interval - minutes_past_boundary

        # If we are exactly on the boundary and no seconds have elapsed,
        # the candle has just closed — fire immediately.
        if minutes_past_boundary == 0 and second == 0:
            return now.replace(microsecond=0)

        # Otherwise, wait for the next boundary + 1 second buffer
        next_dt = (now + timedelta(minutes=minutes_to_next)).replace(
            second=1, microsecond=0
        )
        return next_dt

    # ------------------------------------------------------------------
    # Market hours
    # ------------------------------------------------------------------

    def is_market_open(self, dt: Optional[datetime] = None) -> bool:
        """Return True when forex/spot-gold is open.

        Forex is open from Sunday 22:00 UTC through Friday 22:00 UTC.
        Saturday and the early hours of Sunday are considered closed.

        Parameters
        ----------
        dt:
            Datetime to evaluate.  Defaults to ``datetime.now(UTC)``.
        """
        if dt is None:
            dt = datetime.now(timezone.utc)

        weekday = dt.weekday()  # 0=Monday … 6=Sunday
        hour = dt.hour

        # Saturday: closed all day
        if weekday == 5:
            return False

        # Sunday: open from 22:00 UTC onwards
        if weekday == 6:
            return hour >= 22

        # Friday: closed from 22:00 UTC onwards
        if weekday == 4:
            return hour < 22

        # Monday–Thursday: always open
        return True

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_loop(
        self,
        callback: Callable[[], None],
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        """Run ``callback`` at each aligned interval until stopped.

        Parameters
        ----------
        callback:
            Zero-argument callable invoked on each candle close.  Any
            exception raised by the callback is caught and logged; the
            loop continues after an error to maintain uptime.
        stop_event:
            A :class:`threading.Event` that — when set — causes the loop
            to exit cleanly after the current iteration completes.
        """
        logger.info(
            "ScanScheduler starting — interval=%dM, callback=%s",
            self.interval,
            getattr(callback, "__name__", repr(callback)),
        )

        while True:
            # Check stop signal before waiting
            if stop_event is not None and stop_event.is_set():
                logger.info("ScanScheduler stop event received — exiting loop")
                break

            # Skip market-closed periods
            if not self.is_market_open():
                next_open = self._seconds_to_market_open()
                logger.info(
                    "Market closed — sleeping %ds until next open",
                    next_open,
                )
                # Sleep in short chunks so we can respond to stop events
                self._interruptible_sleep(next_open, stop_event)
                continue

            # Wait for the next candle close
            try:
                self.wait_for_next_close()
            except Exception as exc:  # noqa: BLE001
                logger.error("Error waiting for next close: %s", exc)
                time.sleep(10)
                continue

            # Check stop signal again after sleeping
            if stop_event is not None and stop_event.is_set():
                logger.info("ScanScheduler stop event received — exiting loop")
                break

            # Execute the scan callback
            try:
                callback()
            except Exception as exc:  # noqa: BLE001
                logger.exception("Callback raised an exception: %s", exc)
                # Continue the loop — one bad scan should not halt the agent

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _seconds_to_market_open(self) -> int:
        """Return seconds until the market next opens (max 24 hours)."""
        now = datetime.now(timezone.utc)
        weekday = now.weekday()

        # Sunday before 22:00 — open in (22 - hour) hours
        if weekday == 6 and now.hour < 22:
            target = now.replace(hour=22, minute=0, second=0, microsecond=0)
            return max(1, int((target - now).total_seconds()))

        # Saturday — market opens Sunday 22:00
        if weekday == 5:
            days_ahead = 1  # Sunday
            target = (now + timedelta(days=days_ahead)).replace(
                hour=22, minute=0, second=0, microsecond=0
            )
            return max(1, int((target - now).total_seconds()))

        # Friday after 22:00 — market opens Sunday 22:00
        if weekday == 4 and now.hour >= 22:
            days_ahead = 2  # Sunday
            target = (now + timedelta(days=days_ahead)).replace(
                hour=22, minute=0, second=0, microsecond=0
            )
            return max(1, int((target - now).total_seconds()))

        # Should not reach here if is_market_open() is called correctly
        return self.interval * 60

    @staticmethod
    def _interruptible_sleep(seconds: int, stop_event: Optional[threading.Event]) -> None:
        """Sleep for ``seconds`` but wake early if ``stop_event`` is set."""
        chunk = 30  # check every 30 seconds
        elapsed = 0
        while elapsed < seconds:
            if stop_event is not None and stop_event.is_set():
                return
            sleep_time = min(chunk, seconds - elapsed)
            time.sleep(sleep_time)
            elapsed += sleep_time
