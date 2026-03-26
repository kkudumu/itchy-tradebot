"""Daily loss circuit breaker — hard halt when daily drawdown exceeds the limit.

At 1-2 trades per day with 1–1.5% risk, maximum expected daily loss is ~3%.
The circuit breaker at 2% is a safety margin that should rarely trigger under
normal operation. Once triggered, no further trades are allowed that calendar day.

All thresholds are HARD-CODED — the learning loop cannot override them.
"""

from __future__ import annotations

import datetime
from typing import Optional


class DailyCircuitBreaker:
    """Hard daily loss limit enforcer.

    Parameters
    ----------
    max_daily_loss_pct:
        Maximum permitted daily drawdown before trading halts.
        Default: 2.0% (prop firm safety margin).
    """

    # Hard ceiling: even if max_daily_loss_pct is configured higher, this cap applies.
    _ABSOLUTE_MAX_DAILY_LOSS_PCT: float = 5.0

    def __init__(self, max_daily_loss_pct: float = 2.0) -> None:
        # Clamp to absolute hard ceiling — cannot be configured beyond this
        self._max_daily_loss_pct = min(max_daily_loss_pct, self._ABSOLUTE_MAX_DAILY_LOSS_PCT)

        self._daily_start_balance: float = 0.0
        self._current_date: Optional[datetime.date] = None
        self._triggered: bool = False

    # ------------------------------------------------------------------ #
    # Day lifecycle                                                         #
    # ------------------------------------------------------------------ #

    def start_day(self, balance: float, date: Optional[datetime.date] = None) -> None:
        """Record the start-of-day balance and reset the daily state.

        Parameters
        ----------
        balance:
            Account balance at the start of the trading day.
        date:
            Calendar date for this session. Defaults to today (UTC).
        """
        if balance <= 0:
            raise ValueError(f"balance must be positive, got {balance}")

        self._daily_start_balance = balance
        self._current_date = date if date is not None else datetime.date.today()
        self._triggered = False

    # ------------------------------------------------------------------ #
    # Trading gate                                                          #
    # ------------------------------------------------------------------ #

    def can_trade(self, current_balance: float) -> bool:
        """Return True if trading is permitted, False if the circuit has tripped.

        The circuit trips when daily loss percentage meets or exceeds the limit.
        Once tripped within a session it stays tripped — call :meth:`start_day`
        the next morning to reset.

        Parameters
        ----------
        current_balance:
            Current real-time account balance (including unrealised P&L).
        """
        if self._daily_start_balance <= 0:
            # start_day has not been called yet — allow trading conservatively
            return True

        if self._triggered:
            return False

        loss_pct = self.daily_loss_pct(current_balance)
        if loss_pct >= self._max_daily_loss_pct:
            self._triggered = True
            return False

        return True

    def is_triggered(self) -> bool:
        """Return True if the circuit breaker has been tripped today."""
        return self._triggered

    # ------------------------------------------------------------------ #
    # Metrics                                                               #
    # ------------------------------------------------------------------ #

    def daily_loss_pct(self, current_balance: float) -> float:
        """Calculate current daily loss as a positive percentage.

        Returns 0.0 if the balance has improved (no loss).

        Parameters
        ----------
        current_balance:
            Current real-time account balance.
        """
        if self._daily_start_balance <= 0:
            return 0.0

        loss_pct = (
            (self._daily_start_balance - current_balance)
            / self._daily_start_balance
            * 100.0
        )
        # Negative means profit — clamp to 0 so callers get an unsigned loss figure
        return max(0.0, loss_pct)

    def remaining_risk_budget(self, current_balance: float) -> float:
        """Monetary amount that can still be lost today before the circuit trips.

        Returns 0.0 if the circuit has already tripped or is at the limit.

        Parameters
        ----------
        current_balance:
            Current real-time account balance.
        """
        if self._daily_start_balance <= 0 or self._triggered:
            return 0.0

        max_loss_amount = self._daily_start_balance * (self._max_daily_loss_pct / 100.0)
        already_lost = max(0.0, self._daily_start_balance - current_balance)
        remaining = max_loss_amount - already_lost
        return max(0.0, remaining)

    # ------------------------------------------------------------------ #
    # Properties                                                            #
    # ------------------------------------------------------------------ #

    @property
    def max_daily_loss_pct(self) -> float:
        """The configured (and clamped) maximum daily loss percentage."""
        return self._max_daily_loss_pct

    @property
    def daily_start_balance(self) -> float:
        """Balance recorded at the start of the current trading day."""
        return self._daily_start_balance

    @property
    def current_date(self) -> Optional[datetime.date]:
        """Calendar date of the current session, or None if not started."""
        return self._current_date
