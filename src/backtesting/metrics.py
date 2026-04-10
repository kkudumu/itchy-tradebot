"""
Performance metrics and prop firm constraint tracking for backtests.

PerformanceMetrics calculates standard quant metrics from a trade list and
equity curve.  PropFirmTracker monitors The5ers-style challenge constraints
(profit target, daily drawdown, total drawdown, time limit) on a per-bar
basis throughout the simulation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Trading days per year used for annualisation
_TRADING_DAYS_PER_YEAR: float = 252.0


# =============================================================================
# PropFirmTrackerProtocol — structural interface for pluggable trackers
# =============================================================================


@runtime_checkable
class PropFirmTrackerProtocol(Protocol):
    """Common interface for prop firm trackers.

    Implementations: ``PropFirmTracker`` (legacy single-phase pct),
    ``MultiPhasePropFirmTracker`` (the5ers 2-step pct),
    ``TopstepCombineTracker`` (TopstepX dollar-based trailing).

    The engine holds a single reference of this type and calls
    ``initialise`` / ``update`` / ``check_pass`` / ``to_dict`` without
    knowing which concrete implementation it's dealing with.
    """

    def initialise(self, initial_balance: float, first_ts: datetime) -> None:
        ...

    def update(self, ts: datetime, balance: float) -> None:
        ...

    def check_pass(self) -> Any:  # returns PropFirmStatus or dict
        ...

    def to_dict(self) -> Dict[str, Any]:
        ...


# =============================================================================
# PropFirmStatus
# =============================================================================

@dataclass
class PropFirmStatus:
    """Snapshot of prop firm challenge state at any point during the backtest."""

    status: str
    """One of: 'passed', 'failed_daily_dd', 'failed_total_dd',
    'failed_timeout', 'ongoing'."""

    profit_pct: float
    """Current profit as a percentage of the initial balance."""

    max_daily_dd_pct: float
    """Worst daily drawdown seen during the simulation (positive number)."""

    max_total_dd_pct: float
    """Worst peak-to-trough drawdown from the initial balance (positive number)."""

    days_elapsed: int
    """Calendar days elapsed since the challenge start date."""

    details: Dict = field(default_factory=dict)
    """Per-day breakdown and summary statistics."""


# =============================================================================
# PropFirmTracker
# =============================================================================

class PropFirmTracker:
    """Track prop firm challenge constraints bar-by-bar during a backtest.

    Parameters
    ----------
    profit_target_pct:
        Profit target as a percentage of initial balance.  Default: 8.0.
    max_daily_dd_pct:
        Maximum permitted daily drawdown as a percentage of the day's
        opening balance.  Exceeding this fails the challenge.  Default: 5.0.
    max_total_dd_pct:
        Maximum permitted total drawdown from initial balance.
        Exceeding this fails the challenge.  Default: 10.0.
    time_limit_days:
        Calendar days allowed to reach the profit target.  If elapsed days
        exceed this without passing, the challenge is a timeout failure.
        Default: 30.
    """

    def __init__(
        self,
        profit_target_pct: float = 8.0,
        max_daily_dd_pct: float = 5.0,
        max_total_dd_pct: float = 10.0,
        time_limit_days: int = 30,
    ) -> None:
        self._profit_target_pct = profit_target_pct
        self._max_daily_dd_pct = max_daily_dd_pct
        self._max_total_dd_pct = max_total_dd_pct
        self._time_limit_days = time_limit_days

        # Internal state
        self._initial_balance: float = 0.0
        self._start_date: Optional[datetime] = None

        # Daily tracking — keyed by date string "YYYY-MM-DD"
        self._daily_open_balance: Dict[str, float] = {}
        self._daily_close_balance: Dict[str, float] = {}

        # Running worst-case records
        self._worst_daily_dd_pct: float = 0.0
        self._worst_total_dd_pct: float = 0.0

        # Per-bar equity log for daily_dd_series()
        self._equity_by_date: Dict[str, List[float]] = {}

        self._status: str = "ongoing"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialise(self, initial_balance: float, start_date: datetime) -> None:
        """Set the challenge starting state.  Must be called before update()."""
        if initial_balance <= 0:
            raise ValueError(f"initial_balance must be positive, got {initial_balance}")
        self._initial_balance = initial_balance
        self._start_date = start_date

    def update(self, timestamp: datetime, balance: float) -> None:
        """Record the balance at a given bar timestamp.

        Should be called on every 5M bar.  Internally aggregates to daily
        granularity for drawdown tracking.

        Parameters
        ----------
        timestamp:
            UTC bar timestamp.
        balance:
            Account balance (realised equity) at this bar.
        """
        if self._initial_balance <= 0:
            return  # initialise() not yet called

        date_key = timestamp.strftime("%Y-%m-%d")

        # Record start-of-day balance on first bar of each day
        if date_key not in self._daily_open_balance:
            self._daily_open_balance[date_key] = balance

        # Always update the running close balance for the day
        self._daily_close_balance[date_key] = balance

        # Maintain per-date equity list for series output
        if date_key not in self._equity_by_date:
            self._equity_by_date[date_key] = []
        self._equity_by_date[date_key].append(balance)

        # Compute daily drawdown from this day's open
        day_open = self._daily_open_balance[date_key]
        if day_open > 0:
            daily_dd = (day_open - balance) / day_open * 100.0
            daily_dd = max(0.0, daily_dd)
            if daily_dd > self._worst_daily_dd_pct:
                self._worst_daily_dd_pct = daily_dd

        # Total drawdown from initial balance
        total_dd = (self._initial_balance - balance) / self._initial_balance * 100.0
        total_dd = max(0.0, total_dd)
        if total_dd > self._worst_total_dd_pct:
            self._worst_total_dd_pct = total_dd

        # Update challenge status
        self._update_status(balance, timestamp)

    def check_pass(self) -> PropFirmStatus:
        """Return the current challenge status.

        Returns
        -------
        PropFirmStatus with the final verdict and supporting statistics.
        """
        days_elapsed = self._days_elapsed()
        profit_pct = self._profit_pct()

        # Per-day drawdown breakdown
        daily_dd_details: Dict[str, float] = {}
        for date_key, open_bal in self._daily_open_balance.items():
            close_bal = self._daily_close_balance.get(date_key, open_bal)
            if open_bal > 0:
                dd = max(0.0, (open_bal - close_bal) / open_bal * 100.0)
                daily_dd_details[date_key] = round(dd, 4)

        return PropFirmStatus(
            status=self._status,
            profit_pct=round(profit_pct, 4),
            max_daily_dd_pct=round(self._worst_daily_dd_pct, 4),
            max_total_dd_pct=round(self._worst_total_dd_pct, 4),
            days_elapsed=days_elapsed,
            details={
                "profit_target_pct": self._profit_target_pct,
                "allowed_daily_dd_pct": self._max_daily_dd_pct,
                "allowed_total_dd_pct": self._max_total_dd_pct,
                "time_limit_days": self._time_limit_days,
                "daily_drawdowns": daily_dd_details,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a dict snapshot suitable for JSON / parquet serialization.

        Mirrors the :class:`PropFirmTrackerProtocol` so the engine can
        call ``to_dict()`` uniformly regardless of which concrete
        tracker is active.
        """
        status = self.check_pass()
        return {
            "style": "legacy_single_phase_pct",
            "status": status.status,
            "profit_pct": status.profit_pct,
            "max_daily_dd_pct": status.max_daily_dd_pct,
            "max_total_dd_pct": status.max_total_dd_pct,
            "days_elapsed": status.days_elapsed,
            "details": status.details,
        }

    def daily_dd_series(self) -> pd.Series:
        """Return a Series of daily drawdowns indexed by date string.

        Each value is the maximum intraday drawdown from that day's opening
        balance, expressed as a positive percentage.
        """
        records: Dict[str, float] = {}
        for date_key, open_bal in self._daily_open_balance.items():
            day_equities = self._equity_by_date.get(date_key, [open_bal])
            min_equity = min(day_equities)
            if open_bal > 0:
                dd = max(0.0, (open_bal - min_equity) / open_bal * 100.0)
            else:
                dd = 0.0
            records[date_key] = round(dd, 4)

        series = pd.Series(records, name="daily_dd_pct")
        series.index = pd.to_datetime(series.index)
        return series.sort_index()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _profit_pct(self) -> float:
        """Current profit as percentage of initial balance."""
        if not self._daily_close_balance or self._initial_balance <= 0:
            return 0.0
        latest_balance = list(self._daily_close_balance.values())[-1]
        return (latest_balance - self._initial_balance) / self._initial_balance * 100.0

    def _days_elapsed(self) -> int:
        """Calendar days elapsed since the challenge start."""
        if not self._start_date or not self._daily_open_balance:
            return 0
        last_date_str = sorted(self._daily_open_balance.keys())[-1]
        last_date = datetime.strptime(last_date_str, "%Y-%m-%d")

        # Normalise both sides to naive UTC dates for subtraction
        start = self._start_date
        if hasattr(start, "tzinfo") and start.tzinfo is not None:
            start = start.replace(tzinfo=None)

        delta = last_date - start
        return max(0, delta.days)

    def _update_status(self, balance: float, timestamp: datetime) -> None:
        """Determine and cache the current challenge status."""
        if self._status in ("passed", "failed_daily_dd", "failed_total_dd", "failed_timeout"):
            # Terminal states are sticky — once failed/passed, keep the verdict.
            return

        # Check failure conditions first (they take precedence over passing)
        if self._worst_total_dd_pct >= self._max_total_dd_pct:
            self._status = "failed_total_dd"
            return

        if self._worst_daily_dd_pct >= self._max_daily_dd_pct:
            self._status = "failed_daily_dd"
            return

        days_elapsed = self._days_elapsed()
        if days_elapsed > self._time_limit_days:
            self._status = "failed_timeout"
            return

        # Check pass condition
        profit = self._profit_pct()
        if profit >= self._profit_target_pct:
            self._status = "passed"
            return

        self._status = "ongoing"


# =============================================================================
# MultiPhasePropFirmTracker
# =============================================================================


class MultiPhasePropFirmTracker:
    """Track a multi-phase prop firm challenge (e.g. The5ers 2-Step High Stakes).

    Models the full lifecycle: Phase 1 (Evaluation) -> Phase 2 (Approval) ->
    Funded.  Each phase has independent drawdown tracking that resets on
    transition.  The daily DD formula follows The5ers rules: the reference is
    MAX(prev_day_closing_equity, prev_day_closing_balance).

    The existing single-phase ``PropFirmTracker`` is preserved for backwards
    compatibility.

    Parameters
    ----------
    account_size:
        Nominal account size (balance resets to this at each phase).
    phase_1_profit_target_pct:
        Phase 1 profit target as a percentage of phase balance.
    phase_1_max_loss_pct:
        Phase 1 maximum total drawdown percentage.
    phase_1_daily_loss_pct:
        Phase 1 maximum daily drawdown percentage.
    phase_2_profit_target_pct:
        Phase 2 profit target as a percentage of phase balance.
    phase_2_max_loss_pct:
        Phase 2 maximum total drawdown percentage.
    phase_2_daily_loss_pct:
        Phase 2 maximum daily drawdown percentage.
    funded_monthly_target_pct:
        Funded phase monthly target percentage (informational).
    funded_max_loss_pct:
        Funded phase maximum total drawdown percentage.
    funded_daily_loss_pct:
        Funded phase maximum daily drawdown percentage.
    """

    # Terminal states -- once entered, status never changes
    _TERMINAL_STATES = frozenset({
        "failed_phase_1", "failed_phase_2", "funded_bust",
    })

    def __init__(
        self,
        account_size: float = 10_000.0,
        phase_1_profit_target_pct: float = 8.0,
        phase_1_max_loss_pct: float = 10.0,
        phase_1_daily_loss_pct: float = 5.0,
        phase_2_profit_target_pct: float = 5.0,
        phase_2_max_loss_pct: float = 10.0,
        phase_2_daily_loss_pct: float = 5.0,
        funded_monthly_target_pct: float = 10.0,
        funded_max_loss_pct: float = 10.0,
        funded_daily_loss_pct: float = 5.0,
    ) -> None:
        self._account_size = account_size

        # Phase 1 config
        self._p1_target = phase_1_profit_target_pct
        self._p1_max_loss = phase_1_max_loss_pct
        self._p1_daily_loss = phase_1_daily_loss_pct

        # Phase 2 config
        self._p2_target = phase_2_profit_target_pct
        self._p2_max_loss = phase_2_max_loss_pct
        self._p2_daily_loss = phase_2_daily_loss_pct

        # Funded config
        self._funded_monthly_target = funded_monthly_target_pct
        self._funded_max_loss = funded_max_loss_pct
        self._funded_daily_loss = funded_daily_loss_pct

        # Runtime state -- populated by initialise()
        self._phase: str = ""
        self._phase_balance: float = 0.0
        self._phase_hwm: float = 0.0  # high-water mark for total DD
        self._max_total_dd_pct: float = 0.0
        self._failure_reason: str = ""
        self._phase_1_passed: bool = False

        # Daily DD tracking
        self._prev_day_close_equity: float = 0.0
        self._prev_day_close_balance: float = 0.0
        self._current_day_date: Optional[str] = None  # "YYYY-MM-DD"

        # Funded monthly tracking
        self._funded_monthly_returns: List[float] = []
        self._funded_month_start_equity: float = 0.0
        self._funded_current_month: Optional[str] = None  # "YYYY-MM"

        # Latest equity (for status reporting)
        self._latest_equity: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialise(self, starting_balance: float, start_time: datetime) -> None:
        """Set up Phase 1 with the given starting balance and time."""
        self._phase = "phase_1_active"
        self._phase_balance = starting_balance
        self._phase_hwm = starting_balance
        self._latest_equity = starting_balance
        self._max_total_dd_pct = 0.0
        self._failure_reason = ""
        self._phase_1_passed = False

        # Daily DD: initialise "previous day close" to starting balance
        self._prev_day_close_equity = starting_balance
        self._prev_day_close_balance = starting_balance
        self._current_day_date = start_time.strftime("%Y-%m-%d")

        # Funded tracking reset
        self._funded_monthly_returns = []
        self._funded_month_start_equity = 0.0
        self._funded_current_month = None

    def update(self, timestamp: datetime, equity: float) -> None:
        """Process a bar update -- handles phase logic, DD checks, transitions.

        Parameters
        ----------
        timestamp:
            UTC timestamp of the bar.
        equity:
            Current account equity at this bar.
        """
        if self._phase in self._TERMINAL_STATES:
            return  # sticky terminal state

        if not self._phase:
            return  # initialise() not yet called

        self._latest_equity = equity
        date_key = timestamp.strftime("%Y-%m-%d")

        # --- Day boundary detection ---
        if date_key != self._current_day_date:
            # A new day has started.
            # The *previous* day's closing values become the DD reference.
            # _prev_day_close_equity was updated each bar, so it holds
            # the last equity of the old day already.  But we also need
            # to snapshot the balance for the daily DD formula.
            # Balance for DD purposes = phase_balance (doesn't change
            # intra-day in our model), so _prev_day_close_balance is
            # kept as the phase_balance at end of old day.
            self._prev_day_close_balance = self._phase_balance
            # _prev_day_close_equity already holds last bar's equity
            # (set at end of each bar below).
            self._current_day_date = date_key

        # --- Total DD check (from phase balance) ---
        # Check total DD first: when both limits are breached on the same
        # bar the total DD failure takes precedence.
        if equity > self._phase_hwm:
            self._phase_hwm = equity
        total_dd_pct = (self._phase_balance - equity) / self._phase_balance * 100.0
        total_dd_pct = max(0.0, total_dd_pct)
        if total_dd_pct > self._max_total_dd_pct:
            self._max_total_dd_pct = total_dd_pct

        total_dd_limit = self._current_total_loss_limit()
        if total_dd_pct > total_dd_limit:
            self._fail("total_dd")
            self._prev_day_close_equity = equity
            return

        # --- Daily DD check (The5ers formula) ---
        daily_dd_ref = max(self._prev_day_close_equity, self._prev_day_close_balance)
        if daily_dd_ref > 0:
            daily_dd_frac = (daily_dd_ref - equity) / daily_dd_ref
            daily_dd_limit = self._current_daily_loss_limit() / 100.0
            if daily_dd_frac > daily_dd_limit:
                self._fail("daily_dd")
                self._prev_day_close_equity = equity
                return

        # --- Profit / phase transition check ---
        profit_pct = (equity - self._phase_balance) / self._phase_balance * 100.0

        if self._phase == "phase_1_active":
            if profit_pct >= self._p1_target:
                self._phase_1_passed = True
                self._transition_to_phase_2()
        elif self._phase == "phase_2_active":
            if profit_pct >= self._p2_target:
                self._transition_to_funded(equity)
        elif self._phase == "funded_active":
            self._track_funded_month(timestamp, equity)

        # Always update the running "previous close equity" for next bar
        self._prev_day_close_equity = equity

    def check_pass(self) -> dict:
        """Protocol-compatible alias for :meth:`get_status`.

        Returns the same dict as ``get_status()`` so external callers
        can rely on the uniform :class:`PropFirmTrackerProtocol` name.
        """
        return self.get_status()

    def to_dict(self) -> dict:
        """Return a dict snapshot suitable for JSON / parquet serialization.

        Mirrors the :class:`PropFirmTrackerProtocol` signature so the
        engine can call ``to_dict()`` uniformly regardless of which
        concrete tracker is active.
        """
        status = self.get_status()
        return {
            "style": "the5ers_pct_phased",
            **status,
        }

    def get_status(self) -> dict:
        """Return a dict describing the current tracker state."""
        profit_pct = 0.0
        if self._phase_balance > 0 and self._phase not in ("", ):
            profit_pct = (self._latest_equity - self._phase_balance) / self._phase_balance * 100.0

        # Include the current open month's return if in funded phase
        monthly_returns = list(self._funded_monthly_returns)
        if (self._phase == "funded_active"
                and self._funded_current_month is not None
                and self._funded_month_start_equity > 0):
            current_month_return = (
                (self._latest_equity - self._funded_month_start_equity)
                / self._funded_month_start_equity
                * 100.0
            )
            monthly_returns.append(round(current_month_return, 4))

        return {
            "phase": self._phase,
            "profit_pct": round(profit_pct, 4),
            "phase_balance": self._phase_balance,
            "max_total_dd_pct": round(self._max_total_dd_pct, 4),
            "phase_1_passed": self._phase_1_passed,
            "failure_reason": self._failure_reason,
            "funded_monthly_returns": monthly_returns,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _current_daily_loss_limit(self) -> float:
        """Return the daily loss limit % for the current phase."""
        if self._phase.startswith("phase_1"):
            return self._p1_daily_loss
        elif self._phase.startswith("phase_2"):
            return self._p2_daily_loss
        else:
            return self._funded_daily_loss

    def _current_total_loss_limit(self) -> float:
        """Return the total loss limit % for the current phase."""
        if self._phase.startswith("phase_1"):
            return self._p1_max_loss
        elif self._phase.startswith("phase_2"):
            return self._p2_max_loss
        else:
            return self._funded_max_loss

    def _fail(self, reason: str) -> None:
        """Transition to a terminal failure state."""
        if self._phase == "phase_1_active":
            self._phase = "failed_phase_1"
        elif self._phase == "phase_2_active":
            self._phase = "failed_phase_2"
        elif self._phase == "funded_active":
            self._phase = "funded_bust"
        self._failure_reason = reason

    def _reset_dd_tracking(self) -> None:
        """Reset drawdown tracking for a new phase."""
        self._max_total_dd_pct = 0.0
        self._phase_hwm = self._account_size
        self._prev_day_close_equity = self._account_size
        self._prev_day_close_balance = self._account_size
        self._current_day_date = None

    def _transition_to_phase_2(self) -> None:
        """Move from Phase 1 to Phase 2, resetting balance and DD."""
        self._phase = "phase_2_active"
        self._phase_balance = self._account_size
        self._latest_equity = self._account_size
        self._reset_dd_tracking()

    def _transition_to_funded(self, equity: float) -> None:
        """Move from Phase 2 to Funded, resetting balance and DD."""
        self._phase = "funded_active"
        self._phase_balance = self._account_size
        self._latest_equity = self._account_size
        self._reset_dd_tracking()
        # Start funded monthly tracking
        self._funded_month_start_equity = self._account_size
        self._funded_current_month = None
        self._funded_monthly_returns = []

    def _track_funded_month(self, timestamp: datetime, equity: float) -> None:
        """Track per-calendar-month returns in the funded phase."""
        month_key = timestamp.strftime("%Y-%m")

        if self._funded_current_month is None:
            # First update in funded phase
            self._funded_current_month = month_key
            self._funded_month_start_equity = self._phase_balance
            return

        if month_key != self._funded_current_month:
            # Month boundary crossed -- close out the old month
            if self._funded_month_start_equity > 0:
                month_return = (
                    (self._prev_day_close_equity - self._funded_month_start_equity)
                    / self._funded_month_start_equity
                    * 100.0
                )
                self._funded_monthly_returns.append(round(month_return, 4))

            # Start tracking the new month
            self._funded_month_start_equity = self._prev_day_close_equity
            self._funded_current_month = month_key


# =============================================================================
# PerformanceMetrics
# =============================================================================

class PerformanceMetrics:
    """Calculate comprehensive backtest performance metrics.

    All ratio calculations use daily returns derived from the equity curve.
    Annualisation assumes 252 trading days per year.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(
        self,
        trades: List[dict],
        equity_curve: pd.Series,
        initial_balance: float,
    ) -> dict:
        """Calculate the full metrics suite from a trade list and equity curve.

        Parameters
        ----------
        trades:
            List of trade dicts produced by the backtesting simulation.
            Expected keys: r_multiple, entry_time, exit_time, pnl (optional).
        equity_curve:
            DatetimeIndex Series of portfolio equity at each 5M bar.
        initial_balance:
            Starting portfolio value.

        Returns
        -------
        dict with all metric keys listed in the module docstring.
        """
        if not trades:
            return self._empty_metrics()

        r_multiples = [float(t.get("r_multiple") or 0.0) for t in trades]
        wins = [r for r in r_multiples if r > 0]
        losses = [r for r in r_multiples if r <= 0]

        win_rate = len(wins) / len(trades) if trades else 0.0
        avg_r = float(np.mean(r_multiples)) if r_multiples else 0.0
        avg_win_r = float(np.mean(wins)) if wins else 0.0
        avg_loss_r = float(np.mean(losses)) if losses else 0.0
        expectancy = win_rate * avg_win_r + (1 - win_rate) * avg_loss_r

        # Profit factor: sum of winning R / abs(sum of losing R)
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        # Consecutive win/loss streaks
        max_cons_wins, max_cons_losses = self._consecutive_streaks(r_multiples)

        # Trade durations
        avg_duration = self._avg_duration(trades)

        # Drawdown metrics
        max_dd_pct, dd_peak_dt, dd_trough_dt = self.max_drawdown(equity_curve)
        max_daily_dd = self._max_daily_drawdown(equity_curve)

        # Return metrics
        final_equity = float(equity_curve.iloc[-1]) if not equity_curve.empty else initial_balance
        total_return_pct = (final_equity - initial_balance) / initial_balance * 100.0

        # Daily returns for ratio calculations
        daily_equity = equity_curve.resample("1D").last().dropna()
        daily_returns = daily_equity.pct_change().dropna()

        sharpe = self.sharpe_ratio(daily_returns)
        sortino = self.sortino_ratio(daily_returns)
        calmar = self.calmar_ratio(total_return_pct, max_dd_pct)

        return {
            "total_trades": len(trades),
            "win_rate": round(win_rate, 4),
            "avg_r_multiple": round(avg_r, 4),
            "avg_win_r": round(avg_win_r, 4),
            "avg_loss_r": round(avg_loss_r, 4),
            "expectancy": round(expectancy, 4),
            "profit_factor": round(profit_factor, 4) if math.isfinite(profit_factor) else None,
            "gross_profit_r": round(gross_profit, 4),
            "gross_loss_r": round(gross_loss, 4),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "calmar_ratio": round(calmar, 4),
            "max_drawdown_pct": round(max_dd_pct, 4),
            "max_daily_dd_pct": round(max_daily_dd, 4),
            "total_return_pct": round(total_return_pct, 4),
            "avg_trade_duration": avg_duration,
            "consecutive_wins_max": max_cons_wins,
            "consecutive_losses_max": max_cons_losses,
            "dd_peak_date": dd_peak_dt,
            "dd_trough_date": dd_trough_dt,
            "initial_balance": initial_balance,
            "final_equity": round(final_equity, 2),
        }

    def sharpe_ratio(
        self, returns: pd.Series, risk_free: float = 0.0
    ) -> float:
        """Annualised Sharpe ratio from daily returns.

        Parameters
        ----------
        returns:
            Series of daily fractional returns (not percentages).
        risk_free:
            Daily risk-free rate.  Default: 0.0.

        Returns
        -------
        Annualised Sharpe, or 0.0 when there is insufficient data.
        """
        if returns.empty or len(returns) < 2:
            return 0.0
        excess = returns - risk_free
        std = float(excess.std(ddof=1))
        if std == 0.0 or not math.isfinite(std):
            return 0.0
        mean_excess = float(excess.mean())
        return float(mean_excess / std * math.sqrt(_TRADING_DAYS_PER_YEAR))

    def sortino_ratio(
        self, returns: pd.Series, risk_free: float = 0.0
    ) -> float:
        """Sortino ratio using downside deviation only.

        Parameters
        ----------
        returns:
            Series of daily fractional returns.
        risk_free:
            Daily risk-free rate.  Default: 0.0.

        Returns
        -------
        Annualised Sortino, or 0.0 when there is insufficient data.
        """
        if returns.empty or len(returns) < 2:
            return 0.0
        excess = returns - risk_free
        downside = excess[excess < 0]
        if downside.empty:
            return float("inf")  # no losing days — infinite Sortino
        downside_std = float(np.sqrt(np.mean(downside ** 2)))
        if downside_std == 0.0 or not math.isfinite(downside_std):
            return 0.0
        mean_excess = float(excess.mean())
        return float(mean_excess / downside_std * math.sqrt(_TRADING_DAYS_PER_YEAR))

    def calmar_ratio(
        self, total_return_pct: float, max_drawdown_pct: float
    ) -> float:
        """Calmar ratio: annualised return divided by max drawdown.

        Parameters
        ----------
        total_return_pct:
            Total percentage return over the backtest period.
        max_drawdown_pct:
            Maximum peak-to-trough drawdown as a positive percentage.

        Returns
        -------
        Calmar ratio, or 0.0 when max_drawdown_pct is zero.
        """
        if max_drawdown_pct <= 0.0:
            return 0.0
        return round(total_return_pct / max_drawdown_pct, 4)

    def max_drawdown(
        self, equity: pd.Series
    ) -> Tuple[float, Optional[datetime], Optional[datetime]]:
        """Calculate maximum peak-to-trough drawdown percentage.

        Parameters
        ----------
        equity:
            Time-series of portfolio equity values.

        Returns
        -------
        (max_dd_pct, peak_datetime, trough_datetime)
            max_dd_pct is a positive percentage.  Dates may be None when
            the equity series is empty or has no drawdown.
        """
        if equity.empty or len(equity) < 2:
            return 0.0, None, None

        running_max = equity.cummax()
        drawdown_series = (running_max - equity) / running_max * 100.0
        max_dd = float(drawdown_series.max())

        if max_dd <= 0.0:
            return 0.0, None, None

        trough_idx = drawdown_series.idxmax()
        # Peak is the last high water mark before the trough
        peak_idx = equity.loc[:trough_idx].idxmax()

        peak_dt = peak_idx.to_pydatetime() if hasattr(peak_idx, "to_pydatetime") else None
        trough_dt = trough_idx.to_pydatetime() if hasattr(trough_idx, "to_pydatetime") else None

        return round(max_dd, 4), peak_dt, trough_dt

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _max_daily_drawdown(equity: pd.Series) -> float:
        """Maximum single-day drawdown from day-open to day-low."""
        if equity.empty:
            return 0.0
        daily_open = equity.resample("1D").first().dropna()
        daily_low = equity.resample("1D").min().dropna()
        aligned_open, aligned_low = daily_open.align(daily_low, join="inner")
        if aligned_open.empty:
            return 0.0
        daily_dd = (aligned_open - aligned_low) / aligned_open * 100.0
        return round(float(daily_dd.max()), 4)

    @staticmethod
    def _consecutive_streaks(r_multiples: List[float]) -> Tuple[int, int]:
        """Return (max_consecutive_wins, max_consecutive_losses)."""
        max_wins = max_losses = cur_wins = cur_losses = 0
        for r in r_multiples:
            if r > 0:
                cur_wins += 1
                cur_losses = 0
                max_wins = max(max_wins, cur_wins)
            else:
                cur_losses += 1
                cur_wins = 0
                max_losses = max(max_losses, cur_losses)
        return max_wins, max_losses

    @staticmethod
    def _avg_duration(trades: List[dict]) -> Optional[timedelta]:
        """Average trade duration from entry_time to exit_time."""
        durations = []
        for t in trades:
            entry = t.get("entry_time")
            exit_ = t.get("exit_time")
            if entry and exit_:
                try:
                    delta = exit_ - entry
                    if isinstance(delta, timedelta):
                        durations.append(delta)
                except TypeError:
                    pass
        if not durations:
            return None
        total_seconds = sum(d.total_seconds() for d in durations)
        return timedelta(seconds=total_seconds / len(durations))

    @staticmethod
    def _empty_metrics() -> dict:
        """Return a zero-filled metrics dict when no trades exist."""
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_r_multiple": 0.0,
            "avg_win_r": 0.0,
            "avg_loss_r": 0.0,
            "expectancy": 0.0,
            "profit_factor": None,
            "gross_profit_r": 0.0,
            "gross_loss_r": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "max_daily_dd_pct": 0.0,
            "total_return_pct": 0.0,
            "avg_trade_duration": None,
            "consecutive_wins_max": 0,
            "consecutive_losses_max": 0,
            "dd_peak_date": None,
            "dd_trough_date": None,
            "initial_balance": 0.0,
            "final_equity": 0.0,
        }
