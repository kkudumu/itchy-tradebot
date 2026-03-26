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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Trading days per year used for annualisation
_TRADING_DAYS_PER_YEAR: float = 252.0


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
