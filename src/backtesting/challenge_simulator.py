"""
Challenge simulator with rolling-window and Monte Carlo modes.

Tests how often a strategy config would pass a 2-step prop firm challenge
(e.g. FTMO / The5ers).  Two modes:

- **Rolling windows**: start a new simulated challenge every N trading days
  across the backtest period, measuring pass rate at different entry points.
- **Monte Carlo**: shuffle the trade sequence and replay the challenge many
  times, measuring robustness to ordering effects.

For TopstepX-style combines (dollar-based, single-account), the simulator
dispatches to :class:`TopstepCombineSimulator` instead of running the
rolling/MC pipeline — a TopstepX combine is a single continuous account
that either passes or fails, not a challenge replayed N times.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Result dataclass
# ============================================================================

@dataclass
class ChallengeSimulationResult:
    """Aggregated result from one or both simulation modes."""

    total_windows: int
    phase_1_pass_count: int
    phase_2_pass_count: int  # subset of phase_1 passes
    full_pass_count: int  # passed both phases
    pass_rate: float  # full_pass_count / total_windows
    rolling_pass_rate: float
    monte_carlo_pass_rate: float
    avg_days_phase_1: float
    avg_days_phase_2: float
    failure_breakdown: Dict[str, int]  # {"daily_dd": N, "total_dd": N, ...}
    funded_monthly_returns: List[float] = field(default_factory=list)
    avg_funded_monthly_return: float = 0.0
    months_above_10pct: int = 0


# ============================================================================
# Simulator
# ============================================================================

class ChallengeSimulator:
    """Simulate a 2-step prop firm challenge across trade sequences.

    Parameters
    ----------
    account_size : float
        Starting balance for each simulated challenge attempt.
    phase_1_target_pct : float
        Profit target (%) required to pass Phase 1.
    phase_1_max_loss_pct : float
        Maximum total drawdown (%) allowed in Phase 1 before failing.
    phase_1_daily_loss_pct : float
        Maximum intraday drawdown (%) allowed in Phase 1.
    phase_2_target_pct : float
        Profit target (%) required to pass Phase 2.
    phase_2_max_loss_pct : float
        Maximum total drawdown (%) allowed in Phase 2 before failing.
    phase_2_daily_loss_pct : float
        Maximum intraday drawdown (%) allowed in Phase 2.
    rolling_window_spacing_days : int
        Number of trading days between rolling-window start points.
    """

    def __init__(
        self,
        account_size: float = 10_000.0,
        phase_1_target_pct: float = 8.0,
        phase_1_max_loss_pct: float = 10.0,
        phase_1_daily_loss_pct: float = 5.0,
        phase_2_target_pct: float = 5.0,
        phase_2_max_loss_pct: float = 10.0,
        phase_2_daily_loss_pct: float = 5.0,
        rolling_window_spacing_days: int = 22,
    ) -> None:
        self.account_size = account_size
        self.phase_1_target_pct = phase_1_target_pct
        self.phase_1_max_loss_pct = phase_1_max_loss_pct
        self.phase_1_daily_loss_pct = phase_1_daily_loss_pct
        self.phase_2_target_pct = phase_2_target_pct
        self.phase_2_max_loss_pct = phase_2_max_loss_pct
        self.phase_2_daily_loss_pct = phase_2_daily_loss_pct
        self.rolling_window_spacing_days = rolling_window_spacing_days

    # ------------------------------------------------------------------
    # Core single-challenge simulation
    # ------------------------------------------------------------------

    def _simulate_phase(
        self,
        trades: List[dict],
        target_pct: float,
        max_loss_pct: float,
        daily_loss_pct: float,
    ) -> Tuple[bool, str, int, int, List[float]]:
        """Simulate one phase of a challenge.

        Parameters
        ----------
        trades : list[dict]
            Each dict has at least ``r_multiple`` and ``risk_pct``.  Optionally
            ``day_index`` for daily-DD grouping (defaults to trade index).
        target_pct : float
            Profit target for this phase.
        max_loss_pct : float
            Maximum total drawdown before failure.
        daily_loss_pct : float
            Maximum single-day drawdown before failure.

        Returns
        -------
        (passed, failure_reason, days_used, trades_consumed, funded_returns)
        """
        balance = self.account_size
        initial_balance = balance
        target_balance = initial_balance * (1 + target_pct / 100.0)
        max_dd_balance = initial_balance * (1 - max_loss_pct / 100.0)

        # Track daily balances for daily-DD check
        current_day: Optional[int] = None
        day_open_balance = balance

        days_used = 0
        trades_consumed = 0
        funded_returns: List[float] = []

        for idx, trade in enumerate(trades):
            r = trade["r_multiple"]
            risk_pct = trade["risk_pct"]
            day = trade.get("day_index", idx)

            # New day -> reset daily tracking
            if day != current_day:
                current_day = day
                day_open_balance = balance
                days_used += 1

            # Apply trade PnL
            dollar_pnl = r * (risk_pct / 100.0) * balance
            balance += dollar_pnl

            # Check daily drawdown (loss from day open)
            daily_dd_pct = (day_open_balance - balance) / day_open_balance * 100.0
            if daily_dd_pct >= daily_loss_pct:
                return False, "daily_dd", days_used, idx + 1, funded_returns

            # Check total drawdown (loss from initial)
            total_dd_pct = (initial_balance - balance) / initial_balance * 100.0
            if total_dd_pct >= max_loss_pct:
                return False, "total_dd", days_used, idx + 1, funded_returns

            # Check profit target
            if balance >= target_balance:
                return True, "", days_used, idx + 1, funded_returns

        # Ran out of trades without hitting target
        return False, "no_trades", days_used, len(trades), funded_returns

    def _simulate_challenge(
        self,
        trades: List[dict],
    ) -> Tuple[bool, bool, str, int, int, List[float]]:
        """Simulate the full 2-phase challenge on a trade sequence.

        Returns
        -------
        (passed_phase_1, passed_both, failure_reason,
         days_phase_1, days_phase_2, funded_monthly_returns)
        """
        # Phase 1
        p1_passed, p1_reason, p1_days, p1_consumed, _ = self._simulate_phase(
            trades,
            self.phase_1_target_pct,
            self.phase_1_max_loss_pct,
            self.phase_1_daily_loss_pct,
        )

        if not p1_passed:
            return False, False, p1_reason, p1_days, 0, []

        # Phase 2: use remaining trades
        remaining = trades[p1_consumed:]
        if not remaining:
            return True, False, "no_trades", p1_days, 0, []

        # Assign sequential day indices to remaining trades for phase 2
        # so daily-DD tracking works correctly with the fresh phase.
        phase2_trades = []
        for i, t in enumerate(remaining):
            t2 = dict(t)
            # Preserve relative day ordering if day_index present
            t2["day_index"] = t.get("day_index", i)
            phase2_trades.append(t2)

        p2_passed, p2_reason, p2_days, p2_consumed, _ = self._simulate_phase(
            phase2_trades,
            self.phase_2_target_pct,
            self.phase_2_max_loss_pct,
            self.phase_2_daily_loss_pct,
        )

        if not p2_passed:
            return True, False, p2_reason, p1_days, p2_days, []

        # Funded: compute monthly returns from leftover trades
        funded_trades = phase2_trades[p2_consumed:]
        funded_returns = self._compute_funded_monthly_returns(funded_trades)

        return True, True, "", p1_days, p2_days, funded_returns

    # ------------------------------------------------------------------
    # Funded-period monthly returns
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_funded_monthly_returns(trades: List[dict]) -> List[float]:
        """Compute approximate monthly returns from funded-period trades.

        Groups trades into consecutive blocks of ~22 trading days and
        returns the cumulative return of each block.
        """
        if not trades:
            return []

        DAYS_PER_MONTH = 22
        monthly_returns: List[float] = []
        month_pnl = 0.0
        day_count = 0
        current_day: Optional[int] = None

        for idx, t in enumerate(trades):
            day = t.get("day_index", idx)
            if day != current_day:
                current_day = day
                day_count += 1

            month_pnl += t["r_multiple"] * t["risk_pct"] / 100.0

            if day_count >= DAYS_PER_MONTH:
                monthly_returns.append(month_pnl * 100.0)  # as percentage
                month_pnl = 0.0
                day_count = 0

        # Last partial month
        if day_count > 0:
            monthly_returns.append(month_pnl * 100.0)

        return monthly_returns

    # ------------------------------------------------------------------
    # Rolling window mode
    # ------------------------------------------------------------------

    def run_rolling(
        self,
        trades: List[dict],
        total_trading_days: int,
    ) -> ChallengeSimulationResult:
        """Run the challenge across rolling windows of the trade history.

        A new simulated challenge starts every ``rolling_window_spacing_days``
        trading days.
        """
        spacing = self.rolling_window_spacing_days

        # Build a mapping: day_index -> first trade index on that day
        day_to_first_trade: Dict[int, int] = {}
        for idx, t in enumerate(trades):
            day = t.get("day_index", idx)
            if day not in day_to_first_trade:
                day_to_first_trade[day] = idx

        # Determine start day_indices
        sorted_days = sorted(day_to_first_trade.keys())
        start_days = sorted_days[::spacing]

        # Run a challenge for each window
        p1_pass = 0
        p2_pass = 0
        full_pass = 0
        failure_counts: Dict[str, int] = {}
        days_p1_list: List[int] = []
        days_p2_list: List[int] = []
        all_funded_returns: List[float] = []

        for start_day in start_days:
            first_idx = day_to_first_trade[start_day]
            window_trades = trades[first_idx:]

            passed_p1, passed_both, reason, d1, d2, funded = (
                self._simulate_challenge(window_trades)
            )

            if passed_p1:
                p1_pass += 1
                days_p1_list.append(d1)
            if passed_both:
                p2_pass += 1
                full_pass += 1
                days_p2_list.append(d2)
                all_funded_returns.extend(funded)

            if reason:
                failure_counts[reason] = failure_counts.get(reason, 0) + 1

        total_windows = len(start_days)
        pass_rate = full_pass / total_windows if total_windows > 0 else 0.0
        avg_d1 = float(np.mean(days_p1_list)) if days_p1_list else 0.0
        avg_d2 = float(np.mean(days_p2_list)) if days_p2_list else 0.0
        avg_funded = float(np.mean(all_funded_returns)) if all_funded_returns else 0.0
        months_10 = sum(1 for r in all_funded_returns if r >= 10.0)

        return ChallengeSimulationResult(
            total_windows=total_windows,
            phase_1_pass_count=p1_pass,
            phase_2_pass_count=p2_pass,
            full_pass_count=full_pass,
            pass_rate=pass_rate,
            rolling_pass_rate=pass_rate,
            monte_carlo_pass_rate=0.0,
            avg_days_phase_1=avg_d1,
            avg_days_phase_2=avg_d2,
            failure_breakdown=failure_counts,
            funded_monthly_returns=all_funded_returns,
            avg_funded_monthly_return=avg_funded,
            months_above_10pct=months_10,
        )

    # ------------------------------------------------------------------
    # Monte Carlo mode
    # ------------------------------------------------------------------

    def run_monte_carlo(
        self,
        trades: List[dict],
        n_simulations: int = 10_000,
        seed: Optional[int] = None,
    ) -> ChallengeSimulationResult:
        """Shuffle trade order and replay the challenge *n_simulations* times."""
        rng = np.random.default_rng(seed)

        p1_pass = 0
        p2_pass = 0
        full_pass = 0
        failure_counts: Dict[str, int] = {}
        days_p1_list: List[int] = []
        days_p2_list: List[int] = []
        all_funded_returns: List[float] = []

        # Pre-strip day_index: MC assigns synthetic sequential indices
        base_trades = [
            {"r_multiple": t["r_multiple"], "risk_pct": t["risk_pct"]}
            for t in trades
        ]
        indices = np.arange(len(base_trades))

        for _ in range(n_simulations):
            shuffled_idx = rng.permutation(indices)
            shuffled = [
                {**base_trades[i], "day_index": rank}
                for rank, i in enumerate(shuffled_idx)
            ]

            passed_p1, passed_both, reason, d1, d2, funded = (
                self._simulate_challenge(shuffled)
            )

            if passed_p1:
                p1_pass += 1
                days_p1_list.append(d1)
            if passed_both:
                p2_pass += 1
                full_pass += 1
                days_p2_list.append(d2)
                all_funded_returns.extend(funded)

            if reason:
                failure_counts[reason] = failure_counts.get(reason, 0) + 1

        pass_rate = full_pass / n_simulations if n_simulations > 0 else 0.0
        avg_d1 = float(np.mean(days_p1_list)) if days_p1_list else 0.0
        avg_d2 = float(np.mean(days_p2_list)) if days_p2_list else 0.0
        avg_funded = float(np.mean(all_funded_returns)) if all_funded_returns else 0.0
        months_10 = sum(1 for r in all_funded_returns if r >= 10.0)

        return ChallengeSimulationResult(
            total_windows=n_simulations,
            phase_1_pass_count=p1_pass,
            phase_2_pass_count=p2_pass,
            full_pass_count=full_pass,
            pass_rate=pass_rate,
            rolling_pass_rate=0.0,
            monte_carlo_pass_rate=pass_rate,
            avg_days_phase_1=avg_d1,
            avg_days_phase_2=avg_d2,
            failure_breakdown=failure_counts,
            funded_monthly_returns=all_funded_returns,
            avg_funded_monthly_return=avg_funded,
            months_above_10pct=months_10,
        )

    # ------------------------------------------------------------------
    # Combined run (rolling + Monte Carlo)
    # ------------------------------------------------------------------

    def run(
        self,
        trades: List[dict],
        total_trading_days: int,
        n_mc_simulations: int = 1_000,
        prop_firm_style: Optional[str] = None,
        topstep_config: Optional[Any] = None,
    ) -> Union[ChallengeSimulationResult, Any]:
        """Run both rolling-window and Monte Carlo simulations, merge results.

        When *prop_firm_style* is ``"topstep_combine_dollar"``, delegates
        to :class:`TopstepCombineSimulator` and returns a
        :class:`TopstepCombineResult` instead of the rolling/MC
        aggregate — a TopstepX combine doesn't benefit from rolling-window
        replay since it's a single continuous account.
        """
        if prop_firm_style == "topstep_combine_dollar":
            from src.backtesting.topstep_simulator import TopstepCombineSimulator
            from src.config.models import TopstepCombineConfig

            cfg = topstep_config or TopstepCombineConfig()
            sim = TopstepCombineSimulator(config=cfg)
            return sim.run(trades)

        rolling = self.run_rolling(trades, total_trading_days)
        mc = self.run_monte_carlo(trades, n_simulations=n_mc_simulations)

        # Merge failure breakdowns
        merged_failures: Dict[str, int] = dict(rolling.failure_breakdown)
        for k, v in mc.failure_breakdown.items():
            merged_failures[k] = merged_failures.get(k, 0) + v

        total_windows = rolling.total_windows + mc.total_windows
        full_pass = rolling.full_pass_count + mc.full_pass_count
        combined_pass_rate = (
            (rolling.pass_rate + mc.pass_rate) / 2.0
        )

        all_funded = rolling.funded_monthly_returns + mc.funded_monthly_returns
        avg_funded = float(np.mean(all_funded)) if all_funded else 0.0
        months_10 = sum(1 for r in all_funded if r >= 10.0)

        # Average phase days across both pools
        p1_days_parts = []
        p2_days_parts = []
        if rolling.avg_days_phase_1 > 0:
            p1_days_parts.append(rolling.avg_days_phase_1)
        if mc.avg_days_phase_1 > 0:
            p1_days_parts.append(mc.avg_days_phase_1)
        if rolling.avg_days_phase_2 > 0:
            p2_days_parts.append(rolling.avg_days_phase_2)
        if mc.avg_days_phase_2 > 0:
            p2_days_parts.append(mc.avg_days_phase_2)

        avg_d1 = float(np.mean(p1_days_parts)) if p1_days_parts else 0.0
        avg_d2 = float(np.mean(p2_days_parts)) if p2_days_parts else 0.0

        return ChallengeSimulationResult(
            total_windows=total_windows,
            phase_1_pass_count=rolling.phase_1_pass_count + mc.phase_1_pass_count,
            phase_2_pass_count=rolling.phase_2_pass_count + mc.phase_2_pass_count,
            full_pass_count=full_pass,
            pass_rate=combined_pass_rate,
            rolling_pass_rate=rolling.pass_rate,
            monte_carlo_pass_rate=mc.pass_rate,
            avg_days_phase_1=avg_d1,
            avg_days_phase_2=avg_d2,
            failure_breakdown=merged_failures,
            funded_monthly_returns=all_funded,
            avg_funded_monthly_return=avg_funded,
            months_above_10pct=months_10,
        )
