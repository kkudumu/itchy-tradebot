"""
Monte Carlo engine for prop firm challenge pass-rate estimation.

MonteCarloSimulator runs 10,000+ simulated challenge attempts against a
supplied set of backtest trade results.  Each simulation resamples the
trade sequence (optionally via block bootstrap), applies phased risk
sizing, and enforces all The5ers-style prop firm constraints:

    - 5% daily drawdown limit (from day's opening balance)
    - 2% daily circuit breaker (stop trading once triggered)
    - 10% total drawdown from initial balance
    - 8% profit target
    - 30-calendar-day time limit

Phased risk: position size starts at ``initial_risk_pct`` of balance and
drops to ``reduced_risk_pct`` once the account has recovered past
``phase_threshold_pct`` — simulating a prop firm's recommended risk
reduction protocol for challenge attempts.

Fat-tailed distributions: when ``use_fat_tails=True``, trade R-multiples
are drawn from a fitted non-central t-distribution rather than replayed
verbatim.  This stress-tests edge cases beyond the historical sample.

Block bootstrap: when ``block_bootstrap=True``, whole calendar-day
trade groups are resampled together, preserving intra-day correlations.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.simulation.distributions import BlockBootstrapper, TradeDistribution

logger = logging.getLogger(__name__)

# Convergence is declared when the pass-rate varies by less than this
# percentage over the trailing window.
_CONVERGENCE_TOLERANCE_PCT: float = 1.0

# Number of simulations at the trailing edge of the convergence window.
_CONVERGENCE_WINDOW: int = 1000

# Minimum simulations before convergence is evaluated.
_CONVERGENCE_MIN_SIMS: int = 2000


# =============================================================================
# Result dataclasses
# =============================================================================

@dataclass
class ChallengeOutcome:
    """Outcome of a single simulated challenge attempt."""

    passed: bool
    failure_reason: str
    """One of: 'passed', 'daily_dd', 'total_dd', 'timeout', 'circuit_breaker'."""

    days: int
    """Calendar days elapsed when the challenge ended."""

    final_balance: float
    max_daily_dd: float
    """Worst daily drawdown observed, as a positive percentage of day open."""

    max_total_dd: float
    """Worst peak-to-trough drawdown from initial balance, positive percentage."""

    peak_balance: float
    """Highest balance reached during this attempt."""

    # Full per-day equity curve (used for visualisation)
    equity_curve: List[float] = field(default_factory=list)


@dataclass
class MCResult:
    """Aggregated results of a Monte Carlo simulation run."""

    n_simulations: int
    pass_rate: float
    avg_days: float
    median_days: float
    daily_dd_failure_rate: float
    total_dd_failure_rate: float
    timeout_rate: float
    circuit_breaker_rate: float
    outcomes: List[ChallengeOutcome]
    convergence_reached: bool
    convergence_at: int
    """Simulation index where the pass-rate first stabilised, or n_simulations."""

    # Running pass-rate series for convergence visualisation
    running_pass_rates: List[float] = field(default_factory=list)


# =============================================================================
# MonteCarloSimulator
# =============================================================================

class MonteCarloSimulator:
    """Prop firm challenge Monte Carlo simulation engine.

    Parameters
    ----------
    initial_balance:
        Starting account balance.  Default: 10,000.
    profit_target_pct:
        Profit target as a percentage of initial balance.  Default: 8.0.
    max_daily_dd_pct:
        Maximum daily drawdown allowed as a percentage of that day's
        opening balance.  Breaching this fails the challenge.  Default: 5.0.
    max_total_dd_pct:
        Maximum total drawdown from initial balance allowed.  Default: 10.0.
    daily_circuit_breaker_pct:
        Once daily loss reaches this percentage of the day's open, no
        further trades are taken that day.  Default: 2.0.
    time_limit_days:
        Calendar days before the challenge times out.  Default: 30.
    initial_risk_pct:
        Position risk as a percentage of current balance before the phase
        threshold is reached.  Default: 1.5.
    reduced_risk_pct:
        Position risk after crossing ``phase_threshold_pct`` profit.
        Default: 0.75.
    phase_threshold_pct:
        Profit level at which risk is reduced.  Default: 4.0.
    """

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        profit_target_pct: float = 8.0,
        max_daily_dd_pct: float = 5.0,
        max_total_dd_pct: float = 10.0,
        daily_circuit_breaker_pct: float = 2.0,
        time_limit_days: int = 30,
        initial_risk_pct: float = 1.5,
        reduced_risk_pct: float = 0.75,
        phase_threshold_pct: float = 4.0,
    ) -> None:
        if initial_balance <= 0:
            raise ValueError("initial_balance must be positive.")
        if not (0.0 < profit_target_pct <= 100.0):
            raise ValueError("profit_target_pct must be in (0, 100].")
        if not (0.0 < max_daily_dd_pct <= 100.0):
            raise ValueError("max_daily_dd_pct must be in (0, 100].")
        if not (0.0 < max_total_dd_pct <= 100.0):
            raise ValueError("max_total_dd_pct must be in (0, 100].")
        if not (0.0 < daily_circuit_breaker_pct <= max_daily_dd_pct):
            raise ValueError(
                "daily_circuit_breaker_pct must be in (0, max_daily_dd_pct]."
            )

        self.initial_balance = initial_balance
        self.profit_target_pct = profit_target_pct
        self.max_daily_dd_pct = max_daily_dd_pct
        self.max_total_dd_pct = max_total_dd_pct
        self.daily_circuit_breaker_pct = daily_circuit_breaker_pct
        self.time_limit_days = time_limit_days
        self.initial_risk_pct = initial_risk_pct
        self.reduced_risk_pct = reduced_risk_pct
        self.phase_threshold_pct = phase_threshold_pct

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        trade_results: List[dict],
        n_simulations: int = 10_000,
        use_fat_tails: bool = True,
        block_bootstrap: bool = True,
        seed: Optional[int] = None,
    ) -> MCResult:
        """Run the Monte Carlo simulation.

        Parameters
        ----------
        trade_results:
            List of trade dicts from the backtesting engine.  Each dict
            must have an 'r_multiple' key (float).  For block bootstrap,
            an 'entry_time' key is required (datetime or ISO-8601 string).
        n_simulations:
            Number of challenge attempts to simulate.  10,000 gives
            ±1 pp accuracy at 95% confidence.
        use_fat_tails:
            If True, draw R-multiples from a fitted non-central-t
            distribution instead of replaying raw trades.
        block_bootstrap:
            If True, resample whole calendar-day blocks to preserve
            intra-day correlations.
        seed:
            Optional random seed for reproducibility.

        Returns
        -------
        MCResult with pass_rate and all requested statistics.
        """
        if not trade_results:
            raise ValueError("trade_results must be non-empty.")

        rng = np.random.default_rng(seed)

        # --- prepare fat-tail distribution (fit once, sample per sim) ---
        dist: Optional[TradeDistribution] = None
        if use_fat_tails:
            r_multiples = [
                float(t.get("r_multiple") or 0.0) for t in trade_results
            ]
            dist = TradeDistribution()
            dist.fit(r_multiples)
            logger.info("Fat-tail distribution fitted to %d trades.", len(r_multiples))

        # --- prepare block bootstrapper ---
        bootstrapper: Optional[BlockBootstrapper] = None
        if block_bootstrap:
            bootstrapper = BlockBootstrapper(trade_results)
            logger.info(
                "Block bootstrapper ready: %d day-blocks.", bootstrapper.n_days
            )

        # Estimate trades per simulated challenge from historical frequency
        avg_trades_per_day = self._avg_trades_per_day(trade_results)
        target_trade_count = max(
            5, int(avg_trades_per_day * self.time_limit_days)
        )

        # --- main simulation loop ---
        outcomes: List[ChallengeOutcome] = []
        running_pass_rates: List[float] = []
        convergence_at: int = n_simulations
        convergence_reached: bool = False

        for i in range(n_simulations):
            # Resample trades for this simulation
            sim_trades = self._resample_trades(
                trade_results,
                rng,
                block_bootstrap=block_bootstrap,
                bootstrapper=bootstrapper,
                dist=dist,
                use_fat_tails=use_fat_tails,
                target_count=target_trade_count,
            )

            outcome = self._simulate_single(sim_trades, rng)
            outcomes.append(outcome)

            # Track running pass rate
            n_done = i + 1
            passed_so_far = sum(1 for o in outcomes if o.passed)
            running_pass_rates.append(passed_so_far / n_done * 100.0)

            # Check convergence after minimum window
            if not convergence_reached and n_done >= _CONVERGENCE_MIN_SIMS:
                converged, convergence_at = self._check_convergence(
                    running_pass_rates, n_done
                )
                if converged:
                    convergence_reached = True
                    logger.info(
                        "Monte Carlo converged at simulation %d (pass rate %.2f%%).",
                        convergence_at, running_pass_rates[-1],
                    )

        # --- aggregate ---
        result = self._aggregate(
            outcomes=outcomes,
            n_simulations=n_simulations,
            running_pass_rates=running_pass_rates,
            convergence_reached=convergence_reached,
            convergence_at=convergence_at,
        )

        logger.info(
            "Monte Carlo complete: n=%d pass_rate=%.2f%% converged=%s",
            n_simulations, result.pass_rate, convergence_reached,
        )
        return result

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------

    def _simulate_single(
        self, trades: List[dict], rng: np.random.Generator
    ) -> ChallengeOutcome:
        """Simulate one prop firm challenge attempt from a trade sequence.

        Trades are processed day-by-day.  Each day:
          1. Record opening balance.
          2. Apply phased risk sizing to each trade's R-multiple.
          3. Check circuit breaker after each trade.
          4. Check daily DD limit after each trade.
          5. After all trades on the day, advance to next day.
          6. Check total DD and profit target after each day.
          7. Check time limit at each day boundary.
        """
        balance = self.initial_balance
        peak_balance = self.initial_balance
        max_daily_dd: float = 0.0
        max_total_dd: float = 0.0
        equity_curve: List[float] = [balance]

        # Group trades by day index (trades are already sequenced by day)
        day_groups = self._group_by_day_index(trades)
        n_days = len(day_groups)

        for day_idx, day_trades in enumerate(day_groups):
            if day_idx >= self.time_limit_days:
                return ChallengeOutcome(
                    passed=False,
                    failure_reason="timeout",
                    days=day_idx,
                    final_balance=balance,
                    max_daily_dd=max_daily_dd,
                    max_total_dd=max_total_dd,
                    peak_balance=peak_balance,
                    equity_curve=equity_curve,
                )

            day_open = balance
            daily_loss = 0.0
            circuit_broken = False

            for trade in day_trades:
                if circuit_broken:
                    break  # no more trades this day

                r_multiple = float(trade.get("r_multiple", 0.0))

                # Phased risk: reduce once profit exceeds threshold
                profit_pct = (balance - self.initial_balance) / self.initial_balance * 100.0
                if profit_pct >= self.phase_threshold_pct:
                    risk_pct = self.reduced_risk_pct
                else:
                    risk_pct = self.initial_risk_pct

                # Dollar risk for this trade
                risk_amount = balance * risk_pct / 100.0
                pnl = r_multiple * risk_amount

                balance += pnl

                # Update peak balance
                if balance > peak_balance:
                    peak_balance = balance

                # Daily drawdown check (from this day's opening balance)
                if day_open > 0:
                    current_daily_dd = (day_open - balance) / day_open * 100.0
                    current_daily_dd = max(0.0, current_daily_dd)
                    if current_daily_dd > max_daily_dd:
                        max_daily_dd = current_daily_dd
                    daily_loss = current_daily_dd

                    # Circuit breaker: suspend trading for the day
                    if daily_loss >= self.daily_circuit_breaker_pct:
                        circuit_broken = True

                    # Hard daily DD limit — challenge fails immediately
                    if daily_loss >= self.max_daily_dd_pct:
                        equity_curve.append(balance)
                        return ChallengeOutcome(
                            passed=False,
                            failure_reason="daily_dd",
                            days=day_idx + 1,
                            final_balance=balance,
                            max_daily_dd=max_daily_dd,
                            max_total_dd=max_total_dd,
                            peak_balance=peak_balance,
                            equity_curve=equity_curve,
                        )

                # Total DD check (from initial balance)
                total_dd = (self.initial_balance - balance) / self.initial_balance * 100.0
                total_dd = max(0.0, total_dd)
                if total_dd > max_total_dd:
                    max_total_dd = total_dd

                if total_dd >= self.max_total_dd_pct:
                    equity_curve.append(balance)
                    return ChallengeOutcome(
                        passed=False,
                        failure_reason="total_dd",
                        days=day_idx + 1,
                        final_balance=balance,
                        max_daily_dd=max_daily_dd,
                        max_total_dd=max_total_dd,
                        peak_balance=peak_balance,
                        equity_curve=equity_curve,
                    )

            equity_curve.append(balance)

            # Profit target check at end of each day
            profit_pct = (balance - self.initial_balance) / self.initial_balance * 100.0
            if profit_pct >= self.profit_target_pct:
                return ChallengeOutcome(
                    passed=True,
                    failure_reason="passed",
                    days=day_idx + 1,
                    final_balance=balance,
                    max_daily_dd=max_daily_dd,
                    max_total_dd=max_total_dd,
                    peak_balance=peak_balance,
                    equity_curve=equity_curve,
                )

        # Exhausted all trades without hitting any terminal condition
        # If time limit not yet reached, this is a timeout (not enough trades)
        return ChallengeOutcome(
            passed=False,
            failure_reason="timeout",
            days=min(n_days, self.time_limit_days),
            final_balance=balance,
            max_daily_dd=max_daily_dd,
            max_total_dd=max_total_dd,
            peak_balance=peak_balance,
            equity_curve=equity_curve,
        )

    def _resample_trades(
        self,
        trades: List[dict],
        rng: np.random.Generator,
        block_bootstrap: bool,
        bootstrapper: Optional[BlockBootstrapper],
        dist: Optional[TradeDistribution],
        use_fat_tails: bool,
        target_count: int,
    ) -> List[dict]:
        """Generate a resampled trade sequence for one simulation run.

        Block bootstrap path: resample day-blocks from bootstrapper,
        then optionally replace R-multiples with fat-tail samples.

        IID path: draw trade indices with replacement, optionally using
        the fat-tail distribution for R-multiples.
        """
        if block_bootstrap and bootstrapper is not None:
            # Resample enough day-blocks to cover the time limit
            n_days_needed = self.time_limit_days + 5  # small buffer
            resampled = bootstrapper.resample(n_days_needed, rng)

            if use_fat_tails and dist is not None:
                # Replace the R-multiples with fat-tail samples while
                # preserving all other trade metadata (win/loss sign is
                # determined by the distribution's win probability)
                n = len(resampled)
                new_r = dist.sample(n, rng)
                resampled = [
                    {**t, "r_multiple": float(new_r[i])}
                    for i, t in enumerate(resampled)
                ]

            return resampled

        # IID bootstrap (no day-block grouping)
        indices = rng.integers(0, len(trades), size=target_count)
        resampled_iid = [trades[i] for i in indices]

        if use_fat_tails and dist is not None:
            new_r = dist.sample(target_count, rng)
            resampled_iid = [
                {**t, "r_multiple": float(new_r[i])}
                for i, t in enumerate(resampled_iid)
            ]

        return resampled_iid

    # ------------------------------------------------------------------
    # Convergence detection
    # ------------------------------------------------------------------

    @staticmethod
    def _check_convergence(
        running_pass_rates: List[float],
        n_done: int,
    ) -> Tuple[bool, int]:
        """Test whether the running pass rate has stabilised.

        Convergence criterion: the standard deviation of the last
        ``_CONVERGENCE_WINDOW`` pass-rate values is below a threshold
        that corresponds to ±``_CONVERGENCE_TOLERANCE_PCT`` pp.

        Returns
        -------
        (converged, convergence_at_index)
        """
        if n_done < _CONVERGENCE_MIN_SIMS + _CONVERGENCE_WINDOW:
            return False, n_done

        recent = running_pass_rates[-_CONVERGENCE_WINDOW:]
        std_recent = float(np.std(recent))

        if std_recent <= _CONVERGENCE_TOLERANCE_PCT:
            return True, n_done - _CONVERGENCE_WINDOW
        return False, n_done

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(
        outcomes: List[ChallengeOutcome],
        n_simulations: int,
        running_pass_rates: List[float],
        convergence_reached: bool,
        convergence_at: int,
    ) -> MCResult:
        """Compute summary statistics from all outcomes."""
        n = len(outcomes)
        if n == 0:
            return MCResult(
                n_simulations=0,
                pass_rate=0.0,
                avg_days=0.0,
                median_days=0.0,
                daily_dd_failure_rate=0.0,
                total_dd_failure_rate=0.0,
                timeout_rate=0.0,
                circuit_breaker_rate=0.0,
                outcomes=[],
                convergence_reached=False,
                convergence_at=0,
                running_pass_rates=[],
            )

        passed = [o for o in outcomes if o.passed]
        failed_daily = [o for o in outcomes if o.failure_reason == "daily_dd"]
        failed_total = [o for o in outcomes if o.failure_reason == "total_dd"]
        failed_timeout = [o for o in outcomes if o.failure_reason == "timeout"]
        failed_circuit = [o for o in outcomes if o.failure_reason == "circuit_breaker"]

        all_days = [o.days for o in outcomes]

        return MCResult(
            n_simulations=n_simulations,
            pass_rate=len(passed) / n * 100.0,
            avg_days=float(np.mean(all_days)) if all_days else 0.0,
            median_days=float(np.median(all_days)) if all_days else 0.0,
            daily_dd_failure_rate=len(failed_daily) / n * 100.0,
            total_dd_failure_rate=len(failed_total) / n * 100.0,
            timeout_rate=len(failed_timeout) / n * 100.0,
            circuit_breaker_rate=len(failed_circuit) / n * 100.0,
            outcomes=outcomes,
            convergence_reached=convergence_reached,
            convergence_at=convergence_at,
            running_pass_rates=running_pass_rates,
        )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _avg_trades_per_day(trades: List[dict]) -> float:
        """Estimate average trades per calendar day from the trade list."""
        if not trades:
            return 1.0

        # Try to extract dates from entry_time
        dates = set()
        for t in trades:
            raw = t.get("entry_time")
            if raw is None:
                continue
            if hasattr(raw, "strftime"):
                dates.add(raw.strftime("%Y-%m-%d"))
            elif isinstance(raw, str) and len(raw) >= 10:
                dates.add(raw[:10])

        n_days = len(dates) if dates else 1
        return len(trades) / n_days

    @staticmethod
    def _group_by_day_index(trades: List[dict]) -> List[List[dict]]:
        """Group sequenced trades into days based on entry_time.

        Trades without entry_time are distributed ~evenly across days
        by position in the list using a default trades-per-day estimate.
        """
        if not trades:
            return []

        # Attempt date-based grouping
        day_map: Dict[str, List[dict]] = {}
        no_date: List[dict] = []

        for trade in trades:
            raw = trade.get("entry_time")
            date_str: Optional[str] = None

            if raw is not None:
                if hasattr(raw, "strftime"):
                    date_str = raw.strftime("%Y-%m-%d")
                elif isinstance(raw, str) and len(raw) >= 10:
                    date_str = raw[:10]

            if date_str is not None:
                if date_str not in day_map:
                    day_map[date_str] = []
                day_map[date_str].append(trade)
            else:
                no_date.append(trade)

        if day_map:
            groups = [day_map[k] for k in sorted(day_map.keys())]
            # Append undated trades as a final pseudo-day if any
            if no_date:
                groups.append(no_date)
            return groups

        # No date info — split into pseudo-days of 3 trades each
        chunk_size = 3
        groups = []
        for i in range(0, len(trades), chunk_size):
            groups.append(trades[i : i + chunk_size])
        return groups
