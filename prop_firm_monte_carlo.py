"""
Monte Carlo Simulator for Prop Firm Challenge Pass Rate Optimization
Target: The5ers High Stakes Classic — XAU/USD
$10K account, 8% target, 5% daily DD, 10% total DD, unlimited time
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from enum import Enum


class Outcome(Enum):
    PASS = "PASS"
    TOTAL_DD_BUST = "TOTAL_DD_BUST"
    DAILY_DD_BUST = "DAILY_DD_BUST"
    MAX_DAYS_REACHED = "MAX_DAYS_REACHED"


@dataclass
class ChallengeParams:
    """Prop firm challenge parameters."""
    starting_balance: float = 10_000.0
    profit_target_pct: float = 8.0       # 8% = $800
    daily_dd_pct: float = 5.0            # 5% = $500 max loss from day-open equity
    total_dd_pct: float = 10.0           # 10% = $1000 max loss from starting balance
    max_trading_days: int = 500          # "unlimited" but capped for simulation


@dataclass
class SystemParams:
    """Trading system parameters."""
    win_rate: float = 0.55
    reward_risk_ratio: float = 2.0       # avg winner / avg loser in R
    risk_per_trade_pct: float = 1.0      # % of starting balance risked per trade
    trades_per_day: int = 2              # max trades per day
    trading_days_per_week: int = 5
    intraday_correlation: float = 0.3    # correlation between same-day trades on gold


@dataclass
class AdaptiveParams:
    """Adaptive risk parameters (equity-based)."""
    enabled: bool = False
    # Risk multipliers based on equity state (as % from start)
    deep_drawdown_threshold: float = -3.0   # below this: multiply risk by deep_dd_mult
    deep_drawdown_mult: float = 0.5
    mild_drawdown_threshold: float = 0.0    # below this: multiply risk by mild_dd_mult
    mild_drawdown_mult: float = 0.75
    early_profit_threshold: float = 4.0     # below this (but above 0): multiply by early_mult
    early_profit_mult: float = 1.2
    mid_profit_threshold: float = 6.0       # below this: multiply by mid_mult
    mid_profit_mult: float = 1.0
    near_target_mult: float = 0.6           # above mid_profit_threshold: use this


@dataclass
class SimResult:
    """Result of a single simulation run."""
    outcome: Outcome
    final_equity_pct: float       # % change from starting balance
    trading_days: int
    total_trades: int
    max_drawdown_pct: float       # worst peak-to-trough
    equity_curve: Optional[List[float]] = None  # optional, for visualization


@dataclass
class MonteCarloResult:
    """Aggregated Monte Carlo results."""
    n_simulations: int
    pass_rate: float
    total_dd_bust_rate: float
    daily_dd_bust_rate: float
    timeout_rate: float
    mean_days_to_pass: float          # conditional on passing
    median_days_to_pass: float
    mean_days_to_bust: float          # conditional on busting
    mean_max_drawdown_pct: float
    pass_rate_std_error: float
    results: List[SimResult] = field(default_factory=list)


def get_adaptive_risk_multiplier(equity_pct: float, params: AdaptiveParams) -> float:
    """
    Compute risk multiplier based on current equity relative to start.

    Args:
        equity_pct: Current equity as % change from starting balance
        params: Adaptive risk parameters

    Returns:
        Risk multiplier (e.g., 0.5 means half normal risk)
    """
    if not params.enabled:
        return 1.0

    if equity_pct < params.deep_drawdown_threshold:
        return params.deep_drawdown_mult
    elif equity_pct < params.mild_drawdown_threshold:
        return params.mild_drawdown_mult
    elif equity_pct < params.early_profit_threshold:
        return params.early_profit_mult
    elif equity_pct < params.mid_profit_threshold:
        return params.mid_profit_mult
    else:
        return params.near_target_mult


def generate_correlated_trades(
    n_trades: int,
    win_rate: float,
    reward_risk_ratio: float,
    risk_pct: float,
    correlation: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generate correlated intraday trade PnL values.

    Uses a Gaussian copula to induce correlation between trade outcomes.
    Each trade is Bernoulli (win/loss) with correlated latent variables.

    Args:
        n_trades: Number of trades to generate
        win_rate: Probability of a winning trade
        reward_risk_ratio: Winner size / Loser size
        risk_pct: Risk per trade as % of starting balance
        correlation: Pairwise correlation between trades
        rng: NumPy random generator

    Returns:
        Array of PnL values as % of starting balance
    """
    if n_trades == 0:
        return np.array([])

    if n_trades == 1 or correlation == 0:
        # Independent trades
        wins = rng.random(n_trades) < win_rate
        pnl = np.where(wins, risk_pct * reward_risk_ratio, -risk_pct)
        return pnl

    # Gaussian copula for correlated Bernoulli outcomes
    # Build correlation matrix
    cov = np.full((n_trades, n_trades), correlation)
    np.fill_diagonal(cov, 1.0)

    # Generate correlated normals
    z = rng.multivariate_normal(np.zeros(n_trades), cov)

    # Convert to uniform via CDF, then to Bernoulli
    from scipy.stats import norm
    u = norm.cdf(z)
    wins = u < win_rate

    pnl = np.where(wins, risk_pct * reward_risk_ratio, -risk_pct)
    return pnl


def simulate_single_challenge(
    challenge: ChallengeParams,
    system: SystemParams,
    adaptive: AdaptiveParams,
    rng: np.random.Generator,
    store_curve: bool = False
) -> SimResult:
    """
    Simulate a single prop firm challenge attempt.

    Tracks:
    - Equity curve (as % from starting balance)
    - Daily PnL (resets at each day open)
    - High water mark for total drawdown
    - Daily drawdown from day-open equity
    - Profit target

    Args:
        challenge: Prop firm rules
        system: Trading system parameters
        adaptive: Adaptive risk parameters
        rng: NumPy random generator
        store_curve: Whether to store the full equity curve

    Returns:
        SimResult with outcome and statistics
    """
    equity_pct = 0.0          # % change from starting balance
    high_water_mark = 0.0     # highest equity_pct reached
    total_trades = 0
    equity_curve = [0.0] if store_curve else None

    for day in range(1, challenge.max_trading_days + 1):
        day_open_equity_pct = equity_pct  # equity at start of this trading day

        # Determine today's risk per trade (adaptive)
        risk_mult = get_adaptive_risk_multiplier(equity_pct, adaptive)
        today_risk_pct = system.risk_per_trade_pct * risk_mult

        # Determine number of trades today (can vary, here we use fixed)
        n_trades_today = system.trades_per_day

        # Generate correlated intraday trades
        trade_pnls = generate_correlated_trades(
            n_trades=n_trades_today,
            win_rate=system.win_rate,
            reward_risk_ratio=system.reward_risk_ratio,
            risk_pct=today_risk_pct,
            correlation=system.intraday_correlation,
            rng=rng
        )

        # Process each trade sequentially within the day
        for pnl in trade_pnls:
            equity_pct += pnl
            total_trades += 1

            # Update high water mark
            if equity_pct > high_water_mark:
                high_water_mark = equity_pct

            if store_curve:
                equity_curve.append(equity_pct)

            # Check profit target
            if equity_pct >= challenge.profit_target_pct:
                max_dd = high_water_mark - min(equity_curve) if store_curve else high_water_mark
                return SimResult(
                    outcome=Outcome.PASS,
                    final_equity_pct=equity_pct,
                    trading_days=day,
                    total_trades=total_trades,
                    max_drawdown_pct=max_dd,
                    equity_curve=equity_curve
                )

            # Check daily drawdown (from day-open equity)
            daily_dd = day_open_equity_pct - equity_pct
            if daily_dd >= challenge.daily_dd_pct:
                max_dd = high_water_mark - min(equity_curve) if store_curve else high_water_mark
                return SimResult(
                    outcome=Outcome.DAILY_DD_BUST,
                    final_equity_pct=equity_pct,
                    trading_days=day,
                    total_trades=total_trades,
                    max_drawdown_pct=max_dd,
                    equity_curve=equity_curve
                )

            # Check total drawdown (from starting balance = 0%)
            if equity_pct <= -challenge.total_dd_pct:
                max_dd = high_water_mark - equity_pct
                return SimResult(
                    outcome=Outcome.TOTAL_DD_BUST,
                    final_equity_pct=equity_pct,
                    trading_days=day,
                    total_trades=total_trades,
                    max_drawdown_pct=max_dd,
                    equity_curve=equity_curve
                )

    # Max days reached (shouldn't happen with 500 days for positive expectancy)
    max_dd = high_water_mark - min(equity_curve) if store_curve else high_water_mark
    return SimResult(
        outcome=Outcome.MAX_DAYS_REACHED,
        final_equity_pct=equity_pct,
        trading_days=challenge.max_trading_days,
        total_trades=total_trades,
        max_drawdown_pct=max_dd,
        equity_curve=equity_curve
    )


def run_monte_carlo(
    challenge: ChallengeParams,
    system: SystemParams,
    adaptive: AdaptiveParams,
    n_simulations: int = 100_000,
    store_curves: int = 0,
    seed: int = 42
) -> MonteCarloResult:
    """
    Run Monte Carlo simulation of prop firm challenge.

    Args:
        challenge: Prop firm rules
        system: Trading system parameters
        adaptive: Adaptive risk parameters
        n_simulations: Number of simulation iterations
        store_curves: Number of equity curves to store (0 = none, for visualization)
        seed: Random seed for reproducibility

    Returns:
        MonteCarloResult with aggregated statistics
    """
    rng = np.random.default_rng(seed)
    results: List[SimResult] = []

    for i in range(n_simulations):
        store = (i < store_curves)
        result = simulate_single_challenge(challenge, system, adaptive, rng, store_curve=store)
        results.append(result)

    # Aggregate
    outcomes = [r.outcome for r in results]
    n_pass = sum(1 for o in outcomes if o == Outcome.PASS)
    n_total_dd = sum(1 for o in outcomes if o == Outcome.TOTAL_DD_BUST)
    n_daily_dd = sum(1 for o in outcomes if o == Outcome.DAILY_DD_BUST)
    n_timeout = sum(1 for o in outcomes if o == Outcome.MAX_DAYS_REACHED)

    pass_rate = n_pass / n_simulations

    # Days to pass (conditional on passing)
    pass_days = [r.trading_days for r in results if r.outcome == Outcome.PASS]
    bust_days = [r.trading_days for r in results if r.outcome != Outcome.PASS]

    # Standard error of pass rate: SE = sqrt(p(1-p)/n)
    se = np.sqrt(pass_rate * (1 - pass_rate) / n_simulations)

    return MonteCarloResult(
        n_simulations=n_simulations,
        pass_rate=pass_rate,
        total_dd_bust_rate=n_total_dd / n_simulations,
        daily_dd_bust_rate=n_daily_dd / n_simulations,
        timeout_rate=n_timeout / n_simulations,
        mean_days_to_pass=np.mean(pass_days) if pass_days else float('inf'),
        median_days_to_pass=np.median(pass_days) if pass_days else float('inf'),
        mean_days_to_bust=np.mean(bust_days) if bust_days else float('inf'),
        mean_max_drawdown_pct=np.mean([r.max_drawdown_pct for r in results]),
        pass_rate_std_error=se,
        results=results
    )


def run_parameter_grid(
    challenge: ChallengeParams,
    win_rates: List[float],
    rr_ratios: List[float],
    risk_pcts: List[float],
    n_simulations: int = 50_000,
    trades_per_day: int = 2,
    seed: int = 42
) -> Dict[Tuple[float, float, float], float]:
    """
    Run parameter grid scan and return pass rates.

    Returns dict mapping (win_rate, rr_ratio, risk_pct) -> pass_rate
    """
    grid_results = {}
    total = len(win_rates) * len(rr_ratios) * len(risk_pcts)
    count = 0

    for wr in win_rates:
        for rr in rr_ratios:
            for risk in risk_pcts:
                count += 1
                system = SystemParams(
                    win_rate=wr,
                    reward_risk_ratio=rr,
                    risk_per_trade_pct=risk,
                    trades_per_day=trades_per_day
                )
                adaptive = AdaptiveParams(enabled=False)
                result = run_monte_carlo(
                    challenge, system, adaptive,
                    n_simulations=n_simulations, seed=seed
                )
                grid_results[(wr, rr, risk)] = result.pass_rate
                print(f"[{count}/{total}] WR={wr:.0%} RR={rr}:1 Risk={risk}% "
                      f"-> Pass={result.pass_rate:.1%} (SE={result.pass_rate_std_error:.2%})")

    return grid_results


# ============================================================
# BLOCK BOOTSTRAP FOR PRESERVING INTRA-DAY TRADE GROUPS
# ============================================================

def block_bootstrap_simulate(
    historical_daily_pnls: List[List[float]],
    challenge: ChallengeParams,
    n_simulations: int = 100_000,
    seed: int = 42
) -> MonteCarloResult:
    """
    Block bootstrap Monte Carlo using historical trade data.

    Instead of generating synthetic trades from (WR, R:R), this resamples
    entire trading days (blocks of intra-day trades) from historical data.
    This preserves:
    - Intra-day trade correlation structure
    - Daily trade count variation
    - Non-normal PnL distributions
    - Clustering of wins/losses within days

    Args:
        historical_daily_pnls: List of daily PnL groups.
            Each element is a list of individual trade PnLs for one day.
            Example: [[1.2, -0.8], [2.1], [-1.0, -1.0, 0.5], ...]
        challenge: Prop firm parameters
        n_simulations: Number of bootstrap iterations
        seed: Random seed

    Returns:
        MonteCarloResult
    """
    rng = np.random.default_rng(seed)
    n_days_available = len(historical_daily_pnls)
    results: List[SimResult] = []

    # Precompute daily net PnLs for efficiency
    daily_nets = [sum(day) for day in historical_daily_pnls]

    for _ in range(n_simulations):
        equity_pct = 0.0
        high_water_mark = 0.0
        total_trades = 0
        outcome = Outcome.MAX_DAYS_REACHED
        final_day = challenge.max_trading_days

        for day in range(1, challenge.max_trading_days + 1):
            # Sample a random historical day (with replacement)
            idx = rng.integers(0, n_days_available)
            day_trades = historical_daily_pnls[idx]
            day_open = equity_pct

            busted = False
            for pnl in day_trades:
                equity_pct += pnl
                total_trades += 1

                if equity_pct > high_water_mark:
                    high_water_mark = equity_pct

                # Profit target
                if equity_pct >= challenge.profit_target_pct:
                    outcome = Outcome.PASS
                    final_day = day
                    busted = True
                    break

                # Daily DD
                if (day_open - equity_pct) >= challenge.daily_dd_pct:
                    outcome = Outcome.DAILY_DD_BUST
                    final_day = day
                    busted = True
                    break

                # Total DD
                if equity_pct <= -challenge.total_dd_pct:
                    outcome = Outcome.TOTAL_DD_BUST
                    final_day = day
                    busted = True
                    break

            if busted:
                break

        results.append(SimResult(
            outcome=outcome,
            final_equity_pct=equity_pct,
            trading_days=final_day,
            total_trades=total_trades,
            max_drawdown_pct=high_water_mark - equity_pct if equity_pct < 0 else high_water_mark
        ))

    # Aggregate (same as run_monte_carlo)
    n = n_simulations
    n_pass = sum(1 for r in results if r.outcome == Outcome.PASS)
    pass_rate = n_pass / n

    pass_days = [r.trading_days for r in results if r.outcome == Outcome.PASS]
    bust_days = [r.trading_days for r in results if r.outcome != Outcome.PASS]

    return MonteCarloResult(
        n_simulations=n,
        pass_rate=pass_rate,
        total_dd_bust_rate=sum(1 for r in results if r.outcome == Outcome.TOTAL_DD_BUST) / n,
        daily_dd_bust_rate=sum(1 for r in results if r.outcome == Outcome.DAILY_DD_BUST) / n,
        timeout_rate=sum(1 for r in results if r.outcome == Outcome.MAX_DAYS_REACHED) / n,
        mean_days_to_pass=np.mean(pass_days) if pass_days else float('inf'),
        median_days_to_pass=np.median(pass_days) if pass_days else float('inf'),
        mean_days_to_bust=np.mean(bust_days) if bust_days else float('inf'),
        mean_max_drawdown_pct=np.mean([r.max_drawdown_pct for r in results]),
        pass_rate_std_error=np.sqrt(pass_rate * (1 - pass_rate) / n),
        results=results
    )


# ============================================================
# VISUALIZATION
# ============================================================

def plot_equity_fans(results: List[SimResult], title: str = "Monte Carlo Equity Curves"):
    """
    Plot equity curve fan chart from simulation results.
    Only works with results that have stored equity curves.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    curves_with_data = [r for r in results if r.equity_curve is not None]
    if not curves_with_data:
        print("No equity curves stored. Run with store_curves > 0.")
        return

    for r in curves_with_data:
        color = 'green' if r.outcome == Outcome.PASS else 'red'
        alpha = 0.3 if len(curves_with_data) > 20 else 0.6
        ax.plot(r.equity_curve, color=color, alpha=alpha, linewidth=0.5)

    # Draw barrier lines
    ax.axhline(y=8.0, color='green', linestyle='--', linewidth=2, label='Profit Target (+8%)')
    ax.axhline(y=-10.0, color='red', linestyle='--', linewidth=2, label='Total DD (-10%)')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    ax.set_xlabel('Trade Number')
    ax.set_ylabel('Equity (% from start)')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig('equity_fans.png', dpi=150)
    plt.close()
    print("Saved equity_fans.png")


def plot_pass_rate_convergence(results: List[SimResult]):
    """
    Plot pass rate convergence as simulation count increases.
    Shows that pass rate estimate stabilizes, confirming sufficient iterations.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    n = len(results)
    cumulative_passes = np.cumsum([1 if r.outcome == Outcome.PASS else 0 for r in results])
    x = np.arange(1, n + 1)
    running_pass_rate = cumulative_passes / x

    ax.plot(x, running_pass_rate, color='blue', linewidth=1)

    # Add confidence bands (1 SE)
    se = np.sqrt(running_pass_rate * (1 - running_pass_rate) / x)
    ax.fill_between(x, running_pass_rate - se, running_pass_rate + se, alpha=0.2, color='blue')

    ax.set_xlabel('Number of Simulations')
    ax.set_ylabel('Cumulative Pass Rate')
    ax.set_title('Pass Rate Convergence')
    ax.set_xscale('log')
    plt.tight_layout()
    plt.savefig('pass_rate_convergence.png', dpi=150)
    plt.close()
    print("Saved pass_rate_convergence.png")


def plot_outcome_distribution(result: MonteCarloResult):
    """Plot distribution of outcomes and days to completion."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Outcome pie chart
    labels = ['Pass', 'Total DD Bust', 'Daily DD Bust', 'Timeout']
    sizes = [result.pass_rate, result.total_dd_bust_rate,
             result.daily_dd_bust_rate, result.timeout_rate]
    colors = ['#2ecc71', '#e74c3c', '#e67e22', '#95a5a6']
    nonzero = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0.001]
    if nonzero:
        labels_nz, sizes_nz, colors_nz = zip(*nonzero)
        axes[0].pie(sizes_nz, labels=labels_nz, colors=colors_nz, autopct='%1.1f%%')
        axes[0].set_title('Outcome Distribution')

    # Days to pass histogram
    pass_days = [r.trading_days for r in result.results if r.outcome == Outcome.PASS]
    if pass_days:
        axes[1].hist(pass_days, bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
        axes[1].axvline(np.median(pass_days), color='darkgreen', linestyle='--',
                        label=f'Median: {np.median(pass_days):.0f} days')
        axes[1].set_xlabel('Trading Days')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Days to Pass (conditional on passing)')
        axes[1].legend()

    # Final equity distribution
    final_equities = [r.final_equity_pct for r in result.results]
    axes[2].hist(final_equities, bins=80, color='#3498db', alpha=0.7, edgecolor='black')
    axes[2].axvline(8.0, color='green', linestyle='--', label='Target +8%')
    axes[2].axvline(-10.0, color='red', linestyle='--', label='Total DD -10%')
    axes[2].set_xlabel('Final Equity (%)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Final Equity Distribution')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('outcome_distribution.png', dpi=150)
    plt.close()
    print("Saved outcome_distribution.png")


# ============================================================
# RECOMMENDED ITERATIONS FOR STATISTICAL PRECISION
# ============================================================
#
# Standard error of pass rate estimate:
#   SE = sqrt(p(1-p)/N)
#
# For p ≈ 0.60 (typical pass rate):
#   N = 10,000  → SE = 0.49%  (±1% at 95% CI)
#   N = 50,000  → SE = 0.22%  (±0.44% at 95% CI)
#   N = 100,000 → SE = 0.15%  (±0.31% at 95% CI)
#   N = 500,000 → SE = 0.07%  (±0.14% at 95% CI)
#
# For <1% standard error: N ≥ p(1-p)/(0.01)^2 = 0.24/0.0001 = 2,400
#   (easily achieved with N=10,000+)
#
# For <0.5% standard error: N ≥ 9,600
# For <0.2% standard error: N ≥ 60,000
#
# RECOMMENDATION: 100,000 iterations for production analysis
#   - Provides ±0.3% precision at 95% confidence
#   - Runs in ~30-60 seconds on modern hardware
#   - Sufficient for comparing strategies with >1% difference


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PROP FIRM MONTE CARLO SIMULATOR")
    print("The5ers High Stakes Classic — XAU/USD")
    print("=" * 70)

    challenge = ChallengeParams()

    # --- Baseline: Static risk ---
    print("\n--- BASELINE: Static 1.0% risk ---")
    system_baseline = SystemParams(
        win_rate=0.55,
        reward_risk_ratio=2.0,
        risk_per_trade_pct=1.0,
        trades_per_day=2,
        intraday_correlation=0.3
    )
    adaptive_off = AdaptiveParams(enabled=False)
    result_baseline = run_monte_carlo(
        challenge, system_baseline, adaptive_off,
        n_simulations=100_000, store_curves=200
    )
    print(f"  Pass Rate:        {result_baseline.pass_rate:.1%} "
          f"(SE: {result_baseline.pass_rate_std_error:.2%})")
    print(f"  Total DD Bust:    {result_baseline.total_dd_bust_rate:.1%}")
    print(f"  Daily DD Bust:    {result_baseline.daily_dd_bust_rate:.1%}")
    print(f"  Mean Days (pass): {result_baseline.mean_days_to_pass:.0f}")
    print(f"  Median Days:      {result_baseline.median_days_to_pass:.0f}")

    # --- Adaptive risk ---
    print("\n--- ADAPTIVE: Equity-based risk scaling ---")
    adaptive_on = AdaptiveParams(enabled=True)
    result_adaptive = run_monte_carlo(
        challenge, system_baseline, adaptive_on,
        n_simulations=100_000, store_curves=200
    )
    print(f"  Pass Rate:        {result_adaptive.pass_rate:.1%} "
          f"(SE: {result_adaptive.pass_rate_std_error:.2%})")
    print(f"  Total DD Bust:    {result_adaptive.total_dd_bust_rate:.1%}")
    print(f"  Daily DD Bust:    {result_adaptive.daily_dd_bust_rate:.1%}")
    print(f"  Mean Days (pass): {result_adaptive.mean_days_to_pass:.0f}")
    print(f"  Median Days:      {result_adaptive.median_days_to_pass:.0f}")

    # --- Two-phase: 1.5% until +4%, then 1.0% ---
    print("\n--- TWO-PHASE: 1.5% risk until +4%, then 1.0% ---")
    system_twophase = SystemParams(
        win_rate=0.55,
        reward_risk_ratio=2.0,
        risk_per_trade_pct=1.5,  # base risk (phase 1)
        trades_per_day=2,
        intraday_correlation=0.3
    )
    adaptive_twophase = AdaptiveParams(
        enabled=True,
        deep_drawdown_threshold=-3.0,
        deep_drawdown_mult=0.5,          # 0.75% risk in deep DD
        mild_drawdown_threshold=0.0,
        mild_drawdown_mult=0.75,         # 1.125% risk in mild DD
        early_profit_threshold=4.0,
        early_profit_mult=1.0,           # 1.5% risk in phase 1
        mid_profit_threshold=4.0,
        mid_profit_mult=0.667,           # 1.0% risk in phase 2
        near_target_mult=0.667           # 1.0% risk near target
    )
    result_twophase = run_monte_carlo(
        challenge, system_twophase, adaptive_twophase,
        n_simulations=100_000, store_curves=200
    )
    print(f"  Pass Rate:        {result_twophase.pass_rate:.1%} "
          f"(SE: {result_twophase.pass_rate_std_error:.2%})")
    print(f"  Total DD Bust:    {result_twophase.total_dd_bust_rate:.1%}")
    print(f"  Daily DD Bust:    {result_twophase.daily_dd_bust_rate:.1%}")
    print(f"  Mean Days (pass): {result_twophase.mean_days_to_pass:.0f}")
    print(f"  Median Days:      {result_twophase.median_days_to_pass:.0f}")

    # --- Parameter Grid Scan ---
    print("\n" + "=" * 70)
    print("PARAMETER GRID SCAN")
    print("=" * 70)
    grid = run_parameter_grid(
        challenge,
        win_rates=[0.45, 0.50, 0.55, 0.60, 0.65],
        rr_ratios=[1.5, 2.0, 2.5, 3.0],
        risk_pcts=[0.5, 1.0, 1.5, 2.0],
        n_simulations=50_000,
        trades_per_day=2
    )

    # --- Visualizations ---
    print("\n--- Generating Visualizations ---")
    plot_equity_fans(result_adaptive.results, "Adaptive Risk — Equity Curve Fan")
    plot_pass_rate_convergence(result_adaptive.results)
    plot_outcome_distribution(result_adaptive)

    print("\nDone.")
