#!/usr/bin/env python3
"""
Monte Carlo Simulator for Prop Firm Challenge
The5ers High Stakes Classic: $10K, 8% target, 5% daily DD, 10% total DD
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import json
import sys

# ============================================================
# CONFIGURATION
# ============================================================

NUM_SIMULATIONS = 10000
MAX_TRADING_DAYS = 22  # ~30 calendar days
PROFIT_TARGET_PCT = 8.0
DAILY_DD_LIMIT_PCT = 5.0
TOTAL_DD_LIMIT_PCT = 10.0
DAILY_CIRCUIT_BREAKER_PCT = 2.0
STARTING_BALANCE = 10000.0

np.random.seed(42)

# ============================================================
# CORE SIMULATOR
# ============================================================

@dataclass
class SimResult:
    passed: bool
    days_to_pass: int
    hit_daily_dd: bool
    hit_total_dd: bool
    final_equity_pct: float
    max_drawdown_pct: float
    total_trades: int
    daily_dd_events: int  # how many days hit daily DD


def simulate_one_trial(
    win_rate: float,
    rr_distribution: List[Tuple[float, float]],  # [(probability, rr_multiple), ...]
    risk_schedule: str,  # 'fixed_X', 'phased', 'adaptive', 'anti_martingale'
    base_risk_pct: float,
    trades_per_day_range: Tuple[int, int],  # (min, max) trades per day
    max_days: int = MAX_TRADING_DAYS,
    pause_after_consecutive_losses: int = 0,  # 0 = disabled
    slippage_pct: float = 0.0,  # additional loss per trade as % of account
) -> SimResult:

    equity = STARTING_BALANCE
    high_water_mark = STARTING_BALANCE
    day = 0
    total_trades = 0
    consecutive_losses = 0
    daily_dd_events = 0
    hit_daily_dd = False
    hit_total_dd = False
    passed = False

    for day in range(1, max_days + 1):
        day_start_equity = equity
        daily_loss = 0.0
        daily_pnl = 0.0

        # Determine trades for this day
        num_trades = np.random.randint(trades_per_day_range[0], trades_per_day_range[1] + 1)

        for t in range(num_trades):
            # Check pause rule
            if pause_after_consecutive_losses > 0 and consecutive_losses >= pause_after_consecutive_losses:
                # Skip remaining trades this day
                consecutive_losses = 0  # reset after pause
                break

            # Determine risk for this trade
            current_profit_pct = ((equity - STARTING_BALANCE) / STARTING_BALANCE) * 100

            if risk_schedule == 'phased':
                if current_profit_pct < 2.0:
                    risk_pct = base_risk_pct * 1.25  # early: aggressive
                elif current_profit_pct < 5.0:
                    risk_pct = base_risk_pct  # mid: standard
                else:
                    risk_pct = base_risk_pct * 0.75  # late: conservative
            elif risk_schedule == 'adaptive':
                # Start aggressive, switch to conservative at 4%
                if current_profit_pct < 4.0:
                    risk_pct = base_risk_pct * 1.5
                else:
                    risk_pct = base_risk_pct * 0.75
            elif risk_schedule == 'anti_martingale':
                if consecutive_losses == 0 or total_trades == 0:
                    risk_pct = base_risk_pct
                elif consecutive_losses >= 2:
                    risk_pct = base_risk_pct * 0.5
                else:
                    risk_pct = base_risk_pct * 0.75
                # After a win, could increase but capped
                # (simplified: we track consecutive losses only)
            else:
                # fixed
                risk_pct = base_risk_pct

            # Check daily circuit breaker
            daily_loss_pct = -((equity - day_start_equity) / day_start_equity) * 100 if equity < day_start_equity else 0
            if daily_loss_pct >= DAILY_CIRCUIT_BREAKER_PCT:
                break  # stop trading for the day

            # Cap risk so we don't exceed daily DD limit in one trade
            remaining_daily_dd = DAILY_DD_LIMIT_PCT - daily_loss_pct
            risk_pct = min(risk_pct, remaining_daily_dd * 0.9)  # 90% safety margin

            if risk_pct <= 0.05:  # too little room
                break

            risk_amount = equity * (risk_pct / 100.0)

            # Determine outcome
            if np.random.random() < win_rate:
                # Winner - sample from RR distribution
                r = np.random.random()
                cumulative = 0.0
                rr = rr_distribution[-1][1]  # default to last
                for prob, rr_val in rr_distribution:
                    cumulative += prob
                    if r <= cumulative:
                        rr = rr_val
                        break
                pnl = risk_amount * rr - (equity * slippage_pct / 100)
                consecutive_losses = 0
            else:
                # Loser
                pnl = -risk_amount - (equity * slippage_pct / 100)
                consecutive_losses += 1

            equity += pnl
            total_trades += 1
            high_water_mark = max(high_water_mark, equity)

            # Check total DD
            total_dd = ((high_water_mark - equity) / STARTING_BALANCE) * 100
            if total_dd >= TOTAL_DD_LIMIT_PCT:
                hit_total_dd = True
                return SimResult(
                    passed=False, days_to_pass=day, hit_daily_dd=hit_daily_dd,
                    hit_total_dd=True, final_equity_pct=((equity - STARTING_BALANCE) / STARTING_BALANCE) * 100,
                    max_drawdown_pct=total_dd, total_trades=total_trades, daily_dd_events=daily_dd_events
                )

            # Check daily DD
            daily_dd = ((day_start_equity - equity) / day_start_equity) * 100 if equity < day_start_equity else 0
            if daily_dd >= DAILY_DD_LIMIT_PCT:
                daily_dd_events += 1
                hit_daily_dd = True
                # Account blown for the day - in The5ers this is a violation
                return SimResult(
                    passed=False, days_to_pass=day, hit_daily_dd=True,
                    hit_total_dd=False, final_equity_pct=((equity - STARTING_BALANCE) / STARTING_BALANCE) * 100,
                    max_drawdown_pct=total_dd, total_trades=total_trades, daily_dd_events=daily_dd_events
                )

            # Check if passed
            profit_pct = ((equity - STARTING_BALANCE) / STARTING_BALANCE) * 100
            if profit_pct >= PROFIT_TARGET_PCT:
                return SimResult(
                    passed=True, days_to_pass=day, hit_daily_dd=False,
                    hit_total_dd=False, final_equity_pct=profit_pct,
                    max_drawdown_pct=((high_water_mark - equity) / STARTING_BALANCE) * 100 if equity < high_water_mark else 0,
                    total_trades=total_trades, daily_dd_events=0
                )

        # End of day - check daily DD one more time
        daily_dd = ((day_start_equity - equity) / day_start_equity) * 100 if equity < day_start_equity else 0
        if daily_dd >= DAILY_DD_LIMIT_PCT:
            daily_dd_events += 1

    # Ran out of days
    profit_pct = ((equity - STARTING_BALANCE) / STARTING_BALANCE) * 100
    return SimResult(
        passed=profit_pct >= PROFIT_TARGET_PCT, days_to_pass=max_days,
        hit_daily_dd=hit_daily_dd, hit_total_dd=hit_total_dd,
        final_equity_pct=profit_pct,
        max_drawdown_pct=((high_water_mark - min(equity, high_water_mark)) / STARTING_BALANCE) * 100,
        total_trades=total_trades, daily_dd_events=daily_dd_events
    )


def run_simulation(
    win_rate: float,
    rr_distribution: List[Tuple[float, float]],
    risk_schedule: str,
    base_risk_pct: float,
    trades_per_day_range: Tuple[int, int],
    max_days: int = MAX_TRADING_DAYS,
    num_sims: int = NUM_SIMULATIONS,
    pause_after_consecutive_losses: int = 0,
    slippage_pct: float = 0.0,
) -> Dict:

    results = []
    for _ in range(num_sims):
        r = simulate_one_trial(
            win_rate, rr_distribution, risk_schedule, base_risk_pct,
            trades_per_day_range, max_days, pause_after_consecutive_losses, slippage_pct
        )
        results.append(r)

    passed = [r for r in results if r.passed]
    failed_daily_dd = [r for r in results if r.hit_daily_dd]
    failed_total_dd = [r for r in results if r.hit_total_dd]

    pass_rate = len(passed) / len(results) * 100
    avg_days = np.mean([r.days_to_pass for r in passed]) if passed else 0
    median_days = np.median([r.days_to_pass for r in passed]) if passed else 0
    daily_dd_rate = len(failed_daily_dd) / len(results) * 100
    total_dd_rate = len(failed_total_dd) / len(results) * 100
    avg_trades = np.mean([r.total_trades for r in results])

    return {
        'pass_rate': pass_rate,
        'avg_days': avg_days,
        'median_days': median_days,
        'daily_dd_rate': daily_dd_rate,
        'total_dd_rate': total_dd_rate,
        'avg_trades': avg_trades,
        'timeout_rate': 100 - pass_rate - daily_dd_rate - total_dd_rate,
    }


# ============================================================
# ANALYSIS 1: COMPREHENSIVE PASS RATE TABLE
# ============================================================

def analysis_1():
    print("\n" + "=" * 100)
    print("ANALYSIS 1: COMPREHENSIVE PASS RATE TABLE")
    print("=" * 100)

    win_rates = [0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60]
    rr_ratios = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    risk_levels = [0.75, 1.0, 1.25, 1.5]

    # Using phased risk, 1-2 trades/day (avg ~3-5/week with some rest days)
    for risk_pct in risk_levels:
        print(f"\n{'─' * 100}")
        print(f"BASE RISK: {risk_pct}% (Phased: {risk_pct*1.25:.2f}% early → {risk_pct:.2f}% mid → {risk_pct*0.75:.2f}% late)")
        print(f"Trades/day: 1-2 (circuit breaker at 2% daily loss)")
        print(f"{'─' * 100}")

        # Header
        header = f"{'WR':>5} |"
        for rr in rr_ratios:
            header += f" {rr:.1f}:1 Pass% (Days) |"
        print(header)
        print("-" * len(header))

        for wr in win_rates:
            row = f"{wr*100:.0f}%  |"
            for rr in rr_ratios:
                rr_dist = [(1.0, rr)]  # fixed RR for this analysis
                result = run_simulation(
                    win_rate=wr,
                    rr_distribution=rr_dist,
                    risk_schedule='phased',
                    base_risk_pct=risk_pct,
                    trades_per_day_range=(1, 2),
                    max_days=22,
                    num_sims=5000,
                )
                days_str = f"{result['avg_days']:.0f}" if result['pass_rate'] > 0 else "-"
                row += f"  {result['pass_rate']:5.1f}% ({days_str:>3}d)    |"
            print(row)
            sys.stdout.flush()

    # Now show daily DD and total DD risk for key combinations
    print(f"\n{'=' * 100}")
    print("FAILURE MODE BREAKDOWN (1.0% base risk, phased)")
    print(f"{'=' * 100}")

    header = f"{'WR':>5} |"
    for rr in [1.5, 2.0, 2.5, 3.0]:
        header += f" {rr:.1f}:1 DailyDD%/TotalDD%/Timeout% |"
    print(header)
    print("-" * len(header))

    for wr in win_rates:
        row = f"{wr*100:.0f}%  |"
        for rr in [1.5, 2.0, 2.5, 3.0]:
            rr_dist = [(1.0, rr)]
            result = run_simulation(
                win_rate=wr,
                rr_distribution=rr_dist,
                risk_schedule='phased',
                base_risk_pct=1.0,
                trades_per_day_range=(1, 2),
                max_days=22,
                num_sims=5000,
            )
            row += f"  {result['daily_dd_rate']:4.1f}/{result['total_dd_rate']:4.1f}/{result['timeout_rate']:4.1f}      |"
        print(row)
        sys.stdout.flush()


# ============================================================
# ANALYSIS 2: FAT-TAILED "RUNNER" DISTRIBUTION
# ============================================================

def analysis_2():
    print("\n" + "=" * 100)
    print("ANALYSIS 2: FAT-TAILED 'RUNNER' DISTRIBUTION vs FIXED RR")
    print("=" * 100)

    # Fat-tailed distribution: 60% at 1.5:1, 25% at 3:1, 10% at 5:1, 5% at 8:1
    fat_tail_dist = [(0.60, 1.5), (0.25, 3.0), (0.10, 5.0), (0.05, 8.0)]
    # Weighted average RR = 0.6*1.5 + 0.25*3 + 0.10*5 + 0.05*8 = 0.9 + 0.75 + 0.5 + 0.4 = 2.55
    avg_rr = sum(p * r for p, r in fat_tail_dist)
    print(f"\nFat-tail distribution: 60%@1.5R, 25%@3R, 10%@5R, 5%@8R")
    print(f"Weighted average RR: {avg_rr:.2f}:1")
    print(f"Compared against: Fixed 2.0:1 and Fixed 2.5:1")

    fixed_2_dist = [(1.0, 2.0)]
    fixed_25_dist = [(1.0, 2.5)]

    win_rates = [0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60]

    print(f"\n{'WR':>5} | {'Fixed 2.0:1':>20} | {'Fixed 2.5:1':>20} | {'Fat-Tail (avg 2.55)':>20} | {'Delta vs 2.0':>12} | {'Delta vs 2.5':>12}")
    print("-" * 110)

    for wr in win_rates:
        r_fixed2 = run_simulation(wr, fixed_2_dist, 'phased', 1.0, (1, 2), 22, 5000)
        r_fixed25 = run_simulation(wr, fixed_25_dist, 'phased', 1.0, (1, 2), 22, 5000)
        r_fat = run_simulation(wr, fat_tail_dist, 'phased', 1.0, (1, 2), 22, 5000)

        d2 = f"{r_fixed2['pass_rate']:5.1f}% ({r_fixed2['avg_days']:4.1f}d)"
        d25 = f"{r_fixed25['pass_rate']:5.1f}% ({r_fixed25['avg_days']:4.1f}d)"
        df = f"{r_fat['pass_rate']:5.1f}% ({r_fat['avg_days']:4.1f}d)"
        delta2 = f"{r_fat['pass_rate'] - r_fixed2['pass_rate']:+.1f}%"
        delta25 = f"{r_fat['pass_rate'] - r_fixed25['pass_rate']:+.1f}%"

        print(f"{wr*100:.0f}%  | {d2:>20} | {d25:>20} | {df:>20} | {delta2:>12} | {delta25:>12}")
        sys.stdout.flush()

    # Now model: trailing stops reduce WR by ~5-8% but produce fat tails
    print(f"\n{'─' * 100}")
    print("KEY QUESTION: Does fat-tail compensate for lower win rate from trailing stops?")
    print("Assumption: Trailing stops reduce WR by 5-8% compared to fixed TP")
    print(f"{'─' * 100}")

    comparisons = [
        (0.55, 0.48, "55% fixed → 48% trailing"),
        (0.55, 0.50, "55% fixed → 50% trailing"),
        (0.52, 0.45, "52% fixed → 45% trailing"),
        (0.52, 0.48, "52% fixed → 48% trailing"),
        (0.50, 0.45, "50% fixed → 45% trailing"),
        (0.50, 0.48, "50% fixed → 48% trailing"),
    ]

    print(f"\n{'Scenario':<30} | {'Fixed 2:1 Pass%':>15} | {'Fat-Tail Pass%':>15} | {'Winner':>10}")
    print("-" * 85)

    for fixed_wr, trail_wr, label in comparisons:
        r_fixed = run_simulation(fixed_wr, fixed_2_dist, 'phased', 1.0, (1, 2), 22, 5000)
        r_trail = run_simulation(trail_wr, fat_tail_dist, 'phased', 1.0, (1, 2), 22, 5000)
        winner = "Fat-Tail" if r_trail['pass_rate'] > r_fixed['pass_rate'] else "Fixed"
        print(f"{label:<30} | {r_fixed['pass_rate']:>14.1f}% | {r_trail['pass_rate']:>14.1f}% | {winner:>10}")
        sys.stdout.flush()


# ============================================================
# ANALYSIS 3: 30-DAY FEASIBILITY
# ============================================================

def analysis_3():
    print("\n" + "=" * 100)
    print("ANALYSIS 3: 30-DAY FEASIBILITY BY TRADE FREQUENCY")
    print("=" * 100)

    # trades_per_week mapped to trades_per_day
    # 2/week = ~0-1/day, 3/week = ~0-1/day, 5/week = 1/day, 7/week = 1-2/day, 10/week = 1-3/day
    freq_configs = [
        ("2/week", (0, 1)),   # some days 0, some 1
        ("3/week", (0, 1)),
        ("5/week", (1, 1)),   # 1 trade every trading day
        ("7/week", (1, 2)),   # 1-2 per day
        ("10/week", (1, 3)),  # 1-3 per day
    ]

    # For 2/week and 3/week, we need special handling since (0,1) means avg 0.5/day
    # Let me use more precise ranges
    freq_configs = [
        ("~2/wk (10 total)", (0, 1)),
        ("~3/wk (15 total)", (1, 1)),  # ~15 in 22 days with some off days
        ("~5/wk (22 total)", (1, 1)),
        ("~7/wk (30 total)", (1, 2)),
        ("~10/wk (44 total)", (2, 2)),
    ]

    win_rates = [0.48, 0.50, 0.52, 0.55]
    risk_levels = [0.75, 1.0, 1.25, 1.5]
    rr_dist = [(1.0, 2.0)]  # fixed 2:1 baseline

    for wr in win_rates:
        print(f"\n{'─' * 90}")
        print(f"WIN RATE: {wr*100:.0f}%, RR: 2.0:1, Phased Risk")
        print(f"{'─' * 90}")

        header = f"{'Frequency':<22} |"
        for risk in risk_levels:
            header += f" Risk {risk:.2f}% |"
        print(header)
        print("-" * 70)

        for freq_label, tpd in freq_configs:
            row = f"{freq_label:<22} |"
            for risk in risk_levels:
                result = run_simulation(wr, rr_dist, 'phased', risk, tpd, 22, 5000)
                row += f"  {result['pass_rate']:5.1f}%   |"
            print(row)
            sys.stdout.flush()

    # Minimum frequency for >50% pass rate
    print(f"\n{'=' * 90}")
    print("MINIMUM FREQUENCY FOR >50% PASS RATE (22 trading days)")
    print(f"{'=' * 90}")

    for wr in [0.50, 0.52, 0.55]:
        for risk in [1.0, 1.25, 1.5]:
            for tpd_max in range(1, 5):
                result = run_simulation(wr, rr_dist, 'phased', risk, (1, tpd_max), 22, 5000)
                if result['pass_rate'] >= 50:
                    print(f"WR {wr*100:.0f}%, Risk {risk:.1f}%, RR 2:1 → Need {tpd_max} trades/day max → Pass rate: {result['pass_rate']:.1f}%")
                    break
            else:
                print(f"WR {wr*100:.0f}%, Risk {risk:.1f}%, RR 2:1 → Cannot reach 50% pass rate even at 4 trades/day")
            sys.stdout.flush()


# ============================================================
# ANALYSIS 4: OPTIMAL RISK SIZING
# ============================================================

def analysis_4():
    print("\n" + "=" * 100)
    print("ANALYSIS 4: OPTIMAL RISK SIZING FOR 30-DAY TARGET")
    print("=" * 100)

    strategies = [
        ("Fixed 0.75%", 'fixed', 0.75),
        ("Fixed 1.00%", 'fixed', 1.0),
        ("Fixed 1.25%", 'fixed', 1.25),
        ("Fixed 1.50%", 'fixed', 1.5),
        ("Fixed 2.00%", 'fixed', 2.0),
        ("Phased 1.00%", 'phased', 1.0),       # 1.25 → 1.0 → 0.75
        ("Phased 1.25%", 'phased', 1.25),       # 1.56 → 1.25 → 0.94
        ("Adaptive 1.00%", 'adaptive', 1.0),     # 1.5% until 4%, then 0.75%
        ("Adaptive 1.25%", 'adaptive', 1.25),
        ("AntiMart 1.00%", 'anti_martingale', 1.0),
        ("AntiMart 1.25%", 'anti_martingale', 1.25),
    ]

    rr_dist = [(1.0, 2.0)]

    for wr in [0.50, 0.52, 0.55]:
        print(f"\n{'─' * 100}")
        print(f"WIN RATE: {wr*100:.0f}%, RR: 2.0:1, 1-2 trades/day")
        print(f"{'─' * 100}")
        print(f"{'Strategy':<20} | {'Pass%':>6} | {'AvgDays':>7} | {'DailyDD%':>8} | {'TotalDD%':>8} | {'Timeout%':>8}")
        print("-" * 75)

        for label, sched, risk in strategies:
            result = run_simulation(wr, rr_dist, sched, risk, (1, 2), 22, 5000)
            print(f"{label:<20} | {result['pass_rate']:5.1f}% | {result['avg_days']:6.1f}d | {result['daily_dd_rate']:7.1f}% | {result['total_dd_rate']:7.1f}% | {result['timeout_rate']:7.1f}%")
        sys.stdout.flush()

    # Also test with fat-tail distribution
    fat_tail_dist = [(0.60, 1.5), (0.25, 3.0), (0.10, 5.0), (0.05, 8.0)]
    print(f"\n{'─' * 100}")
    print(f"WITH FAT-TAIL DISTRIBUTION (avg 2.55:1), WR 50%, 1-2 trades/day")
    print(f"{'─' * 100}")
    print(f"{'Strategy':<20} | {'Pass%':>6} | {'AvgDays':>7} | {'DailyDD%':>8} | {'TotalDD%':>8} | {'Timeout%':>8}")
    print("-" * 75)

    for label, sched, risk in strategies:
        result = run_simulation(0.50, fat_tail_dist, sched, risk, (1, 2), 22, 5000)
        print(f"{label:<20} | {result['pass_rate']:5.1f}% | {result['avg_days']:6.1f}d | {result['daily_dd_rate']:7.1f}% | {result['total_dd_rate']:7.1f}% | {result['timeout_rate']:7.1f}%")
    sys.stdout.flush()


# ============================================================
# ANALYSIS 5: DAILY DRAWDOWN DEEP DIVE
# ============================================================

def analysis_5():
    print("\n" + "=" * 100)
    print("ANALYSIS 5: DAILY DRAWDOWN DEEP DIVE")
    print("=" * 100)

    # Analytical probability of hitting 5% daily DD
    print("\n--- Analytical: Probability of hitting 5% daily loss ---")
    print("(Assumes all losses hit in sequence on same day)")

    configs = [
        ("2 trades/day @ 1.0%", 2, 1.0),
        ("2 trades/day @ 1.25%", 2, 1.25),
        ("2 trades/day @ 1.5%", 2, 1.5),
        ("3 trades/day @ 1.0%", 3, 1.0),
        ("3 trades/day @ 1.25%", 3, 1.25),
        ("3 trades/day @ 1.5%", 3, 1.5),
    ]

    for wr in [0.48, 0.50, 0.52, 0.55]:
        print(f"\nWin Rate: {wr*100:.0f}%")
        print(f"{'Config':<25} | {'Max Daily Loss':>15} | {'P(all lose)':>12} | {'P(5% DD)':>10} | {'P(5% DD w/ slip)':>16}")
        print("-" * 90)

        for label, n_trades, risk in configs:
            max_loss = n_trades * risk
            p_all_lose = (1 - wr) ** n_trades

            # Can we hit 5% DD?
            trades_to_5pct = int(np.ceil(5.0 / risk))
            if trades_to_5pct <= n_trades:
                p_5pct = (1 - wr) ** trades_to_5pct
            else:
                p_5pct = 0.0  # Can't reach 5% in one day

            # With slippage ($3-5 per stop on gold, roughly 0.03-0.05% on $10K)
            effective_risk = risk + 0.04  # ~$4 slippage
            trades_to_5pct_slip = int(np.ceil(5.0 / effective_risk))
            if trades_to_5pct_slip <= n_trades:
                p_5pct_slip = (1 - wr) ** trades_to_5pct_slip
            else:
                p_5pct_slip = 0.0

            print(f"{label:<25} | {max_loss:>14.1f}% | {p_all_lose:>11.2%} | {p_5pct:>9.2%} | {p_5pct_slip:>15.2%}")

    # Monte Carlo: daily DD probability
    print(f"\n{'─' * 80}")
    print("Monte Carlo: Daily DD hit rate over 22 trading days")
    print(f"{'─' * 80}")

    for wr in [0.50, 0.52, 0.55]:
        print(f"\nWR {wr*100:.0f}%, RR 2.0:1, Phased risk")
        for n_trades_max in [1, 2, 3]:
            for risk in [0.75, 1.0, 1.25, 1.5]:
                result = run_simulation(
                    wr, [(1.0, 2.0)], 'phased', risk, (1, n_trades_max), 22, 5000
                )
                print(f"  {n_trades_max} max trades/day, {risk:.2f}% risk → Daily DD blow: {result['daily_dd_rate']:.1f}%, Total DD blow: {result['total_dd_rate']:.1f}%")
        sys.stdout.flush()

    # Should you take trade 2 after losing trade 1?
    print(f"\n{'─' * 80}")
    print("CONDITIONAL ANALYSIS: Should you trade after a losing trade 1?")
    print("Comparing: always take trade 2 vs skip trade 2 after loss")
    print(f"{'─' * 80}")

    for wr in [0.50, 0.52, 0.55]:
        # "Always trade 2": (1,2) trades/day
        r_always = run_simulation(wr, [(1.0, 2.0)], 'phased', 1.0, (1, 2), 22, 5000)
        # "Only 1 trade/day" (conservative)
        r_one = run_simulation(wr, [(1.0, 2.0)], 'phased', 1.0, (1, 1), 22, 5000)

        print(f"WR {wr*100:.0f}%: Always 1-2 trades → {r_always['pass_rate']:.1f}% pass, {r_always['daily_dd_rate']:.1f}% daily DD")
        print(f"WR {wr*100:.0f}%: Max 1 trade/day  → {r_one['pass_rate']:.1f}% pass, {r_one['daily_dd_rate']:.1f}% daily DD")
        print()


# ============================================================
# ANALYSIS 6: LOSING STREAK ANALYSIS
# ============================================================

def analysis_6():
    print("\n" + "=" * 100)
    print("ANALYSIS 6: LOSING STREAK ANALYSIS")
    print("=" * 100)

    # Analytical consecutive loss probabilities
    print("\n--- Probability of N consecutive losses ---")
    print(f"{'Streak':>8} |", end="")
    for wr in [0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60]:
        print(f" WR {wr*100:.0f}%  |", end="")
    print()
    print("-" * 80)

    for streak in [2, 3, 4, 5, 6, 7, 8]:
        row = f"{streak:>8} |"
        for wr in [0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60]:
            p = (1 - wr) ** streak
            row += f" {p:>6.2%}  |"
        print(row)

    # Probability of experiencing streak at least once in N trades
    print(f"\n--- P(at least one streak of N losses) in 22 trades ---")
    for wr in [0.50, 0.52, 0.55]:
        print(f"\nWR = {wr*100:.0f}%, 22 trades:")
        for streak in [3, 4, 5, 6]:
            # Approximate: P(at least one) ≈ 1 - (1 - p_streak)^(N-streak+1)
            p_streak = (1 - wr) ** streak
            n_windows = 22 - streak + 1
            p_at_least_one = 1 - (1 - p_streak) ** n_windows
            print(f"  P(streak of {streak}+ at least once) ≈ {p_at_least_one:.1%}")

    # Impact of 3-loss pause rule on pass rate
    print(f"\n{'─' * 80}")
    print("3-LOSS PAUSE RULE: Impact on pass rate")
    print("Rule: After 3 consecutive losses, stop trading for rest of day")
    print(f"{'─' * 80}")

    for wr in [0.48, 0.50, 0.52, 0.55]:
        r_no_pause = run_simulation(wr, [(1.0, 2.0)], 'phased', 1.0, (1, 2), 22, 5000, pause_after_consecutive_losses=0)
        r_pause_3 = run_simulation(wr, [(1.0, 2.0)], 'phased', 1.0, (1, 2), 22, 5000, pause_after_consecutive_losses=3)
        r_pause_2 = run_simulation(wr, [(1.0, 2.0)], 'phased', 1.0, (1, 2), 22, 5000, pause_after_consecutive_losses=2)

        print(f"WR {wr*100:.0f}%:")
        print(f"  No pause:     {r_no_pause['pass_rate']:.1f}% pass, {r_no_pause['daily_dd_rate']:.1f}% daily DD blow")
        print(f"  Pause after 2: {r_pause_2['pass_rate']:.1f}% pass, {r_pause_2['daily_dd_rate']:.1f}% daily DD blow")
        print(f"  Pause after 3: {r_pause_3['pass_rate']:.1f}% pass, {r_pause_3['daily_dd_rate']:.1f}% daily DD blow")
        sys.stdout.flush()


# ============================================================
# RUN ALL ANALYSES
# ============================================================

if __name__ == "__main__":
    print("MONTE CARLO PROP FIRM CHALLENGE ANALYSIS")
    print(f"Simulations per scenario: {NUM_SIMULATIONS}")
    print(f"Account: ${STARTING_BALANCE:.0f}, Target: +{PROFIT_TARGET_PCT}%, Daily DD: {DAILY_DD_LIMIT_PCT}%, Total DD: {TOTAL_DD_LIMIT_PCT}%")
    print(f"Max trading days: {MAX_TRADING_DAYS}")

    analysis_1()
    analysis_2()
    analysis_3()
    analysis_4()
    analysis_5()
    analysis_6()

    print("\n\nALL ANALYSES COMPLETE.")
