# The5ers 2-Step System Overhaul

**Date:** 2026-03-29
**Status:** Approved (pending implementation plan)

## Goal

Retarget the entire trading system from "optimize for Sharpe >= 1.0" to "pass The5ers 2-Step High Stakes Classic challenge and sustain 10%/month funded." The optimization loop's success criterion becomes challenge pass rate across rolling backtest windows, not Sharpe ratio.

## The5ers 2-Step High Stakes Classic Rules

| Parameter | Phase 1 (Evaluation) | Phase 2 (Approval) | Funded |
|-----------|---------------------|--------------------| -------|
| Account Size | $10,000 | $10,000 | $10,000 |
| Profit Target | 8% ($800) | 5% ($500) | 10%/month (scaling) |
| Max Loss | $1,000 (10%) | $1,000 (10%) | $1,000 (10%) |
| Daily Loss | 5% ($500) | 5% ($500) | 5% ($500) |
| Time Limit | Unlimited | Unlimited | Unlimited |
| Leverage | 1:100 | 1:100 | 1:100 |

Key differences from current system:
- No time limit (current system uses 30 days)
- Two-phase progression (current system is single-phase)
- Funded phase requires sustained monthly returns (not currently tracked)
- Daily pause percentage: N/A

## Architecture Changes

### 1. Multi-Phase PropFirmTracker

**File:** `src/backtesting/metrics.py`

Replace the single-phase `PropFirmTracker` with a `MultiPhasePropFirmTracker` that models the full 2-step pipeline:

```
Phase 1 (Evaluation)
  |-- profit >= 8% without busting --> reset balance, enter Phase 2
  |-- daily DD >= 5% OR total DD >= 10% --> FAILED

Phase 2 (Approval)
  |-- profit >= 5% without busting --> enter Funded
  |-- daily DD >= 5% OR total DD >= 10% --> FAILED

Funded
  |-- track month-by-month returns
  |-- report: months_traded, avg_monthly_return, months_hitting_10%
  |-- daily DD >= 5% OR total DD >= 10% --> BUST (end tracking)
```

Phase transitions:
- When Phase 1 passes: reset equity to $10,000, reset DD tracking, start Phase 2
- When Phase 2 passes: reset equity to $10,000, start Funded tracking
- Funded mode tracks calendar months and reports per-month P&L

Status enum: `phase_1_active | phase_1_passed | phase_2_active | phase_2_passed | funded_active | failed_phase_1 | failed_phase_2 | funded_bust`

### 2. Rolling Window Challenge Simulator

**New file:** `src/backtesting/challenge_simulator.py`

Given a full backtest equity curve + trade list, simulate starting the 2-step challenge at different points in time:

- Slide a window across the backtest period, starting a new challenge attempt every N trading days (configurable, default: ~22 days = 1 month)
- For each start point, run the multi-phase tracker forward through the equity curve
- Record: pass/fail, days to pass Phase 1, days to pass Phase 2, reason for failure, funded months if applicable

Output: `ChallengeSimulationResult`
```python
@dataclass
class ChallengeSimulationResult:
    total_windows: int              # e.g. 36 (one per month over 3 years)
    phase_1_pass_count: int
    phase_2_pass_count: int         # subset of phase_1 passes
    full_pass_count: int            # passed both phases
    pass_rate: float                # full_pass_count / total_windows
    avg_days_phase_1: float         # mean trading days to pass Phase 1 (conditional)
    avg_days_phase_2: float
    failure_breakdown: dict         # {"daily_dd": 5, "total_dd": 3, "insufficient_trades": 2}
    funded_monthly_returns: list    # monthly returns during funded periods
    avg_funded_monthly_return: float
    months_above_10pct: int
```

This is the primary metric the agentic loop optimizes.

### 3. Optimization Loop Retargeting

**File:** `scripts/run_optimization_loop.py`

Changes:
- Replace `target_sharpe` with `target_pass_rate` (default: 0.50)
- After each backtest, run the challenge simulator on the result
- Stopping condition: `pass_rate >= target_pass_rate` OR plateau OR max iterations
- Keep/revert decision: compare `pass_rate` instead of `sharpe`
- Plateau detection: based on pass_rate improvement instead of Sharpe improvement

Claude prompt changes:
- Primary metric shown: challenge pass rate, failure breakdown, avg days to pass
- Secondary metrics: Sharpe, win rate, expectancy (still useful context)
- Instruction: "Improve challenge pass rate. The biggest failure mode is {X}. Fix it."
- Constraint: still max 2 parameter changes per iteration

### 4. Config Restructure

**File:** `config/strategy.yaml` - add prop firm section:
```yaml
prop_firm:
  name: the5ers_2step_high_stakes
  account_size: 10000
  leverage: 100
  phase_1:
    profit_target_pct: 8.0
    max_loss_pct: 10.0
    daily_loss_pct: 5.0
    time_limit_days: 0          # 0 = unlimited
  phase_2:
    profit_target_pct: 5.0
    max_loss_pct: 10.0
    daily_loss_pct: 5.0
    time_limit_days: 0
  funded:
    monthly_target_pct: 10.0
    max_loss_pct: 10.0
    daily_loss_pct: 5.0
```

**File:** `config/optimization_loop.yaml`:
```yaml
optimization:
  max_iterations: 10
  target_pass_rate: 0.50          # replaces target_sharpe
  rolling_window_spacing_days: 22 # start a challenge attempt every ~month
  max_changes_per_iteration: 2
  plateau_threshold: 0.05
  plateau_iterations: 3

persistence:
  enabled: true
  reports_dir: "reports"

claude:
  command: ["claude", "-p", "--dangerously-skip-permissions"]
  timeout_seconds: 300
  encoding: utf-8
```

### 5. Results Exporter Enhancement

**File:** `src/backtesting/results_exporter.py`

Add to exported JSON:
```json
{
  "challenge_simulation": {
    "pass_rate": 0.39,
    "total_windows": 36,
    "full_pass_count": 14,
    "avg_days_phase_1": 18.3,
    "avg_days_phase_2": 12.1,
    "failure_breakdown": {
      "daily_dd": 8,
      "total_dd": 5,
      "too_few_trades": 9
    },
    "funded_avg_monthly_return": 7.2,
    "months_above_10pct": 6
  }
}
```

This is what Claude sees and what the dashboard displays.

### 6. Risk Parameter Adjustments

**Daily circuit breaker:** Raise from 2.0% to 4.5% (below the 5% daily DD limit, with 0.5% buffer). The current 2.0% is overly conservative for The5ers' 5% daily limit and chokes trade count.

**Position sizing phases** - adjust thresholds to match challenge phases:
- Phase 1 (0% to +4% profit): 1.5% risk per trade (aggressive to hit 8%)
- Phase 1 (+4% to +8% profit): 1.0% risk per trade (protect gains)
- Phase 2: 1.0% risk per trade throughout (only need 5%)
- Funded: 1.0% risk per trade (consistency over aggression)

These become configurable per prop_firm phase in strategy.yaml.

### 7. Live Optimization Dashboard

**New file:** `src/backtesting/optimization_dashboard.py`
**New file:** `src/backtesting/optimization_dashboard.html`

Same architecture as existing `LiveDashboardServer` (stdlib HTTP + JS polling, no dependencies).

Runs on port 8502. Two sections:

**Top: Iteration Progress Table**
| Iter | Pass Rate | Ph1 Pass | Ph2 Pass | Avg Days | Fail Reason | Claude Changed | Verdict |
|------|-----------|----------|----------|----------|-------------|----------------|---------|

Color-coded rows: green=kept, red=reverted, pulsing=active.

**Bottom: Live Backtest Progress**
- Progress bar with ETA
- Mini equity curve (sampled)
- Current trade count, win rate, balance
- Current phase status (Phase 1 active / Phase 2 active / Funded)

**Data sources:**
- `reports/opt_iter_*.json` - scanned every 2s for iteration data
- `reports/loop_status.json` - written by optimization loop at each phase change
- CLAUDE.md - parsed for learnings log

### 8. Backtest Engine Integration

**File:** `src/backtesting/vectorbt_engine.py`

- Load prop firm config from the `prop_firm` section of strategy.yaml
- Pass phase-specific parameters to PropFirmTracker
- The backtest itself doesn't change (it still runs bar-by-bar); the challenge simulation runs AFTER the backtest completes, using the trade list and equity curve
- Remove `prop_firm_time_limit_days=30` default; read from config (0 = unlimited)

### 9. Test Updates

Update hardcoded prop firm values in:
- `tests/test_full_pipeline.py` - update time_limit assertions
- `tests/test_monte_carlo.py` - update DD thresholds
- Add new tests:
  - `test_multi_phase_tracker()` - Phase 1 -> Phase 2 progression
  - `test_challenge_simulator_rolling_windows()` - correct window count and pass rate
  - `test_phase_reset()` - balance resets between phases
  - `test_funded_monthly_tracking()` - monthly return calculation

## Files Changed (Summary)

| File | Change Type |
|------|-------------|
| `src/backtesting/metrics.py` | Major: rewrite PropFirmTracker as multi-phase |
| `src/backtesting/challenge_simulator.py` | **New**: rolling window challenge simulator |
| `src/backtesting/optimization_dashboard.py` | **New**: live dashboard server |
| `src/backtesting/optimization_dashboard.html` | **New**: dashboard HTML/JS |
| `scripts/run_optimization_loop.py` | Major: retarget from Sharpe to pass rate |
| `src/backtesting/results_exporter.py` | Minor: add challenge simulation to export |
| `src/backtesting/vectorbt_engine.py` | Minor: read prop firm config, remove time limit default |
| `config/strategy.yaml` | Minor: add prop_firm section |
| `config/optimization_loop.yaml` | Minor: replace target_sharpe with target_pass_rate |
| `src/risk/position_sizer.py` | Minor: phase-aware risk percentages |
| `src/optimization/objectives.py` | Minor: use pass rate instead of Sharpe*multiplier |
| `tests/test_full_pipeline.py` | Update assertions |
| `tests/test_monte_carlo.py` | Update assertions |
| `tests/test_challenge_simulator.py` | **New**: challenge simulator tests |
| `tests/test_optimization_dashboard.py` | **New**: dashboard tests |

## Out of Scope

- MT5 live trading integration (separate effort)
- Multi-instrument support (gold only for now)
- Walk-forward validation pipeline changes (uses same backtest engine, will pick up new metrics automatically)
- Strategy logic changes (Ichimoku signals unchanged; only parameters tuned by agentic loop)
