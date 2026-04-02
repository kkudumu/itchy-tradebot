# Forward Test Pipeline — Post-Challenge Funded Simulation with Continued Discovery

## Problem

The discovery loop finds SHAP rules and Claude hypotheses but never applies them. There's no simulation of what happens after passing The5ers challenge — the funded phase where you trade real money month-over-month. And there's no mechanism for continued learning during funded trading.

## Architecture

Three-stage pipeline on contiguous data:

```
Stage 1: Discovery (2023-2024)
  - Run 22-day rolling windows
  - SHAP analysis every 3 windows
  - Claude generates hypotheses
  - APPLY top hypothesis config changes to subsequent windows (close the loop)
  - Revert if performance degrades
  - Output: optimized config with validated rules

Stage 2: Challenge Simulation (Jan-Feb 2025)
  - Run optimized config through Phase 1 (8% target, 10% max DD, 5% daily DD)
  - If Phase 1 passes, continue to Phase 2 (5% target, same DD limits)
  - NO config changes during challenge — pure execution
  - DD tracking resets between phases (matches The5ers rules)
  - Output: pass/fail verdict, trades, equity curve

Stage 3: Funded Forward Test (Mar-Dec 2025)
  - Trade with validated config, 10% max DD constraint
  - Monthly P&L tracking (target: consistent profitability)
  - CONTINUED DISCOVERY with cautious leash:
    - SHAP analysis every 3 months on accumulated funded trades
    - Claude hypotheses for new patterns in 2025 market conditions
    - Only apply new rules if:
      a) Walk-forward validated (2+ OOS windows)
      b) Max DD stays under 5% (half the 10% funded limit)
    - Compare: base config vs evolved config performance
  - Output: monthly P&L report, new edges discovered, funded survival status
```

## Fixes to Existing Code

### Fix 1: Close the Apply Loop (Discovery Phase)

Currently `DiscoveryOrchestrator.process_window()` runs SHAP and generates hypotheses but never applies them.

**Change:** After `run_full_cycle()` returns hypotheses, call `apply_hypothesis_to_config()` from `src/discovery/rule_applier.py` to merge the top hypothesis into the config for the next window. Track what changed. If the next window's performance degrades (lower pass rate or higher DD), revert the change.

**Where:** `src/discovery/orchestrator.py` — `process_window()` and `run()` methods.

### Fix 2: Forward Test Runner

New file: `src/discovery/forward_test.py`

**ForwardTestRunner** class:
- Takes: optimized config (from discovery), candle data (2025), account_size
- Phase 1: Run IchimokuBacktester for first 22 trading days with MultiPhasePropFirmTracker
  - If profit >= 8% and DD limits respected → Phase 1 passed
  - If DD breached → challenge failed, stop
- Phase 2: Reset DD tracking, run next 22 trading days
  - If profit >= 5% → Phase 2 passed, enter funded
  - If DD breached → challenge failed, stop
- Funded: Run remaining data in monthly chunks (22 trading days each)
  - Track monthly return, max DD, daily DD
  - Run discovery engine (SHAP every 3 months) with cautious settings
  - Apply only walk-forward validated rules where DD stays under 5%
  - Log monthly P&L report

### Fix 3: Live Dashboard Integration

Wire `LiveDashboardServer` into:
- `run_discovery_loop.py` — pass to each window's backtest
- `ForwardTestRunner` — pass to challenge and funded backtests

The dashboard shows real-time: equity curve, trade markers on candlestick chart, backtest progress, P&L metrics.

## Funded Discovery Settings

| Setting | Value | Rationale |
|---------|-------|-----------|
| SHAP interval | Every 3 months (~66 trading days) | Enough trades for reliable analysis |
| Min trades for SHAP | 50 | Statistical significance |
| Max DD for new rules | 5% | Half the 10% funded limit — safety margin |
| Walk-forward requirement | 2+ OOS windows | Same as discovery phase |
| Config changes per month | Max 1 | Cautious — don't over-optimize live |
| Revert threshold | Next month DD > 7% | Auto-revert if new rules hurt performance |

## Monthly Report Format

```
Month N: Mar 2025
  Starting balance: $10,800
  Ending balance:   $11,340
  Return:           +5.0%
  Max DD:           2.3%
  Daily DD peak:    1.1%
  Trades:           28
  Win rate:          39%
  New rules applied: 0
  Status:           FUNDED (healthy)
```

## Entry Points

**Full pipeline (discovery → challenge → funded):**
```bash
python scripts/run_forward_test.py \
    --discovery-data data/xauusd_1m_2023_2024.parquet \
    --forward-data data/xauusd_1m_2025.parquet \
    --strategy sss \
    --enable-claude
```

**Forward test only (skip discovery, use current config):**
```bash
python scripts/run_forward_test.py \
    --forward-data data/xauusd_1m_2025.parquet \
    --strategy sss \
    --skip-discovery
```

## Success Criteria

1. Discovery phase applies rules and improves pass rate over baseline
2. Challenge simulation passes Phase 1 + Phase 2 on 2025 data
3. Funded phase is profitable for 6+ of 10 months
4. Max DD never exceeds 10% during funded phase
5. New patterns discovered in 2025 are logged even if not applied
6. Live dashboard shows real-time charts during all phases
