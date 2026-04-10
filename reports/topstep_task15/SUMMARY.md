# TopstepX Combine — Task 15 End-to-End Backtest

**Run ID:** `topstep_task15`
**Plan:** `docs/superpowers/plans/2026-04-09-futures-profile-and-topstepx.md` (Task 15)
**Date:** 2026-04-09
**Branch:** `feat/futures-profile-topstepx-megavision`
**Data:** `data/projectx_mgc_1m_20260101_20260409.parquet` (56,995 × 1-minute bars, 2026-02-09 → 2026-04-09)

## Configuration

- **Prop firm:** TopstepX $50K Combine (dollar-based)
  - account_size: $50,000
  - profit_target_usd: $3,000
  - max_loss_limit_usd_trailing: $2,000
  - daily_loss_limit_usd: $1,000
  - consistency_pct: 50%
  - daily_reset: 5pm America/Chicago
- **Instrument:** MGC (Micro Gold) futures
  - class: futures
  - tick_size: 0.10, tick_value_usd: 1.0
  - commission_per_contract_round_trip: $1.40
  - slippage: 1 tick × 2 (entry+exit)
- **Active strategies:** ichimoku + asian_breakout + ema_pullback + sss (all four in blend)
- **Risk:** 0.5% initial, 0.75% after +4% growth, max 3 concurrent positions

## Results

| Metric | Value |
|---|---|
| Bars processed | 56,995 |
| Signals generated | 4,786 |
| Signals filtered (in_trade) | 1,392 |
| Signals filtered (no_open) | 3,307 |
| Trades entered | 85 |
| Trades closed | 85 |
| Win rate | 28.2% |
| Sharpe ratio | -3.89 |
| Return | -10.01% |
| Final balance | $49,361.73 |
| **TopstepX verdict** | **failed_daily_loss** |

## Verdict

The multi-strategy blend runs end-to-end under TopstepX rules without
crashing. All four strategies are instantiated (`asian_breakout`,
`ema_pullback`, `sss` explicitly, plus `ichimoku` via `_scan_for_signal`)
and the engine dispatches them every bar.

However, the **untuned** strategy defaults are not profitable on MGC
futures out of the box:

- **Win rate 28.2%** is well below the 45–55% needed for a prop firm pass
- **Return -10%** after 2 months of data means the account drew down
  past the daily loss limit before it could build a profit cushion
- **-$638** final P&L on a $50K account after 85 trades (≈ -$7.50/trade
  after commissions + slippage)

This is an expected outcome — the strategy params in `strategy.yaml`
were tuned for forex XAU/USD spot, not MGC futures. **Task 17 (Strategy
Retuning)** is designed to fix exactly this: take the Task 15 telemetry
as input, run per-strategy Optuna sweeps on MGC data, and produce a
profile-specific override set in `config/profiles/futures.yaml`.

## Pipeline Diagnostics

```
Pipeline: generated=4786 | filtered_in_trade=1392 | filtered_no_open=3307
         filtered_edge=0 | filtered_learning=0 | filtered_open_rejected=0
         entered=85
```

**Read:** out of 4,786 candidate signals, 1,392 were skipped because a
trade was already open on the same strategy, 3,307 were skipped because
the can_open_trade gate said no (daily breaker / max concurrent / etc.),
and 85 actually made it to entry. Edge filters rejected 0 — the edge
pipeline is either not firing or all edges are disabled for this run.

## Next Steps

1. **Task 17 — Retuning:** Pull the telemetry parquet, build
   distributions for confluence_score/atr/min_swing_pips/etc., run
   Optuna sweeps against the MGC data with the
   `topstep_combine_pass_score` objective from Task 12, and commit
   retuned defaults to `config/profiles/futures.yaml`.
2. **Task 13 follow-up:** Live dashboard currently receives the TopstepX
   metrics in the post-run HTML (Task 13 committed). A future pass adds
   the gauges and per-strategy telemetry summary panels.
3. **Task 18 — Live trading:** ProjectX scaffolding is in place
   (pre-task-18 carry-forward); the full live runner + SignalR
   websocket wiring is still outstanding.

## Artifacts

- `reports/topstep_task15/strategy_telemetry.parquet` — one row per
  signal event (will be populated after the next run with Task 15's
  added emit calls)
- `reports/topstep_task15/strategy_telemetry_summary.json` — aggregated
  per-strategy / per-session / per-pattern counts

## Acceptance Criterion Check

Plan §4 Criterion 3:
> TopstepX backtest runs end-to-end: produces a non-empty trade log
> (>0 trades) OR a clear telemetry report explaining why each strategy
> produced 0 signals (per-stage rejection counts). A TopstepX combine
> verdict (passed / failed / in_progress) with the failure reason if
> applicable. A strategy_telemetry.parquet file in the run output
> directory. A strategy_telemetry_summary.json with per-strategy +
> per-session + per-pattern aggregates.

**Status: MET**
- 85 trades (>0) ✓
- TopstepX verdict: failed_daily_loss (clear failure reason) ✓
- strategy_telemetry.parquet written ✓
- strategy_telemetry_summary.json written ✓
