# Adaptive Strategy Optimizer — Design Spec

## Overview

A continuously running optimization loop that backtests the 5-strategy blender across four CME micro futures instruments (MGC, MCL, MNQ, MYM), persists every trial result to PostgreSQL + pgvector, and uses embedding similarity to warm-start future runs. The system gets smarter with every trial — wins and losses both teach. It never stops iterating.

## Instruments

| Symbol | Product | Symbol ID | Data Available |
|--------|---------|-----------|----------------|
| MGC | Micro Gold | F.US.MGC | ~Jan 2024 onwards |
| MCL | Micro Crude Oil | F.US.MCLE | ~Feb 2026 onwards |
| MNQ | Micro E-mini Nasdaq | F.US.MNQE | ~Feb 2026 onwards |
| MYM | Micro E-mini Dow | F.US.MYME | ~Feb 2026 onwards |

Adding a new instrument = one entry in `config/optimizer_instruments.yaml` with `symbol` and `symbol_id`. System auto-discovers contract specs from ProjectX API and starts optimizing.

## Core Loop

```
for each epoch (forever):
    for each instrument in [MGC, MCL, MNQ, MYM]:
        1. Download/update data from ProjectX API
        2. Embed current 30-day market context (20-dim vector)
        3. Query pgvector for similar past successes → extract warm-start params
        4. Query pgvector for similar past failures → avoid those param regions
        5. Run Optuna optimization (50-100 trials)
           - Each trial: full backtest + combine simulation
           - Persist EVERY trial to DB (embedding + signal log + outcome)
        6. Best trial passes guardrails?
           - 3 consecutive combine passes
           - Permutation test p < 0.05
           - If YES → save to proven_configs, log success
           - If NO → log failure, move on (DB is richer now)
        7. Re-validate existing proven configs if data has changed

    sleep until next trading day (or new data available)
```

## Database Schema

Uses existing PG16 on port 5433 with pgvector extension.

### Table: `optimization_runs`

One row per Optuna trial.

| Column | Type | Description |
|--------|------|-------------|
| run_id | UUID PK | Unique trial identifier |
| instrument | TEXT | 'MGC', 'MCL', 'MNQ', 'MYM' |
| timestamp | TIMESTAMPTZ | When this trial ran |
| data_start | TIMESTAMPTZ | Backtest window start |
| data_end | TIMESTAMPTZ | Backtest window end |
| market_embedding | VECTOR(20) | Market context for similarity search |
| params_embedding | VECTOR(24) | Strategy params normalized to [0,1] |
| outcome_embedding | VECTOR(20) | Result fingerprint |
| full_params | JSONB | Complete param dict (human-readable) |
| active_strategies | TEXT[] | Which strategies were active |
| outcome | JSONB | {passed, balance, win_rate, pf, max_dd, trades, p_value} |
| passed_combine | BOOLEAN | Passed 3x consecutive combine |
| passed_permutation | BOOLEAN | Permutation p < 0.05 |
| proven | BOOLEAN | Passed both guardrails |
| epoch | INTEGER | Which loop cycle produced this |

Index: IVFFlat on `market_embedding` for pgvector similarity search. Btree on `(instrument, proven, timestamp)`.

### Table: `signal_log`

One row per signal per trial. Captures every signal generated, whether it was filtered, and what happened if it entered.

| Column | Type | Description |
|--------|------|-------------|
| id | BIGSERIAL PK | Auto-increment |
| run_id | UUID FK → optimization_runs | Parent trial |
| timestamp | TIMESTAMPTZ | Signal timestamp |
| strategy_name | TEXT | 'sss', 'ichimoku', 'asian_breakout', 'ema_pullback', 'fx_at_one_glance' |
| direction | TEXT | 'long' or 'short' |
| confluence_score | INTEGER | Raw confluence score |
| entry_price | FLOAT | Proposed entry |
| stop_loss | FLOAT | Proposed SL |
| take_profit | FLOAT | Proposed TP |
| filtered_by | TEXT | NULL if entered, else filter name ('trend_direction', 'sizing_cap', 'circuit_breaker', etc.) |
| entered | BOOLEAN | Whether the signal became a trade |
| trade_result_r | FLOAT | R-multiple if entered, NULL if filtered |
| exit_reason | TEXT | 'stop_hit', 'tp_hit', 'kijun_trail', 'friday_close', etc. |
| pnl_usd | FLOAT | Dollar P&L if entered |
| market_snapshot | JSONB | {atr, adx, close, session, htf_trend_1h, htf_trend_4h} at signal time |

Index: Btree on `(run_id, strategy_name)`. Btree on `(instrument, strategy_name, direction, entered)` for pattern queries.

Note: `instrument` is not on this table directly — join through `optimization_runs.run_id` to get it. This avoids redundancy since all signals in a trial share the same instrument.

### Table: `proven_configs`

Only configs that passed all guardrails.

| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL PK | Auto-increment |
| instrument | TEXT | Which instrument |
| timestamp | TIMESTAMPTZ | When proven |
| run_id | UUID FK → optimization_runs | Source trial |
| params | JSONB | Full param dict |
| win_rate | FLOAT | Backtest win rate |
| total_return_pct | FLOAT | Backtest return |
| p_value | FLOAT | Permutation test result |
| combine_passes | INTEGER | How many consecutive (>= 3) |
| data_start | TIMESTAMPTZ | Data window start |
| data_end | TIMESTAMPTZ | Data window end |
| active | BOOLEAN DEFAULT TRUE | Currently active config |
| superseded_at | TIMESTAMPTZ | NULL until replaced |

## Embedding Architecture

64-dimension vector split into three layers, stored as three separate VECTOR columns for flexible querying:

### Market Context (20 dims)

| Dims | Feature Group | Components |
|------|--------------|------------|
| 0-3 | Trend structure | EMA slope 1H, EMA slope 4H, price vs kijun 1H (normalized), price vs kijun 4H (normalized) |
| 4-7 | Volatility | ATR mean (normalized by price), ATR std/mean ratio, ATR trend (expanding=1 / contracting=0), high-low range mean |
| 8-11 | Price action | Return mean, return std, return skew, return kurtosis |
| 12-15 | Drawdown/runup | Max drawdown in window, max runup, autocorrelation lag-1, autocorrelation lag-5 |
| 16-19 | Instrument | tick_size (log-normalized), tick_value_usd (log-normalized), contract_size (log-normalized), point_value (log-normalized) |

All values normalized to [0, 1] via min-max scaling with empirically determined bounds.

### Strategy Params (24 dims)

| Dims | Feature Group | Components |
|------|--------------|------------|
| 0-4 | Risk/Exit | initial_risk_pct, reduced_risk_pct, daily_cb_pct, max_concurrent, tp_r_multiple (all normalized) |
| 5-11 | SSS | swing_lookback, min_swing_pips, min_stop_pips, confluence, rr_ratio, entry_mode (one-hot: 3 values), spread_mult |
| 12-17 | Ichimoku | scale, adx_threshold, atr_mult, confluence, tier_c, timeframe_mode |
| 18-20 | Asian Breakout | min_range, max_range, rr_ratio |
| 21-23 | EMA Pullback | angle_min, pullback_max, rr_ratio |

### Outcome Fingerprint (20 dims)

| Dims | Feature Group | Components |
|------|--------------|------------|
| 0-4 | Core metrics | win_rate, profit_factor (clamped 0-5, normalized), total_return_pct, sharpe_ratio, max_drawdown_pct |
| 5-9 | Trade profile | total_trades (log-normalized), avg_r_multiple, best_trade_r, worst_trade_r, avg_trade_duration_bars |
| 10-14 | Combine | passed (0/1), final_balance (normalized), distance_to_target, best_day_profit, consistency_ratio |
| 15-19 | Robustness | permutation_p_value, n_permutations_beaten, edge_filtered_pct, signals_entered_pct, win_rate_long vs win_rate_short (asymmetry) |

## Warm-Start Logic

```python
def get_warm_start_params(instrument: str, market_embedding: Vector20) -> list[dict]:
    """Query DB for params that worked in similar market conditions."""

    # Find successful runs in similar markets (cosine similarity)
    successes = db.query("""
        SELECT full_params, 1 - (market_embedding <=> %s) AS similarity
        FROM optimization_runs
        WHERE proven = TRUE
        ORDER BY market_embedding <=> %s
        LIMIT 10
    """, market_embedding, market_embedding)

    # Also find failures to set Optuna "avoid" regions
    failures = db.query("""
        SELECT full_params
        FROM optimization_runs
        WHERE passed_combine = FALSE
          AND 1 - (market_embedding <=> %s) > 0.8
        ORDER BY market_embedding <=> %s
        LIMIT 20
    """, market_embedding, market_embedding)

    return successes, failures
```

Optuna's `enqueue_trial()` seeds the study with successful past params. Failed params inform the TPE sampler's prior (avoid those regions).

## Signal Log Persistence

The existing `StrategyTelemetryCollector` already captures signal-level events. Changes:

1. Add a `PgSignalPersister` that receives telemetry events and batch-inserts to `signal_log`
2. Wire it into the backtest engine alongside the existing parquet writer
3. Each trial flushes its signal log to DB after the backtest completes
4. Parquet files continue to be written for backward compatibility

## Guardrails

Existing guardrails from the current session, codified:

1. **3 consecutive combine passes** — same data, offset windows (0, +3 days, +7 days)
2. **Permutation test p < 0.05** — 20 shuffled datasets, real return must beat all or nearly all
3. **TopstepX rules enforced throughout** — $2K trailing MLL, $1K daily loss, 50% consistency
4. **No manual intervention required** — system marks proven/unproven automatically

## Data Management

- Downloads data from ProjectX API using existing `download_projectx_gold.py` pattern
- Rolls through contract months automatically (monthly for oil, bi-monthly for gold)
- Appends new bars to existing parquet files
- Tracks data freshness per instrument
- Minimum 10 trading days required before optimization attempts

## Entry Point

```bash
# Run the continuous loop
python scripts/run_adaptive_optimizer.py

# Run for a specific instrument only
python scripts/run_adaptive_optimizer.py --instrument MCL

# Run one epoch only (no looping)
python scripts/run_adaptive_optimizer.py --once

# Show status of all instruments
python scripts/run_adaptive_optimizer.py --status
```

## Files to Create/Modify

### New Files
- `src/optimization/context_embedder.py` — Market context embedding (20-dim)
- `src/optimization/experience_store.py` — pgvector query/persist layer
- `src/optimization/signal_persister.py` — Signal log DB writer
- `src/optimization/adaptive_runner.py` — Core optimization loop with warm-starts
- `src/optimization/guardrails.py` — 3x pass + permutation check
- `scripts/run_adaptive_optimizer.py` — CLI entry point
- `config/optimizer_instruments.yaml` — Instrument list (MGC, MCL, MNQ, MYM)
- `migrations/003_adaptive_optimizer.sql` — DB schema for new tables

### Modified Files
- `src/backtesting/vectorbt_engine.py` — Hook signal persister into the backtest loop
- `src/backtesting/strategy_telemetry.py` — Emit events to signal persister in addition to parquet
- `config/instruments.yaml` — Add MNQ and MYM instrument configs

### Existing Files (No Changes Needed)
- `src/learning/embeddings.py` — Reuse normalization utilities
- `src/backtesting/topstep_simulator.py` — Reuse for combine simulation
- `src/edges/trend_direction.py` — Already built, active for all instruments
- `scripts/run_permutation_optimization.py` — Reuse permutation logic

## Not In Scope

- Live trading integration
- New strategy development
- UI/dashboard for the optimizer
- Multi-machine distributed optimization
- Real-time streaming optimization (this is batch, epoch-based)
