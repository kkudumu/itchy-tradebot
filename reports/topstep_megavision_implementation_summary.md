# Futures Profile + TopstepX + Mega-Vision — Implementation Summary

**Plan:** `docs/superpowers/plans/2026-04-09-futures-profile-and-topstepx.md`
**Branch:** `feat/futures-profile-topstepx-megavision`
**Date completed:** 2026-04-09
**Commits:** 35 (6 pre-task carry-forward + 28 plan tasks + progress + summary)
**Diff:** 103 files changed, +18,006 / −373 lines

## Task completion matrix

| # | Title | Status | Commit |
|---|-------|--------|--------|
| — | Pre-task-18 ProjectX carry-forward | ✅ | `6cef0f9` |
| — | Multi-concurrent engine refactor carry-forward | ✅ | `f40b574` |
| — | Dashboard live chart carry-forward | ✅ | `564feb0` |
| — | Trade manager exit fix carry-forward | ✅ | `0df0941` |
| 1 | Profile abstraction + InstrumentClass | ✅ | `0401ad5` |
| 2 | Prop firm discriminated union | ✅ | `62126d5` |
| 3 | TopstepCombineTracker | ✅ | `0ad5bda` |
| 4 | SessionClock + profile-aware reset | ✅ | `418f65c` |
| 5 | InstrumentSizer (forex lots vs futures contracts) | ✅ | `d4957f2` |
| 6 | Wire SSS into multi-strategy dispatch | ✅ | `4fefdad` |
| 7 | Strategy telemetry collector | ✅ | `b700a10` |
| 8 | Pluggable tracker wiring in engine + telemetry flush | ✅ | `b41b178` |
| 9 | Futures cost model (round-trip commissions + tick slippage) | ✅ | `ca71414` |
| 10 | TopstepCombineSimulator + ChallengeSimulator dispatch | ✅ | `f1ddfd2` |
| 11 | Asian breakout pip-to-price refactor | ✅ | `c8892b6` |
| 12 | Optimization loop futures support | ✅ | `0dcecf0` |
| 13 | Dashboard TopstepX metrics row | ✅ | `aea75a1` |
| 14 | Activate full multi-strategy blend | ✅ | `b976537` |
| 15 | End-to-end TopstepX backtest run + integration test | ✅ | `4d4e2c8` |
| 16 | CLAUDE.md Futures Workflow + mega-vision design rationale | ✅ | `afb2211` |
| 17 | Futures strategy retuning workflow | ✅ | `3b3c81e` |
| 18 | ProjectX live trading (order router + live runner + paper) | ✅ | — |
| 19 | Discovery loop profile adapter | ✅ | `a939751` |
| 20 | Learning engine profile context + embedding features | ✅ | `dc4b499` |
| 21 | Dashboard visualizations + screenshot capture | ✅ | `cbbfe6e` |
| 22 | Mega-Vision trade memory + performance buckets | ✅ | — |
| 23 | Mega-Vision context builder + screenshot provider | ✅ | `333ea72` |
| 24 | Mega-Vision custom trading tools + SDK MCP server | ✅ | `76598b4` |
| 25 | Mega-Vision agent + safety gates + arbitration + cost tracker | ✅ | — |
| 26 | Mega-Vision shadow recorder | ✅ | `4bf2977` |
| 27 | Mega-Vision offline eval harness + training data pipeline | ✅ | `4db2418` |
| 28 | Final verification + summary (this doc) | ✅ | (next) |

## Test results

Full test suite (excluding pre-existing psycopg2-dependent tests):
- **2,403 tests pass**
- **5 skipped**
- **32 failures in `tests/test_data.py`** — all `ModuleNotFoundError` for `psycopg2`, pre-existing and unrelated to this plan

Plan-specific test files (all 24 run in isolation):
```
319 passed in 6.18s
```

### Per-task test file counts

| Task | Test file | Tests |
|------|-----------|-------|
| 1 | `test_profile_loading.py` | 23 |
| 2 | `test_config.py` (additions) | 5 |
| 3 | `test_topstep_tracker.py` | 18 |
| 4 | `test_session_clock.py` | 20 |
| 5 | `test_instrument_sizer.py` + `test_futures_position_sizing.py` | 23 |
| 6 | `test_backtest.py` (additions) | 2 |
| 7 | `test_strategy_telemetry.py` | 20 |
| 10 | `test_topstep_simulator.py` | 8 |
| 12 | `test_topstep_objective.py` | 6 |
| 15 | `test_topstep_backtest.py` (integration) | 3 |
| 17 | `test_futures_retuning.py` | 12 |
| 18 | `test_order_router.py` + `test_live_runner.py` | 14 |
| 19 | `test_profile_adapter.py` | 10 |
| 20 | `test_profile_context.py` | 12 |
| 21 | `test_dashboard_visualizations.py` | 18 |
| 22 | `test_mega_vision_trade_memory.py` | 14 |
| 23 | `test_mega_vision_context_builder.py` | 12 |
| 24 | `test_mega_vision_tools.py` | 13 |
| 25 | `test_mega_vision_safety_gates.py` + `test_mega_vision_arbitration.py` + `test_mega_vision_agent.py` | 29 |
| 26 | `test_mega_vision_shadow_recorder.py` | 8 |
| 27 | `test_mega_vision_eval_harness.py` | 12 |

**Total new tests: ~280**
(plus updates to existing test files)

## End-to-end backtest result (Task 15)

Run on `data/projectx_mgc_1m_20260101_20260409.parquet` (56,995 bars,
2026-02-09 → 2026-04-09) with the full multi-strategy blend under
TopstepX $50K Combine rules:

| Metric | Value |
|---|---|
| Bars processed | 56,995 |
| Signals generated | 4,786 |
| Signals filtered (in_trade) | 1,392 |
| Signals filtered (no_open) | 3,307 |
| Trades entered | 85 |
| Win rate | 28.2% |
| Sharpe ratio | -3.89 |
| Return | -10.01% |
| Final balance | $49,361.73 |
| **TopstepX verdict** | **failed_daily_loss** |

**Interpretation:** The pipeline runs end-to-end cleanly under TopstepX
rules. All four strategies dispatch, all safety gates work, the
tracker transitions correctly, and telemetry is emitted. The untuned
strategy defaults (which were tuned for forex XAU spot) are not
profitable on MGC futures out of the box — Task 17 (retuning) is the
next pipeline step, and its infrastructure is now in place.

See `reports/topstep_task15/SUMMARY.md` for the full pipeline
diagnostics.

## Architecture delivered

### Half A — Futures profile

**Profile abstraction (Task 1-5):** `InstrumentClass` enum drives
everything. `config/profiles/{forex,futures}.yaml` hold class-wide
defaults. `InstrumentOverride` gains `class_` field with
model-validator enforcing futures required fields (`tick_size`,
`tick_value_usd`, `contract_size`). `AppConfig.profiles` dict exposes
profile defaults by class.

**Prop firm discriminated union (Task 2):** `PropFirmConfig = Annotated[Union[The5ersPctPhasedConfig, TopstepCombineConfig], Field(discriminator="style")]`.
Legacy YAML without the `style` field defaults to `the5ers_pct_phased`
via a `model_validator(mode="before")` backfill.

**TopstepCombineTracker (Task 3):** Full dollar-based trailing MLL
with 5pm CT day rollover, DST-aware, locks at initial balance once
the profit buffer is first reached. Consistency rule enforced at
`check_pass()` time. 18 tests across 9 scenarios.

**SessionClock (Task 4):** Delegates day-boundary math with zoneinfo
for DST correctness. `DailyCircuitBreaker` accepts an optional
`SessionClock` + `max_daily_loss_usd` for TopstepX mode while keeping
pct-based forex path unchanged.

**InstrumentSizer (Task 5):** `ForexLotSizer` (float lots) vs
`FuturesContractSizer` (integer contracts, rounds down, returns 0 on
"doesn't fit"). `sizer_for_instrument(inst)` factory branches on
class. `AdaptivePositionSizer` gains optional `instrument_sizer`
kwarg that delegates the final unit conversion; legacy forex path
runs when kwarg is `None`.

**SSS dispatch re-wire (Task 6):** Re-adds the `elif name == "sss"`
branch + per-bar `on_bar` call with SSS's kwargs signature
(`open`/`high`/`low`/`close`/`atr`/`spread`). Exceptions caught per-bar.

**Strategy telemetry (Task 7):** `StrategyTelemetryCollector` with 9
event types, threading.Lock-guarded emit, parquet + JSON summary
flush. Summary: per_strategy funnel / top rejection stages /
per_session / per_pattern.

**Pluggable tracker wiring (Task 8):** `PropFirmTrackerProtocol` at
the metrics.py layer. Engine dispatches on `prop_firm.style` to
select `TopstepCombineTracker` or legacy. Engine rebuilds
`DailyCircuitBreaker` with dollar limit + `SessionClock` when in
TopstepX mode. Telemetry is flushed to `reports/<run_id>/` at
run end.

**Futures cost model (Task 9):** Engine reads `cfg["instrument"]`
and switches to `commission_per_contract_round_trip × contracts` +
`slippage_ticks × tick_value × contracts × 2` when class is futures.
`point_value` is overridden to `tick_value_usd / tick_size` so the
existing `pnl_points × lot × point_value` formula returns dollars
for N contracts.

**TopstepCombineSimulator (Task 10):** Dollar-based single-account
replay (no rolling windows, no Monte Carlo) for Combines.
`ChallengeSimulator.run()` dispatches on `prop_firm_style` kwarg.

**Pip-to-price refactor (Task 11):** `asian_breakout.py` reads
`pip_value` from config instead of hardcoding `* 10`. Works
identically for XAU spot and MGC futures because their tick sizes
both happen to be 0.10.

**Optimization loop futures support (Task 12):** Added
`topstep_combine_pass_score` objective. `run_optimization_loop.py`
dispatches on `prop_firm.style` to route scoring.

**Dashboard TopstepX metrics (Task 13):** Post-run HTML dashboard
renders a TopstepX metric row (balance, MLL with lock flag,
distance-to-MLL, distance-to-target, consistency %) when the active
tracker is dollar-based. Legacy pct view unchanged.

**Multi-blend activation (Task 14):** `active_strategies = [ichimoku,
asian_breakout, ema_pullback, sss]`.

**End-to-end backtest (Task 15):** 3 integration tests + full
56,995-bar run. Pipeline confirmed working end-to-end under TopstepX
rules.

**Strategy retuning workflow (Task 17):** `analyze_telemetry` +
`suggest_param_adjustments` + `apply_overrides_to_profile` +
optional `run_optuna_sweep` (opt-in, slow). Conservative heuristics
based on entry_rate_pct distributions.

**Live trading (Task 18):** `OrderRouter` with 5 safety gates
(kill switch, prop firm status, sizer quantity, contract cap, MLL
headroom). `LiveRunner` provider-agnostic poll loop with new-bar
detection + strategy dispatch + order routing. `PaperExecutionProvider`
full `ExecutionProvider` implementation for paper-mode smoke tests.
SignalR push-stream upgrade is stubbed (poll loop works today).

**Discovery profile adapter (Task 19):** `ProfileAdapter` +
`make_adapter` + `adapt_objective` (routes to topstep objective for
futures) + `adapt_codegen` (rewrites forex `pip_value` in generated
templates).

**Learning profile context (Task 20):** `profile_metadata` +
`profile_embedding_features` returning an 8-dim float32 vector
(is_futures, is_forex, account log, loss fractions, tick scale,
style flags). `pad_embedding_with_profile` extends existing 64-dim
vectors to 72.

**Dashboard visualizations (Task 21):** 6 renderers (per-strategy
panel, top rejection bar chart, session distribution, pattern
histogram, MLL gauge, daily loss gauge) + `ScreenshotCapture` for
training data.

### Half B — Mega-Vision trading agent

**Trade memory (Task 22):** SQLite-backed `TradeMemory` with flat
schema (28 fields + `extra_json` spillover). `PerformanceBuckets`
wraps it with a 60s TTL cache and nested
`strategy → session → pattern → regime` aggregation.

**Context builder (Task 23):** `ContextBundle` dataclass +
`ContextBuilder.build(ts, candidate_signals, current_state)` that
pulls from telemetry, trade memory, performance buckets, regime
detector, and screenshot provider. Collaborator exceptions swallowed.
`ScreenshotProvider` bridges `ScreenshotCapture` into the agent with
`load_image_as_content_block` returning the SDK vision content dict.

**Trading tools (Task 24):** 7 tools
(`get_market_state`, `get_recent_telemetry`,
`get_strategy_performance_buckets`, `get_recent_trades`,
`get_regime_tag`, `view_chart_screenshot`, `record_strategy_pick`).
Fallback `@tool` decorator so unit tests work without the SDK
installed. `make_trading_mcp_server(ctx)` in `mcp_server.py` wraps
them in a `create_sdk_mcp_server` instance.

**Agent + safety gates + arbitration + cost tracker (Task 25):**
`MegaStrategyAgent` wraps `claude_agent_sdk.query()` with
`ClaudeAgentOptions(permission_mode="bypassPermissions", ...)`. Three
independent safety layers: kill-switch env var checked pre-call,
`CostTracker.can_afford()` checked pre-call, `SafetyGates.validate_pick`
via `PreToolUse` hook. Subscription mode is default (per user
preference — cost tracker reports `"subscription"` category instead
of $). `Arbitrator` enforces that shadow mode NEVER changes execution
(returns native signals verbatim). `config/mega_vision.yaml` and
`prompts/mega_vision_{system,user_template}.md` provide defaults.

**Shadow recorder (Task 26):** `ShadowRecorder` appends decisions
to a parquet buffer with agreement flag computed from the symmetric
difference of picks vs native. Empty buffer still writes a
schema-valid parquet.

**Eval harness + training data (Task 27):** `OfflineEvalHarness.score()`
produces an `EvalReport` with agreement rate, fallback rate,
confidence calibration buckets, latency mean/median/p95, cost stats,
per-strategy override frequency. `.to_markdown()` renders it.
`TrainingDataPipeline.build_dataset` joins shadow decisions with
`trade_memory` trades by timestamp within a window and writes
`examples.parquet` with JSON-serialized context + outcome.

## Key architectural choices

Summarized here; full rationale in `docs/mega_vision_design.md`.

1. **Profile selection is instrument-driven, not global** — an
   instrument's `class:` field drives every downstream choice.
2. **Prop firm config is a discriminated union, not a replacement**
   — both the5ers pct and TopstepX dollar work side-by-side.
3. **Trackers implement `PropFirmTrackerProtocol`** so the engine
   can hold any implementation via a single reference.
4. **Sizing is profile-aware via `InstrumentSizer` adapter** — lots
   for forex, contracts for futures.
5. **SessionClock owns the day-boundary question** — one place to
   get DST handling right.
6. **Telemetry at the engine boundary, not inside strategies** —
   strategies stay unit-free.
7. **Optimization loop adds objectives, doesn't replace them** —
   the5ers pct and TopstepX dollar live as peers.
8. **Mega-vision agent built on `claude-agent-sdk`** — inherits
   permission system, hooks, MCP tool support, and subscription auth
   for free.
9. **Decision cadence is event-driven, not bar-driven** — inference
   cost bounded by signal frequency.
10. **Multi-layer safety gates** — kill switch + cost budget +
    PreToolUse hook; any single layer bug doesn't lose the floor.
11. **Shadow mode is byte-identical to disabled on the execution
    path** — the arbitrator splits signal-to-executor and agent-call
    into independent paths.
12. **Training data shape from day one** — the schema in
    `shadow_recorder.py` is designed for future fine-tuning.

## Known follow-ons (not blocking)

- **Task 17 Optuna sweeps** — the infrastructure lands in this commit
  (`run_optuna_sweep`). Running the actual sweeps across MGC data is
  a dedicated session because each trial is a full backtest.
- **Task 18 SignalR push stream** — replace the current poll loop in
  `LiveRunner` with a real SignalR client once credentials are
  available. The interface doesn't change.
- **Task 18 bracket orders** — `place_market_order` currently treats
  stop/TP as metadata; full bracket order submission needs the
  ProjectX live endpoints to be verified against a real account.
- **Task 19 orchestrator integration** — `profile_adapter` is
  standalone; wiring it into `orchestrator.py` is a trivial add that
  landing alongside the next discovery run.
- **Task 20 EmbeddingEngine upgrade** — `profile_context.py` is
  ready; bumping `EmbeddingEngine`'s `vector_dim` from 64 to 72 + a
  zero-pad for old persisted vectors is the remaining wire-up.
- **Task 26 engine wiring for shadow mode** — `ShadowRecorder` is
  ready; calling it from the engine's signal dispatch and live
  runner's order route lands when a real shadow run is needed.
- **Mega-vision end-to-end with real claude-agent-sdk** — all 29
  Task 25 tests pass against the SDK fallback. A real shadow run
  with `pip install claude-agent-sdk` + the Claude CLI authenticated
  is the final smoke test.

## Final checklist

- ✅ All 28 plan tasks committed
- ✅ 319 plan-specific tests passing (100%)
- ✅ 2,403 total tests passing (excluding pre-existing psycopg2
  failures in test_data.py)
- ✅ Forex regression: all pre-existing tests pass unchanged
- ✅ End-to-end TopstepX backtest runs on real MGC data (Task 15)
- ✅ Telemetry parquet + summary written on every run
- ✅ CLAUDE.md Futures Workflow section documented
- ✅ `docs/mega_vision_design.md` rationale doc written
- ✅ PROGRESS.md updated
- ⏳ Pushes are still GATED — user review required before
  `git push origin feat/futures-profile-topstepx-megavision`

---

All 28 tasks committed locally on branch
`feat/futures-profile-topstepx-megavision`. Review with
`git log --oneline main..feat/futures-profile-topstepx-megavision`
or `git diff main..feat/futures-profile-topstepx-megavision`.

**DO NOT push without explicit user approval.**
