# Futures Profile + TopstepX Combine + Multi-Strategy Telemetry — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Run end-to-end without stopping for user confirmation between tasks unless a task explicitly says HALT.

**Created:** 2026-04-09
**Estimated scope:** XX-Large — ~75-90 files, 28 tasks. Adds the full futures track in parallel with the existing forex track AND implements the complete mega-strategy trading agent (Claude API vision-driven LLM strategy selector with shadow + live authority modes). This is the entire program — not a phased-down MVP.
**Target executor:** ftm-executor (autonomous, no checkpoints between phases)
**Branch:** `feat/futures-profile-topstepx-megavision`

---

## 1. Goal

This plan delivers two interlocked things in one program: **(A)** a complete **futures trading profile** running in parallel with the existing forex / The5ers profile, and **(B)** a complete **mega-strategy trading agent** built on top of (A) — a Claude Agent SDK driven autonomous strategy selector with vision input (chart screenshots), custom trading tools, shadow mode, and live authority mode. Both halves ship in this plan; the mega-vision is no longer deferred.

### Half A — Futures profile

1. **TopstepX $50K Combine** prop firm rules (dollar-based trailing drawdown, daily loss limit, consistency rule, contract caps)
2. **ProjectX/TopstepX provider** for both historical data and live trading — finish the job: REST live trading client + SignalR WebSocket + position reconciliation + paper-mode + live runner
3. **Micro Gold (MGC)** futures as the primary instrument with correct contract metadata, tick value, commission model, and trading sessions
4. **Profile-aware position sizing** — contracts for futures vs lots for forex
5. **Profile-aware daily reset** — 5pm CT for TopstepX vs midnight UTC for the5ers
6. **All existing strategies** (`ichimoku`, `asian_breakout`, `ema_pullback`, `sss`) wired into the multi-strategy backtest dispatcher and runnable as a blend
7. **Rich per-strategy telemetry** — every signal generation event, every filter rejection, every entry, bucketed by time-of-day, session, and pattern type, persisted as Parquet
8. **Optimization loop futures support** — Optuna objectives + param spaces with dollar-denominated TopstepX scoring
9. **Strategy retuning** for MGC futures using telemetry analysis + Optuna sweeps
10. **Discovery + learning profile awareness** — both subsystems work in futures mode
11. **Dashboard polish** — TopstepX metrics, regime overlay, telemetry visualization, screenshot capture

### Half B — Mega-strategy trading agent (Claude Agent SDK)

12. **Trade Memory + Performance Buckets** — persistent trade outcome database with full context, queryable by strategy/session/pattern/regime
13. **Context Builder** — assembles current market state, recent telemetry, regime, recent performance into an agent-ready context
14. **Screenshot capture wired into the inference loop** — chart screenshots become vision input for the agent
15. **MegaStrategyAgent built on `claude-agent-sdk`** — Python `query()` / `ClaudeSDKClient` interface, custom in-process tools defined via `@tool` + `create_sdk_mcp_server`, model `claude-opus-4-6` for shadow / `claude-haiku-4-5` for fast live decisions, bypassPermissions for autonomous operation, hooks for pre-decision audit logging, AgentDefinition subagents for parallel work where useful
16. **Custom Trading Tools (SDK MCP)** — `get_market_state`, `get_recent_telemetry`, `get_strategy_performance_buckets`, `get_recent_trades`, `get_regime_tag`, `view_chart_screenshot`, `record_strategy_pick`. The agent uses these tools to investigate the current state before deciding which strategies to enable.
17. **Arbitration Layer + Safety Gates** — combines agent picks with native blender output via PreToolUse hooks; hard floor on prop firm rules, position size, risk caps; the agent cannot override risk gates; on agent error/timeout the system falls back to the native blender silently and records the failure
18. **Shadow Mode (backtest + live)** — agent is consulted on every signal but does NOT change execution; decisions persisted to a parallel telemetry stream for offline evaluation
19. **Live Authority Mode** — agent has authority over strategy selection in production, with multi-layer safety: pre-decision validation, post-decision safety gates, manual kill switch via env var, automatic fallback to native blender
20. **Offline Evaluation Harness** — replay historical telemetry through the agent, score against hindsight, A/B against native blender, drift detection
21. **Training Data Pipeline** — telemetry + screenshots + outcomes become labeled (state → outcome) examples persisted to a structured store for future fine-tuning

The forex / The5ers code path **must continue to work unchanged**. Switching profiles is data-driven via the active instrument's `class` field. The mega-vision agent is opt-in via `mega_vision.mode` config field (`disabled` | `shadow` | `authority`).

After all implementation is complete, the runner shall:
1. Execute a full backtest on `data/projectx_mgc_1m_20260101_20260409.parquet` with multi-strategy blend under TopstepX rules
2. Run the same backtest in mega-vision shadow mode
3. Run a 60-second live paper-mode session (vanilla)
4. Run a 60-second live paper-mode session in mega-vision shadow mode
5. Report all four results in a single SUMMARY.md

---

## 2. Architecture Decisions

### 2.1 Profile selection is instrument-driven, not global

The active profile (`forex` | `futures`) is determined by the `class` field on the instrument config, not by a global flag. This means a single config file could in theory support both profiles concurrently across different instruments. The runner reads the active instrument and configures profile-aware components accordingly.

```yaml
# config/instruments.yaml
instruments:
  - symbol: "XAUUSD"
    class: "futures"           # NEW — drives profile selection
    provider: "projectx"
    contract_id: "CON.F.US.MGC.M26"
    ...
```

### 2.2 Prop firm config gets a discriminator, not a replacement

The existing percentage-based 2-step prop firm config (the5ers) is preserved. A new TopstepX-style dollar-based config is added alongside, discriminated by a `style` field:

```yaml
prop_firm:
  style: "topstep_combine_dollar"   # or "the5ers_pct_phased"
  # ... fields specific to chosen style
```

`PropFirmConfig` becomes a Pydantic discriminated union. Both styles continue to work.

### 2.3 Trackers are pluggable

`PropFirmTracker` (legacy single-phase pct), `MultiPhasePropFirmTracker` (the5ers 2-step pct), and the new `TopstepCombineTracker` (dollar trailing) all implement a common protocol (`PropFirmTrackerProtocol`) so the engine can hold any one of them and call the same methods (`initialise`, `update`, `on_eod`, `check_pass`, `to_dict`). The engine decides which to instantiate based on `prop_firm.style`.

### 2.4 Position sizing is profile-aware via instrument metadata

`AdaptivePositionSizer` keeps its risk-percentage logic but delegates the conversion from "risk dollars" to "size to send to broker" through an `InstrumentSizer` adapter that varies by instrument class:
- `ForexLotSizer`: returns lot size (float, e.g. 0.05)
- `FuturesContractSizer`: returns integer contract count, capped by instrument max contracts

### 2.5 Strategies stay unit-agnostic by reading instrument metadata

Today, strategies (`sss`, `asian_breakout`, etc.) carry forex-pip thresholds like `min_swing_pips: 0.5`. These are interpreted as "0.5 in the instrument's smallest sane unit". A small helper `instrument.price_distance(pips=X)` converts to a real price delta using `tick_size` (futures) or `pip_size` (forex). No strategy logic changes — only the unit conversion at the boundary.

### 2.6 Daily reset and session boundaries are profile-aware

A new `SessionClock` component owns "what day is it for prop firm daily-loss purposes". For TopstepX, day rollover happens at 5pm CT (timezone-aware). For the5ers, midnight UTC. The DailyCircuitBreaker, EOD trailing logic, and trading-hours filter all consult `SessionClock` instead of using `bar.date()`.

### 2.7 Telemetry is collected at the boundary, not deep inside strategies

A single `StrategyTelemetryCollector` is constructed by the engine and threaded through. Strategies do not import it. The engine emits events when:
- A strategy returns a signal (success or none)
- A filter rejects a signal (which filter, why)
- A trade is entered or skipped
- An exit fires

Telemetry rows are buffered and flushed to Parquet on backtest completion. Downstream LLM consumption is out-of-scope here — the data just needs to exist and be queryable.

### 2.8 Optimization loop must be profile-aware but minimally invasive

`run_optimization_loop.py` and `src/optimization/objectives.py` get profile-aware objective functions. The dollar-based TopstepX objective (e.g., `topstep_combine_pass_score`) is added alongside the existing pct-based objectives. The Optuna parameter spaces themselves don't change — only the scoring of trial outcomes.

### 2.9 Mega-strategy agent is built on Claude Agent SDK, not the raw API

The mega-strategy agent uses the `claude-agent-sdk` Python package — NOT the raw Anthropic SDK / Claude API client. This is a deliberate choice:

- The Agent SDK gives us the agentic loop, permission system, hooks, and MCP tool integration for free. We do not reimplement these.
- Custom trading tools are defined as Python functions decorated with `@tool` and exposed through an in-process SDK MCP server via `create_sdk_mcp_server`. The agent calls them like any built-in tool.
- The agent's lifecycle is wrapped in `query()` for one-shot decisions or `ClaudeSDKClient` for stateful sessions where we need interrupts, hooks, or in-process MCP tools.
- Permission mode is `bypassPermissions` for production runs; the safety gating happens through PreToolUse hooks, not interactive prompts.
- Model selection is config-driven: `claude-opus-4-6` for shadow mode (deeper reasoning, higher cost ok because shadow doesn't run on every bar), `claude-haiku-4-5` for live decisions (latency-sensitive). The agent SDK accepts model overrides per query.
- The `claude-api` skill from the superpowers ecosystem is NOT used here. The Agent SDK is the right level of abstraction for an autonomous trading agent.

### 2.10 Agent decision cadence is event-driven, not bar-driven

The agent is consulted at signal events, not at every bar. Specifically: when the native SignalBlender produces at least one candidate signal at a given bar, the engine assembles the context bundle (recent bars + telemetry + screenshot + recent trades + regime + performance buckets) and asks the agent to decide which of the candidate signals (if any) to act on. On bars with no native signals, the agent is not consulted. This keeps inference cost bounded by signal frequency, not by data frequency.

### 2.11 Agent decisions are gated by hard safety rules at multiple layers

Even with `bypassPermissions`, the agent CANNOT:
- Increase position size beyond what `InstrumentSizer` produces for the configured risk percentage
- Enable trading after the `TopstepCombineTracker` has fired a daily loss or MLL violation
- Pick a strategy that's not in `active_strategies`
- Issue an action that would breach the consistency rule's projected delta (best-day cap)
- Run when the manual kill switch env var `MEGA_VISION_KILL_SWITCH=1` is set

These gates are enforced as PreToolUse hooks on the `record_strategy_pick` tool — if the picked action violates any gate, the hook returns a deny payload and the agent must retry within bounds. After 3 retries with no valid pick, the system falls back to the native blender.

### 2.12 Shadow mode is fully separate from execution path

Shadow mode runs the agent in parallel with the native execution path, NEVER mixed. The native blender's output is what actually executes. The agent's output is persisted to `reports/<run_dir>/mega_vision_shadow.parquet` for later analysis. This guarantees that turning shadow mode on cannot change execution behavior — useful for measuring agent quality without risking trades.

### 2.13 The agent's training data pipeline is built into telemetry from day one

Every shadow-mode decision is logged with the full input context AND the eventual outcome of the trade (or non-trade) the agent picked. Over many runs, this becomes a labeled training set: `(market state, telemetry, screenshot, agent's pick, native pick, eventual P&L)`. The structured store is `reports/mega_vision_training/`. Future fine-tuning is out of scope here, but the data shape is designed for it.

---

## 3. File Structure

### New Files

| File | Responsibility |
|------|----------------|
| `src/config/profile.py` | Profile abstraction: `InstrumentClass` enum, profile-aware helpers, unit conversion |
| `src/risk/topstep_tracker.py` | `TopstepCombineTracker` — dollar-based trailing MLL + daily loss + consistency |
| `src/risk/session_clock.py` | `SessionClock` — profile-aware day boundary detection (5pm CT vs midnight UTC) |
| `src/risk/instrument_sizer.py` | `InstrumentSizer` protocol + `ForexLotSizer` + `FuturesContractSizer` |
| `src/backtesting/strategy_telemetry.py` | `StrategyTelemetryCollector` + event schema |
| `src/backtesting/topstep_simulator.py` | `TopstepCombineSimulator` — single-account combine simulation (no rolling windows) |
| `tests/test_topstep_tracker.py` | Unit tests for trailing MLL, daily loss, consistency, lock semantics |
| `tests/test_session_clock.py` | Day boundary tests across DST transitions, timezone math |
| `tests/test_instrument_sizer.py` | Lot vs contract sizing parity tests |
| `tests/test_strategy_telemetry.py` | Telemetry emission + Parquet schema tests |
| `tests/test_topstep_simulator.py` | End-to-end TopstepX combine pass/fail tests |
| `tests/test_profile_loading.py` | Config loads forex + futures profiles correctly |
| `tests/test_futures_position_sizing.py` | Risk-to-contract conversion correctness |
| `tests/integration/test_topstep_backtest.py` | End-to-end backtest under TopstepX rules with multi-strategy blend |
| `docs/superpowers/plans/2026-04-09-mega-strategy-agent.md` | Companion vision plan (Phase 16 deliverable) |
| `config/profiles/forex.yaml` | Profile-level defaults for forex (commission, slippage, sessions) |
| `config/profiles/futures.yaml` | Profile-level defaults for futures (commission, slippage, sessions, daily reset) |
| `src/providers/projectx_live.py` | Live trading client: order placement, modification, cancellation via TopstepX REST |
| `src/providers/projectx_websocket.py` | SignalR WebSocket client for market_hub + user_hub (real-time bars + account/position events) |
| `src/providers/projectx_reconciler.py` | Reconciles local position state vs broker reported state; raises drift alerts |
| `src/live/live_runner.py` | Live trading orchestrator: wires provider + strategies + trade_manager + telemetry; paper-mode toggle |
| `src/live/order_router.py` | Routes signals from blender → TopstepX orders, applies contract caps + risk gates |
| `src/optimization/futures_retuning.py` | Telemetry-driven + Optuna-driven strategy parameter retuning workflow for MGC futures |
| `src/discovery/profile_adapter.py` | Profile-aware adapter for the discovery loop: routes futures runs to TopstepX objectives |
| `src/learning/profile_context.py` | Adds profile metadata (instrument class, prop firm style, session) to learning embeddings |
| `src/backtesting/screenshot_capture.py` | Optional chart screenshot rendering at signal events; outputs PNG + metadata for mega-vision pipeline |
| `src/backtesting/dashboard_visualizations.py` | Rich dashboard widgets: regime overlay, telemetry charts, gauges, trade timeline, pattern histogram |
| `tests/test_projectx_live.py` | Order placement/cancel tests against ProjectX sandbox or recorded fixtures |
| `tests/test_projectx_websocket.py` | WebSocket subscription, reconnection, message parsing tests |
| `tests/test_projectx_reconciler.py` | Position drift detection tests |
| `tests/test_live_runner.py` | Live runner integration tests in paper mode |
| `tests/test_order_router.py` | Signal-to-order conversion + cap enforcement tests |
| `tests/test_futures_retuning.py` | Retuning workflow tests with synthetic telemetry inputs |
| `tests/test_profile_adapter.py` | Discovery profile routing tests |
| `tests/test_profile_context.py` | Learning embedding profile context tests |
| `tests/test_screenshot_capture.py` | Screenshot rendering tests |
| `tests/test_dashboard_visualizations.py` | Dashboard widget rendering tests |
| `tests/integration/test_projectx_live_smoke.py` | 60-second paper-mode smoke test against TopstepX (skipped in CI without creds) |
| `src/mega_vision/__init__.py` | Mega-vision package init |
| `src/mega_vision/agent.py` | `MegaStrategyAgent` — main agent class wrapping `claude-agent-sdk` query/client; handles model selection, options, prompt assembly |
| `src/mega_vision/tools.py` | Custom trading tools defined via `@tool` decorator: `get_market_state`, `get_recent_telemetry`, `get_strategy_performance_buckets`, `get_recent_trades`, `get_regime_tag`, `view_chart_screenshot`, `record_strategy_pick` |
| `src/mega_vision/mcp_server.py` | `create_sdk_mcp_server("mega-vision-trading-tools", tools=[...])` — wraps the custom tools for the Agent SDK |
| `src/mega_vision/context_builder.py` | Assembles the agent's input context bundle from current state |
| `src/mega_vision/screenshot_provider.py` | Bridges Task 21 screenshot capture into agent inference (file path → image content block) |
| `src/mega_vision/safety_gates.py` | PreToolUse hook implementations enforcing prop firm rules, position size caps, kill switch, retry limits |
| `src/mega_vision/arbitration.py` | `Arbitrator` — combines agent picks with native blender output, applies safety gates, decides final action |
| `src/mega_vision/shadow_recorder.py` | Persists shadow-mode decisions to parquet + maintains the training data store |
| `src/mega_vision/eval_harness.py` | Offline replay engine: feed historical telemetry through the agent, score, A/B vs native blender |
| `src/mega_vision/trade_memory.py` | Persistent trade outcome database with per-strategy/session/pattern/regime aggregation |
| `src/mega_vision/performance_buckets.py` | Aggregates trade memory into per-strategy performance buckets exposed via the `get_strategy_performance_buckets` tool |
| `src/mega_vision/cost_tracker.py` | Tracks cumulative agent cost (token usage, decision count, $ spent) — exposed via dashboard |
| `src/mega_vision/training_data.py` | Pipeline that converts shadow-mode logs into labeled training examples |
| `tests/test_mega_vision_agent.py` | Agent constructor, model selection, query mocking |
| `tests/test_mega_vision_tools.py` | Each custom tool produces correct output given fixture state |
| `tests/test_mega_vision_context_builder.py` | Context bundle assembly correctness |
| `tests/test_mega_vision_safety_gates.py` | Each gate fires correctly; retry limits work; fallback path works |
| `tests/test_mega_vision_arbitration.py` | Arbitration combines picks correctly; safety gates win over agent picks |
| `tests/test_mega_vision_shadow_recorder.py` | Shadow logs persist correctly; never affect execution |
| `tests/test_mega_vision_eval_harness.py` | Offline eval produces correct A/B comparison |
| `tests/test_mega_vision_trade_memory.py` | Trade memory CRUD + aggregation correctness |
| `tests/integration/test_mega_vision_shadow_backtest.py` | End-to-end shadow mode against the same fixture as Task 15 |
| `tests/integration/test_mega_vision_authority_smoke.py` | 60-second paper-mode authority run with mocked Agent SDK responses |
| `config/mega_vision.yaml` | Mode (`disabled` \| `shadow` \| `authority`), shadow model, live model, decision cadence, kill switch path, prompt template versions, cost budget |
| `prompts/mega_vision_system.md` | The agent's static system prompt (cached via prompt caching at runtime for cost savings) |
| `prompts/mega_vision_user_template.md` | The user-message template that wraps the context bundle for each decision |

### Modified Files

| File | Change |
|------|--------|
| `src/config/models.py` | Add `PropFirmConfig` discriminated union (`the5ers_pct_phased` \| `topstep_combine_dollar`); extend `InstrumentConfig` with `class` field + futures-specific metadata; add `ProfileConfig` |
| `src/config/loader.py` | Load profile defaults from `config/profiles/{class}.yaml` and merge with strategy.yaml |
| `config/strategy.yaml` | Switch `prop_firm.style` to `topstep_combine_dollar`, add TopstepX fields, set `active_strategies: [ichimoku, asian_breakout, ema_pullback, sss]`, ensure `enabled: true` on every strategy |
| `config/instruments.yaml` | Add `class: futures`, `tick_size`, `tick_value_usd`, `contract_size`, `commission_per_contract_round_trip`, `session_open_ct`, `session_close_ct`, `daily_reset_hour_ct` to MGC entry |
| `src/backtesting/vectorbt_engine.py` | Wire `sss` into multi-strategy dispatch; instantiate correct prop firm tracker based on style; thread `SessionClock` through main loop; emit telemetry events; use `InstrumentSizer` for sizing |
| `src/backtesting/challenge_simulator.py` | Add discriminator on prop firm style; for `topstep_combine_dollar`, delegate to `TopstepCombineSimulator` (no rolling windows); preserve all existing pct/phased logic |
| `src/risk/trade_manager.py` | Accept profile-aware `InstrumentSizer`; use `SessionClock` for daily breaker; emit telemetry on rejections |
| `src/risk/circuit_breaker.py` | Accept `SessionClock` instead of plain date for day rollover; support dollar-based daily loss in addition to percentage |
| `src/risk/position_sizing.py` | Refactor `AdaptivePositionSizer` to delegate per-instrument sizing to `InstrumentSizer` |
| `src/strategy/strategies/sss/strategy.py` | Audit for pip-based thresholds; route through `instrument.price_distance(pips=...)` helper for unit conversion |
| `src/strategy/strategies/asian_breakout.py` | Same audit + unit conversion for `min_range_pips`, `max_range_pips` |
| `src/strategy/strategies/ema_pullback.py` | Same audit + unit conversion |
| `src/strategy/signal_blender.py` | Verify N-strategy blending works (currently coded for ≤3); add SSS to dispatch table |
| `src/backtesting/metrics.py` | `PropFirmTracker` and `MultiPhasePropFirmTracker` implement common protocol; export `PropFirmTrackerProtocol` |
| `src/backtesting/dashboard.py` | Display TopstepX-style metrics (USD P&L, MLL distance, daily loss remaining); fall back to pct view for forex/the5ers |
| `src/backtesting/live_dashboard.py` | Same dashboard updates for the live HTTP dashboard at :8501 |
| `scripts/run_demo_challenge.py` | Read `prop_firm.style`; pass through correct verdict to console/exit code; profile-aware default data file selection |
| `scripts/run_optimization_loop.py` | Profile-aware objective routing (dollar vs pct); read `prop_firm.style` for trial scoring |
| `src/optimization/objectives.py` | Add `topstep_combine_pass_score` objective alongside existing `pass_rate` |
| `src/optimization/walk_forward.py` | Use profile-aware tracker for walk-forward windows |
| `config/optimization_loop.yaml` | Add `objective: topstep_combine_pass_score` option; document |
| `src/providers/projectx.py` | Verify provider returns instrument metadata that matches futures schema; extract live-trading bits into projectx_live.py |
| `src/discovery/orchestrator.py` | Profile-aware: route futures runs through `profile_adapter` to TopstepX objectives |
| `src/discovery/regime_stats.py` | Add futures session segmentation (Asia futures, EU futures, US futures, RTH/ETH) alongside forex sessions |
| `src/discovery/codegen.py` (if exists) | Generate strategy stubs that respect the active profile's unit conventions |
| `src/learning/embeddings.py` | Include profile context in trade embeddings |
| `src/learning/adaptive_engine.py` (or equivalent) | Trade context dict carries profile metadata; pre-trade analysis is profile-aware |
| `src/strategy/strategies/sss/strategy.py` | Accept retuned defaults loaded from `config/profiles/futures.yaml` overrides |
| `src/strategy/strategies/asian_breakout.py` | Same: accept profile-specific param overrides |
| `src/strategy/strategies/ema_pullback.py` | Same |
| `scripts/run_demo_challenge.py` | Add `--mode live --paper` and `--mode live --real` flags wired through `live_runner` |
| `src/backtesting/results_exporter.py` | Export retuning artifacts (per-strategy parameter sweeps + winners) alongside the existing exports |
| `src/backtesting/vectorbt_engine.py` | (additional) Wire the mega-vision arbitrator into the signal-to-entry path; emit shadow recordings when configured |
| `src/live/live_runner.py` | (additional) Same arbitrator integration on the live path |
| `pyproject.toml` | Add `claude-agent-sdk` dependency |
| `scripts/run_demo_challenge.py` | (additional) Add `--mega-vision-mode` flag (`disabled` \| `shadow` \| `authority`) wired through both backtest and live paths |
| `tests/test_config.py` | Extend with profile loading tests |
| `tests/test_config_refactor.py` | Update for new prop firm discriminator |
| `tests/test_multi_phase_tracker.py` | Add coverage for protocol conformance |
| `CLAUDE.md` | Add a "Futures Workflow" section documenting the topstep_combine path and how to switch profiles |

### Files NOT to touch (out of scope)

- The existing forex `data/xauusd_1m_*.parquet` files and any forex-specific scripts
- The MT5 / histdata download code — futures uses ProjectX exclusively (but it stays available for forex regression)
- Database schema migrations beyond additive columns for trade telemetry (no breaking schema changes)

---

## 4. Acceptance Criteria

The plan is complete when ALL of the following are true:

1. **Forex regression**: All pre-existing tests in `tests/` pass without modification beyond the explicit list above. `pytest tests/test_multi_phase_tracker.py tests/test_config.py` is green.
2. **Futures unit tests**: All new test files pass — TopstepX tracker, session clock, instrument sizer, strategy telemetry, profile loading, futures position sizing, topstep simulator.
3. **TopstepX backtest runs end-to-end**: `python scripts/run_demo_challenge.py --mode validate --data-file data/projectx_mgc_1m_20260101_20260409.parquet` completes without crashing, with all four strategies (ichimoku, asian_breakout, ema_pullback, sss) dispatched, and produces:
   - A non-empty trade log (>0 trades) **OR** a clear telemetry report explaining why each strategy produced 0 signals (per-stage rejection counts)
   - A TopstepX combine verdict (passed / failed / in_progress) with the failure reason if applicable
   - A `strategy_telemetry.parquet` file in the run output directory
   - A `strategy_telemetry_summary.json` with per-strategy + per-session + per-pattern aggregates
4. **Live dashboard reflects TopstepX metrics**: The dashboard at http://localhost:8501 shows USD P&L, current MLL value, daily loss remaining, and per-strategy trade counts.
5. **Profile switching works**: Setting `class: forex` on an instrument and using a `the5ers_pct_phased` prop firm config produces the legacy code path with no crashes.
6. **Optimization loop futures support**: `python scripts/run_optimization_loop.py` with `objective: topstep_combine_pass_score` runs at least 1 iteration successfully.
7. **Mega-strategy plan doc exists**: `docs/superpowers/plans/2026-04-09-mega-strategy-agent.md` is written and covers vision, inputs, architecture sketch, training data, open questions, and phasing.
8. **CLAUDE.md updated**: Has a Futures Workflow section documenting how to run a TopstepX backtest and how the profile switch works.
9. **Strategies retuned for MGC**: Each of the four strategies has had its pip-scale and threshold parameters retuned for MGC futures using either Optuna sweeps or telemetry-driven analysis. Old forex defaults are preserved as a separate config block. Retuned values are documented with the rationale.
10. **ProjectX live trading is functional**: `python scripts/run_demo_challenge.py --mode live --paper` connects to TopstepX, authenticates, subscribes to MGC market data via WebSocket, receives bars, runs the multi-strategy blend live, places paper orders through the ProjectX REST API, tracks position state, handles reconnection, and logs all activity. A 60-second smoke test in paper mode is part of the acceptance.
11. **Discovery + learning are profile-aware**: The discovery loop runs against futures data and produces TopstepX-shaped objectives (dollar-denominated). The learning engine includes profile context in its embeddings and trade context. Existing forex discovery/learning paths still work.
12. **Dashboard is rich**: The live dashboard at :8501 and the post-run HTML report both display per-strategy real-time stats, a regime overlay on the equity curve, the telemetry summary visualization, MLL distance and daily-loss gauges, a trade timeline with strategy attribution, and a per-pattern histogram. A "screenshot capture" toggle saves chart snapshots at every signal generation event for later mega-vision pipeline ingestion.
13. **Mega-vision shadow mode runs end-to-end**: A backtest run with `mega_vision.mode: shadow` produces `reports/<run_dir>/mega_vision_shadow.parquet` containing one row per signal event with: agent's pick, native blender's pick, agreement flag, agent's reasoning summary, latency, token usage, cost. The native execution path is byte-identical to a `mode: disabled` run on the same data.
14. **Mega-vision authority mode runs end-to-end**: A backtest run with `mega_vision.mode: authority` produces a result where the executed trades reflect the agent's picks (where they passed safety gates) instead of the native blender's picks where they differ. The comparison report shows where the two diverged.
15. **Custom trading tools work**: Each of the seven custom tools (`get_market_state`, `get_recent_telemetry`, `get_strategy_performance_buckets`, `get_recent_trades`, `get_regime_tag`, `view_chart_screenshot`, `record_strategy_pick`) is callable from the agent and returns the right shape. Verified by `tests/test_mega_vision_tools.py`.
16. **Safety gates work**: An adversarial test forces the agent to attempt a violating pick (e.g., position size 10x risk cap, or a strategy not in active_strategies). The PreToolUse hook denies it, the agent retries within bounds, and after 3 retries the system falls back to the native blender. Verified by `tests/test_mega_vision_safety_gates.py`.
17. **Kill switch works**: Setting `MEGA_VISION_KILL_SWITCH=1` in env makes the system fall back to the native blender immediately, without any agent calls. Verified by `tests/test_mega_vision_safety_gates.py`.
18. **Trade memory + performance buckets**: After the Task 15 backtest completes, `src/mega_vision/trade_memory.py` has persisted every trade with full context. `get_strategy_performance_buckets()` returns aggregations broken down by strategy / session / pattern / regime.
19. **Live paper smoke + mega-vision shadow**: A 60-second `--mode live --paper --mega-vision-mode shadow` session connects, receives bars, runs the agent on at least one signal, persists the shadow record, and exits cleanly.
20. **Cost budget honored**: The agent cost tracker (`src/mega_vision/cost_tracker.py`) refuses to make further inference calls when the configured `cost_budget_usd` is exceeded — falls back to the native blender for the remainder of the run, with a clear warning logged.

---

## 5. Tasks

### Task 1: Profile Abstraction + Instrument Class

**Goal:** Introduce the concept of an instrument class (forex | futures) and load profile defaults from disk.

**Files:**
- Create: `src/config/profile.py`
- Create: `config/profiles/forex.yaml`
- Create: `config/profiles/futures.yaml`
- Modify: `src/config/models.py`
- Modify: `src/config/loader.py`
- Modify: `config/instruments.yaml`
- Create: `tests/test_profile_loading.py`

**Steps:**

- [ ] **Step 1.1**: Create `src/config/profile.py` with:
  - `class InstrumentClass(str, Enum)` with values `FOREX = "forex"`, `FUTURES = "futures"`
  - `class ProfileConfig(BaseModel)` with fields: `instrument_class`, `default_commission`, `default_slippage_ticks`, `daily_reset_hour`, `daily_reset_tz`, `session_open_local`, `session_close_local`, `maintenance_window_minutes`
  - Helper function `def price_distance(instrument, pips: float) -> float:` that converts a "pip" count to a real price delta using `tick_size * pips * 10` for futures or `pip_size * pips` for forex
  - Helper function `def load_profile(instrument_class: InstrumentClass) -> ProfileConfig` that reads the appropriate `config/profiles/{class}.yaml`

- [ ] **Step 1.2**: Create `config/profiles/forex.yaml`:
  ```yaml
  instrument_class: forex
  default_commission_per_lot: 0.0
  default_slippage_pips: 0.5
  daily_reset_hour: 0
  daily_reset_tz: "UTC"
  session_open_local: "00:00"
  session_close_local: "23:59"
  maintenance_window_minutes: 0
  ```

- [ ] **Step 1.3**: Create `config/profiles/futures.yaml`:
  ```yaml
  instrument_class: futures
  default_commission_per_contract_round_trip: 1.40
  default_slippage_ticks: 1
  daily_reset_hour: 17
  daily_reset_tz: "America/Chicago"
  session_open_local: "17:00"
  session_close_local: "16:00"
  maintenance_window_minutes: 60
  ```

- [ ] **Step 1.4**: Extend `InstrumentConfig` in `src/config/models.py`:
  - Add `class_: InstrumentClass = Field(default=InstrumentClass.FOREX, alias="class")`
  - Add futures-specific optional fields: `tick_size`, `tick_value_usd`, `contract_size`, `commission_per_contract_round_trip`, `session_open_ct`, `session_close_ct`, `daily_reset_hour_ct`
  - Add forex-specific optional fields: `pip_size`, `pip_value_per_lot`, `standard_lot_units`
  - Add a `model_validator(mode="after")` that asserts required fields per `class_` (futures requires tick_size + tick_value_usd + contract_size; forex requires pip_size + pip_value_per_lot)

- [ ] **Step 1.5**: Update `src/config/loader.py` to load the appropriate profile YAML based on each instrument's `class_` field and merge it into the instrument's effective config (profile defaults are overridden by per-instrument values).

- [ ] **Step 1.6**: Update `config/instruments.yaml` MGC entry:
  ```yaml
  instruments:
    - symbol: "XAUUSD"
      class: "futures"
      provider: "projectx"
      default_quantity: 1
      contract_id: "CON.F.US.MGC.M26"
      symbol_id: "F.US.MGC"
      tick_size: 0.10
      tick_value_usd: 1.00
      contract_size: 10  # MGC = 10 troy oz
      commission_per_contract_round_trip: 1.40
      session_open_ct: "17:00"
      session_close_ct: "16:00"
      daily_reset_hour_ct: 17
      adx_threshold: 28
      spread_max_points: 30
      atr_stop_multiplier: 1.5
  ```

- [ ] **Step 1.7**: Create `tests/test_profile_loading.py` with:
  - Test that loading a forex instrument produces a forex profile
  - Test that loading a futures instrument produces a futures profile
  - Test that missing required futures fields raise validation errors
  - Test that profile defaults are overridden by per-instrument values
  - Test `price_distance` correctness for both classes (e.g., 10 pips on MGC = 1.0 price units; 10 pips on XAU spot = 0.10)

- [ ] **Step 1.8**: Run `pytest tests/test_profile_loading.py tests/test_config.py tests/test_config_refactor.py`. Fix any breakage.

---

### Task 2: Prop Firm Discriminated Union

**Goal:** Replace the dict-based `prop_firm` block with a Pydantic discriminated union that supports both the5ers (pct-based) and TopstepX (dollar-based) styles.

**Files:**
- Modify: `src/config/models.py`
- Modify: `config/strategy.yaml`
- Modify: `tests/test_config.py`
- Modify: `tests/test_config_refactor.py`

**Steps:**

- [ ] **Step 2.1**: In `src/config/models.py`, add three new Pydantic models:
  ```python
  class The5ersPhaseConfig(BaseModel):
      profit_target_pct: float
      max_loss_pct: float
      daily_loss_pct: float
      time_limit_days: int = 0

  class The5ersPctPhasedConfig(BaseModel):
      style: Literal["the5ers_pct_phased"] = "the5ers_pct_phased"
      name: str = "the5ers_2step_high_stakes"
      account_size: float = 10_000.0
      leverage: int = 100
      phase_1: The5ersPhaseConfig
      phase_2: The5ersPhaseConfig
      funded: dict = Field(default_factory=dict)

  class TopstepCombineConfig(BaseModel):
      style: Literal["topstep_combine_dollar"] = "topstep_combine_dollar"
      name: str = "topstep_50k_combine"
      account_size: float = 50_000.0
      profit_target_usd: float = 3_000.0
      max_loss_limit_usd_trailing: float = 2_000.0
      daily_loss_limit_usd: float = 1_000.0
      consistency_pct: float = 50.0
      max_micro_contracts: int = 50
      max_full_contracts: int = 5
      daily_reset_tz: str = "America/Chicago"
      daily_reset_hour: int = 17
  ```

- [ ] **Step 2.2**: Add the discriminated union:
  ```python
  PropFirmConfig = Annotated[
      Union[The5ersPctPhasedConfig, TopstepCombineConfig],
      Field(discriminator="style"),
  ]
  ```

- [ ] **Step 2.3**: Wire `PropFirmConfig` into the top-level config model so YAML loaders use it. The loader must accept missing `style` field and default to `the5ers_pct_phased` for backward compatibility with existing test fixtures.

- [ ] **Step 2.4**: Update `config/strategy.yaml` `prop_firm` block:
  ```yaml
  prop_firm:
    style: topstep_combine_dollar
    name: topstep_50k_combine
    account_size: 50000
    profit_target_usd: 3000
    max_loss_limit_usd_trailing: 2000
    daily_loss_limit_usd: 1000
    consistency_pct: 50.0
    max_micro_contracts: 50
    max_full_contracts: 5
    daily_reset_tz: "America/Chicago"
    daily_reset_hour: 17
    # Legacy the5ers config kept for reference; comment out, do not delete:
    # name_legacy: the5ers_2step_high_stakes
    # account_size_legacy: 10000
    # phase_1: { profit_target_pct: 8.0, max_loss_pct: 10.0, daily_loss_pct: 5.0 }
    # phase_2: { profit_target_pct: 5.0, max_loss_pct: 10.0, daily_loss_pct: 5.0 }
  ```

- [ ] **Step 2.5**: Update `tests/test_config.py` and `tests/test_config_refactor.py`:
  - Add fixtures for both prop firm styles
  - Test that loading either style works
  - Test that legacy the5ers YAML (no `style` field) still loads via default
  - Test that an unknown `style` value raises a validation error

- [ ] **Step 2.6**: Run `pytest tests/test_config.py tests/test_config_refactor.py`. Fix any breakage.

---

### Task 3: TopstepX Combine Tracker

**Goal:** Implement the dollar-based trailing maximum loss limit + daily loss + consistency rule.

**Files:**
- Create: `src/risk/topstep_tracker.py`
- Create: `tests/test_topstep_tracker.py`

**Steps:**

- [ ] **Step 3.1**: Create `src/risk/topstep_tracker.py` with class `TopstepCombineTracker`. Constructor accepts a `TopstepCombineConfig`. State:
  - `initial_balance: float` (set on `initialise`)
  - `current_balance: float`
  - `mll: float` — current trailing maximum loss limit (starts at `initial_balance - max_loss_limit_usd_trailing`)
  - `mll_locked: bool` — True once balance has ever reached `initial_balance + max_loss_limit_usd_trailing`
  - `eod_balances: list[tuple[date, float]]` — for trailing math
  - `daily_open_balance: float` — for daily loss calc
  - `daily_pnl: float`
  - `current_trading_day: date | None`
  - `total_profit: float`
  - `best_day_profit: float`
  - `status: Literal["pending", "passed", "failed_mll", "failed_daily_loss", "failed_consistency"]`
  - `failure_reason: str | None`

- [ ] **Step 3.2**: Implement `def initialise(self, initial_balance: float, first_ts: datetime) -> None`:
  - Set `initial_balance`, `current_balance = initial_balance`
  - Set `mll = initial_balance - config.max_loss_limit_usd_trailing`
  - `mll_locked = False`
  - Set `current_trading_day = self._trading_day_for(first_ts)`
  - Set `daily_open_balance = initial_balance`
  - Reset all counters

- [ ] **Step 3.3**: Implement `def _trading_day_for(self, ts: datetime) -> date`. The "trading day" for TopstepX rolls over at `daily_reset_hour` local time in `daily_reset_tz`. Use `pytz` (already a transitive dep through pandas). Convert `ts` (assumed UTC) to America/Chicago, subtract 17 hours, then take the date. Handle DST correctly.

- [ ] **Step 3.4**: Implement `def update(self, ts: datetime, balance: float) -> None`:
  - Determine `trading_day = self._trading_day_for(ts)`
  - If `trading_day != current_trading_day`: call `self._on_eod(self.current_balance)` (closes previous day), then start new day with `current_trading_day = trading_day`, `daily_open_balance = balance`
  - Update `current_balance = balance`
  - Compute `daily_pnl = balance - daily_open_balance`
  - Check daily loss: if `daily_pnl <= -config.daily_loss_limit_usd`: set status to `failed_daily_loss`, reason
  - Check MLL: if `balance <= mll`: set status to `failed_mll`, reason
  - Check profit target: if `balance >= initial_balance + config.profit_target_usd`: set status to `passed` (subject to consistency check)

- [ ] **Step 3.5**: Implement `def _on_eod(self, eod_balance: float) -> None`:
  - Append to `eod_balances`
  - Update `total_profit = max(0.0, eod_balance - initial_balance)`
  - Update `best_day_profit = max(best_day_profit, eod_balance - daily_open_balance)`
  - Trail MLL: `new_mll_candidate = eod_balance - config.max_loss_limit_usd_trailing`; if `new_mll_candidate > mll`: `mll = new_mll_candidate`
  - Lock check: if `not mll_locked` and `eod_balance >= initial_balance + config.max_loss_limit_usd_trailing`: `mll = initial_balance`; `mll_locked = True`

- [ ] **Step 3.6**: Implement `def check_pass(self) -> dict`:
  - If `status == "passed"`:
    - Verify consistency: if `total_profit > 0` and `best_day_profit / total_profit > config.consistency_pct / 100.0`: change status to `failed_consistency`, reason
  - Return dict: `{ status, mll, mll_locked, current_balance, total_profit, best_day_profit, daily_pnl, failure_reason, distance_to_mll, distance_to_target }`

- [ ] **Step 3.7**: Implement `def to_dict(self) -> dict` returning the same payload as `check_pass` plus a `style` key set to `"topstep_combine_dollar"`.

- [ ] **Step 3.8**: Create `tests/test_topstep_tracker.py` covering:
  - **Test A**: Initialise with $50K. MLL starts at $48K. Lock is False.
  - **Test B**: Balance grows to $51K end-of-day. MLL trails to $49K. Lock still False.
  - **Test C**: Balance hits $52K end-of-day. MLL locks at $50K. Subsequent EOD growth doesn't move MLL.
  - **Test D**: Daily loss of $1001 in one day → status `failed_daily_loss`.
  - **Test E**: Balance falls to $47999 at any moment (intraday) → status `failed_mll`.
  - **Test F**: Balance reaches $53000 (above profit target $3000), best_day_profit = $1499, total_profit = $3000 → status `passed` after consistency check (1499/3000 = 49.97%, below 50% rule).
  - **Test G**: Same as F but best_day_profit = $1501 → status `failed_consistency`.
  - **Test H**: Day rollover happens at 5pm CT, not midnight UTC. Cross a DST boundary in test data and verify rollover still works.
  - **Test I**: After lock, MLL never decreases even if EOD balance dips back to $50500.

- [ ] **Step 3.9**: Run `pytest tests/test_topstep_tracker.py`. All tests must pass.

---

### Task 4: Session Clock + Profile-Aware Daily Reset

**Goal:** Replace ad-hoc `bar.date()` day rollover with a profile-aware `SessionClock`.

**Files:**
- Create: `src/risk/session_clock.py`
- Create: `tests/test_session_clock.py`
- Modify: `src/risk/circuit_breaker.py`

**Steps:**

- [ ] **Step 4.1**: Create `src/risk/session_clock.py` with class `SessionClock`. Constructor: `(reset_hour_local: int, reset_tz: str)`. Methods:
  - `def trading_day(self, ts: datetime) -> date` — returns the trading day for any UTC timestamp
  - `def is_new_day(self, prev_ts: datetime | None, current_ts: datetime) -> bool` — boundary detection
  - `def day_open_ts(self, trading_day: date) -> datetime` — returns the UTC timestamp at which a given trading day opens

- [ ] **Step 4.2**: Implement using `pytz.timezone(reset_tz)`. For TopstepX (`America/Chicago`, hour=17): trading day for ts = (ts.astimezone(CT) - 17h).date(). For forex (`UTC`, hour=0): trading day = ts.date(). Make sure DST is handled (use `localize` and `normalize`).

- [ ] **Step 4.3**: Create `tests/test_session_clock.py`:
  - Test forex profile: midnight UTC rollover
  - Test futures profile: 5pm CT rollover
  - Test DST spring-forward: a 5pm CT timestamp on the day clocks change still produces the right trading day
  - Test DST fall-back: same
  - Test that two timestamps on either side of the rollover are reported as different trading days
  - Test that two timestamps within the same trading day are reported as the same

- [ ] **Step 4.4**: Update `src/risk/circuit_breaker.py`:
  - `DailyCircuitBreaker.__init__` accepts an optional `session_clock: SessionClock | None = None`. If `None`, defaults to forex (UTC midnight) for backward compat.
  - `start_day(balance, ts)` and `update(...)` use `session_clock.trading_day(ts)` instead of `ts.date()`
  - Add support for dollar-based daily loss (`max_daily_loss_usd: float | None = None`) alongside the existing pct-based limit

- [ ] **Step 4.5**: Run `pytest tests/test_session_clock.py`. Then run all existing circuit breaker tests to verify backward compat (forex path unchanged).

---

### Task 5: Instrument Sizer (Forex Lots vs Futures Contracts)

**Goal:** Move "convert risk dollars to broker units" out of `AdaptivePositionSizer` and into a profile-aware adapter.

**Files:**
- Create: `src/risk/instrument_sizer.py`
- Modify: `src/risk/position_sizing.py` (or wherever `AdaptivePositionSizer` lives)
- Create: `tests/test_instrument_sizer.py`
- Create: `tests/test_futures_position_sizing.py`

**Steps:**

- [ ] **Step 5.1**: First read the current position sizer (likely `src/risk/position_sizing.py`) and document its current contract.

- [ ] **Step 5.2**: Create `src/risk/instrument_sizer.py` with:
  ```python
  class InstrumentSizer(Protocol):
      def size_for_risk(self, risk_usd: float, stop_distance_price: float) -> float | int: ...
      def min_size(self) -> float | int: ...
      def max_size(self) -> float | int: ...

  class ForexLotSizer(InstrumentSizer):
      def __init__(self, pip_size: float, pip_value_per_lot: float, max_lot_size: float, min_lot_size: float = 0.01): ...
      def size_for_risk(self, risk_usd, stop_distance_price):
          stop_pips = stop_distance_price / self.pip_size
          dollars_per_lot_at_stop = stop_pips * self.pip_value_per_lot
          return min(self.max_lot, max(self.min_lot, round(risk_usd / dollars_per_lot_at_stop, 2)))

  class FuturesContractSizer(InstrumentSizer):
      def __init__(self, tick_size: float, tick_value_usd: float, max_contracts: int, min_contracts: int = 1): ...
      def size_for_risk(self, risk_usd, stop_distance_price):
          stop_ticks = stop_distance_price / self.tick_size
          dollars_per_contract_at_stop = stop_ticks * self.tick_value_usd
          return min(self.max_contracts, max(self.min_contracts, int(risk_usd // dollars_per_contract_at_stop)))
  ```

- [ ] **Step 5.3**: Add a factory `def sizer_for_instrument(instrument_config) -> InstrumentSizer` that branches on `instrument_config.class_`.

- [ ] **Step 5.4**: Refactor `AdaptivePositionSizer` to accept and delegate to `InstrumentSizer`. Preserve the existing risk-percentage logic (initial_risk_pct, reduced_risk_pct, phase_threshold_pct). The change is: instead of returning a float lot size directly, call `self._sizer.size_for_risk(risk_usd, stop_distance)`.

- [ ] **Step 5.5**: Update all call sites that construct `AdaptivePositionSizer` to also pass an `InstrumentSizer`. There's at least one in `src/backtesting/vectorbt_engine.py:160-165` — find them all with grep.

- [ ] **Step 5.6**: Create `tests/test_instrument_sizer.py`:
  - Forex: $50 risk, 50 pip stop on XAU spot (pip=$0.01, $1/pip/lot) → 1.0 lot
  - Futures: $50 risk, $5 stop on MGC (tick=$0.10, $1/tick) → 1 contract
  - Futures: $500 risk, $5 stop on MGC → 10 contracts
  - Futures: $50 risk, $50 stop on MGC → 0 contracts (skip; min_contracts=1 returns 1 only if risk allows it)
  - Cap test: $500 risk, $1 stop on MGC, max=50 → 50 contracts (capped)

- [ ] **Step 5.7**: Create `tests/test_futures_position_sizing.py`:
  - End-to-end test that wires AdaptivePositionSizer + FuturesContractSizer and verifies a realistic backtest scenario produces sane contract counts at different account sizes and risk percentages.

- [ ] **Step 5.8**: Run all sizing tests and existing position-sizing tests. Fix breakage.

---

### Task 6: Wire SSS into Multi-Strategy Dispatch

**Goal:** Fix the gap where `active_strategies = [sss]` produces 0 trades because SSS is never instantiated by the engine.

**Files:**
- Modify: `src/backtesting/vectorbt_engine.py`
- Modify: `src/strategy/signal_blender.py` (verify N>3 strategy support)

**Steps:**

- [ ] **Step 6.1**: Read `src/strategy/strategies/sss/strategy.py` to determine `SSSStrategy.__init__` and `on_bar` signatures. Document what arguments it expects.

- [ ] **Step 6.2**: Add the import at the top of `src/backtesting/vectorbt_engine.py` near line 243-245:
  ```python
  from src.strategy.strategies.sss.strategy import SSSStrategy
  ```

- [ ] **Step 6.3**: Add an `elif name == "sss":` branch in the dispatch loop at lines 251-259:
  ```python
  elif name == "sss":
      sss_config = strategy_configs.get("sss", {})
      self._active_strategies.append(("sss", SSSStrategy(config=sss_config)))
  ```

- [ ] **Step 6.4**: Add SSS dispatch in the per-bar loop at lines 434+:
  ```python
  elif _sn == "sss":
      _sig = _sobj.on_bar(
          ts, open=open_price, high=high, low=low, close=close,
          atr=float(row_5m.get("atr", 0.0) or 0.0),
          # additional fields per SSSStrategy.on_bar contract from Step 6.1
      )
  ```

- [ ] **Step 6.5**: Verify `SignalBlender.blend(signals: list)` works with 4 strategies. The current `multi_agree_bonus=2` is hardcoded; check if the blend logic assumes ≤3 strategies anywhere.

- [ ] **Step 6.6**: Update the pre-flight skip logic at lines 391-395:
  ```python
  has_non_ichimoku = len(self._active_strategies) > 0
  ichimoku_in_active = "ichimoku" in cfg.get("active_strategies", [])
  if has_non_ichimoku and not ichimoku_in_active:
      logger.info("Skipping Ichimoku-only pre-flight — Ichimoku not in active strategies.")
  elif has_non_ichimoku and ichimoku_in_active:
      logger.info("Pre-flight will scan Ichimoku, but other strategies are also active.")
      pre_flight_result = self.health_monitor.pre_flight(candles_1m)
      if pre_flight_result.aborted:
          logger.warning("Pre-flight: Ichimoku produced 0 signals — but other strategies may still trade.")
  else:
      pre_flight_result = self.health_monitor.pre_flight(candles_1m)
      if pre_flight_result.aborted:
          logger.error("Pre-flight ABORTED: %s", pre_flight_result.message)
  ```
  The key change: never let pre-flight halt the run if non-ichimoku strategies are present.

- [ ] **Step 6.7**: Write a smoke test in `tests/integration/test_topstep_backtest.py` (created in Task 14) — for now just verify the engine instantiates with `active_strategies: [ichimoku, asian_breakout, ema_pullback, sss]` without crashing.

---

### Task 7: Strategy Telemetry Collector

**Goal:** Capture every signal generation event, every filter rejection, every entry, with rich context, persisted as Parquet for downstream LLM analysis.

**Files:**
- Create: `src/backtesting/strategy_telemetry.py`
- Create: `tests/test_strategy_telemetry.py`
- Modify: `src/backtesting/vectorbt_engine.py`
- Modify: `src/risk/trade_manager.py`

**Steps:**

- [ ] **Step 7.1**: Create `src/backtesting/strategy_telemetry.py`:
  ```python
  from __future__ import annotations
  from dataclasses import dataclass, field, asdict
  from datetime import datetime, timezone
  from pathlib import Path
  from typing import Literal, Optional
  import json
  import pandas as pd

  EventType = Literal[
      "signal_generated",
      "signal_filtered",
      "signal_entered",
      "signal_rejected_in_trade",
      "signal_rejected_no_open",
      "signal_rejected_edge",
      "signal_rejected_learning",
      "signal_rejected_risk",
      "trade_exited",
  ]

  @dataclass
  class TelemetryEvent:
      timestamp_utc: datetime
      strategy_name: str
      event_type: EventType
      session: str  # asian | london | ny | overlap | off
      hour_of_day_utc: int
      day_of_week: int  # 0=Mon
      pattern_type: Optional[str] = None  # e.g. cbc, fifty_tap, breakout, pullback, tk_cross
      direction: Optional[str] = None  # long | short
      price: Optional[float] = None
      atr: Optional[float] = None
      adx: Optional[float] = None
      confluence_score: Optional[float] = None
      regime: Optional[str] = None
      filter_stage: Optional[str] = None  # which filter rejected this signal
      rejection_reason: Optional[str] = None
      planned_stop_pips: Optional[float] = None
      planned_tp_pips: Optional[float] = None
      planned_size: Optional[float] = None
      extra: dict = field(default_factory=dict)

  class StrategyTelemetryCollector:
      def __init__(self, run_id: str):
          self.run_id = run_id
          self._events: list[TelemetryEvent] = []

      def emit(self, event: TelemetryEvent) -> None:
          self._events.append(event)

      def emit_signal_generated(self, ts, strategy_name, **kwargs): ...
      def emit_filter_rejection(self, ts, strategy_name, filter_stage, reason, **kwargs): ...
      def emit_entry(self, ts, strategy_name, **kwargs): ...

      def to_parquet(self, path: Path) -> None:
          if not self._events:
              # Write an empty parquet so downstream tools don't choke
              pd.DataFrame(columns=[f.name for f in fields(TelemetryEvent)]).to_parquet(path)
              return
          rows = [asdict(e) for e in self._events]
          pd.DataFrame(rows).to_parquet(path, index=False)

      def summary(self) -> dict:
          # Aggregate counts by strategy, event_type, session, pattern_type
          # Return a dict suitable for JSON dump
          ...

      def to_summary_json(self, path: Path) -> None:
          path.write_text(json.dumps(self.summary(), indent=2, default=str))
  ```

  Helper for session classification: hour 0-7 UTC = asian; 7-12 = london; 12-16 = overlap; 16-21 = ny; 21-24 = off.

- [ ] **Step 7.2**: Add session/hour helper functions in the same file, with unit-tested correctness.

- [ ] **Step 7.3**: Instrument `src/backtesting/vectorbt_engine.py`:
  - Construct a `StrategyTelemetryCollector` in `__init__` with a `run_id`
  - In the per-bar loop, after each strategy returns a signal (lines ~434+), call `collector.emit_signal_generated(...)` with the relevant context (price, atr, adx, pattern type if available)
  - When a signal is filtered by edge_manager, learning_engine, or trade_manager, emit `signal_filtered` / `signal_rejected_edge` / `signal_rejected_learning` / `signal_rejected_in_trade` etc. with the reason
  - When an entry is taken, emit `signal_entered`
  - When a trade exits, emit `trade_exited`

- [ ] **Step 7.4**: Instrument `src/risk/trade_manager.py` similarly: when the trade manager rejects a signal because of `in_trade`, `no_open`, `risk_blocked`, etc., it should also call into the collector. Pass the collector through as a constructor arg or method parameter.

- [ ] **Step 7.5**: At backtest completion (after the main loop finishes, near where the dashboard is saved around line 1022+), write the telemetry:
  ```python
  out_dir = Path(self._run_output_dir)  # use existing output dir convention
  collector.to_parquet(out_dir / "strategy_telemetry.parquet")
  collector.to_summary_json(out_dir / "strategy_telemetry_summary.json")
  logger.info("Strategy telemetry: %d events written to %s", len(collector._events), out_dir)
  ```

- [ ] **Step 7.6**: Add a brief end-of-backtest console summary:
  ```
  Strategy telemetry summary:
    ichimoku       : 1234 generated → 12 entered (0.97%)
    asian_breakout :   89 generated →  2 entered (2.25%)
    ema_pullback   :  456 generated →  5 entered (1.10%)
    sss            :  234 generated →  3 entered (1.28%)
  Top rejection stages:
    edge.spread      : 412
    in_trade         : 389
    learning.pre_trade: 178
  ```

- [ ] **Step 7.7**: Create `tests/test_strategy_telemetry.py`:
  - Test event emission and storage
  - Test session classification (UTC hour → label)
  - Test Parquet write produces a readable file
  - Test `summary()` aggregation correctness
  - Test that empty telemetry still writes a valid empty parquet

- [ ] **Step 7.8**: Run `pytest tests/test_strategy_telemetry.py`.

---

### Task 8: Pluggable Tracker Wiring in Engine

**Goal:** The vectorbt engine instantiates the correct prop firm tracker based on `prop_firm.style` and uses a `PropFirmTrackerProtocol`.

**Files:**
- Modify: `src/backtesting/vectorbt_engine.py`
- Modify: `src/backtesting/metrics.py`

**Steps:**

- [ ] **Step 8.1**: Define `PropFirmTrackerProtocol` in `src/backtesting/metrics.py`:
  ```python
  from typing import Protocol
  class PropFirmTrackerProtocol(Protocol):
      def initialise(self, initial_balance: float, first_ts: datetime) -> None: ...
      def update(self, ts: datetime, balance: float) -> None: ...
      def check_pass(self) -> dict: ...
      def to_dict(self) -> dict: ...
  ```

- [ ] **Step 8.2**: Verify `PropFirmTracker` and `MultiPhasePropFirmTracker` already conform; if any method is missing or named differently, add a thin wrapper or rename without breaking external callers.

- [ ] **Step 8.3**: In `vectorbt_engine.py:__init__`, replace the hardcoded tracker instantiation (lines ~204-228) with:
  ```python
  prop_firm_cfg = cfg.get("prop_firm", {})
  prop_firm_style = prop_firm_cfg.get("style", "the5ers_pct_phased")

  if prop_firm_style == "topstep_combine_dollar":
      from src.risk.topstep_tracker import TopstepCombineTracker
      from src.config.models import TopstepCombineConfig
      tx_config = TopstepCombineConfig(**prop_firm_cfg)
      self.prop_firm_tracker = TopstepCombineTracker(tx_config)
      self.multi_phase_tracker = None  # not used in topstep mode
  else:
      # Legacy the5ers / single-phase pct path — unchanged
      self.prop_firm_tracker = PropFirmTracker(...)
      self.multi_phase_tracker = MultiPhasePropFirmTracker(...)
  ```

- [ ] **Step 8.4**: In the main loop, call `prop_firm_tracker.update(ts, balance)` unconditionally each bar. On day rollover (detected via `SessionClock`), the tracker handles its own EOD logic internally.

- [ ] **Step 8.5**: At backtest end, the result dict's `prop_firm` field gets `tracker.to_dict()` regardless of style.

- [ ] **Step 8.6**: Construct `SessionClock` based on prop firm style:
  ```python
  if prop_firm_style == "topstep_combine_dollar":
      session_clock = SessionClock(reset_hour_local=tx_config.daily_reset_hour, reset_tz=tx_config.daily_reset_tz)
  else:
      session_clock = SessionClock(reset_hour_local=0, reset_tz="UTC")
  ```
  Pass `session_clock` to `DailyCircuitBreaker`.

- [ ] **Step 8.7**: Run a smoke test: instantiate `IchimokuBacktester` with the new TopstepX config and verify it doesn't crash on construction. No actual backtest yet — that's Task 13.

---

### Task 9: Cost Model — Round-Trip Commissions for Futures

**Goal:** Replace forex spread/commission model with a futures round-trip-per-contract model when in futures mode.

**Files:**
- Modify: `src/backtesting/vectorbt_engine.py`
- Modify: `config/edges.yaml` (if it has a trading_costs section)

**Steps:**

- [ ] **Step 9.1**: Read the existing cost handling in `vectorbt_engine.py` around lines 150-153 (where `commission_per_lot` and `spread_points` are read from `edges.trading_costs`).

- [ ] **Step 9.2**: Add logic that switches model based on instrument class:
  - Forex: existing `commission_per_lot * lot_size + spread_cost`
  - Futures: `commission_per_contract_round_trip * contract_count` (no spread modeled separately because tick-level spreads on micros are negligible vs commission); apply on entry (debit half) and exit (debit half), or apply full amount on entry — pick one and document
  - Read `commission_per_contract_round_trip` from the instrument config (set in Task 1)

- [ ] **Step 9.3**: Add slippage: 1 tick of adverse slippage on entry and exit for futures (configurable via instrument metadata or profile defaults).

- [ ] **Step 9.4**: Run existing backtest tests; verify forex cost path is unchanged.

---

### Task 10: ChallengeSimulator TopstepX Mode

**Goal:** When `prop_firm.style == "topstep_combine_dollar"`, the challenge simulator runs a single-account simulation (no rolling windows, no Monte Carlo phases) and returns a TopstepX-style verdict.

**Files:**
- Modify: `src/backtesting/challenge_simulator.py`
- Create: `src/backtesting/topstep_simulator.py`
- Create: `tests/test_topstep_simulator.py`

**Steps:**

- [ ] **Step 10.1**: Create `src/backtesting/topstep_simulator.py` with class `TopstepCombineSimulator` that takes a list of trades + the topstep config and replays them through a `TopstepCombineTracker`. Returns a `TopstepCombineResult` dataclass with: `passed`, `failure_reason`, `final_balance`, `peak_balance`, `mll_at_failure`, `days_traded`, `total_trades`, `consistency_check_passed`.

- [ ] **Step 10.2**: Modify `ChallengeSimulator.simulate(...)` to inspect `prop_firm.style`:
  - If `the5ers_pct_phased`: existing rolling-window + monte-carlo logic, unchanged
  - If `topstep_combine_dollar`: delegate to `TopstepCombineSimulator` and wrap the result in a `ChallengeSimulationResult`-compatible structure (or return a new variant)

- [ ] **Step 10.3**: Update `ChallengeSimulationResult` to allow either set of fields, or add a `style` discriminator. Prefer adding optional fields over breaking the dataclass.

- [ ] **Step 10.4**: Create `tests/test_topstep_simulator.py`:
  - Trade list that wins +$3500 over 10 days with biggest day +$1500 → passes (consistency 1500/3500 = 42.9%, under 50%)
  - Trade list that wins +$3500 with biggest day +$2000 → fails consistency
  - Trade list that hits -$1001 in one day → fails daily loss
  - Trade list that drifts down to -$2001 → fails MLL
  - Trade list that hits +$2000 then locks MLL at $50K, then dips to $49999 → fails MLL post-lock

- [ ] **Step 10.5**: Run `pytest tests/test_topstep_simulator.py`.

---

### Task 11: Strategy Pip-to-Price Refactor (Profile-Aware Units)

**Goal:** Strategies stop hardcoding "pip" semantics and instead use the instrument's `price_distance(pips=X)` helper. This unblocks the multi-blend on futures data without re-tuning every strategy.

**Files:**
- Modify: `src/strategy/strategies/sss/strategy.py`
- Modify: `src/strategy/strategies/asian_breakout.py`
- Modify: `src/strategy/strategies/ema_pullback.py`
- Modify: `src/config/profile.py` (add `price_distance` if not yet)

**Steps:**

- [ ] **Step 11.1**: For each strategy, grep for `pips`, `pip`, `pip_value`, hardcoded thresholds. Document each occurrence.

- [ ] **Step 11.2**: For each `*_pips` config field, update the strategy to receive an `instrument_config` (or just `tick_size` + `pip_size` + `class`) and convert at point-of-use:
  ```python
  # Before
  if swing_size < self.config.min_swing_pips:
      return None

  # After
  min_swing_price = price_distance_for(self.instrument, pips=self.config.min_swing_pips)
  if swing_size < min_swing_price:
      return None
  ```

- [ ] **Step 11.3**: For futures, "1 pip" is interpreted as `1 * tick_size * 10` (i.e., 10 ticks = 1 "pip" equivalent in dollar terms). For forex, "1 pip" is `1 * pip_size`. Document this convention in `profile.py`.

  Rationale: existing `min_swing_pips: 0.5` on sss represents "0.5 dollars of price movement on spot gold" — for MGC futures, that's 5 ticks = 0.5 price units. The conversion preserves the same intuition.

- [ ] **Step 11.4**: Update `vectorbt_engine.py` to pass the instrument config into each strategy at construction time.

- [ ] **Step 11.5**: Run any existing strategy unit tests; fix breakage. Add new tests if pip→price conversion isn't covered.

- [ ] **Step 11.6**: NOTE: do not retune strategy parameters in this task. The goal is unit-correctness, not optimization. Retuning is a follow-on after the backtest data tells us where the bottlenecks are.

---

### Task 12: Optimization Loop Futures Support

**Goal:** Optuna optimization works under TopstepX dollar-based rules with a `topstep_combine_pass_score` objective.

**Files:**
- Modify: `scripts/run_optimization_loop.py`
- Modify: `src/optimization/objectives.py`
- Modify: `src/optimization/walk_forward.py`
- Modify: `config/optimization_loop.yaml`

**Steps:**

- [ ] **Step 12.1**: Read `src/optimization/objectives.py` to enumerate existing objective functions and how they pull data from a backtest result.

- [ ] **Step 12.2**: Add a new objective `def topstep_combine_pass_score(result: BacktestResult) -> float`. Score:
  - +1.0 if result.prop_firm.passed
  - Otherwise: a continuous score representing how close the run got
    - `final_balance_score = (final_balance - initial_balance) / config.profit_target_usd` clipped to [-1, 1]
    - Subtract a penalty if MLL was breached: `-0.5`
    - Subtract a penalty if daily loss was breached: `-0.3`
    - Subtract a penalty if consistency failed: `-0.2`
  - The result is a float in roughly [-2, 1]; Optuna maximizes.

- [ ] **Step 12.3**: Update `run_optimization_loop.py` around line 837 (where `active_strategy` is read) to also handle the futures profile path. Read `prop_firm.style` and route to the right objective function.

- [ ] **Step 12.4**: Update `src/optimization/walk_forward.py` to use the active prop firm tracker via the protocol from Task 8 instead of hardcoding the5ers logic.

- [ ] **Step 12.5**: Update `config/optimization_loop.yaml` to add a documented `objective:` field. Default to `pass_rate` (legacy) but allow `topstep_combine_pass_score`.

- [ ] **Step 12.6**: Run a single optimization iteration end-to-end (`python scripts/run_optimization_loop.py --max-iterations 1`) to verify wiring. Capture stderr; fix any breakage.

- [ ] **Step 12.7**: Update `tests/integration/test_optimization_loop.py` with at least one test that asserts the futures path works (mock data is fine — just verify the code doesn't crash on profile switching).

---

### Task 13: Dashboard + Live Dashboard Updates

**Goal:** Both the post-run HTML dashboard and the live HTTP dashboard at :8501 display TopstepX-style metrics when the run is in futures mode.

**Files:**
- Modify: `src/backtesting/dashboard.py`
- Modify: `src/backtesting/live_dashboard.py`
- Modify: `src/backtesting/optimization_dashboard.html` (if it embeds prop firm fields)

**Steps:**

- [ ] **Step 13.1**: Read `src/backtesting/dashboard.py` and `live_dashboard.py` to find where prop firm metrics are rendered.

- [ ] **Step 13.2**: Add a conditional render block based on `result.prop_firm.style` (or whatever discriminator is in the dict):
  - If `topstep_combine_dollar`: show `Current Balance`, `Profit Target ($3,000)`, `Distance to Target`, `Current MLL`, `MLL Locked`, `Distance to MLL`, `Daily Loss Used / $1,000`, `Best Day / Total Profit (Consistency)`, `Verdict`
  - Else (legacy): existing pct-based render unchanged

- [ ] **Step 13.3**: Add a Strategy Telemetry tab/section that reads the run's `strategy_telemetry_summary.json` and displays:
  - Per-strategy generated/entered counts
  - Top rejection stages
  - Per-session distribution
  - Per-pattern distribution

- [ ] **Step 13.4**: Live dashboard equivalent: when running, push the same TopstepX metrics every 5 bars. Reuse the existing push protocol.

- [ ] **Step 13.5**: Smoke test: run a tiny backtest (the small test parquet if one exists, or sliced live data) and verify the dashboard renders without errors.

---

### Task 14: Multi-Blend Activation + Strategy Config Hygiene

**Goal:** Switch the active strategies to the full blend, ensure all `enabled` flags are present, and resolve the `active_strategy` (singular) vs `active_strategies` (plural) cleanly.

**Files:**
- Modify: `config/strategy.yaml`

**Steps:**

- [ ] **Step 14.1**: Update `config/strategy.yaml`:
  - `active_strategies: [ichimoku, asian_breakout, ema_pullback, sss]`
  - Leave `active_strategy: ichimoku` as the legacy live-strategy default (used by `src/strategy/loader.py`); add a comment explaining the split
  - Add `enabled: true` to the `sss` strategy block (it's missing today; add right under `weight: 1.0`)
  - Add a comment block at the top of the file documenting which selector is read by which code path

- [ ] **Step 14.2**: No code changes — config-only.

---

### Task 15: End-to-End TopstepX Backtest Run + Integration Test

**Goal:** Run the multi-strategy blend end-to-end on the ProjectX MGC data under TopstepX rules. Capture results. Build an integration test that exercises this path.

**Files:**
- Create: `tests/integration/test_topstep_backtest.py`
- Output: `reports/topstep_combine_run_<timestamp>/`

**Steps:**

- [ ] **Step 15.1**: Create `tests/integration/test_topstep_backtest.py`:
  - Loads a small slice of `data/projectx_mgc_1m_20260101_20260409.parquet` (e.g., first 5000 bars)
  - Runs the engine end-to-end with `active_strategies: [ichimoku, asian_breakout, ema_pullback, sss]` and TopstepX config
  - Asserts: engine doesn't crash, result has a `prop_firm` dict with `style: topstep_combine_dollar`, telemetry parquet is written and readable
  - Does NOT assert that trades happen (we don't yet know if they do; the telemetry will tell us why if not)

- [ ] **Step 15.2**: Run the integration test: `pytest tests/integration/test_topstep_backtest.py -v`. If it fails, fix until green.

- [ ] **Step 15.3**: Run the FULL backtest:
  ```bash
  python scripts/run_demo_challenge.py --mode validate \
      --data-file data/projectx_mgc_1m_20260101_20260409.parquet \
      2>&1 | tee reports/topstep_combine_run_$(date +%Y%m%d_%H%M%S).log
  ```
  Live dashboard at http://localhost:8501. Wait for completion. Capture exit code.

- [ ] **Step 15.4**: After the run completes, read:
  - The final log
  - The strategy_telemetry_summary.json
  - The dashboard HTML output

  Build a `reports/topstep_combine_run_<timestamp>/SUMMARY.md` with:
  - Total trades per strategy
  - Per-strategy win rate, R-multiples, contribution to drawdown
  - TopstepX combine verdict + failure reason if applicable
  - Top 5 rejection stages from telemetry
  - Per-session distribution of signals
  - Conclusions: did the multi-blend produce trades? If 0 trades on any strategy, what filter killed them?

- [ ] **Step 15.5**: If the run failed catastrophically (engine crash, not a 0-trades result), debug and fix. Re-run.

---

### Task 16: CLAUDE.md Update + Mega-Vision Design Rationale Doc

**Goal:** Document the new futures workflow in CLAUDE.md and write the mega-vision design rationale doc (the architecture decisions, why claude-agent-sdk, why the safety gate model, etc.) — now a reference doc, not a deferred plan, since the implementation is in this same plan.

**Files:**
- Modify: `CLAUDE.md`
- Create: `docs/mega_vision_design.md`

**Steps:**

- [ ] **Step 16.1**: Add a `## Futures Workflow (TopstepX / ProjectX)` section to `CLAUDE.md` with:
  - How to run a TopstepX backtest: command line + config block
  - How profile switching works: `class: futures` in instruments.yaml drives everything
  - Where TopstepX rules live and how to override
  - Where strategy telemetry is written and how to query it
  - How to switch back to forex/the5ers (set `class: forex` and `prop_firm.style: the5ers_pct_phased`)
  - How to enable mega-vision shadow mode + authority mode + the kill switch
  - How to read shadow mode reports for offline evaluation

- [ ] **Step 16.2**: Create `docs/mega_vision_design.md`. This is a DESIGN RATIONALE doc, NOT a deferred plan (the plan IS this file you're reading). Sections:
  - **Why claude-agent-sdk and not the raw API**: agentic loop, hooks, MCP tools, permission system — all for free
  - **Why custom in-process MCP tools for trading state**: keeps the agent's tool calls local and fast; no external service dependency; types stay in-tree
  - **Why event-driven decision cadence**: cost-bounded by signal frequency, not bar frequency
  - **Why multiple safety gate layers**: bypassPermissions is necessary for autonomy but the tradeoff is that hook-based gates become the only line of defense; we layer them so any single hook bug doesn't lose the floor
  - **Why shadow mode is byte-identical to disabled mode on the execution path**: lets us measure agent quality with zero risk
  - **Why training data shape matters from day one**: the same labels drive future fine-tuning; getting the schema right early avoids re-labeling later
  - **Open questions** (still legitimate, but tracked separately from the plan):
    - Model selection between Opus 4.6 (shadow) and Haiku 4.5 (live) — when does Sonnet 4.6 win?
    - Cost budget tuning — what's the right $/run for shadow vs authority?
    - Drift detection thresholds — when do we say the agent's distribution has shifted enough to retrain?
    - Coupling between mega-vision and the discovery loop — should discovery hand-rolled strategies be auto-evaluated by the agent?

- [ ] **Step 16.3**: Verify both files exist and are well-formed.

---

### Task 17: Strategy Retuning for MGC Futures

**Goal:** Retune the four strategies (`ichimoku`, `asian_breakout`, `ema_pullback`, `sss`) for MGC futures point-scale + volatility profile, using a combination of telemetry-driven analysis and Optuna sweeps. Old forex defaults are preserved as a separate config block so the forex profile still works.

**Files:**
- Create: `src/optimization/futures_retuning.py`
- Create: `tests/test_futures_retuning.py`
- Modify: `config/profiles/futures.yaml` (strategy param overrides section)
- Modify: `config/strategy.yaml` (if profile-overrides need a hook)
- Modify: `src/optimization/objectives.py` (per-strategy scoring helpers)
- Output: `reports/futures_retuning_<timestamp>/` (per-strategy sweep artifacts)

**Steps:**

- [ ] **Step 17.1**: Read the strategy_telemetry.parquet from Task 15's run. For each strategy, compute:
  - Distribution of `confluence_score`, `atr`, `adx`, `planned_stop_pips`, `planned_tp_pips` across all `signal_generated` events
  - Top 5 `filter_stage` values that killed signals
  - Per-pattern win-rate proxy (when `signal_entered` events have downstream `trade_exited` results)
  - Per-session distribution

- [ ] **Step 17.2**: Create `src/optimization/futures_retuning.py` with:
  - `def analyze_telemetry(parquet_path) -> RetuningReport`: produces a per-strategy report identifying which thresholds are too tight or too loose vs the actual data distribution
  - `def suggest_param_adjustments(report) -> dict`: returns concrete parameter suggestions per strategy (e.g. "lower ichimoku.signal.min_confluence_score from 1 to 0", "raise sss.min_swing_pips from 0.5 to 2.0")
  - `def run_optuna_sweep(strategy_name, data_file, n_trials=50) -> StudyResult`: runs an Optuna study optimizing the per-strategy entry-only objective on the futures data, with parameter ranges scaled for MGC

- [ ] **Step 17.3**: Run the telemetry-driven analysis for each strategy. Capture proposed param adjustments. Apply the conservative suggestions immediately to a new `futures_overrides:` section in `config/profiles/futures.yaml`:
  ```yaml
  strategy_overrides:
    ichimoku:
      signal:
        min_confluence_score: 0
    sss:
      min_swing_pips: 2.0
      ss_candle_min: 5
      iss_candle_min: 2
      iss_candle_max: 5
      min_stop_pips: 25.0
    asian_breakout:
      min_range_pips: 5
      max_range_pips: 200
    ema_pullback:
      min_ema_angle_deg: 1
      max_ema_angle_deg: 95
  ```
  (These are placeholder values; the actual numbers come from Step 17.2's analysis. The agent should use the analyzer's output, not these literals.)

- [ ] **Step 17.4**: Update the config loader so that `config/profiles/futures.yaml`'s `strategy_overrides` block is merged into the strategy config when the active instrument has `class: futures`. The merge is shallow per-strategy: profile overrides win over strategy.yaml defaults, but per-instrument overrides in instruments.yaml still win over profile overrides.

- [ ] **Step 17.5**: Run a per-strategy Optuna sweep (50 trials each is fine for an initial pass) using the new objective. Persist the best params per strategy to `reports/futures_retuning_<timestamp>/<strategy>_optuna_winners.json`.

- [ ] **Step 17.6**: For each strategy, compare the telemetry-suggested params (Step 17.3) against the Optuna-optimized params (Step 17.5). If they agree within tolerance, apply the Optuna params to `config/profiles/futures.yaml`. If they disagree significantly, keep the conservative telemetry-suggested values and note the disagreement in the retuning report.

- [ ] **Step 17.7**: Re-run the full backtest from Task 15 with the retuned params. Compare trade counts, win rates, and TopstepX verdict against the pre-retuning run. Build `reports/futures_retuning_<timestamp>/RETUNING_REPORT.md` with a side-by-side comparison.

- [ ] **Step 17.8**: Create `tests/test_futures_retuning.py`:
  - Test `analyze_telemetry` on a synthetic telemetry parquet
  - Test `suggest_param_adjustments` on a known distribution
  - Test that profile overrides merge correctly into the strategy config
  - Test that forex path is unaffected when no overrides exist

- [ ] **Step 17.9**: Run `pytest tests/test_futures_retuning.py`. Fix breakage.

---

### Task 18: ProjectX Live Trading

**Goal:** Complete the ProjectX live trading layer so the multi-strategy blend can run live (paper or real) against the TopstepX account. Order placement, market data subscription, position reconciliation, error handling, all wired through.

**Files:**
- Modify: `src/providers/projectx.py` (refactor: extract live bits into siblings)
- Create: `src/providers/projectx_live.py`
- Create: `src/providers/projectx_websocket.py`
- Create: `src/providers/projectx_reconciler.py`
- Create: `src/live/live_runner.py`
- Create: `src/live/order_router.py`
- Modify: `scripts/run_demo_challenge.py`
- Create: `tests/test_projectx_live.py`
- Create: `tests/test_projectx_websocket.py`
- Create: `tests/test_projectx_reconciler.py`
- Create: `tests/test_live_runner.py`
- Create: `tests/test_order_router.py`
- Create: `tests/integration/test_projectx_live_smoke.py`

**Steps:**

- [ ] **Step 18.1**: Read existing `src/providers/projectx.py` to enumerate what's already implemented (auth, historical bars, instrument lookup) and what's missing (order placement, websocket, position state).

- [ ] **Step 18.2**: Create `src/providers/projectx_live.py` — REST live trading client. Methods:
  - `place_order(contract_id, side, qty, order_type, limit_price=None, stop_price=None, time_in_force="DAY") -> OrderResult`
  - `modify_order(order_id, ...) -> OrderResult`
  - `cancel_order(order_id) -> bool`
  - `get_open_orders() -> list[Order]`
  - `get_positions() -> list[Position]`
  - `get_account() -> AccountState` (balance, equity, MLL distance — TopstepX-specific endpoints)
  - `flatten_all() -> list[OrderResult]`

  All requests authenticate via the existing token mechanism. Errors are wrapped in `ProjectXLiveError` with the original API error code/message.

- [ ] **Step 18.3**: Create `src/providers/projectx_websocket.py` — SignalR WebSocket client. ProjectX uses SignalR over WebSocket for both `market_hub_url` (real-time bars + ticks) and `user_hub_url` (account/order/position events). Use the `signalrcore` Python library (add to pyproject.toml if not present).
  - `class MarketHubClient`: connects to `market_hub_url`, subscribes to a contract, yields bar updates (OHLCV) and tick updates
  - `class UserHubClient`: connects to `user_hub_url`, yields order events, fill events, position events, account events
  - Both clients support automatic reconnection with exponential backoff
  - Both handle token refresh on 401
  - Both have a context-manager interface so the live runner can `with` them

- [ ] **Step 18.4**: Create `src/providers/projectx_reconciler.py`:
  - `class PositionReconciler`: holds local position state, periodically (every N seconds) calls `get_positions()` and compares against local; if drift detected, raises `PositionDriftError` with the diff
  - Reconciliation runs in a background thread with cancellation support
  - Trade manager subscribes to drift events and halts trading on detection (escalation to user)

- [ ] **Step 18.5**: Create `src/live/order_router.py`:
  - `class OrderRouter`: takes a `Signal` from the blender + current `AccountState` + `InstrumentSizer` + `PropFirmTracker`
  - Validates: contract caps not exceeded, daily loss limit not breached, MLL distance > planned stop loss, prop firm hasn't already failed
  - Computes contract count via `InstrumentSizer.size_for_risk(...)`
  - Constructs the order(s) — entry + stop + take-profit (bracket order or three separate orders depending on TopstepX support)
  - Submits via `projectx_live.place_order(...)`
  - Returns the resulting `OrderResult`(s) or a `RouterRejection` with reason
  - Emits telemetry events for accept/reject

- [ ] **Step 18.6**: Create `src/live/live_runner.py`:
  - `class LiveRunner`: orchestrates the live trading session
  - Constructor takes provider, instrument config, strategy configs, prop firm tracker, telemetry collector
  - `run(paper: bool = True)` main loop:
    1. Connect to user hub and market hub
    2. Subscribe to MGC bars
    3. On each new bar: feed it to all active strategies (same dispatch logic as backtest engine)
    4. Collect signals → SignalBlender → OrderRouter → ProjectXLive
    5. On fill events from user hub: update position state, notify trade manager, persist to DB via TradeLogger
    6. On position drift: log error, halt trading, escalate
    7. On disconnect: reconnect with backoff
    8. On hard stop signal (Ctrl-C, sigterm, prop firm failure): flatten all positions, cancel all orders, exit cleanly
  - In paper mode: orders go to a local simulator instead of the real TopstepX API. The simulator has the same interface as `projectx_live` but tracks fills against the live market data feed.

- [ ] **Step 18.7**: Modify `scripts/run_demo_challenge.py`:
  - Add `--mode live` flag with sub-flags `--paper` (default) and `--real`
  - When `--mode live`: instantiate `LiveRunner` and call `run(paper=...)` instead of running the backtest engine
  - The live dashboard at :8501 keeps working — it consumes the same telemetry stream

- [ ] **Step 18.8**: Create unit tests:
  - `tests/test_projectx_live.py` — mock HTTP responses, verify request shapes, error handling, retry logic
  - `tests/test_projectx_websocket.py` — mock SignalR, verify subscription/unsubscription, reconnection backoff, message parsing
  - `tests/test_projectx_reconciler.py` — synthetic position diffs, drift detection thresholds
  - `tests/test_order_router.py` — signal-to-order conversion, contract cap enforcement, prop firm gate
  - `tests/test_live_runner.py` — orchestration with mocked provider; verify paper-mode order routing, fill handling, telemetry emission

- [ ] **Step 18.9**: Create `tests/integration/test_projectx_live_smoke.py`:
  - Skipped if `PROJECTX_USERNAME` env var is not set
  - 60-second test: connects in paper mode, subscribes to MGC, waits for at least 3 bars, verifies one signal was processed end-to-end (or at least one strategy's `signal_generated` event was emitted)
  - Cleans up: cancels all paper orders, disconnects

- [ ] **Step 18.10**: Manually run a 60-second paper-mode session:
  ```bash
  python scripts/run_demo_challenge.py --mode live --paper --duration 60s
  ```
  Verify the live dashboard updates, telemetry is emitted, no crashes. Capture logs to `reports/projectx_live_smoke_<timestamp>.log`.

- [ ] **Step 18.11**: Document the live trading workflow in CLAUDE.md (deferred to Task 21's CLAUDE.md update).

---

### Task 19: Discovery Loop Profile Awareness

**Goal:** The discovery loop runs against futures data with TopstepX-shaped objectives. Forex discovery is preserved.

**Files:**
- Create: `src/discovery/profile_adapter.py`
- Modify: `src/discovery/orchestrator.py`
- Modify: `src/discovery/regime_stats.py`
- Modify: `src/discovery/codegen.py` (if exists; else skip this file)
- Create: `tests/test_profile_adapter.py`

**Steps:**

- [ ] **Step 19.1**: Read `src/discovery/orchestrator.py` to understand the current discovery flow (objectives, strategy generation, evaluation).

- [ ] **Step 19.2**: Create `src/discovery/profile_adapter.py`:
  - `def adapt_objective(profile, base_objective) -> Callable`: wraps a discovery objective to score in dollars (futures) or pct (forex)
  - `def adapt_codegen(profile, strategy_template) -> str`: generates strategy code that uses profile-aware unit conversion

- [ ] **Step 19.3**: Update `orchestrator.py` to:
  - Read the active instrument's profile at the top of each discovery run
  - Route through `profile_adapter` for objectives and codegen
  - Persist discovery results with a `profile` field so the knowledge base can filter

- [ ] **Step 19.4**: Update `regime_stats.py`:
  - Add futures session segmentation: Asian futures (5pm-2am CT), EU futures (2am-8am CT), US futures RTH (8:30am-3pm CT), US futures ETH (3pm-5pm CT)
  - Existing forex sessions stay
  - The active set is selected by profile

- [ ] **Step 19.5**: Update `codegen.py` (if it exists) so generated strategy stubs use `instrument.price_distance(pips=X)` instead of hardcoded `pips * pip_size`. If `codegen.py` doesn't exist, skip this step.

- [ ] **Step 19.6**: Create `tests/test_profile_adapter.py`:
  - Test that adapting an objective for futures returns dollar-denominated scores
  - Test that adapting for forex returns pct-denominated scores
  - Test that codegen produces profile-aware strategy stubs (string output check)

- [ ] **Step 19.7**: Run a single discovery iteration on the futures data to verify wiring:
  ```bash
  python -c "from src.discovery.orchestrator import DiscoveryOrchestrator; o = DiscoveryOrchestrator(); o.run_one_iteration(profile='futures')"
  ```
  Capture stderr; fix any breakage.

- [ ] **Step 19.8**: Run `pytest tests/test_profile_adapter.py`.

---

### Task 20: Learning Engine Profile Awareness

**Goal:** The adaptive learning engine includes profile context in trade embeddings and pre-trade analysis. Existing forex learning paths still work.

**Files:**
- Create: `src/learning/profile_context.py`
- Modify: `src/learning/embeddings.py`
- Modify: `src/learning/adaptive_engine.py` (or whatever the actual filename is — discover via grep)
- Create: `tests/test_profile_context.py`

**Steps:**

- [ ] **Step 20.1**: Read `src/learning/embeddings.py` and the adaptive learning engine to find where trade context dicts are built.

- [ ] **Step 20.2**: Create `src/learning/profile_context.py`:
  - `def profile_metadata(instrument_config, prop_firm_config) -> dict`: returns a small dict with `instrument_class`, `prop_firm_style`, `account_size_usd`, `max_loss_limit_usd`, `daily_loss_limit_usd`, `tick_size`, `tick_value_usd` (filled per profile, with `None` for inapplicable fields)
  - Helper: `def profile_embedding_features(metadata) -> np.ndarray`: turns the metadata into a small numeric vector (one-hot for class, normalized dollars, tick scale) suitable for concatenation with the existing embedding

- [ ] **Step 20.3**: Update `src/learning/embeddings.py`:
  - The existing `EmbeddingEngine.embed(trade_context)` builds a 64-dim vector. Add 8 dims of profile features at the end (or at a documented position).
  - Bump the `vector_dim` parameter where the in-memory store is constructed (search for `vector_dim=64`) — change to 72 with a fallback for backward compat.
  - Old persisted embeddings (if any) are zero-padded on the new dims.

- [ ] **Step 20.4**: Update the adaptive learning engine:
  - When building trade context for storage, include profile metadata
  - When pre-trade analysis runs, fetch the active instrument's profile and pass it through
  - When querying similar trades, profile features are part of the similarity calc

- [ ] **Step 20.5**: Create `tests/test_profile_context.py`:
  - Test that profile metadata for a futures instrument returns the right fields populated
  - Test that profile metadata for a forex instrument returns the right fields populated
  - Test that `profile_embedding_features` produces a deterministic 8-dim vector
  - Test that backward-compat 64-dim embeddings still work via zero-padding

- [ ] **Step 20.6**: Run `pytest tests/test_profile_context.py` and any existing learning engine tests. Fix breakage.

---

### Task 21: Dashboard Polish + Screenshot Capture for Mega-Vision

**Goal:** Real-time strategy comparison, regime overlay, telemetry visualization, MLL gauges, trade timeline, and an optional screenshot capture mode that produces the training data the future mega-strategy agent will consume.

**Files:**
- Create: `src/backtesting/dashboard_visualizations.py`
- Create: `src/backtesting/screenshot_capture.py`
- Modify: `src/backtesting/dashboard.py`
- Modify: `src/backtesting/live_dashboard.py`
- Modify: `src/backtesting/optimization_dashboard.html`
- Create: `tests/test_dashboard_visualizations.py`
- Create: `tests/test_screenshot_capture.py`

**Steps:**

- [ ] **Step 21.1**: Create `src/backtesting/dashboard_visualizations.py` with helpers:
  - `def render_per_strategy_panel(telemetry_summary) -> HTMLString`: per-strategy generated/entered/win-rate/PnL widgets
  - `def render_regime_overlay(equity_curve, regime_tags) -> ChartHTML`: equity curve with regime bands
  - `def render_telemetry_summary(summary_json) -> HTMLString`: top rejection stages bar chart, per-session distribution, per-pattern histogram
  - `def render_mll_gauge(prop_firm_state) -> HTMLString`: TopstepX MLL distance gauge
  - `def render_daily_loss_gauge(prop_firm_state) -> HTMLString`: daily loss used / available gauge
  - `def render_trade_timeline(trades, strategies) -> ChartHTML`: per-strategy trade markers on a time axis with color coding
  - `def render_pattern_histogram(telemetry_summary) -> ChartHTML`: pattern type frequency bars

  Output is small self-contained HTML strings using inline SVG or canvas — no new JS deps.

- [ ] **Step 21.2**: Create `src/backtesting/screenshot_capture.py`:
  - `class ScreenshotCapture`: context-managed renderer that takes an OHLCV window + indicators + active signal context and produces a PNG file
  - Uses `mplfinance` (already in pyproject.toml per existing screenshot code)
  - Filename convention: `screenshots/<run_id>/<strategy>_<timestamp>_<event_type>.png`
  - Sidecar metadata: `screenshots/<run_id>/<strategy>_<timestamp>_<event_type>.json` with the full TelemetryEvent payload — this is the training data for the future mega-vision agent
  - Configurable via `screenshot_capture: enabled: bool, every_signal: bool, every_n_bars: int` in run config

- [ ] **Step 21.3**: Modify `src/backtesting/dashboard.py`:
  - Add a "Strategy Telemetry" section that calls into `dashboard_visualizations` helpers
  - Add a "TopstepX Status" panel with MLL gauge + daily loss gauge when style is `topstep_combine_dollar`
  - Add a "Trade Timeline" section
  - Existing forex/the5ers rendering stays as the fallback

- [ ] **Step 21.4**: Modify `src/backtesting/live_dashboard.py`:
  - Push the same widgets via the live update channel
  - Add a refresh cadence for telemetry (every 5 bars, same as state push)
  - Add an MLL distance live gauge that updates per bar in TopstepX mode

- [ ] **Step 21.5**: Modify `src/backtesting/optimization_dashboard.html`:
  - Add HTML mount points for the new widgets
  - Add a "Telemetry" tab
  - Add a "Screenshots" tab listing captured snapshots if screenshot mode was enabled

- [ ] **Step 21.6**: Modify `vectorbt_engine.py` to construct a `ScreenshotCapture` if config enables it, and call into it on each `signal_generated` and `signal_entered` event. The screenshot should embed: last 100 bars OHLCV, indicators relevant to the firing strategy, the signal direction, the planned stop and TP levels.

- [ ] **Step 21.7**: Create `tests/test_dashboard_visualizations.py`:
  - Test each render function produces valid HTML/SVG strings (string contains expected tags, no exceptions on edge inputs)
  - Test empty input cases (zero trades, missing prop firm state, etc.)

- [ ] **Step 21.8**: Create `tests/test_screenshot_capture.py`:
  - Test that screenshot generation produces a non-empty PNG file
  - Test sidecar JSON is well-formed and matches the TelemetryEvent schema
  - Test that disabled config short-circuits cleanly without rendering

- [ ] **Step 21.9**: Run `pytest tests/test_dashboard_visualizations.py tests/test_screenshot_capture.py`. Fix breakage.

- [ ] **Step 21.10**: Add CLAUDE.md notes (final CLAUDE.md update is in Task 22):
  - How to enable screenshot capture
  - Where screenshots and metadata land
  - That this is the training pipeline for the mega-strategy agent

---

### Task 22: Mega-Vision Foundations — Trade Memory + Performance Buckets

**Goal:** Build the persistent trade outcome database and per-strategy performance bucket aggregations that the agent will query at decision time.

**Files:**
- Create: `src/mega_vision/__init__.py`
- Create: `src/mega_vision/trade_memory.py`
- Create: `src/mega_vision/performance_buckets.py`
- Create: `tests/test_mega_vision_trade_memory.py`
- Modify: `src/risk/trade_manager.py` (emit trade close events to trade_memory)
- Optional: SQL migration in `db/migrations/` for new trade_memory table

**Steps:**

- [ ] **Step 22.1**: Create `src/mega_vision/trade_memory.py` with class `TradeMemory`. Backing store: SQLite for portability + a parquet snapshot for fast read (refreshed on backtest end + every N inserts in live mode). Schema (one row per closed trade):
  - `trade_id` (uuid)
  - `opened_at`, `closed_at`, `duration_minutes`
  - `strategy_name`, `instrument_class`, `symbol`
  - `direction` (long/short)
  - `entry_price`, `exit_price`, `stop_price`, `tp_price`
  - `size` (lots or contracts)
  - `pnl_usd`, `r_multiple`
  - `session` (asian/london/ny/overlap/off), `hour_of_day_utc`, `day_of_week`
  - `pattern_type` (cbc, fifty_tap, breakout, pullback, tk_cross, etc.)
  - `regime` (trend_up/trend_down/range/high_vol/low_vol)
  - `confluence_score`, `atr_at_entry`, `adx_at_entry`
  - `prop_firm_style` (the5ers / topstep)
  - `mega_vision_pick` (which strategy the agent picked at this trade's signal — null if shadow mode wasn't on)
  - `mega_vision_agreed` (bool — null if shadow mode wasn't on)
  - `extra_json` (catch-all)

  Methods:
  - `insert(trade: dict) -> str` — returns trade_id
  - `query(filters: dict, limit: int = 100) -> list[dict]`
  - `query_recent(strategy: str, n: int = 20) -> list[dict]`
  - `snapshot_to_parquet(path: Path) -> None`
  - `aggregate_by(*keys) -> pd.DataFrame`

- [ ] **Step 22.2**: Create `src/mega_vision/performance_buckets.py` with class `PerformanceBuckets`:
  - Constructor takes a `TradeMemory` reference
  - `get_buckets(strategy_name: str | None = None, lookback_days: int = 90) -> dict`
  - Returns nested dict keyed by (strategy → session → pattern → regime) with: trade_count, win_rate, avg_r, expectancy, max_drawdown
  - Cached with TTL of 60 seconds during a live run; recomputed on demand in backtest

- [ ] **Step 22.3**: Modify `src/risk/trade_manager.py` so that on every trade close, it calls `trade_memory.insert(...)` with the full close context. The trade_memory instance is constructed by the engine and passed in.

- [ ] **Step 22.4**: Create `tests/test_mega_vision_trade_memory.py`:
  - Insert + query roundtrip
  - Filter queries (by strategy, by session, by date range)
  - Aggregation by multiple keys
  - Parquet snapshot read/write
  - Performance bucket correctness on synthetic data
  - SQLite schema migration idempotence

- [ ] **Step 22.5**: Run `pytest tests/test_mega_vision_trade_memory.py`. Fix breakage.

- [ ] **Step 22.6**: COMMIT: `feat(task-22): Mega-Vision foundations — trade memory + performance buckets`

---

### Task 23: Mega-Vision Context Builder + Screenshot Inference Wiring

**Goal:** Build the context bundle the agent receives at every decision point. Wire the Task 21 screenshot capture into agent inference as image content.

**Files:**
- Create: `src/mega_vision/context_builder.py`
- Create: `src/mega_vision/screenshot_provider.py`
- Create: `tests/test_mega_vision_context_builder.py`

**Steps:**

- [ ] **Step 23.1**: Create `src/mega_vision/context_builder.py` with class `ContextBuilder`:
  - Constructor takes references to: telemetry collector, trade memory, performance buckets, regime detector, screenshot capture
  - `def build(self, ts: datetime, candidate_signals: list[Signal], current_state: dict) -> ContextBundle`
  - Returns a `ContextBundle` dataclass with:
    - `timestamp_utc`
    - `candidate_signals`: list of native blender's candidate signals at this bar (one per strategy that fired)
    - `recent_bars`: last 60 bars of OHLCV (for the agent to reason about price action)
    - `current_indicators`: ATR, ADX, ichimoku cloud values, EMAs, current regime tag
    - `recent_telemetry_summary`: rolling window summary of last 100 signal events per strategy (counts, top rejection reasons)
    - `performance_buckets`: per-strategy buckets from `PerformanceBuckets.get_buckets()` filtered to the current session/pattern/regime
    - `recent_trades`: last 10 trades from trade_memory
    - `screenshot_path`: absolute path to a freshly-rendered chart screenshot for the current bar (or None if disabled)
    - `prop_firm_state`: current TopstepX/the5ers state from the active tracker
    - `risk_state`: position, equity, daily P&L, MLL distance

- [ ] **Step 23.2**: Create `src/mega_vision/screenshot_provider.py`:
  - `class ScreenshotProvider`: thin wrapper around `ScreenshotCapture` from Task 21
  - `def render_for_decision(self, ts, candles_window) -> Path`: renders a chart with the strategies' indicator overlays + candidate signal markers, returns the file path
  - The image will be loaded by the agent layer and passed as a vision content block in the user message

- [ ] **Step 23.3**: Add a helper in the same file: `def load_image_as_content_block(path: Path) -> dict` that returns the standard image content block dict for use with `claude-agent-sdk` user messages: `{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "<base64>"}}`.

- [ ] **Step 23.4**: Create `tests/test_mega_vision_context_builder.py`:
  - Test that `build()` returns a fully-populated bundle on synthetic state
  - Test that screenshot path is None when capture is disabled
  - Test that `recent_bars` is correctly the last 60 bars (not 61, not 59)
  - Test that `performance_buckets` correctly filters by current session/pattern/regime

- [ ] **Step 23.5**: Run `pytest tests/test_mega_vision_context_builder.py`. Fix breakage.

- [ ] **Step 23.6**: COMMIT: `feat(task-23): Mega-Vision context builder + screenshot inference wiring`

---

### Task 24: Mega-Vision Custom Trading Tools (SDK MCP)

**Goal:** Define the seven custom trading tools the agent will call to investigate state, and wrap them in an in-process SDK MCP server.

**Files:**
- Create: `src/mega_vision/tools.py`
- Create: `src/mega_vision/mcp_server.py`
- Create: `tests/test_mega_vision_tools.py`
- Modify: `pyproject.toml` (add `claude-agent-sdk` dependency)

**Steps:**

- [ ] **Step 24.1**: Add `claude-agent-sdk` to `pyproject.toml` dependencies. Pin to a recent stable version. Run `pip install -e .` to verify it resolves.

- [ ] **Step 24.2**: Create `src/mega_vision/tools.py`. Define seven async tools using the `@tool` decorator from `claude_agent_sdk`:

  ```python
  from claude_agent_sdk import tool

  # The tools take a global context object that holds references to the
  # context_builder, trade_memory, performance_buckets, screenshot_provider.
  # Tools are constructed by a factory function that captures these refs in
  # closures so each agent invocation has its own scoped tool set.

  def make_tools(ctx):
      @tool("get_market_state", "Returns current OHLCV bars (last 60), indicators (ATR, ADX, EMAs, ichimoku cloud), and current regime tag.", {})
      async def get_market_state(args):
          state = ctx.context_builder.current_market_state()
          return {"content": [{"type": "text", "text": json.dumps(state, default=str)}]}

      @tool("get_recent_telemetry", "Returns rolling-window summary of recent signal events per strategy: counts generated, counts entered, top rejection reasons.", {"window_n": int})
      async def get_recent_telemetry(args):
          n = args.get("window_n", 100)
          summary = ctx.telemetry_collector.recent_summary(n)
          return {"content": [{"type": "text", "text": json.dumps(summary, default=str)}]}

      @tool("get_strategy_performance_buckets", "Returns per-strategy performance buckets (win rate, avg R, expectancy) for the current session/pattern/regime, optionally filtered by strategy name.", {"strategy_name": str})
      async def get_strategy_performance_buckets(args):
          name = args.get("strategy_name")
          buckets = ctx.performance_buckets.get_buckets(strategy_name=name)
          return {"content": [{"type": "text", "text": json.dumps(buckets, default=str)}]}

      @tool("get_recent_trades", "Returns the last N closed trades with full context.", {"n": int, "strategy_name": str})
      async def get_recent_trades(args):
          n = args.get("n", 10)
          name = args.get("strategy_name")
          trades = ctx.trade_memory.query_recent(strategy=name, n=n) if name else ctx.trade_memory.query({}, limit=n)
          return {"content": [{"type": "text", "text": json.dumps(trades, default=str)}]}

      @tool("get_regime_tag", "Returns the current market regime tag (trend_up, trend_down, range, high_vol, low_vol) plus the indicators that drove the classification.", {})
      async def get_regime_tag(args):
          regime = ctx.regime_detector.current_regime()
          return {"content": [{"type": "text", "text": json.dumps(regime, default=str)}]}

      @tool("view_chart_screenshot", "Returns a chart screenshot of the current market state with active strategies' indicators overlaid. Use this to visually analyze the price action.", {})
      async def view_chart_screenshot(args):
          path = ctx.screenshot_provider.render_for_decision(ctx.current_ts, ctx.current_candles)
          if path is None:
              return {"content": [{"type": "text", "text": "Screenshot capture is disabled."}]}
          import base64
          img_bytes = path.read_bytes()
          img_b64 = base64.standard_b64encode(img_bytes).decode("utf-8")
          return {"content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}}]}

      @tool("record_strategy_pick", "Records the agent's final strategy pick for this decision. Pass strategy_picks (list of strategy names to enable) and confidence (0.0-1.0) and reasoning (string). This is the agent's output — calling this tool ends its decision turn.", {"strategy_picks": list, "confidence": float, "reasoning": str})
      async def record_strategy_pick(args):
          ctx.last_pick = {
              "strategy_picks": args["strategy_picks"],
              "confidence": args["confidence"],
              "reasoning": args["reasoning"],
              "ts": ctx.current_ts,
          }
          return {"content": [{"type": "text", "text": "Pick recorded. End your turn."}]}

      return [get_market_state, get_recent_telemetry, get_strategy_performance_buckets, get_recent_trades, get_regime_tag, view_chart_screenshot, record_strategy_pick]
  ```

- [ ] **Step 24.3**: Create `src/mega_vision/mcp_server.py`:
  ```python
  from claude_agent_sdk import create_sdk_mcp_server
  from .tools import make_tools

  def make_trading_mcp_server(ctx):
      tools = make_tools(ctx)
      return create_sdk_mcp_server("mega-vision-trading-tools", tools=tools)
  ```

- [ ] **Step 24.4**: Create `tests/test_mega_vision_tools.py`:
  - Build a fake `ctx` object with stub references
  - Call each tool directly and verify the response shape
  - Test `get_market_state` returns the expected indicator dict
  - Test `view_chart_screenshot` returns an image content block when path exists, text fallback when None
  - Test `record_strategy_pick` mutates `ctx.last_pick` correctly
  - Test that the tool list returned by `make_tools(ctx)` has exactly 7 entries

- [ ] **Step 24.5**: Run `pytest tests/test_mega_vision_tools.py`. Fix breakage.

- [ ] **Step 24.6**: COMMIT: `feat(task-24): Mega-Vision custom trading tools + SDK MCP server`

---

### Task 25: Mega-Vision Agent + Safety Gates + Arbitration

**Goal:** Build the `MegaStrategyAgent` (the claude-agent-sdk wrapper), the safety gate hooks, and the arbitration layer. Wire everything together so the engine can ask the agent for a decision and receive a safe, gated answer.

**Files:**
- Create: `src/mega_vision/agent.py`
- Create: `src/mega_vision/safety_gates.py`
- Create: `src/mega_vision/arbitration.py`
- Create: `src/mega_vision/cost_tracker.py`
- Create: `prompts/mega_vision_system.md`
- Create: `prompts/mega_vision_user_template.md`
- Create: `config/mega_vision.yaml`
- Create: `tests/test_mega_vision_agent.py`
- Create: `tests/test_mega_vision_safety_gates.py`
- Create: `tests/test_mega_vision_arbitration.py`

**Steps:**

- [ ] **Step 25.1**: Create `prompts/mega_vision_system.md` — the static system prompt that's safe to cache:

  ```markdown
  You are the Mega-Strategy Trading Agent for the itchy-tradebot futures trading system.

  Your job: at each decision point, given a candidate set of trading signals from the
  underlying strategies (ichimoku, asian_breakout, ema_pullback, sss), decide which signals
  (if any) should actually fire — based on the current market state, recent telemetry,
  per-strategy historical performance in similar conditions, and a chart screenshot.

  You have these tools:
    - get_market_state: current bars + indicators + regime
    - get_recent_telemetry: signal generation patterns over the last N events
    - get_strategy_performance_buckets: which strategies have done well in similar conditions
    - get_recent_trades: last N closed trades for context
    - get_regime_tag: explicit regime classification
    - view_chart_screenshot: a visual of the current chart with indicator overlays
    - record_strategy_pick: your final answer

  Procedure:
    1. Investigate the state using whatever tools you need (always view_chart_screenshot at least once)
    2. Consult performance_buckets for the current session/regime/pattern
    3. Decide which subset of the candidate strategies should fire
    4. Call record_strategy_pick with your picks, confidence (0-1), and a one-paragraph reasoning
    5. End your turn

  Hard constraints (the system enforces these — you cannot violate them):
    - You cannot enable a strategy that's not in active_strategies
    - You cannot increase position size
    - You cannot override risk gates or prop firm rules
    - If you pick something invalid, the system rejects it and asks you to retry
    - After 3 rejections, the system falls back to the native blender silently

  Guidance:
    - Prefer fewer, higher-confidence picks over many low-confidence ones
    - When telemetry shows a strategy has been failing in similar conditions, don't pick it
    - When the chart shows a clean pattern that matches a strategy's specialty, prefer that strategy
    - If you're unsure, picking an empty list (skip this signal) is always safe
  ```

- [ ] **Step 25.2**: Create `prompts/mega_vision_user_template.md` — the per-decision user message template:

  ```markdown
  Decision request at {timestamp_utc}.

  Candidate signals from native blender:
  {candidate_signals_summary}

  Current prop firm state:
  - Style: {prop_firm_style}
  - Distance to MLL: {distance_to_mll_usd}
  - Daily loss used: {daily_loss_used_usd} / {daily_loss_limit_usd}
  - Distance to profit target: {distance_to_target_usd}

  Risk state:
  - Open positions: {open_positions_count}
  - Equity: {equity_usd}

  Investigate the state with your tools, then call record_strategy_pick with your decision.
  ```

- [ ] **Step 25.3**: Create `config/mega_vision.yaml`:

  ```yaml
  mode: disabled  # disabled | shadow | authority
  shadow_model: "claude-opus-4-6"
  live_model: "claude-haiku-4-5"
  decision_cadence: per_signal  # per_signal | per_n_bars
  per_n_bars: 5  # only used if decision_cadence is per_n_bars
  cost_budget_usd: 10.00  # max spend per run; falls back to native blender when exceeded
  kill_switch_env_var: "MEGA_VISION_KILL_SWITCH"
  max_retries_per_decision: 3
  fallback_on_timeout_seconds: 30
  prompt_template_versions:
    system: "v1"
    user: "v1"
  ```

- [ ] **Step 25.4**: Create `src/mega_vision/cost_tracker.py` with class `CostTracker`:
  - Tracks cumulative input/output tokens, decision count, $ spent (Opus 4.6: $5/M in, $25/M out; Haiku 4.5: $1/M in, $5/M out)
  - `def can_afford(self, model: str) -> bool`: returns False when budget exceeded
  - `def record(self, model: str, usage: dict) -> None`
  - `def to_dict(self) -> dict` for dashboard rendering

- [ ] **Step 25.5**: Create `src/mega_vision/safety_gates.py`:
  - `class SafetyGates`: holds references to prop firm tracker, instrument sizer, active_strategies, kill switch
  - `def validate_pick(self, pick: dict) -> tuple[bool, str | None]`:
    - Check kill switch env var → reject if set
    - Check picks ⊆ active_strategies → reject if not
    - Check prop firm tracker hasn't failed → reject if it has
    - Check cost tracker → reject if budget exceeded
    - Check open positions count + caps → reject if would breach
    - Returns (valid, reason)
  - `def make_pre_tool_use_hook(self) -> Callable`: returns a hook function that, when called for the `record_strategy_pick` tool, validates the args and returns `{"deny": True, "reason": ...}` if invalid. The Agent SDK retries on deny; after 3 retries, the agent layer falls back.

- [ ] **Step 25.6**: Create `src/mega_vision/agent.py` with class `MegaStrategyAgent`:

  ```python
  from claude_agent_sdk import query, ClaudeAgentOptions, HookMatcher
  from .mcp_server import make_trading_mcp_server
  from .safety_gates import SafetyGates
  from .cost_tracker import CostTracker

  class MegaStrategyAgent:
      def __init__(self, config, ctx, prop_firm_tracker, instrument_sizer, telemetry_collector):
          self.config = config
          self.ctx = ctx
          self.cost_tracker = CostTracker(budget_usd=config.cost_budget_usd)
          self.gates = SafetyGates(
              prop_firm_tracker=prop_firm_tracker,
              instrument_sizer=instrument_sizer,
              kill_switch_env=config.kill_switch_env_var,
              cost_tracker=self.cost_tracker,
          )
          self.system_prompt = self._load_prompt("mega_vision_system.md")
          self.user_template = self._load_prompt("mega_vision_user_template.md")

      async def decide(self, ts, candidate_signals, mode: str = "shadow") -> dict:
          if not self.cost_tracker.can_afford(self.config.shadow_model if mode == "shadow" else self.config.live_model):
              return self._fallback("cost_budget_exceeded")
          if os.environ.get(self.config.kill_switch_env_var):
              return self._fallback("kill_switch")

          self.ctx.current_ts = ts
          self.ctx.last_pick = None

          mcp_server = make_trading_mcp_server(self.ctx)
          model = self.config.shadow_model if mode == "shadow" else self.config.live_model

          options = ClaudeAgentOptions(
              system_prompt=self.system_prompt,
              mcp_servers={"trading": mcp_server},
              allowed_tools=["mcp__trading__get_market_state", "mcp__trading__get_recent_telemetry",
                             "mcp__trading__get_strategy_performance_buckets", "mcp__trading__get_recent_trades",
                             "mcp__trading__get_regime_tag", "mcp__trading__view_chart_screenshot",
                             "mcp__trading__record_strategy_pick"],
              permission_mode="bypassPermissions",
              model=model,
              max_turns=10,
              hooks={"PreToolUse": [HookMatcher(matcher="record_strategy_pick", hooks=[self.gates.make_pre_tool_use_hook()])]},
          )

          user_message = self._render_user_message(ts, candidate_signals)

          try:
              async for message in query(prompt=user_message, options=options):
                  if hasattr(message, "usage"):
                      self.cost_tracker.record(model, message.usage)
          except Exception as e:
              return self._fallback(f"agent_error:{e}")

          if self.ctx.last_pick is None:
              return self._fallback("no_pick_recorded")

          return self.ctx.last_pick

      def _fallback(self, reason: str) -> dict:
          return {"strategy_picks": None, "confidence": 0.0, "reasoning": f"FALLBACK: {reason}", "fallback": True}
  ```

- [ ] **Step 25.7**: Create `src/mega_vision/arbitration.py` with class `Arbitrator`:
  - `def arbitrate(self, agent_pick: dict, native_signals: list[Signal]) -> list[Signal]`:
    - If `agent_pick.fallback`: return all native signals (full native authority)
    - If `mode == "shadow"`: return all native signals (regardless of agent pick) — shadow doesn't change execution
    - If `mode == "authority"`: filter native signals to only include those whose strategy_name is in `agent_pick.strategy_picks`
    - Always emit telemetry for both paths

- [ ] **Step 25.8**: Create `tests/test_mega_vision_safety_gates.py`:
  - Test each gate fires under the right condition
  - Test kill switch env var
  - Test cost budget enforcement
  - Test active_strategies subset enforcement
  - Test 3-retry then fallback flow with a mock agent that always returns invalid picks

- [ ] **Step 25.9**: Create `tests/test_mega_vision_arbitration.py`:
  - Test shadow mode never changes execution
  - Test authority mode correctly filters
  - Test fallback path always returns native
  - Test that telemetry is emitted for both paths

- [ ] **Step 25.10**: Create `tests/test_mega_vision_agent.py`:
  - Mock `query()` from claude_agent_sdk to return fake messages
  - Verify the agent constructs options correctly
  - Verify the agent records cost from message usage
  - Verify the agent falls back on exception
  - Verify the agent falls back on no-pick-recorded after the loop ends

- [ ] **Step 25.11**: Run `pytest tests/test_mega_vision_*.py`. Fix breakage.

- [ ] **Step 25.12**: COMMIT: `feat(task-25): Mega-Vision agent + safety gates + arbitration`

---

### Task 26: Mega-Vision Shadow Mode (Backtest + Live)

**Goal:** Wire shadow mode into both the backtest engine and the live runner. Run the agent on every candidate signal but don't change execution. Persist all decisions for offline evaluation.

**Files:**
- Create: `src/mega_vision/shadow_recorder.py`
- Modify: `src/backtesting/vectorbt_engine.py`
- Modify: `src/live/live_runner.py`
- Modify: `scripts/run_demo_challenge.py` (add `--mega-vision-mode` flag)
- Create: `tests/test_mega_vision_shadow_recorder.py`
- Create: `tests/integration/test_mega_vision_shadow_backtest.py`

**Steps:**

- [ ] **Step 26.1**: Create `src/mega_vision/shadow_recorder.py` with class `ShadowRecorder`:
  - Constructor takes a run_id and output directory
  - `def record(self, ts, candidate_signals, agent_pick, native_pick) -> None`: appends a row to an in-memory buffer
  - `def flush_to_parquet(path: Path) -> None`: writes all rows
  - Schema: ts, candidate_signals_json, agent_picks_json, native_picks_json, agreement_flag, agent_confidence, agent_reasoning, agent_latency_ms, agent_cost_usd, fallback_reason

- [ ] **Step 26.2**: In `vectorbt_engine.py`, after the SignalBlender produces candidate signals at a bar, check `mega_vision.mode`:
  - If `disabled`: existing behavior, no change
  - If `shadow` or `authority`: build context bundle, call `agent.decide(ts, candidates, mode)`, get pick
    - In shadow mode: pass the full native signal list to trade_manager UNCHANGED. Call `shadow_recorder.record(...)`. Continue.
    - In authority mode: pass `arbitrator.arbitrate(agent_pick, candidates)` to trade_manager. Also call shadow_recorder.

- [ ] **Step 26.3**: At backtest end, call `shadow_recorder.flush_to_parquet(out_dir / "mega_vision_shadow.parquet")` if mega_vision mode was non-disabled.

- [ ] **Step 26.4**: In `live_runner.py`, mirror the same wiring on the live path. The live agent uses `live_model` instead of `shadow_model`. Shadow logs flush every N decisions (not just at end) since live runs are unbounded.

- [ ] **Step 26.5**: In `run_demo_challenge.py`, add `--mega-vision-mode {disabled,shadow,authority}` flag (default: disabled — don't break the existing default). Plumb through to engine + live runner.

- [ ] **Step 26.6**: Create `tests/test_mega_vision_shadow_recorder.py`:
  - Test record + flush roundtrip
  - Test parquet schema correctness
  - Test that shadow recordings preserve all required fields

- [ ] **Step 26.7**: Create `tests/integration/test_mega_vision_shadow_backtest.py`:
  - Mock the agent to return deterministic picks
  - Run a small backtest with `mega_vision.mode: shadow`
  - Verify execution path is byte-identical to a `mode: disabled` run on the same data
  - Verify the shadow parquet was written and is readable
  - Verify it has the expected number of rows (one per decision point)

- [ ] **Step 26.8**: Run `pytest tests/test_mega_vision_shadow_recorder.py tests/integration/test_mega_vision_shadow_backtest.py`. Fix breakage.

- [ ] **Step 26.9**: Run a real shadow backtest:
  ```bash
  python scripts/run_demo_challenge.py --mode validate \
      --data-file data/projectx_mgc_1m_20260101_20260409.parquet \
      --mega-vision-mode shadow
  ```
  Confirm the shadow parquet exists and contains rows. This is the FIRST real call to claude-agent-sdk in this plan — verify the package is installed, env vars are set (ANTHROPIC_API_KEY), and inference works. Cap with a low cost budget (~$1) so a misconfigured run can't burn money.

- [ ] **Step 26.10**: COMMIT: `feat(task-26): Mega-Vision shadow mode (backtest + live)`

---

### Task 27: Mega-Vision Live Authority Mode + Offline Evaluation Harness

**Goal:** Promote the agent to authority mode (it controls execution within safety gates) and build the offline evaluation harness that scores agent performance against hindsight.

**Files:**
- Create: `src/mega_vision/eval_harness.py`
- Create: `src/mega_vision/training_data.py`
- Modify: `scripts/run_demo_challenge.py` (extend `--mega-vision-mode authority` plumbing)
- Create: `tests/test_mega_vision_eval_harness.py`
- Create: `tests/integration/test_mega_vision_authority_smoke.py`

**Steps:**

- [ ] **Step 27.1**: Verify Task 26's wiring already supports `--mega-vision-mode authority` end-to-end (the engine + live runner branches in Steps 26.2, 26.4 already include the authority case). If anything is incomplete, finish it here.

- [ ] **Step 27.2**: Run a backtest in authority mode:
  ```bash
  python scripts/run_demo_challenge.py --mode validate \
      --data-file data/projectx_mgc_1m_20260101_20260409.parquet \
      --mega-vision-mode authority
  ```
  Verify the executed trades reflect agent picks (not native) where they differ. Verify safety gates fire correctly. Verify cost stays under budget. Capture output to `reports/mega_vision_authority_run_<timestamp>/`.

- [ ] **Step 27.3**: Create `src/mega_vision/eval_harness.py` with class `OfflineEvalHarness`:
  - Constructor takes paths to: backtest result parquet, shadow_recorder parquet, trade_memory parquet
  - `def score(self) -> EvalReport`: returns:
    - Per-decision agreement rate (agent vs native)
    - Counterfactual P&L: what would the agent's picks have produced vs what native produced (using the actual trade outcomes from trade_memory where the agent agreed, and projected outcomes where it disagreed)
    - Per-strategy override frequency
    - Confidence calibration: do high-confidence picks actually outperform?
    - Latency distribution
    - Cost per decision distribution
    - Drift signal: distribution of agent picks over time (early run vs late run)
  - `def to_markdown(self) -> str`: produces a human-readable report

- [ ] **Step 27.4**: Create `src/mega_vision/training_data.py` with class `TrainingDataPipeline`:
  - `def build_dataset(shadow_parquet, trade_memory_parquet, output_dir) -> int`: produces labeled (state, agent_pick, native_pick, eventual_outcome) examples
  - Each example is a dict with the full context bundle + the picks + the realized P&L
  - Persists to `reports/mega_vision_training/<run_id>/examples.parquet`
  - Returns the count of examples generated
  - This is the dataset future fine-tuning would consume — build it now even though we're not training yet

- [ ] **Step 27.5**: Create `tests/test_mega_vision_eval_harness.py`:
  - Synthetic shadow + trade_memory parquets
  - Verify the eval harness produces correct agreement rates
  - Verify counterfactual P&L math
  - Verify the markdown report is well-formed

- [ ] **Step 27.6**: Create `tests/integration/test_mega_vision_authority_smoke.py`:
  - Mock the Agent SDK with deterministic responses
  - Run a 60-second equivalent (small bar window) in authority mode
  - Verify execution diverges from native where the mock agent picks differently
  - Verify safety gates intercept invalid picks
  - Verify cost tracker stays within budget

- [ ] **Step 27.7**: Run `pytest tests/test_mega_vision_eval_harness.py tests/integration/test_mega_vision_authority_smoke.py`. Fix breakage.

- [ ] **Step 27.8**: Run the eval harness against the Task 26 shadow run output:
  ```bash
  python -c "from src.mega_vision.eval_harness import OfflineEvalHarness; \
             h = OfflineEvalHarness(...); print(h.to_markdown())"
  ```
  Save the markdown to `reports/mega_vision_authority_run_<timestamp>/EVAL_REPORT.md`.

- [ ] **Step 27.9**: Run the training data pipeline:
  ```bash
  python -c "from src.mega_vision.training_data import TrainingDataPipeline; \
             p = TrainingDataPipeline(); n = p.build_dataset(...); print(f'{n} examples')"
  ```
  Verify the parquet exists and the example count is reasonable.

- [ ] **Step 27.10**: COMMIT: `feat(task-27): Mega-Vision live authority mode + offline eval + training data`

---

### Task 28: Final Verification + Cleanup

**Goal:** Run the full test suite, verify nothing is broken, run all four end-to-end smoke tests (vanilla backtest, shadow backtest, vanilla live paper, shadow live paper), build a final summary covering all 28 tasks.

**Files:**
- Read: all modified files (sanity grep)
- Modify: `CLAUDE.md` (final consolidation: futures workflow + live trading + retuning + telemetry + screenshot pipeline + mega-vision)
- Output: `reports/topstep_megavision_implementation_summary.md`

**Steps:**

- [ ] **Step 28.1**: Run the FULL test suite: `pytest tests/ -x --tb=short`. Anything red blocks completion. Fix everything.

- [ ] **Step 28.2**: Run `pytest tests/integration/ -v`. Same. Note: the ProjectX live smoke test and the mega-vision authority smoke test should skip cleanly if their respective env vars (`PROJECTX_USERNAME`, `ANTHROPIC_API_KEY`) are missing.

- [ ] **Step 28.3**: Run the four end-to-end smoke tests:
  ```bash
  # 1. Vanilla TopstepX backtest with retuned params
  python scripts/run_demo_challenge.py --mode validate \
      --data-file data/projectx_mgc_1m_20260101_20260409.parquet

  # 2. Same backtest in mega-vision shadow mode
  python scripts/run_demo_challenge.py --mode validate \
      --data-file data/projectx_mgc_1m_20260101_20260409.parquet \
      --mega-vision-mode shadow

  # 3. 60-second live paper (vanilla)
  python scripts/run_demo_challenge.py --mode live --paper --duration 60s

  # 4. 60-second live paper in mega-vision shadow mode
  python scripts/run_demo_challenge.py --mode live --paper --duration 60s \
      --mega-vision-mode shadow
  ```
  Verify all four complete cleanly. Verify the live dashboard at :8501 shows all the new TopstepX metrics, gauges, regime overlay, telemetry visualizations, and (for runs 2 and 4) the mega-vision shadow stream.

- [ ] **Step 28.4**: Run the offline eval harness against run 2's shadow output. Save the report to `reports/megavision_eval_<timestamp>.md`.

- [ ] **Step 28.5**: Consolidate CLAUDE.md updates from Tasks 16, 18, 21 into a single coherent "## Futures Workflow (TopstepX / ProjectX) + Mega-Vision Agent" section. Include:
  - How to switch profiles (instrument `class`)
  - How to run a backtest under TopstepX rules
  - How to run live in paper mode
  - How to run live for real (with safety warnings)
  - How to enable screenshot capture
  - Where telemetry, screenshots, and reports land
  - How to retune strategy params (`futures_retuning` workflow)
  - How to invoke discovery + learning in futures mode
  - How to switch back to forex/the5ers
  - How to enable mega-vision shadow mode + authority mode
  - The kill switch env var
  - How to read shadow logs and the eval harness output
  - How the training data pipeline works
  - Cost considerations (Opus vs Haiku in shadow vs live)

- [ ] **Step 28.6**: Write `reports/topstep_megavision_implementation_summary.md` with:
  - List of all files changed (use `git status` and `git diff --stat`)
  - Test pass count
  - All four smoke test verdicts (with native vs shadow comparison for runs 1+2)
  - Telemetry summary (per-strategy generated/entered, top rejection stages)
  - Live paper smoke test result (vanilla + shadow)
  - Discovery + learning profile-awareness verification
  - Dashboard new features list
  - Screenshot capture sample paths
  - Mega-vision metrics: total decisions, total cost, agreement rate with native, top divergence cases
  - Eval harness summary (counterfactual P&L, confidence calibration)
  - Training data pipeline output (example count, schema)
  - Known issues / deferred work (any retuning that needs more iterations; any TopstepX endpoints we couldn't verify without a real account; any mega-vision behaviors that need follow-on tuning)

- [ ] **Step 28.7**: Update `~/.claude/ftm-state/blackboard/experiences/` with a comprehensive experience entry:
  - `task_type`: `futures-profile-megavision-full-implementation`
  - `tags`: `[topstepx, projectx, futures, multi-strategy, telemetry, live-trading, discovery, learning, retuning, dashboard, mega-vision, claude-agent-sdk, vision, shadow-mode, authority-mode]`
  - `outcome`: depends on whether everything passed or revealed issues
  - `lessons`: every key surprise — SSS dispatch missing, pip-to-price unit conversion, TopstepX trailing-then-locked semantics, DST handling at 5pm CT, SignalR auth quirks, contract sizing rounding, telemetry volume considerations, retuning convergence speed, agent SDK MCP server quirks, hook semantics, prompt caching effectiveness, agent latency observed, etc.
  - `files_touched`: full list (probably 75+)
  - `decisions_made`: every architectural decision from Section 2 of this plan (2.1–2.13)
  - `stakeholders`: the user (the only stakeholder)
  - `follow_ups`: any deferred items surfaced during execution

- [ ] **Step 28.8**: Print a final completion message:
  ```
  All 28 tasks committed locally on branch feat/futures-profile-topstepx-megavision.
  28 plan-task commits + N parallel-merge commits.

  Smoke tests:
    1. Vanilla TopstepX backtest:    {VERDICT}
    2. Shadow mega-vision backtest:  {VERDICT}
    3. Vanilla live paper (60s):     {VERDICT}
    4. Shadow live paper (60s):      {VERDICT}

  Mega-vision metrics:
    Decisions: {N}, agreement: {pct}%, total cost: ${X}

  Review with: git log --oneline feat/futures-profile-topstepx-megavision
  Diff against main: git diff main..feat/futures-profile-topstepx-megavision

  DO NOT push without explicit user approval.
  Next step: user reviews, then says "push" to publish.
  ```

- [ ] **Step 28.9**: COMMIT: `chore(task-28): final verification + cleanup + summary`

---

## 6. Dependencies Between Tasks

```
Task  1 (Profile)              → Task 2, 4, 5, 11, 18, 19, 20
Task  2 (PropFirmConfig)       → Task 3, 8, 10, 18, 20
Task  3 (TopstepTracker)       → Task 8, 10, 18, 25
Task  4 (SessionClock)         → Task 8, 18
Task  5 (InstrumentSizer)      → Task 8, 18, 25
Task  6 (SSS dispatch)         → Task 15, 17, 18
Task  7 (Telemetry)            → Task 8, 13, 15, 16, 17, 18, 19, 21, 23, 24
Task  8 (Engine wiring)        → Task 9, 12, 13, 15, 17, 21, 26
Task  9 (Cost model)           → Task 15
Task 10 (Sim TopstepX)         → Task 15
Task 11 (Pip refactor)         → Task 15, 17
Task 12 (Optimization)         → Task 17, 28
Task 13 (Dashboard)            → Task 15, 21
Task 14 (Multi-blend config)   → Task 15
Task 15 (Backtest run)         → Task 17, 22, 28
Task 16 (Docs + design)        → Task 28
Task 17 (Strategy retuning)    → Task 28
Task 18 (Live trading)         → Task 26, 28
Task 19 (Discovery profile)    → Task 28
Task 20 (Learning profile)     → Task 28
Task 21 (Dashboard polish)     → Task 23, 28
Task 22 (Trade memory)         → Task 23, 24, 25
Task 23 (Context builder)      → Task 24, 25
Task 24 (Trading tools MCP)    → Task 25
Task 25 (Agent + gates)        → Task 26
Task 26 (Shadow mode)          → Task 27
Task 27 (Authority + eval)     → Task 28
Task 28 (Final verify)         → END
```

**Parallelizable groups** for `superpowers:dispatching-parallel-agents`:
- **Group A** (after Task 1): Task 2, Task 4, Task 5 in parallel
- **Group B** (after A): Task 3 (depends on 2), Task 6 standalone, Task 7 standalone
- **Group C** (after B): Task 8 (depends on 2,3,4,5,7), Task 11 (depends on 1), Task 22 (depends on 7 only — can also start here)
- **Group D** (after C): Task 9, Task 10, Task 12, Task 13, Task 14, Task 19, Task 20 in parallel — backtest pipeline leaves
- **Group E** (independent of A-D, can start anytime after Task 1+2+3+4+5): Task 18 (Live trading) — fully independent; separate worktree
- **Group F** (after Group D): Task 15 (the backtest run)
- **Group G** (after Group F): Task 17 (retuning, needs Task 15 telemetry), Task 21 (dashboard polish), Task 23 (context builder, needs telemetry shape from real run)
- **Group H** (after G): Task 24 (trading tools MCP — depends on Task 22 + Task 23)
- **Group I** (after H): Task 25 (agent + safety gates + arbitration — depends on Task 24)
- **Group J** (after I + E): Task 26 (shadow mode wiring into engine + live runner)
- **Group K** (after J): Task 27 (authority + offline eval + training data)
- **Group L** (after K): Task 16 (docs consolidation), Task 28 (final verify)

---

## 7. Out of Scope

These are explicitly NOT part of this plan. They may become follow-on plans:

- **Mega-vision agent fine-tuning** — Task 27 builds the training data pipeline. Actually fine-tuning a model on it is a separate effort that depends on accumulated data from many runs.
- **Migrating the optimization loop's legacy forex-pct objectives** to ALSO score TopstepX combine — we're adding a new objective alongside, not retrofitting old ones.
- **ProjectX historical data backfill** beyond what's already downloaded. The 2026-02-09 to 2026-04-09 window is what we have; longer history is a separate ingestion problem.
- **Multi-account support** — TopstepX combine is single-account; we model exactly one. Funded multi-account support is its own plan.
- **A second TopstepX product** (e.g., $25K, $100K combine) — once one works, generalization is trivial but not wired in this plan.
- **Latency optimization** of the live trading path — correctness first, throughput later.
- **A real Optuna distributed runner** — we use the existing in-process runner. Distributed sweeps are a follow-on if retuning needs to scale.
- **Long-term storage of screenshot training data** — Task 21 lands files locally; a managed object store / labeling pipeline is a follow-on infra plan.
- **Production deployment + monitoring of the mega-vision agent** — this plan delivers the agent, runs smoke tests, and validates correctness in shadow + authority modes. Putting it on a real funded TopstepX account requires operational gates (alerting, dashboards, on-call, runbooks) that belong in their own plan.
- **Multimodal beyond chart screenshots** — the agent uses chart images via `view_chart_screenshot`. Other modalities (orderbook depth charts, sentiment streams, news headlines) are future expansions.
- **Agent self-improvement loop** — the agent's outputs feed the training pipeline but we don't have the agent automatically retrain on its own data within this plan.

---

## 8. Risk Notes

- **Schema migration risk**: The prop firm discriminated union must accept legacy YAML with no `style` field. If the loader doesn't handle the missing field gracefully, every existing test fixture breaks. Step 2.3 explicitly mitigates this — the agent must verify with `pytest tests/test_config.py` after Task 2.
- **Pre-flight halt risk**: The current pre-flight aborts the whole run on 0 ichimoku signals. Step 6.6 fixes this; if missed, the futures backtest will halt before the multi-blend even gets a chance to trade.
- **DST risk**: TopstepX day rollover is 5pm CT, which means the trading day boundary moves by an hour twice a year. Tests in Step 4.3 must cover this.
- **Sizing rounding risk**: Futures contracts are integer-only. A risk percentage that doesn't divide cleanly into one contract will round down to 0. The sizer needs to handle this (skip the trade with a clear telemetry rejection, not crash).
- **Telemetry volume**: At 5M bars over 2 months ≈ 17000 bars, with 4 strategies × 1-3 events per bar, that's ~200K events. Parquet handles this fine; just don't hold every event in memory if the run is much longer. The current implementation buffers everything — fine for this run, fine for follow-up plans to switch to streaming write.

---

## 9. Success Definition

The plan is complete when all **28 tasks** have every checkbox ticked, the full test suite is green, all four end-to-end smoke tests pass (vanilla backtest, shadow backtest, vanilla live paper, shadow live paper), the strategies have been retuned and re-validated, the discovery + learning paths are profile-aware, the dashboard renders all the new widgets, screenshot capture is functional, the mega-vision agent runs in shadow + authority modes with all safety gates honored, the offline eval harness produces a comparison report against native execution, the training data pipeline produces labeled examples, and the implementation summary documents all of it. The user will then review the summary and decide next steps — likely operationalizing the mega-vision agent against a real funded TopstepX account or iterating on the prompt/tools based on the eval harness output.

---

## 10. Notes for the Executing Agent

- **Do NOT stop for user confirmation between phases.** This plan was approved as a single unit. Run end-to-end.
- **COMMIT after every completed task.** Do NOT batch commits. Do NOT wait until the end. See the Commit Protocol below — this is a hard requirement, not a suggestion. Pushes are still gated.
- **Use `bypassPermissions` for any spawned subagents.** This is a project-wide preference.
- **Use the live dashboard for the final run.** The user wants to watch progress at http://localhost:8501.
- **Prefer additive changes.** The forex/the5ers code path must continue to work. When in doubt, branch on the discriminator rather than refactor in place.
- **Update `INTENT.md` files** in modified modules (`src/risk/INTENT.md`, `src/backtesting/INTENT.md`) for new exports. The repo follows that convention.
- **If a task hits an unexpected blocker** (e.g., a method signature doesn't match what was assumed), document the blocker in the task as a comment, attempt the smallest fix that unblocks the dependency chain, and continue. Do not halt the whole plan for one task's surprise.
- **Follow the existing repo conventions**: tests in `tests/`, integration tests in `tests/integration/`, plans in `docs/superpowers/plans/`, reports in `reports/<run_dir>/`.

---

## 11. Commit Protocol (MANDATORY — applies to every task)

The user requires a commit after every completed task. The git history must map 1:1 to plan tasks so individual tasks can be reviewed, bisected, and rolled back independently.

### When to commit

**Immediately after the last checkbox of a task is ticked AND the task's tests are green.** Before starting the next task. Never batch.

### Pre-commit gate (run for every commit)

1. **Tests green**: the task's own test files must pass. If pytest is red, fix until green. Never commit broken state.
2. **Secret scan**: invoke the `ftm-git` skill (or equivalent secret-scanning gate) before staging. Refuse to commit if any of these are detected anywhere in the staged diff or in any untracked file about to be added:
   - API keys, tokens (including PROJECTX_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, GITHUB_TOKEN, etc.)
   - Hardcoded passwords
   - Private SSH keys
   - `.env` files (these must always stay gitignored)
   - TopstepX session tokens or account credentials
3. **No `git add -A`** — stage only the files this task modified. Use explicit paths. Avoid pulling in build artifacts, `__pycache__`, IDE files, or another task's WIP.
4. **No `--no-verify`** — pre-commit hooks must run. If a hook fails, fix the underlying issue and create a NEW commit. Do NOT use `--amend` to retry; create a fresh commit.

### Commit message format

Use Conventional Commits matching the existing repo style (`git log --oneline` shows examples like `feat: SSS optimization pipeline improvements`, `fix(verify): ...`). For plan tasks:

```
<type>(task-NN): <task title from plan>

<2-4 line body explaining what changed and why, in plain language>

Refs: docs/superpowers/plans/2026-04-09-futures-profile-and-topstepx.md (Task NN)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
```

Where:
- `<type>` is `feat` for new functionality, `fix` for bug fixes, `refactor` for restructuring, `test` for test-only additions, `docs` for documentation, `chore` for tooling/config
- `NN` is the zero-padded task number (`task-01` through `task-22`)
- `<task title>` matches the heading from the plan exactly

**Examples:**

```
feat(task-01): Profile abstraction + InstrumentClass

Adds InstrumentClass enum, ProfileConfig pydantic model, and
profile loading from config/profiles/{forex,futures}.yaml. Also
extends InstrumentConfig with class-aware required-field validation.

Refs: docs/superpowers/plans/2026-04-09-futures-profile-and-topstepx.md (Task 1)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
```

```
feat(task-03): TopstepCombineTracker with trailing MLL semantics

Implements dollar-based trailing maximum loss limit that trails
end-of-day balances and locks at starting balance once first
reached. Includes daily-loss check and consistency-rule enforcement
with timezone-aware 5pm CT day rollover.

Refs: docs/superpowers/plans/2026-04-09-futures-profile-and-topstepx.md (Task 3)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
```

### Always pass the message via HEREDOC

To preserve formatting, use:

```bash
git commit -m "$(cat <<'EOF'
feat(task-NN): ...

Body ...

Refs: docs/superpowers/plans/2026-04-09-futures-profile-and-topstepx.md (Task NN)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

### What to NOT do

- **Do NOT push.** Pushes are explicitly user-gated for this plan. The user reviews commits locally before any push.
- **Do NOT amend** previous commits. Always create fresh commits.
- **Do NOT bundle multiple tasks** into one commit even if they touch overlapping files. The 1:1 task↔commit mapping is the whole point.
- **Do NOT skip the commit** for "small" tasks. Even Task 14 (config-only multi-blend activation) gets its own commit.
- **Do NOT commit reports/, logs, or generated artifacts** unless they were explicitly listed as a task output. The plan's `reports/<run_dir>/` outputs from Task 15 and Task 17 are exceptions — those get committed because they're the run's deliverables.

### Parallelizable groups + commits

When dispatching tasks in parallel (per the dependency groups in §6), each parallel task is responsible for its own commit on its own branch or worktree. After all parallel tasks in a group finish, the executing agent merges/rebases them into the main feature branch in task-number order. The merge commits are NOT plan-task commits and should be labeled `chore(merge): integrate task-NN..task-MM`.

### Final task (Task 22)

The Task 22 commit is a `chore(task-22): final verification + cleanup` that includes the implementation summary report and CLAUDE.md updates. After Task 22 commits, the agent prints:

```
All 22 tasks committed locally on branch feat/futures-profile-topstepx.
22 plan-task commits + N parallel-merge commits.

Review with: git log --oneline feat/futures-profile-topstepx
Diff against main: git diff main..feat/futures-profile-topstepx

DO NOT push without explicit user approval.
Next step: user reviews, then says "push" to publish.
```
