# FTM Executor — Progress

**Plan:** Futures Profile + TopstepX Combine + Mega-Strategy Trading Agent
**Plan path:** `docs/superpowers/plans/2026-04-09-futures-profile-and-topstepx.md`
**Branch:** `feat/futures-profile-topstepx-megavision`
**Started:** 2026-04-09
**Status:** IN PROGRESS

## Execution Summary

| Wave | Tasks | Status | Started | Completed |
|------|-------|--------|---------|-----------|
| 0 | Pre-work: uncommitted carry-forward | COMPLETE | 2026-04-09 | 2026-04-09 |
| 1 | 1 (Profile) | PENDING | — | — |
| 2 | 2, 4, 5 (PropFirm union, SessionClock, InstrumentSizer) | PENDING | — | — |
| 3 | 3 (TopstepTracker), 6 (SSS dispatch), 7 (Telemetry) | PENDING | — | — |
| 4 | 8 (Engine wiring), 11 (Pip refactor), 22 (Trade memory) | PENDING | — | — |
| 5 | 9 (Cost), 10 (Sim), 12 (Opt), 13 (Dashboard), 14 (Multi-blend), 19 (Discovery), 20 (Learning) | PENDING | — | — |
| 6 | 18 (Live trading) — parallel | PENDING | — | — |
| 7 | 15 (End-to-end backtest run) | PENDING | — | — |
| 8 | 17 (Retuning), 21 (Dashboard polish), 23 (Context builder) | PENDING | — | — |
| 9 | 24 (Trading tools MCP) | PENDING | — | — |
| 10 | 25 (Agent + safety gates) | PENDING | — | — |
| 11 | 26 (Shadow mode) | PENDING | — | — |
| 12 | 27 (Authority + eval) | PENDING | — | — |
| 13 | 16 (Docs), 28 (Final verify) | PENDING | — | — |

## Task Status

| # | Title | Files | Status | Audit | Notes |
|---|-------|-------|--------|-------|-------|
| 1 | Profile abstraction + InstrumentClass | src/config/profile.py, config/profiles/*, src/config/models.py | PENDING | — | |
| 2 | Prop firm discriminated union | src/config/models.py, config/strategy.yaml | PENDING | — | |
| 3 | TopstepCombineTracker | src/risk/topstep_tracker.py | PENDING | — | |
| 4 | SessionClock + profile-aware reset | src/risk/session_clock.py, circuit_breaker.py | PENDING | — | |
| 5 | InstrumentSizer (lots vs contracts) | src/risk/instrument_sizer.py, position_sizing.py | PENDING | — | |
| 6 | Wire SSS into multi-strategy dispatch | src/backtesting/vectorbt_engine.py | PENDING | — | SSS temporarily removed in pre-work commit |
| 7 | Strategy telemetry collector | src/backtesting/strategy_telemetry.py | PENDING | — | |
| 8 | Pluggable tracker wiring in engine | src/backtesting/vectorbt_engine.py, metrics.py | PENDING | — | |
| 9 | Cost model (futures commissions) | src/backtesting/vectorbt_engine.py | PENDING | — | |
| 10 | ChallengeSimulator TopstepX mode | src/backtesting/topstep_simulator.py | PENDING | — | |
| 11 | Strategy pip-to-price refactor | sss, asian_breakout, ema_pullback strategies | PENDING | — | |
| 12 | Optimization loop futures support | scripts/run_optimization_loop.py, objectives.py | PENDING | — | |
| 13 | Dashboard + live dashboard updates | dashboard.py, live_dashboard.py | PENDING | — | |
| 14 | Multi-blend activation (config only) | config/strategy.yaml | PENDING | — | |
| 15 | End-to-end TopstepX backtest run | tests/integration/test_topstep_backtest.py | PENDING | — | Runs full backtest on MGC data |
| 16 | CLAUDE.md + mega-vision design rationale doc | CLAUDE.md, docs/mega_vision_design.md | PENDING | — | |
| 17 | Strategy retuning for MGC futures | src/optimization/futures_retuning.py | PENDING | — | |
| 18 | ProjectX live trading | src/providers/projectx_*, src/live/* | PENDING | — | Pre-Task-18 scaffolding already in place |
| 19 | Discovery loop profile awareness | src/discovery/profile_adapter.py | PENDING | — | |
| 20 | Learning engine profile awareness | src/learning/profile_context.py | PENDING | — | |
| 21 | Dashboard polish + screenshot capture | dashboard_visualizations.py, screenshot_capture.py | PENDING | — | |
| 22 | Mega-Vision foundations (trade memory) | src/mega_vision/trade_memory.py | PENDING | — | |
| 23 | Mega-Vision context builder + screenshots | src/mega_vision/context_builder.py | PENDING | — | |
| 24 | Mega-Vision custom trading tools (SDK MCP) | src/mega_vision/tools.py, mcp_server.py | PENDING | — | |
| 25 | Mega-Vision agent + safety gates + arbitration | src/mega_vision/agent.py | PENDING | — | |
| 26 | Mega-Vision shadow mode (backtest + live) | src/mega_vision/shadow_recorder.py | PENDING | — | |
| 27 | Mega-Vision live authority + offline eval + training data | src/mega_vision/eval_harness.py | PENDING | — | |
| 28 | Final verification + cleanup | CLAUDE.md, reports/topstep_megavision_implementation_summary.md | PENDING | — | |

## Activity Log

### 2026-04-09 — Phase A complete
Carried forward 6 prior-work commits to `feat/futures-profile-topstepx-megavision`:
- docs(plans): futures plan + the5ers overhaul
- feat(dashboard): live candlestick chart
- fix(trade-manager): exit at stop level
- refactor(engine): multi-concurrent trade loop (SSS deferred to Task 6)
- feat(providers): ProjectX provider scaffolding (pre-Task-18)
- chore: optimization loop tweaks + helper scripts
