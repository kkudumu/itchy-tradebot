# Mega-Vision Trading Agent — Design Rationale

This document is a design-rationale reference, not a plan. The plan lives
at `docs/superpowers/plans/2026-04-09-futures-profile-and-topstepx.md`
(Tasks 22–27 cover the agent implementation). This doc explains *why*
those tasks chose the architecture they did, so future maintainers can
judge edge cases and tradeoffs without re-deriving the whole decision
tree.

## TL;DR

The mega-vision agent is a Claude-powered strategy *selector* that runs
on top of the deterministic signal blender. At each signal event, it
gets the current market state + telemetry + performance buckets +
chart screenshot and chooses which candidate strategies (if any)
should actually fire. It cannot override risk gates, size positions,
or enable disabled strategies. It can only filter and reorder.

It runs in one of three modes:
- **disabled** — native blender only
- **shadow** — agent is consulted on every signal but native execution is
  byte-identical to disabled mode; decisions are logged for offline eval
- **authority** — agent's pick replaces the native selection (within
  safety gates)

The goal is *not* "replace human strategy tuning" — it's "let the agent
pick up regime-dependent patterns that the deterministic strategies
miss, using vision + telemetry + performance history that the
strategies themselves don't see."

## 1. Why `claude-agent-sdk` and not the raw Claude API

The Agent SDK wraps four things we'd otherwise have to reimplement:

1. **The agentic tool-use loop** — pass tools, agent decides which to
   call, result goes back to agent, repeat until agent calls a
   terminal tool. Raw API requires manual orchestration of every turn.
2. **The permission system + hooks** — `PreToolUse` hooks intercept
   tool calls before they execute. Our `record_strategy_pick` tool
   uses this to enforce safety gates.
3. **In-process MCP servers** — `create_sdk_mcp_server` lets us expose
   Python functions as tools without spinning up a separate process or
   network endpoint. Tool implementations stay in-tree as normal Python
   methods.
4. **Model selection per query** — `ClaudeAgentOptions(model=...)`
   takes a model ID like `claude-opus-4-6` or `claude-haiku-4-5-20251001`
   without threading it through every call site.

Crucially, the SDK authenticates via the installed Claude CLI session,
inheriting the user's subscription. That means we don't need an
`ANTHROPIC_API_KEY` env var and the user isn't billed per-call for
inference — which matters because live authority mode could easily
make thousands of calls per day.

## 2. Why in-process MCP trading tools

The agent's tools read strategy state, telemetry, performance buckets,
recent trades, regime tags, and chart screenshots. All of this lives
inside the same Python process that runs the backtest / live runner.

Moving these tools to a separate MCP server (HTTP or subprocess) would
add:
- Latency (network or pipe IPC on every tool call)
- Serialization overhead (pickling DataFrames, embeddings, images)
- A whole new failure mode (tool server crashes, reconnection logic)
- A deployment surface (now we have two processes to start)

Keeping them in-process via `create_sdk_mcp_server` means:
- Tools are Python functions with direct access to the same objects
  the engine is using
- No IPC overhead — the agent can call `get_market_state` hundreds of
  times in a single decision if it wants
- Types stay in-tree; no protobuf / JSONSchema dance

The only argument for splitting them out would be "multi-tenant agent
serving" — which we don't have and won't need for a single-user trading
bot.

## 3. Why event-driven decision cadence (not bar-driven)

The native blender produces candidate signals at roughly 1–5% of bars
(the rest are no-op). If the agent ran on every bar, we'd be paying
inference cost for ~95% dead calls. Instead, the agent is only
consulted when the blender has at least one candidate signal to
evaluate.

This makes inference cost scale with signal frequency, not data
frequency. On the MGC 1-minute data, that's roughly:
- 57K bars × ~0.5% entry rate ≈ 285 decisions over 2 months
- At Haiku 4.5 rates (~$1/M in, $5/M out), budget is trivial
- At Opus 4.6 rates (shadow mode only, ~$5/M in, $25/M out), still
  under $10 for a full backtest

It also makes the agent's context bundle smaller — it only sees the
state around events it needs to reason about, not an endless stream of
bars.

## 4. Why multiple safety gate layers

`permission_mode="bypassPermissions"` means the agent can call any of
its tools without asking for interactive approval. That's necessary
for autonomy but removes the usual human-in-the-loop safety net.

So we compensate with three independent gate layers:

1. **Pre-tool-use hook on `record_strategy_pick`** — before the pick
   is recorded, `SafetyGates.validate_pick()` runs five checks (kill
   switch env var, picks ⊆ active_strategies, prop firm hasn't failed,
   cost budget, position/contract cap). A failed check returns `deny`
   and the agent must retry within bounds.
2. **3-retry ceiling** — if the agent can't produce a valid pick in 3
   attempts, the system falls back to the native blender silently and
   logs the failure. This prevents infinite retry loops on an agent
   that's fundamentally confused.
3. **Kill switch env var** — checked BEFORE calling the agent at all.
   Setting `MEGA_VISION_KILL_SWITCH=1` skips all agent calls and uses
   the native blender, without needing to restart the process or
   change config. This is the fastest way to disable the agent in an
   emergency.

Any single layer could have a bug. Having three means a bug in one
doesn't lose the floor.

## 5. Why shadow mode is byte-identical to disabled mode on the execution path

Shadow mode is how we measure agent quality without risking money.
If shadow mode could *accidentally* change execution — because of a
logging side effect, a tool call that mutated state, a retry that
happened to succeed once out of three — its measurements would be
compromised.

The architecture enforces this by splitting execution and agent calls
at the `Arbitrator` layer: the native blender's output goes to the
trade manager unchanged, and the agent's output goes to
`shadow_recorder.record(...)`. The arbitrator in shadow mode returns
the native signals verbatim regardless of agent output.

Authority mode flips the arbitrator to filter native signals by the
agent's picks — but shadow and authority are different code paths,
not different config flags on the same path.

This makes it safe to turn shadow mode on in production with zero
risk that the agent could execute a trade.

## 6. Why training data shape matters from day one

The shadow recorder persists every decision to parquet with:
- `timestamp_utc` — when the decision was made
- `candidate_signals_json` — what the blender offered
- `agent_pick_json` — what the agent chose
- `native_pick_json` — what the blender would have picked
- `agreement_flag` — were they the same?
- `agent_reasoning` — the agent's one-paragraph explanation
- `agent_latency_ms`, `agent_cost_usd` — operational metrics
- `fallback_reason` — null unless the agent errored

Every trade that subsequently fires gets joined back to its originating
decision row via the decision timestamp. Over many runs, we accumulate
a labeled dataset: `(market state, telemetry, screenshot, agent pick,
eventual P&L)`.

That dataset is the training set for future fine-tuning. Getting the
schema right now avoids having to re-label or re-run a year of shadow
mode later. Even if we never fine-tune, the eval harness uses the same
schema to compute counterfactual P&L and confidence calibration.

## Open questions (not blockers, but track)

- **Model selection** — When does Sonnet 4.6 win over Haiku 4.5 in
  live mode? Expectation: it wins when the context bundle gets bigger
  (e.g., 60 bars + 6 tools + screenshot) and Haiku starts missing
  subtle regime shifts. But we need the eval harness numbers to
  confirm.
- **Cost budget tuning** — `cost_budget_usd: 10.00` per run is a guess.
  The true number comes from running the agent against a week of real
  data and measuring actual spend + decision count.
- **Drift detection thresholds** — At what point does the agent's
  pick distribution shift enough to retrain? We don't have a metric
  yet. One candidate: KL divergence between early-week and late-week
  pick distributions.
- **Discovery loop coupling** — Should new strategies generated by
  the discovery loop auto-register as candidates the agent can pick
  from? Or should they always go through a shadow-mode validation run
  first? Leaning toward the latter — shadow validates new strategies
  on real data without risking a run.

## References

- Plan: `docs/superpowers/plans/2026-04-09-futures-profile-and-topstepx.md`
  (Tasks 22–27)
- Claude Agent SDK docs: https://docs.anthropic.com/en/api/agent-sdk
- Prop firm rules: CLAUDE.md "Futures Workflow" section
