# Futures Profile + Mega-Vision — Operational Gaps Fill Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Run end-to-end without stopping for user confirmation between tasks.

**Created:** 2026-04-09
**Companion to:** `docs/superpowers/plans/2026-04-09-futures-profile-and-topstepx.md` (the main plan)
**Estimated scope:** Medium-Large — ~25-35 files, 16 tasks
**Target executor:** ftm-executor (autonomous, no checkpoints between phases)
**Branch:** `feat/futures-megavision-gaps`
**Prerequisite:** The main plan (28 tasks) must be fully executed and merged before starting this plan. This plan fills the operational gaps identified in a post-plan-review of the main plan.

---

## 1. Goal

The main plan delivers the futures profile + mega-vision agent end-to-end. This follow-up plan addresses **15 operational gaps** that the main plan didn't cover but that will cause pain during or after the first real run:

1. **Native kill switch** for live trading when mega-vision is disabled
2. **OCO bracket orders** — atomic entry+stop+TP instead of three separate orders
3. **Partial fill handling** in the order router
4. **Consecutive loss circuit breaker** — halt after N losses in a row
5. **Disk space / log rotation** policy for telemetry, screenshots, shadow logs, training data
6. **Agent rate limiting** — token bucket on top of the existing cost budget
7. **Prompt versioning** — every shadow recording pinned to a specific prompt hash
8. **Chaos / fault injection tests** — websocket drops, API 500s, malformed data, token refresh failures
9. **Heartbeat + error reporting** — liveness check + fatal error capture for live mode
10. **Config schema validation fail-fast** — typos in YAML fail loudly instead of silently defaulting
11. **Backtest vs live parity check** — same bars, same decisions, regression test
12. **Reconnect-with-open-position scenario** — reconciler handles broker closing a position during a disconnect
13. **Prompt cache hit rate verification** — test that prompt caching actually caches
14. **Human feedback loop** for training data labeling
15. **Operational runbook** — what to do when things go wrong

Each gap is its own task. All are mostly independent and parallelizable. Task 16 is the final verification + cleanup.

---

## 2. Prerequisites

- The main plan is fully committed on `feat/futures-profile-topstepx-megavision` and merged (or at least its commits are in the base branch this plan branches from)
- All main-plan tests are green
- The main-plan smoke tests have been run at least once so there is real telemetry/shadow data to reference
- `ANTHROPIC_API_KEY` is set in the environment for any tasks that touch the agent path
- `PROJECTX_USERNAME` + `PROJECTX_API_KEY` are set for any tasks that touch the live provider

---

## 3. Commit Protocol

Same as the main plan — commit after every completed task, Conventional Commits format with task number, secret scanning via `ftm-git` skill before every commit, no pushes without explicit user approval, no `--amend`, no `--no-verify`. Message format:

```
<type>(gap-NN): <task title>

<body>

Refs: docs/superpowers/plans/2026-04-09-futures-megavision-gaps.md (Task NN)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
```

Use `gap-NN` prefix (not `task-NN`) so git log cleanly distinguishes main-plan commits from gap-fill commits.

---

## 4. Dependencies Between Tasks

```
Task  1 (Kill switch)          → Task 16
Task  2 (OCO brackets)         → Task 16
Task  3 (Partial fills)        → Task 2, Task 16
Task  4 (Consecutive loss)     → Task 16
Task  5 (Disk rotation)        → Task 16
Task  6 (Agent rate limit)     → Task 16
Task  7 (Prompt versioning)    → Task 13, Task 16
Task  8 (Chaos tests)          → Task 1, Task 2, Task 3, Task 12, Task 16
Task  9 (Heartbeat)            → Task 1, Task 16
Task 10 (Config validation)    → Task 16
Task 11 (Backtest/live parity) → Task 16
Task 12 (Reconnect-w-position) → Task 16
Task 13 (Cache verification)   → Task 16
Task 14 (Human feedback)       → Task 16
Task 15 (Runbook)              → Task 16
Task 16 (Final verify)         → END
```

**Parallelizable groups:** Tasks 1, 2, 4, 5, 6, 10, 11, 14, 15 are fully independent — can run in parallel immediately. Tasks 3, 7, 8, 9, 12, 13 depend on one other task but otherwise can parallelize.

---

## 5. Tasks

### Task 1: Native-Mode Kill Switch + Signal Handlers

**Goal:** Add a kill switch for the native live trading path so the bot can be halted cleanly when mega-vision is disabled, and handle `SIGTERM` / `SIGINT` gracefully in live mode.

**Files:**
- Create: `src/live/kill_switch.py`
- Modify: `src/live/live_runner.py`
- Create: `tests/test_kill_switch.py`

**Steps:**

- [ ] **Step 1.1**: Create `src/live/kill_switch.py` with class `KillSwitch`:
  - Constructor: `(env_var: str = "ITCHY_KILL_SWITCH", file_path: Path | None = None)`
  - `def is_active(self) -> bool`: returns True if env var is set to `1`/`true`/`yes`, OR if `file_path` exists on disk
  - `def arm(self) -> None`: writes the file (for manual trigger from another shell)
  - `def disarm(self) -> None`: removes the file
  - Checks are cheap (stat + env lookup) so they can happen every bar in the live loop

- [ ] **Step 1.2**: Modify `src/live/live_runner.py`:
  - Construct a `KillSwitch` instance at startup
  - In the main loop, check `kill_switch.is_active()` before processing each new bar
  - If active: log a clear warning, call `flatten_all_positions()`, cancel all open orders, break the main loop, exit cleanly
  - Register signal handlers for `SIGTERM` and `SIGINT` that set an internal `self._shutdown_requested = True` flag; the main loop checks this flag the same way as the kill switch
  - On Windows, `SIGTERM` is approximated via `SIGBREAK` — use `signal.signal(signal.SIGBREAK, ...)` if available, fall back to `SIGINT` only

- [ ] **Step 1.3**: Add a helper script `scripts/arm_kill_switch.sh` (Unix) and `scripts/arm_kill_switch.bat` (Windows) that touches the kill switch file. Documented in the runbook (Task 15).

- [ ] **Step 1.4**: Create `tests/test_kill_switch.py`:
  - Test `is_active()` with env var set vs unset
  - Test `is_active()` with file present vs absent
  - Test `arm()` and `disarm()` roundtrip
  - Test a mocked `live_runner` main loop that shuts down cleanly when kill switch is armed mid-run (verify flatten_all was called before break)

- [ ] **Step 1.5**: Run `pytest tests/test_kill_switch.py`. Fix breakage.

- [ ] **Step 1.6**: COMMIT: `feat(gap-01): native-mode kill switch + signal handlers`

---

### Task 2: OCO Bracket Orders (Atomic Entry + Stop + TP)

**Goal:** Replace the "pick one" decision from main-plan Task 18.5 with an explicit OCO bracket order model. Entry + stop loss + take profit are submitted as an atomic unit so there's no race where the entry fills but the stop fails to attach.

**Files:**
- Modify: `src/providers/projectx_live.py`
- Modify: `src/live/order_router.py`
- Create: `tests/test_order_router_brackets.py`

**Steps:**

- [ ] **Step 2.1**: Research TopstepX / ProjectX bracket order support. The `/Order/place` endpoint (or equivalent) typically accepts a bracket object with `entry`, `stop_loss`, and `take_profit` sub-objects. If the real API does not support atomic brackets, document the limitation and use the "submit entry, then submit stop + TP on fill event" fallback with aggressive error handling (cancel entry if stop submission fails).

- [ ] **Step 2.2**: Add `def place_bracket_order(self, contract_id, side, qty, entry_price, entry_type, stop_price, tp_price, time_in_force="DAY") -> BracketOrderResult` to `src/providers/projectx_live.py`. Returns a dataclass with `bracket_id`, `entry_order_id`, `stop_order_id`, `tp_order_id`, `status`.

- [ ] **Step 2.3**: On API error during bracket submission, the method MUST guarantee that no partial order state is left. If entry is placed but stop fails: immediately cancel entry and raise `BracketSubmissionError`. If the cancel also fails: raise `CriticalBrokerStateError` which the live_runner handles by setting the kill switch (from Task 1).

- [ ] **Step 2.4**: Modify `src/live/order_router.py`:
  - `OrderRouter.route(signal, ...)` now constructs bracket args from the signal (entry price, computed stop distance, computed TP distance)
  - Calls `place_bracket_order` instead of three separate order calls
  - Handles `BracketSubmissionError` by emitting a router rejection with reason
  - Handles `CriticalBrokerStateError` by emitting a router rejection + logging a critical error + triggering the kill switch

- [ ] **Step 2.5**: Create `tests/test_order_router_brackets.py`:
  - Test happy-path bracket submission (mock the provider)
  - Test entry-fails-immediately → no cleanup needed, rejection returned
  - Test entry-succeeds-then-stop-fails → entry is cancelled, rejection returned
  - Test entry-succeeds-then-stop-fails-then-cancel-also-fails → CriticalBrokerStateError raised, kill switch triggered
  - Test that bracket_id is persisted in router state for later lookup

- [ ] **Step 2.6**: Run `pytest tests/test_order_router_brackets.py`. Fix breakage.

- [ ] **Step 2.7**: COMMIT: `feat(gap-02): OCO bracket orders with atomic entry+stop+TP guarantees`

---

### Task 3: Partial Fill Handling

**Goal:** The order router correctly handles partial fills of the entry order and reports accurate position state downstream.

**Files:**
- Modify: `src/live/order_router.py`
- Modify: `src/providers/projectx_reconciler.py`
- Modify: `src/backtesting/strategy_telemetry.py` (add event type)
- Create: `tests/test_partial_fills.py`

**Steps:**

- [ ] **Step 3.1**: Add a new telemetry event type `partial_fill` to `src/backtesting/strategy_telemetry.py` schema with fields: `expected_qty`, `filled_qty`, `remaining_qty`, `strategy_name`, `bracket_id`.

- [ ] **Step 3.2**: In `src/live/order_router.py`, subscribe to fill events from the user hub. On each fill event:
  - Look up the matching bracket by order ID
  - Update the bracket's filled_qty counter
  - If `filled_qty < expected_qty`: this is a partial fill
    - Emit a `partial_fill` telemetry event
    - The stop-loss and take-profit orders' quantities must be adjusted to match the actual filled quantity — modify them via the provider's `modify_order` API
    - If the remaining quantity is never filled within a timeout window (default: 60 seconds): cancel the remainder, leave the already-filled portion with stop+TP attached
  - If `filled_qty == expected_qty`: normal full fill, no special handling

- [ ] **Step 3.3**: Update `src/providers/projectx_reconciler.py` so the local position state reflects actual filled quantity, not expected quantity. Drift detection must account for bracket state (don't falsely alert when a position is mid-partial-fill).

- [ ] **Step 3.4**: Create `tests/test_partial_fills.py`:
  - Test single partial fill → stop and TP quantities adjusted correctly
  - Test multiple sequential partial fills summing to expected qty
  - Test partial fill with timeout expiry → remainder cancelled, filled portion retained
  - Test reconciler doesn't false-alert during mid-partial-fill state
  - Test telemetry event emitted with correct fields

- [ ] **Step 3.5**: Run `pytest tests/test_partial_fills.py`. Fix breakage.

- [ ] **Step 3.6**: COMMIT: `feat(gap-03): partial fill handling in order router and reconciler`

---

### Task 4: Consecutive Loss Circuit Breaker

**Goal:** Halt trading for the rest of the trading day after N consecutive losing trades, independent of prop firm dollar limits. Catches "strategy is clearly broken right now" before it eats the daily loss limit.

**Files:**
- Create: `src/risk/consecutive_loss_breaker.py`
- Modify: `src/risk/trade_manager.py`
- Modify: `config/strategy.yaml` (add new config block)
- Create: `tests/test_consecutive_loss_breaker.py`

**Steps:**

- [ ] **Step 4.1**: Create `src/risk/consecutive_loss_breaker.py` with class `ConsecutiveLossBreaker`:
  - Constructor: `(max_consecutive_losses: int = 3, session_clock: SessionClock)`
  - State: `current_streak: int`, `halted_until_day: date | None`
  - `def on_trade_close(self, trade: dict, ts: datetime) -> None`:
    - If `trade["r_multiple"] < 0`: increment `current_streak`
    - Else: reset `current_streak = 0`
    - If `current_streak >= max_consecutive_losses`: set `halted_until_day = session_clock.trading_day(ts) + timedelta(days=1)`, log a clear warning
  - `def is_halted(self, ts: datetime) -> bool`:
    - If `halted_until_day is None`: False
    - Else: check if the current trading day (per SessionClock) is before `halted_until_day`
  - `def reset(self) -> None`: manual reset (called at start of new trading day or on explicit user action)
  - `def to_dict(self) -> dict`: for dashboard + telemetry

- [ ] **Step 4.2**: Modify `src/risk/trade_manager.py`:
  - Accept an optional `consecutive_loss_breaker` in the constructor
  - On trade close: call `breaker.on_trade_close(trade, ts)`
  - Before allowing a new entry: check `breaker.is_halted(ts)` — if halted, reject the signal with a clear telemetry rejection event (`rejection_reason: "consecutive_loss_halt"`)

- [ ] **Step 4.3**: Add the new config block to `config/strategy.yaml`:
  ```yaml
  risk:
    # ... existing fields ...
    consecutive_loss_breaker:
      enabled: true
      max_consecutive_losses: 3
  ```
  Update the Pydantic `RiskConfig` model accordingly with a new `ConsecutiveLossBreakerConfig` nested model and a validator.

- [ ] **Step 4.4**: Create `tests/test_consecutive_loss_breaker.py`:
  - Test streak increments on loss, resets on win
  - Test halt triggers at exactly N consecutive losses (not N-1, not N+1)
  - Test halt persists through the rest of the current trading day
  - Test halt clears at the start of the next trading day
  - Test halt works under both forex SessionClock (midnight UTC) and futures SessionClock (5pm CT)
  - Test DST transition doesn't accidentally clear the halt
  - Test that trade_manager rejects new entries with the right telemetry event type when halted

- [ ] **Step 4.5**: Run `pytest tests/test_consecutive_loss_breaker.py`. Fix breakage.

- [ ] **Step 4.6**: COMMIT: `feat(gap-04): consecutive loss circuit breaker with session-aware day boundaries`

---

### Task 5: Disk Space / Log Rotation Policy

**Goal:** Telemetry parquets, screenshots, shadow logs, training data, and run logs rotate and clean up automatically so a long-running live session doesn't fill the disk.

**Files:**
- Create: `config/retention.yaml`
- Create: `src/ops/retention_manager.py`
- Create: `scripts/cleanup_reports.py`
- Modify: `.gitignore`
- Modify: `src/backtesting/dashboard_visualizations.py` (disk usage widget)
- Create: `tests/test_retention_manager.py`

**Steps:**

- [ ] **Step 5.1**: Create `config/retention.yaml` with:
  ```yaml
  retention:
    telemetry_parquets:
      keep_days: 30
      max_total_size_gb: 5.0
    screenshots:
      keep_days: 7
      max_total_size_gb: 10.0
      # when both limits hit, delete oldest first
    shadow_logs:
      keep_days: 90
      max_total_size_gb: 2.0
    training_data:
      keep_days: 365
      max_total_size_gb: 20.0
      # training data is valuable — retention is long
    run_logs:
      keep_days: 14
      max_total_size_gb: 1.0
  ```

- [ ] **Step 5.2**: Create `src/ops/retention_manager.py` with class `RetentionManager`:
  - Constructor loads `config/retention.yaml`
  - `def scan(self) -> RetentionReport`: walks each configured directory, reports current size + oldest file + estimated next action
  - `def cleanup(self, dry_run: bool = True) -> list[Path]`: applies the policy, returns list of files that would be deleted (or were deleted)
  - Policy: delete files older than `keep_days` first; if still over `max_total_size_gb`, delete oldest-first until under the limit

- [ ] **Step 5.3**: Create `scripts/cleanup_reports.py`:
  - CLI: `python scripts/cleanup_reports.py [--dry-run] [--category telemetry|screenshots|shadow_logs|training_data|run_logs|all]`
  - Default: dry-run mode, prints what would be deleted
  - `--apply` flag to actually delete

- [ ] **Step 5.4**: Update `.gitignore` to ensure these paths are excluded:
  ```
  reports/
  !reports/.gitkeep
  .ftm-drafts/
  data/*.parquet
  !data/.gitkeep
  mega_vision_training/
  ```
  (some of these may already be in .gitignore — verify and only add what's missing)

- [ ] **Step 5.5**: Add a "Disk Usage" widget to `src/backtesting/dashboard_visualizations.py` that calls `RetentionManager.scan()` and displays the current state as a simple bar chart per category.

- [ ] **Step 5.6**: Create `tests/test_retention_manager.py`:
  - Test scan on synthetic directory structure
  - Test cleanup deletes files older than keep_days
  - Test cleanup enforces size limit by deleting oldest-first
  - Test dry-run doesn't delete anything
  - Test unknown categories are skipped cleanly
  - Test that a broken retention.yaml produces a clear error

- [ ] **Step 5.7**: Run `pytest tests/test_retention_manager.py`. Fix breakage.

- [ ] **Step 5.8**: COMMIT: `feat(gap-05): disk space rotation policy + retention manager + cleanup CLI`

---

### Task 6: Agent Rate Limiting (Token Bucket)

**Goal:** Token-bucket rate limiter on top of the existing cost budget, so a runaway decision loop can't blow the budget before the cost tracker catches up.

**Files:**
- Create: `src/mega_vision/rate_limiter.py`
- Modify: `src/mega_vision/agent.py`
- Modify: `config/mega_vision.yaml`
- Create: `tests/test_agent_rate_limiter.py`

**Steps:**

- [ ] **Step 6.1**: Create `src/mega_vision/rate_limiter.py` with class `TokenBucketRateLimiter`:
  - Constructor: `(max_decisions_per_minute: int, max_decisions_per_hour: int)`
  - Implements a standard token bucket with two buckets (per-minute and per-hour)
  - `def try_acquire(self) -> bool`: returns True if a token is available in both buckets (and consumes one), False otherwise
  - Thread-safe via `threading.Lock` — matters for the live runner if it ever goes multi-threaded
  - `def status(self) -> dict`: current token counts + time-until-refill for dashboard display

- [ ] **Step 6.2**: Modify `config/mega_vision.yaml` to add:
  ```yaml
  rate_limit:
    max_decisions_per_minute: 10
    max_decisions_per_hour: 200
  ```

- [ ] **Step 6.3**: Modify `src/mega_vision/agent.py`:
  - Construct a `TokenBucketRateLimiter` from the config
  - At the top of `decide()`, call `rate_limiter.try_acquire()` — if False, fall back to the native blender with reason `rate_limited`
  - Rate limit checks happen BEFORE cost budget checks (a cheaper, faster guard)

- [ ] **Step 6.4**: Create `tests/test_agent_rate_limiter.py`:
  - Test that `try_acquire` returns True up to the per-minute limit
  - Test that it returns False after the limit is hit
  - Test that tokens refill after 60 seconds (use time mocking)
  - Test that the per-hour bucket is independent
  - Test thread safety (concurrent try_acquire from multiple threads)
  - Test that the agent's `decide()` method falls back to native when rate limited

- [ ] **Step 6.5**: Run `pytest tests/test_agent_rate_limiter.py`. Fix breakage.

- [ ] **Step 6.6**: COMMIT: `feat(gap-06): agent rate limiting with token bucket`

---

### Task 7: Prompt Versioning for Shadow Recordings

**Goal:** Every shadow recording is pinned to a specific prompt hash so that later prompt iterations don't invalidate historical shadow data.

**Files:**
- Modify: `src/mega_vision/agent.py`
- Modify: `src/mega_vision/shadow_recorder.py`
- Modify: `src/mega_vision/eval_harness.py`
- Create: `tests/test_prompt_versioning.py`

**Steps:**

- [ ] **Step 7.1**: In `src/mega_vision/agent.py`, at constructor time:
  - Read `prompts/mega_vision_system.md` and `prompts/mega_vision_user_template.md` contents
  - Compute SHA256 hashes of each
  - Store as `self.system_prompt_hash`, `self.user_template_hash`
  - Also read the `prompt_template_versions.system` and `prompt_template_versions.user` from config (these are human-readable version labels like `v1`, `v2`)
  - Expose `self.prompt_version_info() -> dict` with both the hash and the label

- [ ] **Step 7.2**: Modify `src/mega_vision/shadow_recorder.py`:
  - Add columns to the schema: `system_prompt_hash`, `system_prompt_version_label`, `user_template_hash`, `user_template_version_label`
  - Every `record()` call must populate these fields
  - The agent passes them in when calling `shadow_recorder.record(...)`

- [ ] **Step 7.3**: Modify `src/mega_vision/eval_harness.py`:
  - Add a `def filter_by_version(self, system_hash: str | None, user_hash: str | None) -> "OfflineEvalHarness"` method
  - The default `score()` method warns (but doesn't fail) if the shadow parquet contains mixed prompt versions, since comparing across versions is apples-to-oranges
  - Add a `def versions_present(self) -> list[dict]`: returns the unique (system_hash, user_hash, labels) combinations in the shadow data

- [ ] **Step 7.4**: Create `tests/test_prompt_versioning.py`:
  - Test that the agent computes hashes deterministically (same content → same hash)
  - Test that modifying the prompt file changes the hash
  - Test that shadow recordings include the hash
  - Test that eval_harness.versions_present() correctly enumerates
  - Test that eval_harness warns on mixed-version data without crashing
  - Test that filter_by_version correctly subsets the data

- [ ] **Step 7.5**: Run `pytest tests/test_prompt_versioning.py`. Fix breakage.

- [ ] **Step 7.6**: COMMIT: `feat(gap-07): prompt versioning in shadow recorder and eval harness`

---

### Task 8: Chaos / Fault Injection Tests

**Goal:** Inject realistic failure modes into the live trading path and verify the system degrades gracefully instead of crashing or entering an inconsistent state.

**Files:**
- Create: `tests/chaos/__init__.py`
- Create: `tests/chaos/conftest.py`
- Create: `tests/chaos/test_websocket_drop.py`
- Create: `tests/chaos/test_api_errors.py`
- Create: `tests/chaos/test_malformed_data.py`
- Create: `tests/chaos/test_token_refresh_failure.py`

**Steps:**

- [ ] **Step 8.1**: Create `tests/chaos/conftest.py` with fixtures for:
  - `mock_projectx_provider`: a fully mocked ProjectX provider where individual method calls can be configured to raise specific errors on specific calls
  - `mock_websocket_hub`: a mocked SignalR hub that can simulate drops, reconnects, and malformed messages
  - `mock_bar_stream`: a mocked bar source that can inject NaN values, out-of-order timestamps, and duplicate bars

- [ ] **Step 8.2**: Create `tests/chaos/test_websocket_drop.py`:
  - Test: websocket drops while a position is open → reconnect → broker reports the position is still open → local state unchanged, reconnection is logged
  - Test: websocket drops while a position is open → reconnect → broker reports the position is CLOSED (stopped out during disconnect) → local state updates, telemetry event emitted, position counter decremented
  - Test: websocket drops while an order is pending → reconnect → broker reports the order is filled → local state transitions from "pending" to "filled"
  - Test: websocket drops and reconnect fails 5 times → backoff retries → eventually succeeds; verify no orders were placed during the outage

- [ ] **Step 8.3**: Create `tests/chaos/test_api_errors.py`:
  - Test: API returns 500 during order placement → router returns rejection, no partial state left
  - Test: API returns 429 during order placement → retry with backoff up to SDK default, then fail cleanly
  - Test: API returns 401 during order placement (token expired) → trigger token refresh, retry once, succeed
  - Test: API returns 401 during order placement AND refresh also fails → critical error, kill switch triggered
  - Test: API times out during order cancellation → reconciler detects drift, halts trading

- [ ] **Step 8.4**: Create `tests/chaos/test_malformed_data.py`:
  - Test: bar with NaN close → skipped with telemetry event, no crash
  - Test: bar with timestamp in the future → rejected, warning logged
  - Test: bar with timestamp before the last bar (out of order) → rejected, warning logged
  - Test: duplicate bar → idempotent handling (second one is a no-op)
  - Test: bar with zero or negative volume → still processed but with a warning

- [ ] **Step 8.5**: Create `tests/chaos/test_token_refresh_failure.py`:
  - Test: token refresh fails once → retry succeeds, no interruption
  - Test: token refresh fails 3 times in a row → halt trading, log critical error, exit cleanly
  - Test: token refresh succeeds but returns malformed response → halt trading, log error

- [ ] **Step 8.6**: Run `pytest tests/chaos/ -v`. Fix breakage. If a real failure mode is discovered during testing, fix the underlying code in the live trading path (Task 18 of main plan) — the chaos tests are allowed to surface real bugs in already-committed code.

- [ ] **Step 8.7**: COMMIT: `test(gap-08): chaos fault injection tests for live trading path`

---

### Task 9: Heartbeat + Error Reporting for Live Mode

**Goal:** The live runner writes a heartbeat file every 60s so an external monitor can detect silent crashes. Fatal exceptions write a crash report before exit.

**Files:**
- Create: `src/live/heartbeat.py`
- Modify: `src/live/live_runner.py`
- Modify: `src/backtesting/dashboard_visualizations.py` (add heartbeat status widget)
- Create: `tests/test_heartbeat.py`

**Steps:**

- [ ] **Step 9.1**: Create `src/live/heartbeat.py` with class `Heartbeat`:
  - Constructor: `(file_path: Path, interval_seconds: int = 60)`
  - `def write(self) -> None`: writes the current timestamp, PID, and a JSON blob with recent activity counters (bars processed, orders placed, decisions made)
  - `def last_heartbeat_age_seconds(self) -> float | None`: for external monitors
  - `def is_stale(self, max_age_seconds: int = 180) -> bool`: for dashboard display

- [ ] **Step 9.2**: Modify `src/live/live_runner.py`:
  - Construct a `Heartbeat` instance pointing at `reports/live_heartbeat.json`
  - Call `heartbeat.write()` every N bars (enough to hit the 60s interval given bar cadence; 12 bars on 5M = 60 minutes, so for live we use a time-based check instead: "has it been 60s since last write? if yes, write")
  - Install a `sys.excepthook` that, on uncaught exception, writes a crash report to `reports/fatal_<timestamp>.log` containing: exception traceback, heartbeat state, current position state, last N telemetry events

- [ ] **Step 9.3**: Add a "Liveness" widget to `dashboard_visualizations.py` that reads the heartbeat file and renders a green/yellow/red dot based on staleness:
  - Green: < 90s old
  - Yellow: 90s – 180s
  - Red: > 180s or file missing

- [ ] **Step 9.4**: Create `tests/test_heartbeat.py`:
  - Test `write()` produces a readable JSON file with expected fields
  - Test `last_heartbeat_age_seconds()` returns a reasonable value after a write
  - Test `is_stale()` returns False for a fresh heartbeat, True for an old one
  - Test that the sys.excepthook writes a crash report on an injected exception (using a subprocess fixture)

- [ ] **Step 9.5**: Run `pytest tests/test_heartbeat.py`. Fix breakage.

- [ ] **Step 9.6**: COMMIT: `feat(gap-09): live-mode heartbeat + crash report capture`

---

### Task 10: Config Schema Validation Fail-Fast

**Goal:** Every YAML config file is validated at load time. Typos and missing fields fail immediately with a useful error including file path and field name.

**Files:**
- Modify: `src/config/models.py`
- Modify: `src/config/loader.py`
- Create: `src/config/validators.py`
- Create: `scripts/validate_config.py`
- Create: `tests/test_config_validation.py`

**Steps:**

- [ ] **Step 10.1**: Audit every Pydantic model added by the main plan. For each: verify that `extra` is set to `"forbid"` (typos become errors) or explicitly `"allow"` with documentation of why.

- [ ] **Step 10.2**: Create `src/config/validators.py` with a central `def validate_all_configs(config_dir: Path) -> ValidationReport` function:
  - Attempts to load every YAML file in `config_dir` through its corresponding Pydantic model
  - Catches Pydantic `ValidationError` and wraps each in a useful error message: file path, field path, expected type, actual value, suggestion (e.g., "did you mean 'shadow'?")
  - Returns a `ValidationReport` with `errors: list[ConfigError]` and `warnings: list[str]`

- [ ] **Step 10.3**: Modify `src/config/loader.py`:
  - On load, use `validate_all_configs()` BEFORE returning anything
  - If there are errors: raise `ConfigValidationError` with the full report as the message
  - If there are only warnings: log them and continue

- [ ] **Step 10.4**: Create `scripts/validate_config.py`:
  - CLI: `python scripts/validate_config.py`
  - Dry-run all config loaders, print a formatted report
  - Exit code 0 on success, 1 on any error
  - Use this in CI / pre-commit to catch config issues before runtime

- [ ] **Step 10.5**: Create `tests/test_config_validation.py`:
  - Test that a valid config loads cleanly
  - Test that a typo in `mega_vision.mode` (e.g., `shaddow`) produces a clear error naming the field + file
  - Test that a missing required field in `instruments.yaml` produces a clear error
  - Test that an extra unknown field produces an error (thanks to `extra="forbid"`)
  - Test that a malformed YAML (syntax error) produces a clear error with line number
  - Test that `validate_all_configs` catches multiple errors in one pass

- [ ] **Step 10.6**: Run `pytest tests/test_config_validation.py`. Fix breakage.

- [ ] **Step 10.7**: COMMIT: `feat(gap-10): fail-fast config schema validation with useful errors`

---

### Task 11: Backtest vs Live Parity Check

**Goal:** Guarantee that running the same strategy on the same bars via the backtest engine vs the live runner produces identical trade decisions. Catches subtle drift between the two paths (bar boundary handling, timezone math, fill semantics).

**Files:**
- Create: `tests/integration/test_backtest_live_parity.py`
- Create: `src/live/replay_mode.py` (a recorded-market mode for the live runner)

**Steps:**

- [ ] **Step 11.1**: Create `src/live/replay_mode.py` with class `ReplayMarketHub`:
  - Implements the same interface as `projectx_websocket.MarketHubClient` but is backed by a pre-recorded parquet file instead of a live WebSocket
  - Emits bars at a configurable speed (instant / 1x / 10x / 100x real-time)
  - Used by the live runner when `--mode live --replay <parquet_file>` is passed
  - Allows the test harness to feed the same bars through both the backtest engine and the live runner

- [ ] **Step 11.2**: Extend `scripts/run_demo_challenge.py` to accept `--mode live --replay <parquet>` as an alternative to `--paper` or `--real`. In replay mode, orders go to a local simulator (same as paper mode), but market data comes from the recorded parquet instead of the SignalR WebSocket.

- [ ] **Step 11.3**: Create `tests/integration/test_backtest_live_parity.py`:
  - Load a small slice of `data/projectx_mgc_1m_20260101_20260409.parquet`
  - Run the backtest engine on the slice → capture the list of trade entries (timestamp, strategy, direction, price, size)
  - Run the live runner in replay mode on the same slice → capture the same list
  - Assert the two lists are identical
  - If they differ: print a diff showing the first divergent entry for debugging
  - Cover both forex and futures profiles

- [ ] **Step 11.4**: Run the test. If it fails, the drift has been caught — identify and fix the underlying cause. Common culprits: bar timestamp handling, `datetime.date()` vs `SessionClock.trading_day()`, different order fill semantics, different commission application timing.

- [ ] **Step 11.5**: Once the test passes, keep it in `tests/integration/` so it runs as part of the suite going forward. It's a regression guard.

- [ ] **Step 11.6**: COMMIT: `test(gap-11): backtest vs live parity regression test + replay mode`

---

### Task 12: Reconnect-With-Open-Position Scenario

**Goal:** The reconciler correctly handles the case where the broker closes a position (stop loss fills) while the local client is disconnected. On reconnect, local state updates to match broker truth.

**Files:**
- Modify: `src/providers/projectx_reconciler.py`
- Modify: `src/risk/trade_manager.py` (accept position state updates from reconciler)
- Create: `tests/test_reconciler_open_position.py`

**Steps:**

- [ ] **Step 12.1**: Modify `src/providers/projectx_reconciler.py`:
  - On every successful reconnect to the user hub, immediately poll `get_positions()` and compare against local position state
  - Classify diffs: local-only (broker closed, we don't know) / broker-only (broker opened, we don't know — this should never happen legitimately) / quantity-diff
  - For local-only: emit a `position_closed_during_disconnect` telemetry event with the full context (entry price, estimated close price from the last bar, estimated R-multiple based on stop/TP levels), update local state to "closed", notify trade_manager
  - For broker-only: this is a critical drift — halt trading, set kill switch
  - For quantity-diff: halt trading, require manual intervention

- [ ] **Step 12.2**: Modify `src/risk/trade_manager.py` to accept position state updates from the reconciler:
  - `def apply_reconciliation(self, diff: PositionDiff) -> None`: updates internal state, emits telemetry

- [ ] **Step 12.3**: Create `tests/test_reconciler_open_position.py`:
  - Scenario 1: local has position, reconnect, broker has same position → no diff, no action
  - Scenario 2: local has position, reconnect, broker has no position → emit `position_closed_during_disconnect`, update local, notify trade_manager
  - Scenario 3: local has no position, reconnect, broker has a position → critical drift, kill switch triggered
  - Scenario 4: local has 5 contracts, reconnect, broker has 3 contracts → quantity diff, halt trading, manual intervention required
  - Scenario 5: rapid reconnect cycles (drop, reconnect, drop, reconnect) → idempotent, no duplicate telemetry events

- [ ] **Step 12.4**: Run `pytest tests/test_reconciler_open_position.py`. Fix breakage.

- [ ] **Step 12.5**: COMMIT: `feat(gap-12): reconciler handles broker-closed positions during disconnects`

---

### Task 13: Prompt Cache Hit Rate Verification

**Goal:** Verify that the agent's system prompt caching actually works. Detect silent invalidators (timestamps in prompts, unsorted JSON in context, etc.) that would make caching a no-op.

**Files:**
- Modify: `src/mega_vision/cost_tracker.py`
- Modify: `src/mega_vision/agent.py`
- Create: `tests/test_prompt_caching.py`
- Modify: `src/backtesting/dashboard_visualizations.py` (cache hit rate widget)

**Steps:**

- [ ] **Step 13.1**: Modify `src/mega_vision/cost_tracker.py` to track cache metrics per decision:
  - `cache_creation_tokens: int`
  - `cache_read_tokens: int`
  - `uncached_input_tokens: int`
  - `def cache_hit_rate(self) -> float`: ratio of cache_read / (cache_read + uncached), across all recorded decisions
  - `def cache_health_check(self) -> dict`: returns `{healthy: bool, hit_rate: float, warning: str | None}`; healthy when hit_rate >= 0.5 after at least 10 decisions

- [ ] **Step 13.2**: Modify `src/mega_vision/agent.py` to pull cache metrics from each query's usage object and feed them to the cost tracker. The agent SDK's usage object includes `cache_creation_input_tokens` and `cache_read_input_tokens` fields — propagate these correctly.

- [ ] **Step 13.3**: Add a "Prompt Cache Health" widget to `dashboard_visualizations.py` that shows the current cache hit rate as a gauge (target: >= 50% after warmup). Red when below target, green when above.

- [ ] **Step 13.4**: Create `tests/test_prompt_caching.py`:
  - Mock the agent SDK to return usage with specific cache_creation / cache_read values
  - Test that the cost tracker correctly computes hit rate
  - Test the cache_health_check heuristic
  - Test with zero decisions (no NaN / divide-by-zero)
  - Integration-style test: run the agent twice on identical context (with a real or mocked claude-agent-sdk that honors caching), assert second call has `cache_read_tokens > 0`
  - Test that a silent invalidator (e.g., injecting a unique UUID into the context on every call) makes the hit rate drop to near zero — this is the "audit" test that catches regressions

- [ ] **Step 13.5**: Run `pytest tests/test_prompt_caching.py`. Fix breakage.

- [ ] **Step 13.6**: COMMIT: `feat(gap-13): prompt cache hit rate verification + health check`

---

### Task 14: Human Feedback Loop for Training Data Labeling

**Goal:** Give the user a simple CLI to walk through shadow decisions and mark each as good / bad / neutral with optional notes. These qualitative labels enrich the training data for future fine-tuning.

**Files:**
- Create: `scripts/label_shadow_decisions.py`
- Modify: `src/mega_vision/training_data.py`
- Create: `tests/test_label_shadow_cli.py`

**Steps:**

- [ ] **Step 14.1**: Create `scripts/label_shadow_decisions.py`:
  - CLI: `python scripts/label_shadow_decisions.py --shadow-parquet <path> [--resume]`
  - Walks through shadow recordings one at a time
  - For each decision, displays:
    - Timestamp + session + regime
    - Candidate signals from native blender
    - Agent's pick + confidence + reasoning
    - Eventual outcome (if known from trade_memory)
    - Option to view the screenshot inline (opens in default image viewer)
  - Prompts: `[g]ood / [b]ad / [n]eutral / [s]kip / [q]uit`
  - On `g` / `b` / `n`: optionally prompt for a one-line note, then persist
  - Labels are written to a sidecar file: `<shadow_parquet_stem>_labels.parquet` with columns: `decision_ts`, `human_label`, `note`, `labeled_at`
  - `--resume` flag: skip decisions that already have a label
  - Uses `readline` for history and arrow-key editing

- [ ] **Step 14.2**: Modify `src/mega_vision/training_data.py`:
  - `TrainingDataPipeline.build_dataset` now accepts an optional `labels_parquet: Path` argument
  - When labels are provided, the output examples include the `human_label` and `note` fields
  - Unlabeled decisions are still included (with `human_label: None`) so the pipeline can run before labeling is complete
  - Add a `def labeled_count(self) -> int` helper for progress display

- [ ] **Step 14.3**: Create `tests/test_label_shadow_cli.py`:
  - Use a synthetic shadow parquet + simulated stdin
  - Test that labels are correctly written to the sidecar file
  - Test `--resume` skips already-labeled decisions
  - Test quit mid-session saves progress
  - Test that the training data pipeline correctly merges labels

- [ ] **Step 14.4**: Run `pytest tests/test_label_shadow_cli.py`. Fix breakage.

- [ ] **Step 14.5**: COMMIT: `feat(gap-14): human feedback CLI for shadow decision labeling`

---

### Task 15: Operational Runbook

**Goal:** A single `docs/runbook.md` that documents every operational procedure the user might need when something goes wrong in production (or during a live paper run).

**Files:**
- Create: `docs/runbook.md`

**Steps:**

- [ ] **Step 15.1**: Create `docs/runbook.md` with the following sections:

  1. **How to halt trading immediately**
     - Native mode: `bash scripts/arm_kill_switch.sh` OR `kill -TERM <live_runner_pid>`
     - Mega-vision mode: `export MEGA_VISION_KILL_SWITCH=1` (for the calling shell) OR `touch ~/.claude/ftm-state/megavision_kill`
     - Both at once: arm both switches
     - Confirmation: check `reports/live_heartbeat.json` shows `status: halted`

  2. **How to flatten all positions manually**
     - Via the provider: `python -c "from src.providers.projectx_live import ProjectXLive; p = ProjectXLive.from_env(); p.flatten_all()"`
     - Via the TopstepX web UI as a fallback
     - Always verify with `p.get_positions()` that the flatten was successful

  3. **How to recover from a halted state**
     - Identify WHY it halted (check `reports/fatal_*.log` and `reports/live_heartbeat.json`)
     - If prop firm limit was hit: wait for the trading day reset (5pm CT), the halt auto-clears
     - If consecutive loss breaker fired: investigate the losing streak first; manual reset via `python -c "from src.risk.consecutive_loss_breaker import ConsecutiveLossBreaker; ...; breaker.reset()"` only if you understand why
     - If critical drift (reconciler): investigate broker state first, never blindly reset

  4. **How to reconcile a state mismatch**
     - Run the reconciler manually: `python scripts/reconcile_positions.py`
     - Compare output with broker's reported state in the TopstepX web UI
     - If they disagree: broker is truth. Update local state via `trade_manager.apply_reconciliation(...)` or simply restart the live runner (it reconciles at startup)

  5. **How to roll back a bad prompt version**
     - Shadow/authority mode uses `prompts/mega_vision_system.md` and `prompts/mega_vision_user_template.md`
     - Version history: use git (`git log prompts/`) to find the previous version
     - Revert: `git checkout <prev-sha> -- prompts/mega_vision_system.md`
     - The config field `prompt_template_versions.system` should be bumped to reflect the rollback (e.g., `v2` → `v1_rollback`)
     - Shadow recordings made during the bad version should be filtered out when scoring (use `eval_harness.filter_by_version(system_hash=<prev_hash>, ...)`)

  6. **How to rotate credentials**
     - ProjectX: log in to TopstepX web UI, regenerate API key, update `.env`, restart live runner
     - Anthropic: log in to Anthropic Console, regenerate API key, update `.env`, restart
     - Never commit `.env` to git (already in `.gitignore` but double-check)
     - After rotation, run `scripts/validate_config.py` to verify the new creds work

  7. **How to investigate a telemetry anomaly**
     - Load the latest telemetry parquet: `python -c "import pandas as pd; df = pd.read_parquet('reports/<run>/strategy_telemetry.parquet'); print(df.groupby(['strategy_name', 'event_type']).size())"`
     - Common anomalies: one strategy generating 10x the signals of others (threshold issue); one rejection stage dominating (filter too aggressive); per-session skew (timezone bug)
     - Follow-up: `df[df.strategy_name == 'sss'].event_type.value_counts()` etc.

  8. **How to halt the agent but keep native trading running**
     - Set `export MEGA_VISION_KILL_SWITCH=1` — the agent will fall back to the native blender immediately
     - Native trading continues undisturbed
     - To re-enable: unset the env var and wait for the next decision point

  9. **How to dry-run the config before a live session**
     - `python scripts/validate_config.py` — catches typos, missing fields, malformed YAML before any code runs
     - `python scripts/run_demo_challenge.py --mode validate --data-file <small_parquet>` — runs a tiny backtest to verify end-to-end wiring without touching live markets

  10. **How to monitor disk usage**
      - `python scripts/cleanup_reports.py --dry-run` — shows what would be deleted under current retention policy
      - `du -sh reports/ data/ screenshots/ mega_vision_training/` — raw disk usage
      - Apply cleanup: `python scripts/cleanup_reports.py --apply`

  11. **Who to contact**
      - TopstepX support: (user's responsibility — placeholder)
      - Anthropic API issues: `https://status.anthropic.com`
      - Self-help: `docs/runbook.md` (this file), `CLAUDE.md`, `docs/mega_vision_design.md`

- [ ] **Step 15.2**: Link the runbook from CLAUDE.md under the "Operations" section (create that section if it doesn't exist after the main plan's CLAUDE.md updates).

- [ ] **Step 15.3**: Verify that every command in the runbook is actually runnable — no typos, no references to nonexistent scripts. Run each command in dry-run mode where possible.

- [ ] **Step 15.4**: COMMIT: `docs(gap-15): operational runbook for futures + mega-vision`

---

### Task 16: Final Verification + Summary

**Goal:** Verify all 15 gaps are addressed, run the full test suite, update CLAUDE.md, write the gap-plan summary.

**Files:**
- Modify: `CLAUDE.md`
- Output: `reports/futures_megavision_gaps_summary.md`

**Steps:**

- [ ] **Step 16.1**: Run the FULL test suite: `pytest tests/ -x --tb=short`. Everything must be green including the new chaos tests, parity test, kill switch tests, etc.

- [ ] **Step 16.2**: Run the chaos test suite specifically: `pytest tests/chaos/ -v`. These are allowed to take longer because they simulate network failures. Fix anything red.

- [ ] **Step 16.3**: Run a smoke test that exercises the new operational features:
  ```bash
  # 1. Validate config fails fast on a known-bad config
  python scripts/validate_config.py  # should pass
  # (then manually break a field, rerun, verify error message is useful, then revert)

  # 2. Dry-run the cleanup
  python scripts/cleanup_reports.py --dry-run

  # 3. Run a tiny backtest that produces telemetry + shadow data
  python scripts/run_demo_challenge.py --mode validate \
      --data-file data/projectx_mgc_1m_20260101_20260409.parquet \
      --mega-vision-mode shadow \
      --max-bars 500

  # 4. Verify the shadow parquet contains prompt version columns
  python -c "import pandas as pd; df = pd.read_parquet('reports/<latest>/mega_vision_shadow.parquet'); print(df.columns.tolist())"

  # 5. Verify the cost tracker reports cache hit rate
  python -c "from src.mega_vision.cost_tracker import CostTracker; ..."

  # 6. Verify the consecutive loss breaker fires correctly on synthetic data
  pytest tests/test_consecutive_loss_breaker.py -v

  # 7. Verify the heartbeat widget updates during a live replay
  python scripts/run_demo_challenge.py --mode live --replay <small_parquet> --duration 30s
  ```

- [ ] **Step 16.4**: Add an "Operational Hygiene" subsection to CLAUDE.md's futures workflow section with:
  - How to arm / disarm the kill switch
  - Where the heartbeat file lives
  - How to run cleanup
  - Where the runbook is
  - Where the chaos tests live and when to run them

- [ ] **Step 16.5**: Write `reports/futures_megavision_gaps_summary.md` with:
  - List of all 16 gap-plan tasks with completion status
  - Summary of new files created (diff vs main plan completion state)
  - Test pass count (should be significantly higher than main-plan baseline due to new chaos + parity + gap-specific tests)
  - Any discovered issues during execution (e.g., "the parity test caught a timezone bug in the backtest engine that was fixed in Task 11 — see commit sha")
  - Any remaining known issues / follow-ups

- [ ] **Step 16.6**: Update the blackboard experience entry from the main plan with additional `follow_ups` documenting:
  - Which gaps were covered in this plan
  - Any new gaps discovered during gap-plan execution that should be a follow-on plan

- [ ] **Step 16.7**: Print a final completion message:
  ```
  Gap plan complete. 16 gap-fill commits landed on branch feat/futures-megavision-gaps.
  All 15 operational gaps identified in the post-plan review are addressed.

  Combined with the main plan: 28 (main) + 16 (gap) = 44 total commits.

  Review:
    git log --oneline feat/futures-megavision-gaps
    git diff main..feat/futures-megavision-gaps

  DO NOT push without explicit user approval.
  Next step: user reviews both plans' summaries, then says "push" to publish.
  ```

- [ ] **Step 16.8**: COMMIT: `chore(gap-16): final verification + gap-plan summary`

---

## 6. Out of Scope

These remain out of scope even after the gap plan, same reasoning as the main plan:

- Mega-vision agent fine-tuning (plan builds the data pipeline, not the training)
- Production deployment + monitoring (Sentry, Datadog, etc.)
- CI/CD pipeline automation
- Multi-account / multi-product TopstepX variants
- Latency optimization of the live path
- Regression tests against a captured "known-good" full run (the parity test covers drift, not quality regressions)
- Agent self-improvement loop
- Multimodal inputs beyond chart screenshots

---

## 7. Notes for the Executing Agent

- **Prerequisite check**: verify the main plan is committed and merged (or at minimum, its commits are in the current branch base) before starting Task 1. If prerequisites are missing, STOP and report.
- **Do NOT stop for user confirmation between tasks.** Same as main plan.
- **Commit after every task** using `gap-NN` prefix. Same protocol as main plan §11.
- **Use `bypassPermissions` for any spawned subagents.**
- **Prefer additive changes.** Don't refactor main-plan code unless a gap fix requires it. If refactoring is needed, document why in the commit message.
- **If a gap fix reveals a main-plan bug**, fix it as part of the gap task and note the bug in the commit message. Do NOT start a "main plan v2" — keep the fix scoped to the current gap task.
- **Parallelize aggressively**. Most gaps are independent (see dependency graph in §4). Dispatch Tasks 1, 2, 4, 5, 6, 10, 11, 14, 15 concurrently via `superpowers:dispatching-parallel-agents`.
