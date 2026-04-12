# LLM Meta-Optimizer — Design Spec

## Overview

An LLM reasoning layer that runs after each Optuna epoch, analyzing all trial results, signal log patterns, and chart screenshots to produce config changes and code patches. Sits on top of the existing Optuna optimizer — Optuna grinds parameters, the LLM makes structural improvements that parameter tweaking can't reach.

Uses `claude -p --dangerously-skip-permissions` via the CLI (not the API) so it runs on the existing subscription with persistent output capture.

## How It Fits

```
Optuna epoch finishes (50 trials)
         │
         ▼
  LLM Meta-Optimizer
  (one claude -p call per instrument)
         │
    ┌────┴────┐
    │         │
  Config    Code patch
  changes   on temp branch
    │         │
  Apply     Backtest
  directly    │
    │       Better? ──No──→ delete branch
    │         │
    │        Yes → merge
    │         │
    └────��────┘
         │
    Next Optuna epoch
    (improved code + params)
```

## Execution Method

```bash
claude -p "$(cat /tmp/llm_analysis_prompt.txt)" \
    --dangerously-skip-permissions \
    --output-format json \
    2>&1 | tee reports/llm_analysis/MCL_epoch_3.txt
```

The prompt is assembled by the AnalysisBuilder from DB queries and chart screenshots, written to a temp file, then piped to `claude -p`. Output is captured to `reports/llm_analysis/` for persistent memory — the next epoch's prompt includes a summary of prior analysis so the LLM builds on its own reasoning.

## What the LLM Sees

### Layer 1: Trial Summary (~2K tokens)

Aggregated from `optimization_runs` table:
- Instrument, epoch number, total trials, pass rate
- Best/worst trial metrics
- Per-strategy signal counts, entry counts, win rates, avg R
- Top rejection reasons (sizing_cap, circuit_breaker, trend_direction, etc.)

### Layer 2: Pattern Analysis (~5K tokens)

Pre-computed SQL queries from `signal_log`, grouped by:
- `trade_type` + `direction` + `htf_trend_1h` → win rate, avg R, count
- Strategy × session (asian/london/ny) → which sessions produce winners
- Stop distance distribution → are stops too tight/wide
- Trade duration → winners vs losers bar count
- Confluence score buckets → does higher confluence = better WR

Example output the LLM sees:
```
SSS cbc_pattern long + bearish_1h:  15 entered, 2 won (13% WR), avg R: -0.6
SSS fifty_tap short + bearish_1h:    6 entered, 4 won (67% WR), avg R: +0.8
FXAOG kijun_bounce long + bullish:   8 entered, 6 won (75% WR), avg R: +1.2
FXAOG kumo_breakout long + bearish:  4 entered, 0 won (0% WR), avg R: -1.0
EMA_PB all: avg stop distance = $0.07 (3.5 ticks), 100% of losses = -1.0R
```

### Layer 3: Chart Screenshots (~20-40K tokens)

mplfinance candlestick charts rendered as PNG, included in the prompt:
- Top 3 winning trades: 1H chart with entry/exit/SL/TP marked, indicators overlaid
- Top 3 losing trades: same format
- Best trial equity curve
- 4H overview of full data period with major swing structure

Generated using the existing `enable_screenshots` backtester infrastructure, saved to `reports/llm_analysis/charts/`.

### Layer 4: Prior Analysis History (~1-2K tokens)

Summary of the LLM's own prior reasoning for this instrument (from `reports/llm_analysis/`):
```
Epoch 1: "Identified EMA PB stops too tight. Suggested ATR-based stops."
Epoch 2: "ATR stops merged. WR improved 18% → 31%. Now investigating
          why SSS longs dominate in a bear trend."
```

This gives the LLM continuity across epochs — it remembers what it already tried.

## What the LLM Outputs

The prompt instructs the LLM to produce a structured response with three sections:

### 1. Reasoning Document

Free-form analysis stored for history. What patterns it found, what the charts showed, what it thinks the edge is, what's broken.

### 2. Config Changes

JSON block with parameter overrides to apply directly:
```json
{
  "config_changes": {
    "strategies.sss.entry_mode": "fifty_tap",
    "strategies.ema_pullback.enabled": false,
    "risk.initial_risk_pct": 0.3,
    "strategies.ichimoku.adx.threshold": 12
  }
}
```

Applied to `config/strategy.yaml` and `config/edges.yaml` by the ConfigApplicator. No sandbox needed — these are just param tweaks.

### 3. Code Patches

JSON block with file edits:
```json
{
  "code_patches": [
    {
      "file": "src/strategy/strategies/ema_pullback.py",
      "description": "Use ATR-based stop instead of slow_ema",
      "search": "sl = float(self._armed_slow_ema or ema_slow)",
      "replace": "sl = float(close - atr * 2.0) if direction == 'long' else float(close + atr * 2.0)"
    }
  ]
}
```

Applied to a temp git branch. Backtest runs on the branch. If the patched version produces better results (higher combine pass rate or better return), merge to main. If worse, delete the branch.

## Database Changes

### signal_log: Add columns

```sql
ALTER TABLE signal_log ADD COLUMN IF NOT EXISTS trade_type TEXT;
ALTER TABLE signal_log ADD COLUMN IF NOT EXISTS reasoning_summary TEXT;
```

`trade_type` captures: walking_dragon, tk_crossover, kumo_breakout, kijun_bounce, cbc_pattern, fifty_tap, ema_pullback_breakout, asian_range_break, etc.

`reasoning_summary` captures the Signal.reasoning dict as a short text summary.

### New table: llm_analysis

```sql
CREATE TABLE IF NOT EXISTS llm_analysis (
    id              SERIAL PRIMARY KEY,
    instrument      TEXT NOT NULL,
    epoch           INTEGER NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reasoning       TEXT NOT NULL,
    config_changes  JSONB,
    code_patches    JSONB,
    config_applied  BOOLEAN NOT NULL DEFAULT FALSE,
    code_merged     BOOLEAN NOT NULL DEFAULT FALSE,
    backtest_improvement DOUBLE PRECISION,
    raw_output_path TEXT
);
```

## Components

### AnalysisBuilder (`src/optimization/llm_analysis_builder.py`)

Queries the DB and generates the full prompt:
1. Query `optimization_runs` for trial summary
2. Query `signal_log` for pattern breakdowns (grouped by trade_type × direction × trend)
3. Generate chart screenshots via mplfinance
4. Load prior analysis history from `reports/llm_analysis/`
5. Assemble into a single prompt text file

### LLMAnalyst (`src/optimization/llm_analyst.py`)

Executes the claude CLI call and parses the response:
1. Write assembled prompt to temp file
2. Run: `claude -p "$(cat prompt.txt)" --dangerously-skip-permissions`
3. Capture output to `reports/llm_analysis/{instrument}_epoch_{N}.txt`
4. Parse the structured sections (reasoning, config_changes, code_patches)
5. Store to `llm_analysis` table

### ConfigApplicator (`src/optimization/config_applicator.py`)

Applies config changes to YAML files:
1. Read current strategy.yaml / edges.yaml
2. Apply the JSON path → value overrides
3. Write back
4. Git commit the config change

### CodeSandbox (`src/optimization/code_sandbox.py`)

Tests code patches on a temp branch:
1. `git checkout -b llm-patch-{instrument}-{epoch}`
2. Apply each patch (search/replace in the target file)
3. Run backtest with the patched code
4. Compare results to the pre-patch baseline
5. If better → `git checkout main && git merge llm-patch-...`
6. If worse → `git checkout main && git branch -D llm-patch-...`

### Integration into AdaptiveRunner

After `optimize_instrument()` finishes the Optuna trials:

```python
# In adaptive_runner.py, after Optuna epoch completes:
if not instrument_is_proven:
    analysis = self._llm_analyst.analyze(instrument, epoch)
    if analysis.config_changes:
        self._config_applicator.apply(analysis.config_changes)
    if analysis.code_patches:
        improved = self._code_sandbox.test_patches(
            instrument, analysis.code_patches, data, config
        )
        if improved:
            self._code_sandbox.merge()
```

## Prompt Structure

The prompt sent to `claude -p` follows this template:

```
You are a quantitative trading strategy analyst. You have access to the
complete results of an optimization run on {instrument}.

Your job: analyze all the data below, find patterns, identify what's
working and what's broken, and produce specific actionable changes.

## Trial Summary
{layer_1_summary}

## Signal Pattern Analysis
{layer_2_patterns}

## Chart Analysis
[screenshots of best/worst trades and equity curve]

## Your Prior Analysis
{layer_4_prior_reasoning}

## Current Strategy Code
{relevant strategy source files}

---

Respond with EXACTLY three sections:

### REASONING
Your analysis of what's working, what's failing, and why.

### CONFIG_CHANGES
```json
{"config_changes": { "dotted.path": value, ... }}
```

### CODE_PATCHES
```json
{"code_patches": [{"file": "path", "description": "why", "search": "old code", "replace": "new code"}]}
```

Rules:
- Only suggest code patches you are confident will improve results
- Config changes are safe to apply — be aggressive with these
- Code patches will be tested on a branch — suggest bold structural fixes
- Reference specific signal patterns and trade types in your reasoning
- If a strategy is a net loser on this instrument, disable it via config
```

## File Structure

```
NEW:
  src/optimization/llm_analysis_builder.py  — Assemble prompt from DB + charts
  src/optimization/llm_analyst.py           — Execute claude -p, parse response
  src/optimization/config_applicator.py     — Apply config changes to YAML
  src/optimization/code_sandbox.py          — Temp branch, apply patches, test, merge/delete
  reports/llm_analysis/                     — Persistent output storage

MODIFIED:
  src/optimization/adaptive_runner.py       — Hook LLM analyst after Optuna epoch
  src/optimization/signal_persister.py      — Persist trade_type + reasoning_summary
  src/database/migrations/004_llm_analysis.sql — New columns + table
  src/backtesting/vectorbt_engine.py        — Pass trade_type from Signal to telemetry
```

## Cost

- ~4 claude CLI calls per epoch (one per instrument)
- Each call: ~30-60K tokens input (summary + patterns + screenshots + code)
- Output: ~2-5K tokens (reasoning + changes)
- Total per epoch: negligible on CLI subscription
- Chart generation: ~5 seconds per chart, ~20 charts per epoch

## Not In Scope

- Real-time signal-level LLM gating (that's mega-vision, separate system)
- Multi-turn conversation with the LLM (single prompt-response per instrument)
- LLM choosing which instruments to optimize (the loop handles all 4)
- Training custom models on the signal data (using Claude as-is)
