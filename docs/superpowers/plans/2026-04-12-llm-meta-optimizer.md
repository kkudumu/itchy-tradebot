# LLM Meta-Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an LLM reasoning layer that runs after each Optuna epoch, analyzes all trial data + signal patterns + chart screenshots via `claude -p`, and produces config changes (applied directly) and code patches (tested on a temp git branch before merging).

**Architecture:** After Optuna finishes 50 trials for an instrument, the AnalysisBuilder queries the DB for trial summaries, signal log patterns, and generates chart screenshots. This is assembled into a prompt, sent to `claude -p --dangerously-skip-permissions`. The response is parsed into config changes and code patches. Config changes apply directly. Code patches go to a temp git branch, get backtested, and only merge if they improve results.

**Tech Stack:** Claude CLI (`claude -p`), PostgreSQL + pgvector (existing), mplfinance (chart generation), subprocess (CLI execution), git (branch sandbox).

---

## File Structure

```
NEW FILES:
  src/database/migrations/004_llm_meta_optimizer.sql  — Add columns to signal_log + llm_analysis table
  src/optimization/llm_analysis_builder.py            — Query DB, generate charts, assemble prompt
  src/optimization/llm_analyst.py                     — Execute claude -p, parse response
  src/optimization/config_applicator.py               — Apply dotted-path config changes to YAML
  src/optimization/code_sandbox.py                    — Git branch sandbox for code patches
  reports/llm_analysis/                               — Persistent output directory

MODIFIED FILES:
  src/optimization/signal_persister.py                — Add trade_type + reasoning_summary to inserts
  src/optimization/adaptive_runner.py                 — Hook LLM analyst after Optuna epoch
```

---

### Task 1: DB Migration — Add trade_type column + llm_analysis table

**Files:**
- Create: `src/database/migrations/004_llm_meta_optimizer.sql`

- [ ] **Step 1: Write the migration**

```sql
-- 004_llm_meta_optimizer.sql
-- Adds trade_type and reasoning_summary to signal_log,
-- creates llm_analysis table for LLM reasoning history.

ALTER TABLE signal_log ADD COLUMN IF NOT EXISTS trade_type TEXT;
ALTER TABLE signal_log ADD COLUMN IF NOT EXISTS reasoning_summary TEXT;

CREATE TABLE IF NOT EXISTS llm_analysis (
    id                    SERIAL PRIMARY KEY,
    instrument            TEXT NOT NULL,
    epoch                 INTEGER NOT NULL,
    timestamp             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reasoning             TEXT NOT NULL,
    config_changes        JSONB,
    code_patches          JSONB,
    config_applied        BOOLEAN NOT NULL DEFAULT FALSE,
    code_merged           BOOLEAN NOT NULL DEFAULT FALSE,
    backtest_improvement  DOUBLE PRECISION,
    raw_output_path       TEXT
);

CREATE INDEX IF NOT EXISTS idx_llm_analysis_instrument
    ON llm_analysis (instrument, epoch DESC);
```

- [ ] **Step 2: Apply migration**

Run:
```bash
python -c "
import psycopg2
conn = psycopg2.connect(host='localhost', port=5433, dbname='trading', user='postgres', password='postgres')
conn.autocommit = True
cur = conn.cursor()
with open('src/database/migrations/004_llm_meta_optimizer.sql') as f:
    cur.execute(f.read())
print('Migration applied')
conn.close()
"
```

- [ ] **Step 3: Commit**

```bash
git add src/database/migrations/004_llm_meta_optimizer.sql
git commit -m "feat(llm-meta-1): add trade_type column and llm_analysis table"
```

---

### Task 2: Signal Persister — Add trade_type and reasoning_summary

**Files:**
- Modify: `src/optimization/signal_persister.py`

- [ ] **Step 1: Update the row tuple construction to include trade_type and reasoning_summary**

In `signal_persister.py`, the `persist_signals` method builds row tuples. Add two new fields after `market_snapshot`:

The INSERT SQL becomes:
```sql
INSERT INTO signal_log (
    run_id, timestamp, strategy_name, direction,
    confluence_score, entry_price, stop_loss, take_profit,
    filtered_by, entered, trade_result_r, exit_reason,
    pnl_usd, market_snapshot, trade_type, reasoning_summary
) VALUES %s
```

Each row tuple appends:
```python
ev.get("trade_type", ""),
ev.get("reasoning_summary", ""),
```

- [ ] **Step 2: Commit**

```bash
git add src/optimization/signal_persister.py
git commit -m "feat(llm-meta-2): persist trade_type and reasoning_summary to signal_log"
```

---

### Task 3: Analysis Builder — Assemble LLM Prompt from DB + Charts

**Files:**
- Create: `src/optimization/llm_analysis_builder.py`

- [ ] **Step 1: Implement AnalysisBuilder**

```python
"""Assemble the LLM analysis prompt from DB queries and chart screenshots.

Queries optimization_runs and signal_log for the target instrument,
generates mplfinance charts for top wins/losses, loads prior analysis
history, and writes the complete prompt to a file.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_REPORTS_DIR = _PROJECT_ROOT / "reports" / "llm_analysis"


class AnalysisBuilder:
    """Build the structured prompt for the LLM meta-optimizer."""

    def __init__(self, db_pool) -> None:
        self._pool = db_pool

    def build_prompt(
        self,
        instrument: str,
        epoch: int,
        data_file: str | None = None,
    ) -> tuple[str, Path]:
        """Build full prompt and write to file. Returns (prompt_text, file_path)."""
        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        charts_dir = _REPORTS_DIR / "charts"
        charts_dir.mkdir(exist_ok=True)

        # Layer 1: Trial summary
        summary = self._query_trial_summary(instrument)

        # Layer 2: Signal pattern analysis
        patterns = self._query_signal_patterns(instrument)

        # Layer 3: Chart screenshots (paths — the prompt will reference them)
        chart_paths = self._generate_charts(instrument, charts_dir, data_file)

        # Layer 4: Prior analysis history
        prior = self._load_prior_analysis(instrument)

        # Layer 5: Relevant strategy source code
        strategy_code = self._load_strategy_source()

        # Assemble prompt
        prompt = self._assemble(instrument, epoch, summary, patterns,
                                chart_paths, prior, strategy_code)

        prompt_path = _REPORTS_DIR / f"{instrument}_epoch_{epoch}_prompt.md"
        prompt_path.write_text(prompt, encoding="utf-8")
        return prompt, prompt_path

    def _query_trial_summary(self, instrument: str) -> str:
        """Query optimization_runs for aggregate stats."""
        with self._pool.get_cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) as trials,
                       SUM(CASE WHEN passed_combine THEN 1 ELSE 0 END) as passes,
                       MAX((outcome->>'total_return_pct')::float) as best_return,
                       MIN((outcome->>'total_return_pct')::float) as worst_return,
                       AVG((outcome->>'total_trades')::float) as avg_trades,
                       AVG((outcome->>'win_rate')::float) as avg_wr
                FROM optimization_runs WHERE instrument = %s
            """, (instrument,))
            row = cur.fetchone()

        if not row or row["trials"] == 0:
            return "No trials found for this instrument."

        return (
            f"Total trials: {row['trials']}\n"
            f"Combine passes: {row['passes']} ({row['passes']/row['trials']*100:.1f}%)\n"
            f"Best return: {row['best_return']:.2f}%\n"
            f"Worst return: {row['worst_return']:.2f}%\n"
            f"Avg trades per trial: {row['avg_trades']:.0f}\n"
            f"Avg win rate: {row['avg_wr']*100:.1f}%\n"
        )

    def _query_signal_patterns(self, instrument: str) -> str:
        """Query signal_log for pattern breakdowns."""
        lines = []

        with self._pool.get_cursor() as cur:
            # Pattern: strategy × trade_type × direction × trend
            cur.execute("""
                SELECT s.strategy_name, s.trade_type, s.direction,
                       s.market_snapshot->>'htf_trend_1h' as trend_1h,
                       COUNT(*) FILTER (WHERE s.entered) as entered,
                       COUNT(*) FILTER (WHERE s.entered AND s.trade_result_r > 0) as wins,
                       AVG(s.trade_result_r) FILTER (WHERE s.entered) as avg_r,
                       COUNT(*) as total_signals
                FROM signal_log s
                JOIN optimization_runs r ON s.run_id = r.run_id
                WHERE r.instrument = %s
                GROUP BY s.strategy_name, s.trade_type, s.direction,
                         s.market_snapshot->>'htf_trend_1h'
                HAVING COUNT(*) FILTER (WHERE s.entered) >= 3
                ORDER BY COUNT(*) FILTER (WHERE s.entered) DESC
                LIMIT 30
            """, (instrument,))
            rows = cur.fetchall()

        if not rows:
            return "No signal patterns with enough data yet."

        lines.append("Strategy | TradeType | Dir | HTF Trend | Entered | Wins | WR | Avg R")
        lines.append("-" * 80)
        for r in rows:
            wr = r["wins"] / r["entered"] * 100 if r["entered"] > 0 else 0
            avg_r = r["avg_r"] or 0
            lines.append(
                f"{r['strategy_name']:12s} | {(r['trade_type'] or ''):16s} | "
                f"{r['direction']:5s} | {(r['trend_1h'] or '?'):8s} | "
                f"{r['entered']:4d} | {r['wins']:3d} | {wr:5.1f}% | {avg_r:+.2f}"
            )

        # Top rejection reasons
        with self._pool.get_cursor() as cur:
            cur.execute("""
                SELECT s.filtered_by, COUNT(*) as cnt
                FROM signal_log s
                JOIN optimization_runs r ON s.run_id = r.run_id
                WHERE r.instrument = %s AND s.filtered_by IS NOT NULL
                GROUP BY s.filtered_by
                ORDER BY cnt DESC LIMIT 5
            """, (instrument,))
            rejections = cur.fetchall()

        if rejections:
            lines.append("\nTop rejection reasons:")
            for r in rejections:
                lines.append(f"  {r['cnt']:6d}x {r['filtered_by']}")

        return "\n".join(lines)

    def _generate_charts(
        self, instrument: str, charts_dir: Path, data_file: str | None,
    ) -> list[str]:
        """Generate trade charts. Returns list of file paths."""
        # For now, return empty — charts will be added when mplfinance
        # integration is wired up. The prompt works without them.
        return []

    def _load_prior_analysis(self, instrument: str) -> str:
        """Load prior LLM analysis from the DB."""
        with self._pool.get_cursor() as cur:
            cur.execute("""
                SELECT epoch, reasoning, config_applied, code_merged
                FROM llm_analysis
                WHERE instrument = %s
                ORDER BY epoch DESC LIMIT 5
            """, (instrument,))
            rows = cur.fetchall()

        if not rows:
            return "No prior analysis history."

        lines = []
        for r in rows:
            applied = "config applied" if r["config_applied"] else "config skipped"
            merged = "code merged" if r["code_merged"] else "no code changes"
            # Truncate reasoning to first 200 chars
            summary = (r["reasoning"] or "")[:200].replace("\n", " ")
            lines.append(f"Epoch {r['epoch']}: [{applied}, {merged}] {summary}")

        return "\n".join(lines)

    def _load_strategy_source(self) -> str:
        """Load relevant strategy source files for the LLM to read."""
        files = [
            "src/strategy/strategies/sss/strategy.py",
            "src/strategy/strategies/ema_pullback.py",
            "src/strategy/strategies/asian_breakout.py",
            "src/strategy/strategies/ichimoku.py",
            "src/edges/trend_direction.py",
        ]
        parts = []
        for f in files:
            path = _PROJECT_ROOT / f
            if path.exists():
                content = path.read_text(encoding="utf-8")
                # Truncate large files to first 150 lines
                lines = content.split("\n")[:150]
                parts.append(f"### {f}\n```python\n" + "\n".join(lines) + "\n```\n")
        return "\n".join(parts)

    def _assemble(
        self,
        instrument: str,
        epoch: int,
        summary: str,
        patterns: str,
        chart_paths: list[str],
        prior: str,
        strategy_code: str,
    ) -> str:
        """Assemble the full prompt."""
        chart_section = ""
        if chart_paths:
            chart_section = "## Chart Screenshots\n" + "\n".join(
                f"![{Path(p).stem}]({p})" for p in chart_paths
            )
        else:
            chart_section = "## Charts\nNo chart screenshots available this epoch."

        return f"""You are a quantitative trading strategy analyst. You have access to the
complete results of an optimization run on {instrument}.

Your job: analyze all the data below, find patterns, identify what's
working and what's broken, and produce specific actionable changes.

## Trial Summary
{summary}

## Signal Pattern Analysis
{patterns}

{chart_section}

## Your Prior Analysis
{prior}

## Current Strategy Code
{strategy_code}

---

Respond with EXACTLY three sections:

### REASONING
Your analysis of what's working, what's failing, and why. Reference
specific signal patterns, trade types, and win rates. Be precise.

### CONFIG_CHANGES
```json
{{"config_changes": {{"dotted.path.to.param": value}}}}
```

Dotted paths target strategy.yaml keys. Examples:
- "strategies.sss.entry_mode": "fifty_tap"
- "strategies.ema_pullback.enabled": false
- "risk.initial_risk_pct": 0.3
- "strategies.ichimoku.adx.threshold": 12
- "edges.trend_direction.enabled": true

### CODE_PATCHES
```json
{{"code_patches": [{{"file": "src/path/to/file.py", "description": "why this change helps", "search": "exact old code to find", "replace": "new code to put there"}}]}}
```

Rules:
- Only suggest code patches you are confident will improve results
- Config changes are safe — be aggressive
- Code patches will be tested on a branch — suggest bold structural fixes
- Reference specific signal patterns and trade types in your reasoning
- If a strategy is a net loser on this instrument, disable it via config
- If stops are too tight or wide, fix the code that computes them
- If a pattern has >60% WR, suggest increasing its weight or priority
- If a pattern has <20% WR, suggest filtering it out
"""
```

- [ ] **Step 2: Commit**

```bash
git add src/optimization/llm_analysis_builder.py
git commit -m "feat(llm-meta-3): add analysis builder — assembles LLM prompt from DB + code"
```

---

### Task 4: LLM Analyst — Execute claude -p and Parse Response

**Files:**
- Create: `src/optimization/llm_analyst.py`

- [ ] **Step 1: Implement LLMAnalyst**

```python
"""Execute claude -p with the assembled prompt and parse the structured response."""
from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_REPORTS_DIR = _PROJECT_ROOT / "reports" / "llm_analysis"


@dataclass
class AnalysisResult:
    """Parsed output from the LLM analyst."""
    reasoning: str = ""
    config_changes: dict[str, Any] = field(default_factory=dict)
    code_patches: list[dict[str, str]] = field(default_factory=list)
    raw_output: str = ""
    output_path: str = ""


class LLMAnalyst:
    """Run claude -p on the assembled prompt and parse the response."""

    def __init__(self, db_pool) -> None:
        self._pool = db_pool

    def analyze(
        self,
        instrument: str,
        epoch: int,
        prompt: str,
    ) -> AnalysisResult:
        """Send prompt to claude -p, parse response, persist to DB."""
        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        output_path = _REPORTS_DIR / f"{instrument}_epoch_{epoch}.txt"

        # Execute claude -p
        logger.info("Calling claude -p for %s epoch %d...", instrument, epoch)
        raw_output = self._call_claude(prompt)

        # Save raw output
        output_path.write_text(raw_output, encoding="utf-8")
        logger.info("LLM output saved to %s (%d chars)", output_path, len(raw_output))

        # Parse structured response
        result = self._parse_response(raw_output)
        result.raw_output = raw_output
        result.output_path = str(output_path)

        # Persist to DB
        self._persist(instrument, epoch, result)

        return result

    def _call_claude(self, prompt: str) -> str:
        """Execute claude -p with the prompt."""
        try:
            proc = subprocess.run(
                ["claude", "-p", prompt, "--dangerously-skip-permissions"],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=str(_PROJECT_ROOT),
            )
            if proc.returncode != 0:
                logger.warning("claude -p returned code %d: %s", proc.returncode, proc.stderr[:500])
            return proc.stdout or proc.stderr or ""
        except FileNotFoundError:
            logger.error("claude CLI not found — is it installed and on PATH?")
            return ""
        except subprocess.TimeoutExpired:
            logger.error("claude -p timed out after 300s")
            return ""
        except Exception as exc:
            logger.error("claude -p failed: %s", exc)
            return ""

    def _parse_response(self, raw: str) -> AnalysisResult:
        """Parse the three sections from the LLM output."""
        result = AnalysisResult()

        # Extract REASONING section
        reasoning_match = re.search(
            r"###?\s*REASONING\s*\n(.*?)(?=###?\s*CONFIG_CHANGES|$)",
            raw, re.DOTALL | re.IGNORECASE,
        )
        if reasoning_match:
            result.reasoning = reasoning_match.group(1).strip()

        # Extract CONFIG_CHANGES JSON
        config_match = re.search(
            r"```json\s*\n\s*\{[^`]*?\"config_changes\"[^`]*?\}\s*\n\s*```",
            raw, re.DOTALL,
        )
        if config_match:
            try:
                parsed = json.loads(config_match.group().strip().strip("`").strip("json").strip())
                result.config_changes = parsed.get("config_changes", {})
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse config_changes JSON: %s", exc)

        # Extract CODE_PATCHES JSON
        patches_match = re.search(
            r"```json\s*\n\s*\{[^`]*?\"code_patches\"[^`]*?\}\s*\n\s*```",
            raw, re.DOTALL,
        )
        if patches_match:
            try:
                parsed = json.loads(patches_match.group().strip().strip("`").strip("json").strip())
                result.code_patches = parsed.get("code_patches", [])
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse code_patches JSON: %s", exc)

        return result

    def _persist(self, instrument: str, epoch: int, result: AnalysisResult) -> None:
        """Save analysis to llm_analysis table."""
        try:
            with self._pool.get_cursor() as cur:
                cur.execute("""
                    INSERT INTO llm_analysis (
                        instrument, epoch, reasoning, config_changes,
                        code_patches, raw_output_path
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    instrument, epoch, result.reasoning,
                    json.dumps(result.config_changes),
                    json.dumps(result.code_patches),
                    result.output_path,
                ))
        except Exception as exc:
            logger.warning("Failed to persist LLM analysis: %s", exc)
```

- [ ] **Step 2: Commit**

```bash
git add src/optimization/llm_analyst.py
git commit -m "feat(llm-meta-4): add LLM analyst — execute claude -p and parse response"
```

---

### Task 5: Config Applicator — Apply Dotted-Path Changes to YAML

**Files:**
- Create: `src/optimization/config_applicator.py`

- [ ] **Step 1: Implement ConfigApplicator**

```python
"""Apply dotted-path config changes to YAML files."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class ConfigApplicator:
    """Apply LLM-suggested config changes to strategy.yaml and edges.yaml."""

    def apply(self, changes: dict[str, object]) -> list[str]:
        """Apply dotted-path changes. Returns list of changes applied."""
        if not changes:
            return []

        strategy_path = _PROJECT_ROOT / "config" / "strategy.yaml"
        edges_path = _PROJECT_ROOT / "config" / "edges.yaml"

        applied = []

        # Load both files
        with strategy_path.open() as f:
            strategy = yaml.safe_load(f) or {}
        with edges_path.open() as f:
            edges = yaml.safe_load(f) or {}

        for dotted_key, value in changes.items():
            parts = dotted_key.split(".")

            # Route to the right file
            if parts[0] == "edges":
                target = edges
                parts = parts[1:]  # strip "edges." prefix
            else:
                target = strategy

            # Navigate to the parent, create intermediate dicts if needed
            node = target
            for part in parts[:-1]:
                if part not in node or not isinstance(node[part], dict):
                    node[part] = {}
                node = node[part]

            old_value = node.get(parts[-1])
            node[parts[-1]] = value
            applied.append(f"{dotted_key}: {old_value} -> {value}")
            logger.info("Config: %s = %s (was %s)", dotted_key, value, old_value)

        # Write back
        with strategy_path.open("w") as f:
            yaml.dump(strategy, f, default_flow_style=False, sort_keys=False)
        with edges_path.open("w") as f:
            yaml.dump(edges, f, default_flow_style=False, sort_keys=False)

        # Git commit
        if applied:
            try:
                subprocess.run(
                    ["git", "add", str(strategy_path), str(edges_path)],
                    cwd=str(_PROJECT_ROOT), capture_output=True,
                )
                msg = "feat(llm-meta): config changes\n\n" + "\n".join(applied)
                subprocess.run(
                    ["git", "commit", "-m", msg],
                    cwd=str(_PROJECT_ROOT), capture_output=True,
                )
            except Exception as exc:
                logger.warning("Git commit failed: %s", exc)

        return applied
```

- [ ] **Step 2: Commit**

```bash
git add src/optimization/config_applicator.py
git commit -m "feat(llm-meta-5): add config applicator — apply dotted-path YAML changes"
```

---

### Task 6: Code Sandbox — Git Branch Patch Testing

**Files:**
- Create: `src/optimization/code_sandbox.py`

- [ ] **Step 1: Implement CodeSandbox**

```python
"""Git branch sandbox for testing LLM code patches."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class CodeSandbox:
    """Test code patches on a temporary git branch."""

    def __init__(self) -> None:
        self._branch_name: str | None = None

    def test_patches(
        self,
        instrument: str,
        epoch: int,
        patches: list[dict[str, str]],
        backtest_fn: Callable,
        data: Any,
        config: dict,
        baseline_return: float,
    ) -> bool:
        """Apply patches on a temp branch, backtest, return True if improved."""
        if not patches:
            return False

        branch = f"llm-patch-{instrument.lower()}-epoch{epoch}"
        self._branch_name = branch

        try:
            # Create temp branch
            self._git("checkout", "-b", branch)

            # Apply patches
            applied = 0
            for patch in patches:
                if self._apply_patch(patch):
                    applied += 1
                else:
                    logger.warning("Patch failed: %s", patch.get("description", "unknown"))

            if applied == 0:
                logger.info("No patches applied successfully — aborting")
                self._cleanup(branch)
                return False

            # Commit patches
            self._git("add", "-A")
            desc = "; ".join(p.get("description", "LLM patch") for p in patches)
            self._git("commit", "-m", f"llm-meta: {desc}")

            # Run backtest on patched code
            logger.info("Running backtest on patched branch %s...", branch)
            result = backtest_fn(data, config, instrument, 50_000.0)

            if result is None:
                logger.warning("Patched backtest failed — reverting")
                self._cleanup(branch)
                return False

            patched_return = result.get("total_return_pct", -999)
            logger.info(
                "Patch result: %.2f%% return (baseline: %.2f%%)",
                patched_return, baseline_return,
            )

            if patched_return > baseline_return:
                # Merge to main
                self._git("checkout", "main")
                self._git("merge", branch, "--no-ff", "-m",
                          f"Merge LLM patch for {instrument} epoch {epoch}: "
                          f"{patched_return:.2f}% > {baseline_return:.2f}%")
                self._git("branch", "-d", branch)
                logger.info("Patch MERGED — improved %.2f%% → %.2f%%",
                           baseline_return, patched_return)
                return True
            else:
                logger.info("Patch did NOT improve — discarding")
                self._cleanup(branch)
                return False

        except Exception as exc:
            logger.error("Code sandbox error: %s", exc)
            self._cleanup(branch)
            return False

    def _apply_patch(self, patch: dict[str, str]) -> bool:
        """Apply a search/replace patch to a file."""
        filepath = _PROJECT_ROOT / patch.get("file", "")
        if not filepath.exists():
            logger.warning("Patch target not found: %s", filepath)
            return False

        search = patch.get("search", "")
        replace = patch.get("replace", "")
        if not search or not replace:
            return False

        content = filepath.read_text(encoding="utf-8")
        if search not in content:
            logger.warning("Search string not found in %s", filepath)
            return False

        new_content = content.replace(search, replace, 1)
        filepath.write_text(new_content, encoding="utf-8")
        logger.info("Patched %s: %s", filepath.name, patch.get("description", ""))
        return True

    def _cleanup(self, branch: str) -> None:
        """Return to main and delete the temp branch."""
        try:
            self._git("checkout", "main")
            self._git("branch", "-D", branch)
        except Exception:
            pass
        self._branch_name = None

    @staticmethod
    def _git(*args: str) -> str:
        """Run a git command in the project root."""
        result = subprocess.run(
            ["git"] + list(args),
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr}")
        return result.stdout
```

- [ ] **Step 2: Commit**

```bash
git add src/optimization/code_sandbox.py
git commit -m "feat(llm-meta-6): add code sandbox — git branch testing for LLM patches"
```

---

### Task 7: Hook LLM Analyst into AdaptiveRunner

**Files:**
- Modify: `src/optimization/adaptive_runner.py`

- [ ] **Step 1: Add LLM analyst imports and initialization**

In `adaptive_runner.py`, add to the imports at the top:
```python
from src.optimization.llm_analysis_builder import AnalysisBuilder
from src.optimization.llm_analyst import LLMAnalyst
from src.optimization.config_applicator import ConfigApplicator
from src.optimization.code_sandbox import CodeSandbox
```

In `__init__`, add:
```python
self._analysis_builder = AnalysisBuilder(db_pool=db_pool)
self._llm_analyst = LLMAnalyst(db_pool=db_pool)
self._config_applicator = ConfigApplicator()
self._code_sandbox = CodeSandbox()
```

- [ ] **Step 2: Add LLM analysis call after Optuna epoch**

At the end of `optimize_instrument()`, after the guardrail checks (whether passed or not), add:

```python
# LLM Meta-Optimizer: analyze results and suggest improvements
try:
    prompt, prompt_path = self._analysis_builder.build_prompt(
        instrument=sym, epoch=self._epoch, data_file=instrument.get("data_file"),
    )
    analysis = self._llm_analyst.analyze(sym, self._epoch, prompt)

    if analysis.config_changes:
        applied = self._config_applicator.apply(analysis.config_changes)
        logger.info("LLM config changes applied: %s", applied)

    if analysis.code_patches:
        best_return = best_result.get("total_return_pct", 0) if best_result else 0
        improved = self._code_sandbox.test_patches(
            instrument=sym, epoch=self._epoch,
            patches=analysis.code_patches,
            backtest_fn=self._backtest_fn,
            data=data, config=base_config,
            baseline_return=best_return,
        )
        if improved:
            logger.info("LLM code patch MERGED for %s", sym)

except Exception as exc:
    logger.warning("LLM meta-optimizer failed for %s: %s", sym, exc)
```

- [ ] **Step 3: Commit**

```bash
git add src/optimization/adaptive_runner.py
git commit -m "feat(llm-meta-7): hook LLM meta-optimizer into adaptive runner"
```

---

### Task 8: Smoke Test — Run One LLM Analysis

- [ ] **Step 1: Run the LLM analyst on MGC manually**

```bash
python -c "
import sys; sys.path.insert(0, '.')
import os
for line in open('.env').read().splitlines():
    line = line.strip()
    if line and not line.startswith('#') and '=' in line:
        k, v = line.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip())

from src.database.connection import DatabasePool
from src.optimization.llm_analysis_builder import AnalysisBuilder
from src.optimization.llm_analyst import LLMAnalyst

pool = DatabasePool()
pool.initialise()

builder = AnalysisBuilder(db_pool=pool)
prompt, path = builder.build_prompt('MGC', epoch=99)
print(f'Prompt written to {path} ({len(prompt)} chars)')
print('First 500 chars:')
print(prompt[:500])

analyst = LLMAnalyst(db_pool=pool)
result = analyst.analyze('MGC', 99, prompt)
print(f'\\nReasoning: {result.reasoning[:300]}...')
print(f'Config changes: {result.config_changes}')
print(f'Code patches: {len(result.code_patches)}')
print(f'Output: {result.output_path}')

pool.close()
"
```

Expected: Claude CLI runs, produces reasoning + config changes + possibly code patches. Output saved to `reports/llm_analysis/MGC_epoch_99.txt`.

- [ ] **Step 2: Verify output file exists**

```bash
ls -la reports/llm_analysis/MGC_epoch_99.txt
cat reports/llm_analysis/MGC_epoch_99.txt | head -50
```

- [ ] **Step 3: Commit any fixes**

```bash
git add -A
git commit -m "test(llm-meta-8): smoke test LLM meta-optimizer on MGC"
```
