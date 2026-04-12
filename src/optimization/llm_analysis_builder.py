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
            # Pattern: strategy x trade_type x direction x trend
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
