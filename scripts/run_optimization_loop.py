"""
Agentic Self-Optimization Loop orchestrator.

Runs a backtest → analyze → adjust → re-run loop by spawning the Claude CLI
via subprocess.  Each iteration:

1. Loads the current config (strategy.yaml + edges.yaml).
2. Slices historical data to the last N optimization years.
3. Runs IchimokuBacktester (or StrategyBacktester when available).
4. Exports a structured JSON report via ResultsExporter.
5. Persists trades to pgvector via TradePersistence (optional).
6. Checks stopping conditions (target met, plateau, max iterations).
7. Builds a structured prompt for Claude CLI and spawns it.
8. Detects config changes via git diff.
9. Re-runs the backtest with the new config.
10. Reverts config if metrics worsened; otherwise keeps the change.
11. Appends a one-liner learning to CLAUDE.md.

Stopping conditions
-------------------
- Challenge pass rate meets or exceeds ``target_pass_rate``
- Metrics plateau: 3 consecutive iterations with < 5% pass-rate improvement
- Maximum iterations reached

CLI usage
---------
    python scripts/run_optimization_loop.py \\
        --data-file data/xauusd_1m.parquet \\
        --max-iterations 10 \\
        --target-pass-rate 0.50

Config is read from ``config/optimization_loop.yaml``; CLI flags override it.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Forward-compatible backtester import
# ---------------------------------------------------------------------------
try:
    from src.backtesting.strategy_backtester import StrategyBacktester as _BacktesterClass
except ImportError:
    from src.backtesting.vectorbt_engine import IchimokuBacktester as _BacktesterClass  # type: ignore[assignment]

from src.backtesting.results_exporter import ResultsExporter
from src.backtesting.trade_persistence import TradePersistence
from src.learning.claude_md_writer import ClaudeMdWriter

logger = logging.getLogger(__name__)

# Repo root — scripts live one level below the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent

_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "optimization_loop.yaml"
_STRATEGY_CONFIG_PATH = _REPO_ROOT / "config" / "strategy.yaml"
_EDGES_CONFIG_PATH = _REPO_ROOT / "config" / "edges.yaml"
_CLAUDE_MD_PATH = _REPO_ROOT / "CLAUDE.md"


# ---------------------------------------------------------------------------
# OptimizationLoop
# ---------------------------------------------------------------------------


class OptimizationLoop:
    """Orchestrate the agentic backtest-optimize loop.

    Parameters
    ----------
    config:
        Merged configuration dict, typically loaded from
        ``config/optimization_loop.yaml`` plus CLI overrides.
    data_file:
        Path to the historical data parquet file.
    reports_dir:
        Directory for JSON run reports.  Overrides ``config.persistence.reports_dir``.
    claude_md_path:
        Path to CLAUDE.md for learning persistence.
    db_pool:
        Optional database pool.  When ``None``, trade persistence is skipped.
    embedding_engine:
        Optional embedding engine.  Used by TradePersistence.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        data_file: Optional[str] = None,
        reports_dir: Optional[str] = None,
        claude_md_path: Optional[str] = None,
        db_pool: Optional[Any] = None,
        embedding_engine: Optional[Any] = None,
    ) -> None:
        self._config = config
        opt = config.get("optimization", {})
        pers = config.get("persistence", {})
        self._max_iterations: int = int(opt.get("max_iterations", 10))
        self._target_pass_rate: float = float(opt.get("target_pass_rate", 0.50))
        self._optimization_years: int = int(opt.get("optimization_years", 2))
        self._plateau_threshold: float = float(opt.get("plateau_threshold", 0.05))
        self._plateau_iterations: int = int(opt.get("plateau_iterations", 3))
        self._max_changes: int = int(opt.get("max_changes_per_iteration", 2))

        self._data_file = Path(data_file) if data_file else None
        resolved_reports = reports_dir or pers.get("reports_dir", "reports")
        self._exporter = ResultsExporter(reports_dir=str(resolved_reports))
        self._persistence_enabled: bool = bool(pers.get("enabled", True))

        resolved_claude_md = claude_md_path or str(_CLAUDE_MD_PATH)
        self._claude_writer = ClaudeMdWriter(claude_md_path=resolved_claude_md)

        self._trade_persistence = TradePersistence(
            db_pool=db_pool,
            embedding_engine=embedding_engine,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Execute the full optimization loop.

        Returns
        -------
        dict
            Summary containing ``best_config``, ``best_metrics``,
            ``iteration_history``, ``stop_reason``, and
            ``final_validation_metrics`` (when full-dataset validation runs).
        """
        if self._data_file is None or not self._data_file.exists():
            raise FileNotFoundError(
                f"Data file not found: {self._data_file}. "
                "Pass --data-file or set data_file in config."
            )

        logger.info("Loading data from %s", self._data_file)
        full_df = pd.read_parquet(self._data_file)

        opt_df = self._slice_data(full_df, years=self._optimization_years)
        logger.info(
            "Optimization window: %s → %s (%d rows)",
            opt_df.index.min(),
            opt_df.index.max(),
            len(opt_df),
        )

        # Start unified tabbed dashboard (port 8501)
        # Tab 1: Live Trading (equity curve, trades)
        # Tab 2: Optimization (iterations, pass rates, Claude's reasoning)
        _dashboard = None
        self._live_dashboard = None
        try:
            from src.backtesting.live_dashboard import LiveDashboardServer
            from src.backtesting.optimization_dashboard import OptimizationDashboardServer

            # Create the live dashboard (manages state + candle buffer)
            self._live_dashboard = LiveDashboardServer(port=0, auto_open=False)
            # Don't start its HTTP server — we just use it as a state container

            # Create the unified dashboard server that proxies live state
            resolved_reports = self._config.get("persistence", {}).get("reports_dir", "reports")
            _dashboard = OptimizationDashboardServer(
                port=8501, reports_dir=str(resolved_reports), auto_open=True,
                live_dashboard=self._live_dashboard,
            )
            _dashboard.start()
            logger.info("Dashboard at http://localhost:8501 (Live Trading + Optimization tabs)")
        except Exception as exc:
            logger.warning("Could not start dashboard: %s", exc)

        history: List[Dict[str, Any]] = []
        best_metrics: Dict[str, Any] = {}
        best_config: Dict[str, Any] = {}
        stop_reason: str = "max_iterations"

        for iteration in range(1, self._max_iterations + 1):
            logger.info("=== Iteration %d / %d ===", iteration, self._max_iterations)

            run_id = f"opt_iter_{iteration:03d}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            config_snapshot = self._load_config()

            # ---- Run backtest ------------------------------------------------
            result = self._run_backtest(opt_df, config_snapshot)
            metrics = result.metrics if hasattr(result, "metrics") else result.get("metrics", {})
            current_sharpe = float(metrics.get("sharpe", 0.0))

            # Get challenge simulation from result
            challenge_sim = None
            if hasattr(result, "challenge_simulation"):
                challenge_sim = result.challenge_simulation
            elif isinstance(result, dict):
                challenge_sim = result.get("challenge_simulation")

            current_pass_rate = challenge_sim.pass_rate if challenge_sim else 0.0

            logger.info(
                "Iteration %d: PassRate=%.1f%%  Sharpe=%.3f  WinRate=%.1f%%  Trades=%d",
                iteration,
                current_pass_rate * 100,
                current_sharpe,
                float(metrics.get("win_rate", 0.0)) * 100,
                int(metrics.get("total_trades", metrics.get("trade_count", 0))),
            )

            # ---- Export report -----------------------------------------------
            previous_runs = self._exporter.list_runs()
            report_path = self._exporter.export_run_report(
                result=result,
                config=config_snapshot,
                run_id=run_id,
                previous_runs=[self._exporter.load_run_report(r["run_id"]) for r in previous_runs[-3:]],
            )
            logger.info("Report written: %s", report_path)

            # ---- Persist trades (optional) -----------------------------------
            if self._persistence_enabled:
                trades = result.trades if hasattr(result, "trades") else result.get("trades", [])
                self._trade_persistence.persist_run(
                    run_id=run_id,
                    trades=trades,
                    config_snapshot=config_snapshot,
                    metrics=metrics,
                )

            # ---- Update best -------------------------------------------------
            if not best_metrics or current_pass_rate > float(best_metrics.get("pass_rate", float("-inf"))):
                best_metrics = dict(metrics)
                best_metrics["pass_rate"] = current_pass_rate
                best_config = dict(config_snapshot)

            entry = {
                "iteration": iteration,
                "run_id": run_id,
                "metrics": dict(metrics),
                "sharpe": current_sharpe,
                "pass_rate": current_pass_rate,
            }
            history.append(entry)

            # ---- Check target -----------------------------------------------
            if current_pass_rate >= self._target_pass_rate:
                stop_reason = "target_reached"
                logger.info(
                    "Target pass rate %.0f%% reached (current=%.1f%%). Stopping.",
                    self._target_pass_rate * 100,
                    current_pass_rate * 100,
                )
                break

            # ---- Check plateau -----------------------------------------------
            if self._check_plateau(history):
                stop_reason = "plateau"
                logger.info("Plateau detected after %d iterations. Stopping.", iteration)
                break

            # ---- Last iteration — no point spawning Claude ------------------
            if iteration == self._max_iterations:
                break

            # ---- Build Claude prompt ----------------------------------------
            current_report = self._exporter.load_run_report(run_id)
            previous_report: Optional[Dict[str, Any]] = None
            if len(history) >= 2:
                prev_run_id = history[-2]["run_id"]
                try:
                    previous_report = self._exporter.load_run_report(prev_run_id)
                except FileNotFoundError:
                    pass

            similar_configs = self._trade_persistence.get_similar_past_configs(
                config_embedding=self._dummy_embedding(),
                top_k=5,
            )
            learnings_summary = self._claude_writer.get_learnings_summary()

            prompt = self._build_claude_prompt(
                report=current_report,
                previous_report=previous_report,
                similar_configs=similar_configs,
                learnings=learnings_summary,
            )

            # ---- Spawn Claude CLI -------------------------------------------
            logger.info("Spawning Claude CLI for iteration %d …", iteration)
            claude_output = self._spawn_claude(prompt)
            logger.info("Claude output (%d chars): %s", len(claude_output), claude_output[:200] if claude_output else "(empty)")

            # Save Claude's full reasoning to the iteration report
            _reasoning_path = Path(self._exporter._reports_dir) / f"{run_id}_claude_reasoning.txt"
            try:
                _reasoning_path.write_text(claude_output or "(no output)", encoding="utf-8")
            except OSError:
                pass
            # Also append to the JSON report
            try:
                _report_json_path = Path(self._exporter._reports_dir) / f"{run_id}.json"
                if _report_json_path.exists():
                    _rdata = json.loads(_report_json_path.read_text(encoding="utf-8"))
                    _rdata["claude_reasoning"] = claude_output or "(no output)"
                    _rdata["claude_prompt"] = prompt[:5000]  # First 5K of prompt for context
                    _report_json_path.write_text(json.dumps(_rdata, indent=2, default=str), encoding="utf-8")
            except (OSError, json.JSONDecodeError):
                pass

            # ---- Detect config changes via git diff -------------------------
            changes_diff = self._detect_changes()
            if not changes_diff.strip():
                logger.info("Claude made no config changes in iteration %d.", iteration)
                self._claude_writer.append_learning(
                    run_id=run_id,
                    changes_made="(no changes)",
                    metrics_before=dict(metrics),
                    metrics_after=dict(metrics),
                    verdict="kept",
                )
                continue

            logger.info("Detected config changes:\n%s", changes_diff[:800])

            # ---- Re-run backtest with new config ----------------------------
            new_config_snapshot = self._load_config()
            new_result = self._run_backtest(opt_df, new_config_snapshot)
            new_metrics = (
                new_result.metrics if hasattr(new_result, "metrics") else new_result.get("metrics", {})
            )
            new_sharpe = float(new_metrics.get("sharpe", 0.0))

            # Get challenge simulation from new result
            new_challenge_sim = None
            if hasattr(new_result, "challenge_simulation"):
                new_challenge_sim = new_result.challenge_simulation
            elif isinstance(new_result, dict):
                new_challenge_sim = new_result.get("challenge_simulation")
            new_pass_rate = new_challenge_sim.pass_rate if new_challenge_sim else 0.0

            # ---- Decide keep / revert ---------------------------------------
            if new_pass_rate >= current_pass_rate:
                verdict = "kept"
                logger.info(
                    "Config improved: PassRate %.1f%% → %.1f%% (Sharpe %.3f → %.3f). Keeping changes.",
                    current_pass_rate * 100,
                    new_pass_rate * 100,
                    current_sharpe,
                    new_sharpe,
                )
                # Update best
                if new_pass_rate > float(best_metrics.get("pass_rate", float("-inf"))):
                    best_metrics = dict(new_metrics)
                    best_metrics["pass_rate"] = new_pass_rate
                    best_config = dict(new_config_snapshot)
                # Update current iteration with new metrics
                history[-1]["metrics"] = dict(new_metrics)
                history[-1]["sharpe"] = new_sharpe
                history[-1]["pass_rate"] = new_pass_rate
            else:
                verdict = "reverted"
                logger.info(
                    "Config worsened: PassRate %.1f%% → %.1f%%. Reverting.",
                    current_pass_rate * 100,
                    new_pass_rate * 100,
                )
                self._revert_changes()

            # ---- Append learning to CLAUDE.md --------------------------------
            self._claude_writer.append_learning(
                run_id=run_id,
                changes_made=self._summarise_diff(changes_diff),
                metrics_before={
                    "pass_rate": current_pass_rate,
                    "sharpe": metrics.get("sharpe"),
                    "win_rate": metrics.get("win_rate"),
                    "max_dd": metrics.get("max_drawdown"),
                },
                metrics_after={
                    "pass_rate": new_pass_rate,
                    "sharpe": new_metrics.get("sharpe"),
                    "win_rate": new_metrics.get("win_rate"),
                    "max_dd": new_metrics.get("max_drawdown"),
                },
                verdict=verdict,
            )

        # ---- Final validation on full dataset --------------------------------
        final_validation_metrics: Optional[Dict[str, Any]] = None
        run_full = self._config.get("validation", {}).get("run_full_dataset", True)
        if run_full and best_config:
            logger.info("Running final validation on full dataset …")
            best_backtester = _BacktesterClass(config=best_config)
            final_result = best_backtester.run(full_df)
            final_validation_metrics = (
                final_result.metrics
                if hasattr(final_result, "metrics")
                else final_result.get("metrics", {})
            )
            final_challenge_sim = None
            if hasattr(final_result, "challenge_simulation"):
                final_challenge_sim = final_result.challenge_simulation
            final_pass_rate = final_challenge_sim.pass_rate if final_challenge_sim else 0.0
            logger.info(
                "Final validation: PassRate=%.1f%%  Sharpe=%.3f  WinRate=%.1f%%",
                final_pass_rate * 100,
                float(final_validation_metrics.get("sharpe", 0.0)),
                float(final_validation_metrics.get("win_rate", 0.0)) * 100,
            )

        summary = {
            "best_config": best_config,
            "best_metrics": best_metrics,
            "iteration_history": history,
            "stop_reason": stop_reason,
            "total_iterations": len(history),
            "final_validation_metrics": final_validation_metrics,
            "learnings_summary": self._claude_writer.get_learnings_summary(),
        }
        logger.info(
            "Optimization complete. Stop reason: %s. Best PassRate: %.1f%%  Best Sharpe: %.3f",
            stop_reason,
            float(best_metrics.get("pass_rate", 0.0)) * 100,
            float(best_metrics.get("sharpe", 0.0)),
        )
        # Stop dashboard
        if _dashboard is not None:
            try:
                _dashboard.stop()
            except Exception:
                pass

        return summary

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_backtest(self, df: pd.DataFrame, config: Dict[str, Any]) -> Any:
        """Instantiate the backtester and run it on *df*."""
        backtester = _BacktesterClass(config=config)

        # Start live trade dashboard if not already running
        live_dash = None
        if self._live_dashboard is not None:
            live_dash = self._live_dashboard

        return backtester.run(df, live_dashboard=live_dash)

    def _build_claude_prompt(
        self,
        report: Dict[str, Any],
        previous_report: Optional[Dict[str, Any]],
        similar_configs: List[Any],
        learnings: str,
    ) -> str:
        """Build the structured prompt sent to Claude CLI.

        The prompt includes: challenge simulation results, failure breakdown,
        secondary metrics, similar past configs from pgvector, accumulated
        learnings, and a strict constraint to make at most
        ``max_changes_per_iteration`` parameter changes.
        """
        run_id = report.get("run_id", "unknown")
        metrics = report.get("metrics", {})
        config_snap = report.get("config_snapshot", {})
        challenge_data = report.get("challenge_simulation")

        lines: List[str] = [
            "You are a quantitative trading strategy optimizer for a The5ers 2-Step prop firm challenge.",
            "",
            f"## Current Run: {run_id}",
        ]

        # Challenge simulation results
        if challenge_data:
            lines += [
                "### Challenge Simulation Results",
                f"- Pass Rate:              {challenge_data.get('pass_rate', 0):.1%}",
                f"- Phase 1 Passes:         {challenge_data.get('phase_1_pass_count', 0)}/{challenge_data.get('total_windows', 0)}",
                f"- Phase 2 Passes:         {challenge_data.get('full_pass_count', 0)}/{challenge_data.get('total_windows', 0)}",
                f"- Avg Days Phase 1:       {challenge_data.get('avg_days_phase_1', 0):.0f}",
                f"- Avg Days Phase 2:       {challenge_data.get('avg_days_phase_2', 0):.0f}",
                "",
                "### Failure Breakdown",
            ]
            failure_breakdown = challenge_data.get("failure_breakdown", {})
            if failure_breakdown:
                for reason, count in failure_breakdown.items():
                    lines.append(f"- {reason}: {count}")
                biggest_fail = max(failure_breakdown, key=failure_breakdown.get)
                lines.append("")
                lines.append(f"**BIGGEST FAILURE MODE: {biggest_fail}** -- prioritize fixing this.")
            lines.append("")
        else:
            lines += [
                "### Challenge Simulation Results",
                "No challenge simulation data available (backtest may not have produced it).",
                "",
            ]

        # Secondary metrics
        lines += [
            "### Secondary Metrics",
            f"- Sharpe Ratio:   {metrics.get('sharpe', 'N/A')}",
            f"- Win Rate:       {metrics.get('win_rate', 'N/A')}",
            f"- Trade Count:    {metrics.get('trade_count', 'N/A')}",
            f"- Max Drawdown:   {metrics.get('max_drawdown', 'N/A')}",
            f"- Profit Factor:  {metrics.get('profit_factor', 'N/A')}",
            "",
        ]

        # Include trade-level detail so Claude can see WHY trades lose
        best_trades = report.get("best_trades", [])
        worst_trades = report.get("worst_trades", [])
        if worst_trades or best_trades:
            lines += ["### Trade Details (worst first)"]
            for t in (worst_trades + best_trades)[:10]:
                lines.append(
                    f"- {t.get('direction', '?')} @ {t.get('entry_price', 0):.2f} -> "
                    f"{t.get('exit_price', 0):.2f}  R={t.get('r_multiple', 0):.2f}  "
                    f"reason: {t.get('reason', '?')}"
                )
            lines.append("")

        # Similar past configs
        if similar_configs:
            lines += ["## Similar Past Configurations (from pgvector)"]
            for sc in similar_configs:
                d = sc.to_dict() if hasattr(sc, "to_dict") else sc
                lines.append(
                    f"- Run {d.get('run_id')}: "
                    f"Similarity={d.get('similarity', 0.0):.3f}  "
                    f"WinRate={d.get('win_rate', 0.0):.1%}  "
                    f"AvgR={d.get('avg_r', 0.0):.2f}  "
                    f"Trades={d.get('trade_count', 0)}"
                )
            lines.append("")
        else:
            lines += ["## Similar Past Configurations", "None available yet.", ""]

        # Accumulated learnings
        lines += [
            "## Accumulated Strategy Learnings",
            learnings,
            "",
        ]

        # Current config snapshot
        config_yaml = yaml.dump(config_snap, default_flow_style=False, allow_unicode=True)
        lines += [
            "## Current Config Snapshot",
            "```yaml",
            config_yaml.strip(),
            "```",
            "",
        ]

        # Instructions
        lines += [
            "## Your Task",
            f"Improve the challenge pass rate. Target: >= {self._target_pass_rate:.0%}",
            "",
            "ANALYSIS REQUIRED (write this BEFORE making changes):",
            "1. What is the BIGGEST problem? (e.g., too few trades, all trades losing, SL too tight)",
            "2. WHY is this happening? Trace the logic: which strategy is firing, what's the entry quality, why are exits failing?",
            "3. What specific parameter change would fix the root cause?",
            "4. What is your confidence level and what could go wrong?",
            "",
            "Write at least 3-5 paragraphs of detailed reasoning before making any edits.",
            "",
            "STRICT CONSTRAINTS:",
            f"1. Make AT MOST {self._max_changes} parameter changes total across both config files.",
            "2. You may edit ANY of these files:",
            "   - config/strategy.yaml — strategy parameters",
            "   - config/edges.yaml — edge filter toggles and params",
            "   - src/strategy/strategies/asian_breakout.py — Asian Range Breakout logic",
            "   - src/strategy/strategies/ema_pullback.py — EMA Pullback State Machine logic",
            "   - src/strategy/strategies/ichimoku.py — Ichimoku 15M strategy logic",
            "   - src/risk/exit_manager.py — exit/trailing stop logic",
            "   - src/strategy/signal_blender.py — signal selection logic",
            "3. If you see a code bug causing losses (e.g., wrong SL/TP logic, bad entry timing),",
            "   FIX THE CODE. Don't just tweak parameters around a broken algorithm.",
            "4. Each change must be justified by the data above.",
            "5. Focus on the biggest failure mode above.",
            "",
            "Apply your changes directly to the config files now.",
            "After making changes, explain what you changed and why.",
        ]

        return "\n".join(lines)

    def _spawn_claude(self, prompt: str) -> str:
        """Run Claude CLI with *prompt* on stdin and return stdout.

        Uses the command and timeout from ``config.claude``.
        Returns an empty string if Claude is not installed or times out.
        """
        claude_cfg = self._config.get("claude", {})
        command: List[str] = claude_cfg.get(
            "command", ["claude", "-p", "--dangerously-skip-permissions"]
        )
        timeout: int = int(claude_cfg.get("timeout_seconds", 300))

        try:
            completed = subprocess.run(
                command,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(_REPO_ROOT),
                encoding="utf-8",
            )
            if completed.returncode != 0:
                logger.warning(
                    "_spawn_claude: Claude exited with code %d. stderr: %s",
                    completed.returncode,
                    completed.stderr[:500] if completed.stderr else "",
                )
            return completed.stdout or ""
        except FileNotFoundError:
            logger.error(
                "_spawn_claude: Claude CLI not found. "
                "Install with: npm install -g @anthropic-ai/claude-code"
            )
            return ""
        except subprocess.TimeoutExpired:
            logger.error("_spawn_claude: Claude CLI timed out after %ds.", timeout)
            return ""
        except OSError as exc:
            logger.error("_spawn_claude: OS error spawning Claude: %s", exc)
            return ""

    def _detect_changes(self) -> str:
        """Return git diff output for config and strategy source files.

        Returns an empty string if git is unavailable or there are no changes.
        """
        config_files = [
            str(_STRATEGY_CONFIG_PATH.relative_to(_REPO_ROOT)),
            str(_EDGES_CONFIG_PATH.relative_to(_REPO_ROOT)),
            "src/strategy/strategies/asian_breakout.py",
            "src/strategy/strategies/ema_pullback.py",
            "src/strategy/strategies/ichimoku.py",
            "src/risk/exit_manager.py",
            "src/strategy/signal_blender.py",
        ]
        try:
            completed = subprocess.run(
                ["git", "diff", "--"] + config_files,
                capture_output=True,
                text=True,
                cwd=str(_REPO_ROOT),
            )
            return completed.stdout or ""
        except FileNotFoundError:
            logger.warning("_detect_changes: git not found.")
            return ""
        except OSError as exc:
            logger.error("_detect_changes: OS error running git diff: %s", exc)
            return ""

    def _revert_changes(self) -> None:
        """Revert config and strategy source files to HEAD using git checkout."""
        config_files = [
            str(_STRATEGY_CONFIG_PATH.relative_to(_REPO_ROOT)),
            str(_EDGES_CONFIG_PATH.relative_to(_REPO_ROOT)),
            "src/strategy/strategies/asian_breakout.py",
            "src/strategy/strategies/ema_pullback.py",
            "src/strategy/strategies/ichimoku.py",
            "src/risk/exit_manager.py",
            "src/strategy/signal_blender.py",
        ]
        try:
            completed = subprocess.run(
                ["git", "checkout", "--"] + config_files,
                capture_output=True,
                text=True,
                cwd=str(_REPO_ROOT),
            )
            if completed.returncode != 0:
                logger.error(
                    "_revert_changes: git checkout failed (code %d): %s",
                    completed.returncode,
                    completed.stderr,
                )
            else:
                logger.info("_revert_changes: config files reverted to HEAD.")
        except FileNotFoundError:
            logger.warning("_revert_changes: git not found — cannot revert.")
        except OSError as exc:
            logger.error("_revert_changes: OS error: %s", exc)

    def _check_plateau(self, history: List[Dict[str, Any]]) -> bool:
        """Return True when the last N iterations show < threshold% improvement.

        Parameters
        ----------
        history:
            Ordered list of iteration result dicts (ascending by iteration).

        Returns
        -------
        bool
            ``True`` when ``plateau_iterations`` consecutive iterations each
            improved pass rate by less than ``plateau_threshold`` relative to
            the previous iteration.  ``False`` otherwise or when the history
            is shorter than ``plateau_iterations``.
        """
        n = self._plateau_iterations
        if len(history) < n:
            return False

        recent = history[-n:]
        for i in range(1, len(recent)):
            prev_pass_rate = float(recent[i - 1].get("pass_rate", 0.0))
            curr_pass_rate = float(recent[i].get("pass_rate", 0.0))
            if prev_pass_rate == 0.0:
                # Avoid division-by-zero; treat as improvement to be safe
                return False
            improvement = (curr_pass_rate - prev_pass_rate) / abs(prev_pass_rate)
            if improvement >= self._plateau_threshold:
                return False
        return True

    def _load_config(self) -> Dict[str, Any]:
        """Read and merge strategy.yaml and edges.yaml from disk.

        Returns
        -------
        dict
            Flat merged config with ``edges`` sub-dict for the edge settings.
        """
        config: Dict[str, Any] = {}

        try:
            strategy_text = _STRATEGY_CONFIG_PATH.read_text(encoding="utf-8")
            strategy_cfg: Dict[str, Any] = yaml.safe_load(strategy_text) or {}

            # Determine the active strategy name
            active = strategy_cfg.get("active_strategy", "ichimoku")

            # Flatten the active strategy's sub-sections (ichimoku, adx, atr, signal)
            # into top-level keys that IchimokuBacktester and SignalEngine expect.
            strategy_params = strategy_cfg.get("strategies", {}).get(active, {})
            _KEY_MAP = {
                # strategies.ichimoku.atr.stop_multiplier -> atr_stop_multiplier
                ("atr", "stop_multiplier"): "atr_stop_multiplier",
                ("atr", "period"): "atr_period",
                ("adx", "period"): "adx_period",
                ("adx", "threshold"): "adx_threshold",
                ("ichimoku", "tenkan_period"): "tenkan_period",
                ("ichimoku", "kijun_period"): "kijun_period",
                ("ichimoku", "senkou_b_period"): "senkou_b_period",
                ("signal", "min_confluence_score"): "min_confluence_score",
                ("signal", "tier_a_plus"): "tier_a_plus",
                ("signal", "tier_b"): "tier_b",
                ("signal", "tier_c"): "tier_c",
                ("signal", "timeframes"): "timeframes",
            }
            for (section, key), flat_key in _KEY_MAP.items():
                sub = strategy_params.get(section, {})
                if isinstance(sub, dict) and key in sub:
                    config[flat_key] = sub[key]

            # Flatten risk and exit sections (already one level deep)
            for section in ("risk", "exit"):
                vals = strategy_cfg.get(section, {})
                if isinstance(vals, dict):
                    config.update(vals)

            # Keep active_strategy for reference
            config["active_strategy"] = active

            # Expose prop_firm and active_strategies for challenge simulation
            if "prop_firm" in strategy_cfg:
                config["prop_firm"] = strategy_cfg["prop_firm"]
            if "active_strategies" in strategy_cfg:
                config["active_strategies"] = strategy_cfg["active_strategies"]

            # Expose per-strategy configs (asian_breakout, ema_pullback) for multi-strategy backtesting
            all_strategies = strategy_cfg.get("strategies", {})
            for strat_name in ("asian_breakout", "ema_pullback"):
                if strat_name in all_strategies:
                    config[strat_name] = all_strategies[strat_name]
        except (OSError, yaml.YAMLError) as exc:
            logger.error("_load_config: failed to read strategy.yaml: %s", exc)

        try:
            edges_text = _EDGES_CONFIG_PATH.read_text(encoding="utf-8")
            edges_cfg: Dict[str, Any] = yaml.safe_load(edges_text) or {}
            config["edges"] = edges_cfg
        except (OSError, yaml.YAMLError) as exc:
            logger.error("_load_config: failed to read edges.yaml: %s", exc)

        return config

    def _slice_data(self, df: pd.DataFrame, years: int = 2) -> pd.DataFrame:
        """Return the last *years* of data from *df*.

        Assumes the DataFrame has a ``DatetimeIndex``.  Falls back to the
        full DataFrame when the index type is not datetime.

        Parameters
        ----------
        df:
            Full historical OHLCV DataFrame.
        years:
            Number of years to retain (counting back from the last row).

        Returns
        -------
        pd.DataFrame
            Slice containing only the last *years* of data.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning(
                "_slice_data: DataFrame index is not DatetimeIndex (%s). "
                "Returning full dataset.",
                type(df.index).__name__,
            )
            return df

        cutoff = df.index.max() - timedelta(days=years * 365)
        sliced = df[df.index >= cutoff]
        if sliced.empty:
            logger.warning(
                "_slice_data: slice for last %d years is empty — returning full dataset.",
                years,
            )
            return df
        return sliced

    @staticmethod
    def _summarise_diff(diff: str) -> str:
        """Extract changed parameter names from a unified diff string.

        Parses lines beginning with ``-`` or ``+`` (not ``---``/``+++``) and
        returns a compact summary like ``"tenkan_period, adx_threshold"``.
        Falls back to ``"config changes"`` when parsing yields nothing.
        """
        changed: List[str] = []
        for line in diff.splitlines():
            if (line.startswith("-") and not line.startswith("---")) or (
                line.startswith("+") and not line.startswith("+++")
            ):
                stripped = line[1:].strip()
                # Grab the key portion (everything before the colon)
                if ":" in stripped:
                    key = stripped.split(":")[0].strip()
                    if key and key not in changed:
                        changed.append(key)
        return ", ".join(changed[:5]) if changed else "config changes"

    @staticmethod
    def _dummy_embedding():
        """Return a zero embedding for similarity queries when no real embedding is available."""
        import numpy as np  # local import to keep top-level imports minimal
        return np.zeros(64, dtype=float)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def load_loop_config(config_path: Path) -> Dict[str, Any]:
    """Load and return the optimization loop YAML config."""
    try:
        text = config_path.read_text(encoding="utf-8")
        return yaml.safe_load(text) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Could not load loop config from %s: %s", config_path, exc)
        return {}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Agentic self-optimization loop for the Ichimoku trading strategy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-file",
        required=True,
        help="Path to the historical data parquet file (e.g. data/xauusd_1m.parquet).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override max iterations from config.",
    )
    parser.add_argument(
        "--target-pass-rate",
        type=float,
        default=None,
        help="Override target challenge pass rate from config (e.g. 0.50 for 50%%).",
    )
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG_PATH),
        help="Path to optimization_loop.yaml.",
    )
    parser.add_argument(
        "--reports-dir",
        default=None,
        help="Override the reports directory.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg = load_loop_config(Path(args.config))

    # Apply CLI overrides
    opt = cfg.setdefault("optimization", {})
    if args.max_iterations is not None:
        opt["max_iterations"] = args.max_iterations
    if args.target_pass_rate is not None:
        opt["target_pass_rate"] = args.target_pass_rate

    loop = OptimizationLoop(
        config=cfg,
        data_file=args.data_file,
        reports_dir=args.reports_dir,
    )

    summary = loop.run()

    # Print final summary to stdout as JSON
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
