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
- Sharpe ratio exceeds ``target_sharpe``
- Metrics plateau: 3 consecutive iterations with < 5% Sharpe improvement
- Maximum iterations reached

CLI usage
---------
    python scripts/run_optimization_loop.py \\
        --data-file data/xauusd_1m.parquet \\
        --max-iterations 10 \\
        --target-sharpe 1.0

Config is read from ``config/optimization_loop.yaml``; CLI flags override it.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import textwrap
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
        self._target_sharpe: float = float(opt.get("target_sharpe", 1.0))
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
            logger.info(
                "Iteration %d: Sharpe=%.3f  WinRate=%.1f%%  MaxDD=%.1f%%",
                iteration,
                current_sharpe,
                float(metrics.get("win_rate", 0.0)) * 100,
                float(metrics.get("max_drawdown", 0.0)) * 100,
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
            if not best_metrics or current_sharpe > float(best_metrics.get("sharpe", float("-inf"))):
                best_metrics = dict(metrics)
                best_config = dict(config_snapshot)

            entry = {
                "iteration": iteration,
                "run_id": run_id,
                "metrics": dict(metrics),
                "sharpe": current_sharpe,
            }
            history.append(entry)

            # ---- Check target -----------------------------------------------
            if current_sharpe >= self._target_sharpe:
                stop_reason = "target_reached"
                logger.info(
                    "Target Sharpe %.2f reached (current=%.3f). Stopping.",
                    self._target_sharpe,
                    current_sharpe,
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
            logger.debug("Claude output: %s", claude_output[:500] if claude_output else "(empty)")

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

            # ---- Decide keep / revert ---------------------------------------
            if new_sharpe >= current_sharpe:
                verdict = "kept"
                logger.info(
                    "Config improved: Sharpe %.3f → %.3f. Keeping changes.",
                    current_sharpe,
                    new_sharpe,
                )
                # Update best
                if new_sharpe > float(best_metrics.get("sharpe", float("-inf"))):
                    best_metrics = dict(new_metrics)
                    best_config = dict(new_config_snapshot)
                # Update current iteration with new metrics
                history[-1]["metrics"] = dict(new_metrics)
                history[-1]["sharpe"] = new_sharpe
            else:
                verdict = "reverted"
                logger.info(
                    "Config worsened: Sharpe %.3f → %.3f. Reverting.",
                    current_sharpe,
                    new_sharpe,
                )
                self._revert_changes()

            # ---- Append learning to CLAUDE.md --------------------------------
            self._claude_writer.append_learning(
                run_id=run_id,
                changes_made=self._summarise_diff(changes_diff),
                metrics_before={
                    "sharpe": metrics.get("sharpe"),
                    "win_rate": metrics.get("win_rate"),
                    "max_dd": metrics.get("max_drawdown"),
                },
                metrics_after={
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
            logger.info(
                "Final validation: Sharpe=%.3f  WinRate=%.1f%%",
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
            "Optimization complete. Stop reason: %s. Best Sharpe: %.3f",
            stop_reason,
            float(best_metrics.get("sharpe", 0.0)),
        )
        return summary

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_backtest(self, df: pd.DataFrame, config: Dict[str, Any]) -> Any:
        """Instantiate the backtester and run it on *df*."""
        backtester = _BacktesterClass(config=config)
        return backtester.run(df)

    def _build_claude_prompt(
        self,
        report: Dict[str, Any],
        previous_report: Optional[Dict[str, Any]],
        similar_configs: List[Any],
        learnings: str,
    ) -> str:
        """Build the structured prompt sent to Claude CLI.

        The prompt includes: current metrics, previous-run comparison,
        similar past configs from pgvector, accumulated learnings, and
        a strict constraint to make at most ``max_changes_per_iteration``
        parameter changes.
        """
        run_id = report.get("run_id", "unknown")
        metrics = report.get("metrics", {})
        comparison = report.get("comparison", {})
        config_snap = report.get("config_snapshot", {})

        lines: List[str] = [
            "You are a quantitative trading strategy optimizer.",
            "",
            f"## Current Run: {run_id}",
            "### Performance Metrics",
            f"- Sharpe Ratio:   {metrics.get('sharpe', 'N/A')}",
            f"- Win Rate:       {metrics.get('win_rate', 'N/A')}",
            f"- Max Drawdown:   {metrics.get('max_drawdown', 'N/A')}",
            f"- Total Return:   {metrics.get('total_return', 'N/A')}",
            f"- Profit Factor:  {metrics.get('profit_factor', 'N/A')}",
            f"- Trade Count:    {metrics.get('trade_count', 'N/A')}",
            f"- Expectancy:     {metrics.get('expectancy', 'N/A')}",
            "",
        ]

        # Previous run comparison
        if previous_report and comparison.get("has_previous"):
            prev_metrics = previous_report.get("metrics", {})
            lines += [
                "## Previous Run Comparison",
                f"- Previous Sharpe: {prev_metrics.get('sharpe', 'N/A')}",
                f"- Sharpe delta:    {comparison.get('sharpe', {}).get('delta', 'N/A')}",
                f"- WinRate delta:   {comparison.get('win_rate', {}).get('delta', 'N/A')}",
                f"- MaxDD delta:     {comparison.get('max_drawdown', {}).get('delta', 'N/A')}",
                "",
            ]
        else:
            lines += ["## Previous Run Comparison", "No previous run available.", ""]

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
            textwrap.dedent(f"""
            Analyze the metrics above and suggest improvements to the strategy configuration.

            STRICT CONSTRAINTS:
            1. Make AT MOST {self._max_changes} parameter changes total across both config files.
            2. Edit ONLY the following files:
               - config/strategy.yaml
               - config/edges.yaml
            3. Do NOT change Python source files.
            4. Each change must be justified by the data above.
            5. Prioritize changes that address the biggest weakness visible in the metrics.

            Target: Sharpe Ratio >= {self._target_sharpe:.2f}

            Apply your changes directly to the config files now.
            After making changes, output a brief explanation of what you changed and why.
            """).strip(),
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
        """Return git diff output for the two config files.

        Returns an empty string if git is unavailable or there are no changes.
        """
        config_files = [
            str(_STRATEGY_CONFIG_PATH.relative_to(_REPO_ROOT)),
            str(_EDGES_CONFIG_PATH.relative_to(_REPO_ROOT)),
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
        """Revert both config files to HEAD using git checkout."""
        config_files = [
            str(_STRATEGY_CONFIG_PATH.relative_to(_REPO_ROOT)),
            str(_EDGES_CONFIG_PATH.relative_to(_REPO_ROOT)),
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
            improved Sharpe by less than ``plateau_threshold`` relative to
            the previous iteration.  ``False`` otherwise or when the history
            is shorter than ``plateau_iterations``.
        """
        n = self._plateau_iterations
        if len(history) < n:
            return False

        recent = history[-n:]
        for i in range(1, len(recent)):
            prev_sharpe = float(recent[i - 1].get("sharpe", 0.0))
            curr_sharpe = float(recent[i].get("sharpe", 0.0))
            if prev_sharpe == 0.0:
                # Avoid division-by-zero; treat as improvement to be safe
                return False
            improvement = (curr_sharpe - prev_sharpe) / abs(prev_sharpe)
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
            # Flatten nested dicts one level (e.g., ichimoku.tenkan_period → tenkan_period)
            for section, values in strategy_cfg.items():
                if isinstance(values, dict):
                    config.update(values)
                else:
                    config[section] = values
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
        "--target-sharpe",
        type=float,
        default=None,
        help="Override target Sharpe ratio from config.",
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
    if args.target_sharpe is not None:
        opt["target_sharpe"] = args.target_sharpe

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
