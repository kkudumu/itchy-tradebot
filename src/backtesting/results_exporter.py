"""
Backtest Results Exporter — writes structured JSON reports after each run.

Each report contains aggregate metrics, worst/best trades, per-edge stats,
the full config snapshot, and a delta comparison to the previous run.
Reports are capped at 50 KB (suitable for Claude CLI context windows).

Usage example::

    from src.backtesting.results_exporter import ResultsExporter
    exporter = ResultsExporter(reports_dir="reports")
    path = exporter.export_run_report(result, config, run_id="run_001")
    report = exporter.load_run_report("run_001")
    runs = exporter.list_runs()
"""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Hard limit: keep each report under 50 KB so it fits in a Claude CLI context.
_MAX_REPORT_BYTES: int = 50 * 1024  # 50 KB

# Number of extreme trades to include (top N and bottom N by R-multiple).
_EXTREME_TRADE_COUNT: int = 5

# Fields kept per trade in the report (reduces size vs full trade dict).
_TRADE_FIELDS: Tuple[str, ...] = (
    "entry_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "direction",
    "r_multiple",
    "pnl",
    "confluence_score",
    "signal_tier",
    "edge_results",
)

# Metric keys extracted for list_runs() summary.
_SUMMARY_METRICS: Tuple[str, ...] = (
    "sharpe",
    "win_rate",
    "trade_count",
    "total_return",
    "max_drawdown",
    "expectancy",
)


class ResultsExporter:
    """Exports backtest run reports to disk as structured JSON files.

    Parameters
    ----------
    reports_dir:
        Directory where ``run_{N}.json`` files are written.
        Created automatically if it does not exist.
    """

    def __init__(self, reports_dir: str = "reports") -> None:
        self._reports_dir = Path(reports_dir)
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export_run_report(
        self,
        result: Any,
        config: Dict[str, Any],
        run_id: str,
        previous_runs: Optional[List[Dict[str, Any]]] = None,
    ) -> Path:
        """Serialise a BacktestResult + config into a JSON report file.

        Parameters
        ----------
        result:
            A ``BacktestResult`` instance (or any object with ``.metrics``
            and ``.trades`` attributes, or a plain dict).
        config:
            Full configuration dict (e.g. merged edges.yaml + strategy.yaml).
        run_id:
            Unique identifier for this run (e.g. ``"run_001"``).
        previous_runs:
            Optional list of previously loaded report dicts used to compute
            deltas.  Pass ``None`` when there is no history yet.

        Returns
        -------
        Path
            Absolute path of the written JSON file.
        """
        metrics = self._extract_metrics(result)
        trades = self._extract_trades(result)

        best_trades, worst_trades = self._select_extreme_trades(trades)
        edge_stats = self._compute_edge_stats(trades)

        previous_metrics = self._get_latest_metrics(previous_runs)
        comparison = self._compute_comparison(metrics, previous_metrics)

        report: Dict[str, Any] = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "best_trades": best_trades,
            "worst_trades": worst_trades,
            "edge_stats": edge_stats,
            "config_snapshot": config,
            "comparison": comparison,
        }

        path = self._reports_dir / f"{run_id}.json"
        serialised = json.dumps(report, indent=2, default=str)

        byte_count = len(serialised.encode("utf-8"))
        if byte_count > _MAX_REPORT_BYTES:
            logger.warning(
                "Report %s is %d bytes (limit %d KB). "
                "Trimming config_snapshot and truncating trade context.",
                run_id,
                byte_count,
                _MAX_REPORT_BYTES // 1024,
            )
            report = self._trim_report(report)
            serialised = json.dumps(report, indent=2, default=str)
            byte_count = len(serialised.encode("utf-8"))
            if byte_count > _MAX_REPORT_BYTES:
                logger.error(
                    "Report %s still %d bytes after trimming — writing anyway.",
                    run_id,
                    byte_count,
                )

        path.write_text(serialised, encoding="utf-8")
        logger.info("Wrote report %s (%d bytes) to %s", run_id, byte_count, path)
        return path

    def load_run_report(self, run_id: str) -> Dict[str, Any]:
        """Read a previously saved report from disk.

        Parameters
        ----------
        run_id:
            The run identifier used when the report was exported.

        Returns
        -------
        dict
            The full report dictionary.

        Raises
        ------
        FileNotFoundError
            If no report file exists for ``run_id``.
        """
        path = self._reports_dir / f"{run_id}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"No report found for run_id '{run_id}' at {path}"
            )
        text = path.read_text(encoding="utf-8")
        return json.loads(text)

    def list_runs(self) -> List[Dict[str, Any]]:
        """Return a summary list of all saved runs, sorted by timestamp.

        Each entry contains: ``run_id``, ``timestamp``, and the subset of
        metrics defined in ``_SUMMARY_METRICS``.

        Returns
        -------
        list[dict]
            Summaries sorted ascending by timestamp (oldest first).
        """
        summaries: List[Dict[str, Any]] = []
        for json_file in sorted(self._reports_dir.glob("*.json")):
            try:
                text = json_file.read_text(encoding="utf-8")
                report = json.loads(text)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Could not read report file %s: %s", json_file, exc)
                continue

            summary: Dict[str, Any] = {
                "run_id": report.get("run_id", json_file.stem),
                "timestamp": report.get("timestamp", ""),
            }
            metrics = report.get("metrics", {})
            for key in _SUMMARY_METRICS:
                summary[key] = metrics.get(key)
            summaries.append(summary)

        summaries.sort(key=lambda s: s.get("timestamp") or "")
        return summaries

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_metrics(self, result: Any) -> Dict[str, Any]:
        """Pull the metrics dict out of a BacktestResult or plain dict."""
        if isinstance(result, dict):
            raw = result.get("metrics", {})
        else:
            raw = getattr(result, "metrics", {}) or {}

        # Normalise to plain serialisable dict.
        metrics: Dict[str, Any] = {}
        desired = (
            "sharpe",
            "sortino",
            "calmar",
            "win_rate",
            "max_drawdown",
            "total_return",
            "profit_factor",
            "expectancy",
            "trade_count",
        )
        for key in desired:
            val = raw.get(key)
            if val is not None:
                try:
                    metrics[key] = float(val)
                except (TypeError, ValueError):
                    metrics[key] = val
        # Keep any extra keys the engine adds.
        for key, val in raw.items():
            if key not in metrics:
                try:
                    metrics[key] = float(val)
                except (TypeError, ValueError):
                    metrics[key] = val
        return metrics

    def _extract_trades(self, result: Any) -> List[Dict[str, Any]]:
        """Pull the trades list out of a BacktestResult or plain dict."""
        if isinstance(result, dict):
            trades = result.get("trades", [])
        else:
            trades = getattr(result, "trades", []) or []
        return list(trades)

    def _select_extreme_trades(
        self,
        trades: List[Dict[str, Any]],
        n: int = _EXTREME_TRADE_COUNT,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Return (best_trades, worst_trades) — top/bottom N by R-multiple.

        Parameters
        ----------
        trades:
            Full trade list from the backtest result.
        n:
            How many top and bottom trades to include.

        Returns
        -------
        tuple[list, list]
            ``(best_trades, worst_trades)`` — each is a list of at most *n*
            trade dicts containing only ``_TRADE_FIELDS``.
        """
        if not trades:
            return [], []

        def _r(t: Dict[str, Any]) -> float:
            try:
                return float(t.get("r_multiple", 0.0) or 0.0)
            except (TypeError, ValueError):
                return 0.0

        sorted_by_r = sorted(trades, key=_r, reverse=True)

        def _trim(t: Dict[str, Any]) -> Dict[str, Any]:
            return {k: t.get(k) for k in _TRADE_FIELDS}

        best = [_trim(t) for t in sorted_by_r[:n]]
        worst = [_trim(t) for t in sorted_by_r[-n:]]
        # worst list should be ascending (worst first).
        worst.reverse()
        return best, worst

    def _compute_edge_stats(
        self,
        trades: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Compute per-edge pass/fail/win rates directly from the trade list.

        Each trade dict is expected to contain an ``edge_results`` key that
        maps edge name → ``True`` (passed) / ``False`` (failed) / other truthy.

        Parameters
        ----------
        trades:
            Full trade list from the backtest result.

        Returns
        -------
        dict
            ``{edge_name: {total, passed, failed, pass_rate, win_rate}}``
        """
        if not trades:
            return {}

        # Accumulate per-edge counters.
        # Structure: {edge: {"pass": int, "fail": int, "win_when_pass": int, "win_when_fail": int}}
        accum: Dict[str, Dict[str, int]] = {}

        for trade in trades:
            edge_results = trade.get("edge_results") or {}
            r_mult = trade.get("r_multiple", 0.0)
            try:
                won = float(r_mult or 0.0) > 0.0
            except (TypeError, ValueError):
                won = False

            for edge_name, passed in edge_results.items():
                if edge_name not in accum:
                    accum[edge_name] = {
                        "pass": 0,
                        "fail": 0,
                        "win_when_pass": 0,
                        "win_when_fail": 0,
                    }
                if passed:
                    accum[edge_name]["pass"] += 1
                    if won:
                        accum[edge_name]["win_when_pass"] += 1
                else:
                    accum[edge_name]["fail"] += 1
                    if won:
                        accum[edge_name]["win_when_fail"] += 1

        stats: Dict[str, Dict[str, Any]] = {}
        for edge_name, counts in accum.items():
            total_pass = counts["pass"]
            total_fail = counts["fail"]
            total = total_pass + total_fail
            pass_rate = round(total_pass / total, 4) if total > 0 else 0.0
            win_rate_pass = (
                round(counts["win_when_pass"] / total_pass, 4)
                if total_pass > 0
                else 0.0
            )
            win_rate_fail = (
                round(counts["win_when_fail"] / total_fail, 4)
                if total_fail > 0
                else 0.0
            )
            stats[edge_name] = {
                "total": total,
                "passed": total_pass,
                "failed": total_fail,
                "pass_rate": pass_rate,
                "win_rate_when_passed": win_rate_pass,
                "win_rate_when_failed": win_rate_fail,
            }

        return stats

    def _compute_comparison(
        self,
        current_metrics: Dict[str, Any],
        previous_metrics: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute delta between current run and previous run metrics.

        Parameters
        ----------
        current_metrics:
            Metrics dict for the current run.
        previous_metrics:
            Metrics dict for the most recent previous run, or ``None``.

        Returns
        -------
        dict
            ``{metric: {"current": v, "previous": v, "delta": v}}`` for
            numeric metrics that appear in both runs.  When there is no
            previous run the dict contains only ``{"has_previous": False}``.
        """
        if not previous_metrics:
            return {"has_previous": False}

        comparison: Dict[str, Any] = {"has_previous": True}
        all_keys = set(current_metrics) | set(previous_metrics)
        for key in sorted(all_keys):
            curr = current_metrics.get(key)
            prev = previous_metrics.get(key)
            try:
                curr_f = float(curr) if curr is not None else None
                prev_f = float(prev) if prev is not None else None
            except (TypeError, ValueError):
                continue
            if curr_f is None or prev_f is None:
                continue
            delta = round(curr_f - prev_f, 6)
            comparison[key] = {
                "current": round(curr_f, 6),
                "previous": round(prev_f, 6),
                "delta": delta,
            }
        return comparison

    def _get_latest_metrics(
        self,
        previous_runs: Optional[List[Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        """Extract the metrics from the most recent previous run."""
        if not previous_runs:
            return None
        # Sort by timestamp descending; take the first.
        sorted_runs = sorted(
            previous_runs,
            key=lambda r: r.get("timestamp") or "",
            reverse=True,
        )
        return sorted_runs[0].get("metrics") or None

    def _trim_report(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Aggressively trim the report to stay under the 50 KB limit.

        Strategies (applied in order until the report fits):
        1. Remove the ``edge_results`` sub-dict from each trade entry
           (those are already summarised in ``edge_stats``).
        2. Truncate ``config_snapshot`` to first-level keys only.
        3. Drop the ``comparison`` section if still too large.
        """
        trimmed = copy.deepcopy(report)

        # Step 1: strip edge_results from individual trades.
        for section in ("best_trades", "worst_trades"):
            for trade in trimmed.get(section, []):
                trade.pop("edge_results", None)

        serialised = json.dumps(trimmed, indent=2, default=str)
        if len(serialised.encode("utf-8")) <= _MAX_REPORT_BYTES:
            return trimmed

        # Step 2: truncate config snapshot to top-level keys only.
        snapshot = trimmed.get("config_snapshot", {})
        trimmed["config_snapshot"] = {
            k: "(truncated — see full config files)" for k in snapshot
        }

        serialised = json.dumps(trimmed, indent=2, default=str)
        if len(serialised.encode("utf-8")) <= _MAX_REPORT_BYTES:
            return trimmed

        # Step 3: drop comparison section.
        trimmed["comparison"] = {"has_previous": report.get("comparison", {}).get("has_previous", False), "note": "truncated"}
        return trimmed
