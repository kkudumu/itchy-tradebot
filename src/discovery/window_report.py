"""Per-window and summary report generation for the discovery orchestrator.

Generates JSON reports for each rolling window (trades, SHAP findings,
hypotheses, config changes, challenge result) and a summary report
tracking edge lifecycle across all windows.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class WindowReportGenerator:
    """Generate per-window and summary JSON reports.

    Parameters
    ----------
    reports_dir:
        Directory where window_*.json and discovery_summary.json are written.
    """

    def __init__(self, reports_dir: str = "reports/discovery") -> None:
        self._reports_dir = Path(reports_dir)
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_window_report(
        self,
        window_id: str,
        window_index: int,
        trades: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        challenge_result: Optional[Dict[str, Any]] = None,
        shap_findings: Optional[Dict[str, Any]] = None,
        regime: Optional[str] = None,
        config_changes: Optional[List[str]] = None,
        hypotheses: Optional[List[Dict[str, Any]]] = None,
        pattern_analysis: Optional[Dict[str, Any]] = None,
        codegen_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Generate and persist a per-window report.

        Returns the report dict.
        """
        report: Dict[str, Any] = {
            "window_id": window_id,
            "window_index": window_index,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "trade_count": len(trades),
            "metrics": metrics,
            "challenge_result": challenge_result or {},
            "shap_findings": shap_findings or {},
            "regime": regime,
            "config_changes": config_changes or [],
            "hypotheses": hypotheses or [],
            "pattern_analysis": pattern_analysis or {},
            "codegen_results": codegen_results or [],
        }

        path = self._reports_dir / f"window_{window_id}.json"
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        logger.info("Window report written: %s", path)

        return report

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report across all window reports.

        Reads all window_*.json files in the reports directory and
        aggregates edge lifecycle stats.
        """
        window_reports = self._load_all_window_reports()

        total_windows = len(window_reports)
        phase_1_pass = sum(
            1 for r in window_reports
            if r.get("challenge_result", {}).get("passed_phase_1", False)
        )
        phase_2_pass = sum(
            1 for r in window_reports
            if r.get("challenge_result", {}).get("passed_phase_2", False)
        )

        # Count hypotheses
        all_hypotheses = []
        for r in window_reports:
            all_hypotheses.extend(r.get("hypotheses", []))

        proposed = sum(1 for h in all_hypotheses if h.get("status") != "rejected")
        validated = sum(1 for h in all_hypotheses if h.get("status") == "validated")

        # Count SHAP discoveries
        total_shap_rules = sum(
            r.get("shap_findings", {}).get("rules_count", 0)
            for r in window_reports
        )

        # Count config changes
        total_changes = sum(len(r.get("config_changes", [])) for r in window_reports)

        # Count codegen results
        total_codegen = sum(len(r.get("codegen_results", [])) for r in window_reports)

        summary: Dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_windows": total_windows,
            "phase_1_pass_count": phase_1_pass,
            "phase_2_pass_count": phase_2_pass,
            "phase_1_pass_rate": phase_1_pass / total_windows if total_windows else 0.0,
            "phase_2_pass_rate": phase_2_pass / total_windows if total_windows else 0.0,
            "edges_discovered": total_shap_rules,
            "edges_validated": validated,
            "edges_absorbed": total_codegen,
            "total_hypotheses_proposed": proposed,
            "total_hypotheses_validated": validated,
            "total_config_changes": total_changes,
            "per_window_summary": [
                {
                    "window_id": r["window_id"],
                    "trade_count": r.get("trade_count", 0),
                    "regime": r.get("regime"),
                    "challenge_p1": r.get("challenge_result", {}).get("passed_phase_1", False),
                    "challenge_p2": r.get("challenge_result", {}).get("passed_phase_2", False),
                    "shap_rules": r.get("shap_findings", {}).get("rules_count", 0),
                    "config_changes": len(r.get("config_changes", [])),
                }
                for r in window_reports
            ],
        }

        path = self._reports_dir / "discovery_summary.json"
        path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        logger.info("Summary report written: %s (%d windows)", path, total_windows)

        return summary

    def _load_all_window_reports(self) -> List[Dict[str, Any]]:
        """Load all window_*.json files, sorted by window_index."""
        reports = []
        for p in sorted(self._reports_dir.glob("window_*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                reports.append(data)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not load report %s: %s", p, exc)
        reports.sort(key=lambda r: r.get("window_index", 0))
        return reports

    def get_dashboard_payload(self) -> Dict[str, Any]:
        """Build a payload suitable for the OptimizationDashboardServer.

        Returns a dict with discovery findings formatted for the
        dashboard's Optimization tab.
        """
        summary = self.generate_summary_report()
        return {
            "discovery": {
                "total_windows": summary["total_windows"],
                "phase_1_pass_rate": summary["phase_1_pass_rate"],
                "phase_2_pass_rate": summary["phase_2_pass_rate"],
                "edges_discovered": summary["edges_discovered"],
                "edges_absorbed": summary["edges_absorbed"],
                "hypotheses": summary["total_hypotheses_proposed"],
                "per_window": summary["per_window_summary"],
            },
        }
