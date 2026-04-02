# tests/test_window_report.py
"""Tests for per-window and summary report generation."""

import json
import tempfile
from pathlib import Path

import pytest


class TestWindowReport:
    def test_generate_per_window_report(self):
        from src.discovery.window_report import WindowReportGenerator

        gen = WindowReportGenerator(reports_dir=str(Path(tempfile.mkdtemp()) / "reports"))
        report = gen.generate_window_report(
            window_id="w_003",
            window_index=3,
            trades=[{"r_multiple": 1.5}, {"r_multiple": -0.8}],
            metrics={"win_rate": 0.50, "total_trades": 2, "sharpe_ratio": 1.2},
            challenge_result={"passed_phase_1": True, "passed_phase_2": False},
            shap_findings={"top_features": ["adx_value", "sess_london"], "rules_count": 3},
            regime="trending_bullish",
            config_changes=["Adjusted min_confluence_score 4->3"],
            hypotheses=[{"description": "ADX + London boosts WR", "confidence": "high"}],
        )

        assert report["window_id"] == "w_003"
        assert report["metrics"]["win_rate"] == 0.50
        assert report["challenge_result"]["passed_phase_1"] is True
        assert len(report["config_changes"]) == 1

    def test_report_saved_to_disk(self):
        from src.discovery.window_report import WindowReportGenerator

        reports_dir = str(Path(tempfile.mkdtemp()) / "reports")
        gen = WindowReportGenerator(reports_dir=reports_dir)
        gen.generate_window_report(
            window_id="w_001",
            window_index=1,
            trades=[],
            metrics={},
        )

        path = Path(reports_dir) / "window_w_001.json"
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["window_id"] == "w_001"

    def test_report_includes_timestamp(self):
        from src.discovery.window_report import WindowReportGenerator

        gen = WindowReportGenerator(reports_dir=str(Path(tempfile.mkdtemp()) / "reports"))
        report = gen.generate_window_report(
            window_id="w_001", window_index=1, trades=[], metrics={},
        )

        assert "generated_at" in report


class TestSummaryReport:
    def test_generate_summary_from_window_reports(self):
        from src.discovery.window_report import WindowReportGenerator

        reports_dir = str(Path(tempfile.mkdtemp()) / "reports")
        gen = WindowReportGenerator(reports_dir=reports_dir)

        # Generate 3 window reports
        for i in range(3):
            gen.generate_window_report(
                window_id=f"w_{i:03d}",
                window_index=i,
                trades=[{"r_multiple": 1.0}],
                metrics={"win_rate": 0.35 + i * 0.05, "total_trades": 10 + i},
                challenge_result={"passed_phase_1": i >= 1, "passed_phase_2": i >= 2},
                config_changes=[f"change_{i}"] if i > 0 else [],
            )

        summary = gen.generate_summary_report()
        assert summary["total_windows"] == 3
        assert "edges_discovered" in summary
        assert "edges_validated" in summary
        assert "edges_absorbed" in summary
        assert "phase_1_pass_count" in summary
        assert "phase_2_pass_count" in summary

    def test_summary_tracks_edge_lifecycle(self):
        from src.discovery.window_report import WindowReportGenerator

        reports_dir = str(Path(tempfile.mkdtemp()) / "reports")
        gen = WindowReportGenerator(reports_dir=reports_dir)

        gen.generate_window_report(
            window_id="w_000", window_index=0, trades=[], metrics={},
            shap_findings={"rules_count": 2},
        )
        gen.generate_window_report(
            window_id="w_001", window_index=1, trades=[], metrics={},
            hypotheses=[
                {"description": "H1", "status": "proposed"},
                {"description": "H2", "status": "validated"},
            ],
        )

        summary = gen.generate_summary_report()
        assert summary["total_hypotheses_proposed"] >= 1

    def test_summary_saved_to_disk(self):
        from src.discovery.window_report import WindowReportGenerator

        reports_dir = str(Path(tempfile.mkdtemp()) / "reports")
        gen = WindowReportGenerator(reports_dir=reports_dir)
        gen.generate_window_report(
            window_id="w_000", window_index=0, trades=[], metrics={},
        )
        gen.generate_summary_report()

        path = Path(reports_dir) / "discovery_summary.json"
        assert path.exists()
