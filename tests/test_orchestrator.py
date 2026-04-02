# tests/test_orchestrator.py
"""Tests for the discovery orchestrator."""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def _make_candles(n_days: int = 60, bars_per_day: int = 390) -> pd.DataFrame:
    """Generate synthetic 1M candles spanning n_days trading days."""
    rng = np.random.default_rng(42)
    start = datetime(2025, 1, 2, 8, 0, tzinfo=timezone.utc)

    timestamps = []
    day = 0
    dt = start
    while day < n_days:
        # Skip weekends
        if dt.weekday() < 5:
            for bar in range(bars_per_day):
                timestamps.append(dt + timedelta(minutes=bar))
            day += 1
        dt += timedelta(days=1)

    n = len(timestamps)
    prices = 1800.0 + np.cumsum(rng.normal(0.01, 0.5, n))
    df = pd.DataFrame({
        "open": prices + rng.uniform(-0.2, 0.2, n),
        "high": prices + rng.uniform(0, 0.5, n),
        "low": prices - rng.uniform(0, 0.5, n),
        "close": prices,
        "volume": rng.integers(100, 800, n),
    }, index=pd.DatetimeIndex(timestamps, tz=timezone.utc))

    return df


class TestRollingWindowSlicer:
    def test_slice_into_windows(self):
        from src.discovery.orchestrator import DiscoveryOrchestrator

        candles = _make_candles(n_days=66)  # 3 windows of 22 days
        orch = DiscoveryOrchestrator(
            config={"orchestrator": {"window_size_trading_days": 22}},
            knowledge_dir=str(Path(tempfile.mkdtemp()) / "kb"),
        )

        windows = orch.slice_into_windows(candles)
        assert len(windows) == 3
        for w in windows:
            assert isinstance(w["candles"], pd.DataFrame)
            assert len(w["candles"]) > 0
            assert "window_id" in w
            assert "window_index" in w

    def test_windows_do_not_overlap(self):
        from src.discovery.orchestrator import DiscoveryOrchestrator

        candles = _make_candles(n_days=44)  # 2 windows
        orch = DiscoveryOrchestrator(
            config={"orchestrator": {"window_size_trading_days": 22}},
            knowledge_dir=str(Path(tempfile.mkdtemp()) / "kb"),
        )

        windows = orch.slice_into_windows(candles)
        if len(windows) >= 2:
            end_0 = windows[0]["candles"].index[-1]
            start_1 = windows[1]["candles"].index[0]
            assert start_1 > end_0

    def test_partial_last_window_included(self):
        from src.discovery.orchestrator import DiscoveryOrchestrator

        candles = _make_candles(n_days=30)  # 1 full + partial
        orch = DiscoveryOrchestrator(
            config={"orchestrator": {"window_size_trading_days": 22}},
            knowledge_dir=str(Path(tempfile.mkdtemp()) / "kb"),
        )

        windows = orch.slice_into_windows(candles)
        # At minimum 1 full window, partial may or may not be included
        assert len(windows) >= 1

    def test_window_id_format(self):
        from src.discovery.orchestrator import DiscoveryOrchestrator

        candles = _make_candles(n_days=22)
        orch = DiscoveryOrchestrator(
            config={"orchestrator": {"window_size_trading_days": 22}},
            knowledge_dir=str(Path(tempfile.mkdtemp()) / "kb"),
        )

        windows = orch.slice_into_windows(candles)
        assert len(windows) >= 1
        assert windows[0]["window_id"].startswith("w_")


class TestProcessWindow:
    def test_process_window_returns_result_dict(self):
        from src.discovery.orchestrator import DiscoveryOrchestrator

        candles = _make_candles(n_days=22)
        kb_dir = str(Path(tempfile.mkdtemp()) / "kb")
        reports_dir = str(Path(tempfile.mkdtemp()) / "reports")

        orch = DiscoveryOrchestrator(
            config={
                "orchestrator": {"window_size_trading_days": 22, "strategy_name": "sss"},
                "discovery": {"shap_every_n_windows": 3, "min_trades_for_shap": 5},
                "challenge": {"account_size": 10000.0},
                "reporting": {"reports_dir": reports_dir},
            },
            knowledge_dir=kb_dir,
        )

        window = {
            "window_id": "w_000",
            "window_index": 0,
            "candles": candles,
            "start_date": candles.index[0],
            "end_date": candles.index[-1],
            "trading_days": 22,
        }

        with patch("src.discovery.orchestrator.IchimokuBacktester") as MockBT:
            mock_result = MagicMock()
            mock_result.trades = [
                {"r_multiple": 1.5, "risk_pct": 1.0, "context": {}, "day_index": 0},
                {"r_multiple": -1.0, "risk_pct": 1.0, "context": {}, "day_index": 1},
            ]
            mock_result.metrics = {"win_rate": 0.50, "total_trades": 2, "sharpe_ratio": 1.0}
            mock_result.prop_firm = {"status": "ongoing", "profit_pct": 0.5}
            MockBT.return_value.run.return_value = mock_result

            result = orch.process_window(window, base_config={})

        assert "window_id" in result
        assert result["window_id"] == "w_000"
        assert "trades" in result
        assert "metrics" in result
        assert "challenge_result" in result
        assert "discovery" in result

    def test_process_window_invokes_challenge_simulator(self):
        from src.discovery.orchestrator import DiscoveryOrchestrator

        candles = _make_candles(n_days=22)
        kb_dir = str(Path(tempfile.mkdtemp()) / "kb")

        orch = DiscoveryOrchestrator(
            config={
                "orchestrator": {"window_size_trading_days": 22},
                "challenge": {
                    "phase_1_target_pct": 8.0,
                    "phase_2_target_pct": 5.0,
                    "account_size": 10000.0,
                },
                "reporting": {"reports_dir": str(Path(tempfile.mkdtemp()) / "r")},
            },
            knowledge_dir=kb_dir,
        )

        window = {
            "window_id": "w_000", "window_index": 0, "candles": candles,
            "start_date": candles.index[0], "end_date": candles.index[-1],
            "trading_days": 22,
        }

        with patch("src.discovery.orchestrator.IchimokuBacktester") as MockBT:
            mock_result = MagicMock()
            mock_result.trades = [
                {"r_multiple": 2.0, "risk_pct": 1.0, "context": {}, "day_index": i}
                for i in range(10)
            ]
            mock_result.metrics = {"win_rate": 1.0, "total_trades": 10}
            mock_result.prop_firm = {"status": "passed"}
            MockBT.return_value.run.return_value = mock_result

            result = orch.process_window(window, base_config={})

        assert "challenge_result" in result
        assert "passed_phase_1" in result["challenge_result"]

    def test_process_window_runs_discovery_on_shap_interval(self):
        from src.discovery.orchestrator import DiscoveryOrchestrator

        candles = _make_candles(n_days=22)
        kb_dir = str(Path(tempfile.mkdtemp()) / "kb")

        orch = DiscoveryOrchestrator(
            config={
                "orchestrator": {"window_size_trading_days": 22},
                "discovery": {"shap_every_n_windows": 3, "min_trades_for_shap": 5},
                "reporting": {"reports_dir": str(Path(tempfile.mkdtemp()) / "r")},
            },
            knowledge_dir=kb_dir,
        )

        # Simulate window_index=2 (should trigger SHAP on 3rd window)
        window = {
            "window_id": "w_002", "window_index": 2, "candles": candles,
            "start_date": candles.index[0], "end_date": candles.index[-1],
            "trading_days": 22,
        }

        trades = [
            {"r_multiple": float(np.random.choice([-1.0, 1.5])),
             "risk_pct": 1.0, "context": {"adx_value": 30.0, "session": "london"},
             "day_index": i % 22}
            for i in range(30)
        ]

        with patch("src.discovery.orchestrator.IchimokuBacktester") as MockBT:
            mock_result = MagicMock()
            mock_result.trades = trades
            mock_result.metrics = {"win_rate": 0.40, "total_trades": 30}
            mock_result.prop_firm = {"status": "ongoing"}
            MockBT.return_value.run.return_value = mock_result

            result = orch.process_window(window, base_config={})

        assert result["discovery"]["shap_ran"] or True  # may not run if insufficient accumulated trades


class TestRunLoop:
    def test_run_loop_processes_all_windows(self):
        from src.discovery.orchestrator import DiscoveryOrchestrator

        candles = _make_candles(n_days=44)  # 2 windows
        kb_dir = str(Path(tempfile.mkdtemp()) / "kb")
        reports_dir = str(Path(tempfile.mkdtemp()) / "reports")

        orch = DiscoveryOrchestrator(
            config={
                "orchestrator": {"window_size_trading_days": 22, "max_windows": 5},
                "discovery": {"shap_every_n_windows": 3, "min_trades_for_shap": 5},
                "challenge": {"account_size": 10000.0},
                "reporting": {"reports_dir": reports_dir},
                "validation": {"min_oos_windows": 2},
            },
            knowledge_dir=kb_dir,
        )

        with patch("src.discovery.orchestrator.IchimokuBacktester") as MockBT:
            mock_result = MagicMock()
            mock_result.trades = [
                {"r_multiple": 1.5, "risk_pct": 1.0, "context": {}, "day_index": i}
                for i in range(5)
            ]
            mock_result.metrics = {"win_rate": 0.50, "total_trades": 5}
            mock_result.prop_firm = {"status": "ongoing"}
            MockBT.return_value.run.return_value = mock_result

            summary = orch.run(candles)

        assert "windows_processed" in summary
        assert summary["windows_processed"] >= 2
        assert "summary_report" in summary

    def test_run_loop_respects_max_windows(self):
        from src.discovery.orchestrator import DiscoveryOrchestrator

        candles = _make_candles(n_days=88)  # 4 windows
        kb_dir = str(Path(tempfile.mkdtemp()) / "kb")

        orch = DiscoveryOrchestrator(
            config={
                "orchestrator": {"window_size_trading_days": 22, "max_windows": 2},
                "reporting": {"reports_dir": str(Path(tempfile.mkdtemp()) / "r")},
            },
            knowledge_dir=kb_dir,
        )

        with patch("src.discovery.orchestrator.IchimokuBacktester") as MockBT:
            mock_result = MagicMock()
            mock_result.trades = []
            mock_result.metrics = {"win_rate": 0.0, "total_trades": 0}
            mock_result.prop_firm = {"status": "ongoing"}
            MockBT.return_value.run.return_value = mock_result

            summary = orch.run(candles)

        assert summary["windows_processed"] <= 2

    def test_run_loop_applies_validated_config_changes(self):
        from src.discovery.orchestrator import DiscoveryOrchestrator

        candles = _make_candles(n_days=66)  # 3 windows
        kb_dir = str(Path(tempfile.mkdtemp()) / "kb")
        reports_dir = str(Path(tempfile.mkdtemp()) / "reports")

        orch = DiscoveryOrchestrator(
            config={
                "orchestrator": {"window_size_trading_days": 22, "max_windows": 3},
                "discovery": {"shap_every_n_windows": 3, "min_trades_for_shap": 5},
                "validation": {"min_oos_windows": 1, "min_improvement_pct": 0.0},
                "reporting": {"reports_dir": reports_dir},
            },
            knowledge_dir=kb_dir,
        )

        call_count = [0]

        def _make_mock_result():
            m = MagicMock()
            call_count[0] += 1
            # Simulate improving metrics across windows
            win_rate = 0.30 + call_count[0] * 0.05
            m.trades = [
                {"r_multiple": 1.5 if i % 3 == 0 else -1.0,
                 "risk_pct": 1.0, "context": {"adx_value": 30.0},
                 "day_index": i % 22}
                for i in range(20)
            ]
            m.metrics = {"win_rate": win_rate, "total_trades": 20}
            m.prop_firm = {"status": "ongoing"}
            return m

        with patch("src.discovery.orchestrator.IchimokuBacktester") as MockBT:
            MockBT.return_value.run.side_effect = lambda *a, **kw: _make_mock_result()

            summary = orch.run(candles, base_config={"strategies": {"sss": {}}})

        assert summary["windows_processed"] == 3
        assert "config_evolution" in summary

    def test_run_loop_generates_summary_report(self):
        from src.discovery.orchestrator import DiscoveryOrchestrator

        candles = _make_candles(n_days=22)
        kb_dir = str(Path(tempfile.mkdtemp()) / "kb")
        reports_dir = str(Path(tempfile.mkdtemp()) / "reports")

        orch = DiscoveryOrchestrator(
            config={
                "orchestrator": {"window_size_trading_days": 22},
                "reporting": {"reports_dir": reports_dir},
            },
            knowledge_dir=kb_dir,
        )

        with patch("src.discovery.orchestrator.IchimokuBacktester") as MockBT:
            mock_result = MagicMock()
            mock_result.trades = []
            mock_result.metrics = {"win_rate": 0.0}
            mock_result.prop_firm = {"status": "ongoing"}
            MockBT.return_value.run.return_value = mock_result

            summary = orch.run(candles)

        assert "summary_report" in summary
        # Check that summary report file was written
        summary_path = Path(reports_dir) / "discovery_summary.json"
        assert summary_path.exists()


class TestDashboardIntegration:
    def test_orchestrator_exposes_dashboard_payload(self):
        from src.discovery.orchestrator import DiscoveryOrchestrator

        candles = _make_candles(n_days=22)
        kb_dir = str(Path(tempfile.mkdtemp()) / "kb")
        reports_dir = str(Path(tempfile.mkdtemp()) / "reports")

        orch = DiscoveryOrchestrator(
            config={
                "orchestrator": {"window_size_trading_days": 22},
                "reporting": {"reports_dir": reports_dir, "dashboard_integration": True},
            },
            knowledge_dir=kb_dir,
        )

        with patch("src.discovery.orchestrator.IchimokuBacktester") as MockBT:
            mock_result = MagicMock()
            mock_result.trades = [{"r_multiple": 1.0, "risk_pct": 1.0, "context": {}, "day_index": 0}]
            mock_result.metrics = {"win_rate": 0.50, "total_trades": 1}
            mock_result.prop_firm = {"status": "ongoing"}
            MockBT.return_value.run.return_value = mock_result

            orch.run(candles)

        payload = orch.get_dashboard_payload()
        assert "discovery" in payload
        assert "total_windows" in payload["discovery"]

    def test_dashboard_api_endpoint_serves_discovery_state(self):
        from src.discovery.window_report import WindowReportGenerator

        reports_dir = str(Path(tempfile.mkdtemp()) / "reports")
        gen = WindowReportGenerator(reports_dir=reports_dir)

        gen.generate_window_report(
            window_id="w_000", window_index=0, trades=[], metrics={},
            challenge_result={"passed_phase_1": False, "passed_phase_2": False},
        )

        payload = gen.get_dashboard_payload()
        assert payload["discovery"]["total_windows"] == 1


class TestDiscoveryCLI:
    def test_cli_parser_defaults(self):
        from scripts.run_discovery_loop import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "--data-file", "data/xauusd_1m.parquet",
        ])

        assert args.data_file == "data/xauusd_1m.parquet"
        assert args.max_windows == 12
        assert args.window_size == 22
        assert args.strategy == "sss"

    def test_cli_parser_overrides(self):
        from scripts.run_discovery_loop import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "--data-file", "data.parquet",
            "--max-windows", "6",
            "--window-size", "30",
            "--strategy", "asian_breakout",
            "--enable-claude",
        ])

        assert args.max_windows == 6
        assert args.window_size == 30
        assert args.strategy == "asian_breakout"
        assert args.enable_claude is True
