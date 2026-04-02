# tests/test_discovery_integration_e2e.py
"""End-to-end integration test for the full discovery loop.

Tests the complete orchestrator with synthetic data, mocked backtester,
and real memory/reporting/validation components.
"""

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def _make_candles(n_days: int = 66, bars_per_day: int = 100) -> pd.DataFrame:
    """Generate synthetic 1M candles spanning n_days trading days."""
    rng = np.random.default_rng(42)
    start = datetime(2025, 1, 2, 8, 0, tzinfo=timezone.utc)

    timestamps = []
    day = 0
    dt = start
    while day < n_days:
        if dt.weekday() < 5:
            for bar in range(bars_per_day):
                timestamps.append(dt + timedelta(minutes=bar))
            day += 1
        dt += timedelta(days=1)

    n = len(timestamps)
    prices = 1800.0 + np.cumsum(rng.normal(0.01, 0.5, n))
    return pd.DataFrame({
        "open": prices + rng.uniform(-0.2, 0.2, n),
        "high": prices + rng.uniform(0, 0.5, n),
        "low": prices - rng.uniform(0, 0.5, n),
        "close": prices,
        "volume": rng.integers(100, 800, n),
    }, index=pd.DatetimeIndex(timestamps, tz=timezone.utc))


def _make_mock_bt_result(window_index: int, rng) -> MagicMock:
    """Create a mock BacktestResult with realistic trade data."""
    n_trades = rng.integers(5, 25)
    win_rate = 0.30 + window_index * 0.03  # slightly improving
    trades = []
    for i in range(n_trades):
        is_win = rng.random() < win_rate
        r = float(rng.uniform(0.5, 3.0) if is_win else rng.uniform(-2.0, -0.1))
        trades.append({
            "r_multiple": r,
            "risk_pct": 1.0,
            "day_index": i % 22,
            "context": {
                "adx_value": float(rng.uniform(15, 45)),
                "session": rng.choice(["london", "new_york", "asian"]),
                "confluence_score": int(rng.integers(1, 8)),
                "cloud_direction_4h": float(rng.choice([0.0, 0.5, 1.0])),
            },
        })

    result = MagicMock()
    result.trades = trades
    result.metrics = {
        "win_rate": win_rate,
        "total_trades": n_trades,
        "sharpe_ratio": float(rng.uniform(0.5, 2.0)),
        "max_drawdown_pct": float(rng.uniform(2.0, 8.0)),
        "expectancy": float(rng.uniform(-0.3, 0.5)),
    }
    result.prop_firm = {"status": "ongoing", "profit_pct": float(rng.uniform(-3.0, 5.0))}
    return result


class TestEndToEndDiscoveryLoop:
    """Full end-to-end test: orchestrator with all real components except backtester."""

    def test_full_loop_three_windows(self):
        from src.discovery.orchestrator import DiscoveryOrchestrator

        candles = _make_candles(n_days=66)  # 3 windows of 22 days
        tmp = Path(tempfile.mkdtemp())
        kb_dir = str(tmp / "kb")
        reports_dir = str(tmp / "reports")

        orch = DiscoveryOrchestrator(
            config={
                "orchestrator": {
                    "window_size_trading_days": 22,
                    "max_windows": 3,
                    "strategy_name": "sss",
                },
                "discovery": {
                    "shap_every_n_windows": 3,
                    "min_trades_for_shap": 10,
                },
                "challenge": {
                    "account_size": 10000.0,
                    "phase_1_target_pct": 8.0,
                    "phase_2_target_pct": 5.0,
                },
                "validation": {
                    "min_oos_windows": 2,
                    "min_improvement_pct": 1.0,
                },
                "reporting": {
                    "reports_dir": reports_dir,
                },
            },
            knowledge_dir=kb_dir,
        )

        rng = np.random.default_rng(42)
        call_count = [0]

        def mock_run(*args, **kwargs):
            result = _make_mock_bt_result(call_count[0], rng)
            call_count[0] += 1
            return result

        with patch("src.discovery.orchestrator.IchimokuBacktester") as MockBT:
            MockBT.return_value.run.side_effect = mock_run

            summary = orch.run(
                candles=candles,
                base_config={"strategies": {"sss": {"min_confluence_score": 4}}},
                enable_claude=False,
            )

        # Verify core outputs
        assert summary["windows_processed"] == 3
        assert "summary_report" in summary
        assert "config_evolution" in summary
        assert "per_window_results" in summary

        # Verify reports were written
        reports_path = Path(reports_dir)
        assert (reports_path / "discovery_summary.json").exists()

        window_reports = list(reports_path.glob("window_*.json"))
        assert len(window_reports) == 3

        # Verify summary report content
        summary_data = json.loads(
            (reports_path / "discovery_summary.json").read_text(encoding="utf-8")
        )
        assert summary_data["total_windows"] == 3

        # Verify memory layer was populated
        kb_path = Path(kb_dir)
        assert kb_path.exists()

    def test_full_loop_with_layered_memory(self):
        from src.discovery.orchestrator import DiscoveryOrchestrator
        from src.discovery.memory import LayeredMemory

        candles = _make_candles(n_days=44)  # 2 windows
        tmp = Path(tempfile.mkdtemp())
        kb_dir = str(tmp / "kb")

        orch = DiscoveryOrchestrator(
            config={
                "orchestrator": {"window_size_trading_days": 22, "max_windows": 2},
                "discovery": {"shap_every_n_windows": 3, "min_trades_for_shap": 5},
                "reporting": {"reports_dir": str(tmp / "reports")},
            },
            knowledge_dir=kb_dir,
        )

        rng = np.random.default_rng(99)
        call_count = [0]

        def mock_run(*args, **kwargs):
            result = _make_mock_bt_result(call_count[0], rng)
            call_count[0] += 1
            return result

        with patch("src.discovery.orchestrator.IchimokuBacktester") as MockBT:
            MockBT.return_value.run.side_effect = mock_run
            orch.run(candles, base_config={}, enable_claude=False)

        # Check that short-term memory is populated
        mem = orch._memory
        # At least the last window should be in short-term
        assert len(mem.get_recent_contexts()) >= 1

    def test_dashboard_payload_after_loop(self):
        from src.discovery.orchestrator import DiscoveryOrchestrator

        candles = _make_candles(n_days=22)
        tmp = Path(tempfile.mkdtemp())

        orch = DiscoveryOrchestrator(
            config={
                "orchestrator": {"window_size_trading_days": 22},
                "reporting": {"reports_dir": str(tmp / "reports")},
            },
            knowledge_dir=str(tmp / "kb"),
        )

        with patch("src.discovery.orchestrator.IchimokuBacktester") as MockBT:
            mock_result = MagicMock()
            mock_result.trades = [{"r_multiple": 1.0, "risk_pct": 1.0, "context": {}, "day_index": 0}]
            mock_result.metrics = {"win_rate": 1.0, "total_trades": 1}
            mock_result.prop_firm = {"status": "passed"}
            MockBT.return_value.run.return_value = mock_result

            orch.run(candles, base_config={})

        payload = orch.get_dashboard_payload()
        assert "discovery" in payload
        assert payload["discovery"]["total_windows"] == 1
