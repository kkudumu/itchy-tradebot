"""Tests for the discovery engine orchestrator."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


class TestDiscoveryRunner:
    def test_should_run_shap_respects_accumulation_window(self):
        from src.discovery.runner import DiscoveryRunner

        runner = DiscoveryRunner(
            shap_every_n_windows=3,
            knowledge_dir=str(Path(tempfile.mkdtemp()) / "kb"),
        )
        assert runner.should_run_shap(window_index=0) is False
        assert runner.should_run_shap(window_index=1) is False
        assert runner.should_run_shap(window_index=2) is True
        assert runner.should_run_shap(window_index=3) is False
        assert runner.should_run_shap(window_index=5) is True

    def test_accumulate_trades(self):
        from src.discovery.runner import DiscoveryRunner

        runner = DiscoveryRunner(
            knowledge_dir=str(Path(tempfile.mkdtemp()) / "kb"),
        )
        trades_w1 = [{"r_multiple": 1.0, "context": {}}]
        trades_w2 = [{"r_multiple": -0.5, "context": {}}]

        runner.accumulate_window("w_001", trades_w1)
        runner.accumulate_window("w_002", trades_w2)

        all_trades = runner.get_accumulated_trades()
        assert len(all_trades) == 2

    def test_analyze_returns_insight_when_enough_trades(self):
        from src.discovery.runner import DiscoveryRunner

        runner = DiscoveryRunner(
            min_trades_for_shap=5,
            knowledge_dir=str(Path(tempfile.mkdtemp()) / "kb"),
        )
        # Add enough trades
        rng = np.random.default_rng(42)
        trades = []
        for _ in range(20):
            trades.append({
                "r_multiple": float(rng.choice([-1.0, 1.5])),
                "context": {"adx_value": float(rng.uniform(10, 40)), "session": "london"},
            })

        runner.accumulate_window("w_001", trades)
        insight = runner.analyze(strategy_name="sss")
        assert insight is not None

    def test_analyze_returns_none_when_too_few_trades(self):
        from src.discovery.runner import DiscoveryRunner

        runner = DiscoveryRunner(
            min_trades_for_shap=50,
            knowledge_dir=str(Path(tempfile.mkdtemp()) / "kb"),
        )
        runner.accumulate_window("w_001", [{"r_multiple": 1.0, "context": {}}])
        insight = runner.analyze(strategy_name="sss")
        assert insight is None
