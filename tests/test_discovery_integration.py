"""Integration test: full discovery cycle from trades to rules."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


class TestFullDiscoveryCycle:
    def _generate_trades(self, n=100, seed=42):
        """Generate realistic-ish trade data."""
        rng = np.random.default_rng(seed)
        trades = []
        for _ in range(n):
            win = rng.random() < 0.37
            r = float(rng.uniform(0.5, 3.0) if win else rng.uniform(-2.0, -0.1))
            trades.append({
                "r_multiple": r,
                "context": {
                    "cloud_direction_4h": float(rng.choice([0.0, 0.5, 1.0])),
                    "adx_value": float(rng.uniform(10, 50)),
                    "atr_value": float(rng.uniform(1, 10)),
                    "session": rng.choice(["london", "new_york", "asian"]),
                    "confluence_score": int(rng.integers(1, 8)),
                },
            })
        return trades

    def test_full_cycle_produces_insight_and_rules(self):
        from src.discovery.runner import DiscoveryRunner

        kb_dir = str(Path(tempfile.mkdtemp()) / "kb")
        runner = DiscoveryRunner(
            shap_every_n_windows=3,
            min_trades_for_shap=20,
            knowledge_dir=kb_dir,
        )

        # Simulate 3 windows of trading
        for i in range(3):
            trades = self._generate_trades(n=40, seed=42 + i)
            result = runner.run_full_cycle(
                window_id=f"w_{i:03d}",
                window_index=i,
                trades=trades,
                strategy_name="sss",
                base_config={"strategies": {"sss": {"min_confluence_score": 4}}},
                enable_claude=False,  # skip actual CLI call in tests
            )

        # Window 2 (index=2) should trigger SHAP
        assert result["shap_ran"] is True
        assert result["insight"] is not None
        assert len(result["insight"].feature_importance) > 0

    def test_knowledge_base_persists_across_windows(self):
        from src.discovery.runner import DiscoveryRunner
        from src.discovery.knowledge_base import KnowledgeBase

        kb_dir = str(Path(tempfile.mkdtemp()) / "kb")
        runner = DiscoveryRunner(
            shap_every_n_windows=3,
            min_trades_for_shap=20,
            knowledge_dir=kb_dir,
        )

        for i in range(3):
            trades = self._generate_trades(n=30, seed=i)
            runner.accumulate_window(f"w_{i:03d}", trades)

        kb = KnowledgeBase(base_dir=kb_dir)
        all_trades = kb.get_accumulated_trades()
        assert len(all_trades) == 90  # 3 windows * 30 trades
