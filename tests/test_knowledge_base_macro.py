# tests/test_knowledge_base_macro.py
"""Tests for regime finding persistence in the knowledge base."""

import json
import pytest
from pathlib import Path


class TestRegimePersistence:
    def _make_kb(self, tmp_path):
        from src.discovery.knowledge_base import KnowledgeBase
        return KnowledgeBase(base_dir=str(tmp_path / "agent_knowledge"))

    def test_save_and_load_regime_stats(self, tmp_path):
        kb = self._make_kb(tmp_path)
        stats = {
            "risk_on": {"count": 15, "win_rate": 0.53, "avg_r": 0.42},
            "risk_off": {"count": 8, "win_rate": 0.25, "avg_r": -0.31},
            "dollar_driven": {"count": 12, "win_rate": 0.33, "avg_r": -0.05},
            "inflation_fear": {"count": 5, "win_rate": 0.60, "avg_r": 0.85},
            "mixed": {"count": 20, "win_rate": 0.35, "avg_r": 0.10},
        }
        kb.save_regime_stats(stats, window_id="w_003")
        loaded = kb.load_regime_stats("w_003")

        assert loaded["risk_on"]["win_rate"] == 0.53
        assert loaded["inflation_fear"]["count"] == 5

    def test_save_and_load_macro_panel_summary(self, tmp_path):
        kb = self._make_kb(tmp_path)
        summary = {
            "window_id": "w_003",
            "date_range": "2025-01-06 to 2025-01-31",
            "dxy_mean": 103.5,
            "dxy_std": 0.8,
            "regime_distribution": {
                "risk_on": 5,
                "risk_off": 3,
                "mixed": 12,
            },
            "events_in_window": 3,
        }
        kb.save_macro_summary(summary, window_id="w_003")
        loaded = kb.load_macro_summary("w_003")

        assert loaded["dxy_mean"] == 103.5
        assert loaded["events_in_window"] == 3

    def test_list_regime_stats_across_windows(self, tmp_path):
        kb = self._make_kb(tmp_path)
        kb.save_regime_stats({"risk_on": {"count": 10}}, window_id="w_001")
        kb.save_regime_stats({"risk_off": {"count": 5}}, window_id="w_002")

        all_stats = kb.list_regime_stats()
        assert len(all_stats) == 2
