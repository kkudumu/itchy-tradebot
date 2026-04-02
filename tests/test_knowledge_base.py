"""Tests for the discovery agent JSON knowledge base."""

import json
import tempfile
from pathlib import Path

import pytest


class TestKnowledgeBase:
    def _make_kb(self, tmp_path):
        from src.discovery.knowledge_base import KnowledgeBase
        return KnowledgeBase(base_dir=str(tmp_path / "agent_knowledge"))

    def test_save_and_load_hypothesis(self, tmp_path):
        kb = self._make_kb(tmp_path)
        hyp = {
            "id": "hyp_001",
            "description": "London session + ADX > 30 improves win rate",
            "source": "shap_interaction",
            "strategy": "sss",
            "status": "proposed",
            "evidence": {"feature_a": "sess_london", "feature_b": "adx_trending", "lift": 1.35},
        }
        kb.save_hypothesis(hyp)
        loaded = kb.load_hypothesis("hyp_001")
        assert loaded["description"] == hyp["description"]
        assert loaded["status"] == "proposed"

    def test_update_hypothesis_status(self, tmp_path):
        kb = self._make_kb(tmp_path)
        hyp = {"id": "hyp_002", "description": "test", "status": "proposed"}
        kb.save_hypothesis(hyp)
        kb.update_hypothesis_status("hyp_002", "validated", metrics={"win_rate_delta": 0.08})
        loaded = kb.load_hypothesis("hyp_002")
        assert loaded["status"] == "validated"
        assert loaded["validation_metrics"]["win_rate_delta"] == 0.08

    def test_save_and_load_shap_rules(self, tmp_path):
        kb = self._make_kb(tmp_path)
        rules = [
            {"condition": "adx>0.5 AND sess_london>=0.5", "lift": 1.3, "recommendation": "strong_filter"},
        ]
        kb.save_shap_rules(rules, window_id="w_001")
        loaded = kb.load_shap_rules("w_001")
        assert len(loaded) == 1
        assert loaded[0]["lift"] == 1.3

    def test_list_hypotheses_by_status(self, tmp_path):
        kb = self._make_kb(tmp_path)
        kb.save_hypothesis({"id": "h1", "status": "proposed"})
        kb.save_hypothesis({"id": "h2", "status": "validated"})
        kb.save_hypothesis({"id": "h3", "status": "proposed"})

        proposed = kb.list_hypotheses(status="proposed")
        assert len(proposed) == 2
        validated = kb.list_hypotheses(status="validated")
        assert len(validated) == 1

    def test_get_accumulated_trades(self, tmp_path):
        kb = self._make_kb(tmp_path)
        trades_w1 = [{"r_multiple": 1.0}, {"r_multiple": -0.5}]
        trades_w2 = [{"r_multiple": 2.0}]
        kb.save_window_trades(trades_w1, window_id="w_001")
        kb.save_window_trades(trades_w2, window_id="w_002")
        all_trades = kb.get_accumulated_trades()
        assert len(all_trades) == 3
