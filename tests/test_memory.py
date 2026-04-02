# tests/test_memory.py
"""Tests for the three-tier layered memory system."""

import json
import tempfile
from pathlib import Path
from typing import Dict

import pytest
import yaml


class TestShortTermMemory:
    def test_store_and_retrieve_window_context(self):
        from src.discovery.memory import LayeredMemory

        mem = LayeredMemory(knowledge_dir=str(Path(tempfile.mkdtemp()) / "kb"))
        mem.store_short_term("w_001", {
            "trades": [{"r_multiple": 1.5}],
            "metrics": {"win_rate": 0.45},
            "regime": "trending_bullish",
        })

        ctx = mem.get_short_term("w_001")
        assert ctx is not None
        assert ctx["metrics"]["win_rate"] == 0.45

    def test_short_term_cleared_on_new_window(self):
        from src.discovery.memory import LayeredMemory

        mem = LayeredMemory(knowledge_dir=str(Path(tempfile.mkdtemp()) / "kb"))
        mem.store_short_term("w_001", {"trades": [{"r_multiple": 1.0}]})
        mem.store_short_term("w_002", {"trades": [{"r_multiple": 2.0}]})

        # Short-term only keeps current + previous window
        assert mem.get_short_term("w_002") is not None
        assert mem.get_short_term("w_001") is not None  # previous still accessible

    def test_short_term_max_retention(self):
        from src.discovery.memory import LayeredMemory

        mem = LayeredMemory(
            knowledge_dir=str(Path(tempfile.mkdtemp()) / "kb"),
            short_term_max_windows=2,
        )
        mem.store_short_term("w_001", {"data": 1})
        mem.store_short_term("w_002", {"data": 2})
        mem.store_short_term("w_003", {"data": 3})

        assert mem.get_short_term("w_001") is None  # evicted
        assert mem.get_short_term("w_002") is not None
        assert mem.get_short_term("w_003") is not None


class TestWorkingMemory:
    def test_save_and_load_pattern(self):
        from src.discovery.memory import LayeredMemory

        kb_dir = str(Path(tempfile.mkdtemp()) / "kb")
        mem = LayeredMemory(knowledge_dir=kb_dir)
        pattern = {
            "id": "pat_001",
            "type": "shap_interaction",
            "features": ["adx_value", "sess_london"],
            "lift": 1.35,
            "windows_seen": ["w_001", "w_004"],
            "status": "candidate",
        }
        mem.save_working_pattern(pattern)
        loaded = mem.load_working_pattern("pat_001")

        assert loaded is not None
        assert loaded["lift"] == 1.35
        assert loaded["status"] == "candidate"

    def test_list_patterns_by_status(self):
        from src.discovery.memory import LayeredMemory

        kb_dir = str(Path(tempfile.mkdtemp()) / "kb")
        mem = LayeredMemory(knowledge_dir=kb_dir)
        mem.save_working_pattern({"id": "p1", "status": "candidate"})
        mem.save_working_pattern({"id": "p2", "status": "validated"})
        mem.save_working_pattern({"id": "p3", "status": "candidate"})

        candidates = mem.list_working_patterns(status="candidate")
        assert len(candidates) == 2
        validated = mem.list_working_patterns(status="validated")
        assert len(validated) == 1

    def test_promote_pattern_to_validated(self):
        from src.discovery.memory import LayeredMemory

        kb_dir = str(Path(tempfile.mkdtemp()) / "kb")
        mem = LayeredMemory(knowledge_dir=kb_dir)
        mem.save_working_pattern({"id": "p1", "status": "candidate", "lift": 1.2})
        mem.promote_pattern("p1", "validated", oos_results={"windows_passed": 3})

        p = mem.load_working_pattern("p1")
        assert p["status"] == "validated"
        assert p["oos_results"]["windows_passed"] == 3


class TestLongTermMemory:
    def test_absorb_edge_into_yaml(self):
        from src.discovery.memory import LayeredMemory

        tmp = Path(tempfile.mkdtemp())
        edges_path = tmp / "edges.yaml"
        edges_path.write_text(yaml.dump({
            "regime_filter": {"enabled": False, "params": {"adx_min": 28}},
            "strategy_profiles": {"sss": {}},
        }), encoding="utf-8")

        mem = LayeredMemory(
            knowledge_dir=str(tmp / "kb"),
            edges_yaml_path=str(edges_path),
        )
        mem.absorb_edge(
            edge_name="regime_filter",
            params={"adx_min": 25},
            source_pattern_id="pat_001",
        )

        reloaded = yaml.safe_load(edges_path.read_text(encoding="utf-8"))
        assert reloaded["regime_filter"]["enabled"] is True
        assert reloaded["regime_filter"]["params"]["adx_min"] == 25

    def test_absorb_records_provenance(self):
        from src.discovery.memory import LayeredMemory

        tmp = Path(tempfile.mkdtemp())
        edges_path = tmp / "edges.yaml"
        edges_path.write_text(yaml.dump({
            "regime_filter": {"enabled": False, "params": {}},
        }), encoding="utf-8")

        mem = LayeredMemory(
            knowledge_dir=str(tmp / "kb"),
            edges_yaml_path=str(edges_path),
        )
        mem.absorb_edge(
            edge_name="regime_filter",
            params={"adx_min": 25},
            source_pattern_id="pat_001",
        )

        absorption_log = mem.get_absorption_log()
        assert len(absorption_log) == 1
        assert absorption_log[0]["edge_name"] == "regime_filter"
        assert absorption_log[0]["source_pattern_id"] == "pat_001"

    def test_revert_absorbed_edge(self):
        from src.discovery.memory import LayeredMemory

        tmp = Path(tempfile.mkdtemp())
        edges_path = tmp / "edges.yaml"
        original = {
            "regime_filter": {"enabled": False, "params": {"adx_min": 28}},
        }
        edges_path.write_text(yaml.dump(original), encoding="utf-8")

        mem = LayeredMemory(
            knowledge_dir=str(tmp / "kb"),
            edges_yaml_path=str(edges_path),
        )
        mem.absorb_edge("regime_filter", {"adx_min": 25}, "pat_001")
        mem.revert_absorption("regime_filter")

        reloaded = yaml.safe_load(edges_path.read_text(encoding="utf-8"))
        assert reloaded["regime_filter"]["enabled"] is False
        assert reloaded["regime_filter"]["params"]["adx_min"] == 28
