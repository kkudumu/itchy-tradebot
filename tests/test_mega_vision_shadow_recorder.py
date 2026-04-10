"""Tests for mega-vision shadow recorder (plan Task 26)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.mega_vision.shadow_recorder import ShadowRecorder


UTC = timezone.utc


@dataclass
class FakeSignal:
    strategy_name: str
    direction: str = "long"
    entry_price: float = 2000.0
    stop_loss: float = 1995.0
    take_profit: float = 2010.0
    confluence_score: float = 3.0


def _ts():
    return datetime(2026, 4, 9, 14, 0, tzinfo=UTC)


class TestShadowRecorder:
    def test_empty_flush_writes_valid_parquet(self, tmp_path: Path):
        rec = ShadowRecorder(run_id="empty", out_dir=tmp_path)
        rec.flush_to_parquet()
        out = tmp_path / "mega_vision_shadow.parquet"
        assert out.exists()
        df = pd.read_parquet(out)
        assert len(df) == 0

    def test_record_single_decision(self, tmp_path: Path):
        rec = ShadowRecorder(run_id="test", out_dir=tmp_path)
        candidates = [FakeSignal("sss"), FakeSignal("ichimoku")]
        agent_pick = {
            "strategy_picks": ["sss"],
            "confidence": 0.8,
            "reasoning": "regime matches sss pattern",
            "fallback": False,
        }
        native_picks = [FakeSignal("sss")]
        rec.record(
            _ts(),
            candidates,
            agent_pick,
            native_picks,
            latency_ms=150.0,
            cost_usd=0.02,
        )
        assert rec.record_count == 1

    def test_agreement_flag_true_when_sets_match(self, tmp_path: Path):
        rec = ShadowRecorder(run_id="agree", out_dir=tmp_path)
        rec.record(
            _ts(),
            [FakeSignal("sss")],
            {"strategy_picks": ["sss"], "confidence": 0.9, "reasoning": ""},
            [FakeSignal("sss")],
        )
        rec.flush_to_parquet()
        df = pd.read_parquet(tmp_path / "mega_vision_shadow.parquet")
        assert bool(df.iloc[0]["agreement_flag"]) is True

    def test_agreement_flag_false_when_sets_differ(self, tmp_path: Path):
        rec = ShadowRecorder(run_id="disagree", out_dir=tmp_path)
        rec.record(
            _ts(),
            [FakeSignal("sss"), FakeSignal("ichimoku")],
            {"strategy_picks": ["sss"], "confidence": 0.7, "reasoning": ""},
            [FakeSignal("sss"), FakeSignal("ichimoku")],
        )
        rec.flush_to_parquet()
        df = pd.read_parquet(tmp_path / "mega_vision_shadow.parquet")
        assert bool(df.iloc[0]["agreement_flag"]) is False

    def test_fallback_reason_populated(self, tmp_path: Path):
        rec = ShadowRecorder(run_id="fb", out_dir=tmp_path)
        rec.record(
            _ts(),
            [FakeSignal("sss")],
            {
                "strategy_picks": None,
                "confidence": 0.0,
                "reasoning": "kill_switch",
                "fallback": True,
            },
            [FakeSignal("sss")],
        )
        rec.flush_to_parquet()
        df = pd.read_parquet(tmp_path / "mega_vision_shadow.parquet")
        assert df.iloc[0]["fallback_reason"] == "kill_switch"

    def test_none_pick_handled(self, tmp_path: Path):
        rec = ShadowRecorder(run_id="none", out_dir=tmp_path)
        rec.record(_ts(), [FakeSignal("sss")], None, [FakeSignal("sss")])
        rec.flush_to_parquet()
        df = pd.read_parquet(tmp_path / "mega_vision_shadow.parquet")
        assert len(df) == 1

    def test_multiple_records(self, tmp_path: Path):
        rec = ShadowRecorder(run_id="multi", out_dir=tmp_path)
        for i in range(5):
            rec.record(
                _ts(),
                [FakeSignal("sss")],
                {"strategy_picks": ["sss"], "confidence": 0.5, "reasoning": str(i)},
                [FakeSignal("sss")],
            )
        rec.flush_to_parquet()
        df = pd.read_parquet(tmp_path / "mega_vision_shadow.parquet")
        assert len(df) == 5

    def test_clear(self, tmp_path: Path):
        rec = ShadowRecorder(run_id="clear", out_dir=tmp_path)
        rec.record(_ts(), [], None, [])
        assert rec.record_count == 1
        rec.clear()
        assert rec.record_count == 0
