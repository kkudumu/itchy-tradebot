"""Tests for mega-vision eval harness + training data pipeline (Task 27)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.mega_vision.eval_harness import EvalReport, OfflineEvalHarness
from src.mega_vision.training_data import TrainingDataPipeline


UTC = timezone.utc


def _write_shadow_parquet(path: Path, rows: list) -> None:
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _row(
    ts: str,
    agreement: bool = True,
    confidence: float = 0.8,
    latency_ms: float = 100.0,
    picks: list | None = None,
    fallback_reason: str | None = None,
    cost_usd: float | None = None,
) -> dict:
    agent_picks = {
        "strategy_picks": picks or ["sss"],
        "confidence": confidence,
        "reasoning": "test",
        "fallback": fallback_reason is not None,
    }
    return {
        "ts_utc": ts,
        "candidate_signals_json": json.dumps([{"strategy_name": "sss"}]),
        "agent_picks_json": json.dumps(agent_picks),
        "native_picks_json": json.dumps([{"strategy_name": "sss"}]),
        "agreement_flag": agreement,
        "agent_confidence": confidence,
        "agent_reasoning": "test",
        "agent_latency_ms": latency_ms,
        "agent_cost_usd": cost_usd,
        "fallback_reason": fallback_reason,
    }


class TestOfflineEvalHarness:
    def test_missing_file_raises(self, tmp_path: Path):
        harness = OfflineEvalHarness(tmp_path / "does_not_exist.parquet")
        with pytest.raises(FileNotFoundError):
            harness.score()

    def test_empty_shadow_returns_zeros(self, tmp_path: Path):
        path = tmp_path / "shadow.parquet"
        _write_shadow_parquet(path, [])
        report = OfflineEvalHarness(path).score()
        assert report.total_decisions == 0
        assert report.agreement_count == 0

    def test_all_agreements(self, tmp_path: Path):
        path = tmp_path / "shadow.parquet"
        _write_shadow_parquet(
            path,
            [
                _row("2026-04-09T10:00:00+00:00", agreement=True),
                _row("2026-04-09T10:05:00+00:00", agreement=True),
                _row("2026-04-09T10:10:00+00:00", agreement=True),
            ],
        )
        report = OfflineEvalHarness(path).score()
        assert report.total_decisions == 3
        assert report.agreement_count == 3
        assert report.agreement_rate == 1.0

    def test_mixed_agreements(self, tmp_path: Path):
        path = tmp_path / "shadow.parquet"
        _write_shadow_parquet(
            path,
            [
                _row("2026-04-09T10:00:00+00:00", agreement=True),
                _row("2026-04-09T10:05:00+00:00", agreement=False),
            ],
        )
        report = OfflineEvalHarness(path).score()
        assert report.agreement_rate == 0.5

    def test_latency_stats(self, tmp_path: Path):
        path = tmp_path / "shadow.parquet"
        _write_shadow_parquet(
            path,
            [
                _row(f"2026-04-09T10:0{i}:00+00:00", latency_ms=float(100 + i * 10))
                for i in range(10)
            ],
        )
        report = OfflineEvalHarness(path).score()
        assert report.latency_mean_ms == pytest.approx(145.0)
        assert report.latency_median_ms == pytest.approx(145.0)

    def test_cost_stats(self, tmp_path: Path):
        path = tmp_path / "shadow.parquet"
        _write_shadow_parquet(
            path,
            [
                _row("2026-04-09T10:00:00+00:00", cost_usd=0.01),
                _row("2026-04-09T10:05:00+00:00", cost_usd=0.02),
            ],
        )
        report = OfflineEvalHarness(path).score()
        assert report.total_cost_usd == pytest.approx(0.03)
        assert report.cost_per_decision_usd == pytest.approx(0.015)

    def test_fallback_counted(self, tmp_path: Path):
        path = tmp_path / "shadow.parquet"
        _write_shadow_parquet(
            path,
            [
                _row("2026-04-09T10:00:00+00:00", fallback_reason="kill_switch"),
                _row("2026-04-09T10:05:00+00:00"),
            ],
        )
        report = OfflineEvalHarness(path).score()
        assert report.fallback_count == 1

    def test_per_strategy_override_frequency(self, tmp_path: Path):
        path = tmp_path / "shadow.parquet"
        _write_shadow_parquet(
            path,
            [
                _row("2026-04-09T10:00:00+00:00", picks=["sss"]),
                _row("2026-04-09T10:05:00+00:00", picks=["sss", "ichimoku"]),
                _row("2026-04-09T10:10:00+00:00", picks=["ichimoku"]),
            ],
        )
        report = OfflineEvalHarness(path).score()
        assert report.per_strategy_override_frequency["sss"] == 2
        assert report.per_strategy_override_frequency["ichimoku"] == 2

    def test_to_markdown_includes_headings(self, tmp_path: Path):
        path = tmp_path / "shadow.parquet"
        _write_shadow_parquet(
            path,
            [_row("2026-04-09T10:00:00+00:00")],
        )
        md = OfflineEvalHarness(path).to_markdown()
        assert "# Mega-Vision Offline Eval Report" in md
        assert "## Agreement" in md
        assert "## Latency" in md


class TestTrainingDataPipeline:
    def test_empty_shadow_writes_empty_parquet(self, tmp_path: Path):
        shadow = tmp_path / "shadow.parquet"
        _write_shadow_parquet(shadow, [])
        pipeline = TrainingDataPipeline()
        n = pipeline.build_dataset(
            shadow_parquet=shadow,
            trade_memory_parquet=None,
            output_dir=tmp_path / "out",
        )
        assert n == 0
        out_file = tmp_path / "out" / "examples.parquet"
        assert out_file.exists()

    def test_builds_examples_without_trade_memory(self, tmp_path: Path):
        shadow = tmp_path / "shadow.parquet"
        _write_shadow_parquet(
            shadow,
            [
                _row("2026-04-09T10:00:00+00:00"),
                _row("2026-04-09T10:05:00+00:00"),
            ],
        )
        pipeline = TrainingDataPipeline()
        n = pipeline.build_dataset(
            shadow_parquet=shadow,
            trade_memory_parquet=None,
            output_dir=tmp_path / "out",
        )
        assert n == 2

    def test_joins_with_trade_memory(self, tmp_path: Path):
        shadow = tmp_path / "shadow.parquet"
        _write_shadow_parquet(
            shadow,
            [_row("2026-04-09T10:00:00+00:00")],
        )
        # Trade that opened 30 minutes after the decision
        trade_memory = tmp_path / "trades.parquet"
        pd.DataFrame(
            [
                {
                    "trade_id": "t1",
                    "opened_at": "2026-04-09T10:30:00+00:00",
                    "closed_at": "2026-04-09T11:00:00+00:00",
                    "pnl_usd": 50.0,
                    "r_multiple": 1.0,
                }
            ]
        ).to_parquet(trade_memory)
        pipeline = TrainingDataPipeline()
        n = pipeline.build_dataset(
            shadow_parquet=shadow,
            trade_memory_parquet=trade_memory,
            output_dir=tmp_path / "out",
        )
        assert n == 1
        df = pd.read_parquet(tmp_path / "out" / "examples.parquet")
        outcome = json.loads(df.iloc[0]["outcome_json"])
        assert outcome["trade_id"] == "t1"
        assert outcome["pnl_usd"] == 50.0
