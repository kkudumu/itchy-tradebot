"""Tests for the futures retuning workflow (plan Task 17)."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.backtesting.strategy_telemetry import StrategyTelemetryCollector
from src.optimization.futures_retuning import (
    RetuningReport,
    analyze_telemetry,
    apply_overrides_to_profile,
    suggest_param_adjustments,
    write_report,
)


UTC = timezone.utc


def _ts(hour: int = 10, day: int = 15) -> datetime:
    return datetime(2026, 1, day, hour, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# analyze_telemetry
# ---------------------------------------------------------------------------


class TestAnalyzeTelemetry:
    def _make_telemetry(self, tmp_path: Path) -> Path:
        c = StrategyTelemetryCollector(run_id="retune_test")
        # sss: 10 generated, 1 entered (10% entry rate)
        for i in range(10):
            c.emit_signal_generated(
                _ts(10 + i % 5),
                "sss",
                confluence_score=2 + i % 3,
                atr=1.5,
                planned_stop_pips=3.0 + i * 0.5,
                pattern_type="cbc",
            )
        c.emit_entry(_ts(14), "sss", planned_size=1.0)
        # 1 rejection on edge.spread
        c.emit_filter_rejection(_ts(11), "sss", "edge.spread", "wide")

        # asian_breakout: 2 generated, 0 entered
        for i in range(2):
            c.emit_signal_generated(_ts(7 + i), "asian_breakout", confluence_score=1)

        out = tmp_path / "retune_test.parquet"
        c.to_parquet(out)
        return out

    def test_empty_parquet_returns_empty_report(self, tmp_path: Path) -> None:
        c = StrategyTelemetryCollector(run_id="empty")
        out = tmp_path / "empty.parquet"
        c.to_parquet(out)
        report = analyze_telemetry(out)
        assert report.total_events == 0
        assert report.per_strategy == {}

    def test_populated_report_has_per_strategy_stats(self, tmp_path: Path) -> None:
        parquet = self._make_telemetry(tmp_path)
        report = analyze_telemetry(parquet)

        assert report.total_events == 14  # 10 sss gen + 1 entered + 1 rej + 2 ab gen
        assert "sss" in report.per_strategy
        assert "asian_breakout" in report.per_strategy

        sss = report.per_strategy["sss"]
        assert sss.signals_generated == 10
        assert sss.signals_entered == 1
        assert sss.entry_rate_pct == pytest.approx(10.0)
        # Confluence scores emitted: 2, 3, 4, 2, 3, 4, 2, 3, 4, 2
        assert sss.confluence_score_mean == pytest.approx(2.9, abs=0.1)

    def test_rejection_stages_captured(self, tmp_path: Path) -> None:
        parquet = self._make_telemetry(tmp_path)
        report = analyze_telemetry(parquet)
        sss = report.per_strategy["sss"]
        assert "edge.spread" in sss.top_rejection_stages

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            analyze_telemetry(tmp_path / "does_not_exist.parquet")


# ---------------------------------------------------------------------------
# suggest_param_adjustments
# ---------------------------------------------------------------------------


class TestSuggestParamAdjustments:
    def test_low_entry_rate_suggests_loosening(self) -> None:
        report = RetuningReport(
            per_strategy={
                "sss": _stats("sss", gen=100, ent=0),
                "ichimoku": _stats("ichimoku", gen=100, ent=0),
                "asian_breakout": _stats("asian_breakout", gen=100, ent=0),
                "ema_pullback": _stats("ema_pullback", gen=100, ent=0),
            }
        )
        sugg = suggest_param_adjustments(report)
        assert "sss" in sugg
        assert sugg["sss"]["min_swing_pips"] == 0.3
        assert sugg["sss"]["min_confluence_score"] == 0
        assert sugg["ichimoku"]["signal"]["min_confluence_score"] == 0
        assert sugg["asian_breakout"]["min_range_pips"] == 1
        assert sugg["ema_pullback"]["min_ema_angle_deg"] == 0.5

    def test_high_entry_rate_suggests_tightening(self) -> None:
        report = RetuningReport(
            per_strategy={
                "sss": _stats("sss", gen=600, ent=50),  # 8.3% entry rate
            }
        )
        sugg = suggest_param_adjustments(report)
        assert sugg["sss"]["min_swing_pips"] == 1.0

    def test_small_planned_stop_on_sss_bumps_min_stop(self) -> None:
        report = RetuningReport(
            per_strategy={
                "sss": _stats("sss", gen=50, ent=1, planned_stop_pips_mean=3.0),
            }
        )
        sugg = suggest_param_adjustments(report)
        assert sugg["sss"]["min_stop_pips"] == 10.0

    def test_insufficient_data_returns_empty(self) -> None:
        report = RetuningReport(
            per_strategy={"sss": _stats("sss", gen=3, ent=0)}
        )
        sugg = suggest_param_adjustments(report)
        assert sugg == {}


def _stats(
    name: str,
    gen: int = 0,
    ent: int = 0,
    planned_stop_pips_mean: float | None = None,
):
    from src.optimization.futures_retuning import StrategyTelemetryStats

    return StrategyTelemetryStats(
        strategy_name=name,
        signals_generated=gen,
        signals_entered=ent,
        entry_rate_pct=(ent / gen * 100.0) if gen else 0.0,
        planned_stop_pips_mean=planned_stop_pips_mean,
    )


# ---------------------------------------------------------------------------
# apply_overrides_to_profile
# ---------------------------------------------------------------------------


class TestApplyOverridesToProfile:
    def test_creates_file_when_missing(self, tmp_path: Path) -> None:
        profile = tmp_path / "futures.yaml"
        apply_overrides_to_profile(
            {"sss": {"min_swing_pips": 2.0}},
            profile,
        )
        assert profile.exists()
        import yaml

        data = yaml.safe_load(profile.read_text())
        assert data["strategy_overrides"]["sss"]["min_swing_pips"] == 2.0

    def test_merges_with_existing(self, tmp_path: Path) -> None:
        profile = tmp_path / "futures.yaml"
        profile.write_text(
            "strategy_overrides:\n  sss:\n    min_swing_pips: 1.0\n    ss_candle_min: 5\n",
            encoding="utf-8",
        )
        apply_overrides_to_profile(
            {"sss": {"min_swing_pips": 2.0, "iss_candle_min": 4}},
            profile,
        )
        import yaml

        data = yaml.safe_load(profile.read_text())
        sss = data["strategy_overrides"]["sss"]
        assert sss["min_swing_pips"] == 2.0  # updated
        assert sss["ss_candle_min"] == 5  # preserved
        assert sss["iss_candle_min"] == 4  # added

    def test_preserves_unrelated_top_level_keys(self, tmp_path: Path) -> None:
        profile = tmp_path / "futures.yaml"
        profile.write_text(
            "instrument_class: futures\ndaily_reset_hour: 17\n",
            encoding="utf-8",
        )
        apply_overrides_to_profile(
            {"sss": {"min_swing_pips": 2.0}},
            profile,
        )
        import yaml

        data = yaml.safe_load(profile.read_text())
        assert data["instrument_class"] == "futures"
        assert data["daily_reset_hour"] == 17
        assert "strategy_overrides" in data


class TestWriteReport:
    def test_writes_json(self, tmp_path: Path) -> None:
        report = RetuningReport(
            total_events=10,
            source_parquet="foo.parquet",
            per_strategy={
                "sss": _stats("sss", gen=10, ent=1),
            },
        )
        out = tmp_path / "report.json"
        write_report(report, out)
        data = json.loads(out.read_text())
        assert data["total_events"] == 10
        assert "sss" in data["per_strategy"]
