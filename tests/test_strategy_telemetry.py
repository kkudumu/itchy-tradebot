"""Tests for the strategy telemetry collector (plan Task 7)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.backtesting.strategy_telemetry import (
    StrategyTelemetryCollector,
    TelemetryEvent,
    classify_session,
    ts_to_event_fields,
)


UTC = timezone.utc


def _ts(hour: int, weekday_day: int = 15) -> datetime:
    """Build a UTC timestamp with a given hour on 2026-01-15 (Thursday)."""
    return datetime(2026, 1, weekday_day, hour, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# classify_session
# ---------------------------------------------------------------------------


class TestClassifySession:
    def test_asian_hours(self) -> None:
        for h in (0, 3, 6):
            assert classify_session(h) == "asian"

    def test_london_hours(self) -> None:
        for h in (7, 9, 11):
            assert classify_session(h) == "london"

    def test_overlap_hours(self) -> None:
        for h in (12, 14, 15):
            assert classify_session(h) == "overlap"

    def test_ny_hours(self) -> None:
        for h in (16, 18, 20):
            assert classify_session(h) == "ny"

    def test_off_hours(self) -> None:
        for h in (21, 22, 23):
            assert classify_session(h) == "off"


class TestTsToEventFields:
    def test_extracts_hour_and_session(self) -> None:
        ts = _ts(14)
        fields = ts_to_event_fields(ts)
        assert fields["hour_of_day_utc"] == 14
        assert fields["session"] == "overlap"

    def test_day_of_week_is_weekday(self) -> None:
        # 2026-01-15 is a Thursday → weekday() returns 3
        fields = ts_to_event_fields(_ts(10, weekday_day=15))
        assert fields["day_of_week"] == 3


# ---------------------------------------------------------------------------
# StrategyTelemetryCollector
# ---------------------------------------------------------------------------


class TestCollectorEmit:
    def test_empty_collector_has_zero_events(self) -> None:
        c = StrategyTelemetryCollector(run_id="test_empty")
        assert c.event_count == 0

    def test_emit_signal_generated(self) -> None:
        c = StrategyTelemetryCollector(run_id="t1")
        c.emit_signal_generated(
            _ts(14),
            "sss",
            direction="long",
            price=2000.5,
            atr=1.5,
            confluence_score=3,
            pattern_type="cbc",
        )
        assert c.event_count == 1
        evt = c.events()[0]
        assert evt.strategy_name == "sss"
        assert evt.event_type == "signal_generated"
        assert evt.direction == "long"
        assert evt.pattern_type == "cbc"
        assert evt.session == "overlap"
        assert evt.hour_of_day_utc == 14

    def test_emit_filter_rejection(self) -> None:
        c = StrategyTelemetryCollector(run_id="t2")
        c.emit_filter_rejection(
            _ts(10),
            "sss",
            filter_stage="edge.spread",
            rejection_reason="spread too wide",
        )
        evt = c.events()[0]
        assert evt.event_type == "signal_rejected_edge"
        assert evt.filter_stage == "edge.spread"
        assert evt.rejection_reason == "spread too wide"

    def test_emit_entry(self) -> None:
        c = StrategyTelemetryCollector(run_id="t3")
        c.emit_entry(
            _ts(14),
            "ichimoku",
            direction="short",
            price=2000.0,
            planned_size=0.5,
            planned_stop_pips=20.0,
        )
        evt = c.events()[0]
        assert evt.event_type == "signal_entered"
        assert evt.planned_size == 0.5

    def test_emit_trade_exited(self) -> None:
        c = StrategyTelemetryCollector(run_id="t4")
        c.emit_trade_exited(
            _ts(15),
            "asian_breakout",
            direction="long",
            price=2003.0,
            realized_r=1.5,
        )
        evt = c.events()[0]
        assert evt.event_type == "trade_exited"
        assert evt.realized_r == 1.5


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestParquetRoundtrip:
    def test_empty_collector_writes_valid_empty_parquet(self, tmp_path: Path) -> None:
        c = StrategyTelemetryCollector(run_id="empty_test")
        out_path = tmp_path / "telemetry.parquet"
        c.to_parquet(out_path)
        assert out_path.exists()
        df = pd.read_parquet(out_path)
        assert len(df) == 0
        # Schema columns should still be present
        for col in ("timestamp_utc", "strategy_name", "event_type", "session"):
            assert col in df.columns

    def test_populated_collector_roundtrips(self, tmp_path: Path) -> None:
        c = StrategyTelemetryCollector(run_id="populated")
        c.emit_signal_generated(_ts(14), "sss", direction="long", pattern_type="cbc")
        c.emit_filter_rejection(_ts(14), "sss", "edge.spread", "spread too wide")
        c.emit_entry(_ts(15), "sss", direction="long", planned_size=1.0)
        c.emit_trade_exited(_ts(16), "sss", realized_r=1.5)

        out_path = tmp_path / "populated.parquet"
        c.to_parquet(out_path)
        df = pd.read_parquet(out_path)
        assert len(df) == 4
        assert set(df["event_type"]) == {
            "signal_generated",
            "signal_rejected_edge",
            "signal_entered",
            "trade_exited",
        }

    def test_extra_dict_serialized_as_json(self, tmp_path: Path) -> None:
        c = StrategyTelemetryCollector(run_id="extra")
        c.emit_signal_generated(_ts(14), "sss", direction="long", custom_key="custom_value")
        out_path = tmp_path / "extra.parquet"
        c.to_parquet(out_path)
        df = pd.read_parquet(out_path)
        assert len(df) == 1
        extra = json.loads(df.iloc[0]["extra"])
        assert extra == {"custom_key": "custom_value"}


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------


class TestSummary:
    def _make_fixture(self) -> StrategyTelemetryCollector:
        c = StrategyTelemetryCollector(run_id="summary_test")
        # 3 sss signal_generated events in different sessions
        c.emit_signal_generated(_ts(3), "sss", pattern_type="cbc")  # asian
        c.emit_signal_generated(_ts(10), "sss", pattern_type="fifty_tap")  # london
        c.emit_signal_generated(_ts(14), "sss", pattern_type="cbc")  # overlap
        # 1 sss entry
        c.emit_entry(_ts(14), "sss")
        # 2 rejections on different stages
        c.emit_filter_rejection(_ts(10), "sss", "edge.spread", "wide")
        c.emit_filter_rejection(_ts(10), "sss", "edge.spread", "wide")
        c.emit_filter_rejection(_ts(11), "sss", "learning.pre_trade", "low win rate")
        # 1 ichimoku signal
        c.emit_signal_generated(_ts(14), "ichimoku", pattern_type="tk_cross")
        return c

    def test_per_strategy_counts(self) -> None:
        summary = self._make_fixture().summary()
        sss = summary["per_strategy"]["sss"]
        assert sss["generated"] == 3
        assert sss["entered"] == 1
        assert sss["rejected"] == 3
        # Entry rate = 1/3 * 100 = 33.33%
        assert sss["entry_rate_pct"] == pytest.approx(33.33)

    def test_per_session_counts(self) -> None:
        summary = self._make_fixture().summary()
        # Fixture events by hour:
        #   asian  (h=3):  1 signal_generated
        #   london (h=10): 1 generated + 2 rejected (edge.spread)
        #                  (h=11): 1 rejected (learning.pre_trade) → 4 total
        #   overlap(h=14): 1 sss generated + 1 sss entry + 1 ichimoku gen → 3 total
        assert summary["per_session"]["asian"] == 1
        assert summary["per_session"]["london"] == 4
        assert summary["per_session"]["overlap"] == 3

    def test_per_pattern_counts(self) -> None:
        summary = self._make_fixture().summary()
        assert summary["per_pattern"]["cbc"] == 2
        assert summary["per_pattern"]["fifty_tap"] == 1
        assert summary["per_pattern"]["tk_cross"] == 1

    def test_top_rejection_stages_sorted(self) -> None:
        summary = self._make_fixture().summary()
        stages = summary["top_rejection_stages"]
        # edge.spread (2) should come before learning.pre_trade (1)
        assert list(stages.keys())[0] == "edge.spread"
        assert stages["edge.spread"] == 2
        assert stages["learning.pre_trade"] == 1

    def test_to_summary_json(self, tmp_path: Path) -> None:
        c = self._make_fixture()
        out_path = tmp_path / "summary.json"
        c.to_summary_json(out_path)
        data = json.loads(out_path.read_text())
        assert data["total_events"] == 8
        assert data["run_id"] == "summary_test"
