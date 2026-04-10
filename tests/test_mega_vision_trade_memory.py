"""Tests for TradeMemory + PerformanceBuckets (plan Task 22)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.mega_vision.performance_buckets import PerformanceBuckets
from src.mega_vision.trade_memory import TradeMemory, TradeRecord


UTC = timezone.utc


@pytest.fixture
def mem():
    return TradeMemory(db_path=":memory:")


def _make_trade(**overrides) -> dict:
    base = {
        "opened_at": "2026-04-09T10:00:00+00:00",
        "closed_at": "2026-04-09T11:00:00+00:00",
        "duration_minutes": 60.0,
        "strategy_name": "sss",
        "instrument_class": "futures",
        "symbol": "MGC",
        "direction": "long",
        "entry_price": 2000.0,
        "exit_price": 2005.0,
        "stop_price": 1995.0,
        "tp_price": 2010.0,
        "size": 1.0,
        "pnl_usd": 50.0,
        "r_multiple": 1.0,
        "session": "ny",
        "pattern_type": "cbc",
        "regime": "trend_up",
        "confluence_score": 3.0,
        "prop_firm_style": "topstep_combine_dollar",
    }
    base.update(overrides)
    return base


class TestInsertQuery:
    def test_insert_dict_returns_id(self, mem):
        tid = mem.insert(_make_trade())
        assert tid
        assert mem.count() == 1

    def test_insert_record_dataclass(self, mem):
        record = TradeRecord(strategy_name="sss", pnl_usd=100.0)
        tid = mem.insert(record)
        assert tid == record.trade_id

    def test_query_all(self, mem):
        mem.insert(_make_trade(strategy_name="sss", pnl_usd=50.0))
        mem.insert(_make_trade(strategy_name="ichimoku", pnl_usd=-20.0))
        rows = mem.query({})
        assert len(rows) == 2

    def test_query_by_strategy(self, mem):
        mem.insert(_make_trade(strategy_name="sss"))
        mem.insert(_make_trade(strategy_name="ichimoku"))
        mem.insert(_make_trade(strategy_name="sss"))
        sss_trades = mem.query({"strategy_name": "sss"})
        assert len(sss_trades) == 2

    def test_query_recent_limit(self, mem):
        for i in range(5):
            mem.insert(_make_trade(pnl_usd=float(i)))
        recent = mem.query_recent(n=3)
        assert len(recent) == 3

    def test_unknown_fields_go_into_extra_json(self, mem):
        import json

        mem.insert(_make_trade(custom_metric=0.75))
        rows = mem.query({})
        extra = json.loads(rows[0]["extra_json"])
        assert extra["custom_metric"] == 0.75


class TestAggregation:
    def test_aggregate_by_strategy(self, mem):
        mem.insert(_make_trade(strategy_name="sss", pnl_usd=100.0, r_multiple=1.0))
        mem.insert(_make_trade(strategy_name="sss", pnl_usd=-50.0, r_multiple=-0.5))
        mem.insert(_make_trade(strategy_name="ichimoku", pnl_usd=25.0, r_multiple=0.5))
        df = mem.aggregate_by("strategy_name")
        assert isinstance(df, pd.DataFrame)
        sss_row = df[df["strategy_name"] == "sss"].iloc[0]
        assert sss_row["trade_count"] == 2
        assert sss_row["total_pnl"] == 50.0
        assert sss_row["win_count"] == 1
        assert sss_row["win_rate"] == 0.5

    def test_aggregate_empty_returns_empty_df(self, mem):
        df = mem.aggregate_by("strategy_name")
        assert df.empty

    def test_aggregate_unknown_key_raises(self, mem):
        mem.insert(_make_trade())
        with pytest.raises(KeyError):
            mem.aggregate_by("does_not_exist")


class TestSnapshotToParquet:
    def test_snapshot_writes_parquet(self, mem, tmp_path: Path):
        mem.insert(_make_trade())
        mem.insert(_make_trade(strategy_name="ichimoku"))
        out = tmp_path / "snapshot.parquet"
        mem.snapshot_to_parquet(out)
        assert out.exists()
        df = pd.read_parquet(out)
        assert len(df) == 2


class TestPerformanceBuckets:
    def test_buckets_empty_when_memory_empty(self, mem):
        pb = PerformanceBuckets(trade_memory=mem)
        assert pb.get_buckets() == {}

    def test_buckets_aggregates_by_strategy_session_pattern_regime(self, mem):
        mem.insert(_make_trade(strategy_name="sss", session="ny", pattern_type="cbc", regime="trend_up", pnl_usd=100.0, r_multiple=1.0))
        mem.insert(_make_trade(strategy_name="sss", session="ny", pattern_type="cbc", regime="trend_up", pnl_usd=-50.0, r_multiple=-0.5))
        mem.insert(_make_trade(strategy_name="sss", session="london", pattern_type="cbc", regime="range", pnl_usd=25.0, r_multiple=0.5))
        pb = PerformanceBuckets(trade_memory=mem)
        buckets = pb.get_buckets()
        ny_bucket = buckets["sss"]["ny"]["cbc"]["trend_up"]
        assert ny_bucket["trade_count"] == 2
        assert ny_bucket["win_count"] == 1
        assert ny_bucket["win_rate"] == 0.5
        assert ny_bucket["total_pnl"] == 50.0
        assert ny_bucket["avg_r"] == pytest.approx(0.25)

    def test_get_buckets_filtered_by_strategy(self, mem):
        mem.insert(_make_trade(strategy_name="sss"))
        mem.insert(_make_trade(strategy_name="ichimoku"))
        pb = PerformanceBuckets(trade_memory=mem)
        filtered = pb.get_buckets(strategy_name="sss")
        assert "sss" in filtered
        assert "ichimoku" not in filtered

    def test_invalidate_clears_cache(self, mem):
        pb = PerformanceBuckets(trade_memory=mem)
        pb.get_buckets()
        mem.insert(_make_trade())
        # Stale cache
        cached = pb.get_buckets()
        assert cached == {}
        pb.invalidate()
        fresh = pb.get_buckets()
        assert "sss" in fresh
