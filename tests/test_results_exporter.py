"""
Unit tests for src/backtesting/results_exporter.py

All external I/O (filesystem) is isolated to a tmp_path fixture — no
real-world directories are touched.  No database or network access.

Test categories
---------------
1.  export_run_report — file creation, JSON validity, size guard
2.  Report structure — required top-level keys present
3.  worst/best trade selection — correct ordering and count
4.  Edge stats — aggregation from trade edge_results
5.  Comparison delta computation
6.  load_run_report — round-trip fidelity
7.  list_runs — enumeration, sorting, summary fields
8.  Edge cases — empty trades, no previous runs, large trade lists
9.  Config snapshot fidelity
10. Run ID uniqueness / no collision
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from src.backtesting.results_exporter import (
    ResultsExporter,
    _MAX_REPORT_BYTES,
    _EXTREME_TRADE_COUNT,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _make_trade(
    r_multiple: float = 1.0,
    won: bool = True,
    edge_results: Dict[str, bool] | None = None,
) -> Dict[str, Any]:
    """Return a minimal trade dict with controllable R-multiple and edge results."""
    return {
        "entry_time": "2024-01-02T08:00:00+00:00",
        "exit_time": "2024-01-02T12:00:00+00:00",
        "entry_price": 1900.0,
        "exit_price": 1905.0 if won else 1895.0,
        "direction": "long",
        "r_multiple": r_multiple,
        "pnl": r_multiple * 100.0,
        "confluence_score": 5,
        "signal_tier": "B",
        "edge_results": edge_results or {
            "time_of_day": True,
            "day_of_week": True,
            "regime_filter": True,
        },
    }


def _make_trades(n: int, base_r: float = 1.0) -> List[Dict[str, Any]]:
    """Return n trades with R-multiples from 0 to n-1 (ascending)."""
    return [
        _make_trade(r_multiple=float(i), won=i > 0)
        for i in range(n)
    ]


def _make_metrics(
    sharpe: float = 1.5,
    win_rate: float = 0.6,
    trade_count: int = 30,
    total_return: float = 0.12,
    max_drawdown: float = -0.05,
    expectancy: float = 0.8,
) -> Dict[str, Any]:
    return {
        "sharpe": sharpe,
        "win_rate": win_rate,
        "trade_count": trade_count,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "expectancy": expectancy,
        "profit_factor": 2.0,
        "sortino": 2.1,
        "calmar": 1.8,
    }


def _make_config() -> Dict[str, Any]:
    return {
        "strategy": {
            "ichimoku": {"tenkan_period": 9, "kijun_period": 26},
            "adx": {"period": 14, "threshold": 28},
        },
        "edges": {
            "time_of_day": {"enabled": True, "params": {"start_utc": "08:00"}},
            "regime_filter": {"enabled": True, "params": {"adx_min": 28}},
        },
    }


def _make_result(
    trades: List[Dict[str, Any]] | None = None,
    metrics: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Create a plain-dict BacktestResult for testing (no engine dependency)."""
    return {
        "trades": trades if trades is not None else _make_trades(10),
        "metrics": metrics if metrics is not None else _make_metrics(),
    }


@pytest.fixture
def exporter(tmp_path: Path) -> ResultsExporter:
    return ResultsExporter(reports_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# 1. export_run_report — file creation, JSON validity, size guard
# ---------------------------------------------------------------------------

class TestExportRunReport:

    def test_creates_file_in_reports_dir(self, exporter: ResultsExporter, tmp_path: Path) -> None:
        result = _make_result()
        path = exporter.export_run_report(result, _make_config(), run_id="run_001")
        assert path.exists()
        assert path.parent == tmp_path

    def test_file_is_valid_json(self, exporter: ResultsExporter) -> None:
        result = _make_result()
        path = exporter.export_run_report(result, _make_config(), run_id="run_002")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_file_under_50kb(self, exporter: ResultsExporter) -> None:
        result = _make_result()
        path = exporter.export_run_report(result, _make_config(), run_id="run_003")
        assert path.stat().st_size <= _MAX_REPORT_BYTES

    def test_returns_path_object(self, exporter: ResultsExporter) -> None:
        result = _make_result()
        path = exporter.export_run_report(result, _make_config(), run_id="run_004")
        assert isinstance(path, Path)

    def test_run_id_in_filename(self, exporter: ResultsExporter, tmp_path: Path) -> None:
        result = _make_result()
        path = exporter.export_run_report(result, _make_config(), run_id="my_special_run")
        assert path.name == "my_special_run.json"

    def test_accepts_backtest_result_object(self, exporter: ResultsExporter) -> None:
        """Should work with an object that has .trades and .metrics attributes."""
        from types import SimpleNamespace
        obj = SimpleNamespace(
            trades=_make_trades(5),
            metrics=_make_metrics(),
        )
        path = exporter.export_run_report(obj, _make_config(), run_id="run_obj")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["run_id"] == "run_obj"


# ---------------------------------------------------------------------------
# 2. Report structure — required top-level keys
# ---------------------------------------------------------------------------

class TestReportStructure:

    def _load(self, exporter: ResultsExporter, run_id: str = "struct_test") -> Dict[str, Any]:
        result = _make_result()
        path = exporter.export_run_report(result, _make_config(), run_id=run_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def test_has_run_id(self, exporter: ResultsExporter) -> None:
        data = self._load(exporter)
        assert data["run_id"] == "struct_test"

    def test_has_timestamp(self, exporter: ResultsExporter) -> None:
        data = self._load(exporter, "ts_test")
        assert "timestamp" in data
        # Should be a parseable ISO-8601 string.
        datetime.fromisoformat(data["timestamp"])

    def test_has_metrics(self, exporter: ResultsExporter) -> None:
        data = self._load(exporter, "metrics_test")
        assert "metrics" in data
        assert isinstance(data["metrics"], dict)

    def test_metrics_contain_required_keys(self, exporter: ResultsExporter) -> None:
        data = self._load(exporter, "metrics_keys_test")
        metrics = data["metrics"]
        for key in ("sharpe", "win_rate", "max_drawdown", "expectancy", "total_return", "trade_count"):
            assert key in metrics, f"Missing metrics key: {key}"

    def test_has_best_trades(self, exporter: ResultsExporter) -> None:
        data = self._load(exporter, "best_test")
        assert "best_trades" in data
        assert isinstance(data["best_trades"], list)

    def test_has_worst_trades(self, exporter: ResultsExporter) -> None:
        data = self._load(exporter, "worst_test")
        assert "worst_trades" in data
        assert isinstance(data["worst_trades"], list)

    def test_has_edge_stats(self, exporter: ResultsExporter) -> None:
        data = self._load(exporter, "edge_test")
        assert "edge_stats" in data
        assert isinstance(data["edge_stats"], dict)

    def test_has_config_snapshot(self, exporter: ResultsExporter) -> None:
        data = self._load(exporter, "config_test")
        assert "config_snapshot" in data
        assert isinstance(data["config_snapshot"], dict)

    def test_has_comparison(self, exporter: ResultsExporter) -> None:
        data = self._load(exporter, "cmp_test")
        assert "comparison" in data


# ---------------------------------------------------------------------------
# 3. worst/best trade selection
# ---------------------------------------------------------------------------

class TestSelectExtremeTrades:

    def test_best_trades_are_highest_r(self, exporter: ResultsExporter) -> None:
        trades = _make_trades(20)  # R-multiples 0..19
        best, _ = exporter._select_extreme_trades(trades)
        r_values = [t["r_multiple"] for t in best]
        assert all(r >= 15.0 for r in r_values), f"Expected top 5 R >= 15, got {r_values}"

    def test_worst_trades_are_lowest_r(self, exporter: ResultsExporter) -> None:
        trades = _make_trades(20)
        _, worst = exporter._select_extreme_trades(trades)
        r_values = [t["r_multiple"] for t in worst]
        assert all(r <= 4.0 for r in r_values), f"Expected bottom 5 R <= 4, got {r_values}"

    def test_count_does_not_exceed_n(self, exporter: ResultsExporter) -> None:
        trades = _make_trades(3)
        best, worst = exporter._select_extreme_trades(trades, n=5)
        assert len(best) <= 5
        assert len(worst) <= 5

    def test_exactly_five_trades_returned_when_enough(self, exporter: ResultsExporter) -> None:
        trades = _make_trades(20)
        best, worst = exporter._select_extreme_trades(trades)
        assert len(best) == _EXTREME_TRADE_COUNT
        assert len(worst) == _EXTREME_TRADE_COUNT

    def test_empty_trades_returns_empty_lists(self, exporter: ResultsExporter) -> None:
        best, worst = exporter._select_extreme_trades([])
        assert best == []
        assert worst == []

    def test_worst_trades_ordered_worst_first(self, exporter: ResultsExporter) -> None:
        trades = _make_trades(10)  # R-multiples 0..9
        _, worst = exporter._select_extreme_trades(trades)
        r_values = [t["r_multiple"] for t in worst]
        assert r_values == sorted(r_values), "worst_trades should be ordered worst (lowest R) first"

    def test_best_trades_ordered_best_first(self, exporter: ResultsExporter) -> None:
        trades = _make_trades(10)
        best, _ = exporter._select_extreme_trades(trades)
        r_values = [t["r_multiple"] for t in best]
        assert r_values == sorted(r_values, reverse=True), "best_trades should be ordered best (highest R) first"


# ---------------------------------------------------------------------------
# 4. Edge stats
# ---------------------------------------------------------------------------

class TestComputeEdgeStats:

    def test_basic_pass_fail_counts(self, exporter: ResultsExporter) -> None:
        trades = [
            _make_trade(r_multiple=2.0, won=True,  edge_results={"tod": True}),
            _make_trade(r_multiple=-1.0, won=False, edge_results={"tod": False}),
            _make_trade(r_multiple=1.5, won=True,  edge_results={"tod": True}),
        ]
        stats = exporter._compute_edge_stats(trades)
        assert "tod" in stats
        assert stats["tod"]["passed"] == 2
        assert stats["tod"]["failed"] == 1
        assert stats["tod"]["total"] == 3

    def test_pass_rate_correct(self, exporter: ResultsExporter) -> None:
        trades = [
            _make_trade(r_multiple=1.0, won=True,  edge_results={"e1": True}),
            _make_trade(r_multiple=1.0, won=True,  edge_results={"e1": True}),
            _make_trade(r_multiple=-1.0, won=False, edge_results={"e1": False}),
            _make_trade(r_multiple=-1.0, won=False, edge_results={"e1": False}),
        ]
        stats = exporter._compute_edge_stats(trades)
        assert stats["e1"]["pass_rate"] == pytest.approx(0.5, abs=1e-4)

    def test_win_rate_when_passed(self, exporter: ResultsExporter) -> None:
        trades = [
            _make_trade(r_multiple=2.0, won=True,  edge_results={"e2": True}),
            _make_trade(r_multiple=-1.0, won=False, edge_results={"e2": True}),
        ]
        stats = exporter._compute_edge_stats(trades)
        # 1 win out of 2 passes
        assert stats["e2"]["win_rate_when_passed"] == pytest.approx(0.5, abs=1e-4)

    def test_empty_trades_returns_empty_dict(self, exporter: ResultsExporter) -> None:
        stats = exporter._compute_edge_stats([])
        assert stats == {}

    def test_multiple_edges_tracked_independently(self, exporter: ResultsExporter) -> None:
        trades = [
            _make_trade(r_multiple=1.0, won=True,  edge_results={"a": True, "b": False}),
            _make_trade(r_multiple=1.0, won=True,  edge_results={"a": True, "b": True}),
        ]
        stats = exporter._compute_edge_stats(trades)
        assert stats["a"]["passed"] == 2
        assert stats["a"]["failed"] == 0
        assert stats["b"]["passed"] == 1
        assert stats["b"]["failed"] == 1

    def test_trade_without_edge_results_skipped_gracefully(self, exporter: ResultsExporter) -> None:
        trades = [
            {"r_multiple": 1.0, "pnl": 100.0},  # no edge_results key
            _make_trade(r_multiple=1.0, won=True, edge_results={"e3": True}),
        ]
        stats = exporter._compute_edge_stats(trades)
        assert stats["e3"]["total"] == 1


# ---------------------------------------------------------------------------
# 5. Comparison delta
# ---------------------------------------------------------------------------

class TestComputeComparison:

    def test_no_previous_returns_has_previous_false(self, exporter: ResultsExporter) -> None:
        cmp = exporter._compute_comparison(_make_metrics(), None)
        assert cmp == {"has_previous": False}

    def test_delta_computed_correctly(self, exporter: ResultsExporter) -> None:
        current = {"sharpe": 1.8, "win_rate": 0.65}
        previous = {"sharpe": 1.5, "win_rate": 0.60}
        cmp = exporter._compute_comparison(current, previous)
        assert cmp["has_previous"] is True
        assert cmp["sharpe"]["delta"] == pytest.approx(0.3, abs=1e-5)
        assert cmp["win_rate"]["delta"] == pytest.approx(0.05, abs=1e-5)

    def test_negative_delta_for_regression(self, exporter: ResultsExporter) -> None:
        current = {"sharpe": 1.0}
        previous = {"sharpe": 1.5}
        cmp = exporter._compute_comparison(current, previous)
        assert cmp["sharpe"]["delta"] < 0

    def test_missing_key_in_previous_not_included(self, exporter: ResultsExporter) -> None:
        current = {"sharpe": 1.5, "win_rate": 0.6}
        previous = {"sharpe": 1.2}  # no win_rate
        cmp = exporter._compute_comparison(current, previous)
        # win_rate has no previous counterpart → should be absent
        assert "win_rate" not in cmp

    def test_comparison_includes_current_and_previous_values(self, exporter: ResultsExporter) -> None:
        current = {"sharpe": 2.0}
        previous = {"sharpe": 1.5}
        cmp = exporter._compute_comparison(current, previous)
        assert cmp["sharpe"]["current"] == pytest.approx(2.0, abs=1e-5)
        assert cmp["sharpe"]["previous"] == pytest.approx(1.5, abs=1e-5)


# ---------------------------------------------------------------------------
# 6. load_run_report — round-trip fidelity
# ---------------------------------------------------------------------------

class TestLoadRunReport:

    def test_round_trip_preserves_run_id(self, exporter: ResultsExporter) -> None:
        exporter.export_run_report(_make_result(), _make_config(), run_id="rt_001")
        loaded = exporter.load_run_report("rt_001")
        assert loaded["run_id"] == "rt_001"

    def test_round_trip_preserves_metrics(self, exporter: ResultsExporter) -> None:
        metrics = _make_metrics(sharpe=2.5, win_rate=0.71)
        exporter.export_run_report(_make_result(metrics=metrics), _make_config(), run_id="rt_002")
        loaded = exporter.load_run_report("rt_002")
        assert loaded["metrics"]["sharpe"] == pytest.approx(2.5, abs=1e-4)
        assert loaded["metrics"]["win_rate"] == pytest.approx(0.71, abs=1e-4)

    def test_round_trip_preserves_config_snapshot(self, exporter: ResultsExporter) -> None:
        config = _make_config()
        exporter.export_run_report(_make_result(), config, run_id="rt_003")
        loaded = exporter.load_run_report("rt_003")
        assert loaded["config_snapshot"]["strategy"]["ichimoku"]["tenkan_period"] == 9

    def test_load_nonexistent_run_raises_file_not_found(self, exporter: ResultsExporter) -> None:
        with pytest.raises(FileNotFoundError, match="no_such_run"):
            exporter.load_run_report("no_such_run")

    def test_round_trip_preserves_edge_stats(self, exporter: ResultsExporter) -> None:
        trades = [
            _make_trade(r_multiple=1.5, won=True,  edge_results={"tod": True}),
            _make_trade(r_multiple=-1.0, won=False, edge_results={"tod": False}),
        ]
        exporter.export_run_report(_make_result(trades=trades), _make_config(), run_id="rt_004")
        loaded = exporter.load_run_report("rt_004")
        assert "tod" in loaded["edge_stats"]


# ---------------------------------------------------------------------------
# 7. list_runs — enumeration and sorting
# ---------------------------------------------------------------------------

class TestListRuns:

    def test_returns_empty_list_when_no_runs(self, exporter: ResultsExporter) -> None:
        assert exporter.list_runs() == []

    def test_returns_all_saved_runs(self, exporter: ResultsExporter) -> None:
        for i in range(3):
            exporter.export_run_report(_make_result(), _make_config(), run_id=f"run_{i:03d}")
        runs = exporter.list_runs()
        assert len(runs) == 3

    def test_each_entry_has_run_id(self, exporter: ResultsExporter) -> None:
        exporter.export_run_report(_make_result(), _make_config(), run_id="lr_001")
        runs = exporter.list_runs()
        assert "run_id" in runs[0]

    def test_each_entry_has_summary_metrics(self, exporter: ResultsExporter) -> None:
        exporter.export_run_report(_make_result(), _make_config(), run_id="lr_002")
        runs = exporter.list_runs()
        for key in ("sharpe", "win_rate", "trade_count"):
            assert key in runs[0], f"Missing summary key: {key}"

    def test_sorted_by_timestamp(self, exporter: ResultsExporter) -> None:
        for i in range(5):
            exporter.export_run_report(_make_result(), _make_config(), run_id=f"ord_{i:03d}")
        runs = exporter.list_runs()
        timestamps = [r["timestamp"] for r in runs]
        assert timestamps == sorted(timestamps)

    def test_run_ids_unique_no_collision(self, exporter: ResultsExporter) -> None:
        for i in range(10):
            exporter.export_run_report(_make_result(), _make_config(), run_id=f"uniq_{i:03d}")
        runs = exporter.list_runs()
        ids = [r["run_id"] for r in runs]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# 8. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_trades_list_still_produces_valid_report(self, exporter: ResultsExporter) -> None:
        result = _make_result(trades=[])
        path = exporter.export_run_report(result, _make_config(), run_id="empty_trades")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["best_trades"] == []
        assert data["worst_trades"] == []
        assert data["edge_stats"] == {}

    def test_missing_previous_runs_gives_no_comparison(self, exporter: ResultsExporter) -> None:
        result = _make_result()
        path = exporter.export_run_report(result, _make_config(), run_id="no_prev")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["comparison"]["has_previous"] is False

    def test_with_previous_runs_gives_comparison(self, exporter: ResultsExporter) -> None:
        prev_report = {
            "run_id": "prev_run",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "metrics": _make_metrics(sharpe=1.0),
        }
        result = _make_result(metrics=_make_metrics(sharpe=1.8))
        path = exporter.export_run_report(
            result, _make_config(), run_id="with_prev",
            previous_runs=[prev_report],
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["comparison"]["has_previous"] is True
        assert "sharpe" in data["comparison"]

    def test_large_trade_list_stays_under_50kb(self, exporter: ResultsExporter) -> None:
        """Even 500 trades should produce a report under 50 KB (we only keep top/bottom 5)."""
        trades = _make_trades(500)
        result = _make_result(trades=trades)
        path = exporter.export_run_report(result, _make_config(), run_id="big_run")
        assert path.stat().st_size <= _MAX_REPORT_BYTES

    def test_single_trade_works(self, exporter: ResultsExporter) -> None:
        result = _make_result(trades=[_make_trade(r_multiple=3.0)])
        path = exporter.export_run_report(result, _make_config(), run_id="one_trade")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data["best_trades"]) == 1
        assert len(data["worst_trades"]) == 1

    def test_none_r_multiple_handled(self, exporter: ResultsExporter) -> None:
        """Trades where r_multiple is None should not crash."""
        trade = _make_trade(r_multiple=1.0)
        trade["r_multiple"] = None
        result = _make_result(trades=[trade])
        path = exporter.export_run_report(result, _make_config(), run_id="none_r")
        assert path.exists()


# ---------------------------------------------------------------------------
# 9. Config snapshot fidelity
# ---------------------------------------------------------------------------

class TestConfigSnapshot:

    def test_config_snapshot_includes_strategy(self, exporter: ResultsExporter) -> None:
        config = _make_config()
        path = exporter.export_run_report(_make_result(), config, run_id="cfg_001")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "strategy" in data["config_snapshot"]

    def test_config_snapshot_includes_edges(self, exporter: ResultsExporter) -> None:
        config = _make_config()
        path = exporter.export_run_report(_make_result(), config, run_id="cfg_002")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "edges" in data["config_snapshot"]

    def test_config_snapshot_preserves_nested_values(self, exporter: ResultsExporter) -> None:
        config = _make_config()
        path = exporter.export_run_report(_make_result(), config, run_id="cfg_003")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["config_snapshot"]["strategy"]["ichimoku"]["kijun_period"] == 26

    def test_empty_config_does_not_crash(self, exporter: ResultsExporter) -> None:
        path = exporter.export_run_report(_make_result(), {}, run_id="cfg_empty")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["config_snapshot"] == {}


# ---------------------------------------------------------------------------
# 10. Run ID uniqueness / no collision
# ---------------------------------------------------------------------------

class TestRunIdNoCollision:

    def test_separate_run_ids_produce_separate_files(self, exporter: ResultsExporter, tmp_path: Path) -> None:
        exporter.export_run_report(_make_result(), _make_config(), run_id="col_a")
        exporter.export_run_report(_make_result(), _make_config(), run_id="col_b")
        assert (tmp_path / "col_a.json").exists()
        assert (tmp_path / "col_b.json").exists()

    def test_same_run_id_overwrites(self, exporter: ResultsExporter, tmp_path: Path) -> None:
        """Re-exporting with the same run_id should overwrite, not create a second file."""
        exporter.export_run_report(_make_result(metrics=_make_metrics(sharpe=1.0)), _make_config(), run_id="ow_001")
        exporter.export_run_report(_make_result(metrics=_make_metrics(sharpe=2.5)), _make_config(), run_id="ow_001")
        files = list(tmp_path.glob("ow_001*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert data["metrics"]["sharpe"] == pytest.approx(2.5, abs=1e-4)

    def test_reports_dir_created_automatically(self, tmp_path: Path) -> None:
        nested = tmp_path / "nested" / "reports"
        exp = ResultsExporter(reports_dir=str(nested))
        exp.export_run_report(_make_result(), _make_config(), run_id="auto_dir")
        assert nested.exists()
