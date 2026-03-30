"""
Unit tests for scripts/run_optimization_loop.py

All external I/O (subprocess, filesystem, database) is mocked.
Tests use tmp_path for config/report file isolation.

Test categories
---------------
1.  OptimizationLoop.__init__ — config parsing and component setup
2.  _build_claude_prompt — sections, constraints, target Sharpe
3.  _spawn_claude — subprocess invocation and error handling
4.  _detect_changes — git diff on config files
5.  _check_plateau — plateau detection logic
6.  _load_config — reads and merges both YAML files
7.  _slice_data — date slicing correctness
8.  run() — full loop with mocked backtester and subprocess
9.  Revert when metrics worsen
10. Learning appended after each iteration
11. Final validation on full dataset
12. Graceful degradation without DB
"""

from __future__ import annotations

import subprocess
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest
import yaml

# ---------------------------------------------------------------------------
# Helpers — minimal BacktestResult-like object and data factories
# ---------------------------------------------------------------------------


class _FakeResult:
    """Minimal stand-in for BacktestResult."""

    def __init__(self, sharpe: float = 0.8, win_rate: float = 0.55) -> None:
        self.metrics: Dict[str, Any] = {
            "sharpe": sharpe,
            "win_rate": win_rate,
            "max_drawdown": 0.08,
            "total_return": 0.12,
            "profit_factor": 1.4,
            "trade_count": 40,
            "expectancy": 0.5,
            "sortino": 1.1,
            "calmar": 0.9,
        }
        self.trades: List[Dict[str, Any]] = [
            {
                "instrument": "XAU/USD",
                "direction": "long",
                "entry_price": 1900.0,
                "exit_price": 1910.0,
                "r_multiple": 1.0,
                "pnl": 100.0,
                "entry_time": "2024-01-02T08:00:00+00:00",
                "exit_time": "2024-01-02T12:00:00+00:00",
            }
        ]
        self.equity_curve = pd.Series([10000.0, 10100.0])
        self.prop_firm: Dict[str, Any] = {}
        self.daily_pnl = pd.Series([100.0])
        self.skipped_signals: int = 0
        self.total_signals: int = 5


def _make_loop_config(
    max_iterations: int = 3,
    target_sharpe: float = 2.0,
    reports_dir: Optional[str] = None,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "optimization": {
            "max_iterations": max_iterations,
            "target_sharpe": target_sharpe,
            "max_changes_per_iteration": 2,
            "optimization_years": 2,
            "plateau_threshold": 0.05,
            "plateau_iterations": 3,
        },
        "persistence": {
            "enabled": False,  # off by default in tests
            "reports_dir": reports_dir or "reports",
        },
        "claude": {
            "command": ["echo", "mock"],
            "timeout_seconds": 10,
        },
        "validation": {
            "run_full_dataset": False,
        },
    }
    return cfg


def _make_df(rows: int = 200) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame with a DatetimeIndex spanning 4 years."""
    end = datetime(2024, 1, 1, tzinfo=timezone.utc)
    start = end - timedelta(days=4 * 365)
    index = pd.date_range(start=start, end=end, periods=rows)
    return pd.DataFrame(
        {
            "open": np.random.uniform(1850, 1950, rows),
            "high": np.random.uniform(1860, 1960, rows),
            "low": np.random.uniform(1840, 1940, rows),
            "close": np.random.uniform(1850, 1950, rows),
            "volume": np.random.uniform(1000, 5000, rows),
        },
        index=index,
    )


def _make_strategy_yaml(tmp_path: Path) -> Path:
    content = textwrap.dedent("""\
        ichimoku:
          tenkan_period: 9
          kijun_period: 26
          senkou_b_period: 52
        adx:
          period: 14
          threshold: 28
        risk:
          initial_risk_pct: 1.5
    """)
    p = tmp_path / "strategy.yaml"
    p.write_text(content)
    return p


def _make_edges_yaml(tmp_path: Path) -> Path:
    content = textwrap.dedent("""\
        time_of_day:
          enabled: true
          params:
            start_utc: "08:00"
            end_utc: "17:00"
        regime_filter:
          enabled: true
          params:
            adx_min: 28
    """)
    p = tmp_path / "edges.yaml"
    p.write_text(content)
    return p


def _make_parquet(tmp_path: Path, rows: int = 200) -> Path:
    df = _make_df(rows)
    p = tmp_path / "data.parquet"
    df.to_parquet(p)
    return p


# ---------------------------------------------------------------------------
# Import the module under test with patched repo-root paths
# ---------------------------------------------------------------------------

import importlib
import sys


def _import_loop_module(tmp_path: Path):
    """Import run_optimization_loop with strategy/edges paths pointing to tmp_path."""
    import scripts.run_optimization_loop as mod
    # Patch the module-level path constants used by _load_config, _detect_changes etc.
    mod._STRATEGY_CONFIG_PATH = tmp_path / "strategy.yaml"
    mod._EDGES_CONFIG_PATH = tmp_path / "edges.yaml"
    mod._CLAUDE_MD_PATH = tmp_path / "CLAUDE.md"
    mod._REPO_ROOT = tmp_path
    return mod


# Ensure the scripts package is importable
_scripts_init = Path(__file__).resolve().parent.parent / "scripts" / "__init__.py"
if not _scripts_init.exists():
    _scripts_init.touch()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_configs(tmp_path: Path):
    """Create temporary strategy.yaml, edges.yaml, and data.parquet in tmp_path."""
    _make_strategy_yaml(tmp_path)
    _make_edges_yaml(tmp_path)
    _make_parquet(tmp_path)
    return tmp_path


@pytest.fixture()
def loop_mod(tmp_configs):
    """Return the module with patched path constants."""
    return _import_loop_module(tmp_configs)


@pytest.fixture()
def basic_loop(loop_mod, tmp_configs):
    """Return an OptimizationLoop instance with minimal config."""
    cfg = _make_loop_config(reports_dir=str(tmp_configs / "reports"))
    return loop_mod.OptimizationLoop(
        config=cfg,
        data_file=str(tmp_configs / "data.parquet"),
        reports_dir=str(tmp_configs / "reports"),
        claude_md_path=str(tmp_configs / "CLAUDE.md"),
    )


# ---------------------------------------------------------------------------
# 1. __init__ tests
# ---------------------------------------------------------------------------


class TestOptimizationLoopInit:
    def test_config_values_parsed(self, loop_mod, tmp_configs):
        cfg = _make_loop_config(max_iterations=7, target_sharpe=1.5)
        cfg["optimization"]["plateau_threshold"] = 0.03
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_configs / "data.parquet"),
            reports_dir=str(tmp_configs / "reports"),
        )
        assert loop._max_iterations == 7
        assert loop._target_sharpe == 1.5
        assert loop._plateau_threshold == pytest.approx(0.03)

    def test_no_db_pool_graceful(self, loop_mod, tmp_configs):
        """OptimizationLoop initialises without a db_pool (graceful degradation)."""
        cfg = _make_loop_config()
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_configs / "data.parquet"),
            db_pool=None,
        )
        # TradePersistence must exist even when db_pool is None
        assert loop._trade_persistence is not None

    def test_reports_dir_created(self, loop_mod, tmp_configs):
        reports = tmp_configs / "nested" / "reports"
        cfg = _make_loop_config(reports_dir=str(reports))
        loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_configs / "data.parquet"),
            reports_dir=str(reports),
        )
        assert reports.exists()


# ---------------------------------------------------------------------------
# 2. _build_claude_prompt tests
# ---------------------------------------------------------------------------


class TestBuildClaudePrompt:
    def _make_report(self, sharpe: float = 0.8) -> Dict[str, Any]:
        return {
            "run_id": "opt_iter_001",
            "metrics": {
                "sharpe": sharpe,
                "win_rate": 0.55,
                "max_drawdown": 0.08,
                "total_return": 0.12,
                "profit_factor": 1.4,
                "trade_count": 40,
                "expectancy": 0.5,
            },
            "comparison": {"has_previous": False},
            "config_snapshot": {"tenkan_period": 9, "kijun_period": 26},
        }

    def test_prompt_contains_run_id(self, basic_loop):
        report = self._make_report()
        prompt = basic_loop._build_claude_prompt(report, None, [], "No learnings.")
        assert "opt_iter_001" in prompt

    def test_prompt_contains_metrics(self, basic_loop):
        report = self._make_report(sharpe=1.23)
        prompt = basic_loop._build_claude_prompt(report, None, [], "No learnings.")
        assert "1.23" in prompt
        assert "0.55" in prompt  # win_rate

    def test_prompt_contains_max_changes_constraint(self, basic_loop):
        report = self._make_report()
        prompt = basic_loop._build_claude_prompt(report, None, [], "No learnings.")
        assert "AT MOST 2" in prompt

    def test_prompt_contains_target_sharpe(self, basic_loop):
        report = self._make_report()
        prompt = basic_loop._build_claude_prompt(report, None, [], "No learnings.")
        # target_sharpe is 2.0 in basic_loop fixture
        assert "2.0" in prompt or "2.00" in prompt

    def test_prompt_contains_previous_comparison(self, basic_loop):
        report = self._make_report(sharpe=0.9)
        prev_report = self._make_report(sharpe=0.7)
        report["comparison"] = {
            "has_previous": True,
            "sharpe": {"current": 0.9, "previous": 0.7, "delta": 0.2},
        }
        prompt = basic_loop._build_claude_prompt(report, prev_report, [], "")
        assert "Previous Run Comparison" in prompt
        assert "0.7" in prompt

    def test_prompt_contains_similar_configs(self, basic_loop):
        sc = MagicMock()
        sc.to_dict.return_value = {
            "run_id": "run_old_001",
            "similarity": 0.95,
            "win_rate": 0.60,
            "avg_r": 1.2,
            "trade_count": 55,
        }
        report = self._make_report()
        prompt = basic_loop._build_claude_prompt(report, None, [sc], "")
        assert "run_old_001" in prompt
        assert "0.95" in prompt or "0.950" in prompt

    def test_prompt_contains_learnings(self, basic_loop):
        report = self._make_report()
        learnings = "Run 001: changed tenkan_period → Sharpe 0.5→0.8 (kept)"
        prompt = basic_loop._build_claude_prompt(report, None, [], learnings)
        assert learnings in prompt

    def test_prompt_contains_config_yaml_block(self, basic_loop):
        report = self._make_report()
        prompt = basic_loop._build_claude_prompt(report, None, [], "")
        assert "```yaml" in prompt
        assert "tenkan_period" in prompt

    def test_prompt_mentions_config_files(self, basic_loop):
        report = self._make_report()
        prompt = basic_loop._build_claude_prompt(report, None, [], "")
        assert "config/strategy.yaml" in prompt
        assert "config/edges.yaml" in prompt


# ---------------------------------------------------------------------------
# 3. _spawn_claude tests
# ---------------------------------------------------------------------------


class TestSpawnClaude:
    def test_spawn_returns_stdout(self, basic_loop):
        completed = MagicMock()
        completed.returncode = 0
        completed.stdout = "Changed tenkan_period to 7."
        completed.stderr = ""
        with patch("subprocess.run", return_value=completed) as mock_run:
            result = basic_loop._spawn_claude("test prompt")
        assert result == "Changed tenkan_period to 7."
        mock_run.assert_called_once()

    def test_spawn_passes_input_to_stdin(self, basic_loop):
        completed = MagicMock(returncode=0, stdout="ok", stderr="")
        with patch("subprocess.run", return_value=completed) as mock_run:
            basic_loop._spawn_claude("my prompt text")
        _, kwargs = mock_run.call_args
        assert kwargs.get("input") == "my prompt text"

    def test_spawn_returns_empty_on_file_not_found(self, basic_loop):
        with patch("subprocess.run", side_effect=FileNotFoundError("claude not found")):
            result = basic_loop._spawn_claude("prompt")
        assert result == ""

    def test_spawn_returns_empty_on_timeout(self, basic_loop):
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=300),
        ):
            result = basic_loop._spawn_claude("prompt")
        assert result == ""

    def test_spawn_logs_nonzero_exit(self, basic_loop, caplog):
        import logging

        completed = MagicMock(returncode=1, stdout="", stderr="error msg")
        with patch("subprocess.run", return_value=completed):
            with caplog.at_level(logging.WARNING):
                basic_loop._spawn_claude("p")
        assert any("code 1" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 4. _detect_changes tests
# ---------------------------------------------------------------------------


class TestDetectChanges:
    def test_returns_diff_output(self, basic_loop):
        completed = MagicMock(returncode=0, stdout="@@ -1 +1 @@ tenkan_period: 9\n", stderr="")
        with patch("subprocess.run", return_value=completed):
            diff = basic_loop._detect_changes()
        assert "tenkan_period" in diff

    def test_returns_empty_when_no_changes(self, basic_loop):
        completed = MagicMock(returncode=0, stdout="", stderr="")
        with patch("subprocess.run", return_value=completed):
            diff = basic_loop._detect_changes()
        assert diff == ""

    def test_returns_empty_on_git_not_found(self, basic_loop):
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            diff = basic_loop._detect_changes()
        assert diff == ""

    def test_diff_includes_both_config_files(self, basic_loop):
        completed = MagicMock(returncode=0, stdout="diff output", stderr="")
        with patch("subprocess.run", return_value=completed) as mock_run:
            basic_loop._detect_changes()
        cmd = mock_run.call_args[0][0]
        # Command must include both config files
        assert any("strategy.yaml" in part for part in cmd)
        assert any("edges.yaml" in part for part in cmd)


# ---------------------------------------------------------------------------
# 5. _check_plateau tests
# ---------------------------------------------------------------------------


class TestCheckPlateau:
    def _entry(self, sharpe: float, iteration: int = 1) -> Dict[str, Any]:
        return {"iteration": iteration, "run_id": f"r{iteration}", "sharpe": sharpe}

    def test_no_plateau_with_short_history(self, basic_loop):
        history = [self._entry(0.5), self._entry(0.6)]
        assert basic_loop._check_plateau(history) is False

    def test_no_plateau_when_improving(self, basic_loop):
        # Each step > 5% improvement
        history = [
            self._entry(0.50),
            self._entry(0.60),  # +20%
            self._entry(0.72),  # +20%
        ]
        assert basic_loop._check_plateau(history) is False

    def test_plateau_detected_stale_metrics(self, basic_loop):
        # Each step < 5% improvement for 3 consecutive iterations
        history = [
            self._entry(1.000),
            self._entry(1.010),  # +1% — stale
            self._entry(1.020),  # +1% — stale
            self._entry(1.030),  # +1% — stale
        ]
        # plateau_iterations=3 so we look at last 3 entries
        assert basic_loop._check_plateau(history) is True

    def test_plateau_false_when_one_good_step(self, basic_loop):
        history = [
            self._entry(1.000),
            self._entry(1.010),  # stale
            self._entry(1.200),  # +18.8% — breaks plateau
            self._entry(1.210),  # stale
        ]
        # Last 3: 1.010 → 1.200 (+18.8%) → 1.210 (+0.8%)
        # The big jump in the middle means no plateau
        assert basic_loop._check_plateau(history) is False

    def test_plateau_zero_previous_sharpe_returns_false(self, basic_loop):
        history = [
            self._entry(0.0),
            self._entry(0.0),
            self._entry(0.0),
        ]
        # Division-by-zero guard should return False (safe side)
        assert basic_loop._check_plateau(history) is False


# ---------------------------------------------------------------------------
# 6. _load_config tests
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_reads_strategy_yaml(self, basic_loop):
        cfg = basic_loop._load_config()
        # strategy.yaml has ichimoku.tenkan_period: 9
        assert cfg.get("tenkan_period") == 9

    def test_reads_edges_yaml(self, basic_loop):
        cfg = basic_loop._load_config()
        assert "edges" in cfg
        assert "time_of_day" in cfg["edges"]

    def test_merged_config_contains_risk(self, basic_loop):
        cfg = basic_loop._load_config()
        assert "initial_risk_pct" in cfg

    def test_missing_strategy_yaml_returns_partial(self, loop_mod, tmp_configs):
        (tmp_configs / "strategy.yaml").unlink()
        cfg_dict = _make_loop_config(reports_dir=str(tmp_configs / "reports"))
        loop = loop_mod.OptimizationLoop(
            config=cfg_dict,
            data_file=str(tmp_configs / "data.parquet"),
            reports_dir=str(tmp_configs / "reports"),
            claude_md_path=str(tmp_configs / "CLAUDE.md"),
        )
        result = loop._load_config()
        # edges still present
        assert "edges" in result


# ---------------------------------------------------------------------------
# 7. _slice_data tests
# ---------------------------------------------------------------------------


class TestSliceData:
    def test_slice_returns_last_2_years(self, basic_loop):
        df = _make_df(rows=1000)
        sliced = basic_loop._slice_data(df, years=2)
        expected_cutoff = df.index.max() - timedelta(days=2 * 365)
        assert sliced.index.min() >= expected_cutoff
        assert len(sliced) < len(df)

    def test_slice_full_dataset_when_too_short(self, basic_loop):
        # Only 30 days of data — slice for 2 years should return full df
        end = datetime(2024, 1, 1, tzinfo=timezone.utc)
        start = end - timedelta(days=30)
        index = pd.date_range(start=start, end=end, periods=100)
        df = pd.DataFrame({"close": np.ones(100)}, index=index)
        sliced = basic_loop._slice_data(df, years=2)
        assert len(sliced) == len(df)

    def test_slice_non_datetime_index_returns_full(self, basic_loop):
        df = pd.DataFrame({"close": np.ones(50)})  # RangeIndex
        sliced = basic_loop._slice_data(df, years=2)
        assert len(sliced) == len(df)

    def test_slice_preserves_columns(self, basic_loop):
        df = _make_df(rows=500)
        sliced = basic_loop._slice_data(df, years=1)
        assert list(sliced.columns) == list(df.columns)


# ---------------------------------------------------------------------------
# 8. run() — full loop integration
# ---------------------------------------------------------------------------


class TestRunLoop:
    def _mock_backtester_factory(self, sharpe: float = 0.8):
        """Return a patch target and a mock class."""
        mock_instance = MagicMock()
        mock_instance.run.return_value = _FakeResult(sharpe=sharpe)
        mock_class = MagicMock(return_value=mock_instance)
        return mock_class

    def test_run_stops_at_target_sharpe(self, loop_mod, tmp_configs):
        """Loop stops when Sharpe > target_sharpe in first iteration."""
        cfg = _make_loop_config(
            max_iterations=5, target_sharpe=0.5,
            reports_dir=str(tmp_configs / "reports"),
        )
        cfg["validation"]["run_full_dataset"] = False
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_configs / "data.parquet"),
            reports_dir=str(tmp_configs / "reports"),
            claude_md_path=str(tmp_configs / "CLAUDE.md"),
        )
        mock_cls = self._mock_backtester_factory(sharpe=0.8)
        with patch.object(loop_mod, "_BacktesterClass", mock_cls):
            summary = loop.run()
        assert summary["stop_reason"] == "target_reached"
        assert summary["total_iterations"] == 1

    def test_run_stops_at_max_iterations(self, loop_mod, tmp_configs):
        cfg = _make_loop_config(
            max_iterations=2, target_sharpe=5.0,
            reports_dir=str(tmp_configs / "reports"),
        )
        cfg["validation"]["run_full_dataset"] = False
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_configs / "data.parquet"),
            reports_dir=str(tmp_configs / "reports"),
            claude_md_path=str(tmp_configs / "CLAUDE.md"),
        )
        mock_cls = self._mock_backtester_factory(sharpe=0.4)
        with patch.object(loop_mod, "_BacktesterClass", mock_cls):
            with patch("subprocess.run") as mock_sub:
                mock_sub.return_value = MagicMock(returncode=0, stdout="", stderr="")
                summary = loop.run()
        assert summary["stop_reason"] == "max_iterations"
        assert summary["total_iterations"] == 2

    def test_run_stops_at_plateau(self, loop_mod, tmp_configs):
        """Loop stops when plateau detected (3 stale iterations)."""
        cfg = _make_loop_config(
            max_iterations=10, target_sharpe=5.0,
            reports_dir=str(tmp_configs / "reports"),
        )
        cfg["optimization"]["plateau_iterations"] = 3
        cfg["optimization"]["plateau_threshold"] = 0.05
        cfg["validation"]["run_full_dataset"] = False

        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_configs / "data.parquet"),
            reports_dir=str(tmp_configs / "reports"),
            claude_md_path=str(tmp_configs / "CLAUDE.md"),
        )

        # Sharpe barely improves each time → plateau after 4 calls (3 improvements)
        call_count = [0]
        sharpe_values = [1.000, 1.005, 1.010, 1.015, 1.020, 1.025]

        def _fake_run(df):
            idx = min(call_count[0], len(sharpe_values) - 1)
            call_count[0] += 1
            return _FakeResult(sharpe=sharpe_values[idx])

        mock_cls = MagicMock()
        mock_cls.return_value.run.side_effect = _fake_run

        with patch.object(loop_mod, "_BacktesterClass", mock_cls):
            with patch("subprocess.run") as mock_sub:
                mock_sub.return_value = MagicMock(returncode=0, stdout="", stderr="")
                summary = loop.run()
        assert summary["stop_reason"] == "plateau"

    def test_run_returns_iteration_history(self, loop_mod, tmp_configs):
        cfg = _make_loop_config(
            max_iterations=2, target_sharpe=5.0,
            reports_dir=str(tmp_configs / "reports"),
        )
        cfg["validation"]["run_full_dataset"] = False
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_configs / "data.parquet"),
            reports_dir=str(tmp_configs / "reports"),
            claude_md_path=str(tmp_configs / "CLAUDE.md"),
        )
        mock_cls = self._mock_backtester_factory(sharpe=0.4)
        with patch.object(loop_mod, "_BacktesterClass", mock_cls):
            with patch("subprocess.run") as mock_sub:
                mock_sub.return_value = MagicMock(returncode=0, stdout="", stderr="")
                summary = loop.run()
        assert len(summary["iteration_history"]) == 2
        assert all("sharpe" in entry for entry in summary["iteration_history"])

    def test_run_missing_data_file_raises(self, loop_mod, tmp_configs):
        cfg = _make_loop_config(reports_dir=str(tmp_configs / "reports"))
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file="/nonexistent/path.parquet",
            reports_dir=str(tmp_configs / "reports"),
        )
        with pytest.raises(FileNotFoundError):
            loop.run()


# ---------------------------------------------------------------------------
# 9. Revert when metrics worsen
# ---------------------------------------------------------------------------


class TestRevertOnWorsening:
    def test_revert_called_when_sharpe_drops(self, loop_mod, tmp_configs):
        """If re-run sharpe < initial sharpe, _revert_changes must be called.

        Requires max_iterations >= 2 so that iteration 1 reaches the Claude
        spawn path (the guard ``if iteration == max_iterations: break`` skips
        Claude on the very last iteration to avoid wasted API calls).
        """
        cfg = _make_loop_config(
            max_iterations=2, target_sharpe=5.0,
            reports_dir=str(tmp_configs / "reports"),
        )
        cfg["validation"]["run_full_dataset"] = False
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_configs / "data.parquet"),
            reports_dir=str(tmp_configs / "reports"),
            claude_md_path=str(tmp_configs / "CLAUDE.md"),
        )

        # Calls: iter-1 backtest → iter-1 re-run (after Claude, worse) → iter-2 backtest
        run_values = [_FakeResult(sharpe=0.8), _FakeResult(sharpe=0.5), _FakeResult(sharpe=0.5)]
        run_idx = [0]

        def _alternating_run(df):
            result = run_values[min(run_idx[0], len(run_values) - 1)]
            run_idx[0] += 1
            return result

        mock_cls = MagicMock()
        mock_cls.return_value.run.side_effect = _alternating_run

        # Fake a non-empty git diff so revert is triggered
        diff_output = "@@ -1 +1 @@\n-tenkan_period: 9\n+tenkan_period: 7\n"
        with patch.object(loop_mod, "_BacktesterClass", mock_cls):
            with patch.object(loop, "_detect_changes", return_value=diff_output):
                with patch.object(loop, "_spawn_claude", return_value="changed tenkan"):
                    with patch.object(loop, "_revert_changes") as mock_revert:
                        loop.run()
        mock_revert.assert_called_once()

    def test_no_revert_when_sharpe_improves(self, loop_mod, tmp_configs):
        """No revert when the re-run sharpe is equal or better.

        Uses max_iterations=2 for the same reason as the revert test above.
        """
        cfg = _make_loop_config(
            max_iterations=2, target_sharpe=5.0,
            reports_dir=str(tmp_configs / "reports"),
        )
        cfg["validation"]["run_full_dataset"] = False
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_configs / "data.parquet"),
            reports_dir=str(tmp_configs / "reports"),
            claude_md_path=str(tmp_configs / "CLAUDE.md"),
        )

        # iter-1 backtest → iter-1 re-run (better) → iter-2 backtest
        run_values = [_FakeResult(sharpe=0.8), _FakeResult(sharpe=1.2), _FakeResult(sharpe=1.2)]
        run_idx = [0]

        def _alternating_run(df):
            result = run_values[min(run_idx[0], len(run_values) - 1)]
            run_idx[0] += 1
            return result

        mock_cls = MagicMock()
        mock_cls.return_value.run.side_effect = _alternating_run

        diff_output = "@@ -1 +1 @@\n-tenkan_period: 9\n+tenkan_period: 7\n"
        with patch.object(loop_mod, "_BacktesterClass", mock_cls):
            with patch.object(loop, "_detect_changes", return_value=diff_output):
                with patch.object(loop, "_spawn_claude", return_value="changed tenkan"):
                    with patch.object(loop, "_revert_changes") as mock_revert:
                        loop.run()
        mock_revert.assert_not_called()


# ---------------------------------------------------------------------------
# 10. Learning appended after each iteration
# ---------------------------------------------------------------------------


class TestLearningAppended:
    def test_learning_written_after_iteration_with_changes(self, loop_mod, tmp_configs):
        """Learning is appended to CLAUDE.md when config changes improve metrics.

        Uses max_iterations=2 so iteration 1 reaches the Claude spawn path.
        """
        cfg = _make_loop_config(
            max_iterations=2, target_sharpe=5.0,
            reports_dir=str(tmp_configs / "reports"),
        )
        cfg["validation"]["run_full_dataset"] = False
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_configs / "data.parquet"),
            reports_dir=str(tmp_configs / "reports"),
            claude_md_path=str(tmp_configs / "CLAUDE.md"),
        )

        # iter-1 backtest → iter-1 re-run (improved) → iter-2 backtest
        run_values = [_FakeResult(sharpe=0.8), _FakeResult(sharpe=1.0), _FakeResult(sharpe=1.0)]
        run_idx = [0]

        def _run(df):
            r = run_values[min(run_idx[0], len(run_values) - 1)]
            run_idx[0] += 1
            return r

        mock_cls = MagicMock()
        mock_cls.return_value.run.side_effect = _run

        diff_output = "@@ -1 +1 @@\n-tenkan_period: 9\n+tenkan_period: 7\n"
        with patch.object(loop_mod, "_BacktesterClass", mock_cls):
            with patch.object(loop, "_detect_changes", return_value=diff_output):
                with patch.object(loop, "_spawn_claude", return_value="ok"):
                    loop.run()

        claude_md = tmp_configs / "CLAUDE.md"
        assert claude_md.exists()
        content = claude_md.read_text()
        assert "Strategy Learnings" in content

    def test_learning_written_even_without_changes(self, loop_mod, tmp_configs):
        """A '(no changes)' learning is appended when Claude makes no diff.

        Uses max_iterations=2 so iteration 1 reaches the Claude spawn path,
        which then detects an empty diff and records '(no changes)'.
        """
        cfg = _make_loop_config(
            max_iterations=2, target_sharpe=5.0,
            reports_dir=str(tmp_configs / "reports"),
        )
        cfg["validation"]["run_full_dataset"] = False
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_configs / "data.parquet"),
            reports_dir=str(tmp_configs / "reports"),
            claude_md_path=str(tmp_configs / "CLAUDE.md"),
        )

        mock_cls = MagicMock()
        mock_cls.return_value.run.return_value = _FakeResult(sharpe=0.8)

        with patch.object(loop_mod, "_BacktesterClass", mock_cls):
            with patch.object(loop, "_detect_changes", return_value=""):
                with patch.object(loop, "_spawn_claude", return_value=""):
                    loop.run()

        claude_md = tmp_configs / "CLAUDE.md"
        assert claude_md.exists()
        content = claude_md.read_text()
        assert "(no changes)" in content


# ---------------------------------------------------------------------------
# 11. Final validation on full dataset
# ---------------------------------------------------------------------------


class TestFinalValidation:
    def test_final_validation_runs_on_full_dataset(self, loop_mod, tmp_configs):
        cfg = _make_loop_config(
            max_iterations=1, target_sharpe=0.5,
            reports_dir=str(tmp_configs / "reports"),
        )
        cfg["validation"]["run_full_dataset"] = True
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_configs / "data.parquet"),
            reports_dir=str(tmp_configs / "reports"),
            claude_md_path=str(tmp_configs / "CLAUDE.md"),
        )

        # First call returns sharpe=0.8 (triggers target_reached), second is validation
        results = [_FakeResult(sharpe=0.8), _FakeResult(sharpe=0.75)]
        call_idx = [0]

        def _run(df):
            r = results[min(call_idx[0], 1)]
            call_idx[0] += 1
            return r

        mock_cls = MagicMock()
        mock_cls.return_value.run.side_effect = _run

        with patch.object(loop_mod, "_BacktesterClass", mock_cls):
            summary = loop.run()

        assert summary["final_validation_metrics"] is not None
        assert "sharpe" in summary["final_validation_metrics"]

    def test_no_final_validation_when_disabled(self, loop_mod, tmp_configs):
        cfg = _make_loop_config(
            max_iterations=1, target_sharpe=0.5,
            reports_dir=str(tmp_configs / "reports"),
        )
        cfg["validation"]["run_full_dataset"] = False
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_configs / "data.parquet"),
            reports_dir=str(tmp_configs / "reports"),
            claude_md_path=str(tmp_configs / "CLAUDE.md"),
        )

        mock_cls = MagicMock()
        mock_cls.return_value.run.return_value = _FakeResult(sharpe=0.8)

        with patch.object(loop_mod, "_BacktesterClass", mock_cls):
            summary = loop.run()

        assert summary["final_validation_metrics"] is None


# ---------------------------------------------------------------------------
# 12. Graceful degradation without DB
# ---------------------------------------------------------------------------


class TestGracefulDegradationNoDB:
    def test_run_succeeds_without_db_pool(self, loop_mod, tmp_configs):
        """The loop must complete successfully even when db_pool=None."""
        cfg = _make_loop_config(
            max_iterations=1, target_sharpe=0.5,
            reports_dir=str(tmp_configs / "reports"),
        )
        cfg["persistence"]["enabled"] = True  # enabled but no pool
        cfg["validation"]["run_full_dataset"] = False
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_configs / "data.parquet"),
            reports_dir=str(tmp_configs / "reports"),
            claude_md_path=str(tmp_configs / "CLAUDE.md"),
            db_pool=None,
        )

        mock_cls = MagicMock()
        mock_cls.return_value.run.return_value = _FakeResult(sharpe=0.8)

        with patch.object(loop_mod, "_BacktesterClass", mock_cls):
            summary = loop.run()

        assert summary["stop_reason"] == "target_reached"

    def test_persist_called_with_trades_when_enabled(self, loop_mod, tmp_configs):
        """TradePersistence.persist_run is called with the trade list."""
        cfg = _make_loop_config(
            max_iterations=1, target_sharpe=0.5,
            reports_dir=str(tmp_configs / "reports"),
        )
        cfg["persistence"]["enabled"] = True
        cfg["validation"]["run_full_dataset"] = False
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_configs / "data.parquet"),
            reports_dir=str(tmp_configs / "reports"),
            claude_md_path=str(tmp_configs / "CLAUDE.md"),
            db_pool=None,
        )

        mock_cls = MagicMock()
        mock_cls.return_value.run.return_value = _FakeResult(sharpe=0.8)

        with patch.object(loop_mod, "_BacktesterClass", mock_cls):
            with patch.object(loop._trade_persistence, "persist_run") as mock_persist:
                loop.run()
        mock_persist.assert_called_once()
        _, kwargs = mock_persist.call_args
        assert "run_id" in kwargs or len(mock_persist.call_args[0]) >= 1


# ---------------------------------------------------------------------------
# 13. _summarise_diff helper
# ---------------------------------------------------------------------------


class TestSummariseDiff:
    def test_extracts_changed_keys(self, loop_mod):
        diff = textwrap.dedent("""\
            --- a/config/strategy.yaml
            +++ b/config/strategy.yaml
            @@ -1,4 +1,4 @@
            -tenkan_period: 9
            +tenkan_period: 7
            -adx_threshold: 28
            +adx_threshold: 25
        """)
        summary = loop_mod.OptimizationLoop._summarise_diff(diff)
        assert "tenkan_period" in summary
        assert "adx_threshold" in summary

    def test_fallback_for_non_yaml_diff(self, loop_mod):
        diff = "binary difference"
        summary = loop_mod.OptimizationLoop._summarise_diff(diff)
        assert summary == "config changes"

    def test_empty_diff_returns_fallback(self, loop_mod):
        summary = loop_mod.OptimizationLoop._summarise_diff("")
        assert summary == "config changes"


# ---------------------------------------------------------------------------
# 14. load_loop_config
# ---------------------------------------------------------------------------


class TestLoadLoopConfig:
    def test_loads_yaml_file(self, tmp_path, loop_mod):
        cfg_file = tmp_path / "loop.yaml"
        cfg_file.write_text("optimization:\n  max_iterations: 5\n")
        result = loop_mod.load_loop_config(cfg_file)
        assert result["optimization"]["max_iterations"] == 5

    def test_returns_empty_dict_on_missing_file(self, tmp_path, loop_mod):
        result = loop_mod.load_loop_config(tmp_path / "nonexistent.yaml")
        assert result == {}
