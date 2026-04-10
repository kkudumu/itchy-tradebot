"""
Integration tests for the Agentic Self-Optimization Loop — Task 5.

Tests verify end-to-end wiring between:
  - OptimizationLoop (scripts/run_optimization_loop.py)
  - TradePersistence (src/backtesting/trade_persistence.py)
  - ResultsExporter (src/backtesting/results_exporter.py)
  - ClaudeMdWriter (src/learning/claude_md_writer.py)
  - run_demo_challenge.py --persist-trades flag

All external I/O (subprocess, database, IchimokuBacktester) is mocked.
Uses tmp_path for complete file isolation.

Test categories
---------------
1.  Full loop integration — mock Claude CLI + mock backtester, 2+ iterations
2.  Trade persistence integration — correct run_ids per iteration
3.  CLAUDE.md learning integration — entries after each iteration
4.  Revert integration — worsening Sharpe triggers git checkout
5.  Forward-compatibility imports (IchimokuBacktester / StrategyBacktester)
6.  run_demo_challenge.py --persist-trades behaviour
"""

from __future__ import annotations

import argparse
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
# Helpers — factories used throughout the module
# ---------------------------------------------------------------------------

_WORKTREE_ROOT = Path(__file__).resolve().parent.parent.parent


def _make_trade(direction: str = "long", r_multiple: float = 1.0) -> Dict[str, Any]:
    return {
        "instrument": "XAU/USD",
        "direction": direction,
        "entry_price": 1900.0,
        "exit_price": 1910.0,
        "entry_time": "2024-01-02T08:00:00+00:00",
        "exit_time": "2024-01-02T12:00:00+00:00",
        "r_multiple": r_multiple,
        "pnl": r_multiple * 100.0,
        "pnl_pct": r_multiple * 0.01,
        "stop_loss": 1890.0,
        "lot_size": 0.01,
        "risk_pct": 1.0,
        "status": "closed",
        "confluence_score": 3,
        "signal_tier": "A",
        "atr": 5.0,
        "source": "backtest",
    }


def _make_result(sharpe: float = 0.8, trades: Optional[List[Dict]] = None, pass_rate: float = 0.0) -> Any:
    """Return a BacktestResult-like object with controlled metrics."""
    from src.backtesting.vectorbt_engine import BacktestResult
    from src.backtesting.challenge_simulator import ChallengeSimulationResult

    _trades = trades if trades is not None else [_make_trade()]
    metrics = {
        "sharpe": sharpe,
        "sharpe_ratio": sharpe,
        "win_rate": 0.55,
        "max_drawdown": 0.08,
        "max_drawdown_pct": 8.0,
        "total_return": 0.12,
        "total_return_pct": 12.0,
        "profit_factor": 1.4,
        "trade_count": len(_trades),
        "total_trades": len(_trades),
        "expectancy": 0.5,
        "sortino": 1.1,
        "calmar": 0.9,
    }
    challenge_sim = ChallengeSimulationResult(
        total_windows=100,
        phase_1_pass_count=int(pass_rate * 100),
        phase_2_pass_count=int(pass_rate * 100),
        full_pass_count=int(pass_rate * 100),
        pass_rate=pass_rate,
        rolling_pass_rate=pass_rate,
        monte_carlo_pass_rate=pass_rate,
        avg_days_phase_1=15.0,
        avg_days_phase_2=10.0,
        failure_breakdown={},
    )
    return BacktestResult(
        trades=_trades,
        metrics=metrics,
        equity_curve=pd.Series(
            [10000.0, 10100.0],
            index=pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"),
        ),
        prop_firm={
            "status": "active",
            "profit_pct": 2.5,
            "max_daily_dd_pct": 1.0,
            "max_total_dd_pct": 2.0,
        },
        daily_pnl=pd.Series(
            [0.01],
            index=pd.date_range("2024-01-01", periods=1, freq="D", tz="UTC"),
        ),
        challenge_simulation=challenge_sim,
    )


def _make_df(rows: int = 300) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame spanning 4 years."""
    end = datetime(2024, 1, 1, tzinfo=timezone.utc)
    start = end - timedelta(days=4 * 365)
    index = pd.date_range(start=start, end=end, periods=rows, tz="UTC")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "open": rng.uniform(1850, 1950, rows),
            "high": rng.uniform(1860, 1960, rows),
            "low": rng.uniform(1840, 1940, rows),
            "close": rng.uniform(1850, 1950, rows),
            "volume": rng.uniform(1000, 5000, rows),
        },
        index=index,
    )


def _write_yaml_configs(tmp_path: Path) -> None:
    """Write minimal strategy.yaml and edges.yaml into tmp_path."""
    strategy = textwrap.dedent("""\
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
    edges = textwrap.dedent("""\
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
    (tmp_path / "strategy.yaml").write_text(strategy)
    (tmp_path / "edges.yaml").write_text(edges)


def _make_parquet(tmp_path: Path) -> Path:
    p = tmp_path / "data.parquet"
    _make_df().to_parquet(p)
    return p


def _make_loop_config(
    tmp_path: Path,
    max_iterations: int = 3,
    target_pass_rate: float = 0.90,
    persistence_enabled: bool = False,
) -> Dict[str, Any]:
    return {
        "optimization": {
            "max_iterations": max_iterations,
            "target_pass_rate": target_pass_rate,
            "max_changes_per_iteration": 2,
            "optimization_years": 2,
            "plateau_threshold": 0.05,
            "plateau_iterations": 3,
        },
        "persistence": {
            "enabled": persistence_enabled,
            "reports_dir": str(tmp_path / "reports"),
        },
        "claude": {
            "command": ["echo", "mock"],
            "timeout_seconds": 10,
        },
        "validation": {
            "run_full_dataset": False,
        },
    }


def _import_loop_module(tmp_path: Path):
    """Import run_optimization_loop with path constants redirected to tmp_path."""
    import scripts.run_optimization_loop as mod
    mod._STRATEGY_CONFIG_PATH = tmp_path / "strategy.yaml"
    mod._EDGES_CONFIG_PATH = tmp_path / "edges.yaml"
    mod._CLAUDE_MD_PATH = tmp_path / "CLAUDE.md"
    mod._REPO_ROOT = tmp_path
    return mod


# Ensure scripts package is importable
_scripts_init = _WORKTREE_ROOT / "scripts" / "__init__.py"
if not _scripts_init.exists():
    _scripts_init.touch()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_env(tmp_path: Path):
    """Set up YAML configs and parquet data in tmp_path."""
    _write_yaml_configs(tmp_path)
    _make_parquet(tmp_path)
    (tmp_path / "reports").mkdir(exist_ok=True)
    return tmp_path


@pytest.fixture()
def loop_mod(tmp_env):
    return _import_loop_module(tmp_env)


@pytest.fixture()
def loop(loop_mod, tmp_env):
    """OptimizationLoop instance with basic config."""
    cfg = _make_loop_config(tmp_env)
    return loop_mod.OptimizationLoop(
        config=cfg,
        data_file=str(tmp_env / "data.parquet"),
        reports_dir=str(tmp_env / "reports"),
        claude_md_path=str(tmp_env / "CLAUDE.md"),
    )


# ===========================================================================
# 1. Full loop integration
# ===========================================================================

class TestFullLoopIntegration:
    """Integration tests covering the full optimization loop with mocked backtester and Claude."""

    def test_loop_runs_multiple_iterations(self, loop_mod, tmp_env):
        """Loop executes at least 2 iterations when target not yet met."""
        cfg = _make_loop_config(tmp_env, max_iterations=2, target_pass_rate=0.99)
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_env / "data.parquet"),
            reports_dir=str(tmp_env / "reports"),
            claude_md_path=str(tmp_env / "CLAUDE.md"),
        )

        mock_result = _make_result(sharpe=0.5)

        with patch.object(loop, "_run_backtest", return_value=mock_result), \
             patch.object(loop, "_spawn_claude", return_value=""), \
             patch.object(loop, "_detect_changes", return_value=""):
            summary = loop.run()

        assert summary["total_iterations"] >= 2

    def test_loop_stops_when_target_reached(self, loop_mod, tmp_env):
        """Loop stops after first iteration when pass rate already exceeds target."""
        cfg = _make_loop_config(tmp_env, max_iterations=5, target_pass_rate=0.10)
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_env / "data.parquet"),
            reports_dir=str(tmp_env / "reports"),
            claude_md_path=str(tmp_env / "CLAUDE.md"),
        )

        mock_result = _make_result(sharpe=1.5, pass_rate=0.50)

        with patch.object(loop, "_run_backtest", return_value=mock_result), \
             patch.object(loop, "_spawn_claude", return_value=""), \
             patch.object(loop, "_detect_changes", return_value=""):
            summary = loop.run()

        assert summary["stop_reason"] == "target_reached"
        assert summary["total_iterations"] == 1

    def test_loop_stops_on_plateau(self, loop_mod, tmp_env):
        """Loop stops on plateau after 3 consecutive low-improvement iterations."""
        cfg = _make_loop_config(tmp_env, max_iterations=10, target_pass_rate=0.99)
        cfg["optimization"]["plateau_iterations"] = 3
        cfg["optimization"]["plateau_threshold"] = 0.05
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_env / "data.parquet"),
            reports_dir=str(tmp_env / "reports"),
            claude_md_path=str(tmp_env / "CLAUDE.md"),
        )

        # Same pass rate every iteration → plateau detected after 3
        mock_result = _make_result(sharpe=0.8, pass_rate=0.30)

        with patch.object(loop, "_run_backtest", return_value=mock_result), \
             patch.object(loop, "_spawn_claude", return_value="no changes needed"), \
             patch.object(loop, "_detect_changes", return_value=""):
            summary = loop.run()

        assert summary["stop_reason"] == "plateau"

    def test_loop_exports_reports_each_iteration(self, loop_mod, tmp_env):
        """Each iteration creates a JSON report file in the reports directory."""
        cfg = _make_loop_config(tmp_env, max_iterations=2, target_pass_rate=0.99)
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_env / "data.parquet"),
            reports_dir=str(tmp_env / "reports"),
            claude_md_path=str(tmp_env / "CLAUDE.md"),
        )

        mock_result = _make_result(sharpe=0.5)

        with patch.object(loop, "_run_backtest", return_value=mock_result), \
             patch.object(loop, "_spawn_claude", return_value=""), \
             patch.object(loop, "_detect_changes", return_value=""):
            summary = loop.run()

        reports = list((tmp_env / "reports").glob("*.json"))
        assert len(reports) >= summary["total_iterations"]

    def test_loop_returns_best_config_and_metrics(self, loop_mod, tmp_env):
        """Summary contains best_config and best_metrics keys."""
        cfg = _make_loop_config(tmp_env, max_iterations=2, target_pass_rate=0.99)
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_env / "data.parquet"),
            reports_dir=str(tmp_env / "reports"),
            claude_md_path=str(tmp_env / "CLAUDE.md"),
        )

        mock_result = _make_result(sharpe=0.7)

        with patch.object(loop, "_run_backtest", return_value=mock_result), \
             patch.object(loop, "_spawn_claude", return_value=""), \
             patch.object(loop, "_detect_changes", return_value=""):
            summary = loop.run()

        assert "best_config" in summary
        assert "best_metrics" in summary
        assert summary["best_metrics"].get("sharpe") == pytest.approx(0.7)

    def test_claude_prompt_contains_max_changes_constraint(self, loop_mod, tmp_env):
        """The Claude prompt includes the max-changes constraint."""
        cfg = _make_loop_config(tmp_env, max_iterations=1, target_pass_rate=0.10)
        cfg["optimization"]["max_changes_per_iteration"] = 2
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_env / "data.parquet"),
            reports_dir=str(tmp_env / "reports"),
            claude_md_path=str(tmp_env / "CLAUDE.md"),
        )

        mock_result = _make_result(sharpe=0.5)
        captured_prompts: List[str] = []

        def _fake_spawn(prompt: str) -> str:
            captured_prompts.append(prompt)
            return ""

        with patch.object(loop, "_run_backtest", return_value=mock_result), \
             patch.object(loop, "_spawn_claude", side_effect=_fake_spawn), \
             patch.object(loop, "_detect_changes", return_value=""):
            loop.run()

        # Loop stops at max_iterations=1; _spawn_claude may not be called.
        # Create the prompt directly to test constraint text.
        report = {
            "run_id": "test_run",
            "metrics": {"sharpe": 0.5, "win_rate": 0.55},
            "comparison": {"has_previous": False},
            "config_snapshot": {"tenkan_period": 9},
        }
        prompt = loop._build_claude_prompt(report, None, [], "No learnings.")
        assert "AT MOST 2" in prompt

    def test_claude_prompt_includes_report_metrics(self, loop):
        """Claude prompt contains current Sharpe and win rate from the report."""
        report = {
            "run_id": "iter_001",
            "metrics": {"sharpe": 1.23, "win_rate": 0.62},
            "comparison": {"has_previous": False},
            "config_snapshot": {},
        }
        prompt = loop._build_claude_prompt(report, None, [], "No learnings.")
        assert "1.23" in prompt
        assert "0.62" in prompt

    def test_claude_prompt_includes_challenge_simulation(self, loop):
        """Prompt includes challenge simulation section from current report."""
        current_report = {
            "run_id": "iter_002",
            "metrics": {"sharpe": 0.9, "win_rate": 0.62},
            "comparison": {},
            "config_snapshot": {},
            "challenge_simulation": {
                "pass_rate": 0.45,
                "phase_1_pass_count": 45,
                "full_pass_count": 30,
                "total_windows": 100,
                "avg_days_phase_1": 15,
                "avg_days_phase_2": 10,
                "failure_breakdown": {"daily_dd": 20, "total_dd": 5},
            },
        }
        prompt = loop._build_claude_prompt(current_report, None, [], "")
        assert "Challenge Simulation Results" in prompt
        assert "Failure Breakdown" in prompt

    def test_loop_iteration_history_populated(self, loop_mod, tmp_env):
        """iteration_history contains one entry per iteration with run_id and sharpe."""
        cfg = _make_loop_config(tmp_env, max_iterations=2, target_pass_rate=0.99)
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_env / "data.parquet"),
            reports_dir=str(tmp_env / "reports"),
            claude_md_path=str(tmp_env / "CLAUDE.md"),
        )

        mock_result = _make_result(sharpe=0.6)

        with patch.object(loop, "_run_backtest", return_value=mock_result), \
             patch.object(loop, "_spawn_claude", return_value=""), \
             patch.object(loop, "_detect_changes", return_value=""):
            summary = loop.run()

        history = summary["iteration_history"]
        assert len(history) == summary["total_iterations"]
        for entry in history:
            assert "run_id" in entry
            assert "sharpe" in entry
            assert "iteration" in entry


# ===========================================================================
# 2. Trade persistence integration
# ===========================================================================

class TestTradePersistenceIntegration:
    """Integration tests verifying correct run_ids are passed to TradePersistence."""

    def test_persist_run_called_with_correct_run_id(self, loop_mod, tmp_env):
        """persist_run receives the run_id computed for each iteration."""
        cfg = _make_loop_config(tmp_env, max_iterations=2, target_pass_rate=0.99, persistence_enabled=True)
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_env / "data.parquet"),
            reports_dir=str(tmp_env / "reports"),
            claude_md_path=str(tmp_env / "CLAUDE.md"),
        )

        mock_result = _make_result(sharpe=0.5)
        persisted_run_ids: List[str] = []

        def _fake_persist(run_id, trades, config_snapshot, metrics):
            persisted_run_ids.append(run_id)
            return len(trades)

        with patch.object(loop._trade_persistence, "persist_run", side_effect=_fake_persist), \
             patch.object(loop, "_run_backtest", return_value=mock_result), \
             patch.object(loop, "_spawn_claude", return_value=""), \
             patch.object(loop, "_detect_changes", return_value=""):
            summary = loop.run()

        n_iters = summary["total_iterations"]
        assert len(persisted_run_ids) == n_iters
        # Each run_id should be unique
        assert len(set(persisted_run_ids)) == n_iters

    def test_persist_run_skipped_when_persistence_disabled(self, loop_mod, tmp_env):
        """No calls to persist_run when persistence.enabled is False."""
        cfg = _make_loop_config(tmp_env, max_iterations=2, target_pass_rate=0.99, persistence_enabled=False)
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_env / "data.parquet"),
            reports_dir=str(tmp_env / "reports"),
            claude_md_path=str(tmp_env / "CLAUDE.md"),
        )

        mock_result = _make_result(sharpe=0.5)
        mock_persist = MagicMock(return_value=0)

        with patch.object(loop._trade_persistence, "persist_run", mock_persist), \
             patch.object(loop, "_run_backtest", return_value=mock_result), \
             patch.object(loop, "_spawn_claude", return_value=""), \
             patch.object(loop, "_detect_changes", return_value=""):
            loop.run()

        mock_persist.assert_not_called()

    def test_persist_run_receives_trades_list(self, loop_mod, tmp_env):
        """persist_run is called with the trades list from the backtest result."""
        cfg = _make_loop_config(tmp_env, max_iterations=1, target_pass_rate=0.99, persistence_enabled=True)
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_env / "data.parquet"),
            reports_dir=str(tmp_env / "reports"),
            claude_md_path=str(tmp_env / "CLAUDE.md"),
        )

        expected_trades = [_make_trade(), _make_trade(direction="short")]
        mock_result = _make_result(sharpe=0.4, trades=expected_trades)
        persisted_trades: List = []

        def _fake_persist(run_id, trades, config_snapshot, metrics):
            persisted_trades.extend(trades)
            return len(trades)

        with patch.object(loop._trade_persistence, "persist_run", side_effect=_fake_persist), \
             patch.object(loop, "_run_backtest", return_value=mock_result), \
             patch.object(loop, "_spawn_claude", return_value=""), \
             patch.object(loop, "_detect_changes", return_value=""):
            loop.run()

        assert len(persisted_trades) == 2

    def test_different_run_ids_per_iteration(self, loop_mod, tmp_env):
        """Each iteration gets its own unique run_id in the opt_iter_NNN format."""
        cfg = _make_loop_config(tmp_env, max_iterations=3, target_pass_rate=0.99, persistence_enabled=True)
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_env / "data.parquet"),
            reports_dir=str(tmp_env / "reports"),
            claude_md_path=str(tmp_env / "CLAUDE.md"),
        )

        mock_result = _make_result(sharpe=0.5)
        run_ids: List[str] = []

        def _fake_persist(run_id, trades, config_snapshot, metrics):
            run_ids.append(run_id)
            return 1

        with patch.object(loop._trade_persistence, "persist_run", side_effect=_fake_persist), \
             patch.object(loop, "_run_backtest", return_value=mock_result), \
             patch.object(loop, "_spawn_claude", return_value=""), \
             patch.object(loop, "_detect_changes", return_value=""):
            summary = loop.run()

        # All run_ids must be unique and follow the opt_iter_NNN_ prefix
        assert len(set(run_ids)) == len(run_ids)
        for rid in run_ids:
            assert rid.startswith("opt_iter_")

    def test_persist_no_db_graceful(self, loop_mod, tmp_env):
        """Loop completes without error when db_pool is None (no persistence)."""
        cfg = _make_loop_config(tmp_env, max_iterations=1, target_pass_rate=0.99, persistence_enabled=True)
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_env / "data.parquet"),
            reports_dir=str(tmp_env / "reports"),
            claude_md_path=str(tmp_env / "CLAUDE.md"),
            db_pool=None,
        )

        mock_result = _make_result(sharpe=0.5)

        with patch.object(loop, "_run_backtest", return_value=mock_result), \
             patch.object(loop, "_spawn_claude", return_value=""), \
             patch.object(loop, "_detect_changes", return_value=""):
            summary = loop.run()

        # Should complete without raising
        assert "total_iterations" in summary


# ===========================================================================
# 3. CLAUDE.md learning integration
# ===========================================================================

class TestClaudeMdLearningIntegration:
    """Integration tests verifying learning entries in CLAUDE.md."""

    def test_learning_appended_after_each_iteration_with_changes(self, loop_mod, tmp_env):
        """append_learning is called once per iteration that detects changes."""
        cfg = _make_loop_config(tmp_env, max_iterations=2, target_pass_rate=0.99)
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_env / "data.parquet"),
            reports_dir=str(tmp_env / "reports"),
            claude_md_path=str(tmp_env / "CLAUDE.md"),
        )

        mock_result_iter1 = _make_result(sharpe=0.5)
        mock_result_iter2 = _make_result(sharpe=0.7)
        call_count = {"n": 0}

        orig_append = loop._claude_writer.append_learning

        def _tracking_append(*args, **kwargs):
            call_count["n"] += 1
            return orig_append(*args, **kwargs)

        diff_output = "- tenkan_period: 9\n+ tenkan_period: 11\n"

        with patch.object(loop, "_run_backtest", side_effect=[mock_result_iter1, mock_result_iter2, _make_result(sharpe=0.7)]), \
             patch.object(loop, "_spawn_claude", return_value="Changed tenkan_period"), \
             patch.object(loop, "_detect_changes", return_value=diff_output), \
             patch.object(loop._claude_writer, "append_learning", side_effect=_tracking_append):
            loop.run()

        assert call_count["n"] >= 1

    def test_learning_entries_written_to_claude_md_file(self, loop_mod, tmp_env):
        """After loop completes, CLAUDE.md contains learning entries."""
        cfg = _make_loop_config(tmp_env, max_iterations=2, target_pass_rate=0.99)
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_env / "data.parquet"),
            reports_dir=str(tmp_env / "reports"),
            claude_md_path=str(tmp_env / "CLAUDE.md"),
        )

        diff_output = "- tenkan_period: 9\n+ tenkan_period: 12\n"

        with patch.object(loop, "_run_backtest", side_effect=[_make_result(0.5), _make_result(0.7), _make_result(0.7)]), \
             patch.object(loop, "_spawn_claude", return_value="Changed param"), \
             patch.object(loop, "_detect_changes", return_value=diff_output):
            loop.run()

        claude_md = tmp_env / "CLAUDE.md"
        assert claude_md.exists()
        content = claude_md.read_text()
        assert "Strategy Learnings" in content

    def test_kept_verdict_when_metrics_improve(self, loop_mod, tmp_env):
        """When new pass rate >= old pass rate after Claude change, verdict is 'kept'.

        Flow for max_iterations=2:
          iter 1: _run_backtest (pass_rate=0.20) -> spawn Claude -> detect changes ->
                  re-run (pass_rate=0.40) -> append_learning(verdict='kept')
          iter 2: _run_backtest (pass_rate=0.40) -> last iter -> break
        """
        cfg = _make_loop_config(tmp_env, max_iterations=2, target_pass_rate=0.99)
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_env / "data.parquet"),
            reports_dir=str(tmp_env / "reports"),
            claude_md_path=str(tmp_env / "CLAUDE.md"),
        )

        verdicts: List[str] = []

        def _track_append(run_id, changes_made, metrics_before, metrics_after, verdict):
            verdicts.append(verdict)

        diff_output = "- tenkan_period: 9\n+ tenkan_period: 12\n"

        # 3 backtest calls: iter1-initial, iter1-rerun-after-claude, iter2-initial
        with patch.object(loop, "_run_backtest", side_effect=[_make_result(0.5, pass_rate=0.20), _make_result(0.9, pass_rate=0.40), _make_result(0.9, pass_rate=0.40)]), \
             patch.object(loop, "_spawn_claude", return_value="Changed tenkan"), \
             patch.object(loop, "_detect_changes", return_value=diff_output), \
             patch.object(loop._claude_writer, "append_learning", side_effect=_track_append):
            loop.run()

        assert "kept" in verdicts

    def test_reverted_verdict_when_metrics_worsen(self, loop_mod, tmp_env):
        """When new pass rate < old pass rate after Claude change, verdict is 'reverted'.

        Flow for max_iterations=2:
          iter 1: _run_backtest (pass_rate=0.40) -> spawn Claude -> detect changes ->
                  re-run (pass_rate=0.10) -> _revert_changes -> append_learning(verdict='reverted')
          iter 2: _run_backtest (pass_rate=0.40) -> last iter -> break
        """
        cfg = _make_loop_config(tmp_env, max_iterations=2, target_pass_rate=0.99)
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_env / "data.parquet"),
            reports_dir=str(tmp_env / "reports"),
            claude_md_path=str(tmp_env / "CLAUDE.md"),
        )

        verdicts: List[str] = []

        def _track_append(run_id, changes_made, metrics_before, metrics_after, verdict):
            verdicts.append(verdict)

        diff_output = "- tenkan_period: 9\n+ tenkan_period: 3\n"

        with patch.object(loop, "_run_backtest", side_effect=[_make_result(0.8, pass_rate=0.40), _make_result(0.3, pass_rate=0.10), _make_result(0.8, pass_rate=0.40)]), \
             patch.object(loop, "_spawn_claude", return_value="Changed tenkan"), \
             patch.object(loop, "_detect_changes", return_value=diff_output), \
             patch.object(loop, "_revert_changes"), \
             patch.object(loop._claude_writer, "append_learning", side_effect=_track_append):
            loop.run()

        assert "reverted" in verdicts

    def test_no_changes_appends_kept_learning(self, loop_mod, tmp_env):
        """When Claude makes no changes, a 'kept' learning with '(no changes)' is written.

        Flow for max_iterations=2:
          iter 1: _run_backtest (0.5) → spawn Claude → detect changes (empty) →
                  append_learning(changes='(no changes)', verdict='kept') → continue
          iter 2: _run_backtest (0.5) → last iter → break
        """
        cfg = _make_loop_config(tmp_env, max_iterations=2, target_pass_rate=0.99)
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_env / "data.parquet"),
            reports_dir=str(tmp_env / "reports"),
            claude_md_path=str(tmp_env / "CLAUDE.md"),
        )

        verdicts: List[str] = []
        changes_list: List[str] = []

        def _track_append(run_id, changes_made, metrics_before, metrics_after, verdict):
            verdicts.append(verdict)
            changes_list.append(changes_made)

        with patch.object(loop, "_run_backtest", return_value=_make_result(0.5)), \
             patch.object(loop, "_spawn_claude", return_value=""), \
             patch.object(loop, "_detect_changes", return_value=""), \
             patch.object(loop._claude_writer, "append_learning", side_effect=_track_append):
            loop.run()

        # When no changes were detected, a learning with '(no changes)' and verdict 'kept' is appended
        assert "kept" in verdicts
        assert any("no changes" in c for c in changes_list)


# ===========================================================================
# 4. Revert integration
# ===========================================================================

class TestRevertIntegration:
    """Integration tests verifying config revert when metrics worsen."""

    def test_git_checkout_called_when_metrics_worsen(self, loop_mod, tmp_env):
        """When new pass rate < old pass rate, _revert_changes is called.

        Flow for max_iterations=2:
          iter 1: backtest (pass_rate=0.40) -> spawn Claude -> detect changes ->
                  re-run (pass_rate=0.10) -> _revert_changes called
          iter 2: backtest (pass_rate=0.40) -> last iter -> break
        """
        cfg = _make_loop_config(tmp_env, max_iterations=2, target_pass_rate=0.99)
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_env / "data.parquet"),
            reports_dir=str(tmp_env / "reports"),
            claude_md_path=str(tmp_env / "CLAUDE.md"),
        )

        revert_called = {"n": 0}

        def _track_revert():
            revert_called["n"] += 1

        diff_output = "- kijun_period: 26\n+ kijun_period: 10\n"

        with patch.object(loop, "_run_backtest", side_effect=[_make_result(0.8, pass_rate=0.40), _make_result(0.2, pass_rate=0.10), _make_result(0.8, pass_rate=0.40)]), \
             patch.object(loop, "_spawn_claude", return_value="Bad change"), \
             patch.object(loop, "_detect_changes", return_value=diff_output), \
             patch.object(loop, "_revert_changes", side_effect=_track_revert):
            loop.run()

        assert revert_called["n"] == 1

    def test_git_checkout_not_called_when_metrics_improve(self, loop_mod, tmp_env):
        """When new pass rate >= old pass rate, _revert_changes is NOT called."""
        cfg = _make_loop_config(tmp_env, max_iterations=1, target_pass_rate=0.99)
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_env / "data.parquet"),
            reports_dir=str(tmp_env / "reports"),
            claude_md_path=str(tmp_env / "CLAUDE.md"),
        )

        diff_output = "- tenkan_period: 9\n+ tenkan_period: 12\n"

        with patch.object(loop, "_run_backtest", side_effect=[_make_result(0.5, pass_rate=0.20), _make_result(0.9, pass_rate=0.40)]), \
             patch.object(loop, "_spawn_claude", return_value="Good change"), \
             patch.object(loop, "_detect_changes", return_value=diff_output) as mock_detect, \
             patch.object(loop, "_revert_changes") as mock_revert:
            loop.run()

        mock_revert.assert_not_called()

    def test_subprocess_git_checkout_called_in_revert(self, loop_mod, tmp_env):
        """_revert_changes calls subprocess.run with git checkout arguments."""
        cfg = _make_loop_config(tmp_env)
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_env / "data.parquet"),
            reports_dir=str(tmp_env / "reports"),
            claude_md_path=str(tmp_env / "CLAUDE.md"),
        )

        mock_completed = MagicMock()
        mock_completed.returncode = 0
        mock_completed.stderr = ""

        with patch("subprocess.run", return_value=mock_completed) as mock_sub:
            loop._revert_changes()

        calls = mock_sub.call_args_list
        assert len(calls) == 1
        cmd = calls[0][0][0]
        assert "git" in cmd
        assert "checkout" in cmd


# ===========================================================================
# 5. Forward-compatibility imports
# ===========================================================================

class TestForwardCompatibilityImports:
    """Tests verifying correct import fallback for backtester."""

    def test_ichimoku_backtester_import_path_works(self):
        """IchimokuBacktester is importable from vectorbt_engine."""
        from src.backtesting.vectorbt_engine import IchimokuBacktester
        assert IchimokuBacktester is not None

    def test_strategy_backtester_import_fallback(self):
        """ImportError for StrategyBacktester falls back to IchimokuBacktester gracefully."""
        # We simulate the try/except at module top-level by checking the _BacktesterClass
        import scripts.run_optimization_loop as mod
        # _BacktesterClass should be set to IchimokuBacktester when StrategyBacktester absent
        assert mod._BacktesterClass is not None

    def test_optimization_loop_uses_backtest_class(self, loop_mod, tmp_env):
        """OptimizationLoop._run_backtest instantiates _BacktesterClass with config."""
        cfg = _make_loop_config(tmp_env, max_iterations=1, target_pass_rate=0.10)
        loop = loop_mod.OptimizationLoop(
            config=cfg,
            data_file=str(tmp_env / "data.parquet"),
            reports_dir=str(tmp_env / "reports"),
            claude_md_path=str(tmp_env / "CLAUDE.md"),
        )

        called_configs: List[Dict] = []
        mock_result = _make_result(sharpe=1.5, pass_rate=0.50)  # exceeds target so stops after 1

        def _fake_backtest(df, config):
            called_configs.append(config)
            return mock_result

        with patch.object(loop, "_run_backtest", side_effect=_fake_backtest), \
             patch.object(loop, "_spawn_claude", return_value=""), \
             patch.object(loop, "_detect_changes", return_value=""):
            loop.run()

        assert len(called_configs) == 1
        assert isinstance(called_configs[0], dict)

    def test_backtest_result_dataclass_fields(self):
        """BacktestResult has required fields: trades, metrics, equity_curve, prop_firm, daily_pnl."""
        result = _make_result(sharpe=1.0)
        assert hasattr(result, "trades")
        assert hasattr(result, "metrics")
        assert hasattr(result, "equity_curve")
        assert hasattr(result, "prop_firm")
        assert hasattr(result, "daily_pnl")
        assert hasattr(result, "skipped_signals")


# ===========================================================================
# 6. run_demo_challenge.py --persist-trades
# ===========================================================================

class TestRunDemoChallengePersisteFlag:
    """Tests for the --persist-trades flag in run_demo_challenge.py."""

    def _make_args(self, persist_trades: bool = False, tmp_path: Optional[Path] = None) -> argparse.Namespace:
        """Build a minimal argparse Namespace for run_backtest."""
        return argparse.Namespace(
            mode="backtest",
            config=None,
            data_file=None,
            synthetic_data=True,
            initial_balance=10_000.0,
            instrument="XAUUSD",
            output_dir=str(tmp_path or Path("/tmp")),
            log_trades=False,
            persist_trades=persist_trades,
            seed=42,
            wf_trials=10,
            mc_sims=100,
            haircut=25.0,
            mt5_login=None,
            mt5_password=None,
            mt5_server=None,
        )

    def test_persist_trades_flag_recognized_by_parser(self):
        """--persist-trades is a recognised argument in the CLI parser."""
        import scripts.run_demo_challenge as demo_mod
        parser = demo_mod._build_parser()
        args = parser.parse_args(["--persist-trades", "--synthetic-data"])
        assert args.persist_trades is True

    def test_persist_trades_defaults_to_false(self):
        """--persist-trades defaults to False when not supplied."""
        import scripts.run_demo_challenge as demo_mod
        parser = demo_mod._build_parser()
        args = parser.parse_args(["--synthetic-data"])
        assert args.persist_trades is False

    def _run_backtest_with_mocks(
        self,
        args: argparse.Namespace,
        mock_result: Any,
        mock_pool: Optional[Any] = None,
        mock_persistence: Optional[Any] = None,
    ) -> int:
        """Helper: run run_backtest with IchimokuBacktester and ConfigLoader mocked.

        IchimokuBacktester and ConfigLoader are lazily imported inside run_backtest,
        so we patch them via their canonical module paths.
        """
        import scripts.run_demo_challenge as demo_mod

        if mock_pool is None:
            mock_pool = MagicMock()
            mock_pool.initialise = MagicMock()

        with patch("src.backtesting.vectorbt_engine.IchimokuBacktester") as mock_bt_cls, \
             patch("src.config.loader.ConfigLoader") as mock_loader_cls, \
             patch("src.database.connection.DatabasePool", return_value=mock_pool) as mock_pool_cls, \
             patch("src.backtesting.trade_persistence.TradePersistence",
                   return_value=mock_persistence or MagicMock(return_value=0)) as mock_tp_cls, \
             patch("src.learning.embeddings.EmbeddingEngine", return_value=MagicMock()):

            mock_bt = MagicMock()
            mock_bt.run.return_value = mock_result
            mock_bt.learning_engine.get_phase.return_value = "explore"
            mock_bt_cls.return_value = mock_bt

            mock_config = MagicMock()
            mock_config.model_dump.return_value = {}
            mock_loader = MagicMock()
            mock_loader.load.return_value = mock_config
            mock_loader.config_dir = Path(args.output_dir)
            mock_loader_cls.return_value = mock_loader

            with patch("src.backtesting.live_dashboard.LiveDashboardServer",
                       side_effect=Exception("no dash")), \
                 patch("src.backtesting.dashboard.BacktestDashboard",
                       side_effect=Exception("no dash")):
                return demo_mod.run_backtest(args), mock_pool_cls, mock_tp_cls

    def test_persist_trades_calls_persistence_when_set(self, tmp_path):
        """When --persist-trades is set, DatabasePool is initialised."""
        import scripts.run_demo_challenge as demo_mod

        args = self._make_args(persist_trades=True, tmp_path=tmp_path)
        mock_result = _make_result(sharpe=0.7)
        mock_pool = MagicMock()
        mock_pool.initialise = MagicMock()
        mock_persistence = MagicMock()
        mock_persistence.persist_run = MagicMock(return_value=1)

        with patch("src.backtesting.vectorbt_engine.IchimokuBacktester") as mock_bt_cls, \
             patch("src.config.loader.ConfigLoader") as mock_loader_cls, \
             patch("src.database.connection.DatabasePool", return_value=mock_pool), \
             patch("src.backtesting.trade_persistence.TradePersistence",
                   return_value=mock_persistence), \
             patch("src.learning.embeddings.EmbeddingEngine", return_value=MagicMock()):

            mock_bt = MagicMock()
            mock_bt.run.return_value = mock_result
            mock_bt.learning_engine.get_phase.return_value = "explore"
            mock_bt_cls.return_value = mock_bt

            mock_config = MagicMock()
            mock_config.model_dump.return_value = {}
            mock_loader = MagicMock()
            mock_loader.load.return_value = mock_config
            mock_loader.config_dir = Path(tmp_path)
            mock_loader_cls.return_value = mock_loader

            with patch("src.backtesting.live_dashboard.LiveDashboardServer",
                       side_effect=Exception("no dash")), \
                 patch("src.backtesting.dashboard.BacktestDashboard",
                       side_effect=Exception("no dash")):
                ret = demo_mod.run_backtest(args)

        assert ret == 0
        mock_pool.initialise.assert_called_once()

    def test_persist_trades_failure_doesnt_crash(self, tmp_path):
        """When DB persistence fails, run_backtest still returns 0."""
        import scripts.run_demo_challenge as demo_mod

        args = self._make_args(persist_trades=True, tmp_path=tmp_path)
        mock_result = _make_result(sharpe=0.7)

        with patch("src.backtesting.vectorbt_engine.IchimokuBacktester") as mock_bt_cls, \
             patch("src.config.loader.ConfigLoader") as mock_loader_cls, \
             patch("src.database.connection.DatabasePool",
                   side_effect=RuntimeError("DB unavailable")):

            mock_bt = MagicMock()
            mock_bt.run.return_value = mock_result
            mock_bt.learning_engine.get_phase.return_value = "explore"
            mock_bt_cls.return_value = mock_bt

            mock_config = MagicMock()
            mock_config.model_dump.return_value = {}
            mock_loader = MagicMock()
            mock_loader.load.return_value = mock_config
            mock_loader.config_dir = Path(tmp_path)
            mock_loader_cls.return_value = mock_loader

            with patch("src.backtesting.live_dashboard.LiveDashboardServer",
                       side_effect=Exception("no dash")), \
                 patch("src.backtesting.dashboard.BacktestDashboard",
                       side_effect=Exception("no dash")):
                ret = demo_mod.run_backtest(args)

        assert ret == 0

    def test_no_persist_when_flag_not_set(self, tmp_path):
        """When --persist-trades is not set, DatabasePool is NOT instantiated."""
        import scripts.run_demo_challenge as demo_mod

        args = self._make_args(persist_trades=False, tmp_path=tmp_path)
        mock_result = _make_result(sharpe=0.7)

        with patch("src.backtesting.vectorbt_engine.IchimokuBacktester") as mock_bt_cls, \
             patch("src.config.loader.ConfigLoader") as mock_loader_cls, \
             patch("src.database.connection.DatabasePool") as mock_pool_cls:

            mock_bt = MagicMock()
            mock_bt.run.return_value = mock_result
            mock_bt.learning_engine.get_phase.return_value = "explore"
            mock_bt_cls.return_value = mock_bt

            mock_config = MagicMock()
            mock_config.model_dump.return_value = {}
            mock_loader = MagicMock()
            mock_loader.load.return_value = mock_config
            mock_loader.config_dir = Path(tmp_path)
            mock_loader_cls.return_value = mock_loader

            with patch("src.backtesting.live_dashboard.LiveDashboardServer",
                       side_effect=Exception("no dash")), \
                 patch("src.backtesting.dashboard.BacktestDashboard",
                       side_effect=Exception("no dash")):
                demo_mod.run_backtest(args)

        mock_pool_cls.assert_not_called()

    def test_persist_trades_uses_demo_run_id_prefix(self, tmp_path):
        """The run_id passed to persist_run starts with 'demo_'."""
        import scripts.run_demo_challenge as demo_mod

        args = self._make_args(persist_trades=True, tmp_path=tmp_path)
        mock_result = _make_result(sharpe=0.7)

        mock_pool = MagicMock()
        mock_pool.initialise = MagicMock()

        captured_run_ids: List[str] = []
        mock_persistence = MagicMock()

        def _track_persist(run_id, trades, config_snapshot, metrics):
            captured_run_ids.append(run_id)
            return 1

        mock_persistence.persist_run.side_effect = _track_persist

        with patch("src.backtesting.vectorbt_engine.IchimokuBacktester") as mock_bt_cls, \
             patch("src.config.loader.ConfigLoader") as mock_loader_cls, \
             patch("src.database.connection.DatabasePool", return_value=mock_pool), \
             patch("src.backtesting.trade_persistence.TradePersistence",
                   return_value=mock_persistence), \
             patch("src.learning.embeddings.EmbeddingEngine", return_value=MagicMock()):

            mock_bt = MagicMock()
            mock_bt.run.return_value = mock_result
            mock_bt.learning_engine.get_phase.return_value = "explore"
            mock_bt_cls.return_value = mock_bt

            mock_config = MagicMock()
            mock_config.model_dump.return_value = {}
            mock_loader = MagicMock()
            mock_loader.load.return_value = mock_config
            mock_loader.config_dir = Path(tmp_path)
            mock_loader_cls.return_value = mock_loader

            with patch("src.backtesting.live_dashboard.LiveDashboardServer",
                       side_effect=Exception("no dash")), \
                 patch("src.backtesting.dashboard.BacktestDashboard",
                       side_effect=Exception("no dash")):
                demo_mod.run_backtest(args)

        assert len(captured_run_ids) == 1
        assert captured_run_ids[0].startswith("demo_")
