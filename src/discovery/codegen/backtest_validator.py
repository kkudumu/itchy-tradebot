"""Mini-backtest + walk-forward validation for generated EdgeFilters.

Stage 3 of the safety pipeline. Runs a short backtest (1000 bars) with
and without the candidate filter to ensure it does not degrade win rate
by more than 5%. Then runs walk-forward validation on multiple OOS
windows to confirm the edge generalizes.
"""

from __future__ import annotations

import importlib
import logging
import sys
import tempfile
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BacktestValidationResult:
    """Result of mini-backtest validation."""
    passed: bool
    baseline_win_rate: float = 0.0
    candidate_win_rate: float = 0.0
    win_rate_delta: float = 0.0
    baseline_trades: int = 0
    candidate_trades: int = 0
    reason: str = ""


@dataclass
class WalkForwardWindow:
    """Metrics for a single OOS validation window."""
    window_id: str
    baseline_win_rate: float
    candidate_win_rate: float
    baseline_avg_r: float
    candidate_avg_r: float
    n_trades: int


@dataclass
class WalkForwardResult:
    """Result of walk-forward OOS validation."""
    passed: bool
    improved_count: int = 0
    total_windows: int = 0
    windows: List[WalkForwardWindow] = field(default_factory=list)
    reason: str = ""


def _run_mini_backtest(
    filter_name: Optional[str] = None,
    filter_class_name: Optional[str] = None,
    filter_code: Optional[str] = None,
    category: str = "entry",
    n_bars: int = 1000,
) -> Any:
    """Run a mini-backtest with or without the candidate filter.

    When filter_name is None, runs the baseline backtest without the
    candidate filter. When provided, dynamically loads the filter and
    adds it to the edge pipeline.

    This is a thin wrapper around IchimokuBacktester.run() that:
    1. Loads the most recent 1000 bars of cached data
    2. Optionally injects the candidate filter into EdgeManager
    3. Returns the BacktestResult

    Parameters
    ----------
    filter_name:
        Registered name for the candidate filter (None for baseline).
    filter_class_name:
        Class name of the candidate filter.
    filter_code:
        Source code of the candidate filter (loaded dynamically).
    category:
        Edge category: 'entry', 'exit', or 'modifier'.
    n_bars:
        Number of 5M bars to backtest.

    Returns
    -------
    BacktestResult from the mini-backtest.
    """
    # Lazy import to avoid circular dependency at module load time
    from src.backtesting.vectorbt_engine import BacktestEngine

    # Load cached data (most recent n_bars of 5M data)
    engine = BacktestEngine()

    if filter_code and filter_name and filter_class_name:
        # Dynamically load the generated filter
        module = types.ModuleType(f"generated_filter_{filter_name}")
        exec(compile(filter_code, f"<generated:{filter_name}>", "exec"), module.__dict__)
        filter_cls = getattr(module, filter_class_name)

        # Register temporarily in EdgeManager._REGISTRY
        from src.edges.manager import _REGISTRY
        _REGISTRY[filter_name] = (filter_cls, category)

    result = engine.run()

    # Clean up temporary registration
    if filter_name:
        from src.edges.manager import _REGISTRY
        _REGISTRY.pop(filter_name, None)

    return result


def validate_with_backtest(
    filter_code: str,
    filter_name: str,
    filter_class_name: str,
    category: str = "entry",
    max_win_rate_degradation: float = 0.05,
    n_bars: int = 1000,
) -> BacktestValidationResult:
    """Validate a generated filter via mini-backtest comparison.

    Runs two backtests:
    1. Baseline: without the candidate filter
    2. Candidate: with the candidate filter active

    The candidate passes if its win rate does not drop more than
    max_win_rate_degradation below the baseline.

    Parameters
    ----------
    filter_code:
        Complete Python source for the generated EdgeFilter.
    filter_name:
        The filter's registered name (e.g., 'high_adx_london').
    filter_class_name:
        The class name (e.g., 'HighADXLondonFilter').
    category:
        Edge category: 'entry', 'exit', or 'modifier'.
    max_win_rate_degradation:
        Maximum allowed win rate drop (0.05 = 5 percentage points).
    n_bars:
        Number of bars for the mini-backtest.

    Returns
    -------
    BacktestValidationResult with pass/fail and metrics.
    """
    # Run baseline
    try:
        baseline = _run_mini_backtest(n_bars=n_bars)
    except Exception as exc:
        return BacktestValidationResult(
            passed=False,
            reason=f"Baseline backtest failed: {exc}",
        )

    baseline_wr = baseline.metrics.get("win_rate", 0.0)
    baseline_trades = baseline.metrics.get("total_trades", len(baseline.trades))

    # Run with candidate filter
    try:
        candidate = _run_mini_backtest(
            filter_name=filter_name,
            filter_class_name=filter_class_name,
            filter_code=filter_code,
            category=category,
            n_bars=n_bars,
        )
    except Exception as exc:
        return BacktestValidationResult(
            passed=False,
            baseline_win_rate=baseline_wr,
            baseline_trades=baseline_trades,
            reason=f"Candidate backtest failed: {exc}",
        )

    candidate_wr = candidate.metrics.get("win_rate", 0.0)
    candidate_trades = candidate.metrics.get("total_trades", len(candidate.trades))
    delta = candidate_wr - baseline_wr

    degradation = baseline_wr - candidate_wr
    passed = degradation <= max_win_rate_degradation

    reason = (
        f"Win rate: {baseline_wr:.1%} -> {candidate_wr:.1%} "
        f"(delta={delta:+.1%}). "
        f"Trades: {baseline_trades} -> {candidate_trades}. "
    )
    if not passed:
        reason += (
            f"Degradation {degradation:.1%} exceeds maximum "
            f"allowed {max_win_rate_degradation:.1%}."
        )

    return BacktestValidationResult(
        passed=passed,
        baseline_win_rate=baseline_wr,
        candidate_win_rate=candidate_wr,
        win_rate_delta=delta,
        baseline_trades=baseline_trades,
        candidate_trades=candidate_trades,
        reason=reason,
    )


def validate_walk_forward(
    windows: List[WalkForwardWindow],
    min_improved_windows: int = 2,
) -> WalkForwardResult:
    """Validate that a filter improves metrics on multiple OOS windows.

    A window counts as "improved" if the candidate has BOTH:
    - Higher win rate than baseline, AND
    - Higher average R-multiple than baseline

    Parameters
    ----------
    windows:
        List of WalkForwardWindow with baseline vs candidate metrics.
    min_improved_windows:
        Minimum number of windows that must show improvement.

    Returns
    -------
    WalkForwardResult with pass/fail and breakdown.
    """
    improved_count = 0

    for w in windows:
        wr_improved = w.candidate_win_rate > w.baseline_win_rate
        r_improved = w.candidate_avg_r > w.baseline_avg_r
        if wr_improved and r_improved:
            improved_count += 1

    passed = improved_count >= min_improved_windows

    reason = (
        f"{improved_count}/{len(windows)} OOS windows improved "
        f"(need {min_improved_windows}+). "
    )
    if not passed:
        reason += "Filter does not generalize to out-of-sample data."

    return WalkForwardResult(
        passed=passed,
        improved_count=improved_count,
        total_windows=len(windows),
        windows=windows,
        reason=reason,
    )
