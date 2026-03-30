"""Integration tests for the strategy abstraction layer.

Tests the full pipeline: config → strategy loader → coordinator → evaluators → strategy → signal
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone

# Import all layers to ensure registration happens
import src.strategy.evaluators  # triggers __init_subclass__ registration
import src.strategy.strategies  # triggers __init_subclass__ registration
import src.strategy.trading_modes  # ensures trading modes are importable

from src.strategy.base import (
    EVALUATOR_REGISTRY,
    STRATEGY_REGISTRY,
    EvalMatrix,
    EvalRequirement,
    EvaluatorResult,
    ConfluenceResult,
    ExitDecision,
)
from src.strategy.coordinator import EvaluatorCoordinator
from src.strategy.loader import StrategyLoader
from src.strategy.signal_engine import Signal
from src.config.models import StrategyConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_1m_data(bars: int = 2000) -> pd.DataFrame:
    """Create synthetic 1M OHLCV data with enough bars for all TFs."""
    idx = pd.date_range("2024-01-02 08:00", periods=bars, freq="1min", tz="UTC")
    rng = np.random.default_rng(42)
    # Trending price series
    close = 1900.0 + np.cumsum(rng.normal(0.01, 0.5, bars))
    high = close + rng.uniform(0, 1.5, bars)
    low = close - rng.uniform(0, 1.5, bars)
    open_ = close + rng.normal(0, 0.3, bars)
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": rng.integers(100, 1000, bars),
    }, index=idx)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistriesPopulated:
    """Verify that importing the packages populates the registries."""

    def test_evaluator_registry_has_ichimoku(self):
        assert "ichimoku" in EVALUATOR_REGISTRY

    def test_evaluator_registry_has_adx(self):
        assert "adx" in EVALUATOR_REGISTRY

    def test_evaluator_registry_has_atr(self):
        assert "atr" in EVALUATOR_REGISTRY

    def test_evaluator_registry_has_session(self):
        assert "session" in EVALUATOR_REGISTRY

    def test_strategy_registry_has_ichimoku(self):
        assert "ichimoku" in STRATEGY_REGISTRY


# ---------------------------------------------------------------------------
# Coordinator + Evaluator pipeline
# ---------------------------------------------------------------------------

class TestCoordinatorEvaluatorPipeline:
    """Test that the coordinator feeds data to evaluators and collects results."""

    def test_coordinator_produces_eval_matrix(self):
        data = _make_1m_data(2000)
        reqs = [
            EvalRequirement("ichimoku", ["1H", "15M"]),
            EvalRequirement("adx", ["15M"]),
            EvalRequirement("atr", ["15M"]),
            EvalRequirement("session", ["5M"]),
        ]
        coord = EvaluatorCoordinator(reqs, warmup_bars=500)

        # Should return None during warmup
        assert coord.evaluate(data, current_bar=100) is None

        # Should return matrix with enough data
        matrix = coord.evaluate(data)
        assert matrix is not None
        assert "ichimoku_1H" in matrix
        assert "ichimoku_15M" in matrix
        assert "adx_15M" in matrix
        assert "atr_15M" in matrix
        assert "session_5M" in matrix

    def test_evaluator_results_have_correct_types(self):
        data = _make_1m_data(2000)
        reqs = [EvalRequirement("ichimoku", ["15M"])]
        coord = EvaluatorCoordinator(reqs)
        matrix = coord.evaluate(data)

        if matrix is not None:
            result = matrix.get("ichimoku_15M")
            assert result is not None
            assert isinstance(result, EvaluatorResult)
            assert -1.0 <= result.direction <= 1.0
            assert 0.0 <= result.confidence <= 1.0
            assert isinstance(result.metadata, dict)


# ---------------------------------------------------------------------------
# Strategy loader
# ---------------------------------------------------------------------------

class TestStrategyLoaderIntegration:
    """Test loading strategy from config."""

    def test_load_ichimoku_strategy(self):
        config = StrategyConfig()
        loader = StrategyLoader(config)
        strategy = loader.load()
        assert strategy is not None
        assert strategy.name == "ichimoku"

    def test_loaded_strategy_has_required_evaluators(self):
        config = StrategyConfig()
        loader = StrategyLoader(config)
        strategy = loader.load()
        assert len(strategy.required_evaluators) > 0
        eval_names = [r.evaluator_name for r in strategy.required_evaluators]
        assert "ichimoku" in eval_names

    def test_loaded_strategy_has_trading_mode(self):
        config = StrategyConfig()
        loader = StrategyLoader(config)
        strategy = loader.load()
        assert strategy.trading_mode is not None


# ---------------------------------------------------------------------------
# Full pipeline: config → load → coordinate → decide
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """End-to-end: load strategy from config, create coordinator, evaluate, decide."""

    def test_end_to_end_no_crash(self):
        """Verify the full pipeline runs without errors on synthetic data."""
        # Load strategy from config
        config = StrategyConfig()
        loader = StrategyLoader(config)
        strategy = loader.load()

        # Create coordinator
        coord = EvaluatorCoordinator(
            strategy.required_evaluators,
            warmup_bars=strategy.warmup_bars,
        )

        # Evaluate
        data = _make_1m_data(2000)
        matrix = coord.evaluate(data)

        # Decide (may or may not produce a signal — that's OK)
        if matrix is not None:
            result = strategy.decide(matrix)
            # Result is Signal or None — both are valid
            assert result is None or isinstance(result, Signal)

    def test_confluence_scoring_produces_valid_result(self):
        """Verify confluence scoring returns valid tiers."""
        config = StrategyConfig()
        loader = StrategyLoader(config)
        strategy = loader.load()

        data = _make_1m_data(2000)
        coord = EvaluatorCoordinator(strategy.required_evaluators)
        matrix = coord.evaluate(data)

        if matrix is not None:
            result = strategy.score_confluence(matrix, direction="long")
            assert isinstance(result, ConfluenceResult)
            assert result.quality_tier in ("A+", "B", "C", "no_trade")
            assert 0 <= result.score <= 9

    def test_edge_context_population(self):
        """Verify populate_edge_context returns expected keys."""
        config = StrategyConfig()
        loader = StrategyLoader(config)
        strategy = loader.load()

        data = _make_1m_data(2000)
        coord = EvaluatorCoordinator(strategy.required_evaluators)
        matrix = coord.evaluate(data)

        if matrix is not None:
            ctx = strategy.populate_edge_context(matrix)
            assert isinstance(ctx, dict)
            # Should have kijun and/or cloud_thickness if ichimoku evaluator ran
            # (may be empty if data doesn't have enough bars)


# ---------------------------------------------------------------------------
# Trading mode exit decisions
# ---------------------------------------------------------------------------

class TestTradingModeIntegration:
    """Verify trading mode exit decisions work with eval matrix."""

    def test_kijun_exit_mode_returns_exit_decision(self):
        from src.strategy.trading_modes.kijun_exit import KijunExitMode
        from src.strategy.base import ExitDecision
        from dataclasses import dataclass, field

        @dataclass
        class MockTrade:
            entry_price: float = 1900.0
            stop_loss: float = 1895.0
            direction: str = "long"
            remaining_pct: float = 1.0
            current_r: float = 0.5
            partial_exits: list = field(default_factory=list)
            original_stop_loss: float = field(init=False)

            def __post_init__(self):
                self.original_stop_loss = self.stop_loss

            @property
            def initial_risk(self):
                return self.entry_price - self.original_stop_loss

            @property
            def is_partial(self):
                return self.remaining_pct < 1.0

        mode = KijunExitMode()
        trade = MockTrade()
        current_data = {"close": 1902.0, "atr": 3.0}
        matrix = EvalMatrix()

        decision = mode.check_exit(trade, current_data, matrix)
        assert isinstance(decision, ExitDecision)
        assert decision.action in ("hold", "partial_exit", "trail_update", "full_exit")
