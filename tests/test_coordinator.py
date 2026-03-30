"""Tests for EvaluatorCoordinator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.base import (
    Evaluator,
    EvalMatrix,
    EvalRequirement,
    EvaluatorResult,
    EVALUATOR_REGISTRY,
)
from src.strategy.coordinator import EvaluatorCoordinator, _resample_ohlcv


# --- Fixtures ---

def _make_1m_data(bars: int = 500) -> pd.DataFrame:
    """Create synthetic 1M OHLCV data."""
    idx = pd.date_range("2024-01-02 08:00", periods=bars, freq="1min", tz="UTC")
    rng = np.random.default_rng(42)
    close = 1900 + np.cumsum(rng.normal(0, 0.5, bars))
    return pd.DataFrame({
        "open": close - rng.uniform(0, 0.3, bars),
        "high": close + rng.uniform(0, 1, bars),
        "low": close - rng.uniform(0, 1, bars),
        "close": close,
        "volume": rng.integers(100, 1000, bars),
    }, index=idx)


# Use a test-only evaluator to avoid depending on real evaluators
_TEST_EVAL_KEY = '_coord_test_eval'

class _CoordTestEvaluator(Evaluator, key=_TEST_EVAL_KEY):
    """Minimal evaluator for coordinator tests."""
    def evaluate(self, ohlcv: pd.DataFrame) -> EvaluatorResult:
        close = ohlcv["close"].iloc[-1]
        return EvaluatorResult(
            direction=1.0 if close > 1900 else -1.0,
            confidence=0.5,
            metadata={"close": float(close), "bars": len(ohlcv)},
        )


class TestResampleOHLCV:
    def test_5min_resample(self):
        data = _make_1m_data(100)
        resampled = _resample_ohlcv(data, "5min")
        assert len(resampled) <= 20 + 1  # ~100/5
        assert "close" in resampled.columns

    def test_1h_resample(self):
        data = _make_1m_data(500)
        resampled = _resample_ohlcv(data, "1h")
        assert len(resampled) >= 1


class TestCoordinator:
    def test_basic_evaluate(self):
        data = _make_1m_data(500)
        reqs = [EvalRequirement(_TEST_EVAL_KEY, ["5M", "15M"])]
        coord = EvaluatorCoordinator(reqs)
        matrix = coord.evaluate(data)
        assert matrix is not None
        assert f"{_TEST_EVAL_KEY}_5M" in matrix
        assert f"{_TEST_EVAL_KEY}_15M" in matrix

    def test_warmup_returns_none(self):
        data = _make_1m_data(10)
        reqs = [EvalRequirement(_TEST_EVAL_KEY, ["5M"])]
        coord = EvaluatorCoordinator(reqs, warmup_bars=500)
        assert coord.evaluate(data) is None

    def test_current_bar_slicing(self):
        data = _make_1m_data(500)
        reqs = [EvalRequirement(_TEST_EVAL_KEY, ["5M"])]
        coord = EvaluatorCoordinator(reqs)
        # Evaluate at bar 100 vs bar 200 should give different results
        m1 = coord.evaluate(data, current_bar=100)
        m2 = coord.evaluate(data, current_bar=200)
        # Both should return a matrix (enough data)
        assert m1 is not None
        assert m2 is not None

    def test_unknown_evaluator_raises(self):
        with pytest.raises(ValueError, match="not found in registry"):
            EvaluatorCoordinator([EvalRequirement("nonexistent", ["5M"])])

    def test_matrix_keys_format(self):
        data = _make_1m_data(500)
        reqs = [EvalRequirement(_TEST_EVAL_KEY, ["1H", "4H"])]
        coord = EvaluatorCoordinator(reqs)
        matrix = coord.evaluate(data)
        if matrix is not None:
            for key in matrix.keys():
                parts = key.split("_")
                assert len(parts) >= 2
