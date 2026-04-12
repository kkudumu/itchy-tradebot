"""Unit tests for src/optimization/guardrails.py.

Covers:
- check_consecutive_passes with all-pass, partial-fail, and edge cases
- check_permutation_significance with significant / not-significant scenarios
- _permute_candles internal helper with OHLC consistency checks
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.optimization.guardrails import (
    check_consecutive_passes,
    check_permutation_significance,
    _permute_candles,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv_df(n: int = 10_000) -> pd.DataFrame:
    """Build a realistic-ish OHLCV DataFrame for testing."""
    rng = np.random.default_rng(123)
    close = 2000.0 + np.cumsum(rng.normal(0, 0.5, n))
    # Ensure prices stay positive
    close = np.maximum(close, 100.0)
    open_ = close + rng.normal(0, 0.2, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.3, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.3, n))
    volume = rng.integers(100, 5000, n).astype(float)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=pd.date_range("2025-01-01", periods=n, freq="min"),
    )


# =============================================================================
# 1. check_consecutive_passes
# =============================================================================


class TestConsecutivePassCheck:
    """Tests for the consecutive-passes guardrail."""

    def test_all_pass_returns_true(self):
        """When backtest_fn always passes, all_passed should be True."""
        data = _make_ohlcv_df()
        result = check_consecutive_passes(
            data=data,
            config={},
            instrument="MGC",
            backtest_fn=lambda d, c, i, b: {
                "passed": True,
                "total_trades": 20,
                "total_return_pct": 5.0,
            },
        )
        assert result["all_passed"] is True
        assert len(result["attempts"]) == 3

    def test_one_fail_returns_false(self):
        """When one of the offset windows fails, all_passed should be False."""
        call_count = {"n": 0}

        def _mock_bt(d, c, i, b):
            call_count["n"] += 1
            if call_count["n"] == 2:
                return {"passed": False, "total_trades": 5, "total_return_pct": -2.0}
            return {"passed": True, "total_trades": 20, "total_return_pct": 5.0}

        data = _make_ohlcv_df()
        result = check_consecutive_passes(
            data=data, config={}, instrument="MGC", backtest_fn=_mock_bt,
        )
        assert result["all_passed"] is False
        assert any(not a["passed"] for a in result["attempts"])

    def test_all_fail_returns_false(self):
        """When every window fails, all_passed is False."""
        data = _make_ohlcv_df()
        result = check_consecutive_passes(
            data=data,
            config={},
            instrument="MGC",
            backtest_fn=lambda d, c, i, b: {
                "passed": False,
                "total_trades": 0,
                "total_return_pct": -10.0,
            },
        )
        assert result["all_passed"] is False

    def test_backtest_fn_returns_none_counts_as_fail(self):
        """If backtest_fn returns None, that window is treated as a failure."""
        data = _make_ohlcv_df()
        result = check_consecutive_passes(
            data=data,
            config={},
            instrument="MGC",
            backtest_fn=lambda d, c, i, b: None,
        )
        assert result["all_passed"] is False
        assert all(not a["passed"] for a in result["attempts"])

    def test_attempts_contain_offset_info(self):
        """Each attempt dict should record the offset used."""
        data = _make_ohlcv_df()
        result = check_consecutive_passes(
            data=data,
            config={},
            instrument="MGC",
            backtest_fn=lambda d, c, i, b: {
                "passed": True,
                "total_trades": 10,
                "total_return_pct": 3.0,
            },
        )
        for attempt in result["attempts"]:
            assert "offset" in attempt

    def test_custom_n_required(self):
        """n_required controls how many windows must pass."""
        data = _make_ohlcv_df()
        # Only 3 default offsets, so n_required=2 should still work
        call_count = {"n": 0}

        def _mock_bt(d, c, i, b):
            call_count["n"] += 1
            # Third call fails
            if call_count["n"] == 3:
                return {"passed": False, "total_trades": 0, "total_return_pct": -1.0}
            return {"passed": True, "total_trades": 10, "total_return_pct": 5.0}

        result = check_consecutive_passes(
            data=data,
            config={},
            instrument="MGC",
            backtest_fn=_mock_bt,
            n_required=2,
        )
        # 2 out of 3 passed and n_required=2
        assert result["all_passed"] is True

    def test_small_data_uses_full_range(self):
        """When data is small (< 5000 bars after offset), use full data."""
        small_data = _make_ohlcv_df(n=3000)
        result = check_consecutive_passes(
            data=small_data,
            config={},
            instrument="MGC",
            backtest_fn=lambda d, c, i, b: {
                "passed": True,
                "total_trades": 5,
                "total_return_pct": 1.0,
            },
        )
        assert result["all_passed"] is True


# =============================================================================
# 2. check_permutation_significance
# =============================================================================


class TestPermutationCheck:
    """Tests for the permutation-based significance guardrail."""

    def test_significant_returns_true(self):
        """When real return beats all permutations, significant should be True."""
        data = _make_ohlcv_df()

        def _mock_bt(d, c, i, b):
            return {
                "passed": True,
                "total_trades": 10,
                "total_return_pct": 1.0,
            }

        result = check_permutation_significance(
            real_return=18.0,
            data=data,
            config={},
            instrument="MGC",
            backtest_fn=_mock_bt,
            n_permutations=20,
        )
        assert result["significant"] is True
        assert result["p_value"] < 0.05
        assert result["real_return"] == 18.0
        assert len(result["permuted_returns"]) == 20

    def test_not_significant_when_permuted_beats_real(self):
        """When permuted returns regularly beat real, p_value is high."""
        data = _make_ohlcv_df()

        def _mock_bt(d, c, i, b):
            return {
                "passed": True,
                "total_trades": 10,
                "total_return_pct": 50.0,  # always beats real_return=1.0
            }

        result = check_permutation_significance(
            real_return=1.0,
            data=data,
            config={},
            instrument="MGC",
            backtest_fn=_mock_bt,
            n_permutations=20,
        )
        assert result["significant"] is False
        assert result["p_value"] >= 0.05

    def test_p_value_calculation(self):
        """Verify p_value = count(permuted >= real) / n_permutations."""
        data = _make_ohlcv_df()
        call_count = {"n": 0}

        def _mock_bt(d, c, i, b):
            call_count["n"] += 1
            # 5 out of 20 permutations beat the real return of 10.0
            if call_count["n"] <= 5:
                return {"passed": True, "total_trades": 10, "total_return_pct": 15.0}
            return {"passed": True, "total_trades": 10, "total_return_pct": 2.0}

        result = check_permutation_significance(
            real_return=10.0,
            data=data,
            config={},
            instrument="MGC",
            backtest_fn=_mock_bt,
            n_permutations=20,
        )
        assert result["p_value"] == 5 / 20

    def test_backtest_fn_returns_none_uses_zero(self):
        """Permutation where backtest_fn returns None treats return as 0."""
        data = _make_ohlcv_df()
        result = check_permutation_significance(
            real_return=5.0,
            data=data,
            config={},
            instrument="MGC",
            backtest_fn=lambda d, c, i, b: None,
            n_permutations=10,
        )
        assert result["significant"] is True
        assert result["p_value"] == 0.0

    def test_mean_permuted_returned(self):
        """Result should contain mean_permuted."""
        data = _make_ohlcv_df()

        def _mock_bt(d, c, i, b):
            return {"passed": True, "total_trades": 10, "total_return_pct": 3.0}

        result = check_permutation_significance(
            real_return=10.0,
            data=data,
            config={},
            instrument="MGC",
            backtest_fn=_mock_bt,
            n_permutations=10,
        )
        assert "mean_permuted" in result
        assert abs(result["mean_permuted"] - 3.0) < 0.01

    def test_custom_p_threshold(self):
        """Custom p_threshold changes the significance boundary."""
        data = _make_ohlcv_df()
        call_count = {"n": 0}

        def _mock_bt(d, c, i, b):
            call_count["n"] += 1
            # 1 out of 20 permutations beats real
            if call_count["n"] == 1:
                return {"passed": True, "total_trades": 10, "total_return_pct": 20.0}
            return {"passed": True, "total_trades": 10, "total_return_pct": 0.5}

        # p_value = 1/20 = 0.05, with threshold=0.04 → not significant
        result = check_permutation_significance(
            real_return=10.0,
            data=data,
            config={},
            instrument="MGC",
            backtest_fn=_mock_bt,
            n_permutations=20,
            p_threshold=0.04,
        )
        assert result["p_value"] == 0.05
        assert result["significant"] is False


# =============================================================================
# 3. _permute_candles internal helper
# =============================================================================


class TestPermuteCandles:
    """Tests for the OHLC candle permutation helper."""

    def test_output_shape_matches_input(self):
        df = _make_ohlcv_df(n=500)
        permuted = _permute_candles(df, seed=42)
        assert permuted.shape == df.shape

    def test_columns_preserved(self):
        df = _make_ohlcv_df(n=500)
        permuted = _permute_candles(df, seed=42)
        assert list(permuted.columns) == list(df.columns)

    def test_high_ge_open_close(self):
        """After permutation, high must be >= max(open, close) on every bar."""
        df = _make_ohlcv_df(n=1000)
        permuted = _permute_candles(df, seed=7)
        bar_max = np.maximum(permuted["open"].values, permuted["close"].values)
        assert (permuted["high"].values >= bar_max - 1e-10).all()

    def test_low_le_open_close(self):
        """After permutation, low must be <= min(open, close) on every bar."""
        df = _make_ohlcv_df(n=1000)
        permuted = _permute_candles(df, seed=7)
        bar_min = np.minimum(permuted["open"].values, permuted["close"].values)
        assert (permuted["low"].values <= bar_min + 1e-10).all()

    def test_first_close_preserved(self):
        """The first close should match the original (anchor point)."""
        df = _make_ohlcv_df(n=500)
        permuted = _permute_candles(df, seed=99)
        assert abs(permuted["close"].iloc[0] - df["close"].iloc[0]) < 1e-10

    def test_different_seeds_produce_different_output(self):
        df = _make_ohlcv_df(n=500)
        p1 = _permute_candles(df, seed=1)
        p2 = _permute_candles(df, seed=2)
        # Very unlikely that all closes match with different seeds
        assert not np.allclose(p1["close"].values, p2["close"].values)

    def test_volume_unchanged(self):
        """Volume should remain unchanged after permutation."""
        df = _make_ohlcv_df(n=500)
        permuted = _permute_candles(df, seed=42)
        assert np.allclose(permuted["volume"].values, df["volume"].values)

    def test_positive_prices(self):
        """All OHLC prices must remain positive after permutation."""
        df = _make_ohlcv_df(n=2000)
        permuted = _permute_candles(df, seed=42)
        for col in ["open", "high", "low", "close"]:
            assert (permuted[col].values > 0).all(), f"{col} has non-positive values"
