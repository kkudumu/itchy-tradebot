"""Tests for the 64-dim context embedder (market + params + outcome)."""

import numpy as np
import pandas as pd
import pytest

from src.optimization.context_embedder import (
    embed_market,
    embed_params,
    embed_outcome,
)


# ── helpers ──────────────────────────────────────────────────────────


def _make_sample_data(n=5000, start_price=100.0):
    rng = np.random.default_rng(42)
    prices = start_price + np.cumsum(rng.normal(0, 0.5, n))
    idx = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices + rng.uniform(0, 1, n),
            "low": prices - rng.uniform(0, 1, n),
            "close": prices + rng.normal(0, 0.3, n),
            "volume": rng.integers(10, 500, n).astype(float),
        },
        index=idx,
    )


def _make_sample_params():
    return {
        "risk": {
            "initial_risk_pct": 0.5,
            "reduced_risk_pct": 0.75,
            "daily_circuit_breaker_pct": 4.5,
            "max_concurrent_positions": 3,
        },
        "exit": {
            "tp_r_multiple": 1.5,
        },
        "strategies": {
            "sss": {
                "swing_lookback_n": 2,
                "min_swing_pips": 0.5,
                "min_stop_pips": 10.0,
                "min_confluence_score": 2,
                "rr_ratio": 2.0,
                "entry_mode": "cbc_only",
                "spread_multiplier": 2.0,
            },
            "ichimoku": {
                "tenkan_period": 9,
                "adx_threshold": 20,
                "atr_stop_multiplier": 2.5,
                "min_confluence_score": 1,
                "tier_c": 1,
            },
            "asian_breakout": {
                "min_range_pips": 3,
                "max_range_pips": 80,
                "rr_ratio": 2.0,
            },
            "ema_pullback": {
                "min_ema_angle_deg": 2,
                "pullback_candles_max": 20,
                "rr_ratio": 2.0,
            },
        },
    }


def _make_sample_outcome():
    return {
        "win_rate": 0.55,
        "profit_factor": 1.8,
        "total_return_pct": 6.5,
        "sharpe_ratio": 1.2,
        "max_drawdown_pct": 3.5,
        "total_trades": 150,
        "avg_r_multiple": 0.3,
        "best_trade_r": 4.5,
        "worst_trade_r": -1.0,
        "avg_trade_duration_bars": 45,
        "passed": True,
        "final_balance": 53000,
        "distance_to_target": 500,
        "best_day_profit": 800,
        "consistency_ratio": 0.65,
        "p_value": 0.03,
        "n_permutations_beaten": 970,
        "edge_filtered_pct": 0.15,
        "signals_entered_pct": 0.40,
        "win_rate_long": 0.58,
        "win_rate_short": 0.52,
    }


# ── market embedding tests ──────────────────────────────────────────


class TestEmbedMarket:
    def test_shape_and_range(self):
        """embed_market returns shape (20,) with all values in [0, 1]."""
        data = _make_sample_data()
        vec = embed_market(
            data,
            tick_size=0.10,
            tick_value_usd=1.0,
            contract_size=10,
            point_value=1.0,
        )
        assert vec.shape == (20,), f"Expected (20,), got {vec.shape}"
        assert np.all(vec >= 0.0), f"Values below 0 found: {vec[vec < 0]}"
        assert np.all(vec <= 1.0), f"Values above 1 found: {vec[vec > 1]}"

    def test_different_data_different_embeddings(self):
        """Different market data should produce different embeddings."""
        data1 = _make_sample_data(n=5000, start_price=100.0)
        data2 = _make_sample_data(n=5000, start_price=2000.0)
        # Override the rng seed to produce genuinely different data
        rng2 = np.random.default_rng(99)
        prices2 = 2000.0 + np.cumsum(rng2.normal(0, 2.0, 5000))
        idx2 = pd.date_range("2026-01-01", periods=5000, freq="1min", tz="UTC")
        data2 = pd.DataFrame(
            {
                "open": prices2,
                "high": prices2 + rng2.uniform(0, 5, 5000),
                "low": prices2 - rng2.uniform(0, 5, 5000),
                "close": prices2 + rng2.normal(0, 1.5, 5000),
                "volume": rng2.integers(10, 500, 5000).astype(float),
            },
            index=idx2,
        )

        kwargs = dict(tick_size=0.10, tick_value_usd=1.0, contract_size=10, point_value=1.0)
        vec1 = embed_market(data1, **kwargs)
        vec2 = embed_market(data2, **kwargs)
        assert not np.allclose(vec1, vec2), "Different data should produce different embeddings"

    def test_short_data(self):
        """embed_market should handle short data (< 240 bars) gracefully."""
        data = _make_sample_data(n=100, start_price=100.0)
        vec = embed_market(
            data,
            tick_size=0.10,
            tick_value_usd=1.0,
            contract_size=10,
            point_value=1.0,
        )
        assert vec.shape == (20,)
        assert np.all(vec >= 0.0)
        assert np.all(vec <= 1.0)

    def test_no_nan_or_inf(self):
        """No NaN or inf in the output."""
        data = _make_sample_data()
        vec = embed_market(
            data,
            tick_size=0.10,
            tick_value_usd=1.0,
            contract_size=10,
            point_value=1.0,
        )
        assert not np.any(np.isnan(vec)), "NaN found in market embedding"
        assert not np.any(np.isinf(vec)), "Inf found in market embedding"


# ── strategy params embedding tests ─────────────────────────────────


class TestEmbedParams:
    def test_shape_and_range(self):
        """embed_params returns shape (24,) with all values in [0, 1]."""
        params = _make_sample_params()
        vec = embed_params(params)
        assert vec.shape == (24,), f"Expected (24,), got {vec.shape}"
        assert np.all(vec >= 0.0), f"Values below 0 found: {vec[vec < 0]}"
        assert np.all(vec <= 1.0), f"Values above 1 found: {vec[vec > 1]}"

    def test_different_params_different_embeddings(self):
        """Different params should produce different embeddings."""
        params1 = _make_sample_params()
        params2 = _make_sample_params()
        params2["risk"]["initial_risk_pct"] = 2.0
        params2["strategies"]["sss"]["rr_ratio"] = 4.0
        params2["strategies"]["asian_breakout"]["min_range_pips"] = 20
        vec1 = embed_params(params1)
        vec2 = embed_params(params2)
        assert not np.allclose(vec1, vec2), "Different params should produce different embeddings"

    def test_missing_strategy_keys(self):
        """embed_params handles missing strategy sub-keys gracefully."""
        params = {
            "risk": {"initial_risk_pct": 0.5},
            "exit": {"tp_r_multiple": 1.5},
            "strategies": {},
        }
        vec = embed_params(params)
        assert vec.shape == (24,)
        assert np.all(vec >= 0.0)
        assert np.all(vec <= 1.0)

    def test_no_nan_or_inf(self):
        """No NaN or inf in the output."""
        params = _make_sample_params()
        vec = embed_params(params)
        assert not np.any(np.isnan(vec)), "NaN found in params embedding"
        assert not np.any(np.isinf(vec)), "Inf found in params embedding"


# ── outcome fingerprint tests ───────────────────────────────────────


class TestEmbedOutcome:
    def test_shape_and_range(self):
        """embed_outcome returns shape (20,) with all values in [0, 1]."""
        outcome = _make_sample_outcome()
        vec = embed_outcome(outcome)
        assert vec.shape == (20,), f"Expected (20,), got {vec.shape}"
        assert np.all(vec >= 0.0), f"Values below 0 found: {vec[vec < 0]}"
        assert np.all(vec <= 1.0), f"Values above 1 found: {vec[vec > 1]}"

    def test_different_outcomes_different_embeddings(self):
        """Different outcomes should produce different embeddings."""
        out1 = _make_sample_outcome()
        out2 = _make_sample_outcome()
        out2["win_rate"] = 0.30
        out2["profit_factor"] = 0.8
        out2["passed"] = False
        out2["total_trades"] = 20
        vec1 = embed_outcome(out1)
        vec2 = embed_outcome(out2)
        assert not np.allclose(vec1, vec2), "Different outcomes should produce different embeddings"

    def test_missing_keys_use_defaults(self):
        """embed_outcome handles missing keys gracefully."""
        outcome = {"win_rate": 0.5, "profit_factor": 1.2}
        vec = embed_outcome(outcome)
        assert vec.shape == (20,)
        assert np.all(vec >= 0.0)
        assert np.all(vec <= 1.0)

    def test_no_nan_or_inf(self):
        """No NaN or inf in the output."""
        outcome = _make_sample_outcome()
        vec = embed_outcome(outcome)
        assert not np.any(np.isnan(vec)), "NaN found in outcome embedding"
        assert not np.any(np.isinf(vec)), "Inf found in outcome embedding"


# ── concatenation test ──────────────────────────────────────────────


class TestFullEmbedding:
    def test_64_dims(self):
        """All three embeddings concatenated = 64 dims."""
        data = _make_sample_data()
        params = _make_sample_params()
        outcome = _make_sample_outcome()

        market_vec = embed_market(
            data,
            tick_size=0.10,
            tick_value_usd=1.0,
            contract_size=10,
            point_value=1.0,
        )
        params_vec = embed_params(params)
        outcome_vec = embed_outcome(outcome)

        full = np.concatenate([market_vec, params_vec, outcome_vec])
        assert full.shape == (64,), f"Expected (64,), got {full.shape}"
        assert np.all(full >= 0.0)
        assert np.all(full <= 1.0)
