"""
Unit tests for the learning module — no database required.

Tests cover:
  1.  Feature vector dimensions (exactly 64)
  2.  Continuous value normalisation clamped to [0, 1]
  3.  Ordinal encoding: -1/0/1 → 0/0.5/1.0
  4.  Circular encoding: hour 0 and 24 produce the same sin/cos
  5.  Boolean encoding: True → 1.0, False → 0.0
  6.  Missing feature keys default to 0
  7.  Two identical contexts produce cosine similarity 1.0
  8.  Confidence ramp-up: 0→0, 10→0.5, 20→1.0, 100→1.0
  9.  Performance stats: known trade list → correct win_rate, avg_r, expectancy
  10. Batch embedding: produces same arrays as individual calls
  11. Vector stability: same input always yields same output
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
import pytest

from src.learning.feature_vector import FeatureVectorBuilder
from src.learning.embeddings import EmbeddingEngine
from src.learning.similarity import SimilaritySearch, SimilarTrade, PerformanceStats


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def builder() -> FeatureVectorBuilder:
    return FeatureVectorBuilder()


@pytest.fixture
def engine() -> EmbeddingEngine:
    return EmbeddingEngine()


@pytest.fixture
def full_context() -> Dict[str, Any]:
    """A reasonably complete context that exercises all feature groups."""
    return {
        # Ichimoku
        "cloud_direction_4h": 1,
        "cloud_direction_1h": 1,
        "tk_cross_15m": 1,
        "chikou_confirmation": True,
        "cloud_thickness_4h": 5.0,
        "cloud_thickness_1h": 3.0,
        "cloud_position_15m": 1,
        "cloud_position_5m": 1,
        "kijun_distance_5m": 2.0,
        "tenkan_kijun_spread": 1.0,
        "cloud_twist_15m": False,
        "cloud_breakout_15m": True,
        # Trend / Momentum
        "adx_value": 35.0,
        "rsi_value": 58.0,
        "bb_width_percentile": 0.6,
        "bb_squeeze": False,
        "atr_value": 10.0,
        "atr_ma": 9.0,
        # Zones
        "nearest_sr_distance": 8.0,
        "zone_confluence_score": 3,
        "in_supply_zone": False,
        "in_demand_zone": True,
        "at_pivot_level": False,
        "zone_strength": 0.7,
        "zone_freshness": "active",
        # Session / Time
        "session": "london",
        "hour": 9.0,
        "day_of_week": 2.0,
        # Signal
        "confluence_score": 6,
        "signal_tier": "B",
        "direction": "long",
        "risk_pct": 1.0,
        "atr_stop_distance": 15.0,
        # Market regime
        "volatility_regime": "normal",
        "spread": 0.5,
        "daily_range": 30.0,
    }


# ---------------------------------------------------------------------------
# 1. Feature vector dimensions
# ---------------------------------------------------------------------------

class TestVectorDimensions:
    def test_empty_context_is_64_dim(self, builder):
        vec = builder.build({})
        assert vec.shape == (64,), f"Expected shape (64,), got {vec.shape}"

    def test_full_context_is_64_dim(self, builder, full_context):
        vec = builder.build(full_context)
        assert vec.shape == (64,), f"Expected shape (64,), got {vec.shape}"

    def test_vector_dim_constant(self):
        assert FeatureVectorBuilder.VECTOR_DIM == 64

    def test_dtype_is_float64(self, builder, full_context):
        vec = builder.build(full_context)
        assert vec.dtype == np.float64


# ---------------------------------------------------------------------------
# 2. Continuous value normalisation
# ---------------------------------------------------------------------------

class TestNormalisation:
    def test_values_in_unit_interval(self, builder, full_context):
        vec = builder.build(full_context)
        assert np.all(vec >= 0.0), f"Below 0: indices {np.where(vec < 0.0)}"
        assert np.all(vec <= 1.0), f"Above 1: indices {np.where(vec > 1.0)}"

    def test_empty_context_all_zeros_or_default(self, builder):
        """Empty context should not produce NaN or out-of-range values."""
        vec = builder.build({})
        assert np.all(np.isfinite(vec))
        assert np.all(vec >= 0.0)
        assert np.all(vec <= 1.0)

    def test_normalize_continuous_clamps_above(self, builder):
        result = builder._normalize_continuous(200.0, 0.0, 100.0)
        assert result == pytest.approx(1.0)

    def test_normalize_continuous_clamps_below(self, builder):
        result = builder._normalize_continuous(-10.0, 0.0, 100.0)
        assert result == pytest.approx(0.0)

    def test_normalize_continuous_midpoint(self, builder):
        result = builder._normalize_continuous(50.0, 0.0, 100.0)
        assert result == pytest.approx(0.5)

    def test_normalize_continuous_equal_bounds(self, builder):
        """When min == max the function should return 0 without raising."""
        result = builder._normalize_continuous(5.0, 5.0, 5.0)
        assert result == 0.0

    def test_adx_normalisation(self, builder):
        ctx_high_adx = {"adx_value": 100.0, "atr_value": 1.0}
        ctx_zero_adx = {"adx_value": 0.0, "atr_value": 1.0}
        vec_high = builder.build(ctx_high_adx)
        vec_zero = builder.build(ctx_zero_adx)
        assert vec_high[15] == pytest.approx(1.0)
        assert vec_zero[15] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. Ordinal encoding
# ---------------------------------------------------------------------------

class TestOrdinalEncoding:
    def test_cloud_direction_bullish(self, builder):
        vec = builder.build({"cloud_direction_4h": 1})
        assert vec[0] == pytest.approx(1.0)

    def test_cloud_direction_bearish(self, builder):
        vec = builder.build({"cloud_direction_4h": -1})
        assert vec[0] == pytest.approx(0.0)

    def test_cloud_direction_neutral(self, builder):
        vec = builder.build({"cloud_direction_4h": 0})
        assert vec[0] == pytest.approx(0.5)

    def test_encode_ordinal_direct(self, builder):
        mapping = {-1: 0.0, 0: 0.5, 1: 1.0}
        assert builder._encode_ordinal(-1, mapping) == pytest.approx(0.0)
        assert builder._encode_ordinal(0, mapping)  == pytest.approx(0.5)
        assert builder._encode_ordinal(1, mapping)  == pytest.approx(1.0)

    def test_encode_ordinal_string_values(self, builder):
        """String "-1", "0", "1" should be coerced to int."""
        mapping = {-1: 0.0, 0: 0.5, 1: 1.0}
        assert builder._encode_ordinal("-1", mapping) == pytest.approx(0.0)
        assert builder._encode_ordinal("1", mapping)  == pytest.approx(1.0)

    def test_encode_ordinal_missing_key(self, builder):
        mapping = {-1: 0.0, 1: 1.0}
        assert builder._encode_ordinal(99, mapping) == 0.0

    def test_encode_ordinal_none(self, builder):
        mapping = {-1: 0.0, 0: 0.5, 1: 1.0}
        assert builder._encode_ordinal(None, mapping) == 0.0

    def test_tk_cross_all_values(self, builder):
        for raw, expected in [(-1, 0.0), (0, 0.5), (1, 1.0)]:
            vec = builder.build({"tk_cross_15m": raw})
            assert vec[2] == pytest.approx(expected), f"Failed for tk_cross={raw}"


# ---------------------------------------------------------------------------
# 4. Circular encoding
# ---------------------------------------------------------------------------

class TestCircularEncoding:
    def test_hour_0_and_24_are_identical(self, builder):
        """Hours 0 and 24 should produce the same circular features."""
        sin0, cos0 = builder._encode_circular(0.0, 24.0)
        sin24, cos24 = builder._encode_circular(24.0, 24.0)
        assert sin0 == pytest.approx(sin24, abs=1e-10)
        assert cos0 == pytest.approx(cos24, abs=1e-10)

    def test_hour_stored_in_vector(self, builder):
        """Hour 0 → sin=0, cos=1 → stored as (0+1)/2=0.5, (1+1)/2=1.0."""
        vec = builder.build({"hour": 0.0})
        assert vec[39] == pytest.approx(0.5, abs=1e-10)  # (sin(0)+1)/2
        assert vec[40] == pytest.approx(1.0, abs=1e-10)  # (cos(0)+1)/2

    def test_hour_12_opposite_to_0(self, builder):
        """Hour 12 is the anti-phase of hour 0."""
        vec0  = builder.build({"hour": 0.0})
        vec12 = builder.build({"hour": 12.0})
        # sin(π) ≈ 0 → stored as 0.5; cos(π) = -1 → stored as 0.0
        assert vec12[39] == pytest.approx(0.5, abs=1e-10)
        assert vec12[40] == pytest.approx(0.0, abs=1e-10)

    def test_circular_period_5_day_of_week(self, builder):
        sin0, cos0 = builder._encode_circular(0.0, 5.0)
        sin5, cos5 = builder._encode_circular(5.0, 5.0)
        assert sin0 == pytest.approx(sin5, abs=1e-10)
        assert cos0 == pytest.approx(cos5, abs=1e-10)

    def test_circular_returns_unit_norm(self, builder):
        """sin² + cos² must equal 1 for all inputs."""
        for val in [0, 6, 12, 18, 23]:
            s, c = builder._encode_circular(float(val), 24.0)
            assert s**2 + c**2 == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 5. Boolean encoding
# ---------------------------------------------------------------------------

class TestBooleanEncoding:
    def test_true_encodes_to_one(self, builder):
        assert builder._encode_bool(True) == 1.0

    def test_false_encodes_to_zero(self, builder):
        assert builder._encode_bool(False) == 0.0

    def test_none_encodes_to_zero(self, builder):
        assert builder._encode_bool(None) == 0.0

    def test_integer_one_encodes_to_one(self, builder):
        assert builder._encode_bool(1) == 1.0

    def test_integer_zero_encodes_to_zero(self, builder):
        assert builder._encode_bool(0) == 0.0

    def test_string_true(self, builder):
        assert builder._encode_bool("true") == 1.0
        assert builder._encode_bool("True") == 1.0

    def test_string_false(self, builder):
        assert builder._encode_bool("false") == 0.0

    def test_chikou_in_vector(self, builder):
        vec_true  = builder.build({"chikou_confirmation": True})
        vec_false = builder.build({"chikou_confirmation": False})
        assert vec_true[3]  == pytest.approx(1.0)
        assert vec_false[3] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 6. Missing features default to 0
# ---------------------------------------------------------------------------

class TestMissingFeatures:
    def test_all_ichimoku_default_to_zero_or_neutral(self, builder):
        vec = builder.build({})
        # Without atr context, kijun/tenkan distances default to 0/atr=1 → mid
        # Cloud directions without value default to 0 (mapping[None]=0)
        assert vec[3]  == pytest.approx(0.0)   # chikou_confirmation
        assert vec[10] == pytest.approx(0.0)   # cloud_twist_15m
        assert vec[11] == pytest.approx(0.0)   # cloud_breakout_15m

    def test_session_defaults_to_no_session(self, builder):
        vec = builder.build({})
        assert vec[35] == pytest.approx(0.0)   # london
        assert vec[36] == pytest.approx(0.0)   # ny
        assert vec[37] == pytest.approx(0.0)   # asian
        assert vec[38] == pytest.approx(0.0)   # overlap

    def test_reserved_dims_are_zero(self, builder, full_context):
        vec = builder.build(full_context)
        reserved = [12, 13, 14, 23, 24, 32, 33, 34, 43, 44, 50, 51, 52, 53, 54, 59, 60, 61, 62, 63]
        for dim in reserved:
            assert vec[dim] == pytest.approx(0.0), f"Reserved dim {dim} is non-zero: {vec[dim]}"

    def test_partial_context_no_exception(self, builder):
        """Partial context with only a few keys should not raise."""
        partial = {"adx_value": 30.0, "session": "london"}
        vec = builder.build(partial)
        assert vec.shape == (64,)
        assert np.all(np.isfinite(vec))


# ---------------------------------------------------------------------------
# 7. Cosine similarity for identical contexts
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_contexts_similarity_one(self, engine, full_context):
        emb_a = engine.create_embedding(full_context)
        emb_b = engine.create_embedding(full_context)
        sim = SimilaritySearch.cosine_similarity(emb_a, emb_b)
        assert sim == pytest.approx(1.0, abs=1e-10)

    def test_zero_vector_similarity(self):
        a = np.zeros(64)
        b = np.zeros(64)
        sim = SimilaritySearch.cosine_similarity(a, b)
        assert sim == 0.0

    def test_orthogonal_similarity(self):
        a = np.zeros(64)
        b = np.zeros(64)
        a[0] = 1.0
        b[1] = 1.0
        sim = SimilaritySearch.cosine_similarity(a, b)
        assert sim == pytest.approx(0.0, abs=1e-10)

    def test_different_contexts_similarity_less_than_one(self, engine):
        ctx_long  = {"cloud_direction_4h": 1, "direction": "long",  "adx_value": 40.0}
        ctx_short = {"cloud_direction_4h": -1, "direction": "short", "adx_value": 20.0}
        emb_l = engine.create_embedding(ctx_long)
        emb_s = engine.create_embedding(ctx_short)
        sim = SimilaritySearch.cosine_similarity(emb_l, emb_s)
        assert sim < 1.0

    def test_pgvector_literal_format(self):
        vec = np.array([0.1, 0.2, 0.3])
        literal = SimilaritySearch._to_pgvector_literal(vec)
        assert literal.startswith("[")
        assert literal.endswith("]")
        parts = literal[1:-1].split(",")
        assert len(parts) == 3
        assert float(parts[0]) == pytest.approx(0.1, abs=1e-6)


# ---------------------------------------------------------------------------
# 8. Confidence ramp-up
# ---------------------------------------------------------------------------

class TestConfidenceRamp:
    def test_zero_trades_zero_confidence(self):
        assert SimilaritySearch.get_confidence(0) == pytest.approx(0.0)

    def test_ten_trades_half_confidence(self):
        assert SimilaritySearch.get_confidence(10) == pytest.approx(0.5)

    def test_twenty_trades_full_confidence(self):
        assert SimilaritySearch.get_confidence(20) == pytest.approx(1.0)

    def test_hundred_trades_capped_at_one(self):
        assert SimilaritySearch.get_confidence(100) == pytest.approx(1.0)

    def test_one_trade_low_confidence(self):
        assert SimilaritySearch.get_confidence(1) == pytest.approx(1.0 / 20.0)

    def test_confidence_monotone_increasing(self):
        values = [SimilaritySearch.get_confidence(n) for n in range(0, 25)]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1]

    def test_negative_input_returns_zero(self):
        """Negative trade counts should be treated as 0."""
        assert SimilaritySearch.get_confidence(-5) == 0.0


# ---------------------------------------------------------------------------
# 9. Performance statistics
# ---------------------------------------------------------------------------

def _make_trades(specs: List[tuple]) -> List[SimilarTrade]:
    """Build SimilarTrade list from (r_multiple, win) tuples."""
    return [
        SimilarTrade(trade_id=i, similarity=0.85, r_multiple=r, win=w)
        for i, (r, w) in enumerate(specs)
    ]


class TestPerformanceStats:
    def test_empty_trades_returns_zeroes(self):
        searcher = SimilaritySearch()
        stats = searcher.get_performance_stats([])
        assert stats.win_rate    == pytest.approx(0.0)
        assert stats.avg_r       == pytest.approx(0.0)
        assert stats.expectancy  == pytest.approx(0.0)
        assert stats.n_trades    == 0
        assert stats.confidence  == pytest.approx(0.0)

    def test_all_winners(self):
        trades = _make_trades([(2.0, True), (3.0, True), (1.5, True)])
        searcher = SimilaritySearch()
        stats = searcher.get_performance_stats(trades)
        assert stats.win_rate   == pytest.approx(1.0)
        assert stats.avg_r      == pytest.approx((2.0 + 3.0 + 1.5) / 3)
        assert stats.avg_win_r  == pytest.approx((2.0 + 3.0 + 1.5) / 3)
        assert stats.avg_loss_r == pytest.approx(0.0)

    def test_all_losers(self):
        trades = _make_trades([(-1.0, False), (-0.5, False)])
        searcher = SimilaritySearch()
        stats = searcher.get_performance_stats(trades)
        assert stats.win_rate   == pytest.approx(0.0)
        assert stats.avg_r      == pytest.approx((-1.0 + -0.5) / 2)
        assert stats.avg_win_r  == pytest.approx(0.0)
        assert stats.avg_loss_r == pytest.approx((-1.0 + -0.5) / 2)

    def test_mixed_win_rate(self):
        # 3 wins at 2R, 2 losses at -1R
        trades = _make_trades([
            (2.0, True), (2.0, True), (2.0, True),
            (-1.0, False), (-1.0, False),
        ])
        searcher = SimilaritySearch()
        stats = searcher.get_performance_stats(trades)
        assert stats.win_rate   == pytest.approx(0.6)
        assert stats.n_trades   == 5
        assert stats.avg_win_r  == pytest.approx(2.0)
        assert stats.avg_loss_r == pytest.approx(-1.0)

    def test_expectancy_formula(self):
        """Expectancy = win_rate * avg_win + loss_rate * avg_loss."""
        trades = _make_trades([
            (2.0, True), (2.0, True),
            (-1.0, False),
        ])
        searcher = SimilaritySearch()
        stats = searcher.get_performance_stats(trades)
        expected_expectancy = (2/3) * 2.0 + (1/3) * (-1.0)
        assert stats.expectancy == pytest.approx(expected_expectancy, rel=1e-6)

    def test_confidence_from_n_trades(self):
        trades = _make_trades([(1.0, True)] * 10)
        searcher = SimilaritySearch()
        stats = searcher.get_performance_stats(trades)
        assert stats.confidence == pytest.approx(0.5)

    def test_n_trades_count(self):
        trades = _make_trades([(1.0, True)] * 7)
        searcher = SimilaritySearch()
        stats = searcher.get_performance_stats(trades)
        assert stats.n_trades == 7


# ---------------------------------------------------------------------------
# 10. Batch embedding
# ---------------------------------------------------------------------------

class TestBatchEmbedding:
    def test_batch_matches_individual(self, engine, full_context):
        ctx2 = {"adx_value": 20.0, "direction": "short", "cloud_direction_4h": -1}
        contexts = [full_context, ctx2, {}]

        individual = [engine.create_embedding(c) for c in contexts]
        batch      = engine.batch_embed(contexts)

        assert len(batch) == len(contexts)
        for i, (ind, bat) in enumerate(zip(individual, batch)):
            np.testing.assert_array_equal(ind, bat, err_msg=f"Mismatch at index {i}")

    def test_batch_empty_list(self, engine):
        result = engine.batch_embed([])
        assert result == []

    def test_batch_single_context(self, engine, full_context):
        result = engine.batch_embed([full_context])
        assert len(result) == 1
        assert result[0].shape == (64,)


# ---------------------------------------------------------------------------
# 11. Vector stability (determinism)
# ---------------------------------------------------------------------------

class TestVectorStability:
    def test_same_input_same_output(self, engine, full_context):
        emb1 = engine.create_embedding(full_context)
        emb2 = engine.create_embedding(full_context)
        np.testing.assert_array_equal(emb1, emb2)

    def test_multiple_calls_same_result(self, builder, full_context):
        vectors = [builder.build(full_context) for _ in range(5)]
        for v in vectors[1:]:
            np.testing.assert_array_equal(vectors[0], v)

    def test_embed_trade_embedding_stable(self, engine, full_context):
        result1 = engine.embed_trade(full_context)
        result2 = engine.embed_trade(full_context)
        assert result1["context_embedding"] == result2["context_embedding"]


# ---------------------------------------------------------------------------
# EmbeddingEngine additional tests
# ---------------------------------------------------------------------------

class TestEmbeddingEngine:
    def test_create_embedding_shape(self, engine, full_context):
        emb = engine.create_embedding(full_context)
        assert emb.shape == (64,)

    def test_embed_trade_keys_present(self, engine, full_context):
        result = engine.embed_trade(full_context)
        assert "context_embedding" in result
        assert "outcome_r" in result
        assert "win" in result

    def test_embed_trade_no_result(self, engine, full_context):
        result = engine.embed_trade(full_context)
        assert result["outcome_r"] is None
        assert result["win"] is None

    def test_embed_trade_with_result(self, engine, full_context):
        result = engine.embed_trade(full_context, {"r_multiple": 2.5})
        assert result["outcome_r"] == pytest.approx(2.5)
        assert result["win"] is True

    def test_embed_trade_losing_result(self, engine, full_context):
        result = engine.embed_trade(full_context, {"r_multiple": -0.8})
        assert result["outcome_r"] == pytest.approx(-0.8)
        assert result["win"] is False

    def test_embed_trade_explicit_win_flag(self, engine, full_context):
        result = engine.embed_trade(full_context, {"r_multiple": 1.0, "win": True})
        assert result["win"] is True

    def test_embedding_to_list_length(self, engine, full_context):
        emb = engine.create_embedding(full_context)
        lst = engine.embedding_to_list(emb)
        assert len(lst) == 64
        assert all(isinstance(v, float) for v in lst)

    def test_embedding_round_trip(self, engine, full_context):
        emb = engine.create_embedding(full_context)
        lst = engine.embedding_to_list(emb)
        reconstructed = engine.embedding_from_list(lst)
        np.testing.assert_array_almost_equal(emb, reconstructed)

    def test_embedding_from_list_wrong_length(self, engine):
        with pytest.raises(ValueError, match="Expected 64"):
            engine.embedding_from_list([0.5] * 32)


# ---------------------------------------------------------------------------
# Session encoding
# ---------------------------------------------------------------------------

class TestSessionEncoding:
    @pytest.mark.parametrize("session,expected_dim,expected_val", [
        ("london",     35, 1.0),
        ("new_york",   36, 1.0),
        ("ny",         36, 1.0),
        ("asian",      37, 1.0),
        ("london_ny",  38, 1.0),
        ("overlap",    38, 1.0),
        ("unknown",    35, 0.0),
    ])
    def test_session_flag(self, builder, session, expected_dim, expected_val):
        vec = builder.build({"session": session})
        assert vec[expected_dim] == pytest.approx(expected_val), (
            f"session={session!r}: dim {expected_dim} expected {expected_val}, got {vec[expected_dim]}"
        )
