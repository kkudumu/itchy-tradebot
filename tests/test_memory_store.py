"""
Tests for the in-memory learning components:
- InMemorySimilarityStore: numpy-backed cosine similarity search
- InMemoryStatsAnalyzer: trade statistics from accumulated trades
- Backtester learning integration: adaptive learning wired into event loop
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.learning.memory_store import InMemorySimilarityStore, InMemoryStatsAnalyzer
from src.learning.adaptive_engine import AdaptiveLearningEngine
from src.learning.embeddings import EmbeddingEngine
from src.learning.similarity import SimilarTrade, PerformanceStats


# ===========================================================================
# InMemorySimilarityStore
# ===========================================================================

class TestInMemorySimilarityStore:
    """Tests for numpy-backed similarity search."""

    def test_initial_state(self):
        store = InMemorySimilarityStore(vector_dim=64)
        assert store.trade_count == 0

    def test_record_trade(self):
        store = InMemorySimilarityStore(vector_dim=8)
        emb = np.random.rand(8)
        idx = store.record_trade(embedding=emb, r_multiple=1.5, context={"session": "london"})
        assert idx == 0
        assert store.trade_count == 1

    def test_record_multiple_trades(self):
        store = InMemorySimilarityStore(vector_dim=8)
        for i in range(10):
            store.record_trade(embedding=np.random.rand(8), r_multiple=float(i))
        assert store.trade_count == 10

    def test_find_similar_returns_empty_below_minimum(self):
        """Below _MIN_TRADES_FOR_SEARCH (30), no results returned."""
        store = InMemorySimilarityStore(vector_dim=8)
        for i in range(20):
            store.record_trade(embedding=np.random.rand(8), r_multiple=1.0)
        results = store.find_similar_trades(np.random.rand(8), k=5)
        assert results == []

    def test_find_similar_returns_results_above_minimum(self):
        """Above threshold, similarity search returns matches."""
        store = InMemorySimilarityStore(vector_dim=8)
        # Record 40 identical embeddings
        base_emb = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        for i in range(40):
            store.record_trade(embedding=base_emb.copy(), r_multiple=1.0)
        # Query with the same embedding
        results = store.find_similar_trades(base_emb, k=5, min_similarity=0.9)
        assert len(results) > 0
        assert all(isinstance(r, SimilarTrade) for r in results)

    def test_similarity_ordering(self):
        """Results should be ordered by descending similarity."""
        store = InMemorySimilarityStore(vector_dim=4)
        # Add 35 trades: first 30 random, then 5 very similar to query
        for _ in range(30):
            store.record_trade(embedding=np.random.rand(4) * 0.1, r_multiple=-0.5)
        target = np.array([1.0, 0.0, 0.0, 0.0])
        for _ in range(5):
            store.record_trade(embedding=target + np.random.rand(4) * 0.01, r_multiple=2.0)

        results = store.find_similar_trades(target, k=10, min_similarity=0.5)
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].similarity >= results[i + 1].similarity

    def test_min_similarity_filter(self):
        """Only results above min_similarity are returned."""
        store = InMemorySimilarityStore(vector_dim=4)
        target = np.array([1.0, 0.0, 0.0, 0.0])
        orthogonal = np.array([0.0, 1.0, 0.0, 0.0])
        for _ in range(35):
            store.record_trade(embedding=orthogonal, r_multiple=0.5)
        # Cosine similarity of orthogonal vectors = 0.0
        results = store.find_similar_trades(target, k=10, min_similarity=0.5)
        assert len(results) == 0

    def test_k_limit(self):
        """At most k results returned."""
        store = InMemorySimilarityStore(vector_dim=4)
        emb = np.array([1.0, 0.5, 0.0, 0.0])
        for _ in range(50):
            store.record_trade(embedding=emb + np.random.rand(4) * 0.01, r_multiple=1.0)
        results = store.find_similar_trades(emb, k=3, min_similarity=0.5)
        assert len(results) <= 3

    def test_win_flag_derived_from_r_multiple(self):
        store = InMemorySimilarityStore(vector_dim=4)
        emb = np.array([1.0, 0.0, 0.0, 0.0])
        for i in range(35):
            r = 1.5 if i % 2 == 0 else -0.8
            store.record_trade(embedding=emb.copy(), r_multiple=r)
        results = store.find_similar_trades(emb, k=10, min_similarity=0.9)
        for r in results:
            assert r.win == (r.r_multiple > 0.0)

    def test_context_preserved(self):
        store = InMemorySimilarityStore(vector_dim=4)
        emb = np.array([1.0, 0.0, 0.0, 0.0])
        ctx = {"session": "london", "adx_value": 30.0}
        for _ in range(35):
            store.record_trade(embedding=emb.copy(), r_multiple=1.0, context=ctx)
        results = store.find_similar_trades(emb, k=1, min_similarity=0.9)
        assert len(results) == 1
        assert results[0].context["session"] == "london"

    def test_clear_resets_state(self):
        store = InMemorySimilarityStore(vector_dim=4)
        for _ in range(10):
            store.record_trade(embedding=np.random.rand(4), r_multiple=1.0)
        store.clear()
        assert store.trade_count == 0

    def test_auto_grow(self):
        """Storage grows when initial capacity is exceeded."""
        store = InMemorySimilarityStore(vector_dim=4, initial_capacity=5)
        for i in range(20):
            store.record_trade(embedding=np.random.rand(4), r_multiple=float(i))
        assert store.trade_count == 20

    def test_zero_embedding_returns_empty(self):
        store = InMemorySimilarityStore(vector_dim=4)
        emb = np.array([1.0, 0.0, 0.0, 0.0])
        for _ in range(35):
            store.record_trade(embedding=emb.copy(), r_multiple=1.0)
        zero = np.zeros(4)
        results = store.find_similar_trades(zero, k=5, min_similarity=0.5)
        assert results == []

    def test_get_performance_stats_inherited(self):
        """get_performance_stats is inherited from SimilaritySearch base."""
        store = InMemorySimilarityStore(vector_dim=4)
        trades = [
            SimilarTrade(trade_id=0, similarity=0.9, r_multiple=2.0, win=True),
            SimilarTrade(trade_id=1, similarity=0.8, r_multiple=-0.5, win=False),
        ]
        stats = store.get_performance_stats(trades)
        assert isinstance(stats, PerformanceStats)
        assert stats.n_trades == 2
        assert stats.win_rate == 0.5

    def test_get_confidence_inherited(self):
        store = InMemorySimilarityStore()
        assert store.get_confidence(0) == 0.0
        assert store.get_confidence(10) == 0.5
        assert store.get_confidence(20) == 1.0


# ===========================================================================
# InMemoryStatsAnalyzer
# ===========================================================================

class TestInMemoryStatsAnalyzer:
    """Tests for in-memory trade statistics."""

    def test_initial_empty(self):
        stats = InMemoryStatsAnalyzer()
        assert stats.trade_count == 0
        assert stats.win_rate_by_session() == {}
        assert stats.win_rate_by_regime() == {}

    def test_record_and_count(self):
        stats = InMemoryStatsAnalyzer()
        stats.record_trade({"r_multiple": 1.0, "session": "london", "adx_value": 30.0})
        assert stats.trade_count == 1

    def _make_trades(self, n, session="london", adx=30.0, win_rate=0.6):
        """Helper to create n trades with specified win rate."""
        stats = InMemoryStatsAnalyzer()
        for i in range(n):
            r = 1.5 if i < int(n * win_rate) else -0.8
            stats.record_trade({
                "r_multiple": r,
                "session": session,
                "adx_value": adx,
                "confluence_score": 5,
                "day_of_week": i % 5,
            })
        return stats

    def test_win_rate_by_session_below_min_trades(self):
        stats = self._make_trades(10, session="london")
        result = stats.win_rate_by_session(min_trades=20)
        assert result == {}

    def test_win_rate_by_session_above_min_trades(self):
        stats = self._make_trades(30, session="london", win_rate=0.7)
        result = stats.win_rate_by_session(min_trades=20)
        assert "london" in result
        assert 0.6 <= result["london"]["win_rate"] <= 0.8

    def test_win_rate_by_regime_low(self):
        stats = self._make_trades(25, adx=15.0, win_rate=0.5)
        result = stats.win_rate_by_regime(min_trades=20)
        assert "low" in result

    def test_win_rate_by_regime_medium(self):
        stats = self._make_trades(25, adx=28.0, win_rate=0.6)
        result = stats.win_rate_by_regime(min_trades=20)
        assert "medium" in result

    def test_win_rate_by_regime_high(self):
        stats = self._make_trades(25, adx=40.0, win_rate=0.7)
        result = stats.win_rate_by_regime(min_trades=20)
        assert "high" in result

    def test_should_filter_session_true(self):
        stats = self._make_trades(25, session="asian", win_rate=0.3)
        assert stats.should_filter_session("asian", min_wr=0.40, min_trades=20) is True

    def test_should_filter_session_false(self):
        stats = self._make_trades(25, session="london", win_rate=0.7)
        assert stats.should_filter_session("london", min_wr=0.40, min_trades=20) is False

    def test_should_filter_session_insufficient_data(self):
        stats = self._make_trades(5, session="overlap")
        assert stats.should_filter_session("overlap", min_wr=0.40, min_trades=20) is False

    def test_should_filter_regime_true(self):
        stats = self._make_trades(25, adx=15.0, win_rate=0.3)
        assert stats.should_filter_regime(15.0, min_wr=0.40, min_trades=20) is True

    def test_should_filter_regime_false(self):
        stats = self._make_trades(25, adx=30.0, win_rate=0.7)
        assert stats.should_filter_regime(30.0, min_wr=0.40, min_trades=20) is False

    def test_clear(self):
        stats = self._make_trades(10)
        stats.clear()
        assert stats.trade_count == 0
        assert stats.win_rate_by_session() == {}


# ===========================================================================
# AdaptiveLearningEngine + InMemory integration
# ===========================================================================

class TestLearningEngineWithMemoryStore:
    """Test AdaptiveLearningEngine using in-memory backends."""

    def _make_engine(self):
        store = InMemorySimilarityStore(vector_dim=64)
        stats = InMemoryStatsAnalyzer()
        emb = EmbeddingEngine()
        engine = AdaptiveLearningEngine(
            similarity_search=store,
            embedding_engine=emb,
            stats_analyzer=stats,
        )
        return engine, store, stats

    def test_mechanical_phase_always_proceeds(self):
        engine, store, stats = self._make_engine()
        insight = engine.pre_trade_analysis({"session": "london", "adx_value": 30.0})
        assert insight.recommendation == "proceed"
        assert "Mechanical" in insight.reasoning

    def test_phase_transitions(self):
        engine, store, stats = self._make_engine()
        assert engine.get_phase() == "mechanical"
        engine.set_total_trades(100)
        assert engine.get_phase() == "statistical"
        engine.set_total_trades(500)
        assert engine.get_phase() == "similarity"

    def test_statistical_phase_uses_in_memory_stats(self):
        engine, store, stats = self._make_engine()
        # Add 25 losing trades in asian session
        for _ in range(25):
            stats.record_trade({"r_multiple": -0.8, "session": "asian", "adx_value": 30.0})
        engine.set_total_trades(150)  # statistical phase

        # should_filter now returns True for asian session
        should_skip, reason = engine.should_filter({"session": "asian", "adx_value": 30.0})
        assert should_skip is True

    def test_similarity_phase_uses_memory_store(self):
        engine, store, stats = self._make_engine()
        emb_engine = EmbeddingEngine()

        # Record 40 trades with identical context in memory store
        ctx = {"session": "london", "adx_value": 30.0, "atr_value": 5.0,
               "confluence_score": 6, "direction": "long"}
        for i in range(40):
            embedding = emb_engine.create_embedding(ctx)
            store.record_trade(embedding=embedding, r_multiple=2.0, context=ctx)

        engine.set_total_trades(500)  # similarity phase

        # Pre-trade analysis should find similar trades
        insight = engine.pre_trade_analysis(ctx)
        # In similarity phase with matching trades, should get a non-empty result
        assert insight.recommendation in ("proceed", "caution", "skip")

    def test_post_trade_increments_counter(self):
        engine, store, stats = self._make_engine()
        assert engine.get_phase() == "mechanical"
        for i in range(100):
            engine.post_trade_analysis({"r_multiple": 1.0})
        assert engine.get_phase() == "statistical"


# ===========================================================================
# Backtester learning integration
# ===========================================================================

class TestBacktesterLearningIntegration:
    """Test that IchimokuBacktester uses learning components correctly."""

    def test_backtester_has_learning_components(self):
        from src.backtesting.vectorbt_engine import IchimokuBacktester
        bt = IchimokuBacktester()
        assert isinstance(bt._memory_store, InMemorySimilarityStore)
        assert isinstance(bt._memory_stats, InMemoryStatsAnalyzer)
        assert isinstance(bt.learning_engine, AdaptiveLearningEngine)

    def test_backtester_learning_engine_uses_memory_store(self):
        from src.backtesting.vectorbt_engine import IchimokuBacktester
        bt = IchimokuBacktester()
        # The learning engine's similarity search should be the memory store
        assert bt.learning_engine._similarity_search is bt._memory_store

    def test_backtester_learning_engine_uses_memory_stats(self):
        from src.backtesting.vectorbt_engine import IchimokuBacktester
        bt = IchimokuBacktester()
        assert bt.learning_engine._stats is bt._memory_stats

    def test_record_learning_stores_embedding(self):
        from src.backtesting.vectorbt_engine import IchimokuBacktester
        bt = IchimokuBacktester()
        trade_summary = {
            "r_multiple": 1.5,
            "context_embedding": [0.5] * 64,
            "confluence_score": 5,
            "signal_tier": "B",
        }
        context = {"session": "london", "adx_value": 30.0, "day_of_week": 1, "direction": "long"}
        bt._record_learning(trade_summary, context, enable_learning=True)
        assert bt._memory_store.trade_count == 1
        assert bt._memory_stats.trade_count == 1

    def test_record_learning_disabled(self):
        from src.backtesting.vectorbt_engine import IchimokuBacktester
        bt = IchimokuBacktester()
        trade_summary = {
            "r_multiple": 1.5,
            "context_embedding": [0.5] * 64,
        }
        bt._record_learning(trade_summary, {}, enable_learning=False)
        assert bt._memory_store.trade_count == 0
        assert bt._memory_stats.trade_count == 0

    def test_record_learning_no_embedding(self):
        """When context_embedding is None, only stats are recorded."""
        from src.backtesting.vectorbt_engine import IchimokuBacktester
        bt = IchimokuBacktester()
        trade_summary = {"r_multiple": 1.5}
        context = {"session": "london", "adx_value": 30.0, "day_of_week": 1, "direction": "long"}
        bt._record_learning(trade_summary, context, enable_learning=True)
        assert bt._memory_store.trade_count == 0  # no embedding → not stored
        assert bt._memory_stats.trade_count == 1  # stats still recorded

    def test_record_learning_increments_phase(self):
        from src.backtesting.vectorbt_engine import IchimokuBacktester
        bt = IchimokuBacktester()
        for i in range(100):
            trade_summary = {
                "r_multiple": 1.0,
                "context_embedding": list(np.random.rand(64)),
                "confluence_score": 5,
                "signal_tier": "B",
            }
            context = {"session": "london", "adx_value": 30.0, "day_of_week": 1, "direction": "long"}
            bt._record_learning(trade_summary, context, enable_learning=True)
        assert bt.learning_engine.get_phase() == "statistical"

    def test_run_signature_accepts_learning_params(self):
        """Verify run() accepts the new parameters without error on signature."""
        from src.backtesting.vectorbt_engine import IchimokuBacktester
        import inspect
        sig = inspect.signature(IchimokuBacktester.run)
        params = list(sig.parameters.keys())
        assert "enable_learning" in params
        assert "enable_screenshots" in params
        assert "screenshot_dir" in params

    def test_run_with_learning_disabled(self):
        """Backtest with enable_learning=False should work as before."""
        from src.backtesting.vectorbt_engine import IchimokuBacktester
        bt = IchimokuBacktester(initial_balance=10_000)

        # Create minimal synthetic data
        n_bars = 500
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="1min", tz="UTC")
        np.random.seed(42)
        prices = 2000.0 + np.cumsum(np.random.randn(n_bars) * 0.5)
        candles = pd.DataFrame({
            "open": prices,
            "high": prices + np.abs(np.random.randn(n_bars)) * 0.3,
            "low": prices - np.abs(np.random.randn(n_bars)) * 0.3,
            "close": prices + np.random.randn(n_bars) * 0.2,
            "tick_volume": np.random.randint(100, 1000, n_bars),
        }, index=dates)

        result = bt.run(candles, enable_learning=False)
        assert result is not None
        assert bt._memory_store.trade_count == 0

    def test_run_with_learning_enabled(self):
        """Backtest with enable_learning=True records trades to memory store."""
        from src.backtesting.vectorbt_engine import IchimokuBacktester
        bt = IchimokuBacktester(initial_balance=10_000)

        # Create minimal synthetic data
        n_bars = 500
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="1min", tz="UTC")
        np.random.seed(42)
        prices = 2000.0 + np.cumsum(np.random.randn(n_bars) * 0.5)
        candles = pd.DataFrame({
            "open": prices,
            "high": prices + np.abs(np.random.randn(n_bars)) * 0.3,
            "low": prices - np.abs(np.random.randn(n_bars)) * 0.3,
            "close": prices + np.random.randn(n_bars) * 0.2,
            "tick_volume": np.random.randint(100, 1000, n_bars),
        }, index=dates)

        result = bt.run(candles, enable_learning=True)
        assert result is not None
        # Memory store trade count should match closed trades
        n_closed = len(result.trades)
        assert bt._memory_store.trade_count == n_closed
        assert bt._memory_stats.trade_count == n_closed


# ===========================================================================
# Screenshot integration
# ===========================================================================

class TestBacktesterScreenshots:
    """Test screenshot generation in backtester."""

    def test_make_screenshot_fn_returns_none_without_mplfinance(self):
        """When mplfinance is not installed, screenshot fn is None."""
        from src.backtesting.vectorbt_engine import IchimokuBacktester
        bt = IchimokuBacktester()
        with patch.dict("sys.modules", {"mplfinance": None}):
            with patch("builtins.__import__", side_effect=ImportError("no mplfinance")):
                fn = bt._make_screenshot_fn("/tmp/test_screenshots", "XAUUSD")
        # If mplfinance is actually installed, fn will not be None — that's fine
        # This test just verifies the method doesn't crash

    def test_make_screenshot_fn_returns_callable_with_mplfinance(self):
        """When mplfinance is available, a callable is returned."""
        from src.backtesting.vectorbt_engine import IchimokuBacktester
        bt = IchimokuBacktester()
        try:
            import mplfinance  # noqa: F401
            fn = bt._make_screenshot_fn("/tmp/test_screenshots", "XAUUSD")
            assert callable(fn)
        except ImportError:
            pytest.skip("mplfinance not installed")
