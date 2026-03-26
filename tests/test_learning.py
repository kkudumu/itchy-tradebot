"""
Unit tests for the adaptive learning system.

Tests cover:
  1. Learning phase detection: 50 → mechanical; 200 → statistical; 600 → similarity
  2. Pre-trade insight: mock similar trades → verify expected performance
  3. Confidence adjustment: verify bounded within +/-0.5
  4. Statistical filters: mock DB stats → verify session/regime filters
  5. Edge reviewer: mock edge stats → verify suggestions
  6. Report generation: mock all data → verify report structure
  7. Cannot exceed risk bounds: learning never overrides risk

All tests are database-free — mock data is injected via public helpers.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List

import pandas as pd
import pytest

from src.learning.adaptive_engine import (
    AdaptiveLearningEngine,
    PreTradeInsight,
    _MAX_CONFIDENCE_DELTA,
    _MECHANICAL_MAX,
    _STATISTICAL_MAX,
)
from src.learning.edge_reviewer import EdgeReviewer, EdgeReviewResult, EdgeSuggestion
from src.learning.report_generator import ReportGenerator, WeeklyReport
from src.learning.similarity import SimilaritySearch, SimilarTrade, PerformanceStats
from src.learning.stats_analyzer import StatsAnalyzer


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_context() -> Dict[str, Any]:
    """Minimal trade context with sufficient fields for all modules."""
    return {
        "session": "london",
        "adx_value": 30.0,
        "confluence_score": 6,
        "signal_tier": "B",
        "direction": "long",
        "atr_value": 10.0,
        "cloud_direction_4h": 1,
        "cloud_direction_1h": 1,
        "tk_cross_15m": 1,
    }


def _make_similar_trade(trade_id: int, r_multiple: float, win: bool) -> SimilarTrade:
    """Helper to create a SimilarTrade for testing."""
    return SimilarTrade(
        trade_id=trade_id,
        similarity=0.85,
        r_multiple=r_multiple,
        win=win,
        context={},
    )


def _make_trade_df(records: List[dict]) -> pd.DataFrame:
    """Build a DataFrame of trade records for injecting into StatsAnalyzer."""
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 1. Learning phase detection
# ---------------------------------------------------------------------------

class TestLearningPhase:
    def test_mechanical_phase_at_zero_trades(self):
        engine = AdaptiveLearningEngine()
        assert engine.get_phase(0) == "mechanical"

    def test_mechanical_phase_at_50_trades(self):
        engine = AdaptiveLearningEngine()
        assert engine.get_phase(50) == "mechanical"

    def test_mechanical_phase_at_boundary(self):
        """99 trades is still mechanical (boundary inclusive)."""
        engine = AdaptiveLearningEngine()
        assert engine.get_phase(_MECHANICAL_MAX) == "mechanical"

    def test_statistical_phase_at_100_trades(self):
        """100 trades crosses into the statistical phase."""
        engine = AdaptiveLearningEngine()
        assert engine.get_phase(_MECHANICAL_MAX + 1) == "statistical"

    def test_statistical_phase_at_200_trades(self):
        engine = AdaptiveLearningEngine()
        assert engine.get_phase(200) == "statistical"

    def test_statistical_phase_at_boundary(self):
        """499 trades is the last statistical trade."""
        engine = AdaptiveLearningEngine()
        assert engine.get_phase(_STATISTICAL_MAX) == "statistical"

    def test_similarity_phase_at_500_trades(self):
        engine = AdaptiveLearningEngine()
        assert engine.get_phase(_STATISTICAL_MAX + 1) == "similarity"

    def test_similarity_phase_at_600_trades(self):
        engine = AdaptiveLearningEngine()
        assert engine.get_phase(600) == "similarity"

    def test_similarity_phase_at_1000_trades(self):
        engine = AdaptiveLearningEngine()
        assert engine.get_phase(1000) == "similarity"

    def test_internal_counter_used_without_override(self):
        engine = AdaptiveLearningEngine()
        engine.set_total_trades(600)
        assert engine.get_phase() == "similarity"

    def test_post_trade_increments_counter(self):
        engine = AdaptiveLearningEngine()
        initial = engine._total_trades
        engine.post_trade_analysis({"r_multiple": 1.5})
        assert engine._total_trades == initial + 1


# ---------------------------------------------------------------------------
# 2. Pre-trade insight with mock similar trades
# ---------------------------------------------------------------------------

class TestPreTradeInsight:
    def test_mechanical_phase_returns_proceed(self, base_context):
        engine = AdaptiveLearningEngine()
        engine.set_total_trades(50)
        insight = engine.pre_trade_analysis(base_context)
        assert insight.recommendation == "proceed"

    def test_mechanical_phase_no_similar_trades(self, base_context):
        engine = AdaptiveLearningEngine()
        engine.set_total_trades(50)
        insight = engine.pre_trade_analysis(base_context)
        assert len(insight.similar_trades) == 0

    def test_mechanical_phase_zero_confidence(self, base_context):
        engine = AdaptiveLearningEngine()
        engine.set_total_trades(50)
        insight = engine.pre_trade_analysis(base_context)
        assert insight.confidence == 0.0

    def test_mechanical_phase_zero_confidence_adjustment(self, base_context):
        engine = AdaptiveLearningEngine()
        engine.set_total_trades(50)
        insight = engine.pre_trade_analysis(base_context)
        assert insight.confidence_adjustment == 0.0

    def test_insight_fields_present(self, base_context):
        """PreTradeInsight must have all required fields."""
        engine = AdaptiveLearningEngine()
        insight = engine.pre_trade_analysis(base_context)
        assert hasattr(insight, "similar_trades")
        assert hasattr(insight, "expected_win_rate")
        assert hasattr(insight, "expected_avg_r")
        assert hasattr(insight, "expected_expectancy")
        assert hasattr(insight, "confidence")
        assert hasattr(insight, "confidence_adjustment")
        assert hasattr(insight, "statistical_filters")
        assert hasattr(insight, "recommendation")
        assert hasattr(insight, "reasoning")

    def test_insight_recommendation_values(self, base_context):
        """recommendation must be one of the three valid states."""
        engine = AdaptiveLearningEngine()
        insight = engine.pre_trade_analysis(base_context)
        assert insight.recommendation in ("proceed", "caution", "skip")

    def test_statistical_phase_checks_session_filter(self):
        """In statistical phase, a bad-performing session triggers 'skip'."""
        # Build a stats analyzer loaded with a session that has low win rate
        records = [
            {"r_multiple": -1.0, "session": "asian", "adx_value": 25.0,
             "confluence_score": 4, "day_of_week": 2},
        ] * 30  # 30 losing trades → 0% win rate in asian session

        df = _make_trade_df(records)
        analyzer = StatsAnalyzer()
        analyzer._inject_trade_cache(df)

        engine = AdaptiveLearningEngine(stats_analyzer=analyzer)
        engine.set_total_trades(200)  # statistical phase

        context = {"session": "asian", "adx_value": 25.0}
        insight = engine.pre_trade_analysis(context)

        # The session filter should trigger a skip recommendation
        assert insight.statistical_filters.get("session_filter") is True
        assert insight.recommendation == "skip"

    def test_similarity_phase_uses_performance_stats(self):
        """In similarity phase, expected performance reflects similar trades."""
        # Build 20 high-confidence winning similar trades
        winning_trades = [
            _make_similar_trade(i, r_multiple=2.0, win=True) for i in range(20)
        ]

        class MockSimilaritySearch:
            def find_similar_trades(self, context_embedding, k=10, min_similarity=0.70,
                                     source_filter=None):
                return winning_trades

            def get_performance_stats(self, similar_trades):
                searcher = SimilaritySearch()
                return searcher.get_performance_stats(similar_trades)

        engine = AdaptiveLearningEngine(
            similarity_search=MockSimilaritySearch(),
        )
        engine.set_total_trades(600)  # similarity phase

        context = {"session": "london", "adx_value": 30.0}
        insight = engine.pre_trade_analysis(context)

        assert insight.expected_win_rate == pytest.approx(1.0)
        assert insight.expected_avg_r == pytest.approx(2.0)
        assert len(insight.similar_trades) == 20


# ---------------------------------------------------------------------------
# 3. Confidence adjustment bounded within +/-0.5
# ---------------------------------------------------------------------------

class TestConfidenceAdjustment:
    def test_mechanical_phase_returns_zero(self, base_context):
        engine = AdaptiveLearningEngine()
        engine.set_total_trades(50)
        adj = engine.get_confidence_adjustment(base_context, base_confluence=6)
        assert adj == 0.0

    def test_no_similarity_search_returns_zero(self, base_context):
        engine = AdaptiveLearningEngine(similarity_search=None)
        engine.set_total_trades(600)
        adj = engine.get_confidence_adjustment(base_context, base_confluence=6)
        assert adj == 0.0

    def test_adjustment_bounded_positive(self):
        """Adjustment must never exceed +0.5."""
        # 20 trades all winning at very high expectancy
        winning_trades = [
            _make_similar_trade(i, r_multiple=10.0, win=True) for i in range(20)
        ]

        class MockSimilaritySearch:
            def find_similar_trades(self, *a, **kw):
                return winning_trades

            def get_performance_stats(self, similar_trades):
                return SimilaritySearch().get_performance_stats(similar_trades)

        engine = AdaptiveLearningEngine(similarity_search=MockSimilaritySearch())
        engine.set_total_trades(600)

        adj = engine.get_confidence_adjustment({}, base_confluence=6)
        assert adj <= _MAX_CONFIDENCE_DELTA

    def test_adjustment_bounded_negative(self):
        """Adjustment must never go below -0.5."""
        # 20 trades all losing at extreme loss
        losing_trades = [
            _make_similar_trade(i, r_multiple=-10.0, win=False) for i in range(20)
        ]

        class MockSimilaritySearch:
            def find_similar_trades(self, *a, **kw):
                return losing_trades

            def get_performance_stats(self, similar_trades):
                return SimilaritySearch().get_performance_stats(similar_trades)

        engine = AdaptiveLearningEngine(similarity_search=MockSimilaritySearch())
        engine.set_total_trades(600)

        adj = engine.get_confidence_adjustment({}, base_confluence=6)
        assert adj >= -_MAX_CONFIDENCE_DELTA

    def test_adjustment_is_float(self, base_context):
        engine = AdaptiveLearningEngine()
        engine.set_total_trades(600)
        adj = engine.get_confidence_adjustment(base_context, base_confluence=5)
        assert isinstance(adj, float)

    def test_low_confidence_no_adjustment(self):
        """Fewer than 5 similar trades → confidence below threshold → no adjustment."""
        few_trades = [
            _make_similar_trade(i, r_multiple=3.0, win=True) for i in range(3)
        ]

        class MockSimilaritySearch:
            def find_similar_trades(self, *a, **kw):
                return few_trades

            def get_performance_stats(self, similar_trades):
                return SimilaritySearch().get_performance_stats(similar_trades)

        engine = AdaptiveLearningEngine(similarity_search=MockSimilaritySearch())
        engine.set_total_trades(600)

        adj = engine.get_confidence_adjustment({}, base_confluence=5)
        # Confidence = 3/20 = 0.15, below MIN_SIMILARITY_CONFIDENCE (0.25)
        assert adj == 0.0


# ---------------------------------------------------------------------------
# 4. Statistical filters
# ---------------------------------------------------------------------------

class TestStatisticalFilters:
    def _make_session_df(self, session: str, n_wins: int, n_losses: int) -> pd.DataFrame:
        """Build a trade DataFrame for a single session."""
        records = []
        for _ in range(n_wins):
            records.append({
                "r_multiple": 1.5, "session": session,
                "adx_value": 28.0, "confluence_score": 5, "day_of_week": 1,
            })
        for _ in range(n_losses):
            records.append({
                "r_multiple": -1.0, "session": session,
                "adx_value": 28.0, "confluence_score": 5, "day_of_week": 1,
            })
        return pd.DataFrame(records)

    def test_session_with_sufficient_trades_and_low_wr_is_filtered(self):
        df = self._make_session_df("asian", n_wins=6, n_losses=24)  # 20% WR
        analyzer = StatsAnalyzer()
        analyzer._inject_trade_cache(df)
        result = analyzer.should_filter_session("asian", min_wr=0.40, min_trades=20)
        assert result is True

    def test_session_with_good_win_rate_is_not_filtered(self):
        df = self._make_session_df("london", n_wins=15, n_losses=5)  # 75% WR
        analyzer = StatsAnalyzer()
        analyzer._inject_trade_cache(df)
        result = analyzer.should_filter_session("london", min_wr=0.40, min_trades=20)
        assert result is False

    def test_session_with_insufficient_trades_not_filtered(self):
        """Less than min_trades → do not filter (insufficient data)."""
        df = self._make_session_df("asian", n_wins=2, n_losses=8)  # 20% WR but only 10 trades
        analyzer = StatsAnalyzer()
        analyzer._inject_trade_cache(df)
        result = analyzer.should_filter_session("asian", min_wr=0.40, min_trades=20)
        assert result is False

    def test_unknown_session_not_filtered(self):
        """A session we've never seen → do not filter."""
        analyzer = StatsAnalyzer()
        analyzer._inject_trade_cache(pd.DataFrame())
        result = analyzer.should_filter_session("london", min_wr=0.40, min_trades=20)
        assert result is False

    def test_regime_filter_low_adx(self):
        """Low ADX regime (< 20) with bad win rate should be filtered."""
        records = [
            {"r_multiple": -1.0, "session": "london", "adx_value": 12.0,
             "confluence_score": 4, "day_of_week": 2}
        ] * 25  # 25 losses in low regime

        df = _make_trade_df(records)
        analyzer = StatsAnalyzer()
        analyzer._inject_trade_cache(df)

        result = analyzer.should_filter_regime(adx=15.0, min_wr=0.40, min_trades=20)
        assert result is True

    def test_regime_filter_high_adx_good_performance(self):
        """High ADX with good performance should not be filtered."""
        records = [
            {"r_multiple": 2.0, "session": "london", "adx_value": 40.0,
             "confluence_score": 7, "day_of_week": 1}
        ] * 25

        df = _make_trade_df(records)
        analyzer = StatsAnalyzer()
        analyzer._inject_trade_cache(df)

        result = analyzer.should_filter_regime(adx=38.0, min_wr=0.40, min_trades=20)
        assert result is False

    def test_win_rate_by_session_returns_dict(self):
        records = [
            {"r_multiple": 1.0, "session": "london", "adx_value": 30.0,
             "confluence_score": 6, "day_of_week": 1}
        ] * 25

        df = _make_trade_df(records)
        analyzer = StatsAnalyzer()
        analyzer._inject_trade_cache(df)

        result = analyzer.win_rate_by_session(min_trades=20)
        assert isinstance(result, dict)
        assert "london" in result
        assert result["london"]["win_rate"] == pytest.approx(1.0)
        assert result["london"]["n_trades"] == 25

    def test_win_rate_by_regime_excludes_small_buckets(self):
        """Buckets below min_trades threshold are excluded from results."""
        records = [
            {"r_multiple": 1.0, "session": "london", "adx_value": 40.0,
             "confluence_score": 7, "day_of_week": 1}
        ] * 5  # Only 5 trades in high regime — below min_trades=20

        df = _make_trade_df(records)
        analyzer = StatsAnalyzer()
        analyzer._inject_trade_cache(df)

        result = analyzer.win_rate_by_regime(min_trades=20)
        assert "high" not in result

    def test_performance_heatmap_returns_dataframe(self):
        records = [
            {"r_multiple": 1.0, "session": "london", "adx_value": 30.0,
             "confluence_score": 6, "day_of_week": 1}
        ] * 5

        df = _make_trade_df(records)
        analyzer = StatsAnalyzer()
        analyzer._inject_trade_cache(df)

        heatmap = analyzer.performance_heatmap()
        assert isinstance(heatmap, pd.DataFrame)

    def test_should_filter_in_statistical_phase(self):
        """should_filter returns (True, reason) for bad session in statistical phase."""
        records = [
            {"r_multiple": -1.0, "session": "asian", "adx_value": 22.0,
             "confluence_score": 4, "day_of_week": 2}
        ] * 25

        df = _make_trade_df(records)
        analyzer = StatsAnalyzer()
        analyzer._inject_trade_cache(df)

        engine = AdaptiveLearningEngine(stats_analyzer=analyzer)
        engine.set_total_trades(200)  # statistical phase

        should_skip, reason = engine.should_filter({"session": "asian", "adx_value": 22.0})
        assert should_skip is True
        assert len(reason) > 0

    def test_should_not_filter_in_mechanical_phase(self, base_context):
        """Mechanical phase: should_filter always returns False."""
        engine = AdaptiveLearningEngine()
        engine.set_total_trades(50)
        should_skip, _ = engine.should_filter(base_context)
        assert should_skip is False


# ---------------------------------------------------------------------------
# 5. Edge reviewer
# ---------------------------------------------------------------------------

class TestEdgeReviewer:
    def _make_edge_df(
        self,
        edge_name: str,
        n_wins: int,
        n_losses: int,
        n_filtered: int = 0,
    ) -> pd.DataFrame:
        """Build a trade DataFrame annotating edge_signals and filter_log."""
        records = []
        for _ in range(n_wins):
            records.append({
                "r_multiple": 2.0,
                "edge_signals": [edge_name],
                "filter_log": [],
            })
        for _ in range(n_losses):
            records.append({
                "r_multiple": -1.0,
                "edge_signals": [edge_name],
                "filter_log": [],
            })
        for _ in range(n_filtered):
            records.append({
                "r_multiple": 1.0,        # these trades passed through
                "edge_signals": [],
                "filter_log": [edge_name],
            })
        return pd.DataFrame(records)

    def test_review_all_edges_returns_list(self):
        reviewer = EdgeReviewer()
        reviewer._inject_trade_cache(pd.DataFrame())
        results = reviewer.review_all_edges()
        assert isinstance(results, list)
        assert len(results) > 0

    def test_edge_review_result_has_required_fields(self):
        df = self._make_edge_df("regime_filter", n_wins=10, n_losses=5)
        reviewer = EdgeReviewer()
        reviewer._inject_trade_cache(df)

        results = reviewer.review_all_edges()
        regime_result = next((r for r in results if r.edge_name == "regime_filter"), None)
        assert regime_result is not None
        assert hasattr(regime_result, "total_trades_affected")
        assert hasattr(regime_result, "win_rate_when_active")
        assert hasattr(regime_result, "avg_r_when_active")
        assert hasattr(regime_result, "filter_rate")
        assert hasattr(regime_result, "marginal_impact")

    def test_win_rate_computed_correctly(self):
        # 15 wins, 5 losses → 75% WR
        df = self._make_edge_df("regime_filter", n_wins=15, n_losses=5)
        reviewer = EdgeReviewer()
        reviewer._inject_trade_cache(df)

        results = reviewer.review_all_edges()
        result = next(r for r in results if r.edge_name == "regime_filter")
        assert result.win_rate_when_active == pytest.approx(0.75)
        assert result.total_trades_affected == 20

    def test_no_trades_returns_zero_filled_result(self):
        reviewer = EdgeReviewer()
        reviewer._inject_trade_cache(pd.DataFrame())

        results = reviewer.review_all_edges()
        assert all(r.total_trades_affected == 0 for r in results)
        assert all(r.win_rate_when_active == 0.0 for r in results)

    def test_suggest_disable_underperforming_edge(self):
        """An enabled edge with low win rate should generate a disable suggestion."""
        # 6 wins, 24 losses → 20% WR — well below threshold
        df = self._make_edge_df("bb_squeeze", n_wins=6, n_losses=24)
        reviewer = EdgeReviewer()
        reviewer._inject_trade_cache(df)
        reviewer._inject_edge_states({"bb_squeeze": True})  # currently enabled

        suggestions = reviewer.suggest_edge_changes()
        bb_suggestion = next(
            (s for s in suggestions if s.edge_name == "bb_squeeze"), None
        )
        assert bb_suggestion is not None
        assert bb_suggestion.current_state is True
        assert bb_suggestion.suggested_state is False

    def test_no_suggestion_when_insufficient_trades(self):
        """Below MIN_TRADES_FOR_SUGGESTION → no suggestions emitted."""
        df = self._make_edge_df("bb_squeeze", n_wins=5, n_losses=20)  # only 25 trades
        reviewer = EdgeReviewer()
        reviewer._inject_trade_cache(df)
        reviewer._inject_edge_states({"bb_squeeze": True})

        # Set min threshold to something the data doesn't meet
        from src.learning import edge_reviewer as er_module
        original = er_module._MIN_TRADES_FOR_SUGGESTION
        er_module._MIN_TRADES_FOR_SUGGESTION = 50
        try:
            suggestions = reviewer.suggest_edge_changes()
            bb_suggestion = next(
                (s for s in suggestions if s.edge_name == "bb_squeeze"), None
            )
            assert bb_suggestion is None
        finally:
            er_module._MIN_TRADES_FOR_SUGGESTION = original

    def test_suggestion_confidence_bounded(self):
        """Suggestion confidence must be in [0, 1]."""
        df = self._make_edge_df("time_of_day", n_wins=4, n_losses=36)  # bad edge
        reviewer = EdgeReviewer()
        reviewer._inject_trade_cache(df)
        reviewer._inject_edge_states({"time_of_day": True})

        suggestions = reviewer.suggest_edge_changes()
        for s in suggestions:
            assert 0.0 <= s.confidence <= 1.0

    def test_filter_rate_computed(self):
        """Filter rate = filtered / (total_trades_in_df + filtered)."""
        df = self._make_edge_df("spread_filter", n_wins=16, n_losses=4, n_filtered=10)
        reviewer = EdgeReviewer()
        reviewer._inject_trade_cache(df)

        results = reviewer.review_all_edges()
        result = next(r for r in results if r.edge_name == "spread_filter")
        # n_total (all rows in df) = 30, n_filtered = 10
        # total_candidates = 30 + 10 = 40; filter_rate = 10 / 40 = 0.25
        assert result.filter_rate == pytest.approx(10 / 40, rel=0.01)

    def test_edge_suggestion_has_reason(self):
        df = self._make_edge_df("equity_curve", n_wins=6, n_losses=24)
        reviewer = EdgeReviewer()
        reviewer._inject_trade_cache(df)
        reviewer._inject_edge_states({"equity_curve": True})

        suggestions = reviewer.suggest_edge_changes()
        for s in suggestions:
            assert isinstance(s.reason, str)
            assert len(s.reason) > 0


# ---------------------------------------------------------------------------
# 6. Report generation
# ---------------------------------------------------------------------------

class TestReportGenerator:
    def _make_period_trades(self, n_wins: int, n_losses: int) -> List[dict]:
        trades = []
        for i in range(n_wins):
            trades.append({
                "id": i,
                "r_multiple": 2.0,
                "pnl": 200.0,
                "signal_tier": "B",
                "session": "london",
                "entry_time": datetime(2025, 3, 1, 10, tzinfo=timezone.utc),
                "exit_time": datetime(2025, 3, 1, 11, tzinfo=timezone.utc),
            })
        for i in range(n_wins, n_wins + n_losses):
            trades.append({
                "id": i,
                "r_multiple": -1.0,
                "pnl": -100.0,
                "signal_tier": "B",
                "session": "london",
                "entry_time": datetime(2025, 3, 1, 14, tzinfo=timezone.utc),
                "exit_time": datetime(2025, 3, 1, 15, tzinfo=timezone.utc),
            })
        return trades

    def test_weekly_report_returns_object(self):
        gen = ReportGenerator()
        gen._inject_period_trades([])
        report = gen.weekly_report()
        assert isinstance(report, WeeklyReport)

    def test_report_has_required_fields(self):
        gen = ReportGenerator()
        gen._inject_period_trades([])
        report = gen.weekly_report()

        assert hasattr(report, "period_start")
        assert hasattr(report, "period_end")
        assert hasattr(report, "total_trades")
        assert hasattr(report, "win_rate")
        assert hasattr(report, "pnl")
        assert hasattr(report, "session_breakdown")
        assert hasattr(report, "edge_review")
        assert hasattr(report, "similarity_quality")
        assert hasattr(report, "suggestions")
        assert hasattr(report, "learning_phase")

    def test_report_win_rate_calculation(self):
        trades = self._make_period_trades(n_wins=6, n_losses=4)
        gen = ReportGenerator()
        gen._inject_period_trades(trades)
        report = gen.weekly_report()

        assert report.total_trades == 10
        assert report.win_rate == pytest.approx(0.6)

    def test_report_pnl_sum(self):
        trades = self._make_period_trades(n_wins=3, n_losses=2)
        gen = ReportGenerator()
        gen._inject_period_trades(trades)
        report = gen.weekly_report()

        # 3 * 2.0R + 2 * (-1.0R) = 6 - 2 = 4.0R
        assert report.pnl == pytest.approx(4.0)

    def test_report_zero_trades(self):
        gen = ReportGenerator()
        gen._inject_period_trades([])
        report = gen.weekly_report()

        assert report.total_trades == 0
        assert report.win_rate == 0.0
        assert report.pnl == 0.0

    def test_report_period_dates(self):
        start = datetime(2025, 3, 1, tzinfo=timezone.utc)
        end = datetime(2025, 3, 8, tzinfo=timezone.utc)
        gen = ReportGenerator()
        gen._inject_period_trades([])
        report = gen.weekly_report(period_start=start, period_end=end)

        assert report.period_start == start
        assert report.period_end == end

    def test_report_includes_edge_review(self):
        gen = ReportGenerator()
        gen._inject_period_trades([])
        report = gen.weekly_report()

        assert isinstance(report.edge_review, list)

    def test_report_includes_similarity_quality(self):
        gen = ReportGenerator()
        gen._inject_period_trades([])
        gen._inject_similarity_quality({
            "total_queries": 50,
            "avg_confidence": 0.72,
            "pct_above_threshold": 0.64,
        })
        report = gen.weekly_report()

        assert report.similarity_quality["total_queries"] == 50
        assert report.similarity_quality["avg_confidence"] == pytest.approx(0.72)

    def test_report_learning_phase(self):
        gen = ReportGenerator()
        gen._inject_period_trades([])
        gen._set_learning_context("statistical", 250)
        report = gen.weekly_report()

        assert report.learning_phase == "statistical"
        assert report.total_trades_lifetime == 250

    def test_to_text_returns_string(self):
        gen = ReportGenerator()
        gen._inject_period_trades(self._make_period_trades(3, 2))
        report = gen.weekly_report()
        text = gen.to_text(report)

        assert isinstance(text, str)
        assert len(text) > 0

    def test_to_text_contains_win_rate(self):
        trades = self._make_period_trades(n_wins=3, n_losses=2)
        gen = ReportGenerator()
        gen._inject_period_trades(trades)
        report = gen.weekly_report()
        text = gen.to_text(report)

        assert "60.0%" in text or "Win rate" in text

    def test_to_text_contains_period(self):
        start = datetime(2025, 3, 1, tzinfo=timezone.utc)
        end = datetime(2025, 3, 8, tzinfo=timezone.utc)
        gen = ReportGenerator()
        gen._inject_period_trades([])
        report = gen.weekly_report(period_start=start, period_end=end)
        text = gen.to_text(report)

        assert "2025-03-01" in text

    def test_to_html_returns_string(self):
        gen = ReportGenerator()
        gen._inject_period_trades(self._make_period_trades(2, 1))
        report = gen.weekly_report()
        html = gen.to_html(report)

        assert isinstance(html, str)
        assert "<html" in html.lower()

    def test_to_html_contains_table(self):
        gen = ReportGenerator()
        gen._inject_period_trades(self._make_period_trades(2, 1))
        report = gen.weekly_report()
        html = gen.to_html(report)

        assert "<table" in html

    def test_setup_breakdown_populated_from_trades(self):
        trades = [
            {"r_multiple": 2.0, "pnl": 200.0, "signal_tier": "A+",
             "session": "london"},
            {"r_multiple": -1.0, "pnl": -100.0, "signal_tier": "B",
             "session": "london"},
        ]
        gen = ReportGenerator()
        gen._inject_period_trades(trades)
        report = gen.weekly_report()

        assert "A+" in report.setup_breakdown or "B" in report.setup_breakdown


# ---------------------------------------------------------------------------
# 7. Learning cannot exceed risk bounds
# ---------------------------------------------------------------------------

class TestRiskBoundary:
    def test_confidence_adjustment_never_exceeds_max_delta(self):
        """No matter how extreme the similar trade performance, delta stays bounded."""
        extreme_winning = [
            _make_similar_trade(i, r_multiple=100.0, win=True) for i in range(100)
        ]

        class MockSimilaritySearch:
            def find_similar_trades(self, *a, **kw):
                return extreme_winning

            def get_performance_stats(self, similar_trades):
                return SimilaritySearch().get_performance_stats(similar_trades)

        engine = AdaptiveLearningEngine(similarity_search=MockSimilaritySearch())
        engine.set_total_trades(600)

        adj = engine.get_confidence_adjustment({}, base_confluence=8)
        assert adj <= _MAX_CONFIDENCE_DELTA

    def test_confidence_adjustment_never_below_negative_max_delta(self):
        extreme_losing = [
            _make_similar_trade(i, r_multiple=-100.0, win=False) for i in range(100)
        ]

        class MockSimilaritySearch:
            def find_similar_trades(self, *a, **kw):
                return extreme_losing

            def get_performance_stats(self, similar_trades):
                return SimilaritySearch().get_performance_stats(similar_trades)

        engine = AdaptiveLearningEngine(similarity_search=MockSimilaritySearch())
        engine.set_total_trades(600)

        adj = engine.get_confidence_adjustment({}, base_confluence=3)
        assert adj >= -_MAX_CONFIDENCE_DELTA

    def test_mechanical_phase_produces_no_adjustments(self, base_context):
        """During mechanical phase the engine should not alter execution in any way."""
        engine = AdaptiveLearningEngine()
        engine.set_total_trades(0)

        insight = engine.pre_trade_analysis(base_context)
        assert insight.confidence_adjustment == 0.0
        assert insight.recommendation == "proceed"

        adj = engine.get_confidence_adjustment(base_context, base_confluence=6)
        assert adj == 0.0

    def test_should_filter_returns_false_in_mechanical(self, base_context):
        """Statistical filters must be inactive in mechanical phase."""
        engine = AdaptiveLearningEngine()
        engine.set_total_trades(99)

        should_skip, _ = engine.should_filter(base_context)
        assert should_skip is False

    def test_suggestion_required_human_approval(self):
        """EdgeSuggestion objects exist but do not self-apply — they need approval."""
        df = pd.DataFrame([
            {"r_multiple": -1.0, "edge_signals": ["time_stop"],
             "filter_log": []} for _ in range(35)
        ])
        reviewer = EdgeReviewer()
        reviewer._inject_trade_cache(df)
        reviewer._inject_edge_states({"time_stop": True})

        suggestions = reviewer.suggest_edge_changes()
        # The suggestion exists (advisory only) — it carries no execution side-effect
        for s in suggestions:
            assert isinstance(s, EdgeSuggestion)
            assert hasattr(s, "suggested_state")  # human must read and act

    def test_skip_recommendation_does_not_modify_context(self, base_context):
        """pre_trade_analysis must not mutate the input context dict."""
        original = dict(base_context)
        engine = AdaptiveLearningEngine()
        engine.pre_trade_analysis(base_context)
        assert base_context == original
