"""Tests for pattern-based confluence scoring adjustments."""

from datetime import datetime, timezone

import pytest


def _make_edge_context(**overrides):
    """Build a minimal EdgeContext for testing."""
    from src.edges.base import EdgeContext

    defaults = {
        "timestamp": datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
        "day_of_week": 3,
        "close_price": 2050.0,
        "high_price": 2052.0,
        "low_price": 2048.0,
        "spread": 0.3,
        "session": "london",
        "adx": 30.0,
        "atr": 5.0,
    }
    defaults.update(overrides)
    return EdgeContext(**defaults)


class TestPatternConfluenceScoring:
    def test_bullish_pattern_adds_confluence_for_long(self):
        from src.discovery.pattern_confluence import PatternConfluenceScorer
        from src.discovery.chart_patterns import ChartPattern

        scorer = PatternConfluenceScorer()
        patterns = [
            ChartPattern(
                pattern_type="double_bottom",
                direction="bullish",
                confidence=0.8,
                key_prices=[2020.0, 2040.0, 2021.0],
                start_index=10,
                end_index=40,
                description="Double bottom",
            )
        ]
        adjustment = scorer.score(patterns, trade_direction="long")
        assert adjustment > 0

    def test_bearish_pattern_subtracts_confluence_for_long(self):
        from src.discovery.pattern_confluence import PatternConfluenceScorer
        from src.discovery.chart_patterns import ChartPattern

        scorer = PatternConfluenceScorer()
        patterns = [
            ChartPattern(
                pattern_type="head_and_shoulders",
                direction="bearish",
                confidence=0.75,
                key_prices=[2040.0, 2060.0, 2040.0, 2030.0],
                start_index=10,
                end_index=50,
                description="H&S",
            )
        ]
        adjustment = scorer.score(patterns, trade_direction="long")
        assert adjustment < 0

    def test_aligned_pattern_boosts_score(self):
        from src.discovery.pattern_confluence import PatternConfluenceScorer
        from src.discovery.chart_patterns import ChartPattern

        scorer = PatternConfluenceScorer()
        patterns = [
            ChartPattern(
                pattern_type="falling_wedge",
                direction="bullish",
                confidence=0.9,
                key_prices=[2050.0, 2040.0, 2045.0, 2035.0],
                start_index=10,
                end_index=50,
                description="Falling wedge",
            )
        ]
        # Bullish pattern + long trade = aligned
        aligned = scorer.score(patterns, trade_direction="long")
        # Bullish pattern + short trade = conflicting
        conflicting = scorer.score(patterns, trade_direction="short")
        assert aligned > conflicting

    def test_low_confidence_patterns_contribute_less(self):
        from src.discovery.pattern_confluence import PatternConfluenceScorer
        from src.discovery.chart_patterns import ChartPattern

        scorer = PatternConfluenceScorer()

        high_conf = [ChartPattern(
            pattern_type="double_bottom", direction="bullish", confidence=0.95,
            key_prices=[], start_index=0, end_index=50, description="",
        )]
        low_conf = [ChartPattern(
            pattern_type="double_bottom", direction="bullish", confidence=0.3,
            key_prices=[], start_index=0, end_index=50, description="",
        )]

        high_adj = scorer.score(high_conf, trade_direction="long")
        low_adj = scorer.score(low_conf, trade_direction="long")
        assert high_adj > low_adj

    def test_no_patterns_returns_zero(self):
        from src.discovery.pattern_confluence import PatternConfluenceScorer

        scorer = PatternConfluenceScorer()
        assert scorer.score([], trade_direction="long") == 0

    def test_multiple_patterns_aggregate(self):
        from src.discovery.pattern_confluence import PatternConfluenceScorer
        from src.discovery.chart_patterns import ChartPattern

        scorer = PatternConfluenceScorer()
        patterns = [
            ChartPattern(
                pattern_type="double_bottom", direction="bullish", confidence=0.8,
                key_prices=[], start_index=0, end_index=30, description="",
            ),
            ChartPattern(
                pattern_type="ascending_triangle", direction="bullish", confidence=0.7,
                key_prices=[], start_index=0, end_index=30, description="",
            ),
        ]
        adj = scorer.score(patterns, trade_direction="long")
        # Two aligned bullish patterns should give more than one
        single = scorer.score(patterns[:1], trade_direction="long")
        assert adj > single


class TestPatternConfluenceEdge:
    def test_edge_filter_allows_when_disabled(self):
        from src.discovery.pattern_confluence import PatternConfluenceEdge

        edge = PatternConfluenceEdge(
            name="pattern_confluence",
            config={"enabled": False},
        )
        ctx = _make_edge_context()
        result = edge.should_allow(ctx)
        assert result.allowed is True

    def test_edge_filter_allows_with_no_patterns(self):
        from src.discovery.pattern_confluence import PatternConfluenceEdge

        edge = PatternConfluenceEdge(
            name="pattern_confluence",
            config={"enabled": True},
        )
        ctx = _make_edge_context()
        result = edge.should_allow(ctx)
        assert result.allowed is True

    def test_edge_filter_blocks_on_conflicting_pattern(self):
        from src.discovery.pattern_confluence import PatternConfluenceEdge
        from src.discovery.chart_patterns import ChartPattern

        edge = PatternConfluenceEdge(
            name="pattern_confluence",
            config={"enabled": True, "min_pattern_score": 0, "block_on_conflict": True},
        )
        ctx = _make_edge_context(
            indicator_values={
                "_active_patterns": [
                    ChartPattern(
                        pattern_type="head_and_shoulders",
                        direction="bearish",
                        confidence=0.9,
                        key_prices=[],
                        start_index=0,
                        end_index=50,
                        description="Strong H&S",
                    ).to_dict()
                ],
                "_trade_direction": "long",
            }
        )
        result = edge.should_allow(ctx)
        assert result.allowed is False
        assert "conflicting" in result.reason.lower() or "bearish" in result.reason.lower()
