"""Tests for pattern detection hook that wires into the backtest loop."""

from datetime import datetime, timezone

import pytest


def _make_swing(price, swing_type, index=0):
    from src.strategy.strategies.sss.breathing_room import SwingPoint
    return SwingPoint(
        index=index,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        price=price,
        swing_type=swing_type,
        bar_count_since_prev=10,
    )


class TestPatternHook:
    def test_detects_patterns_from_swing_history(self):
        from src.discovery.pattern_hook import PatternHook

        hook = PatternHook(atr=5.0)
        swings = [
            _make_swing(2050.0, "high", index=10),
            _make_swing(2030.0, "low", index=20),
            _make_swing(2051.0, "high", index=30),
            _make_swing(2025.0, "low", index=40),
        ]
        patterns = hook.detect(swings)
        assert isinstance(patterns, list)
        # Should find the double top
        assert any(p.pattern_type == "double_top" for p in patterns)

    def test_returns_empty_with_insufficient_swings(self):
        from src.discovery.pattern_hook import PatternHook

        hook = PatternHook(atr=5.0)
        patterns = hook.detect([_make_swing(2050.0, "high", index=10)])
        assert patterns == []

    def test_updates_atr(self):
        from src.discovery.pattern_hook import PatternHook

        hook = PatternHook(atr=5.0)
        hook.update_atr(10.0)
        # With higher ATR, the tolerance is wider
        swings = [
            _make_swing(2050.0, "high", index=10),
            _make_swing(2030.0, "low", index=20),
            _make_swing(2058.0, "high", index=30),  # 8 points apart -- within 10*1.5=15 tol
            _make_swing(2025.0, "low", index=40),
        ]
        patterns = hook.detect(swings)
        assert any(p.pattern_type == "double_top" for p in patterns)

    def test_get_confluence_adjustment(self):
        from src.discovery.pattern_hook import PatternHook

        hook = PatternHook(atr=5.0)
        # Two equal highs with widely-spaced lows (no double bottom)
        swings = [
            _make_swing(2050.0, "high", index=10),
            _make_swing(2030.0, "low", index=20),
            _make_swing(2051.0, "high", index=30),
            _make_swing(2015.0, "low", index=40),  # far from first low -- no double bottom
        ]
        patterns = hook.detect(swings)
        # Should find double top only (bearish)
        assert any(p.pattern_type == "double_top" for p in patterns)

        # Double top is bearish -- should penalize longs
        adj = hook.get_confluence_adjustment(patterns, trade_direction="long")
        assert adj < 0

        # Double top is bearish -- should boost shorts
        adj_short = hook.get_confluence_adjustment(patterns, trade_direction="short")
        assert adj_short > 0
