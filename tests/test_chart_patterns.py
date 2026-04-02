"""Tests for chart pattern detection from swing points."""

from datetime import datetime, timezone

import numpy as np
import pytest


def _make_swing(price: float, swing_type: str, index: int = 0, ts: datetime = None) -> "SwingPoint":
    """Helper: build a SwingPoint for testing."""
    from src.strategy.strategies.sss.breathing_room import SwingPoint
    return SwingPoint(
        index=index,
        timestamp=ts or datetime(2026, 1, 1, tzinfo=timezone.utc),
        price=price,
        swing_type=swing_type,
        bar_count_since_prev=10,
    )


class TestChartPatternDataclass:
    def test_pattern_has_required_fields(self):
        from src.discovery.chart_patterns import ChartPattern

        pattern = ChartPattern(
            pattern_type="double_top",
            direction="bearish",
            confidence=0.85,
            key_prices=[2050.0, 2048.0, 2050.5],
            start_index=100,
            end_index=200,
            description="Double top near 2050 resistance",
        )
        assert pattern.pattern_type == "double_top"
        assert pattern.direction == "bearish"
        assert pattern.confidence == 0.85
        assert len(pattern.key_prices) == 3

    def test_pattern_to_dict(self):
        from src.discovery.chart_patterns import ChartPattern

        pattern = ChartPattern(
            pattern_type="head_and_shoulders",
            direction="bearish",
            confidence=0.75,
            key_prices=[2040.0, 2060.0, 2040.0],
            start_index=50,
            end_index=150,
            description="H&S with neckline at 2030",
        )
        d = pattern.to_dict()
        assert d["pattern_type"] == "head_and_shoulders"
        assert isinstance(d["key_prices"], list)
        assert "confidence" in d
