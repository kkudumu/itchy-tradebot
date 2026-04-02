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


class TestDoubleTopBottom:
    def test_detects_double_top(self):
        from src.discovery.chart_patterns import PatternDetector, ChartPattern

        # Classic double top: high, low, high (roughly equal peaks)
        swings = [
            _make_swing(2050.0, "high", index=10),
            _make_swing(2030.0, "low", index=20),
            _make_swing(2051.0, "high", index=30),  # second peak ~equal
            _make_swing(2025.0, "low", index=40),
        ]
        detector = PatternDetector(atr=5.0)
        patterns = detector.detect_double_tops(swings)

        assert len(patterns) >= 1
        p = patterns[0]
        assert p.pattern_type == "double_top"
        assert p.direction == "bearish"
        assert p.confidence > 0.5

    def test_detects_double_bottom(self):
        from src.discovery.chart_patterns import PatternDetector

        # Classic double bottom: low, high, low (roughly equal troughs)
        swings = [
            _make_swing(2020.0, "low", index=10),
            _make_swing(2040.0, "high", index=20),
            _make_swing(2019.5, "low", index=30),  # second trough ~equal
            _make_swing(2045.0, "high", index=40),
        ]
        detector = PatternDetector(atr=5.0)
        patterns = detector.detect_double_bottoms(swings)

        assert len(patterns) >= 1
        p = patterns[0]
        assert p.pattern_type == "double_bottom"
        assert p.direction == "bullish"

    def test_rejects_unequal_peaks(self):
        from src.discovery.chart_patterns import PatternDetector

        # Peaks too far apart to be a double top
        swings = [
            _make_swing(2050.0, "high", index=10),
            _make_swing(2030.0, "low", index=20),
            _make_swing(2070.0, "high", index=30),  # 20 points higher -- not double top
            _make_swing(2025.0, "low", index=40),
        ]
        detector = PatternDetector(atr=5.0)
        patterns = detector.detect_double_tops(swings)

        assert len(patterns) == 0

    def test_empty_swings_returns_empty(self):
        from src.discovery.chart_patterns import PatternDetector

        detector = PatternDetector(atr=5.0)
        assert detector.detect_double_tops([]) == []
        assert detector.detect_double_bottoms([]) == []


class TestHeadAndShoulders:
    def test_detects_head_and_shoulders(self):
        from src.discovery.chart_patterns import PatternDetector

        # Classic H&S: left shoulder, head (highest), right shoulder (roughly equal to left)
        swings = [
            _make_swing(2040.0, "high", index=10),   # left shoulder
            _make_swing(2025.0, "low", index=15),     # left neckline
            _make_swing(2055.0, "high", index=25),    # head (highest)
            _make_swing(2024.0, "low", index=35),     # right neckline
            _make_swing(2041.0, "high", index=45),    # right shoulder (~left)
            _make_swing(2020.0, "low", index=55),
        ]
        detector = PatternDetector(atr=5.0)
        patterns = detector.detect_head_and_shoulders(swings)

        assert len(patterns) >= 1
        p = patterns[0]
        assert p.pattern_type == "head_and_shoulders"
        assert p.direction == "bearish"
        assert p.confidence > 0.5

    def test_detects_inverse_head_and_shoulders(self):
        from src.discovery.chart_patterns import PatternDetector

        # Inverse H&S: left shoulder, head (lowest), right shoulder (~left)
        swings = [
            _make_swing(2030.0, "low", index=10),     # left shoulder
            _make_swing(2045.0, "high", index=15),     # left neckline
            _make_swing(2015.0, "low", index=25),      # head (lowest)
            _make_swing(2046.0, "high", index=35),     # right neckline
            _make_swing(2029.0, "low", index=45),      # right shoulder (~left)
            _make_swing(2050.0, "high", index=55),
        ]
        detector = PatternDetector(atr=5.0)
        patterns = detector.detect_inverse_head_and_shoulders(swings)

        assert len(patterns) >= 1
        p = patterns[0]
        assert p.pattern_type == "inverse_head_and_shoulders"
        assert p.direction == "bullish"

    def test_rejects_when_head_not_highest(self):
        from src.discovery.chart_patterns import PatternDetector

        # "Head" is not the highest peak -- not valid H&S
        swings = [
            _make_swing(2050.0, "high", index=10),
            _make_swing(2025.0, "low", index=15),
            _make_swing(2045.0, "high", index=25),    # head LOWER than left shoulder
            _make_swing(2024.0, "low", index=35),
            _make_swing(2048.0, "high", index=45),
            _make_swing(2020.0, "low", index=55),
        ]
        detector = PatternDetector(atr=5.0)
        patterns = detector.detect_head_and_shoulders(swings)
        assert len(patterns) == 0


class TestTriangles:
    def test_detects_ascending_triangle(self):
        from src.discovery.chart_patterns import PatternDetector

        # Ascending triangle: flat highs, rising lows
        swings = [
            _make_swing(2050.0, "high", index=10),
            _make_swing(2030.0, "low", index=15),
            _make_swing(2050.5, "high", index=25),
            _make_swing(2035.0, "low", index=30),     # higher low
            _make_swing(2051.0, "high", index=40),
            _make_swing(2040.0, "low", index=45),     # higher low
        ]
        detector = PatternDetector(atr=5.0)
        patterns = detector.detect_triangles(swings)

        assert any(p.pattern_type == "ascending_triangle" for p in patterns)

    def test_detects_descending_triangle(self):
        from src.discovery.chart_patterns import PatternDetector

        # Descending triangle: flat lows, falling highs
        swings = [
            _make_swing(2050.0, "high", index=10),
            _make_swing(2020.0, "low", index=15),
            _make_swing(2045.0, "high", index=25),     # lower high
            _make_swing(2020.5, "low", index=30),
            _make_swing(2040.0, "high", index=40),     # lower high
            _make_swing(2019.5, "low", index=45),
        ]
        detector = PatternDetector(atr=5.0)
        patterns = detector.detect_triangles(swings)

        assert any(p.pattern_type == "descending_triangle" for p in patterns)

    def test_detects_symmetrical_triangle(self):
        from src.discovery.chart_patterns import PatternDetector

        # Symmetrical: lower highs AND higher lows
        swings = [
            _make_swing(2055.0, "high", index=10),
            _make_swing(2025.0, "low", index=15),
            _make_swing(2050.0, "high", index=25),     # lower high
            _make_swing(2030.0, "low", index=30),      # higher low
            _make_swing(2045.0, "high", index=40),     # lower high
            _make_swing(2035.0, "low", index=45),      # higher low
        ]
        detector = PatternDetector(atr=5.0)
        patterns = detector.detect_triangles(swings)

        assert any(p.pattern_type == "symmetrical_triangle" for p in patterns)


class TestWedges:
    def test_detects_rising_wedge(self):
        from src.discovery.chart_patterns import PatternDetector

        # Rising wedge: higher highs AND higher lows, converging
        swings = [
            _make_swing(2040.0, "high", index=10),
            _make_swing(2020.0, "low", index=15),
            _make_swing(2050.0, "high", index=25),     # higher high
            _make_swing(2035.0, "low", index=30),      # higher low (bigger move)
            _make_swing(2055.0, "high", index=40),     # higher high
            _make_swing(2045.0, "low", index=45),      # higher low (bigger move)
        ]
        detector = PatternDetector(atr=5.0)
        patterns = detector.detect_wedges(swings)

        assert any(p.pattern_type == "rising_wedge" for p in patterns)

    def test_detects_falling_wedge(self):
        from src.discovery.chart_patterns import PatternDetector

        # Falling wedge: lower highs AND lower lows, converging
        swings = [
            _make_swing(2060.0, "high", index=10),
            _make_swing(2040.0, "low", index=15),
            _make_swing(2050.0, "high", index=25),     # lower high
            _make_swing(2035.0, "low", index=30),      # lower low (smaller drop)
            _make_swing(2045.0, "high", index=40),     # lower high
            _make_swing(2032.0, "low", index=45),      # lower low (smaller drop)
        ]
        detector = PatternDetector(atr=5.0)
        patterns = detector.detect_wedges(swings)

        assert any(p.pattern_type == "falling_wedge" for p in patterns)

    def test_needs_minimum_swings(self):
        from src.discovery.chart_patterns import PatternDetector

        detector = PatternDetector(atr=5.0)
        swings = [_make_swing(2050.0, "high", index=10)]
        assert detector.detect_wedges(swings) == []
        assert detector.detect_triangles(swings) == []
