from __future__ import annotations
import numpy as np
import pytest
from src.indicators.price_action import detect_patterns, PriceActionResult

def _candles(ohlc_list):
    arr = np.array(ohlc_list, dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

class TestTweezerDetection:
    def test_tweezer_bottom(self):
        o, h, l, c = _candles([
            (1810, 1812, 1800, 1802),
            (1802, 1811, 1800, 1809),
        ])
        result = detect_patterns(o, h, l, c, tick_tolerance=2)
        assert result.tweezer_bottom is True

    def test_tweezer_top(self):
        o, h, l, c = _candles([
            (1800, 1812, 1799, 1810),
            (1810, 1812, 1801, 1803),
        ])
        result = detect_patterns(o, h, l, c, tick_tolerance=2)
        assert result.tweezer_top is True

    def test_same_color_not_tweezer(self):
        o, h, l, c = _candles([
            (1800, 1812, 1800, 1810),
            (1805, 1813, 1800, 1811),
        ])
        result = detect_patterns(o, h, l, c, tick_tolerance=2)
        assert result.tweezer_bottom is False

class TestInsideBar:
    def test_inside_bar_count(self):
        o, h, l, c = _candles([
            (1800, 1815, 1795, 1810),
            (1803, 1808, 1801, 1807),
            (1804, 1809, 1802, 1806),
            (1805, 1807, 1803, 1805),
            (1806, 1818, 1804, 1815),
        ])
        result = detect_patterns(o, h, l, c)
        assert result.inside_bar_count >= 2
        assert result.inside_bar_breakout == 'up'

class TestEngulfing:
    def test_bullish_engulfing(self):
        o, h, l, c = _candles([
            (1810, 1812, 1800, 1802),
            (1800, 1815, 1798, 1812),
        ])
        result = detect_patterns(o, h, l, c)
        assert result.engulfing_bullish is True

    def test_bearish_engulfing(self):
        o, h, l, c = _candles([
            (1800, 1812, 1798, 1810),
            (1812, 1814, 1797, 1798),
        ])
        result = detect_patterns(o, h, l, c)
        assert result.engulfing_bearish is True

class TestPinBar:
    def test_hammer(self):
        o, h, l, c = _candles([(1802, 1805, 1790, 1804)])
        result = detect_patterns(o, h, l, c)
        assert result.pin_bar_bullish is True

    def test_shooting_star(self):
        o, h, l, c = _candles([(1804, 1818, 1801, 1802)])
        result = detect_patterns(o, h, l, c)
        assert result.pin_bar_bearish is True
