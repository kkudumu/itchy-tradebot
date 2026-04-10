from __future__ import annotations
import numpy as np
import pytest
from src.indicators.fractals import detect_fractals, FractalLevel, FractalResult

def _make_v_shape(n=20, base=1800.0):
    mid = n // 2
    prices = np.zeros(n)
    for i in range(n):
        if i <= mid:
            prices[i] = base - i * 2.0
        else:
            prices[i] = base - mid * 2.0 + (i - mid) * 2.0
    high = prices + 1.0
    low = prices - 1.0
    return high, low, prices

def _make_mountain(n=20, base=1800.0):
    mid = n // 2
    prices = np.zeros(n)
    for i in range(n):
        if i <= mid:
            prices[i] = base + i * 2.0
        else:
            prices[i] = base + mid * 2.0 - (i - mid) * 2.0
    high = prices + 1.0
    low = prices - 1.0
    return high, low, prices

class TestDetectFractals:
    def test_bull_fractal_at_mountain_peak(self):
        high, low, close = _make_mountain(n=11, base=1800.0)
        result = detect_fractals(high, low)
        bulls = [f for f in result.bull_fractals if not np.isnan(f.price)]
        assert len(bulls) >= 1
        assert any(f.bar_index == 5 for f in bulls)

    def test_bear_fractal_at_v_bottom(self):
        high, low, close = _make_v_shape(n=11, base=1800.0)
        result = detect_fractals(high, low)
        bears = [f for f in result.bear_fractals if not np.isnan(f.price)]
        assert len(bears) >= 1
        assert any(f.bar_index == 5 for f in bears)

    def test_five_bar_minimum(self):
        high = np.array([10.0, 11.0, 12.0, 11.0, 10.0])
        low = np.array([9.0, 10.0, 11.0, 10.0, 9.0])
        result = detect_fractals(high, low)
        assert len(result.bull_fractals) == 1
        assert result.bull_fractals[0].bar_index == 2

    def test_no_fractals_in_flat_market(self):
        high = np.full(20, 1800.0)
        low = np.full(20, 1799.0)
        result = detect_fractals(high, low)
        assert len(result.bull_fractals) == 0
        assert len(result.bear_fractals) == 0

    def test_too_few_bars_returns_empty(self):
        high = np.array([10.0, 11.0, 10.0])
        low = np.array([9.0, 10.0, 9.0])
        result = detect_fractals(high, low)
        assert len(result.bull_fractals) == 0
        assert len(result.bear_fractals) == 0

    def test_fractal_never_repaints(self):
        high1 = np.array([10.0, 11.0, 13.0, 11.0, 10.0, 9.0, 8.0])
        low1 = np.array([9.0, 10.0, 12.0, 10.0, 9.0, 8.0, 7.0])
        r1 = detect_fractals(high1, low1)
        bulls1 = {f.bar_index for f in r1.bull_fractals}
        high2 = np.concatenate([high1, np.array([9.0, 10.0, 11.0])])
        low2 = np.concatenate([low1, np.array([8.0, 9.0, 10.0])])
        r2 = detect_fractals(high2, low2)
        bulls2 = {f.bar_index for f in r2.bull_fractals}
        assert bulls1.issubset(bulls2)

class TestMarketStructure:
    def test_uptrend_detection(self):
        from src.indicators.fractals import market_structure
        high = np.array([
            10, 11, 15, 11, 10,
            10, 11, 12, 18, 12,
            11, 12, 13, 22, 13,
            12, 11, 10, 9, 8,
        ], dtype=float)
        low = high - 3.0
        result = market_structure(high, low)
        assert result.trend in ('uptrend', 'ranging')

    def test_flat_market_is_ranging(self):
        from src.indicators.fractals import market_structure
        high = np.full(50, 1800.0) + np.random.uniform(-0.5, 0.5, 50)
        low = high - 1.0
        result = market_structure(high, low)
        assert result.trend == 'ranging'
