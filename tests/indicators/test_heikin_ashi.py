from __future__ import annotations
import numpy as np
import pytest
from src.indicators.heikin_ashi import compute_heikin_ashi, ha_trend_signal, HACandle

class TestComputeHeikinAshi:
    def test_close_is_ohlc_average(self):
        o = np.array([100.0, 105.0])
        h = np.array([110.0, 115.0])
        l = np.array([95.0, 100.0])
        c = np.array([105.0, 110.0])
        ha = compute_heikin_ashi(o, h, l, c)
        expected = (105.0 + 115.0 + 100.0 + 110.0) / 4
        assert ha.close[1] == pytest.approx(expected)

    def test_open_is_midpoint_of_previous(self):
        o = np.array([100.0, 105.0, 108.0])
        h = np.array([110.0, 115.0, 118.0])
        l = np.array([95.0, 100.0, 103.0])
        c = np.array([105.0, 110.0, 115.0])
        ha = compute_heikin_ashi(o, h, l, c)
        expected = (ha.open[1] + ha.close[1]) / 2
        assert ha.open[2] == pytest.approx(expected)

    def test_output_length_matches_input(self):
        n = 50
        o = np.random.uniform(100, 110, n)
        h = o + np.random.uniform(0, 5, n)
        l = o - np.random.uniform(0, 5, n)
        c = np.random.uniform(l, h)
        ha = compute_heikin_ashi(o, h, l, c)
        assert len(ha.open) == n

class TestHATrendSignal:
    def test_strong_bullish_no_lower_wick(self):
        candle = HACandle(open=100.0, high=110.0, low=100.0, close=108.0)
        assert ha_trend_signal(candle) == 'strong_bullish'

    def test_strong_bearish_no_upper_wick(self):
        candle = HACandle(open=108.0, high=108.0, low=95.0, close=100.0)
        assert ha_trend_signal(candle) == 'strong_bearish'

    def test_doji(self):
        candle = HACandle(open=100.0, high=105.0, low=95.0, close=100.1)
        assert ha_trend_signal(candle) == 'indecision'
