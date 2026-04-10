from __future__ import annotations
import numpy as np
import pytest
from src.indicators.wave_patterns import (
    WaveAnalyzer, n_value, v_value, e_value, nt_value,
)


class TestTargetCalculations:
    def test_bullish_n_value(self):
        assert n_value(1800.0, 1900.0, 1850.0, 'bullish') == pytest.approx(1950.0)

    def test_bullish_v_value(self):
        assert v_value(1800.0, 1900.0, 1850.0, 'bullish') == pytest.approx(1950.0)

    def test_bullish_e_value(self):
        assert e_value(1800.0, 1900.0, 1850.0, 'bullish') == pytest.approx(2000.0)

    def test_bullish_nt_value(self):
        assert nt_value(1800.0, 1900.0, 1850.0, 'bullish') == pytest.approx(1900.0)

    def test_bearish_n_value(self):
        assert n_value(1900.0, 1800.0, 1850.0, 'bearish') == pytest.approx(1750.0)


class TestWaveClassification:
    def test_n_wave_detected(self):
        analyzer = WaveAnalyzer()
        swings = [
            {'type': 'low', 'price': 1800.0, 'bar': 0},
            {'type': 'high', 'price': 1900.0, 'bar': 20},
            {'type': 'low', 'price': 1850.0, 'bar': 40},
            {'type': 'high', 'price': 1960.0, 'bar': 60},
        ]
        result = analyzer.classify(swings, current_price=1960.0)
        assert result['wave_type'] == 'N'
        assert result['direction'] == 'bullish'

    def test_flat_is_box(self):
        analyzer = WaveAnalyzer()
        swings = [
            {'type': 'low', 'price': 1799.0, 'bar': 0},
            {'type': 'high', 'price': 1801.0, 'bar': 10},
            {'type': 'low', 'price': 1799.5, 'bar': 20},
            {'type': 'high', 'price': 1800.5, 'bar': 30},
        ]
        result = analyzer.classify(swings, current_price=1800.0)
        assert result['wave_type'] in ('box', 'P')


class TestElliottCounting:
    def test_wave3_not_shortest(self):
        from src.indicators.wave_patterns import count_elliott
        swings = [1800, 1850, 1820, 1910, 1870, 1920]
        result = count_elliott(swings, 'bullish')
        assert result is not None
        assert result['confidence'] > 0

    def test_wave2_100pct_retrace_invalid(self):
        from src.indicators.wave_patterns import count_elliott
        swings = [1800, 1850, 1800, 1900, 1850, 1920]
        result = count_elliott(swings, 'bullish')
        assert result is None
