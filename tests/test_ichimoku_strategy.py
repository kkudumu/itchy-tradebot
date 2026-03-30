"""Tests for IchimokuStrategy — the concrete Strategy implementation."""
from __future__ import annotations

import pytest
from datetime import datetime

from src.strategy.base import (
    EvalMatrix,
    EvalRequirement,
    EvaluatorResult,
    ConfluenceResult,
    STRATEGY_REGISTRY,
)
from src.strategy.strategies.ichimoku import IchimokuStrategy
from src.strategy.signal_engine import Signal


def _make_bullish_matrix() -> EvalMatrix:
    """Create an EvalMatrix with all conditions aligned for a bullish signal."""
    m = EvalMatrix()

    # 4H: bullish cloud
    m.set('ichimoku_4H', EvaluatorResult(
        direction=1.0, confidence=0.8,
        metadata={'cloud_direction': 1, 'tk_cross': 1, 'cloud_position': 1,
                  'chikou_confirmed': 1, 'cloud_thickness': 5.0,
                  'tenkan': 1920.0, 'kijun': 1915.0, 'senkou_a': 1910.0,
                  'senkou_b': 1905.0, 'chikou': 1925.0, 'close': 1922.0}
    ))

    # 1H: TK aligned bullish
    m.set('ichimoku_1H', EvaluatorResult(
        direction=1.0, confidence=0.7,
        metadata={'cloud_direction': 1, 'tk_cross': 1, 'cloud_position': 1,
                  'chikou_confirmed': 1, 'cloud_thickness': 3.0,
                  'tenkan': 1918.0, 'kijun': 1916.0, 'senkou_a': 1912.0,
                  'senkou_b': 1908.0, 'chikou': 1920.0, 'close': 1920.0}
    ))

    # 15M: all confirmed bullish
    m.set('ichimoku_15M', EvaluatorResult(
        direction=1.0, confidence=0.9,
        metadata={'cloud_direction': 1, 'tk_cross': 1, 'cloud_position': 1,
                  'chikou_confirmed': 1, 'cloud_thickness': 2.0,
                  'tenkan': 1919.0, 'kijun': 1917.0, 'senkou_a': 1914.0,
                  'senkou_b': 1912.0, 'chikou': 1921.0, 'close': 1920.0}
    ))

    # 5M: close near kijun (entry timing)
    m.set('ichimoku_5M', EvaluatorResult(
        direction=1.0, confidence=0.6,
        metadata={'cloud_direction': 1, 'tk_cross': 1, 'cloud_position': 1,
                  'chikou_confirmed': 1, 'cloud_thickness': 1.5,
                  'tenkan': 1919.5, 'kijun': 1919.0, 'senkou_a': 1916.0,
                  'senkou_b': 1914.0, 'chikou': 1920.0,
                  'close': 1919.2, 'timestamp': datetime(2024, 1, 2, 10, 30)}
    ))

    # ADX: trending
    m.set('adx_15M', EvaluatorResult(
        direction=1.0, confidence=0.35,
        metadata={'adx': 35.0, 'plus_di': 30.0, 'minus_di': 15.0, 'is_trending': True}
    ))

    # ATR
    m.set('atr_15M', EvaluatorResult(
        direction=0.0, confidence=0.5,
        metadata={'atr': 3.0}
    ))

    # Session: london (active)
    m.set('session_5M', EvaluatorResult(
        direction=0.0, confidence=1.0,
        metadata={'session': 'london', 'is_active': True}
    ))

    return m


class TestRegistration:
    def test_registered_in_strategy_registry(self):
        assert 'ichimoku' in STRATEGY_REGISTRY
        assert STRATEGY_REGISTRY['ichimoku'] is IchimokuStrategy

    def test_name_set(self):
        assert IchimokuStrategy.name == 'ichimoku'


class TestDecide:
    def test_bullish_signal_generated(self):
        strategy = IchimokuStrategy()
        matrix = _make_bullish_matrix()
        signal = strategy.decide(matrix)
        assert signal is not None
        assert isinstance(signal, Signal)
        assert signal.direction == "long"
        assert signal.entry_price == 1919.2
        assert signal.stop_loss < signal.entry_price
        assert signal.take_profit > signal.entry_price

    def test_no_signal_when_4h_flat(self):
        strategy = IchimokuStrategy()
        matrix = _make_bullish_matrix()
        # Override 4H to flat
        matrix.set('ichimoku_4H', EvaluatorResult(
            direction=0.0, confidence=0.0,
            metadata={'cloud_direction': 0, 'tk_cross': 0}
        ))
        assert strategy.decide(matrix) is None

    def test_no_signal_when_1h_opposing(self):
        strategy = IchimokuStrategy()
        matrix = _make_bullish_matrix()
        # Override 1H to bearish TK
        matrix.set('ichimoku_1H', EvaluatorResult(
            direction=-1.0, confidence=0.5,
            metadata={'cloud_direction': -1, 'tk_cross': -1}
        ))
        assert strategy.decide(matrix) is None

    def test_no_signal_when_15m_chikou_fails(self):
        strategy = IchimokuStrategy()
        matrix = _make_bullish_matrix()
        matrix.set('ichimoku_15M', EvaluatorResult(
            direction=1.0, confidence=0.5,
            metadata={'tk_cross': 1, 'cloud_position': 1, 'chikou_confirmed': 0, 'cloud_thickness': 2.0}
        ))
        assert strategy.decide(matrix) is None

    def test_no_signal_when_5m_too_far_from_kijun(self):
        strategy = IchimokuStrategy()
        matrix = _make_bullish_matrix()
        # Close far from kijun (distance > 0.5 * ATR = 1.5)
        matrix.set('ichimoku_5M', EvaluatorResult(
            direction=1.0, confidence=0.5,
            metadata={'close': 1925.0, 'kijun': 1919.0, 'tk_cross': 1,
                      'cloud_position': 1, 'chikou_confirmed': 1, 'cloud_thickness': 1.0,
                      'timestamp': datetime(2024, 1, 2, 10, 30)}
        ))
        assert strategy.decide(matrix) is None

    def test_bearish_signal(self):
        strategy = IchimokuStrategy()
        m = EvalMatrix()
        # All bearish aligned
        for tf in ['4H', '1H', '15M']:
            m.set(f'ichimoku_{tf}', EvaluatorResult(
                direction=-1.0, confidence=0.8,
                metadata={'cloud_direction': -1, 'tk_cross': -1, 'cloud_position': -1,
                          'chikou_confirmed': -1, 'cloud_thickness': 3.0,
                          'tenkan': 1900.0, 'kijun': 1905.0, 'close': 1898.0}
            ))
        m.set('ichimoku_5M', EvaluatorResult(
            direction=-1.0, confidence=0.6,
            metadata={'close': 1901.0, 'kijun': 1901.5, 'cloud_direction': -1,
                      'tk_cross': -1, 'cloud_position': -1, 'chikou_confirmed': -1,
                      'cloud_thickness': 1.0, 'timestamp': datetime(2024, 1, 2, 14, 0)}
        ))
        m.set('adx_15M', EvaluatorResult(direction=-1.0, confidence=0.35, metadata={'adx': 35.0}))
        m.set('atr_15M', EvaluatorResult(direction=0.0, confidence=0.5, metadata={'atr': 3.0}))
        m.set('session_5M', EvaluatorResult(direction=0.0, confidence=1.0, metadata={'session': 'london', 'is_active': True}))

        signal = strategy.decide(m)
        assert signal is not None
        assert signal.direction == "short"
        assert signal.stop_loss > signal.entry_price
        assert signal.take_profit < signal.entry_price


class TestScoreConfluence:
    def test_full_score_bullish(self):
        strategy = IchimokuStrategy()
        matrix = _make_bullish_matrix()
        result = strategy.score_confluence(matrix, direction="long", zone_count=1)
        # 5 ichimoku + ADX + session + zone = 8
        assert result.score >= 7
        assert result.quality_tier == "A+"

    def test_minimum_score_tier_c(self):
        strategy = IchimokuStrategy()
        m = EvalMatrix()
        # 4 ichimoku dimensions aligned, but 5M NOT near kijun (distance > 0.5 * ATR)
        # close=1920.0, kijun=1925.0, atr=3.0 → distance=5.0 > 1.5 threshold
        m.set('ichimoku_4H', EvaluatorResult(direction=1.0, confidence=0.5, metadata={'cloud_direction': 1}))
        m.set('ichimoku_1H', EvaluatorResult(direction=1.0, confidence=0.5, metadata={'tk_cross': 1}))
        m.set('ichimoku_15M', EvaluatorResult(direction=1.0, confidence=0.5, metadata={'tk_cross': 1, 'chikou_confirmed': 1}))
        m.set('ichimoku_5M', EvaluatorResult(direction=1.0, confidence=0.5, metadata={'close': 1920.0, 'kijun': 1925.0}))
        m.set('adx_15M', EvaluatorResult(direction=0.0, confidence=0.1, metadata={'adx': 15.0}))
        m.set('atr_15M', EvaluatorResult(direction=0.0, confidence=0.5, metadata={'atr': 3.0}))
        m.set('session_5M', EvaluatorResult(direction=0.0, confidence=0.0, metadata={'session': 'off_hours'}))
        result = strategy.score_confluence(m, direction="long")
        assert result.score == 4
        assert result.quality_tier == "C"

    def test_no_trade_when_below_minimum(self):
        strategy = IchimokuStrategy()
        m = EvalMatrix()
        m.set('ichimoku_4H', EvaluatorResult(direction=1.0, confidence=0.5, metadata={'cloud_direction': 1}))
        m.set('ichimoku_1H', EvaluatorResult(direction=0.0, confidence=0.0, metadata={'tk_cross': 0}))
        m.set('ichimoku_15M', EvaluatorResult(direction=0.0, confidence=0.0, metadata={'tk_cross': 0, 'chikou_confirmed': 0}))
        m.set('ichimoku_5M', EvaluatorResult(direction=0.0, confidence=0.0, metadata={'close': 0.0, 'kijun': float('nan')}))
        m.set('adx_15M', EvaluatorResult(direction=0.0, confidence=0.0, metadata={'adx': 10.0}))
        m.set('atr_15M', EvaluatorResult(direction=0.0, confidence=0.0, metadata={'atr': 0.0}))
        m.set('session_5M', EvaluatorResult(direction=0.0, confidence=0.0, metadata={'session': 'off_hours'}))
        result = strategy.score_confluence(m, direction="long")
        assert result.quality_tier == "no_trade"


class TestPopulateEdgeContext:
    def test_returns_kijun_and_cloud_thickness(self):
        strategy = IchimokuStrategy()
        m = EvalMatrix()
        m.set('ichimoku_5M', EvaluatorResult(direction=0.0, confidence=0.0, metadata={'kijun': 1920.5}))
        m.set('ichimoku_15M', EvaluatorResult(direction=0.0, confidence=0.0, metadata={'cloud_thickness': 5.0}))
        ctx = strategy.populate_edge_context(m)
        assert ctx['kijun'] == 1920.5
        assert ctx['cloud_thickness'] == 5.0

    def test_empty_when_no_data(self):
        strategy = IchimokuStrategy()
        m = EvalMatrix()
        ctx = strategy.populate_edge_context(m)
        assert ctx == {}


class TestSuggestParams:
    def test_returns_expected_keys(self):
        strategy = IchimokuStrategy()

        class MockTrial:
            def suggest_float(self, name, low, high):
                return (low + high) / 2
            def suggest_int(self, name, low, high):
                return (low + high) // 2

        params = strategy.suggest_params(MockTrial())
        expected_keys = {
            'tenkan_period', 'kijun_period', 'senkou_b_period',
            'adx_threshold', 'atr_stop_multiplier', 'tp_r_multiple',
            'kijun_trail_start_r', 'min_confluence_score',
            'initial_risk_pct', 'reduced_risk_pct',
        }
        assert set(params.keys()) == expected_keys
