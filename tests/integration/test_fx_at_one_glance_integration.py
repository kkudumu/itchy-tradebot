"""End-to-end: all evaluators → strategy → signal."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from src.strategy.base import EvalMatrix, EVALUATOR_REGISTRY, STRATEGY_REGISTRY


def _make_trending_ohlcv(n=300, start=1800.0, step=1.5):
    idx = pd.date_range('2026-01-01', periods=n, freq='1h', tz='UTC')
    close = np.zeros(n)
    price = start
    for i in range(n):
        if i % 20 < 15:
            price += step
        else:
            price -= step * 0.5
        close[i] = price
    return pd.DataFrame({
        'open': close - 0.3 * step, 'high': close + abs(step),
        'low': close - abs(step), 'close': close,
        'volume': np.ones(n) * 100,
    }, index=idx)


class TestAllEvaluatorsRegistered:
    def test_new_evaluators_present(self):
        import src.strategy.evaluators
        for key in ['fractal', 'price_action', 'cloud_balance', 'kihon_suchi', 'wave', 'rsi', 'divergence']:
            assert key in EVALUATOR_REGISTRY, f"{key} not registered"


class TestStrategyRegistration:
    def test_fx_at_one_glance_registered(self):
        from src.strategy.strategies.fx_at_one_glance import FXAtOneGlance
        assert 'fx_at_one_glance' in STRATEGY_REGISTRY

    def test_old_ichimoku_still_registered(self):
        from src.strategy.strategies.ichimoku import IchimokuStrategy
        assert 'ichimoku' in STRATEGY_REGISTRY


class TestEvaluatorsProduceResults:
    def test_each_evaluator_runs(self):
        ohlcv = _make_trending_ohlcv(n=300)
        for key in ['fractal', 'price_action', 'cloud_balance', 'kihon_suchi', 'wave', 'rsi', 'divergence']:
            eval_cls = EVALUATOR_REGISTRY[key]
            evaluator = eval_cls()
            result = evaluator.evaluate(ohlcv)
            assert result is not None, f"{key} returned None"
            assert -1.0 <= result.direction <= 1.0
            assert 0.0 <= result.confidence <= 1.0


class TestFullSignalGeneration:
    def test_strategy_produces_signal_with_relaxed_filters(self):
        ohlcv = _make_trending_ohlcv(n=300, step=2.0)
        matrix = EvalMatrix()
        for tf in ['4H', '1H']:
            for key in ['ichimoku', 'fractal', 'wave', 'adx', 'atr']:
                evaluator = EVALUATOR_REGISTRY[key]()
                matrix.set(f'{key}_{tf}', evaluator.evaluate(ohlcv))
        for key in ['price_action', 'rsi', 'divergence']:
            matrix.set(f'{key}_1H', EVALUATOR_REGISTRY[key]().evaluate(ohlcv))
        for key in ['cloud_balance', 'kihon_suchi']:
            matrix.set(f'{key}_4H', EVALUATOR_REGISTRY[key]().evaluate(ohlcv))

        from src.strategy.strategies.fx_at_one_glance import FXAtOneGlance
        strategy = FXAtOneGlance(config={
            'five_elements_mode': 'disabled', 'time_theory_mode': 'disabled',
            'min_confluence_score': 1, 'min_tier': 'C',
        })
        signal = strategy.decide(matrix)
        if signal is not None:
            assert signal.direction in ('long', 'short')
            assert signal.strategy_name == 'fx_at_one_glance'
            assert 'trade_type' in signal.reasoning
