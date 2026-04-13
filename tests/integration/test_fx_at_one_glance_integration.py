"""End-to-end: all evaluators → strategy → signal."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from src.strategy.base import EvalMatrix, EVALUATOR_REGISTRY, STRATEGY_REGISTRY, EvaluatorResult


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


class TestFXAOGConfigParsing:
    def test_nested_strategy_config_is_honoured(self):
        from src.strategy.strategies.fx_at_one_glance import FXAtOneGlance

        strategy = FXAtOneGlance(config={
            'signal': {
                'min_confluence_score': 3,
                'min_tier': 'C',
            },
            'exit': {
                'mode': 'hybrid',
                'partial_close_pct': 50,
                'primary_target': 'v_value',
            },
            'stop_loss': {
                'kijun_buffer_pips': 7,
                'max_stop_pips': 55,
                'min_rr_ratio': 1.2,
            },
            'range_filter': {
                'adx_min': 12,
                'cloud_thickness_min_usd': 0.25,
            },
        })

        assert strategy._min_score == 3
        assert strategy._min_tier == 'C'
        assert strategy._exit_mode == 'hybrid'
        assert strategy._partial_pct == pytest.approx(0.5)
        assert strategy._primary_target_key == 'v_value'
        assert strategy._kijun_buffer == pytest.approx(7.0)
        assert strategy._max_stop == pytest.approx(55.0)
        assert strategy._min_rr == pytest.approx(1.2)
        assert strategy._adx_min == pytest.approx(12.0)
        assert strategy._cloud_thickness_min == pytest.approx(0.25)

    def test_duplicate_setup_cooldown_suppresses_repeat_signal(self):
        from src.strategy.strategies.fx_at_one_glance import FXAtOneGlance

        strategy = FXAtOneGlance(config={
            'signal_cooldown_bars': 12,
            'reentry_price_tolerance_points': 5.0,
            'reentry_stop_tolerance_points': 2.0,
        })
        strategy._decision_counter = 10
        strategy._last_signal_state = {
            'bar': 5,
            'direction': 'long',
            'entry_price': 5103.5,
            'stop_loss': 5098.4,
        }

        assert strategy._should_suppress_signal('long', 5105.0, 5098.9) is True
        assert strategy._should_suppress_signal('short', 5105.0, 5098.9) is False
        assert strategy._should_suppress_signal('long', 5120.0, 5098.9) is False

    def test_range_filter_uses_configured_thresholds(self):
        from src.strategy.strategies.fx_at_one_glance import FXAtOneGlance

        strategy = FXAtOneGlance(config={
            'tf_mode': 'hyperscalp_m15_m5',
            'range_filter': {
                'adx_min': 5,
                'cloud_thickness_min_usd': 0.10,
            },
        })
        matrix = EvalMatrix()
        matrix.set('adx_5M', EvaluatorResult(
            direction=1.0,
            confidence=0.08,
            metadata={'adx': 8.0},
        ))
        matrix.set('ichimoku_5M', EvaluatorResult(
            direction=1.0,
            confidence=1.0,
            metadata={'cloud_thickness': 0.15},
        ))

        assert strategy._filter_range(matrix) is True
