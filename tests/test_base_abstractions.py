"""
Unit tests for src/strategy/base.py — ABCs, dataclasses, and registries.

Test plan
---------
1.  EVALUATOR_REGISTRY: subclass with key= appears in registry
2.  STRATEGY_REGISTRY:  subclass with key= appears in registry
3.  Duplicate key raises ValueError for both registries
4.  EvalMatrix: set, get, missing key → None, __contains__
5.  EvalRequirement construction
6.  ConfluenceResult construction
7.  ExitDecision construction and defaults
8.  Subclass without key= is NOT added to any registry
"""

from __future__ import annotations

import pytest

from src.strategy.base import (
    EVALUATOR_REGISTRY,
    STRATEGY_REGISTRY,
    ConfluenceResult,
    EvalMatrix,
    EvalRequirement,
    EvaluatorResult,
    ExitDecision,
    Evaluator,
    Strategy,
    TradingMode,
)


# ---------------------------------------------------------------------------
# Helpers — concrete stubs used across multiple tests
# ---------------------------------------------------------------------------

class _DummyEvaluator(Evaluator, key='_test_eval'):
    """Minimal concrete Evaluator for registry tests."""

    def evaluate(self, ohlcv):  # type: ignore[override]
        return EvaluatorResult(direction=0.0, confidence=0.0)


class _DummyStrategy(Strategy, key='_test_strat'):
    """Minimal concrete Strategy for registry tests."""

    required_evaluators = []

    def decide(self, eval_matrix):
        return None

    def score_confluence(self, eval_matrix):
        return ConfluenceResult(score=0, quality_tier='no_trade')


class _DummyTradingMode(TradingMode):
    """Minimal concrete TradingMode (no key registration)."""

    def check_exit(self, trade, current_data, eval_results):
        return ExitDecision(action='hold')


# ---------------------------------------------------------------------------
# 1. EVALUATOR_REGISTRY
# ---------------------------------------------------------------------------

class TestEvaluatorRegistry:
    def test_registered_key_present(self):
        assert '_test_eval' in EVALUATOR_REGISTRY

    def test_registered_class_is_correct(self):
        assert EVALUATOR_REGISTRY['_test_eval'] is _DummyEvaluator

    def test_name_attribute_set_on_class(self):
        assert _DummyEvaluator.name == '_test_eval'

    def test_duplicate_key_raises(self):
        with pytest.raises(ValueError, match="Duplicate evaluator key"):
            class _DupEval(Evaluator, key='_test_eval'):
                def evaluate(self, ohlcv):
                    return EvaluatorResult(direction=0.0, confidence=0.0)

    def test_subclass_without_key_not_registered(self):
        class _NoKeyEval(Evaluator):
            def evaluate(self, ohlcv):
                return EvaluatorResult(direction=0.0, confidence=0.0)

        assert _NoKeyEval not in EVALUATOR_REGISTRY.values()


# ---------------------------------------------------------------------------
# 2. STRATEGY_REGISTRY
# ---------------------------------------------------------------------------

class TestStrategyRegistry:
    def test_registered_key_present(self):
        assert '_test_strat' in STRATEGY_REGISTRY

    def test_registered_class_is_correct(self):
        assert STRATEGY_REGISTRY['_test_strat'] is _DummyStrategy

    def test_name_attribute_set_on_class(self):
        assert _DummyStrategy.name == '_test_strat'

    def test_duplicate_key_raises(self):
        with pytest.raises(ValueError, match="Duplicate strategy key"):
            class _DupStrat(Strategy, key='_test_strat'):
                def decide(self, eval_matrix):
                    return None

                def score_confluence(self, eval_matrix):
                    return ConfluenceResult(score=0, quality_tier='no_trade')

    def test_subclass_without_key_not_registered(self):
        class _NoKeyStrat(Strategy):
            def decide(self, eval_matrix):
                return None

            def score_confluence(self, eval_matrix):
                return ConfluenceResult(score=0, quality_tier='no_trade')

        assert _NoKeyStrat not in STRATEGY_REGISTRY.values()


# ---------------------------------------------------------------------------
# 3. EvalMatrix
# ---------------------------------------------------------------------------

class TestEvalMatrix:
    def setup_method(self):
        self.matrix = EvalMatrix()
        self.result = EvaluatorResult(direction=1.0, confidence=0.9, metadata={'atr': 3.2})

    def test_set_and_get(self):
        self.matrix.set('ichimoku_4H', self.result)
        retrieved = self.matrix.get('ichimoku_4H')
        assert retrieved is self.result

    def test_get_missing_key_returns_none(self):
        assert self.matrix.get('nonexistent_1H') is None

    def test_contains_true(self):
        self.matrix.set('adx_1H', self.result)
        assert 'adx_1H' in self.matrix

    def test_contains_false(self):
        assert 'missing_key' not in self.matrix

    def test_keys_reflects_stored_items(self):
        self.matrix.set('ichimoku_4H', self.result)
        self.matrix.set('adx_15M', self.result)
        assert set(self.matrix.keys()) == {'ichimoku_4H', 'adx_15M'}

    def test_overwrite_existing_key(self):
        first = EvaluatorResult(direction=1.0, confidence=0.5)
        second = EvaluatorResult(direction=-1.0, confidence=0.8)
        self.matrix.set('ichimoku_1H', first)
        self.matrix.set('ichimoku_1H', second)
        assert self.matrix.get('ichimoku_1H') is second

    def test_repr_shows_keys(self):
        self.matrix.set('ichimoku_4H', self.result)
        assert 'ichimoku_4H' in repr(self.matrix)


# ---------------------------------------------------------------------------
# 4. EvaluatorResult
# ---------------------------------------------------------------------------

class TestEvaluatorResult:
    def test_basic_construction(self):
        r = EvaluatorResult(direction=0.5, confidence=0.7)
        assert r.direction == 0.5
        assert r.confidence == 0.7
        assert r.metadata == {}

    def test_metadata_populated(self):
        r = EvaluatorResult(direction=-1.0, confidence=1.0, metadata={'cloud': 'below'})
        assert r.metadata['cloud'] == 'below'

    def test_metadata_default_independent(self):
        """Each instance gets its own default dict."""
        a = EvaluatorResult(direction=0.0, confidence=0.0)
        b = EvaluatorResult(direction=0.0, confidence=0.0)
        a.metadata['x'] = 1
        assert 'x' not in b.metadata


# ---------------------------------------------------------------------------
# 5. EvalRequirement
# ---------------------------------------------------------------------------

class TestEvalRequirement:
    def test_basic_construction(self):
        req = EvalRequirement(evaluator_name='ichimoku', timeframes=['4H', '1H', '15M'])
        assert req.evaluator_name == 'ichimoku'
        assert req.timeframes == ['4H', '1H', '15M']

    def test_single_timeframe(self):
        req = EvalRequirement(evaluator_name='adx', timeframes=['15M'])
        assert len(req.timeframes) == 1


# ---------------------------------------------------------------------------
# 6. ConfluenceResult
# ---------------------------------------------------------------------------

class TestConfluenceResult:
    def test_basic_construction(self):
        cr = ConfluenceResult(score=7, quality_tier='A+')
        assert cr.score == 7
        assert cr.quality_tier == 'A+'
        assert cr.breakdown == {}

    def test_with_breakdown(self):
        bd = {'cloud_aligned': True, 'tk_cross': True}
        cr = ConfluenceResult(score=5, quality_tier='B', breakdown=bd)
        assert cr.breakdown['cloud_aligned'] is True

    def test_breakdown_default_independent(self):
        a = ConfluenceResult(score=0, quality_tier='no_trade')
        b = ConfluenceResult(score=0, quality_tier='no_trade')
        a.breakdown['x'] = 1
        assert 'x' not in b.breakdown

    def test_all_quality_tiers(self):
        for tier in ('A+', 'B', 'C', 'no_trade'):
            cr = ConfluenceResult(score=0, quality_tier=tier)
            assert cr.quality_tier == tier


# ---------------------------------------------------------------------------
# 7. ExitDecision
# ---------------------------------------------------------------------------

class TestExitDecision:
    def test_hold_defaults(self):
        ed = ExitDecision(action='hold')
        assert ed.action == 'hold'
        assert ed.new_stop is None
        assert ed.close_pct == 0.0
        assert ed.reason == ''

    def test_partial_exit(self):
        ed = ExitDecision(action='partial_exit', close_pct=0.5, reason='half-off at 1R')
        assert ed.action == 'partial_exit'
        assert ed.close_pct == 0.5
        assert 'half-off' in ed.reason

    def test_trail_update(self):
        ed = ExitDecision(action='trail_update', new_stop=1910.5, reason='trail to breakeven')
        assert ed.action == 'trail_update'
        assert ed.new_stop == pytest.approx(1910.5)

    def test_full_exit(self):
        ed = ExitDecision(action='full_exit', close_pct=1.0, reason='target hit')
        assert ed.action == 'full_exit'
        assert ed.close_pct == 1.0

    def test_all_valid_actions(self):
        for action in ('hold', 'partial_exit', 'trail_update', 'full_exit'):
            ed = ExitDecision(action=action)
            assert ed.action == action


# ---------------------------------------------------------------------------
# 8. TradingMode ABC
# ---------------------------------------------------------------------------

class TestTradingModeABC:
    def test_concrete_subclass_works(self):
        mode = _DummyTradingMode()
        result = mode.check_exit(trade=None, current_data={}, eval_results=EvalMatrix())
        assert isinstance(result, ExitDecision)
        assert result.action == 'hold'

    def test_abstract_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            TradingMode()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# 9. Strategy default method implementations
# ---------------------------------------------------------------------------

class TestStrategyDefaults:
    def setup_method(self):
        self.strategy = _DummyStrategy()

    def test_suggest_params_returns_empty_dict(self):
        result = self.strategy.suggest_params(trial=None)
        assert result == {}

    def test_populate_edge_context_returns_empty_dict(self):
        result = self.strategy.populate_edge_context(EvalMatrix())
        assert result == {}
