"""Unit tests for the evaluator implementations (Task 2).

Verifies that each evaluator:
- Returns an EvaluatorResult with valid fields
- Produces direction and confidence in the correct ranges
- Populates expected metadata keys
- Auto-registers in EVALUATOR_REGISTRY
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.base import EVALUATOR_REGISTRY, EvaluatorResult
# Import evaluators so they self-register
from src.strategy.evaluators import (
    IchimokuEvaluator,
    ADXEvaluator,
    ATREvaluator,
    SessionEvaluator,
)


# ---------------------------------------------------------------------------
# OHLCV fixture helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(
    n: int = 150,
    start_price: float = 1900.0,
    step: float = 1.0,
    session_hour: int = 10,   # 10:00 UTC → London session (active)
) -> pd.DataFrame:
    """Generate a simple upward-trending OHLCV DataFrame with a datetime index.

    Parameters
    ----------
    n:
        Number of 1-minute bars.
    start_price:
        Opening close price.
    step:
        Price increment per bar.
    session_hour:
        UTC hour for the generated timestamps (controls which session is active).

    Returns
    -------
    pd.DataFrame with columns: open, high, low, close, volume
    and a DatetimeIndex in UTC.
    """
    timestamps = pd.date_range(
        start=f'2024-01-15 {session_hour:02d}:00:00',
        periods=n,
        freq='1min',
        tz='UTC',
    )
    close = np.arange(n, dtype=float) * step + start_price
    high = close + 0.5
    low = close - 0.5
    open_ = close - step * 0.5
    volume = np.full(n, 1000.0)

    return pd.DataFrame(
        {'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume},
        index=timestamps,
    )


def _make_flat_ohlcv(n: int = 150, price: float = 1900.0) -> pd.DataFrame:
    """Generate flat OHLCV data — no trend."""
    timestamps = pd.date_range(
        start='2024-01-15 10:00:00',
        periods=n,
        freq='1min',
        tz='UTC',
    )
    close = np.full(n, price, dtype=float)
    high = close + 0.5
    low = close - 0.5
    open_ = close.copy()
    volume = np.full(n, 1000.0)

    return pd.DataFrame(
        {'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume},
        index=timestamps,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_evaluator_result_shape(result: EvaluatorResult) -> None:
    """Check that an EvaluatorResult has the correct types."""
    assert isinstance(result, EvaluatorResult)
    assert isinstance(result.direction, float)
    assert isinstance(result.confidence, float)
    assert isinstance(result.metadata, dict)


# ---------------------------------------------------------------------------
# EVALUATOR_REGISTRY tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_ichimoku_registered(self):
        assert 'ichimoku' in EVALUATOR_REGISTRY
        assert EVALUATOR_REGISTRY['ichimoku'] is IchimokuEvaluator

    def test_adx_registered(self):
        assert 'adx' in EVALUATOR_REGISTRY
        assert EVALUATOR_REGISTRY['adx'] is ADXEvaluator

    def test_atr_registered(self):
        assert 'atr' in EVALUATOR_REGISTRY
        assert EVALUATOR_REGISTRY['atr'] is ATREvaluator

    def test_session_registered(self):
        assert 'session' in EVALUATOR_REGISTRY
        assert EVALUATOR_REGISTRY['session'] is SessionEvaluator

    def test_registry_keys_have_name_attribute(self):
        for key, cls in EVALUATOR_REGISTRY.items():
            assert hasattr(cls, 'name'), f"{cls.__name__} missing .name attribute"
            assert cls.name == key


# ---------------------------------------------------------------------------
# IchimokuEvaluator
# ---------------------------------------------------------------------------

class TestIchimokuEvaluator:
    def setup_method(self):
        self.eval = IchimokuEvaluator()

    def test_returns_evaluator_result(self):
        ohlcv = _make_ohlcv(n=150)
        result = self.eval.evaluate(ohlcv)
        _assert_evaluator_result_shape(result)

    def test_direction_in_valid_set(self):
        ohlcv = _make_ohlcv(n=150)
        result = self.eval.evaluate(ohlcv)
        assert result.direction in (-1.0, 0.0, 1.0), (
            f"Expected direction in (-1, 0, 1), got {result.direction}"
        )

    def test_confidence_in_range(self):
        ohlcv = _make_ohlcv(n=150)
        result = self.eval.evaluate(ohlcv)
        assert 0.0 <= result.confidence <= 1.0, (
            f"Confidence out of range: {result.confidence}"
        )

    def test_metadata_keys_present(self):
        ohlcv = _make_ohlcv(n=150)
        result = self.eval.evaluate(ohlcv)
        expected_keys = {
            'tenkan', 'kijun', 'senkou_a', 'senkou_b', 'chikou',
            'cloud_thickness', 'cloud_direction', 'tk_cross',
            'cloud_position', 'chikou_confirmed',
        }
        assert expected_keys.issubset(set(result.metadata.keys())), (
            f"Missing metadata keys: {expected_keys - set(result.metadata.keys())}"
        )

    def test_flat_market_neutral_direction(self):
        """A perfectly flat market should produce a neutral cloud direction."""
        ohlcv = _make_flat_ohlcv(n=150)
        result = self.eval.evaluate(ohlcv)
        # Flat market: senkou_a == senkou_b → cloud_direction == 0
        assert result.direction == 0.0

    def test_trending_up_positive_direction(self):
        """A strong uptrend should eventually produce a bullish cloud."""
        # Use a large step to get a clear bullish cloud after warm-up
        ohlcv = _make_ohlcv(n=200, step=5.0)
        result = self.eval.evaluate(ohlcv)
        # With a strong uptrend, Senkou A > Senkou B → direction = +1
        assert result.direction == 1.0

    def test_confidence_zero_for_zero_direction_no_signals(self):
        """Flat market with no signals still adds thickness bonus only."""
        ohlcv = _make_flat_ohlcv(n=150)
        result = self.eval.evaluate(ohlcv)
        # Flat: cloud direction = 0, tk=0, cloud_pos=0, chikou=0, thickness=0
        # All dimensions stay 0 → confidence should be 0
        assert result.confidence == 0.0

    def test_minimum_bars_does_not_crash(self):
        """Evaluator should not crash even if data is sparse."""
        ohlcv = _make_ohlcv(n=60)  # Below warmup but should not raise
        # May return NaN in metadata, that is acceptable
        result = self.eval.evaluate(ohlcv)
        assert isinstance(result, EvaluatorResult)


# ---------------------------------------------------------------------------
# ADXEvaluator
# ---------------------------------------------------------------------------

class TestADXEvaluator:
    def setup_method(self):
        self.eval = ADXEvaluator()

    def test_returns_evaluator_result(self):
        ohlcv = _make_ohlcv(n=150)
        result = self.eval.evaluate(ohlcv)
        _assert_evaluator_result_shape(result)

    def test_direction_is_plus_or_minus_one(self):
        ohlcv = _make_ohlcv(n=150)
        result = self.eval.evaluate(ohlcv)
        assert result.direction in (-1.0, 1.0), (
            f"ADX direction must be ±1.0, got {result.direction}"
        )

    def test_confidence_in_range(self):
        ohlcv = _make_ohlcv(n=150)
        result = self.eval.evaluate(ohlcv)
        assert 0.0 <= result.confidence <= 1.0

    def test_metadata_has_adx_key(self):
        ohlcv = _make_ohlcv(n=150)
        result = self.eval.evaluate(ohlcv)
        assert 'adx' in result.metadata

    def test_metadata_has_all_expected_keys(self):
        ohlcv = _make_ohlcv(n=150)
        result = self.eval.evaluate(ohlcv)
        for key in ('adx', 'plus_di', 'minus_di', 'is_trending'):
            assert key in result.metadata, f"Missing metadata key: {key}"

    def test_uptrend_positive_direction(self):
        """Strong uptrend should give +DI > -DI → direction = +1."""
        ohlcv = _make_ohlcv(n=150, step=2.0)
        result = self.eval.evaluate(ohlcv)
        assert result.direction == 1.0

    def test_is_trending_is_bool(self):
        ohlcv = _make_ohlcv(n=150)
        result = self.eval.evaluate(ohlcv)
        assert isinstance(result.metadata['is_trending'], bool)

    def test_confidence_not_exceeds_one(self):
        """ADX / 100 should never exceed 1."""
        ohlcv = _make_ohlcv(n=150, step=10.0)  # High step → high ADX
        result = self.eval.evaluate(ohlcv)
        assert result.confidence <= 1.0


# ---------------------------------------------------------------------------
# ATREvaluator
# ---------------------------------------------------------------------------

class TestATREvaluator:
    def setup_method(self):
        self.eval = ATREvaluator()

    def test_returns_evaluator_result(self):
        ohlcv = _make_ohlcv(n=150)
        result = self.eval.evaluate(ohlcv)
        _assert_evaluator_result_shape(result)

    def test_direction_is_zero(self):
        ohlcv = _make_ohlcv(n=150)
        result = self.eval.evaluate(ohlcv)
        assert result.direction == 0.0, (
            f"ATR direction must always be 0.0, got {result.direction}"
        )

    def test_confidence_in_range(self):
        ohlcv = _make_ohlcv(n=150)
        result = self.eval.evaluate(ohlcv)
        assert 0.0 <= result.confidence <= 1.0

    def test_metadata_has_atr_key(self):
        ohlcv = _make_ohlcv(n=150)
        result = self.eval.evaluate(ohlcv)
        assert 'atr' in result.metadata

    def test_atr_value_is_float(self):
        ohlcv = _make_ohlcv(n=150)
        result = self.eval.evaluate(ohlcv)
        assert isinstance(result.metadata['atr'], float)

    def test_flat_market_fallback_confidence(self):
        """Flat market with no ATR range → confidence fallback = 0.5."""
        ohlcv = _make_flat_ohlcv(n=150)
        result = self.eval.evaluate(ohlcv)
        # Flat bars all have H-L = 1.0, so ATR will be 1.0 throughout.
        # rolling_max == rolling_min → spread = 0 → fallback 0.5
        assert result.confidence == 0.5

    def test_varying_volatility_full_range(self):
        """Bar at peak volatility should have confidence near 1.0."""
        # Build a series where the last bar has the highest ATR
        ohlcv = _make_ohlcv(n=150, step=1.0)
        # Increase final bars' high-low spread significantly
        ohlcv_high = ohlcv.copy()
        ohlcv_high.iloc[-1, ohlcv_high.columns.get_loc('high')] += 100.0
        ohlcv_high.iloc[-1, ohlcv_high.columns.get_loc('low')] -= 100.0
        result = self.eval.evaluate(ohlcv_high)
        # Last bar ATR should be close to the window max → confidence ~ 1.0
        assert result.confidence > 0.5


# ---------------------------------------------------------------------------
# SessionEvaluator
# ---------------------------------------------------------------------------

class TestSessionEvaluator:
    def setup_method(self):
        self.eval = SessionEvaluator()

    def test_returns_evaluator_result(self):
        ohlcv = _make_ohlcv(n=150, session_hour=10)  # London
        result = self.eval.evaluate(ohlcv)
        _assert_evaluator_result_shape(result)

    def test_direction_is_zero(self):
        ohlcv = _make_ohlcv(n=150, session_hour=10)
        result = self.eval.evaluate(ohlcv)
        assert result.direction == 0.0, (
            f"Session direction must always be 0.0, got {result.direction}"
        )

    def test_confidence_is_zero_or_one(self):
        ohlcv_active = _make_ohlcv(n=150, session_hour=10)   # London → active
        ohlcv_inactive = _make_ohlcv(n=150, session_hour=3)  # Asian → inactive
        result_active = self.eval.evaluate(ohlcv_active)
        result_inactive = self.eval.evaluate(ohlcv_inactive)
        assert result_active.confidence in (0.0, 1.0)
        assert result_inactive.confidence in (0.0, 1.0)

    def test_metadata_has_session_key(self):
        ohlcv = _make_ohlcv(n=150, session_hour=10)
        result = self.eval.evaluate(ohlcv)
        assert 'session' in result.metadata

    def test_metadata_has_is_active_key(self):
        ohlcv = _make_ohlcv(n=150, session_hour=10)
        result = self.eval.evaluate(ohlcv)
        assert 'is_active' in result.metadata
        assert isinstance(result.metadata['is_active'], bool)

    def test_london_session_active(self):
        """10:00 UTC falls in London session (08:00–16:00) → active."""
        ohlcv = _make_ohlcv(n=150, session_hour=10)
        result = self.eval.evaluate(ohlcv)
        assert result.confidence == 1.0
        assert result.metadata['is_active'] is True

    def test_off_hours_inactive(self):
        """03:00 UTC is in Asian session, but is_active_session is London+NY only."""
        # Asian session is 00:00–08:00; is_active_session = London | NY
        # 03:00 UTC → not London (starts at 08:00), not NY (starts at 13:00)
        ohlcv = _make_ohlcv(n=150, session_hour=3)
        result = self.eval.evaluate(ohlcv)
        assert result.confidence == 0.0
        assert result.metadata['is_active'] is False

    def test_session_label_is_string(self):
        ohlcv = _make_ohlcv(n=150, session_hour=10)
        result = self.eval.evaluate(ohlcv)
        assert isinstance(result.metadata['session'], str)

    def test_session_label_valid_values(self):
        """Session label must be one of the known session names."""
        valid_labels = {'asian', 'london', 'new_york', 'overlap', 'off_hours'}
        for hour in (3, 10, 14, 18, 22):
            ohlcv = _make_ohlcv(n=150, session_hour=hour)
            result = self.eval.evaluate(ohlcv)
            assert result.metadata['session'] in valid_labels, (
                f"Unexpected session label: {result.metadata['session']} "
                f"for hour={hour}"
            )

    def test_new_york_session_active(self):
        """15:00 UTC → NY session (13:00–21:00) → active."""
        ohlcv = _make_ohlcv(n=150, session_hour=15)
        result = self.eval.evaluate(ohlcv)
        assert result.confidence == 1.0
        assert result.metadata['is_active'] is True
