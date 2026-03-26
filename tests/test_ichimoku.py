"""Comprehensive unit tests for the Ichimoku Indicator Engine.

Synthetic datasets are constructed so that expected values can be
calculated by hand or via simple formulas, then compared against the
vectorized implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.indicators.ichimoku import IchimokuCalculator, IchimokuResult
from src.indicators.signals import IchimokuSignalState, IchimokuSignals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _midpoint_manual(high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
    """Reference (loop-based) midpoint calculation for test comparison."""
    n = len(high)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        h = np.max(high[i - period + 1 : i + 1])
        l = np.min(low[i - period + 1 : i + 1])
        result[i] = (h + l) / 2.0
    return result


def _make_flat_ohlc(n: int = 100, price: float = 1800.0):
    """All bars at the same price — flat market."""
    high = np.full(n, price + 1.0)
    low = np.full(n, price - 1.0)
    close = np.full(n, price)
    return high, low, close


def _make_trending_ohlc(n: int = 100, start: float = 1700.0, step: float = 1.0):
    """Linearly trending upward OHLC data."""
    close = np.arange(n, dtype=float) * step + start
    high = close + 0.5
    low = close - 0.5
    return high, low, close


def _make_known_ohlc():
    """Small synthetic dataset (60 bars) with predictable behaviour.

    Bars 0-29: flat at 1800 (high=1801, low=1799)
    Bars 30-59: flat at 1900 (high=1901, low=1899)

    This creates a clean step-up where:
    - Tenkan (9): reflects 1800 until bar 37 (9 bars into new level), then 1900
    - Kijun (26): reflects 1800 until bar 54, then fully transitions to 1900
    - Senkou A / B are displaced 26 bars forward
    - Chikou is displaced 26 bars backward (NaN at tail)
    """
    n = 60
    high = np.where(np.arange(n) < 30, 1801.0, 1901.0)
    low = np.where(np.arange(n) < 30, 1799.0, 1899.0)
    close = np.where(np.arange(n) < 30, 1800.0, 1900.0)
    return high.astype(float), low.astype(float), close.astype(float)


# ---------------------------------------------------------------------------
# IchimokuCalculator — component calculation
# ---------------------------------------------------------------------------

class TestIchimokuCalculatorInit:
    def test_default_periods(self):
        calc = IchimokuCalculator()
        assert calc.tenkan_period == 9
        assert calc.kijun_period == 26
        assert calc.senkou_b_period == 52

    def test_custom_periods(self):
        calc = IchimokuCalculator(tenkan_period=7, kijun_period=22, senkou_b_period=44)
        assert calc.tenkan_period == 7
        assert calc.kijun_period == 22
        assert calc.senkou_b_period == 44

    def test_invalid_period_raises(self):
        with pytest.raises(ValueError):
            IchimokuCalculator(tenkan_period=0)
        with pytest.raises(ValueError):
            IchimokuCalculator(kijun_period=-1)
        with pytest.raises(ValueError):
            IchimokuCalculator(senkou_b_period=0)


class TestIchimokuCalculatorCalculate:
    def setup_method(self):
        self.calc = IchimokuCalculator()
        self.high, self.low, self.close = _make_trending_ohlc(n=100)
        self.result = self.calc.calculate(self.high, self.low, self.close)

    def test_returns_ichimoku_result(self):
        assert isinstance(self.result, IchimokuResult)

    def test_output_shapes_match_input(self):
        n = len(self.close)
        assert self.result.tenkan_sen.shape == (n,)
        assert self.result.kijun_sen.shape == (n,)
        assert self.result.senkou_a.shape == (n,)
        assert self.result.senkou_b.shape == (n,)
        assert self.result.chikou_span.shape == (n,)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            self.calc.calculate(self.high[:50], self.low, self.close)

    def test_tenkan_nan_in_lookback_period(self):
        # First tenkan_period-1 values must be NaN
        assert np.all(np.isnan(self.result.tenkan_sen[: self.calc.tenkan_period - 1]))

    def test_kijun_nan_in_lookback_period(self):
        assert np.all(np.isnan(self.result.kijun_sen[: self.calc.kijun_period - 1]))

    def test_tenkan_matches_manual(self):
        expected = _midpoint_manual(self.high, self.low, self.calc.tenkan_period)
        valid = ~np.isnan(expected)
        np.testing.assert_allclose(self.result.tenkan_sen[valid], expected[valid], rtol=1e-10)

    def test_kijun_matches_manual(self):
        expected = _midpoint_manual(self.high, self.low, self.calc.kijun_period)
        valid = ~np.isnan(expected)
        np.testing.assert_allclose(self.result.kijun_sen[valid], expected[valid], rtol=1e-10)

    def test_senkou_a_is_displaced_forward(self):
        # senkou_a[i] must be NaN for i < kijun_period (displacement zeros out leading)
        assert np.all(np.isnan(self.result.senkou_a[: self.calc.kijun_period]))

    def test_senkou_b_is_displaced_forward(self):
        # senkou_b[i] must be NaN for i < kijun_period
        assert np.all(np.isnan(self.result.senkou_b[: self.calc.kijun_period]))

    def test_chikou_nan_at_tail(self):
        # Last kijun_period values of chikou must be NaN (no future data)
        assert np.all(np.isnan(self.result.chikou_span[-self.calc.kijun_period :]))

    def test_chikou_equals_close_shifted(self):
        # chikou[i] should equal close[i + kijun_period]
        k = self.calc.kijun_period
        n = len(self.close)
        for i in range(n - k):
            assert self.result.chikou_span[i] == pytest.approx(self.close[i + k])

    def test_known_tenkan_value(self):
        """Manual check for a known-result dataset."""
        high, low, close = _make_known_ohlc()
        result = self.calc.calculate(high, low, close)
        # Bar 8 (index 8): first valid tenkan — all 9 bars at price 1800
        # rolling high=1801, rolling low=1799 → midpoint=1800
        assert result.tenkan_sen[8] == pytest.approx(1800.0)
        # Bar 38 (index 38): 9 bars fully in new level (bars 30-38)
        assert result.tenkan_sen[38] == pytest.approx(1900.0)

    def test_known_kijun_value(self):
        high, low, close = _make_known_ohlc()
        result = self.calc.calculate(high, low, close)
        # Bar 25 (index 25): first valid kijun — bars 0-25 all at 1800
        assert result.kijun_sen[25] == pytest.approx(1800.0)
        # Bar 55 (index 55): 26 bars fully in new level (bars 30-55)
        assert result.kijun_sen[55] == pytest.approx(1900.0)

    def test_senkou_a_value_at_known_position(self):
        high, low, close = _make_known_ohlc()
        result = self.calc.calculate(high, low, close)
        # At bar 25 (raw): tenkan=1800, kijun=1800 → senkou_a_raw=1800
        # Displaced +26 → appears at bar 51
        assert result.senkou_a[51] == pytest.approx(1800.0)

    def test_accepts_list_inputs(self):
        h = list(self.high)
        l = list(self.low)
        c = list(self.close)
        result = self.calc.calculate(h, l, c)
        assert isinstance(result.tenkan_sen, np.ndarray)


class TestIchimokuCloudHelpers:
    def setup_method(self):
        self.calc = IchimokuCalculator()

    def test_cloud_thickness_basic(self):
        a = np.array([100.0, 200.0, np.nan])
        b = np.array([80.0, 220.0, 100.0])
        result = self.calc.cloud_thickness(a, b)
        assert result[0] == pytest.approx(20.0)
        assert result[1] == pytest.approx(20.0)
        assert np.isnan(result[2])

    def test_cloud_direction_bullish(self):
        a = np.array([110.0, 90.0, np.nan])
        b = np.array([100.0, 100.0, 100.0])
        result = self.calc.cloud_direction(a, b)
        assert result[0] == 1    # A > B
        assert result[1] == -1   # A < B
        assert result[2] == 0    # NaN → neutral

    def test_cloud_direction_equal(self):
        a = np.array([100.0])
        b = np.array([100.0])
        result = self.calc.cloud_direction(a, b)
        assert result[0] == 0


# ---------------------------------------------------------------------------
# IchimokuSignals
# ---------------------------------------------------------------------------

class TestTKCross:
    def setup_method(self):
        self.sig = IchimokuSignals()

    def test_bullish_cross(self):
        # tenkan crosses above kijun at bar 2
        tenkan = np.array([90.0, 99.0, 101.0, 103.0])
        kijun = np.array([100.0, 100.0, 100.0, 100.0])
        result = self.sig.tk_cross(tenkan, kijun)
        assert result[2] == 1   # cross happened here
        assert result[0] == 0
        assert result[1] == 0
        assert result[3] == 0

    def test_bearish_cross(self):
        # tenkan crosses below kijun at bar 2
        tenkan = np.array([110.0, 101.0, 99.0, 97.0])
        kijun = np.array([100.0, 100.0, 100.0, 100.0])
        result = self.sig.tk_cross(tenkan, kijun)
        assert result[2] == -1
        assert result[0] == 0

    def test_no_cross_when_always_above(self):
        tenkan = np.array([110.0, 111.0, 112.0])
        kijun = np.array([100.0, 100.0, 100.0])
        result = self.sig.tk_cross(tenkan, kijun)
        assert np.all(result == 0)

    def test_no_cross_when_always_below(self):
        tenkan = np.array([90.0, 89.0, 88.0])
        kijun = np.array([100.0, 100.0, 100.0])
        result = self.sig.tk_cross(tenkan, kijun)
        assert np.all(result == 0)

    def test_nan_input_produces_zero(self):
        tenkan = np.array([np.nan, 101.0])
        kijun = np.array([100.0, 100.0])
        result = self.sig.tk_cross(tenkan, kijun)
        assert result[1] == 0  # previous bar was NaN → no valid cross

    def test_short_array(self):
        result = self.sig.tk_cross(np.array([100.0]), np.array([100.0]))
        assert len(result) == 1
        assert result[0] == 0

    def test_cross_from_equality(self):
        # touching (equal) then crossing up counts as bullish
        tenkan = np.array([100.0, 101.0])
        kijun = np.array([100.0, 100.0])
        result = self.sig.tk_cross(tenkan, kijun)
        assert result[1] == 1

    def test_cross_at_every_bar(self):
        # Oscillating tenkan produces alternating crosses
        tenkan = np.array([90.0, 110.0, 90.0, 110.0])
        kijun = np.array([100.0, 100.0, 100.0, 100.0])
        result = self.sig.tk_cross(tenkan, kijun)
        assert result[1] == 1
        assert result[2] == -1
        assert result[3] == 1


class TestCloudPosition:
    def setup_method(self):
        self.sig = IchimokuSignals()

    def test_above_cloud(self):
        close = np.array([200.0])
        a = np.array([150.0])
        b = np.array([140.0])
        result = self.sig.cloud_position(close, a, b)
        assert result[0] == 1

    def test_below_cloud(self):
        close = np.array([100.0])
        a = np.array([150.0])
        b = np.array([140.0])
        result = self.sig.cloud_position(close, a, b)
        assert result[0] == -1

    def test_inside_cloud(self):
        close = np.array([145.0])
        a = np.array([150.0])
        b = np.array([140.0])
        result = self.sig.cloud_position(close, a, b)
        assert result[0] == 0

    def test_on_cloud_top_boundary(self):
        # Exactly at cloud top → inside (not strictly above)
        close = np.array([150.0])
        a = np.array([150.0])
        b = np.array([140.0])
        result = self.sig.cloud_position(close, a, b)
        assert result[0] == 0

    def test_on_cloud_bottom_boundary(self):
        close = np.array([140.0])
        a = np.array([150.0])
        b = np.array([140.0])
        result = self.sig.cloud_position(close, a, b)
        assert result[0] == 0

    def test_nan_senkou_gives_zero(self):
        close = np.array([200.0])
        a = np.array([np.nan])
        b = np.array([140.0])
        result = self.sig.cloud_position(close, a, b)
        assert result[0] == 0

    def test_inverted_cloud_above(self):
        # Bearish cloud: B > A; price still above both
        close = np.array([200.0])
        a = np.array([140.0])   # A < B in bearish cloud
        b = np.array([150.0])
        result = self.sig.cloud_position(close, a, b)
        assert result[0] == 1

    def test_mixed_array(self):
        close = np.array([200.0, 145.0, 100.0])
        a = np.array([150.0, 150.0, 150.0])
        b = np.array([140.0, 140.0, 140.0])
        result = self.sig.cloud_position(close, a, b)
        assert result[0] == 1
        assert result[1] == 0
        assert result[2] == -1


class TestChikouConfirmation:
    def setup_method(self):
        self.sig = IchimokuSignals()

    def test_bullish_confirmation(self):
        chikou = np.array([1900.0])
        close = np.array([1800.0])
        result = self.sig.chikou_confirmation(chikou, close)
        assert result[0] == 1

    def test_bearish_confirmation(self):
        chikou = np.array([1700.0])
        close = np.array([1800.0])
        result = self.sig.chikou_confirmation(chikou, close)
        assert result[0] == -1

    def test_neutral_equal(self):
        chikou = np.array([1800.0])
        close = np.array([1800.0])
        result = self.sig.chikou_confirmation(chikou, close)
        assert result[0] == 0

    def test_nan_chikou_gives_zero(self):
        chikou = np.array([np.nan])
        close = np.array([1800.0])
        result = self.sig.chikou_confirmation(chikou, close)
        assert result[0] == 0

    def test_integrated_with_calculator(self):
        """chikou[i] = close[i+26]; compare with close[i] → confirms up-trend."""
        high, low, close = _make_trending_ohlc(n=100, step=2.0)
        calc = IchimokuCalculator()
        result = calc.calculate(high, low, close)
        sig = IchimokuSignals()
        conf = sig.chikou_confirmation(result.chikou_span, close)
        # In an uptrend close[i+26] > close[i] → all valid bars should be 1
        valid = ~np.isnan(result.chikou_span)
        assert np.all(conf[valid] == 1)


class TestCloudTwist:
    def setup_method(self):
        self.sig = IchimokuSignals()

    def test_bullish_twist(self):
        # A crosses above B at bar 2
        a = np.array([90.0, 99.0, 101.0, 103.0])
        b = np.array([100.0, 100.0, 100.0, 100.0])
        result = self.sig.cloud_twist(a, b)
        assert result[2] == 1

    def test_bearish_twist(self):
        # A crosses below B at bar 2
        a = np.array([110.0, 101.0, 99.0, 97.0])
        b = np.array([100.0, 100.0, 100.0, 100.0])
        result = self.sig.cloud_twist(a, b)
        assert result[2] == -1

    def test_no_twist(self):
        a = np.array([110.0, 111.0, 112.0])
        b = np.array([100.0, 100.0, 100.0])
        result = self.sig.cloud_twist(a, b)
        assert np.all(result == 0)

    def test_nan_gives_zero(self):
        a = np.array([np.nan, 101.0])
        b = np.array([100.0, 100.0])
        result = self.sig.cloud_twist(a, b)
        assert result[1] == 0

    def test_short_array(self):
        result = self.sig.cloud_twist(np.array([100.0]), np.array([100.0]))
        assert len(result) == 1
        assert result[0] == 0


class TestCloudBreakout:
    def setup_method(self):
        self.sig = IchimokuSignals()

    def test_bullish_breakout(self):
        # bar 0 inside cloud, bar 1 above cloud
        close = np.array([145.0, 200.0])
        a = np.array([150.0, 150.0])
        b = np.array([140.0, 140.0])
        result = self.sig.cloud_breakout(close, a, b)
        assert result[1] == 1

    def test_bearish_breakout(self):
        close = np.array([145.0, 100.0])
        a = np.array([150.0, 150.0])
        b = np.array([140.0, 140.0])
        result = self.sig.cloud_breakout(close, a, b)
        assert result[1] == -1

    def test_no_breakout_stays_inside(self):
        close = np.array([145.0, 146.0])
        a = np.array([150.0, 150.0])
        b = np.array([140.0, 140.0])
        result = self.sig.cloud_breakout(close, a, b)
        assert np.all(result == 0)

    def test_no_breakout_stays_above(self):
        close = np.array([200.0, 201.0])
        a = np.array([150.0, 150.0])
        b = np.array([140.0, 140.0])
        result = self.sig.cloud_breakout(close, a, b)
        assert np.all(result == 0)

    def test_breakout_from_below_to_inside_is_not_bullish(self):
        # crosses from below into inside — not a full bullish breakout
        close = np.array([100.0, 145.0])
        a = np.array([150.0, 150.0])
        b = np.array([140.0, 140.0])
        result = self.sig.cloud_breakout(close, a, b)
        assert result[1] == 0

    def test_nan_cloud_produces_zero(self):
        close = np.array([145.0, 200.0])
        a = np.array([np.nan, np.nan])
        b = np.array([140.0, 140.0])
        result = self.sig.cloud_breakout(close, a, b)
        assert np.all(result == 0)

    def test_short_array(self):
        result = self.sig.cloud_breakout(
            np.array([100.0]), np.array([100.0]), np.array([100.0])
        )
        assert len(result) == 1
        assert result[0] == 0


# ---------------------------------------------------------------------------
# Full integration: IchimokuCalculator + IchimokuSignals
# ---------------------------------------------------------------------------

class TestFullIntegration:
    def setup_method(self):
        self.calc = IchimokuCalculator()
        self.sig = IchimokuSignals()
        self.high, self.low, self.close = _make_trending_ohlc(n=100, step=1.0)
        self.result = self.calc.calculate(self.high, self.low, self.close)

    def test_cloud_direction_bullish_in_uptrend(self):
        """In a steady uptrend Senkou A should exceed Senkou B."""
        direction = self.calc.cloud_direction(self.result.senkou_a, self.result.senkou_b)
        valid = ~(np.isnan(self.result.senkou_a) | np.isnan(self.result.senkou_b))
        # Most valid bars should be bullish (A > B) in an uptrend
        assert np.sum(direction[valid] == 1) > np.sum(direction[valid] == -1)

    def test_cloud_position_above_in_uptrend(self):
        """Price should be above the cloud in a steady uptrend for most bars."""
        pos = self.sig.cloud_position(self.close, self.result.senkou_a, self.result.senkou_b)
        valid = ~(np.isnan(self.result.senkou_a) | np.isnan(self.result.senkou_b))
        above_count = np.sum(pos[valid] == 1)
        total_valid = np.sum(valid)
        assert above_count / total_valid > 0.5

    def test_output_dtypes(self):
        tk = self.sig.tk_cross(self.result.tenkan_sen, self.result.kijun_sen)
        assert tk.dtype == np.int8

    def test_signal_state_at_returns_dataclass(self):
        idx = 80  # Well within data range
        state = self.sig.signal_state_at(
            idx,
            self.result.tenkan_sen,
            self.result.kijun_sen,
            self.close,
            self.result.senkou_a,
            self.result.senkou_b,
            self.result.chikou_span,
        )
        assert isinstance(state, IchimokuSignalState)
        assert state.cloud_direction in (-1, 0, 1)
        assert state.tk_cross in (-1, 0, 1)
        assert state.cloud_position in (-1, 0, 1)
        assert state.chikou_confirmed in (-1, 0, 1)
        assert state.cloud_twist in (-1, 0, 1)
        assert isinstance(state.cloud_thickness, float)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def setup_method(self):
        self.calc = IchimokuCalculator()
        self.sig = IchimokuSignals()

    def test_minimum_length_array(self):
        """Array shorter than any lookback period — everything is NaN."""
        high = np.array([1800.0, 1801.0, 1799.0])
        low = np.array([1795.0, 1796.0, 1794.0])
        close = np.array([1797.0, 1798.0, 1796.0])
        result = self.calc.calculate(high, low, close)
        assert np.all(np.isnan(result.tenkan_sen))
        assert np.all(np.isnan(result.kijun_sen))

    def test_flat_market_tenkan_equals_kijun_approx(self):
        """Flat market with equal high/low spread → tenkan == kijun for shared period."""
        high, low, close = _make_flat_ohlc(n=100, price=1800.0)
        result = self.calc.calculate(high, low, close)
        valid = ~np.isnan(result.kijun_sen)
        np.testing.assert_allclose(result.tenkan_sen[valid], 1800.0, atol=1e-10)
        np.testing.assert_allclose(result.kijun_sen[valid], 1800.0, atol=1e-10)

    def test_no_tk_cross_in_flat_market(self):
        high, low, close = _make_flat_ohlc(n=100)
        result = self.calc.calculate(high, low, close)
        crosses = self.sig.tk_cross(result.tenkan_sen, result.kijun_sen)
        assert np.all(crosses == 0)

    def test_all_nan_inputs(self):
        n = 10
        nans = np.full(n, np.nan)
        result = self.calc.calculate(nans, nans, nans)
        assert np.all(np.isnan(result.tenkan_sen))

    def test_single_element_signals(self):
        """Signal methods on length-1 arrays should not raise."""
        single = np.array([1800.0])
        result = self.sig.tk_cross(single, single)
        assert len(result) == 1

    def test_senkou_spans_displaced_correctly(self):
        """Verify kijun_period displacement in senkou spans."""
        high, low, close = _make_flat_ohlc(n=100)
        result = self.calc.calculate(high, low, close)
        k = self.calc.kijun_period
        # First kijun_period positions should be NaN
        assert np.all(np.isnan(result.senkou_a[:k]))
        assert np.all(np.isnan(result.senkou_b[:k]))
        # After kijun_period, values should be valid (for flat market all 1800)
        # Earliest valid senkou_a position is when raw senkou_a_raw had valid
        # values: raw requires at least tenkan_period and kijun_period bars.
        # For default 9/26/52, tenkan valid from bar 8, kijun from bar 25,
        # so raw senkou_a valid from bar 25; shifted to bar 51.
        assert not np.isnan(result.senkou_a[51])

    def test_cloud_thickness_zero_in_flat(self):
        high, low, close = _make_flat_ohlc(n=100)
        result = self.calc.calculate(high, low, close)
        thickness = self.calc.cloud_thickness(result.senkou_a, result.senkou_b)
        valid = ~np.isnan(thickness)
        if valid.any():
            np.testing.assert_allclose(thickness[valid], 0.0, atol=1e-10)

    def test_large_array_performance(self):
        """Ensure 10000 bars run without timeout (pure performance guard)."""
        import time
        high, low, close = _make_trending_ohlc(n=10_000, step=0.01)
        t0 = time.perf_counter()
        result = self.calc.calculate(high, low, close)
        elapsed = time.perf_counter() - t0
        # Should complete well within 2 seconds on any modern machine
        assert elapsed < 2.0
        assert result.tenkan_sen.shape == (10_000,)
