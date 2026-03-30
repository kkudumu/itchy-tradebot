"""Tests for Asian Range Breakout strategy."""
import datetime as dt
from src.strategy.strategies.asian_breakout import AsianBreakoutStrategy


def _utc(h, m=0):
    return dt.datetime(2024, 3, 15, h, m, tzinfo=dt.timezone.utc)


class TestAsianRangeDetection:
    def test_marks_asian_range_high_low(self):
        strategy = AsianBreakoutStrategy()
        strategy.on_bar(_utc(21), high=2050, low=2040, close=2045)
        strategy.on_bar(_utc(23), high=2055, low=2038, close=2050)
        strategy.on_bar(_utc(2), high=2052, low=2042, close=2048)
        strategy.on_bar(_utc(5), high=2053, low=2039, close=2046)
        assert strategy.asian_high == 2055.0
        assert strategy.asian_low == 2038.0

    def test_rejects_range_too_narrow(self):
        strategy = AsianBreakoutStrategy(config={"min_range_pips": 20})
        strategy.on_bar(_utc(21), high=2050, low=2049, close=2049.5)
        strategy.on_bar(_utc(5), high=2050.5, low=2048.5, close=2049)
        assert strategy.range_valid is False

    def test_rejects_range_too_wide(self):
        strategy = AsianBreakoutStrategy(config={"max_range_pips": 80})
        strategy.on_bar(_utc(21), high=2080, low=2040, close=2060)
        strategy.on_bar(_utc(5), high=2085, low=2035, close=2060)
        assert strategy.range_valid is False


class TestLondonBreakoutSignal:
    def test_long_signal_on_break_above_asian_high(self):
        strategy = AsianBreakoutStrategy(config={"min_range_pips": 20, "max_range_pips": 500, "rr_ratio": 2.0})
        strategy.on_bar(_utc(21), high=2050, low=2040, close=2045)
        strategy.on_bar(_utc(5, 59), high=2050, low=2040, close=2045)
        signal = strategy.on_bar(_utc(6, 5), high=2052, low=2044, close=2051)
        assert signal is not None
        assert signal.direction == "long"
        assert signal.entry_price == 2051.0
        assert signal.stop_loss == 2040.0
        assert signal.take_profit > signal.entry_price

    def test_short_signal_on_break_below_asian_low(self):
        strategy = AsianBreakoutStrategy(config={"min_range_pips": 20, "max_range_pips": 500, "rr_ratio": 2.0})
        strategy.on_bar(_utc(21), high=2050, low=2040, close=2045)
        strategy.on_bar(_utc(5, 59), high=2050, low=2040, close=2045)
        signal = strategy.on_bar(_utc(6, 5), high=2042, low=2038, close=2039)
        assert signal is not None
        assert signal.direction == "short"
        assert signal.stop_loss == 2050.0

    def test_no_signal_outside_london_window(self):
        strategy = AsianBreakoutStrategy()
        strategy.on_bar(_utc(21), high=2050, low=2040, close=2045)
        strategy.on_bar(_utc(5, 59), high=2050, low=2040, close=2045)
        signal = strategy.on_bar(_utc(11), high=2052, low=2044, close=2051)
        assert signal is None

    def test_only_one_signal_per_day(self):
        strategy = AsianBreakoutStrategy(config={"min_range_pips": 20, "max_range_pips": 500})
        strategy.on_bar(_utc(21), high=2050, low=2040, close=2045)
        strategy.on_bar(_utc(5, 59), high=2050, low=2040, close=2045)
        sig1 = strategy.on_bar(_utc(6, 5), high=2052, low=2044, close=2051)
        assert sig1 is not None
        sig2 = strategy.on_bar(_utc(7), high=2055, low=2050, close=2054)
        assert sig2 is None
