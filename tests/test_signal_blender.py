"""Tests for multi-strategy signal blender."""
import datetime as dt
from src.strategy.signal_blender import SignalBlender
from src.strategy.signal_engine import Signal


def _make_signal(direction="long", score=5, strategy_name="test", ts=None):
    return Signal(
        timestamp=ts or dt.datetime(2024, 1, 1, 8, 0, tzinfo=dt.timezone.utc),
        instrument="XAUUSD",
        direction=direction,
        entry_price=2050.0,
        stop_loss=2040.0,
        take_profit=2070.0,
        confluence_score=score,
        quality_tier="B",
        atr=5.0,
        reasoning={"strategy": strategy_name},
    )


class TestBlenderSelection:
    def test_returns_none_when_no_signals(self):
        blender = SignalBlender()
        result = blender.select([])
        assert result is None

    def test_returns_single_signal(self):
        blender = SignalBlender()
        sig = _make_signal(score=5)
        result = blender.select([sig])
        assert result is sig

    def test_picks_highest_confluence(self):
        blender = SignalBlender()
        low = _make_signal(score=3, strategy_name="ichi")
        high = _make_signal(score=7, strategy_name="asian")
        result = blender.select([low, high])
        assert result is high

    def test_multi_agree_bonus(self):
        """When 2+ strategies agree on direction, add bonus to score."""
        blender = SignalBlender(multi_agree_bonus=2)
        sig_a = _make_signal(direction="long", score=4, strategy_name="ichi")
        sig_b = _make_signal(direction="long", score=3, strategy_name="asian")
        sig_c = _make_signal(direction="short", score=5, strategy_name="ema")
        # sig_a and sig_b both long -> each gets +2 bonus
        # sig_a effective=6, sig_b effective=5, sig_c=5
        result = blender.select([sig_a, sig_b, sig_c])
        assert result.reasoning["strategy"] == "ichi"

    def test_conflicting_directions_picks_highest(self):
        blender = SignalBlender()
        long_sig = _make_signal(direction="long", score=6)
        short_sig = _make_signal(direction="short", score=4)
        result = blender.select([long_sig, short_sig])
        assert result.direction == "long"

    def test_tie_breaks_by_input_order(self):
        blender = SignalBlender()
        first = _make_signal(score=5, strategy_name="first")
        second = _make_signal(score=5, strategy_name="second")
        result = blender.select([first, second])
        assert result.reasoning["strategy"] == "first"
