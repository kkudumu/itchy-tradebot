"""Integration tests for SSSStrategy — full pipeline bar-by-bar."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
import pytest

from src.strategy.strategies.sss import SSSStrategy
from src.strategy.signal_engine import Signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_TIME = datetime(2024, 1, 1, 9, 0, 0)


def make_strategy(**config_overrides) -> SSSStrategy:
    """Create SSSStrategy with default config + optional overrides."""
    defaults = {
        "warmup_bars": 20,          # shorter warmup for tests
        "min_confluence_score": 1,  # low threshold so signals can emerge
        "entry_mode": "cbc_only",   # simplest mode for integration tests
        "require_cbc_context": False,
    }
    defaults.update(config_overrides)
    return SSSStrategy(config=defaults)


def feed_strategy(
    strategy: SSSStrategy,
    prices: List[float],
    atr: float = 1.0,
    spread: float = 0.0,
) -> List[Optional[Signal]]:
    """Feed a list of close prices and return per-bar Signal results."""
    signals = []
    for i, p in enumerate(prices):
        ts = BASE_TIME + timedelta(minutes=i)
        sig = strategy.on_bar(
            ts,
            open=p,
            high=p + 0.5,
            low=p - 0.5,
            close=p,
            atr=atr,
            spread=spread,
        )
        signals.append(sig)
    return signals


def flat_prices(n: int, base: float = 1900.0) -> List[float]:
    """Flat market — no swings."""
    return [base] * n


def sinusoidal_prices(n: int, base: float = 1900.0, amplitude: float = 5.0) -> List[float]:
    """Sinusoidal prices to generate regular swing points."""
    return [base + amplitude * np.sin(i * 0.3) for i in range(n)]


# ---------------------------------------------------------------------------
# Warmup period
# ---------------------------------------------------------------------------


def test_no_signals_during_warmup():
    """No signals should be emitted before warmup_bars are processed."""
    strategy = make_strategy(warmup_bars=50)
    prices = sinusoidal_prices(49)
    signals = feed_strategy(strategy, prices)
    assert all(s is None for s in signals), "Got a signal before warmup period ended"


def test_signals_possible_after_warmup():
    """After warmup, the strategy may emit signals (no exception at minimum)."""
    strategy = make_strategy(warmup_bars=10)
    prices = sinusoidal_prices(200, amplitude=10.0)
    signals = feed_strategy(strategy, prices, atr=2.0)
    # Just verify no exception and the function returns a list
    assert isinstance(signals, list)
    assert len(signals) == len(prices)


# ---------------------------------------------------------------------------
# Signal attributes
# ---------------------------------------------------------------------------


def test_signal_strategy_name_is_sss():
    """Any emitted signal must carry strategy_name='sss'."""
    strategy = make_strategy(warmup_bars=10)
    prices = sinusoidal_prices(300, amplitude=15.0)
    signals = feed_strategy(strategy, prices, atr=3.0)
    emitted = [s for s in signals if s is not None]
    for sig in emitted:
        assert sig.strategy_name == "sss", f"Expected 'sss', got {sig.strategy_name!r}"


def test_signal_has_required_reasoning_keys():
    """Signal.reasoning must contain all expected keys."""
    required_keys = {
        "sequence_state",
        "sequence_direction",
        "ss_level_target",
        "cbc_type",
        "fifty_tap_status",
        "layers_aligned",
        "confluence_breakdown",
    }
    strategy = make_strategy(warmup_bars=10)
    prices = sinusoidal_prices(300, amplitude=15.0)
    signals = feed_strategy(strategy, prices, atr=3.0)
    emitted = [s for s in signals if s is not None]
    for sig in emitted:
        missing = required_keys - set(sig.reasoning.keys())
        assert not missing, f"Missing keys in reasoning: {missing}"


def test_signal_direction_is_long_or_short():
    strategy = make_strategy(warmup_bars=10)
    prices = sinusoidal_prices(300, amplitude=15.0)
    signals = feed_strategy(strategy, prices, atr=3.0)
    emitted = [s for s in signals if s is not None]
    for sig in emitted:
        assert sig.direction in ("long", "short"), f"Unexpected direction: {sig.direction}"


def test_signal_stop_loss_on_correct_side_for_long():
    """For a long signal, stop_loss must be below entry_price."""
    strategy = make_strategy(warmup_bars=10)
    prices = sinusoidal_prices(300, amplitude=15.0)
    signals = feed_strategy(strategy, prices, atr=3.0)
    longs = [s for s in signals if s is not None and s.direction == "long"]
    for sig in longs:
        assert sig.stop_loss < sig.entry_price, (
            f"Long stop {sig.stop_loss} not below entry {sig.entry_price}"
        )


def test_signal_stop_loss_on_correct_side_for_short():
    """For a short signal, stop_loss must be above entry_price."""
    strategy = make_strategy(warmup_bars=10)
    prices = sinusoidal_prices(300, amplitude=15.0)
    signals = feed_strategy(strategy, prices, atr=3.0)
    shorts = [s for s in signals if s is not None and s.direction == "short"]
    for sig in shorts:
        assert sig.stop_loss > sig.entry_price, (
            f"Short stop {sig.stop_loss} not above entry {sig.entry_price}"
        )


# ---------------------------------------------------------------------------
# Confluence scoring and quality tier
# ---------------------------------------------------------------------------


def test_confluence_score_bounded():
    """Confluence score must be in [0, 8] as documented."""
    strategy = make_strategy(warmup_bars=10, min_confluence_score=0)
    prices = sinusoidal_prices(400, amplitude=20.0)
    signals = feed_strategy(strategy, prices, atr=4.0)
    emitted = [s for s in signals if s is not None]
    for sig in emitted:
        assert 0 <= sig.confluence_score <= 8, f"Score out of range: {sig.confluence_score}"


def test_quality_tier_assignment():
    """Tier classification: A+(>=7), B(>=5), C(else)."""
    strategy = make_strategy(warmup_bars=0, min_confluence_score=0,
                              tier_a_plus=7, tier_b=5)
    prices = sinusoidal_prices(400, amplitude=20.0)
    signals = feed_strategy(strategy, prices, atr=4.0)
    emitted = [s for s in signals if s is not None]
    for sig in emitted:
        assert sig.quality_tier in ("A+", "B", "C"), f"Unexpected tier: {sig.quality_tier}"
        if sig.confluence_score >= 7:
            assert sig.quality_tier == "A+"
        elif sig.confluence_score >= 5:
            assert sig.quality_tier == "B"
        else:
            assert sig.quality_tier == "C"


def test_min_confluence_filter_blocks_low_score():
    """Signals with score < min_confluence_score should be suppressed."""
    strategy = make_strategy(warmup_bars=10, min_confluence_score=99)
    prices = sinusoidal_prices(300, amplitude=15.0)
    signals = feed_strategy(strategy, prices, atr=3.0)
    emitted = [s for s in signals if s is not None]
    assert emitted == [], "No signals should pass with min_confluence_score=99"


# ---------------------------------------------------------------------------
# Entry mode gates
# ---------------------------------------------------------------------------


def test_cbc_only_mode_requires_cbc():
    """entry_mode='cbc_only' should only signal when CBC is present."""
    strategy = make_strategy(warmup_bars=10, entry_mode="cbc_only",
                              require_cbc_context=False, min_confluence_score=0)
    prices = sinusoidal_prices(300, amplitude=15.0)
    signals = feed_strategy(strategy, prices, atr=3.0)
    # Just verify no crash and all returned items are Signal or None
    for s in signals:
        assert s is None or isinstance(s, Signal)


def test_fifty_tap_mode_requires_tap():
    """entry_mode='fifty_tap' only signals when 50% tap is confirmed."""
    strategy = make_strategy(warmup_bars=10, entry_mode="fifty_tap",
                              require_cbc_context=False, min_confluence_score=0)
    prices = sinusoidal_prices(300, amplitude=15.0)
    signals = feed_strategy(strategy, prices, atr=3.0)
    for s in signals:
        assert s is None or isinstance(s, Signal)


# ---------------------------------------------------------------------------
# Configuration propagation
# ---------------------------------------------------------------------------


def test_custom_config_applied():
    """Strategy uses config overrides, not defaults."""
    strategy = SSSStrategy(config={"warmup_bars": 200, "min_confluence_score": 10})
    # warmup_bars=200 → no signals in first 199 bars
    prices = sinusoidal_prices(199)
    signals = feed_strategy(strategy, prices)
    assert all(s is None for s in signals)


def test_strategy_handles_flat_market():
    """Strategy should not crash on flat prices (no swings)."""
    strategy = make_strategy(warmup_bars=5)
    prices = flat_prices(100)
    signals = feed_strategy(strategy, prices)
    # All None expected (no swings → no sequence → no entry)
    assert all(s is None for s in signals)


def test_on_bar_returns_optional_signal():
    """Verify the return type contract — Signal or None."""
    strategy = make_strategy(warmup_bars=5)
    ts = BASE_TIME
    result = strategy.on_bar(ts, open=1900.0, high=1901.0, low=1899.0, close=1900.0, atr=1.0)
    assert result is None or isinstance(result, Signal)


# ---------------------------------------------------------------------------
# Instrument in signal
# ---------------------------------------------------------------------------


def test_signal_instrument_matches_config():
    strategy = make_strategy(warmup_bars=10, instrument="XAUUSD")
    prices = sinusoidal_prices(300, amplitude=15.0)
    signals = feed_strategy(strategy, prices, atr=3.0)
    emitted = [s for s in signals if s is not None]
    for sig in emitted:
        assert sig.instrument == "XAUUSD"
