"""Tests for TradingMode implementations: KijunExitMode and DefaultExitMode.

Tests cover:
- KijunExitMode: full_exit when stop hit
- KijunExitMode: partial_exit at 2R
- KijunExitMode: trail_update with signal-TF kijun at 1.5R+
- KijunExitMode: no trail below 1.5R
- KijunExitMode: higher-TF kijun takes over at 3R+
- KijunExitMode: hold when nothing triggers
- DefaultExitMode: partial_exit at 2R
- DefaultExitMode: ATR trailing after partial
- Both: returns ExitDecision instances
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pytest

from src.strategy.base import EvalMatrix, EvaluatorResult, ExitDecision
from src.strategy.trading_modes import DefaultExitMode, KijunExitMode


# ---------------------------------------------------------------------------
# Mock trade object
# ---------------------------------------------------------------------------

@dataclass
class MockTrade:
    """Minimal ActiveTrade-compatible object for unit tests."""

    entry_price: float
    stop_loss: float
    direction: str
    remaining_pct: float = 1.0
    current_r: float = 0.0
    partial_exits: List[dict] = field(default_factory=list)
    original_stop_loss: float = field(init=False)

    def __post_init__(self) -> None:
        self.original_stop_loss = self.stop_loss

    @property
    def initial_risk(self) -> float:
        """Initial risk in price units (entry to original stop)."""
        if self.direction == 'long':
            return self.entry_price - self.original_stop_loss
        return self.original_stop_loss - self.entry_price

    @property
    def is_partial(self) -> bool:
        """True if the first partial exit has already been executed."""
        return self.remaining_pct < 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_eval_matrix(
    signal_kijun: Optional[float] = None,
    higher_kijun: Optional[float] = None,
    signal_tf: str = '1H',
) -> EvalMatrix:
    """Build an EvalMatrix populated with kijun values."""
    matrix = EvalMatrix()
    if signal_kijun is not None:
        matrix.set(
            f'ichimoku_{signal_tf}',
            EvaluatorResult(direction=1.0, confidence=0.8, metadata={'kijun': signal_kijun}),
        )
    if higher_kijun is not None:
        matrix.set(
            'ichimoku_4H',
            EvaluatorResult(direction=1.0, confidence=0.9, metadata={'kijun': higher_kijun}),
        )
    return matrix


# ---------------------------------------------------------------------------
# KijunExitMode tests
# ---------------------------------------------------------------------------

class TestKijunExitMode:

    def setup_method(self):
        self.mode = KijunExitMode()

    # --- Long trade fixtures ------------------------------------------------

    def _long_trade(self) -> MockTrade:
        """Long trade: entry=1900, stop=1880, initial_risk=20."""
        return MockTrade(entry_price=1900.0, stop_loss=1880.0, direction='long')

    def _long_partial_trade(self) -> MockTrade:
        """Long trade already at partial stage (stop moved up slightly)."""
        t = MockTrade(entry_price=1900.0, stop_loss=1880.0, direction='long')
        t.remaining_pct = 0.5  # simulate partial exit already done
        t.stop_loss = 1890.0   # stop moved up a little after partial
        return t

    # --- Test 1: full_exit when stop is hit ---------------------------------

    def test_full_exit_stop_hit_long(self):
        """Long trade: price falls to or below stop → full_exit."""
        trade = self._long_trade()
        # Price exactly at stop
        data = {'close': 1880.0}
        decision = self.mode.check_exit(trade, data, EvalMatrix())
        assert isinstance(decision, ExitDecision)
        assert decision.action == 'full_exit'
        assert decision.close_pct == 1.0

    def test_full_exit_stop_hit_short(self):
        """Short trade: price rises to or above stop → full_exit."""
        trade = MockTrade(entry_price=1900.0, stop_loss=1920.0, direction='short')
        data = {'close': 1920.0}
        decision = self.mode.check_exit(trade, data, EvalMatrix())
        assert isinstance(decision, ExitDecision)
        assert decision.action == 'full_exit'
        assert decision.close_pct == 1.0

    def test_full_exit_uses_remaining_pct(self):
        """full_exit closes the remaining fraction, not always 1.0."""
        trade = self._long_partial_trade()
        data = {'close': trade.stop_loss}  # exactly at stop
        decision = self.mode.check_exit(trade, data, EvalMatrix())
        assert decision.action == 'full_exit'
        assert decision.close_pct == trade.remaining_pct

    # --- Test 2: partial_exit at 2R -----------------------------------------

    def test_partial_exit_at_2r_long(self):
        """Long trade reaching 2R → partial_exit at 50%."""
        trade = self._long_trade()
        # entry=1900, stop=1880, risk=20; 2R price = 1940
        data = {'close': 1940.0}
        decision = self.mode.check_exit(trade, data, EvalMatrix())
        assert isinstance(decision, ExitDecision)
        assert decision.action == 'partial_exit'
        assert decision.close_pct == 0.5

    def test_partial_exit_at_2r_short(self):
        """Short trade reaching 2R → partial_exit at 50%."""
        trade = MockTrade(entry_price=1900.0, stop_loss=1920.0, direction='short')
        # risk=20; 2R price = 1860
        data = {'close': 1860.0}
        decision = self.mode.check_exit(trade, data, EvalMatrix())
        assert decision.action == 'partial_exit'
        assert decision.close_pct == 0.5

    def test_no_second_partial_exit(self):
        """After first partial, reaching 2R again does not trigger another partial."""
        trade = self._long_partial_trade()
        # At 2R but already partial
        data = {'close': 1940.0}
        matrix = make_eval_matrix(signal_kijun=None)  # no kijun → should hold
        decision = self.mode.check_exit(trade, data, matrix)
        assert decision.action != 'partial_exit'

    # --- Test 3: trail_update with signal-TF kijun at 1.5R+ ----------------

    def test_trail_update_signal_kijun_at_1_5r(self):
        """After partial, R >= 1.5 with signal-TF kijun → trail_update."""
        trade = self._long_partial_trade()
        # entry=1900, stop=1890 (current after partial), risk=20
        # 1.5R price = 1900 + 1.5*20 = 1930
        data = {'close': 1930.0}
        # kijun at 1895 — improves on current stop (1890)
        matrix = make_eval_matrix(signal_kijun=1895.0)
        decision = self.mode.check_exit(trade, data, matrix)
        assert isinstance(decision, ExitDecision)
        assert decision.action == 'trail_update'
        assert decision.new_stop == 1895.0

    def test_trail_update_uses_15m_kijun_when_no_1h(self):
        """Falls back to 15M kijun when 1H is not available."""
        trade = self._long_partial_trade()
        data = {'close': 1930.0}
        # Only 15M kijun present
        matrix = make_eval_matrix(signal_kijun=1893.0, signal_tf='15M')
        decision = self.mode.check_exit(trade, data, matrix)
        assert decision.action == 'trail_update'
        assert decision.new_stop == 1893.0

    def test_trail_not_applied_if_kijun_below_current_stop(self):
        """Trail is rejected if new kijun level would worsen the stop."""
        trade = self._long_partial_trade()
        # Current stop = 1890; kijun at 1885 → would move stop DOWN (worse for long)
        data = {'close': 1930.0}
        matrix = make_eval_matrix(signal_kijun=1885.0)
        decision = self.mode.check_exit(trade, data, matrix)
        # Should not trail to a worse stop
        assert decision.action != 'trail_update'

    # --- Test 4: no trail below 1.5R ----------------------------------------

    def test_no_trail_below_1_5r(self):
        """Below 1.5R, no trail update even if partial and kijun available."""
        trade = self._long_partial_trade()
        # entry=1900, risk=20; 1.4R price = 1900 + 1.4*20 = 1928
        data = {'close': 1928.0}
        matrix = make_eval_matrix(signal_kijun=1895.0)
        decision = self.mode.check_exit(trade, data, matrix)
        assert decision.action == 'hold'

    def test_no_trail_exactly_at_threshold(self):
        """At exactly 1.5R, trail is permitted."""
        trade = self._long_partial_trade()
        # 1.5R price = 1930
        data = {'close': 1930.0}
        matrix = make_eval_matrix(signal_kijun=1895.0)
        decision = self.mode.check_exit(trade, data, matrix)
        # kijun(1895) > stop(1890) → trail should fire
        assert decision.action == 'trail_update'

    # --- Test 5: higher-TF kijun at 3R+ ------------------------------------

    def test_higher_tf_kijun_at_3r(self):
        """At R >= 3, higher-TF kijun takes over trailing if available."""
        trade = self._long_partial_trade()
        # entry=1900, risk=20; 3R price = 1960
        data = {'close': 1960.0}
        # higher kijun at 1910 (above current stop 1890) → should use this
        matrix = make_eval_matrix(signal_kijun=1905.0, higher_kijun=1910.0)
        decision = self.mode.check_exit(trade, data, matrix)
        assert decision.action == 'trail_update'
        assert decision.new_stop == 1910.0
        assert 'higher-TF Kijun' in decision.reason

    def test_falls_back_to_signal_kijun_when_no_higher(self):
        """At 3R+ but no higher-TF kijun → falls back to signal-TF kijun."""
        trade = self._long_partial_trade()
        data = {'close': 1960.0}
        matrix = make_eval_matrix(signal_kijun=1905.0, higher_kijun=None)
        decision = self.mode.check_exit(trade, data, matrix)
        assert decision.action == 'trail_update'
        assert decision.new_stop == 1905.0
        assert 'signal-TF Kijun' in decision.reason

    # --- Test 6: hold when nothing triggers ---------------------------------

    def test_hold_below_2r_no_partial(self):
        """Below 2R, no partial, no trail → hold."""
        trade = self._long_trade()
        data = {'close': 1920.0}  # 1R exactly, not enough for partial
        decision = self.mode.check_exit(trade, data, EvalMatrix())
        assert isinstance(decision, ExitDecision)
        assert decision.action == 'hold'

    def test_hold_partial_below_1_5r_no_kijun(self):
        """After partial, below 1.5R, no kijun → hold."""
        trade = self._long_partial_trade()
        data = {'close': 1920.0}
        decision = self.mode.check_exit(trade, data, EvalMatrix())
        assert decision.action == 'hold'


# ---------------------------------------------------------------------------
# DefaultExitMode tests
# ---------------------------------------------------------------------------

class TestDefaultExitMode:

    def setup_method(self):
        self.mode = DefaultExitMode()

    def _long_trade(self) -> MockTrade:
        return MockTrade(entry_price=1900.0, stop_loss=1880.0, direction='long')

    def _long_partial_trade(self) -> MockTrade:
        t = MockTrade(entry_price=1900.0, stop_loss=1880.0, direction='long')
        t.remaining_pct = 0.5
        t.stop_loss = 1890.0
        return t

    # --- Test 7: partial_exit at 2R -----------------------------------------

    def test_partial_at_2r_long(self):
        """Long trade at 2R → partial_exit 50%."""
        trade = self._long_trade()
        data = {'close': 1940.0, 'atr': 5.0}
        decision = self.mode.check_exit(trade, data, EvalMatrix())
        assert isinstance(decision, ExitDecision)
        assert decision.action == 'partial_exit'
        assert decision.close_pct == 0.5

    def test_partial_at_2r_short(self):
        """Short trade at 2R → partial_exit 50%."""
        trade = MockTrade(entry_price=1900.0, stop_loss=1920.0, direction='short')
        data = {'close': 1860.0, 'atr': 5.0}
        decision = self.mode.check_exit(trade, data, EvalMatrix())
        assert decision.action == 'partial_exit'
        assert decision.close_pct == 0.5

    def test_full_exit_stop_hit(self):
        """Stop hit → full_exit."""
        trade = self._long_trade()
        data = {'close': 1879.0, 'atr': 5.0}
        decision = self.mode.check_exit(trade, data, EvalMatrix())
        assert decision.action == 'full_exit'

    # --- Test 8: ATR trailing after partial ---------------------------------

    def test_atr_trail_long(self):
        """After partial, R >= 1.5, ATR available → trail_update."""
        trade = self._long_partial_trade()
        # current stop = 1890; 1.5R price = 1930
        # ATR trail = 1930 - 2*5 = 1920 → improves on 1890
        data = {'close': 1930.0, 'atr': 5.0}
        decision = self.mode.check_exit(trade, data, EvalMatrix())
        assert isinstance(decision, ExitDecision)
        assert decision.action == 'trail_update'
        assert decision.new_stop == pytest.approx(1920.0)

    def test_atr_trail_short(self):
        """Short: after partial, ATR trail moves stop down (tighter)."""
        trade = MockTrade(entry_price=1900.0, stop_loss=1920.0, direction='short')
        trade.remaining_pct = 0.5
        trade.stop_loss = 1912.0  # current stop
        # 1.5R short price = 1900 - 1.5*20 = 1870
        # ATR trail = 1870 + 2*5 = 1880 → below current stop (1912) → improves
        data = {'close': 1870.0, 'atr': 5.0}
        decision = self.mode.check_exit(trade, data, EvalMatrix())
        assert decision.action == 'trail_update'
        assert decision.new_stop == pytest.approx(1880.0)

    def test_atr_trail_not_fired_below_1_5r(self):
        """Below 1.5R, no ATR trail even if partial."""
        trade = self._long_partial_trade()
        # 1.4R price = 1928
        data = {'close': 1928.0, 'atr': 5.0}
        decision = self.mode.check_exit(trade, data, EvalMatrix())
        assert decision.action == 'hold'

    def test_atr_trail_rejected_if_worsens_stop(self):
        """ATR trail is skipped if it would move stop further from price."""
        trade = self._long_partial_trade()
        trade.stop_loss = 1925.0  # stop already very tight
        # ATR trail = 1930 - 2*5 = 1920 → below current (1925) → worse for long
        data = {'close': 1930.0, 'atr': 5.0}
        decision = self.mode.check_exit(trade, data, EvalMatrix())
        assert decision.action != 'trail_update'

    def test_no_trail_with_zero_atr(self):
        """When ATR is 0 (or missing), trail update is skipped."""
        trade = self._long_partial_trade()
        data = {'close': 1930.0, 'atr': 0.0}
        decision = self.mode.check_exit(trade, data, EvalMatrix())
        assert decision.action == 'hold'

    def test_no_atr_key_defaults_gracefully(self):
        """Missing 'atr' key in current_data → no trail, no crash."""
        trade = self._long_partial_trade()
        data = {'close': 1930.0}  # no 'atr' key
        decision = self.mode.check_exit(trade, data, EvalMatrix())
        assert decision.action == 'hold'

    def test_hold_below_2r(self):
        """Below 2R, not partial → hold."""
        trade = self._long_trade()
        data = {'close': 1920.0, 'atr': 5.0}
        decision = self.mode.check_exit(trade, data, EvalMatrix())
        assert decision.action == 'hold'


# ---------------------------------------------------------------------------
# Cross-mode: both modes return ExitDecision
# ---------------------------------------------------------------------------

class TestReturnTypes:

    @pytest.mark.parametrize('mode_cls', [KijunExitMode, DefaultExitMode])
    def test_always_returns_exit_decision(self, mode_cls):
        """Both modes always return an ExitDecision instance."""
        mode = mode_cls()
        trade = MockTrade(entry_price=1900.0, stop_loss=1880.0, direction='long')
        data = {'close': 1910.0, 'atr': 3.0}
        result = mode.check_exit(trade, data, EvalMatrix())
        assert isinstance(result, ExitDecision)

    @pytest.mark.parametrize('mode_cls', [KijunExitMode, DefaultExitMode])
    def test_action_is_valid_string(self, mode_cls):
        """Action field is always one of the valid action strings."""
        valid_actions = {'hold', 'partial_exit', 'trail_update', 'full_exit'}
        mode = mode_cls()
        trade = MockTrade(entry_price=1900.0, stop_loss=1880.0, direction='long')
        # Test multiple price scenarios
        for price in [1870.0, 1900.0, 1920.0, 1940.0, 1960.0]:
            data = {'close': price, 'atr': 5.0}
            result = mode.check_exit(trade, data, EvalMatrix())
            assert result.action in valid_actions, (
                f"{mode_cls.__name__} returned invalid action {result.action!r} "
                f"at price={price}"
            )
