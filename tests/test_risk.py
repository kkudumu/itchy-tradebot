"""Comprehensive tests for the risk management layer.

Covers:
- Position sizing: lot calculation and phase switching
- Circuit breaker: daily loss enforcement
- Exit manager: hybrid 50/50 exits, Kijun trailing, no-BE-before-1R rule
- Trade manager: full lifecycle, concurrent limits, integration
- Hard-coded limits: verify absolute caps cannot be exceeded
- Edge cases: threshold boundaries, zero ATR guard, negative PnL
"""

from __future__ import annotations

import datetime

import pytest

from src.risk.circuit_breaker import DailyCircuitBreaker
from src.risk.exit_manager import ActiveTrade, ExitDecision, HybridExitManager
from src.risk.position_sizer import AdaptivePositionSizer, PositionSize
from src.risk.trade_manager import TradeManager


# ============================================================
# Helpers
# ============================================================

def _make_long_trade(
    entry: float = 2000.0,
    stop: float = 1992.0,
    tp: float = 2016.0,
    lot: float = 0.1,
) -> ActiveTrade:
    """Long trade with 8-point risk (stop 8 below entry), 16-point TP (2R)."""
    return ActiveTrade(
        entry_price=entry,
        stop_loss=stop,
        take_profit=tp,
        direction="long",
        lot_size=lot,
        entry_time=datetime.datetime(2024, 1, 15, 10, 0, 0),
    )


def _make_short_trade(
    entry: float = 2000.0,
    stop: float = 2008.0,
    tp: float = 1984.0,
    lot: float = 0.1,
) -> ActiveTrade:
    """Short trade with 8-point risk (stop 8 above entry), 16-point TP (2R)."""
    return ActiveTrade(
        entry_price=entry,
        stop_loss=stop,
        take_profit=tp,
        direction="short",
        lot_size=lot,
        entry_time=datetime.datetime(2024, 1, 15, 10, 0, 0),
    )


def _make_trade_manager(
    balance: float = 10_000.0,
    max_concurrent: int = 1,
) -> TradeManager:
    sizer = AdaptivePositionSizer(initial_balance=balance)
    breaker = DailyCircuitBreaker(max_daily_loss_pct=2.0)
    breaker.start_day(balance, datetime.date(2024, 1, 15))
    exit_mgr = HybridExitManager()
    return TradeManager(sizer, breaker, exit_mgr, max_concurrent=max_concurrent)


# ============================================================
# AdaptivePositionSizer
# ============================================================

class TestAdaptivePositionSizer:

    def test_lot_calculation_10k_account(self):
        """Verify lot size for 10K account, 1.5% risk, ATR=5, multiplier=1.5.

        risk_amount    = 10_000 * 0.015 = 150
        stop_distance  = 5 * 1.5 = 7.5
        lot_size       = 150 / (7.5 * 1.0) = 20.0 → clamped to max_lot=10.0
        """
        sizer = AdaptivePositionSizer(initial_balance=10_000.0, max_lot=10.0)
        pos = sizer.calculate_position_size(
            account_equity=10_000.0,
            atr=5.0,
            atr_multiplier=1.5,
            point_value=1.0,
        )
        # risk_amount=150, stop=7.5, raw_lot=20 → clamped to 10.0
        assert pos.lot_size == 10.0
        assert pos.risk_pct == pytest.approx(1.5)
        assert pos.risk_amount == pytest.approx(150.0)
        assert pos.stop_distance == pytest.approx(7.5)
        assert pos.phase == "aggressive"

    def test_lot_calculation_unclamped(self):
        """Smaller ATR and higher point_value produces unclamped lot."""
        sizer = AdaptivePositionSizer(initial_balance=10_000.0, max_lot=100.0)
        # risk_amount = 10_000 * 0.015 = 150
        # stop = 2.0 * 1.5 = 3.0
        # lot  = 150 / (3.0 * 100.0) = 0.5
        pos = sizer.calculate_position_size(
            account_equity=10_000.0,
            atr=2.0,
            atr_multiplier=1.5,
            point_value=100.0,
        )
        assert pos.lot_size == pytest.approx(0.5, rel=1e-3)

    def test_phase1_aggressive_risk(self):
        """Below +4% profit: should use initial_risk_pct (1.5%)."""
        sizer = AdaptivePositionSizer(initial_balance=10_000.0)
        sizer.update_balance(10_390.0)  # +3.9% — still Phase 1
        assert sizer.get_risk_pct() == pytest.approx(1.5)
        assert sizer.get_phase() == "aggressive"

    def test_phase2_protective_at_exact_threshold(self):
        """Exactly at +4% profit: should switch to reduced_risk_pct (0.75%)."""
        sizer = AdaptivePositionSizer(initial_balance=10_000.0, phase_threshold_pct=4.0)
        sizer.update_balance(10_400.0)  # exactly +4%
        assert sizer.get_risk_pct() == pytest.approx(0.75)
        assert sizer.get_phase() == "protective"

    def test_phase2_protective_above_threshold(self):
        """Above +4% profit: reduced risk applies."""
        sizer = AdaptivePositionSizer(initial_balance=10_000.0)
        sizer.update_balance(11_000.0)  # +10%
        assert sizer.get_risk_pct() == pytest.approx(0.75)
        assert sizer.get_phase() == "protective"

    def test_phase_switch_drops_risk(self):
        """Risk percentage drops when phase switches."""
        sizer = AdaptivePositionSizer(initial_balance=10_000.0)
        risk_before = sizer.get_risk_pct()
        sizer.update_balance(10_500.0)
        risk_after = sizer.get_risk_pct()
        assert risk_before > risk_after

    def test_hard_coded_max_risk_cap(self):
        """Configuring initial_risk_pct above 2% is silently clamped to 2%."""
        sizer = AdaptivePositionSizer(initial_balance=10_000.0, initial_risk_pct=5.0)
        assert sizer.get_risk_pct() <= sizer._MAX_RISK_PCT

    def test_hard_coded_min_risk_floor(self):
        """Configuring reduced_risk_pct below 0.25% is silently clamped."""
        sizer = AdaptivePositionSizer(
            initial_balance=10_000.0,
            reduced_risk_pct=0.01,
            initial_risk_pct=1.5,
        )
        sizer.update_balance(11_000.0)  # trigger Phase 2
        assert sizer.get_risk_pct() >= sizer._MIN_RISK_PCT

    def test_lot_clamped_to_minimum(self):
        """If computed lot is below min_lot, result is min_lot."""
        sizer = AdaptivePositionSizer(
            initial_balance=100.0,
            initial_risk_pct=1.5,
            min_lot=0.01,
        )
        # risk_amount = 1.5, stop = 100 * 1.5 = 150 → lot = 1.5/150 = 0.01
        pos = sizer.calculate_position_size(
            account_equity=100.0, atr=100.0, atr_multiplier=1.5, point_value=1.0
        )
        assert pos.lot_size >= 0.01

    def test_invalid_account_equity_raises(self):
        sizer = AdaptivePositionSizer(initial_balance=10_000.0)
        with pytest.raises(ValueError):
            sizer.calculate_position_size(
                account_equity=0.0, atr=5.0, atr_multiplier=1.5, point_value=1.0
            )

    def test_invalid_atr_raises(self):
        sizer = AdaptivePositionSizer(initial_balance=10_000.0)
        with pytest.raises(ValueError):
            sizer.calculate_position_size(
                account_equity=10_000.0, atr=0.0, atr_multiplier=1.5, point_value=1.0
            )

    def test_position_size_returns_dataclass(self):
        sizer = AdaptivePositionSizer(initial_balance=10_000.0)
        pos = sizer.calculate_position_size(
            account_equity=10_000.0, atr=5.0, atr_multiplier=1.5, point_value=1.0
        )
        assert isinstance(pos, PositionSize)
        assert pos.lot_size > 0
        assert pos.risk_pct > 0


# ============================================================
# DailyCircuitBreaker
# ============================================================

class TestDailyCircuitBreaker:

    def test_no_loss_allows_trading(self):
        cb = DailyCircuitBreaker(max_daily_loss_pct=2.0)
        cb.start_day(10_000.0, datetime.date(2024, 1, 15))
        assert cb.can_trade(10_000.0) is True

    def test_small_loss_allows_trading(self):
        cb = DailyCircuitBreaker(max_daily_loss_pct=2.0)
        cb.start_day(10_000.0)
        # 1% loss — below 2% limit
        assert cb.can_trade(9_900.0) is True

    def test_exact_loss_limit_blocks_trading(self):
        """At exactly the loss threshold, trading should be blocked."""
        cb = DailyCircuitBreaker(max_daily_loss_pct=2.0)
        cb.start_day(10_000.0)
        # Exactly 2% loss
        assert cb.can_trade(9_800.0) is False

    def test_loss_above_limit_blocks_trading(self):
        cb = DailyCircuitBreaker(max_daily_loss_pct=2.0)
        cb.start_day(10_000.0)
        assert cb.can_trade(9_700.0) is False

    def test_circuit_stays_tripped_even_if_balance_recovers(self):
        """Once tripped, circuit stays off for the rest of the day."""
        cb = DailyCircuitBreaker(max_daily_loss_pct=2.0)
        cb.start_day(10_000.0)
        # Trip it
        cb.can_trade(9_700.0)
        assert cb.is_triggered() is True
        # Even if balance recovers (e.g. unrealised reversal), still blocked
        assert cb.can_trade(10_000.0) is False

    def test_start_day_resets_circuit(self):
        """A new day resets the circuit breaker."""
        cb = DailyCircuitBreaker(max_daily_loss_pct=2.0)
        cb.start_day(10_000.0)
        cb.can_trade(9_700.0)  # trip
        assert cb.is_triggered() is True

        # New day
        cb.start_day(9_700.0, datetime.date(2024, 1, 16))
        assert cb.is_triggered() is False
        assert cb.can_trade(9_700.0) is True

    def test_daily_loss_pct_calculation(self):
        cb = DailyCircuitBreaker(max_daily_loss_pct=2.0)
        cb.start_day(10_000.0)
        loss_pct = cb.daily_loss_pct(9_800.0)
        assert loss_pct == pytest.approx(2.0)

    def test_daily_loss_pct_positive_on_profit(self):
        """Profit → loss_pct should return 0 (not negative)."""
        cb = DailyCircuitBreaker(max_daily_loss_pct=2.0)
        cb.start_day(10_000.0)
        assert cb.daily_loss_pct(10_500.0) == pytest.approx(0.0)

    def test_remaining_budget(self):
        cb = DailyCircuitBreaker(max_daily_loss_pct=2.0)
        cb.start_day(10_000.0)
        # After a 1% loss, 1% remains
        remaining = cb.remaining_risk_budget(9_900.0)
        assert remaining == pytest.approx(100.0, rel=1e-3)  # 1% of 10_000

    def test_remaining_budget_zero_when_tripped(self):
        cb = DailyCircuitBreaker(max_daily_loss_pct=2.0)
        cb.start_day(10_000.0)
        cb.can_trade(9_700.0)  # trip
        assert cb.remaining_risk_budget(9_700.0) == pytest.approx(0.0)

    def test_max_loss_capped_at_absolute_limit(self):
        """Configuring loss > 5% is clamped to the absolute cap."""
        cb = DailyCircuitBreaker(max_daily_loss_pct=10.0)
        assert cb.max_daily_loss_pct <= DailyCircuitBreaker._ABSOLUTE_MAX_DAILY_LOSS_PCT


# ============================================================
# HybridExitManager — R-multiple calculation
# ============================================================

class TestRMultipleCalculation:

    def setup_method(self):
        self.mgr = HybridExitManager()

    def test_long_at_entry_is_zero_r(self):
        r = self.mgr.calculate_r_multiple(
            entry_price=2000.0, current_price=2000.0,
            stop_loss=1992.0, direction="long"
        )
        assert r == pytest.approx(0.0)

    def test_long_at_2r(self):
        # entry=2000, stop=1992 → risk=8, 2R target = 2016
        r = self.mgr.calculate_r_multiple(
            entry_price=2000.0, current_price=2016.0,
            stop_loss=1992.0, direction="long"
        )
        assert r == pytest.approx(2.0)

    def test_long_at_1_5r(self):
        r = self.mgr.calculate_r_multiple(
            entry_price=2000.0, current_price=2012.0,
            stop_loss=1992.0, direction="long"
        )
        assert r == pytest.approx(1.5)

    def test_long_in_drawdown_negative_r(self):
        r = self.mgr.calculate_r_multiple(
            entry_price=2000.0, current_price=1996.0,
            stop_loss=1992.0, direction="long"
        )
        assert r == pytest.approx(-0.5)

    def test_short_at_2r(self):
        # entry=2000, stop=2008 → risk=8, 2R target = 1984
        r = self.mgr.calculate_r_multiple(
            entry_price=2000.0, current_price=1984.0,
            stop_loss=2008.0, direction="short"
        )
        assert r == pytest.approx(2.0)

    def test_short_in_drawdown_negative_r(self):
        r = self.mgr.calculate_r_multiple(
            entry_price=2000.0, current_price=2004.0,
            stop_loss=2008.0, direction="short"
        )
        assert r == pytest.approx(-0.5)

    def test_short_at_3r(self):
        r = self.mgr.calculate_r_multiple(
            entry_price=2000.0, current_price=1976.0,
            stop_loss=2008.0, direction="short"
        )
        assert r == pytest.approx(3.0)


# ============================================================
# HybridExitManager — exit decisions
# ============================================================

class TestHybridExitManager:

    def setup_method(self):
        self.mgr = HybridExitManager(
            tp_r_multiple=2.0,
            breakeven_threshold_r=1.0,
            kijun_trail_start_r=1.5,
            higher_tf_kijun_start_r=3.0,
        )

    # --- No-BE before 1R (CRITICAL RULE) ---

    def test_no_breakeven_before_1r_long(self):
        """At 0.5R, stop must NOT be moved to breakeven."""
        trade = _make_long_trade()
        trade.remaining_pct = 0.5  # simulate partial already done
        # At 0.5R: current = 2000 + (0.5 * 8) = 2004
        decision = self.mgr.check_exit(
            trade, current_price=2004.0, kijun_value=2001.0
        )
        # No stop update should push stop to BE (2000.0) or above 1992
        if decision.new_stop is not None:
            assert decision.new_stop <= trade.original_stop_loss, (
                "Stop must not be moved above original stop before 1R"
            )

    def test_no_breakeven_before_1r_no_trail_at_0_5r(self):
        """Trailing only begins at 1.5R — no trail action at 0.5R."""
        trade = _make_long_trade()
        trade.remaining_pct = 0.5
        trade.current_r = 0.5
        trail_stop = self.mgr.get_trailing_stop(trade, kijun_value=2001.0)
        assert trail_stop is None, "No trailing stop update expected at 0.5R"

    def test_no_trail_at_exactly_1r(self):
        """At exactly 1R, Kijun trail has not yet activated (threshold is 1.5R)."""
        trade = _make_long_trade()
        trade.remaining_pct = 0.5
        trade.current_r = 1.0
        trail_stop = self.mgr.get_trailing_stop(trade, kijun_value=1998.0)
        assert trail_stop is None

    # --- Partial exit at 2R ---

    def test_partial_exit_triggered_at_2r(self):
        """At exactly 2R, first 50% partial exit is triggered."""
        trade = _make_long_trade()  # entry=2000, stop=1992, risk=8
        # 2R price = 2000 + 2*8 = 2016
        decision = self.mgr.check_exit(
            trade, current_price=2016.0, kijun_value=2010.0
        )
        assert decision.action == "partial_exit"
        assert decision.close_pct == pytest.approx(0.5)
        assert decision.r_multiple == pytest.approx(2.0)

    def test_partial_exit_not_triggered_below_2r(self):
        """Below 2R, partial exit should not fire."""
        trade = _make_long_trade()
        # 1.9R = 2000 + 1.9*8 = 2015.2
        decision = self.mgr.check_exit(
            trade, current_price=2015.0, kijun_value=2010.0
        )
        assert decision.action == "no_action"

    def test_partial_exit_not_repeated(self):
        """Second call after partial should not close another 50%."""
        trade = _make_long_trade()
        trade.remaining_pct = 0.5  # already partially closed
        trade.partial_exits = [{"pct_closed": 0.5}]
        # At 2.5R — should trail, not partial again
        # 2.5R price = 2000 + 2.5*8 = 2020
        decision = self.mgr.check_exit(
            trade, current_price=2020.0, kijun_value=2012.0
        )
        assert decision.action != "partial_exit"

    # --- Kijun trailing ---

    def test_trail_begins_at_1_5r_long(self):
        """At 1.5R (post-partial), trailing stop should update to Kijun."""
        trade = _make_long_trade()
        trade.remaining_pct = 0.5
        trade.current_r = 1.5  # set so get_trailing_stop can read it
        # 1.5R price = 2000 + 1.5*8 = 2012
        trail_stop = self.mgr.get_trailing_stop(trade, kijun_value=1998.0)
        assert trail_stop == pytest.approx(1998.0)

    def test_trail_at_2_5r_uses_signal_kijun(self):
        """Between 1.5R and 3R: should use signal-TF Kijun, not higher-TF."""
        trade = _make_long_trade()
        trade.remaining_pct = 0.5
        trade.current_r = 2.5
        trail_stop = self.mgr.get_trailing_stop(
            trade, kijun_value=2005.0, higher_tf_kijun=2003.0
        )
        # At 2.5R, still below 3R threshold — uses signal-TF
        assert trail_stop == pytest.approx(2005.0)

    def test_trail_at_3r_plus_uses_higher_tf_kijun(self):
        """At 3R+, should trail with the higher-TF Kijun."""
        trade = _make_long_trade()
        trade.remaining_pct = 0.5
        trade.current_r = 3.5
        trail_stop = self.mgr.get_trailing_stop(
            trade, kijun_value=2010.0, higher_tf_kijun=2006.0
        )
        assert trail_stop == pytest.approx(2006.0)

    def test_trail_at_3r_without_higher_tf_uses_signal(self):
        """At 3R+ but no higher-TF Kijun provided: fall back to signal-TF."""
        trade = _make_long_trade()
        trade.remaining_pct = 0.5
        trade.current_r = 3.5
        trail_stop = self.mgr.get_trailing_stop(
            trade, kijun_value=2010.0, higher_tf_kijun=None
        )
        assert trail_stop == pytest.approx(2010.0)

    def test_trail_kijun_below_entry_not_breakeven(self):
        """Kijun below entry at 2.5R means stop trails below entry — not BE."""
        trade = _make_long_trade(entry=2000.0, stop=1992.0)
        trade.remaining_pct = 0.5
        # Kijun at 1996 — below entry (2000) — this is legitimate trailing
        trade.current_r = 2.5
        trail_stop = self.mgr.get_trailing_stop(trade, kijun_value=1996.0)
        assert trail_stop == pytest.approx(1996.0)
        # And the stop improves from original 1992 → 1996 (closer)
        assert self.mgr._stop_improves(trade, 1996.0) is True

    # --- Full exit (stop hit) ---

    def test_full_exit_when_stop_hit_long(self):
        """Long trade full exit when price drops to stop."""
        trade = _make_long_trade(entry=2000.0, stop=1992.0)
        trade.remaining_pct = 0.5
        decision = self.mgr.check_exit(
            trade, current_price=1992.0, kijun_value=1990.0
        )
        assert decision.action == "full_exit"
        assert decision.close_pct == pytest.approx(0.5)

    def test_full_exit_when_stop_hit_short(self):
        """Short trade full exit when price rises to stop."""
        trade = _make_short_trade(entry=2000.0, stop=2008.0)
        trade.remaining_pct = 0.5
        decision = self.mgr.check_exit(
            trade, current_price=2008.0, kijun_value=2010.0
        )
        assert decision.action == "full_exit"

    # --- Short direction trailing ---

    def test_short_trail_at_1_5r(self):
        """Short trade: at 1.5R, trail stop to Kijun (which should be above entry)."""
        trade = _make_short_trade(entry=2000.0, stop=2008.0)
        trade.remaining_pct = 0.5
        trade.current_r = 1.5
        # For shorts, Kijun above entry means stop is moving toward entry
        trail_stop = self.mgr.get_trailing_stop(trade, kijun_value=2004.0)
        assert trail_stop == pytest.approx(2004.0)
        # Improves: short stop moves DOWN (closer to price, which is below entry)
        # 2004.0 < 2008.0 → improves
        assert self.mgr._stop_improves(trade, 2004.0) is True

    def test_short_no_be_before_1r(self):
        """Short trade: at 0.5R, no trail update should push stop to BE."""
        trade = _make_short_trade(entry=2000.0, stop=2008.0)
        trade.remaining_pct = 0.5
        trade.current_r = 0.5
        trail_stop = self.mgr.get_trailing_stop(trade, kijun_value=2005.0)
        assert trail_stop is None, "No trailing before 1.5R threshold"


# ============================================================
# TradeManager — integration
# ============================================================

class TestTradeManager:

    def test_open_trade_returns_active_trade(self):
        tm = _make_trade_manager()
        trade_id, trade, pos = tm.open_trade(
            entry_price=2000.0, stop_loss=1992.0, take_profit=2016.0,
            direction="long", atr=5.0, point_value=1.0,
            account_equity=10_000.0,
        )
        assert trade_id == 1
        assert isinstance(trade, ActiveTrade)
        assert trade.direction == "long"
        assert trade.lot_size > 0
        assert 1 in tm.active_trade_ids

    def test_max_concurrent_blocks_second_trade(self):
        """With max_concurrent=1, a second open should raise RuntimeError."""
        tm = _make_trade_manager(max_concurrent=1)
        tm.open_trade(
            entry_price=2000.0, stop_loss=1992.0, take_profit=2016.0,
            direction="long", atr=5.0, point_value=1.0,
            account_equity=10_000.0,
        )
        with pytest.raises(RuntimeError, match="Max concurrent"):
            tm.open_trade(
                entry_price=2001.0, stop_loss=1993.0, take_profit=2017.0,
                direction="long", atr=5.0, point_value=1.0,
                account_equity=10_000.0,
            )

    def test_max_concurrent_2_allows_second_trade(self):
        """With max_concurrent=2, second trade should open without error."""
        tm = _make_trade_manager(max_concurrent=2)
        tm.open_trade(
            entry_price=2000.0, stop_loss=1992.0, take_profit=2016.0,
            direction="long", atr=5.0, point_value=1.0,
            account_equity=10_000.0,
        )
        trade_id_2, _, _ = tm.open_trade(
            entry_price=2001.0, stop_loss=1993.0, take_profit=2017.0,
            direction="long", atr=5.0, point_value=1.0,
            account_equity=10_000.0,
        )
        assert trade_id_2 == 2

    def test_circuit_breaker_blocks_open(self):
        """After 2% daily loss, new trades should be blocked."""
        tm = _make_trade_manager(balance=10_000.0)
        # Trip the circuit breaker manually
        tm._breaker.can_trade(9_700.0)  # 3% loss → trips at 2%
        with pytest.raises(RuntimeError, match="circuit breaker"):
            tm.open_trade(
                entry_price=2000.0, stop_loss=1992.0, take_profit=2016.0,
                direction="long", atr=5.0, point_value=1.0,
                account_equity=9_700.0,
            )

    def test_full_trade_lifecycle_long(self):
        """Open → trail update → partial exit → full exit."""
        tm = _make_trade_manager(balance=10_000.0)

        # 1. Open
        trade_id, trade, pos = tm.open_trade(
            entry_price=2000.0, stop_loss=1992.0, take_profit=2016.0,
            direction="long", atr=5.0, point_value=1.0,
            account_equity=10_000.0,
        )

        # 2. Price at 1.0R — no action expected
        decision = tm.update_trade(
            trade_id, current_price=2008.0, kijun_value=1994.0
        )
        assert decision.action == "no_action"

        # 3. Price at 2R — partial exit
        decision = tm.update_trade(
            trade_id, current_price=2016.0, kijun_value=2005.0
        )
        assert decision.action == "partial_exit"
        assert trade.remaining_pct == pytest.approx(0.5)

        # 4. Trail update at 2.5R
        decision = tm.update_trade(
            trade_id, current_price=2020.0, kijun_value=2007.0
        )
        assert decision.action in ("trail_update", "no_action")

        # 5. Stop hit → full exit
        # Manually set stop to 2007 to simulate trail
        trade.stop_loss = 2007.0
        decision = tm.update_trade(
            trade_id, current_price=2007.0, kijun_value=2007.0
        )
        assert decision.action == "full_exit"
        assert trade_id not in tm.active_trade_ids

    def test_close_trade_manual(self):
        """Manual close removes trade from active list."""
        tm = _make_trade_manager()
        trade_id, _, _ = tm.open_trade(
            entry_price=2000.0, stop_loss=1992.0, take_profit=2016.0,
            direction="long", atr=5.0, point_value=1.0,
            account_equity=10_000.0,
        )
        result = tm.close_trade(trade_id, exit_price=2010.0, reason="friday_close")
        assert result["reason"] == "friday_close"
        assert result["pnl_points"] == pytest.approx(10.0)
        assert trade_id not in tm.active_trade_ids

    def test_close_trade_at_loss(self):
        """Closing at a loss produces negative pnl_points."""
        tm = _make_trade_manager()
        trade_id, _, _ = tm.open_trade(
            entry_price=2000.0, stop_loss=1992.0, take_profit=2016.0,
            direction="long", atr=5.0, point_value=1.0,
            account_equity=10_000.0,
        )
        result = tm.close_trade(trade_id, exit_price=1994.0)
        assert result["pnl_points"] < 0

    def test_equity_summary_structure(self):
        tm = _make_trade_manager()
        summary = tm.get_equity_summary()
        assert "balance" in summary
        assert "open_trades" in summary
        assert "realised_pnl" in summary
        assert "phase" in summary
        assert "circuit_triggered" in summary

    def test_invalid_direction_raises(self):
        tm = _make_trade_manager()
        with pytest.raises(ValueError, match="direction"):
            tm.open_trade(
                entry_price=2000.0, stop_loss=1992.0, take_profit=2016.0,
                direction="sideways", atr=5.0, point_value=1.0,
                account_equity=10_000.0,
            )

    def test_update_nonexistent_trade_raises(self):
        tm = _make_trade_manager()
        with pytest.raises(KeyError):
            tm.update_trade(999, current_price=2000.0, kijun_value=1995.0)

    def test_correlation_placeholder_always_true(self):
        tm = _make_trade_manager()
        assert tm.check_correlation("XAUUSD", "long") is True
        assert tm.check_correlation("EURUSD", "short") is True

    def test_can_open_trade_returns_tuple(self):
        tm = _make_trade_manager()
        result = tm.can_open_trade(10_000.0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)


# ============================================================
# Hard-coded limit verification
# ============================================================

class TestHardCodedLimits:
    """Verify that absolute caps cannot be exceeded regardless of configuration."""

    def test_position_sizer_absolute_max_risk(self):
        """_MAX_RISK_PCT is exactly 2.0 and enforced."""
        sizer = AdaptivePositionSizer(initial_balance=10_000.0, initial_risk_pct=99.9)
        # Risk applied in sizing is bounded
        pos = sizer.calculate_position_size(
            account_equity=10_000.0, atr=5.0, atr_multiplier=1.5, point_value=1.0
        )
        assert pos.risk_pct <= AdaptivePositionSizer._MAX_RISK_PCT

    def test_circuit_breaker_absolute_max(self):
        """_ABSOLUTE_MAX_DAILY_LOSS_PCT is 5.0 and enforced."""
        cb = DailyCircuitBreaker(max_daily_loss_pct=100.0)
        assert cb.max_daily_loss_pct == DailyCircuitBreaker._ABSOLUTE_MAX_DAILY_LOSS_PCT

    def test_trade_manager_constants(self):
        """Hard-coded absolute limits are present and correctly valued."""
        assert TradeManager._ABSOLUTE_MAX_RISK == pytest.approx(2.0)
        assert TradeManager._ABSOLUTE_MAX_DAILY_LOSS == pytest.approx(5.0)
        assert TradeManager._ABSOLUTE_MAX_TOTAL_DD == pytest.approx(10.0)

    def test_exit_manager_kijun_trail_min(self):
        """Kijun trail cannot be configured to start before 1R."""
        mgr = HybridExitManager(kijun_trail_start_r=0.1)
        assert mgr._kijun_trail_start_r >= HybridExitManager._ABSOLUTE_KIJUN_START_MIN


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:

    def test_phase_switches_at_exact_boundary(self):
        """Exactly at the phase threshold: should be protective."""
        sizer = AdaptivePositionSizer(
            initial_balance=10_000.0,
            phase_threshold_pct=4.0,
        )
        sizer.update_balance(10_400.0)  # exact +4%
        assert sizer.get_phase() == "protective"

    def test_phase_just_below_boundary(self):
        """Just below threshold (3.999...%): still aggressive."""
        sizer = AdaptivePositionSizer(initial_balance=10_000.0, phase_threshold_pct=4.0)
        sizer.update_balance(10_399.99)
        assert sizer.get_phase() == "aggressive"

    def test_circuit_breaker_at_exactly_limit(self):
        """At exactly 2% daily loss: circuit should trip."""
        cb = DailyCircuitBreaker(max_daily_loss_pct=2.0)
        cb.start_day(10_000.0)
        assert cb.can_trade(9_800.0) is False

    def test_circuit_breaker_at_just_below_limit(self):
        """Just below 2% daily loss: trading still permitted."""
        cb = DailyCircuitBreaker(max_daily_loss_pct=2.0)
        cb.start_day(10_000.0)
        # 1.999% loss
        assert cb.can_trade(9_800.1) is True

    def test_exit_at_exactly_tp_r(self):
        """Partial exit fires at exactly 2.0R."""
        mgr = HybridExitManager(tp_r_multiple=2.0)
        trade = _make_long_trade(entry=2000.0, stop=1992.0)
        # Exactly 2R = 2016
        decision = mgr.check_exit(trade, current_price=2016.0, kijun_value=2008.0)
        assert decision.action == "partial_exit"

    def test_exit_just_below_tp_r(self):
        """Just below 2R: no partial exit."""
        mgr = HybridExitManager(tp_r_multiple=2.0)
        trade = _make_long_trade(entry=2000.0, stop=1992.0)
        # 1.999R — just below
        decision = mgr.check_exit(trade, current_price=2015.99, kijun_value=2008.0)
        assert decision.action == "no_action"

    def test_kijun_trail_not_applied_in_wrong_direction(self):
        """Trailing stop update rejected if Kijun moves against the trade."""
        mgr = HybridExitManager()
        trade = _make_long_trade(entry=2000.0, stop=2005.0)  # stop already high
        trade.remaining_pct = 0.5
        trade.current_r = 2.5
        # Kijun below current stop → would move stop backward → rejected
        assert mgr._stop_improves(trade, 2003.0) is False

    def test_r_multiple_zero_risk_returns_zero(self):
        """If stop equals entry (degenerate), R-multiple returns 0."""
        mgr = HybridExitManager()
        r = mgr.calculate_r_multiple(
            entry_price=2000.0, current_price=2010.0,
            stop_loss=2000.0, direction="long"
        )
        assert r == pytest.approx(0.0)

    def test_position_size_negative_balance_raises(self):
        with pytest.raises(ValueError):
            AdaptivePositionSizer(initial_balance=-1.0)

    def test_circuit_breaker_before_start_day_allows_trading(self):
        """Before start_day is called, can_trade should return True (conservative)."""
        cb = DailyCircuitBreaker()
        assert cb.can_trade(10_000.0) is True

    def test_active_trade_original_stop_preserved(self):
        """original_stop_loss is frozen at entry even when stop is trailed."""
        trade = _make_long_trade(entry=2000.0, stop=1992.0)
        assert trade.original_stop_loss == pytest.approx(1992.0)
        trade.stop_loss = 1998.0  # trail update
        assert trade.original_stop_loss == pytest.approx(1992.0)  # unchanged

    def test_initial_risk_preserved_after_balance_update(self):
        """initial_risk on ActiveTrade is constant regardless of trailing."""
        trade = _make_long_trade(entry=2000.0, stop=1992.0)
        initial_risk = trade.initial_risk
        trade.stop_loss = 1998.0
        assert trade.initial_risk == pytest.approx(initial_risk)

    def test_short_trade_initial_risk(self):
        """Short trade risk = stop - entry."""
        trade = _make_short_trade(entry=2000.0, stop=2008.0)
        assert trade.initial_risk == pytest.approx(8.0)

    def test_long_trade_initial_risk(self):
        """Long trade risk = entry - stop."""
        trade = _make_long_trade(entry=2000.0, stop=1992.0)
        assert trade.initial_risk == pytest.approx(8.0)
