"""Tests for trading cost deduction in the backtester P&L methods."""
from __future__ import annotations

import pytest

from src.backtesting.vectorbt_engine import IchimokuBacktester


class TestUpdateBalanceFromTrade:
    """Test _update_balance_from_trade with and without costs."""

    def _make_backtester(self, commission: float = 0.0, spread: float = 0.0):
        config = {
            "edges": {
                "trading_costs": {
                    "commission_per_lot": commission,
                    "spread_points": spread,
                }
            }
        }
        return IchimokuBacktester(config=config, initial_balance=10_000.0)

    def test_zero_costs_matches_original_behavior(self):
        """With zero costs, P&L is unchanged from the original formula."""
        bt = self._make_backtester(commission=0.0, spread=0.0)
        trade = {"pnl_points": 10.0, "lot_size": 0.1, "remaining_pct": 1.0}
        new_bal = bt._update_balance_from_trade(10_000.0, trade)
        assert new_bal == pytest.approx(10_100.0)

    def test_commission_deducted(self):
        """Commission is deducted: commission_per_lot * lot_size * remaining_pct."""
        bt = self._make_backtester(commission=4.0, spread=0.0)
        trade = {"pnl_points": 10.0, "lot_size": 0.1, "remaining_pct": 1.0}
        new_bal = bt._update_balance_from_trade(10_000.0, trade)
        assert new_bal == pytest.approx(10_099.60)

    def test_spread_deducted(self):
        """Spread cost: spread_points * lot_size * point_value * remaining_pct."""
        bt = self._make_backtester(commission=0.0, spread=0.15)
        trade = {"pnl_points": 10.0, "lot_size": 0.1, "remaining_pct": 1.0}
        new_bal = bt._update_balance_from_trade(10_000.0, trade)
        assert new_bal == pytest.approx(10_098.50)

    def test_both_costs_deducted(self):
        """Both commission and spread are deducted from P&L."""
        bt = self._make_backtester(commission=4.0, spread=0.15)
        trade = {"pnl_points": 10.0, "lot_size": 0.1, "remaining_pct": 1.0}
        new_bal = bt._update_balance_from_trade(10_000.0, trade)
        assert new_bal == pytest.approx(10_098.10)

    def test_costs_on_partial_exit(self):
        """Partial exit (50%) gets proportional costs."""
        bt = self._make_backtester(commission=4.0, spread=0.15)
        trade = {"pnl_points": 10.0, "lot_size": 0.1, "remaining_pct": 0.5}
        new_bal = bt._update_balance_from_trade(10_000.0, trade)
        assert new_bal == pytest.approx(10_049.05)

    def test_costs_on_losing_trade(self):
        """Costs make losses worse."""
        bt = self._make_backtester(commission=4.0, spread=0.15)
        trade = {"pnl_points": -5.0, "lot_size": 0.1, "remaining_pct": 1.0}
        new_bal = bt._update_balance_from_trade(10_000.0, trade)
        assert new_bal == pytest.approx(9_948.10)


class TestPartialPnl:
    """Test _partial_pnl with and without costs."""

    def _make_backtester(self, commission: float = 0.0, spread: float = 0.0):
        config = {
            "edges": {
                "trading_costs": {
                    "commission_per_lot": commission,
                    "spread_points": spread,
                }
            }
        }
        return IchimokuBacktester(config=config, initial_balance=10_000.0)

    def _make_trade(self, direction="long", entry=2000.0, lot=0.1):
        from src.risk.exit_manager import ActiveTrade
        import datetime
        return ActiveTrade(
            entry_price=entry,
            stop_loss=1992.0 if direction == "long" else 2008.0,
            take_profit=2016.0 if direction == "long" else 1984.0,
            direction=direction,
            lot_size=lot,
            entry_time=datetime.datetime(2024, 1, 15, 10, 0, 0),
        )

    def test_partial_pnl_zero_costs(self):
        """No costs: partial pnl = pnl_points * lot * point_value * close_pct."""
        bt = self._make_backtester(commission=0.0, spread=0.0)
        trade = self._make_trade(direction="long", entry=2000.0, lot=0.1)
        pnl = bt._partial_pnl(trade, exit_price=2010.0, close_pct=0.5)
        assert pnl == pytest.approx(50.0)

    def test_partial_pnl_with_costs(self):
        """Costs deducted from partial pnl."""
        bt = self._make_backtester(commission=4.0, spread=0.15)
        trade = self._make_trade(direction="long", entry=2000.0, lot=0.1)
        pnl = bt._partial_pnl(trade, exit_price=2010.0, close_pct=0.5)
        assert pnl == pytest.approx(49.05)

    def test_partial_pnl_short_with_costs(self):
        """Short trade partial exit with costs."""
        bt = self._make_backtester(commission=4.0, spread=0.15)
        trade = self._make_trade(direction="short", entry=2000.0, lot=0.1)
        pnl = bt._partial_pnl(trade, exit_price=1990.0, close_pct=0.5)
        assert pnl == pytest.approx(49.05)
