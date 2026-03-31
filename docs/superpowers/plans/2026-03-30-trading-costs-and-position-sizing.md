# Trading Costs & Position Sizing Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add spread/commission cost deductions to backtest P&L and fix position sizer to use actual stop distance so risk percentages match reality.

**Architecture:** Two independent fixes applied to the risk/backtesting layer. Position sizer gains a `stop_distance_override` parameter; backtester gains cost config fields and deducts costs in P&L methods. Both default to prior behavior when not configured.

**Tech Stack:** Python, pytest, YAML config

**Spec:** `docs/superpowers/specs/2026-03-30-trading-costs-and-position-sizing-design.md`

---

### Task 1: Position Sizer — `stop_distance_override` Parameter

**Files:**
- Modify: `src/risk/position_sizer.py:111-174`
- Test: `tests/test_risk.py`

- [ ] **Step 1: Write failing tests for `stop_distance_override`**

Add to `tests/test_risk.py` inside `class TestAdaptivePositionSizer`:

```python
def test_stop_distance_override_used_when_provided(self):
    """When stop_distance_override is given, sizer ignores atr * multiplier."""
    sizer = AdaptivePositionSizer(initial_balance=10_000.0)
    # risk_amount = 10_000 * 0.015 = 150
    # override stop = 5.0
    # lot = 150 / (5.0 * 100.0) = 0.3
    pos = sizer.calculate_position_size(
        account_equity=10_000.0,
        atr=2.0,
        atr_multiplier=1.5,
        point_value=100.0,
        stop_distance_override=5.0,
    )
    assert pos.lot_size == pytest.approx(0.3, rel=1e-3)
    assert pos.stop_distance == pytest.approx(5.0)

def test_stop_distance_override_none_falls_back_to_atr(self):
    """When override is None, existing atr * multiplier logic is used."""
    sizer = AdaptivePositionSizer(initial_balance=10_000.0)
    # risk_amount = 150, stop = 2.0 * 1.5 = 3.0, lot = 150 / (3.0 * 100.0) = 0.5
    pos = sizer.calculate_position_size(
        account_equity=10_000.0,
        atr=2.0,
        atr_multiplier=1.5,
        point_value=100.0,
        stop_distance_override=None,
    )
    assert pos.lot_size == pytest.approx(0.5, rel=1e-3)
    assert pos.stop_distance == pytest.approx(3.0)

def test_stop_distance_override_zero_raises(self):
    """Zero stop_distance_override should raise ValueError."""
    sizer = AdaptivePositionSizer(initial_balance=10_000.0)
    with pytest.raises(ValueError):
        sizer.calculate_position_size(
            account_equity=10_000.0,
            atr=2.0,
            atr_multiplier=1.5,
            point_value=100.0,
            stop_distance_override=0.0,
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd .worktrees/spread-commission && python -m pytest tests/test_risk.py::TestAdaptivePositionSizer::test_stop_distance_override_used_when_provided tests/test_risk.py::TestAdaptivePositionSizer::test_stop_distance_override_none_falls_back_to_atr tests/test_risk.py::TestAdaptivePositionSizer::test_stop_distance_override_zero_raises -v`

Expected: FAIL — `calculate_position_size() got an unexpected keyword argument 'stop_distance_override'`

- [ ] **Step 3: Implement `stop_distance_override` in position_sizer.py**

In `src/risk/position_sizer.py`, modify `calculate_position_size()`:

```python
def calculate_position_size(
    self,
    account_equity: float,
    atr: float,
    atr_multiplier: float,
    point_value: float,
    instrument: str = "XAUUSD",
    stop_distance_override: float | None = None,
) -> PositionSize:
    if account_equity <= 0:
        raise ValueError(f"account_equity must be positive, got {account_equity}")
    if stop_distance_override is not None:
        if stop_distance_override <= 0:
            raise ValueError(f"stop_distance_override must be positive, got {stop_distance_override}")
    else:
        if atr <= 0:
            raise ValueError(f"atr must be positive, got {atr}")
        if atr_multiplier <= 0:
            raise ValueError(f"atr_multiplier must be positive, got {atr_multiplier}")
    if point_value <= 0:
        raise ValueError(f"point_value must be positive, got {point_value}")

    risk_pct = self.get_risk_pct()
    risk_pct = min(risk_pct, self._MAX_RISK_PCT)

    risk_amount = account_equity * (risk_pct / 100.0)
    stop_distance = stop_distance_override if stop_distance_override is not None else atr * atr_multiplier

    raw_lot = risk_amount / (stop_distance * point_value)

    lot_size = max(self._min_lot, min(raw_lot, self._max_lot))

    return PositionSize(
        lot_size=round(lot_size, 2),
        risk_pct=risk_pct,
        risk_amount=risk_amount,
        stop_distance=stop_distance,
        phase=self.get_phase(),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd .worktrees/spread-commission && python -m pytest tests/test_risk.py::TestAdaptivePositionSizer -v`

Expected: ALL PASS (new tests + existing tests)

- [ ] **Step 5: Commit**

```bash
git add src/risk/position_sizer.py tests/test_risk.py
git commit -m "feat: add stop_distance_override to position sizer"
```

---

### Task 2: TradeManager — Use Actual Stop Distance

**Files:**
- Modify: `src/risk/trade_manager.py:124-178`
- Test: `tests/test_risk.py`

- [ ] **Step 1: Write failing test for actual stop distance in open_trade**

Add to `tests/test_risk.py`:

```python
class TestTradeManagerStopDistance:

    def test_open_trade_uses_actual_stop_distance(self):
        """open_trade should size using abs(entry - stop), not atr * multiplier."""
        tm = _make_trade_manager(balance=10_000.0)
        # entry=2000, stop=1990 → actual stop distance = 10
        # 1.5% risk on 10K = $150, lot = 150 / (10 * 1.0) = 15.0 → clamped to 10.0
        # (trade manager helper uses point_value=1.0 implicitly via default sizer)
        _id, _trade, pos = tm.open_trade(
            entry_price=2000.0,
            stop_loss=1990.0,
            take_profit=2020.0,
            direction="long",
            atr=5.0,           # atr=5 * multiplier=1.5 = 7.5 (OLD would give 20.0→clamped 10.0)
            point_value=1.0,
            account_equity=10_000.0,
            atr_multiplier=1.5,
        )
        # Actual stop distance = 10.0, so lot = 150 / (10 * 1.0) = 15.0 → clamped 10.0
        assert pos.stop_distance == pytest.approx(10.0)

    def test_open_trade_short_uses_actual_stop_distance(self):
        """Short trade: stop above entry, sizer uses abs(entry - stop)."""
        tm = _make_trade_manager(balance=10_000.0)
        # entry=2000, stop=2005 → actual stop = 5
        _id, _trade, pos = tm.open_trade(
            entry_price=2000.0,
            stop_loss=2005.0,
            take_profit=1990.0,
            direction="short",
            atr=3.0,
            point_value=100.0,
            account_equity=10_000.0,
            atr_multiplier=1.5,
        )
        # Actual stop = 5.0, risk = 150, lot = 150 / (5 * 100) = 0.3
        assert pos.stop_distance == pytest.approx(5.0)
        assert pos.lot_size == pytest.approx(0.3, rel=1e-2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd .worktrees/spread-commission && python -m pytest tests/test_risk.py::TestTradeManagerStopDistance -v`

Expected: FAIL — `pos.stop_distance` will be `7.5` (atr*mult) instead of `10.0`

- [ ] **Step 3: Modify `open_trade()` to pass actual stop distance**

In `src/risk/trade_manager.py`, modify the `open_trade` method. Replace lines 157-163:

```python
        actual_stop_distance = abs(entry_price - stop_loss)
        pos = self._sizer.calculate_position_size(
            account_equity=account_equity,
            atr=atr,
            atr_multiplier=atr_multiplier,
            point_value=point_value,
            instrument=instrument,
            stop_distance_override=actual_stop_distance,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd .worktrees/spread-commission && python -m pytest tests/test_risk.py -v`

Expected: ALL PASS (new + existing)

- [ ] **Step 5: Commit**

```bash
git add src/risk/trade_manager.py tests/test_risk.py
git commit -m "fix: position sizing uses actual stop distance instead of atr * multiplier"
```

---

### Task 3: Trading Costs Config

**Files:**
- Modify: `config/edges.yaml`
- Modify: `src/backtesting/vectorbt_engine.py:133-159` (`__init__`)

- [ ] **Step 1: Add `trading_costs` section to edges.yaml**

Append to the end of `config/edges.yaml`:

```yaml

# Trading costs — deducted from P&L per trade (round-turn)
trading_costs:
  commission_per_lot: 4.00     # USD per standard lot, round-turn (The5ers High Stakes)
  spread_points: 0.15          # USD per oz price penalty (15 MT5 points, conservative)
```

- [ ] **Step 2: Load cost config in IchimokuBacktester.__init__()**

In `src/backtesting/vectorbt_engine.py`, after `self._point_value = point_value` (line 148), add:

```python
        # Trading costs (default to 0.0 = no cost deduction for backward compat)
        costs = cfg.get("trading_costs", {})
        self._commission_per_lot = float(costs.get("commission_per_lot", 0.0))
        self._spread_points = float(costs.get("spread_points", 0.0))
```

- [ ] **Step 3: Commit**

```bash
git add config/edges.yaml src/backtesting/vectorbt_engine.py
git commit -m "feat: add trading_costs config section (spread + commission)"
```

---

### Task 4: Cost Deduction in P&L Methods

**Files:**
- Modify: `src/backtesting/vectorbt_engine.py:1177-1206` (`_update_balance_from_trade`, `_partial_pnl`)
- Create: `tests/test_trading_costs.py`

- [ ] **Step 1: Write failing tests for cost deduction**

Create `tests/test_trading_costs.py`:

```python
"""Tests for trading cost deduction in the backtester P&L methods."""
from __future__ import annotations

import pytest

from src.backtesting.vectorbt_engine import IchimokuBacktester


class TestUpdateBalanceFromTrade:
    """Test _update_balance_from_trade with and without costs."""

    def _make_backtester(self, commission: float = 0.0, spread: float = 0.0):
        config = {
            "trading_costs": {
                "commission_per_lot": commission,
                "spread_points": spread,
            }
        }
        return IchimokuBacktester(config=config, initial_balance=10_000.0)

    def test_zero_costs_matches_original_behavior(self):
        """With zero costs, P&L is unchanged from the original formula."""
        bt = self._make_backtester(commission=0.0, spread=0.0)
        # 10 point win, 0.1 lots, point_value=100, remaining=1.0
        # gross pnl = 10 * 0.1 * 100 * 1.0 = $100
        trade = {"pnl_points": 10.0, "lot_size": 0.1, "remaining_pct": 1.0}
        new_bal = bt._update_balance_from_trade(10_000.0, trade)
        assert new_bal == pytest.approx(10_100.0)

    def test_commission_deducted(self):
        """Commission is deducted: commission_per_lot * lot_size * remaining_pct."""
        bt = self._make_backtester(commission=4.0, spread=0.0)
        # gross pnl = 10 * 0.1 * 100 * 1.0 = $100
        # commission = 4.0 * 0.1 * 1.0 = $0.40
        # net = 10_000 + 100 - 0.40 = 10_099.60
        trade = {"pnl_points": 10.0, "lot_size": 0.1, "remaining_pct": 1.0}
        new_bal = bt._update_balance_from_trade(10_000.0, trade)
        assert new_bal == pytest.approx(10_099.60)

    def test_spread_deducted(self):
        """Spread cost: spread_points * lot_size * point_value * remaining_pct."""
        bt = self._make_backtester(commission=0.0, spread=0.15)
        # gross pnl = 10 * 0.1 * 100 * 1.0 = $100
        # spread = 0.15 * 0.1 * 100 * 1.0 = $1.50
        # net = 10_000 + 100 - 1.50 = 10_098.50
        trade = {"pnl_points": 10.0, "lot_size": 0.1, "remaining_pct": 1.0}
        new_bal = bt._update_balance_from_trade(10_000.0, trade)
        assert new_bal == pytest.approx(10_098.50)

    def test_both_costs_deducted(self):
        """Both commission and spread are deducted from P&L."""
        bt = self._make_backtester(commission=4.0, spread=0.15)
        # gross pnl = $100
        # commission = $0.40, spread = $1.50, total cost = $1.90
        # net = 10_000 + 100 - 1.90 = 10_098.10
        trade = {"pnl_points": 10.0, "lot_size": 0.1, "remaining_pct": 1.0}
        new_bal = bt._update_balance_from_trade(10_000.0, trade)
        assert new_bal == pytest.approx(10_098.10)

    def test_costs_on_partial_exit(self):
        """Partial exit (50%) gets proportional costs."""
        bt = self._make_backtester(commission=4.0, spread=0.15)
        # gross pnl = 10 * 0.1 * 100 * 0.5 = $50
        # commission = 4.0 * 0.1 * 0.5 = $0.20
        # spread = 0.15 * 0.1 * 100 * 0.5 = $0.75
        # net = 10_000 + 50 - 0.20 - 0.75 = 10_049.05
        trade = {"pnl_points": 10.0, "lot_size": 0.1, "remaining_pct": 0.5}
        new_bal = bt._update_balance_from_trade(10_000.0, trade)
        assert new_bal == pytest.approx(10_049.05)

    def test_costs_on_losing_trade(self):
        """Costs make losses worse."""
        bt = self._make_backtester(commission=4.0, spread=0.15)
        # gross pnl = -5 * 0.1 * 100 * 1.0 = -$50
        # commission = $0.40, spread = $1.50
        # net = 10_000 - 50 - 1.90 = 9_948.10
        trade = {"pnl_points": -5.0, "lot_size": 0.1, "remaining_pct": 1.0}
        new_bal = bt._update_balance_from_trade(10_000.0, trade)
        assert new_bal == pytest.approx(9_948.10)


class TestPartialPnl:
    """Test _partial_pnl with and without costs."""

    def _make_backtester(self, commission: float = 0.0, spread: float = 0.0):
        config = {
            "trading_costs": {
                "commission_per_lot": commission,
                "spread_points": spread,
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
        # exit at 2010, close 50%: (2010 - 2000) * 0.1 * 100 * 0.5 = $50
        pnl = bt._partial_pnl(trade, exit_price=2010.0, close_pct=0.5)
        assert pnl == pytest.approx(50.0)

    def test_partial_pnl_with_costs(self):
        """Costs deducted from partial pnl."""
        bt = self._make_backtester(commission=4.0, spread=0.15)
        trade = self._make_trade(direction="long", entry=2000.0, lot=0.1)
        # gross = (2010-2000) * 0.1 * 100 * 0.5 = $50
        # spread = 0.15 * 0.1 * 100 * 0.5 = $0.75
        # commission = 4.0 * 0.1 * 0.5 = $0.20
        # net = 50 - 0.75 - 0.20 = $49.05
        pnl = bt._partial_pnl(trade, exit_price=2010.0, close_pct=0.5)
        assert pnl == pytest.approx(49.05)

    def test_partial_pnl_short_with_costs(self):
        """Short trade partial exit with costs."""
        bt = self._make_backtester(commission=4.0, spread=0.15)
        trade = self._make_trade(direction="short", entry=2000.0, lot=0.1)
        # gross = (2000-1990) * 0.1 * 100 * 0.5 = $50
        # costs = $0.75 + $0.20 = $0.95
        # net = $49.05
        pnl = bt._partial_pnl(trade, exit_price=1990.0, close_pct=0.5)
        assert pnl == pytest.approx(49.05)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd .worktrees/spread-commission && python -m pytest tests/test_trading_costs.py -v`

Expected: FAIL — costs are not deducted, so commission/spread tests will see wrong values

- [ ] **Step 3: Implement cost deduction in `_update_balance_from_trade`**

In `src/backtesting/vectorbt_engine.py`, replace the `_update_balance_from_trade` method (lines 1177-1196):

```python
    def _update_balance_from_trade(
        self, balance: float, trade_summary: dict
    ) -> float:
        """Apply trade P&L (minus trading costs) to the running balance."""
        pnl_points = float(trade_summary.get("pnl_points") or 0.0)
        lot_size = float(trade_summary.get("lot_size") or 0.0)
        remaining_pct = float(trade_summary.get("remaining_pct") or 1.0)

        # Gross P&L = points × lot_size × point_value × remaining fraction
        monetary_pnl = pnl_points * lot_size * self._point_value * remaining_pct

        # Trading costs (spread + commission), proportional to fraction closed
        spread_cost = self._spread_points * lot_size * self._point_value * remaining_pct
        commission_cost = self._commission_per_lot * lot_size * remaining_pct

        new_balance = balance + monetary_pnl - spread_cost - commission_cost

        # Prevent balance from going negative (margin call equivalent)
        new_balance = max(new_balance, 0.01)

        # Update the sizer so phase transitions are tracked correctly
        self.trade_manager._sizer.update_balance(new_balance)
        return new_balance
```

- [ ] **Step 4: Implement cost deduction in `_partial_pnl`**

In `src/backtesting/vectorbt_engine.py`, replace the `_partial_pnl` method (lines 1198-1206):

```python
    def _partial_pnl(
        self, trade: ActiveTrade, exit_price: float, close_pct: float
    ) -> float:
        """Calculate monetary P&L for a partial exit, net of trading costs."""
        if trade.direction == "long":
            pnl_points = exit_price - trade.entry_price
        else:
            pnl_points = trade.entry_price - exit_price
        gross_pnl = pnl_points * trade.lot_size * self._point_value * close_pct
        spread_cost = self._spread_points * trade.lot_size * self._point_value * close_pct
        commission_cost = self._commission_per_lot * trade.lot_size * close_pct
        return gross_pnl - spread_cost - commission_cost
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd .worktrees/spread-commission && python -m pytest tests/test_trading_costs.py tests/test_risk.py -v`

Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add src/backtesting/vectorbt_engine.py tests/test_trading_costs.py
git commit -m "feat: deduct spread and commission from backtest P&L"
```

---

### Task 5: Smoke Test — Full Backtest Still Runs

**Files:** None (verification only)

- [ ] **Step 1: Run existing test suite**

Run: `cd .worktrees/spread-commission && python -m pytest tests/ -v --timeout=60`

Expected: ALL PASS — zero-cost defaults mean existing tests are unaffected, new tests validate cost logic

- [ ] **Step 2: Run a quick backtest if data is available**

Run: `cd .worktrees/spread-commission && python -c "from src.backtesting.vectorbt_engine import IchimokuBacktester; print('Import OK')"`

Expected: No import errors

- [ ] **Step 3: Final commit (if any fixups needed)**

Only if smoke test revealed issues that needed fixing.
