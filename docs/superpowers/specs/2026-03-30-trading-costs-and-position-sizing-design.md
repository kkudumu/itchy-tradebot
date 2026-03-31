# Trading Costs & Position Sizing Fix

**Date:** 2026-03-30
**Branch:** `feature/spread-commission`
**Worktree:** `.worktrees/spread-commission`

## Problem

Two issues are producing unrealistic backtest results:

1. **No trading costs modeled.** The backtester calculates P&L as pure price difference — no spread, no commission. The optimizer sees inflated returns and cannot reject strategies that don't overcome real-world friction.

2. **Position sizer uses wrong stop distance.** The sizer calculates `signal.atr * atr_multiplier` as stop distance, but each strategy sets `signal.atr` to different things:
   - **Ichimoku:** 15M ATR → `ATR × 2.5` = correct (matches actual SL placement)
   - **Asian Breakout:** `asian_high - asian_low` (the full range) → sizer computes `range × 2.5`, but the actual stop is just `range` → **lots 2.5× too small**
   - **EMA Pullback:** 5M ATR → sizer computes `ATR × 2.5`, but actual stop is `high - low` of the breakout bar → **lot size doesn't match actual risk**

   Result: 0.5% risk config produces ~0.075-0.2% actual risk. Account is dramatically under-risked.

## The5ers High Stakes Cost Data

| Component | Per Standard Lot (round-turn) |
|-----------|-------------------------------|
| Commission | $4.00 |
| Spread (London/NY typical) | ~$12.50 (12.5 pts = $0.125/oz) |
| Model | Raw spread + separate commission |

Source: The5ers asset specifications, TheTrustedProp review, TradingFinder review (March 2026).

## Design

### 1. Trading Costs Config

Add to `config/edges.yaml`:

```yaml
trading_costs:
  commission_per_lot: 4.00     # USD per standard lot, round-turn (The5ers)
  spread_points: 0.15          # USD per oz, conservative estimate (15 MT5 points)
```

Units: `spread_points` is in price units (same as `pnl_points` in the backtester), i.e., dollars per ounce. This is consistent with how `_update_balance_from_trade()` computes monetary P&L.

### 2. Cost Deduction in `_update_balance_from_trade()`

**Where:** `src/backtesting/vectorbt_engine.py`, method `_update_balance_from_trade()`

**Current:**
```python
monetary_pnl = pnl_points * lot_size * self._point_value * remaining_pct
new_balance = balance + monetary_pnl
```

**New:**
```python
monetary_pnl = pnl_points * lot_size * self._point_value * remaining_pct

# Deduct trading costs (spread + commission) proportional to the fraction closed
spread_cost = self._spread_points * lot_size * self._point_value * remaining_pct
commission_cost = self._commission_per_lot * lot_size * remaining_pct
total_cost = spread_cost + commission_cost

new_balance = balance + monetary_pnl - total_cost
```

**Why deduct at close, not open:**
- Simpler — single deduction point, no need to track "entry cost already applied"
- Spread is a round-turn cost (you pay it once on the full trade)
- Commission is round-turn ($4 covers both open and close)
- The partial exit at 2R correctly applies `remaining_pct` to costs too

**Also apply to `_partial_exit_pnl()`:**
```python
def _partial_exit_pnl(self, trade, exit_price, close_pct):
    pnl_points = (exit_price - trade.entry_price) if trade.direction == "long" else (trade.entry_price - exit_price)
    gross_pnl = pnl_points * trade.lot_size * self._point_value * close_pct
    spread_cost = self._spread_points * trade.lot_size * self._point_value * close_pct
    commission_cost = self._commission_per_lot * trade.lot_size * close_pct
    return gross_pnl - spread_cost - commission_cost
```

### 3. Position Sizer Fix: Use Actual Stop Distance

**Where:** `src/backtesting/vectorbt_engine.py` (call site) and `src/risk/trade_manager.py`

**Current flow:**
```
signal.atr → trade_manager.open_trade(atr=signal.atr) → sizer.calculate_position_size(atr=atr, atr_multiplier=2.5)
    → stop_distance = atr * atr_multiplier   # WRONG for non-Ichimoku strategies
    → lot_size = risk_amount / (stop_distance * point_value)
```

**New flow:**
```
signal.entry_price, signal.stop_loss → actual_stop_distance = abs(entry - stop_loss)
    → lot_size = risk_amount / (actual_stop_distance * point_value)
```

**Changes to `AdaptivePositionSizer.calculate_position_size()`:**

Add a new parameter `stop_distance_override: float | None = None`. When provided, use it directly instead of computing `atr * atr_multiplier`.

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
    risk_pct = self.get_risk_pct()
    risk_pct = min(risk_pct, self._MAX_RISK_PCT)
    risk_amount = account_equity * (risk_pct / 100.0)

    stop_distance = stop_distance_override if stop_distance_override is not None else atr * atr_multiplier

    raw_lot = risk_amount / (stop_distance * point_value)
    lot_size = max(self._min_lot, min(raw_lot, self._max_lot))
    ...
```

**Changes to `TradeManager.open_trade()`:**

Compute `actual_stop = abs(entry_price - stop_loss)` and pass as `stop_distance_override`.

**Changes to vectorbt_engine.py call site (~line 631):**

No change needed — `entry_price` and `stop_loss` are already passed to `open_trade()`.

### 4. Config Loading

**Where:** `src/backtesting/vectorbt_engine.py`, `__init__()` method

Read cost config and store as instance attributes:

```python
costs = config.get("trading_costs", {})
self._commission_per_lot = costs.get("commission_per_lot", 0.0)
self._spread_points = costs.get("spread_points", 0.0)
```

Default to 0.0 so existing tests that don't pass cost config continue to work unchanged.

## What Changes

| File | Change |
|------|--------|
| `config/edges.yaml` | Add `trading_costs` section |
| `src/risk/position_sizer.py` | Add `stop_distance_override` param |
| `src/risk/trade_manager.py` | Compute actual stop distance, pass to sizer |
| `src/backtesting/vectorbt_engine.py` | Load cost config, deduct costs in P&L methods |

## What Doesn't Change

- Signal generation, edge filters, exit logic — untouched
- Optimization loop — it already reads metrics from backtest results; no changes needed
- SpreadFilter edge — separate concern (gates live entries on real-time spread)
- R-multiple calculations in exit_manager — still based on raw price movement
- Existing tests — cost defaults to 0.0, sizer falls back to atr * multiplier when no override

## Expected Impact

### Position Sizing (at 0.5% risk, $10K account)

| Strategy | Current Lot Size | Fixed Lot Size | Actual Risk |
|----------|-----------------|---------------|-------------|
| Ichimoku (15M ATR $7, SL $17.50) | 0.03 | 0.03 | 0.5% ✓ |
| Asian Breakout (range $20, SL $20) | 0.01 | 0.025 | 0.5% ✓ |
| EMA Pullback (5M ATR $3, SL $5) | 0.03 | 0.10 | 0.5% ✓ |

### Cost Impact (103 trades, estimated avg 0.05 lots)

| Component | Per Trade | Total |
|-----------|----------|-------|
| Commission | $0.20 | $20.60 |
| Spread | $0.75 | $77.25 |
| **Total friction** | **$0.95** | **$97.85** |

On $721 gross profit, costs consume ~$98 (13.6%). Not the primary issue — the 8.7% win rate is — but the optimizer now sees accurate net P&L and will reject marginal strategies.

## Testing

- Existing position sizer tests should still pass (no override = old behavior)
- Add test: sizer with `stop_distance_override` produces correct lot size
- Add test: `_update_balance_from_trade` deducts costs correctly
- Add test: zero-cost config produces identical results to current behavior
