# The5ers 2-Step System Overhaul - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retarget the trading system from Sharpe-based optimization to passing The5ers 2-Step High Stakes challenge with 3 strategies, multi-phase prop firm tracking, challenge simulation, and a live optimization dashboard.

**Architecture:** Clean rewrite of signal layer into a strategy plugin system. Three entry strategies (Ichimoku 15M, Asian Range Breakout, EMA State Machine) feed a signal blender that picks the highest-confluence signal. The backtest engine runs the blended signals through shared risk management and exit logic. A challenge simulator runs rolling windows + Monte Carlo on backtest results. The optimization loop targets challenge pass rate instead of Sharpe.

**Tech Stack:** Python 3.13, pandas, numpy, vectorbt (existing), stdlib HTTP server (dashboard), PyYAML

**Spec:** `docs/superpowers/specs/2026-03-29-the5ers-system-overhaul-design.md`

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `src/strategy/strategies/asian_breakout.py` | Asian Range Breakout strategy (Strategy subclass) |
| `src/strategy/strategies/ema_pullback.py` | EMA Pullback State Machine strategy (Strategy subclass) |
| `src/strategy/signal_blender.py` | Multi-strategy signal coordinator |
| `src/backtesting/challenge_simulator.py` | Rolling window + Monte Carlo challenge simulation |
| `src/backtesting/optimization_dashboard.py` | Live HTTP dashboard server |
| `src/backtesting/optimization_dashboard.html` | Dashboard HTML/JS (self-contained) |
| `tests/test_asian_breakout.py` | Asian Range Breakout tests |
| `tests/test_ema_pullback.py` | EMA Pullback State Machine tests |
| `tests/test_signal_blender.py` | Signal blender tests |
| `tests/test_challenge_simulator.py` | Challenge simulator tests |
| `tests/test_multi_phase_tracker.py` | Multi-phase prop firm tracker tests |

### Modified Files
| File | Change |
|------|--------|
| `config/strategy.yaml` | Add `prop_firm` section, add `strategies.asian_breakout` and `strategies.ema_pullback` |
| `config/optimization_loop.yaml` | Replace `target_sharpe` with `target_pass_rate`, add rolling window config |
| `src/backtesting/metrics.py` | Rewrite `PropFirmTracker` as `MultiPhasePropFirmTracker` |
| `src/strategy/strategies/ichimoku.py` | Simplify: drop 4H/1H hard gates, use 15M cloud as trend filter |
| `src/backtesting/vectorbt_engine.py` | Rewrite main loop to use signal blender + strategy plugins |
| `src/backtesting/results_exporter.py` | Add challenge simulation data to exports |
| `scripts/run_optimization_loop.py` | Retarget from Sharpe to pass rate, new Claude prompt |
| `src/strategy/confluence_scorer.py` | Generalize scoring for multi-strategy (strategy-provided scores) |
| `config/edges.yaml` | Enable session-relevant edges (time_of_day, friday_close) |

---

## Task 1: Config - Prop Firm Rules + Multi-Strategy Parameters

**Files:**
- Modify: `config/strategy.yaml`
- Modify: `config/optimization_loop.yaml`
- Modify: `config/edges.yaml`

- [ ] **Step 1: Update strategy.yaml with prop firm rules and new strategy configs**

```yaml
# Append to end of config/strategy.yaml:

prop_firm:
  name: the5ers_2step_high_stakes
  account_size: 10000
  leverage: 100
  phase_1:
    profit_target_pct: 8.0
    max_loss_pct: 10.0
    daily_loss_pct: 5.0
    time_limit_days: 0  # 0 = unlimited
  phase_2:
    profit_target_pct: 5.0
    max_loss_pct: 10.0
    daily_loss_pct: 5.0
    time_limit_days: 0
  funded:
    monthly_target_pct: 10.0
    max_loss_pct: 10.0
    daily_loss_pct: 5.0

strategies:
  ichimoku:
    # Existing ichimoku config stays the same but:
    ichimoku:
      tenkan_period: 9
      kijun_period: 26
      senkou_b_period: 52
    adx:
      period: 14
      threshold: 20  # Lowered from 28 for more signals
    atr:
      period: 14
      stop_multiplier: 2.5
    signal:
      min_confluence_score: 2  # Lowered from 4 - now using 15M trend only
      tier_a_plus: 7
      tier_b: 5
      tier_c: 2
      timeframes:
        - "15M"
        - "5M"  # Dropped 4H and 1H - using 15M as trend filter

  asian_breakout:
    enabled: true
    weight: 1.0
    asian_session_start_utc: "21:00"
    asian_session_end_utc: "06:00"
    london_entry_start_utc: "06:00"
    london_entry_end_utc: "10:00"
    min_range_pips: 20
    max_range_pips: 80
    rr_ratio: 2.0
    atr_period: 14

  ema_pullback:
    enabled: true
    weight: 1.0
    fast_ema: 14
    mid_ema: 18
    slow_ema: 24
    atr_period: 14
    min_ema_angle_deg: 30
    max_ema_angle_deg: 95
    pullback_candles_min: 1
    pullback_candles_max: 3
    breakout_window_bars: 20
    rr_ratio: 2.0

# Update risk section:
risk:
  initial_risk_pct: 1.5
  reduced_risk_pct: 1.0
  phase_threshold_pct: 4.0
  daily_circuit_breaker_pct: 4.5  # Raised from 2.0 - buffer below 5% The5ers limit
  max_concurrent_positions: 1

# Update exit section:
exit:
  strategy: hybrid_50_50
  tp_r_multiple: 1.5
  trail_type: kijun
  breakeven_threshold_r: 1.0
  kijun_trail_start_r: 1.5
  higher_tf_kijun_start_r: 3.0

active_strategies:
  - ichimoku
  - asian_breakout
  - ema_pullback
```

- [ ] **Step 2: Update optimization_loop.yaml**

Replace the full file content:

```yaml
optimization:
  max_iterations: 10
  target_pass_rate: 0.50
  rolling_window_spacing_days: 22
  max_changes_per_iteration: 2
  plateau_threshold: 0.05
  plateau_iterations: 3

persistence:
  enabled: true
  reports_dir: "reports"

claude:
  command: ["claude", "-p", "--dangerously-skip-permissions"]
  timeout_seconds: 300
  encoding: utf-8

validation:
  run_full_dataset: true
```

- [ ] **Step 3: Enable session-relevant edges in edges.yaml**

Set `enabled: true` for these edges:
- `time_of_day` with params: `start_utc: "06:00"`, `end_utc: "21:00"` (London+NY)
- `friday_close` with params: `close_hour_utc: 20` (close positions before weekend)

- [ ] **Step 4: Commit**

```bash
git add config/strategy.yaml config/optimization_loop.yaml config/edges.yaml
git commit -m "config: add The5ers prop firm rules, multi-strategy params, enable session edges"
```

---

## Task 2: Multi-Phase PropFirmTracker

**Files:**
- Create: `tests/test_multi_phase_tracker.py`
- Modify: `src/backtesting/metrics.py`

- [ ] **Step 1: Write failing tests for multi-phase tracker**

```python
# tests/test_multi_phase_tracker.py
"""Tests for MultiPhasePropFirmTracker - The5ers 2-Step High Stakes."""

import datetime as dt
from src.backtesting.metrics import MultiPhasePropFirmTracker


def _utc(year, month, day, hour=0):
    return dt.datetime(year, month, day, hour, tzinfo=dt.timezone.utc)


class TestPhase1:
    def test_starts_in_phase_1(self):
        tracker = MultiPhasePropFirmTracker()
        tracker.initialise(10_000.0, _utc(2024, 1, 1))
        status = tracker.get_status()
        assert status["phase"] == "phase_1_active"
        assert status["profit_pct"] == 0.0

    def test_phase_1_passes_at_8_percent(self):
        tracker = MultiPhasePropFirmTracker()
        tracker.initialise(10_000.0, _utc(2024, 1, 1))
        # Equity grows to $10,800
        tracker.update(_utc(2024, 1, 15), 10_800.0)
        status = tracker.get_status()
        assert status["phase"] == "phase_2_active"
        assert status["phase_1_passed"] is True

    def test_phase_1_fails_on_total_dd(self):
        tracker = MultiPhasePropFirmTracker()
        tracker.initialise(10_000.0, _utc(2024, 1, 1))
        tracker.update(_utc(2024, 1, 2), 8_900.0)  # -11% total DD
        status = tracker.get_status()
        assert status["phase"] == "failed_phase_1"
        assert "total_dd" in status["failure_reason"]

    def test_phase_1_fails_on_daily_dd(self):
        tracker = MultiPhasePropFirmTracker(
            phase_1_daily_loss_pct=5.0,
        )
        tracker.initialise(10_000.0, _utc(2024, 1, 1))
        # Day 1 close at $10,000, Day 2 drops to $9,400 (-6% from prev day equity)
        tracker.update(_utc(2024, 1, 1, 23), 10_000.0)
        tracker.update(_utc(2024, 1, 2, 12), 9_400.0)
        status = tracker.get_status()
        assert status["phase"] == "failed_phase_1"
        assert "daily_dd" in status["failure_reason"]


class TestPhase2:
    def test_phase_2_resets_balance_to_10k(self):
        tracker = MultiPhasePropFirmTracker()
        tracker.initialise(10_000.0, _utc(2024, 1, 1))
        tracker.update(_utc(2024, 1, 15), 10_800.0)  # Phase 1 passed
        status = tracker.get_status()
        assert status["phase"] == "phase_2_active"
        assert status["phase_balance"] == 10_000.0  # Reset

    def test_phase_2_passes_at_5_percent(self):
        tracker = MultiPhasePropFirmTracker()
        tracker.initialise(10_000.0, _utc(2024, 1, 1))
        tracker.update(_utc(2024, 1, 15), 10_800.0)  # Phase 1
        tracker.update(_utc(2024, 2, 1), 10_500.0)   # Phase 2 +5%
        status = tracker.get_status()
        assert status["phase"] == "funded_active"

    def test_phase_2_fails_on_total_dd(self):
        tracker = MultiPhasePropFirmTracker()
        tracker.initialise(10_000.0, _utc(2024, 1, 1))
        tracker.update(_utc(2024, 1, 15), 10_800.0)  # Phase 1
        tracker.update(_utc(2024, 2, 1), 8_900.0)    # -11% from phase 2 start
        status = tracker.get_status()
        assert status["phase"] == "failed_phase_2"


class TestFunded:
    def test_funded_tracks_monthly_returns(self):
        tracker = MultiPhasePropFirmTracker()
        tracker.initialise(10_000.0, _utc(2024, 1, 1))
        tracker.update(_utc(2024, 1, 10), 10_800.0)  # Phase 1
        tracker.update(_utc(2024, 1, 20), 10_500.0)  # Phase 2
        # Funded: simulate 3 months
        tracker.update(_utc(2024, 2, 28), 11_000.0)  # +10% month 1
        tracker.update(_utc(2024, 3, 31), 11_500.0)  # +4.5% month 2
        tracker.update(_utc(2024, 4, 30), 12_200.0)  # +6.1% month 3
        status = tracker.get_status()
        assert status["phase"] == "funded_active"
        assert len(status["funded_monthly_returns"]) == 3
        assert status["funded_monthly_returns"][0] > 9.0  # ~10%


class TestDailyDDFormula:
    def test_daily_dd_uses_max_of_equity_and_balance(self):
        """The5ers: daily_dd_limit = 5% * MAX(prev_close_equity, prev_close_balance)"""
        tracker = MultiPhasePropFirmTracker(
            phase_1_daily_loss_pct=5.0,
        )
        tracker.initialise(10_000.0, _utc(2024, 1, 1))
        # Day 1: equity closes at $10,300 (floating profit)
        tracker.update(_utc(2024, 1, 1, 21), 10_300.0)
        # Day 2: drops from $10,300 ref to $9,750 = -5.34% daily DD
        tracker.update(_utc(2024, 1, 2, 12), 9_750.0)
        status = tracker.get_status()
        # Should fail because 5.34% > 5.0% daily limit
        # Reference point is max(10300, 10000) = 10300
        # DD = (10300 - 9750) / 10300 = 5.34%
        assert status["phase"] == "failed_phase_1"


class TestPhaseTransitionHardReset:
    def test_dd_tracking_resets_between_phases(self):
        tracker = MultiPhasePropFirmTracker()
        tracker.initialise(10_000.0, _utc(2024, 1, 1))
        # Phase 1: some drawdown before passing
        tracker.update(_utc(2024, 1, 5), 9_500.0)   # -5% DD
        tracker.update(_utc(2024, 1, 15), 10_800.0)  # Pass phase 1
        status = tracker.get_status()
        # Phase 2 should have 0 DD, fresh start
        assert status["max_total_dd_pct"] == 0.0
        assert status["phase"] == "phase_2_active"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_multi_phase_tracker.py -v
```
Expected: FAIL (MultiPhasePropFirmTracker not found)

- [ ] **Step 3: Implement MultiPhasePropFirmTracker**

Add the new class to `src/backtesting/metrics.py`. Keep the existing `PropFirmTracker` for backwards compatibility. The new class tracks:
- Current phase (`phase_1_active`, `phase_2_active`, `funded_active`, `failed_*`)
- Per-phase balance (resets to account_size on transition)
- Daily DD using The5ers formula: `5% * MAX(prev_day_closing_equity, prev_day_closing_balance)`
- Total DD from phase start balance
- Funded monthly returns (calendar month buckets)
- Phase transition timestamps

Key methods:
```python
class MultiPhasePropFirmTracker:
    def __init__(
        self,
        account_size: float = 10_000.0,
        phase_1_profit_target_pct: float = 8.0,
        phase_1_max_loss_pct: float = 10.0,
        phase_1_daily_loss_pct: float = 5.0,
        phase_2_profit_target_pct: float = 5.0,
        phase_2_max_loss_pct: float = 10.0,
        phase_2_daily_loss_pct: float = 5.0,
        funded_monthly_target_pct: float = 10.0,
        funded_max_loss_pct: float = 10.0,
        funded_daily_loss_pct: float = 5.0,
    ): ...

    def initialise(self, starting_balance: float, start_time: datetime) -> None: ...
    def update(self, timestamp: datetime, equity: float) -> None: ...
    def get_status(self) -> dict: ...
    def force_close_and_transition(self, equity: float, timestamp: datetime) -> None: ...
```

The `update()` method:
1. Check if phase already terminal (failed/bust) -> return early
2. Track day boundary: if new day, compute prev_day_ref = max(prev_day_closing_equity, prev_day_closing_balance)
3. Check daily DD: `(prev_day_ref - equity) / prev_day_ref >= daily_loss_pct/100` -> fail
4. Check total DD: `(phase_start_balance - equity) / phase_start_balance >= max_loss_pct/100` -> fail
5. Check profit target: `(equity - phase_start_balance) / phase_start_balance >= target_pct/100` -> transition
6. If transitioning: reset phase_start_balance, reset DD tracking, advance phase

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_multi_phase_tracker.py -v
```
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_multi_phase_tracker.py src/backtesting/metrics.py
git commit -m "feat: add MultiPhasePropFirmTracker for The5ers 2-Step challenge"
```

---

## Task 3: Ichimoku Strategy - Simplify to 15M Trend Filter

**Files:**
- Modify: `src/strategy/strategies/ichimoku.py`
- Modify: `tests/test_signal_engine.py`

- [ ] **Step 1: Write test for 15M-only Ichimoku signal generation**

Add to `tests/test_signal_engine.py`:

```python
class TestIchimoku15MMode:
    """Test Ichimoku strategy with 15M cloud as trend filter (no 4H/1H gates)."""

    def test_generates_signal_with_15m_trend_only(self):
        """With 15M cloud bullish + 5M pullback to Kijun, should produce a long signal
        even without 4H/1H alignment."""
        from src.strategy.strategies.ichimoku import IchimokuStrategy
        from src.strategy.base import EvalMatrix, EvaluatorResult

        config = {
            "timeframes": ["15M", "5M"],
            "min_confluence_score": 2,
            "adx_threshold": 20,
        }
        strategy = IchimokuStrategy(config=config)

        # Build eval matrix with only 15M and 5M data (no 4H, no 1H)
        matrix = EvalMatrix()
        matrix.set("ichimoku_15M", EvaluatorResult(
            direction=1.0, confidence=0.8,
            metadata={"cloud_direction": 1, "tk_cross": 1, "cloud_position": 1, "chikou_confirmed": 1},
        ))
        matrix.set("ichimoku_5M", EvaluatorResult(
            direction=1.0, confidence=0.7,
            metadata={"kijun": 2050.0, "close": 2052.0, "atr": 5.0},
        ))
        matrix.set("adx_15M", EvaluatorResult(
            direction=0.0, confidence=0.0,
            metadata={"adx": 25.0, "trending": True},
        ))
        matrix.set("atr_15M", EvaluatorResult(
            direction=0.0, confidence=0.0,
            metadata={"atr": 8.0},
        ))
        matrix.set("session_5M", EvaluatorResult(
            direction=0.0, confidence=0.0,
            metadata={"session": "london"},
        ))

        signal = strategy.decide(matrix)
        assert signal is not None
        assert signal.direction == "long"

    def test_no_signal_when_15m_cloud_flat(self):
        from src.strategy.strategies.ichimoku import IchimokuStrategy
        from src.strategy.base import EvalMatrix, EvaluatorResult

        config = {"timeframes": ["15M", "5M"], "min_confluence_score": 2}
        strategy = IchimokuStrategy(config=config)

        matrix = EvalMatrix()
        matrix.set("ichimoku_15M", EvaluatorResult(
            direction=0.0, confidence=0.0,
            metadata={"cloud_direction": 0, "tk_cross": 0, "cloud_position": 0, "chikou_confirmed": 0},
        ))
        # ... (5M and other evaluators populated but 15M cloud is flat)

        signal = strategy.decide(matrix)
        assert signal is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_signal_engine.py::TestIchimoku15MMode -v
```
Expected: FAIL (current decide() requires 4H/1H evaluators)

- [ ] **Step 3: Modify IchimokuStrategy.decide() to support 15M-only mode**

In `src/strategy/strategies/ichimoku.py`, modify `decide()` (lines 88-228):

The key change: check `self._config.get("timeframes", ["4H", "1H", "15M", "5M"])`. If the list doesn't include "4H" and "1H", skip those gates and use 15M cloud direction as the primary trend filter.

```python
def decide(self, eval_matrix: EvalMatrix) -> Optional[Signal]:
    configured_tfs = self._config.get("timeframes", ["4H", "1H", "15M", "5M"])

    # Determine direction from the highest configured timeframe
    if "4H" in configured_tfs:
        # Original 4H -> 1H -> 15M cascade
        state_4h = eval_matrix.get("ichimoku_4H")
        if state_4h is None or state_4h.metadata.get("cloud_direction") == 0:
            return None
        direction = "long" if state_4h.metadata["cloud_direction"] == 1 else "short"
        # ... existing 1H confirmation logic ...
    else:
        # 15M-only mode: use 15M cloud as trend filter
        state_15m = eval_matrix.get("ichimoku_15M")
        if state_15m is None:
            return None
        cloud_dir = state_15m.metadata.get("cloud_direction", 0)
        if cloud_dir == 0:
            return None
        direction = "long" if cloud_dir == 1 else "short"

    # Continue with 5M entry logic (shared between both modes)
    # ... rest of entry logic using 5M Kijun proximity ...
```

Also update `required_evaluators` to be dynamic based on config timeframes, and reduce `warmup_bars` from 6240 to `kijun_period * 60` (= 1560) when in 15M mode.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_signal_engine.py -v
```
Expected: ALL PASS (old tests still work, new 15M tests pass)

- [ ] **Step 5: Commit**

```bash
git add src/strategy/strategies/ichimoku.py tests/test_signal_engine.py
git commit -m "feat: add 15M-only trend mode to Ichimoku strategy (drop 4H/1H gates)"
```

---

## Task 4: Asian Range Breakout Strategy

**Files:**
- Create: `src/strategy/strategies/asian_breakout.py`
- Create: `tests/test_asian_breakout.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_asian_breakout.py
"""Tests for Asian Range Breakout strategy."""

import datetime as dt
from src.strategy.strategies.asian_breakout import AsianBreakoutStrategy
from src.strategy.base import EvalMatrix, EvaluatorResult, Signal


def _utc(h, m=0):
    return dt.datetime(2024, 3, 15, h, m, tzinfo=dt.timezone.utc)


class TestAsianRangeDetection:
    def test_marks_asian_range_high_low(self):
        strategy = AsianBreakoutStrategy()
        # Feed bars during Asian session (21:00 - 06:00 UTC)
        strategy.on_bar(_utc(21), high=2050, low=2040, close=2045)
        strategy.on_bar(_utc(23), high=2055, low=2038, close=2050)
        strategy.on_bar(_utc(2), high=2052, low=2042, close=2048)
        strategy.on_bar(_utc(5), high=2053, low=2039, close=2046)

        assert strategy.asian_high == 2055.0
        assert strategy.asian_low == 2038.0
        assert strategy.asian_range_pips == 170  # (2055 - 2038) * 10 = 170 pips

    def test_rejects_range_too_narrow(self):
        strategy = AsianBreakoutStrategy(config={"min_range_pips": 20})
        strategy.on_bar(_utc(21), high=2050, low=2049, close=2049.5)
        strategy.on_bar(_utc(5), high=2050.5, low=2048.5, close=2049)
        # Range = 2 pips, below 20 minimum
        assert strategy.range_valid is False

    def test_rejects_range_too_wide(self):
        strategy = AsianBreakoutStrategy(config={"max_range_pips": 80})
        strategy.on_bar(_utc(21), high=2080, low=2040, close=2060)
        strategy.on_bar(_utc(5), high=2085, low=2035, close=2060)
        # Range = 500 pips, above 80 maximum
        assert strategy.range_valid is False


class TestLondonBreakoutSignal:
    def test_long_signal_on_break_above_asian_high(self):
        strategy = AsianBreakoutStrategy(config={
            "min_range_pips": 20, "max_range_pips": 500,
            "rr_ratio": 2.0,
        })
        # Set Asian range
        strategy.on_bar(_utc(21), high=2050, low=2040, close=2045)
        strategy.on_bar(_utc(5, 59), high=2050, low=2040, close=2045)
        # London session bar breaks above
        signal = strategy.on_bar(_utc(6, 5), high=2052, low=2044, close=2051)
        assert signal is not None
        assert signal.direction == "long"
        assert signal.entry_price == 2051.0
        assert signal.stop_loss == 2040.0  # Asian low
        assert signal.take_profit > signal.entry_price  # 2R from entry

    def test_short_signal_on_break_below_asian_low(self):
        strategy = AsianBreakoutStrategy(config={
            "min_range_pips": 20, "max_range_pips": 500,
            "rr_ratio": 2.0,
        })
        strategy.on_bar(_utc(21), high=2050, low=2040, close=2045)
        strategy.on_bar(_utc(5, 59), high=2050, low=2040, close=2045)
        signal = strategy.on_bar(_utc(6, 5), high=2042, low=2038, close=2039)
        assert signal is not None
        assert signal.direction == "short"
        assert signal.stop_loss == 2050.0  # Asian high

    def test_no_signal_outside_london_window(self):
        strategy = AsianBreakoutStrategy()
        strategy.on_bar(_utc(21), high=2050, low=2040, close=2045)
        strategy.on_bar(_utc(5, 59), high=2050, low=2040, close=2045)
        # Bar at 11:00 UTC (after London entry window closes at 10:00)
        signal = strategy.on_bar(_utc(11), high=2052, low=2044, close=2051)
        assert signal is None

    def test_only_one_signal_per_day(self):
        strategy = AsianBreakoutStrategy(config={
            "min_range_pips": 20, "max_range_pips": 500,
        })
        strategy.on_bar(_utc(21), high=2050, low=2040, close=2045)
        strategy.on_bar(_utc(5, 59), high=2050, low=2040, close=2045)
        sig1 = strategy.on_bar(_utc(6, 5), high=2052, low=2044, close=2051)
        assert sig1 is not None
        # Second breakout same day should not fire
        sig2 = strategy.on_bar(_utc(7), high=2055, low=2050, close=2054)
        assert sig2 is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_asian_breakout.py -v
```
Expected: FAIL (module not found)

- [ ] **Step 3: Implement AsianBreakoutStrategy**

```python
# src/strategy/strategies/asian_breakout.py
"""Asian Range Breakout strategy for XAU/USD.

Marks the Asian session (21:00-06:00 UTC) high/low, then trades
the breakout during London session (06:00-10:00 UTC).

Stop-loss: opposite side of Asian range.
Take-profit: entry + (risk * rr_ratio).
Max 1 signal per day.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional

from src.strategy.base import Signal


@dataclass
class _DayState:
    """Tracks state for a single trading day."""
    asian_high: float = 0.0
    asian_low: float = float("inf")
    range_locked: bool = False
    signal_fired: bool = False
    date: Optional[dt.date] = None


class AsianBreakoutStrategy:
    """Asian Range Breakout - structural edge on gold.

    Gold consolidates during Asian session, then breaks during London
    when European institutional flow arrives.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self._asian_start_hour = int(cfg.get("asian_session_start_utc", "21").split(":")[0])
        self._asian_end_hour = int(cfg.get("asian_session_end_utc", "06").split(":")[0])
        self._london_start_hour = int(cfg.get("london_entry_start_utc", "06").split(":")[0])
        self._london_end_hour = int(cfg.get("london_entry_end_utc", "10").split(":")[0])
        self._min_range_pips = float(cfg.get("min_range_pips", 20))
        self._max_range_pips = float(cfg.get("max_range_pips", 80))
        self._rr_ratio = float(cfg.get("rr_ratio", 2.0))
        self._weight = float(cfg.get("weight", 1.0))
        self._day = _DayState()

    @property
    def asian_high(self) -> float:
        return self._day.asian_high

    @property
    def asian_low(self) -> float:
        return self._day.asian_low

    @property
    def asian_range_pips(self) -> float:
        return (self._day.asian_high - self._day.asian_low) * 10  # Gold: 1 pip = $0.10

    @property
    def range_valid(self) -> bool:
        rng = self.asian_range_pips
        return self._min_range_pips <= rng <= self._max_range_pips

    def on_bar(
        self,
        timestamp: dt.datetime,
        high: float,
        low: float,
        close: float,
        atr: float = 0.0,
    ) -> Optional[Signal]:
        hour = timestamp.hour
        today = timestamp.date()

        # New day reset (Asian session starts at 21:00 previous calendar day)
        if self._day.date is None or (hour >= self._asian_start_hour and today != self._day.date):
            self._day = _DayState(date=today)

        # Asian session: accumulate range
        in_asian = (hour >= self._asian_start_hour or hour < self._asian_end_hour)
        if in_asian and not self._day.range_locked:
            self._day.asian_high = max(self._day.asian_high, high)
            if low < self._day.asian_low:
                self._day.asian_low = low
            return None

        # Lock range when Asian session ends
        if not self._day.range_locked and hour >= self._asian_end_hour:
            self._day.range_locked = True

        # London entry window
        if not (self._london_start_hour <= hour < self._london_end_hour):
            return None

        if self._day.signal_fired or not self._day.range_locked or not self.range_valid:
            return None

        # Check for breakout
        risk = self._day.asian_high - self._day.asian_low
        if close > self._day.asian_high:
            self._day.signal_fired = True
            return Signal(
                timestamp=timestamp,
                instrument="XAUUSD",
                direction="long",
                entry_price=close,
                stop_loss=self._day.asian_low,
                take_profit=close + risk * self._rr_ratio,
                confluence_score=int(self._weight * 5),
                quality_tier="B",
                atr=atr,
                reasoning={"strategy": "asian_breakout", "asian_high": self._day.asian_high,
                           "asian_low": self._day.asian_low, "range_pips": self.asian_range_pips},
            )
        elif close < self._day.asian_low:
            self._day.signal_fired = True
            return Signal(
                timestamp=timestamp,
                instrument="XAUUSD",
                direction="short",
                entry_price=close,
                stop_loss=self._day.asian_high,
                take_profit=close - risk * self._rr_ratio,
                confluence_score=int(self._weight * 5),
                quality_tier="B",
                atr=atr,
                reasoning={"strategy": "asian_breakout", "asian_high": self._day.asian_high,
                           "asian_low": self._day.asian_low, "range_pips": self.asian_range_pips},
            )

        return None
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_asian_breakout.py -v
```
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/strategy/strategies/asian_breakout.py tests/test_asian_breakout.py
git commit -m "feat: add Asian Range Breakout strategy for London session gold trading"
```

---

## Task 5: EMA Pullback State Machine Strategy

**Files:**
- Create: `src/strategy/strategies/ema_pullback.py`
- Create: `tests/test_ema_pullback.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ema_pullback.py
"""Tests for EMA Pullback State Machine strategy."""

import datetime as dt
import numpy as np
import pandas as pd
from src.strategy.strategies.ema_pullback import EMAPullbackStrategy


def _utc(h, m=0):
    return dt.datetime(2024, 3, 15, h, m, tzinfo=dt.timezone.utc)


def _make_bar(ts, o, h, l, c, ema_fast, ema_mid, ema_slow, atr=5.0):
    return {
        "timestamp": ts, "open": o, "high": h, "low": l, "close": c,
        "ema_fast": ema_fast, "ema_mid": ema_mid, "ema_slow": ema_slow, "atr": atr,
    }


class TestStateTransitions:
    def test_starts_in_scanning(self):
        strategy = EMAPullbackStrategy()
        assert strategy.state == "SCANNING"

    def test_scanning_to_armed_on_ema_alignment(self):
        strategy = EMAPullbackStrategy(config={"min_ema_angle_deg": 0})
        # EMAs aligned: fast > mid > slow (uptrend)
        strategy.on_bar(**_make_bar(_utc(8), 2050, 2055, 2048, 2053, 2053, 2050, 2045))
        # Pullback candle (close < open, bearish)
        strategy.on_bar(**_make_bar(_utc(8, 5), 2053, 2054, 2049, 2050, 2052, 2050, 2045))
        assert strategy.state == "ARMED"

    def test_armed_to_window_open_after_pullback(self):
        strategy = EMAPullbackStrategy(config={
            "min_ema_angle_deg": 0,
            "pullback_candles_min": 1,
            "pullback_candles_max": 3,
        })
        # Trend established
        strategy.on_bar(**_make_bar(_utc(8), 2050, 2055, 2048, 2053, 2053, 2050, 2045))
        # 1 pullback candle (counter-trend)
        strategy.on_bar(**_make_bar(_utc(8, 5), 2053, 2054, 2049, 2050, 2052, 2050, 2045))
        # Window should open (1 pullback candle meets minimum)
        assert strategy.state in ("ARMED", "WINDOW_OPEN")

    def test_no_signal_when_emas_not_ordered(self):
        strategy = EMAPullbackStrategy()
        # Fast < mid (no trend)
        strategy.on_bar(**_make_bar(_utc(8), 2050, 2055, 2048, 2053, 2045, 2050, 2055))
        assert strategy.state == "SCANNING"


class TestSignalGeneration:
    def test_long_signal_on_breakout(self):
        strategy = EMAPullbackStrategy(config={
            "min_ema_angle_deg": 0,
            "pullback_candles_min": 1,
            "breakout_window_bars": 20,
            "rr_ratio": 2.0,
        })
        # Build trend + pullback manually by walking through states
        # Uptrend: fast > mid > slow
        strategy.on_bar(**_make_bar(_utc(8), 2050, 2055, 2048, 2053, 2053, 2050, 2045))
        # Pullback
        strategy.on_bar(**_make_bar(_utc(8, 5), 2053, 2054, 2049, 2050, 2052, 2050, 2045))
        # Force to WINDOW_OPEN state for testing
        strategy._state = "WINDOW_OPEN"
        strategy._breakout_level = 2054.0  # Previous high
        strategy._stop_level = 2045.0      # Slow EMA
        # Breakout bar
        signal = strategy.on_bar(**_make_bar(_utc(8, 10), 2053, 2056, 2052, 2055, 2054, 2051, 2046))
        assert signal is not None
        assert signal.direction == "long"


class TestEMAAngleFilter:
    def test_rejects_flat_emas(self):
        strategy = EMAPullbackStrategy(config={"min_ema_angle_deg": 30})
        # EMAs ordered but nearly flat (all ~2050)
        strategy.on_bar(**_make_bar(_utc(8), 2050, 2051, 2049, 2050.5, 2050.3, 2050.1, 2050.0))
        # Should stay in SCANNING because EMA angle is too flat
        assert strategy.state == "SCANNING"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_ema_pullback.py -v
```
Expected: FAIL (module not found)

- [ ] **Step 3: Implement EMAPullbackStrategy**

Create `src/strategy/strategies/ema_pullback.py` with a 4-phase state machine:

1. **SCANNING**: Monitor EMA ordering (fast > mid > slow for long, reverse for short). Check EMA angle filter. If aligned + sufficient angle -> transition to ARMED.
2. **ARMED**: Count counter-trend (pullback) candles. If `pullback_candles_min <= count <= pullback_candles_max` -> transition to WINDOW_OPEN. If count exceeds max -> back to SCANNING.
3. **WINDOW_OPEN**: Set breakout level = highest high during pullback (for longs). Monitor for `breakout_window_bars`. If close breaks above level -> emit Signal. If window expires -> back to SCANNING.
4. **ENTRY**: Signal emitted, reset to SCANNING.

Key parameters (from config):
- `fast_ema`, `mid_ema`, `slow_ema`: EMA periods (14, 18, 24)
- `min_ema_angle_deg`: Minimum angle in degrees (30-95)
- `pullback_candles_min/max`: Counter-trend candle range (1-3)
- `breakout_window_bars`: How many bars to wait for breakout (20)
- `rr_ratio`: Risk-reward (2.0)
- SL: slow EMA level at entry time
- TP: entry + (entry - SL) * rr_ratio

EMA angle calculation: `angle = arctan(ema_current - ema_prev) / atr * (180/pi)`. Normalized by ATR so the angle is relative to volatility.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_ema_pullback.py -v
```
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/strategy/strategies/ema_pullback.py tests/test_ema_pullback.py
git commit -m "feat: add EMA Pullback State Machine strategy (4-phase: scan/armed/window/entry)"
```

---

## Task 6: Signal Blender

**Files:**
- Create: `src/strategy/signal_blender.py`
- Create: `tests/test_signal_blender.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_signal_blender.py
"""Tests for multi-strategy signal blender."""

import datetime as dt
from src.strategy.signal_blender import SignalBlender
from src.strategy.base import Signal


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
        # sig_a and sig_b agree (both long), so each gets +2 bonus
        # sig_a effective = 6, sig_b effective = 5, sig_c = 5
        result = blender.select([sig_a, sig_b, sig_c])
        assert result.reasoning["strategy"] == "ichi"  # highest after bonus

    def test_conflicting_directions_picks_highest(self):
        blender = SignalBlender()
        long_sig = _make_signal(direction="long", score=6)
        short_sig = _make_signal(direction="short", score=4)
        result = blender.select([long_sig, short_sig])
        assert result.direction == "long"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_signal_blender.py -v
```

- [ ] **Step 3: Implement SignalBlender**

```python
# src/strategy/signal_blender.py
"""Multi-strategy signal coordinator.

Receives signals from multiple strategies, applies multi-agree bonuses,
and selects the highest-confluence signal for execution.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional

from src.strategy.base import Signal


class SignalBlender:
    """Select the best signal from multiple strategy candidates.

    Parameters
    ----------
    multi_agree_bonus:
        Score bonus added when 2+ strategies agree on direction. Default: 2.
    """

    def __init__(self, multi_agree_bonus: int = 2):
        self._bonus = multi_agree_bonus

    def select(self, signals: List[Signal]) -> Optional[Signal]:
        if not signals:
            return None

        # Count direction agreement
        direction_counts = Counter(s.direction for s in signals)

        # Score each signal: base confluence + bonus if direction has 2+ votes
        scored = []
        for sig in signals:
            effective_score = sig.confluence_score
            if direction_counts[sig.direction] >= 2:
                effective_score += self._bonus
            scored.append((effective_score, sig))

        # Pick highest score (stable: first wins on tie)
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_signal_blender.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/strategy/signal_blender.py tests/test_signal_blender.py
git commit -m "feat: add SignalBlender for multi-strategy signal selection"
```

---

## Task 7: Challenge Simulator (Rolling Window + Monte Carlo)

**Files:**
- Create: `src/backtesting/challenge_simulator.py`
- Create: `tests/test_challenge_simulator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_challenge_simulator.py
"""Tests for rolling window + Monte Carlo challenge simulation."""

from src.backtesting.challenge_simulator import ChallengeSimulator, ChallengeSimulationResult


class TestRollingWindows:
    def test_correct_window_count(self):
        # 3 years of daily trades ~750 trading days, spacing=22 -> ~34 windows
        trades = [{"r_multiple": 0.5, "risk_pct": 1.5, "day_index": i}
                  for i in range(750)]
        sim = ChallengeSimulator(rolling_window_spacing_days=22)
        result = sim.run_rolling(trades, total_trading_days=750)
        assert 30 <= result.total_windows <= 36

    def test_winning_trades_pass_phase_1(self):
        # 20 trades, each +1R with 1.5% risk = +30% total -> easily passes 8%
        trades = [{"r_multiple": 1.0, "risk_pct": 1.5, "day_index": i * 2}
                  for i in range(20)]
        sim = ChallengeSimulator()
        result = sim.run_rolling(trades, total_trading_days=40)
        assert result.phase_1_pass_count > 0

    def test_all_losers_fail(self):
        trades = [{"r_multiple": -1.0, "risk_pct": 1.5, "day_index": i * 2}
                  for i in range(50)]
        sim = ChallengeSimulator()
        result = sim.run_rolling(trades, total_trading_days=100)
        assert result.full_pass_count == 0
        assert result.pass_rate == 0.0

    def test_failure_breakdown_populated(self):
        trades = [{"r_multiple": -1.0, "risk_pct": 1.5, "day_index": i}
                  for i in range(100)]
        sim = ChallengeSimulator()
        result = sim.run_rolling(trades, total_trading_days=100)
        assert "total_dd" in result.failure_breakdown or "daily_dd" in result.failure_breakdown


class TestMonteCarlo:
    def test_monte_carlo_runs_n_simulations(self):
        trades = [{"r_multiple": 0.5, "risk_pct": 1.5} for _ in range(100)]
        sim = ChallengeSimulator()
        result = sim.run_monte_carlo(trades, n_simulations=1000)
        assert result.total_windows == 1000

    def test_monte_carlo_shuffles_order(self):
        # With fixed seed, results should be deterministic
        trades = [{"r_multiple": 0.5 if i % 2 == 0 else -0.8, "risk_pct": 1.5}
                  for i in range(100)]
        sim = ChallengeSimulator()
        r1 = sim.run_monte_carlo(trades, n_simulations=100, seed=42)
        r2 = sim.run_monte_carlo(trades, n_simulations=100, seed=42)
        assert r1.pass_rate == r2.pass_rate


class TestCombinedResult:
    def test_combined_has_both_metrics(self):
        trades = [{"r_multiple": 0.5, "risk_pct": 1.5, "day_index": i}
                  for i in range(200)]
        sim = ChallengeSimulator()
        result = sim.run(trades, total_trading_days=200)
        assert hasattr(result, "rolling_pass_rate")
        assert hasattr(result, "monte_carlo_pass_rate")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_challenge_simulator.py -v
```

- [ ] **Step 3: Implement ChallengeSimulator**

The simulator works with **R-multiples** (position-size independent):
1. **Rolling windows**: Slide across the trade list by `day_index`, starting a fresh challenge every `spacing_days`. For each window, replay trades sequentially against `MultiPhasePropFirmTracker`, converting R-multiples to dollar P&L using the challenge's risk parameters.
2. **Monte Carlo**: Shuffle the trade list N times (preserving day blocks for intra-day correlation), replay each shuffle against the tracker.
3. **Combined**: Run both, return `ChallengeSimulationResult` with rolling and MC metrics.

Key conversion: `dollar_pnl = r_multiple * (risk_pct / 100) * current_balance`

```python
# src/backtesting/challenge_simulator.py
@dataclass
class ChallengeSimulationResult:
    total_windows: int
    phase_1_pass_count: int
    phase_2_pass_count: int
    full_pass_count: int
    pass_rate: float  # full_pass_count / total_windows
    rolling_pass_rate: float  # from rolling windows
    monte_carlo_pass_rate: float  # from MC
    avg_days_phase_1: float
    avg_days_phase_2: float
    failure_breakdown: dict  # {"daily_dd": N, "total_dd": N, ...}
    funded_monthly_returns: list
    avg_funded_monthly_return: float
    months_above_10pct: int
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_challenge_simulator.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/challenge_simulator.py tests/test_challenge_simulator.py
git commit -m "feat: add ChallengeSimulator with rolling window + Monte Carlo"
```

---

## Task 8: Rewrite Backtest Engine for Multi-Strategy

**Files:**
- Modify: `src/backtesting/vectorbt_engine.py`
- Modify: `src/backtesting/multi_tf.py`

This is the largest task. The IchimokuBacktester's main loop (lines 334-599) needs to be generalized to:
1. Run multiple strategies per bar
2. Collect signals from all strategies via the signal blender
3. Feed bar data to each strategy's `on_bar()` method

- [ ] **Step 1: Write integration test for multi-strategy backtest**

Add to `tests/test_backtest.py`:

```python
class TestMultiStrategyBacktest:
    def test_backtest_with_all_strategies_produces_trades(self):
        """Integration: run all 3 strategies on 6 months of synthetic data.
        Should produce more trades than Ichimoku alone."""
        from src.backtesting.vectorbt_engine import IchimokuBacktester
        candles = _make_1m_candles(n=200_000, trend=0.001)  # ~4 months of 1M data
        config = {
            "active_strategies": ["ichimoku", "asian_breakout", "ema_pullback"],
            "timeframes": ["15M", "5M"],
            "min_confluence_score": 2,
            "adx_threshold": 20,
            "atr_stop_multiplier": 2.5,
            "edges": {},
        }
        backtester = IchimokuBacktester(config=config)
        result = backtester.run(candles)
        assert result.metrics.get("total_trades", 0) > 0

    def test_challenge_simulation_in_result(self):
        """Backtest result should include challenge simulation data."""
        from src.backtesting.vectorbt_engine import IchimokuBacktester
        candles = _make_1m_candles(n=200_000, trend=0.001)
        config = {
            "active_strategies": ["asian_breakout"],
            "timeframes": ["15M", "5M"],
            "min_confluence_score": 1,
            "edges": {},
        }
        backtester = IchimokuBacktester(config=config)
        result = backtester.run(candles)
        assert "challenge_simulation" in result.metrics or hasattr(result, "challenge_simulation")
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_backtest.py::TestMultiStrategyBacktest -v
```

- [ ] **Step 3: Modify backtest main loop**

In `src/backtesting/vectorbt_engine.py`, modify `IchimokuBacktester.__init__()` to:
1. Read `active_strategies` from config
2. Instantiate each strategy (Ichimoku, AsianBreakout, EMAPullback)
3. Create a `SignalBlender`

Modify the main loop in `run()`:
- For each 5M bar, call each active strategy's `on_bar()` with the current bar data
- Collect all signals into a list
- Pass to `SignalBlender.select()` to pick the best
- Rest of the loop (edge checking, trade management, exit logic) stays the same

Key changes to the bar loop (around lines 460-565):
```python
# Replace:
#   signal = self._scan_for_signal(tf_data, bar_idx, instrument)
# With:
signals = []
for strategy in self._active_strategies:
    sig = strategy.on_bar(
        timestamp=ts, high=high, low=low, close=close, open=open_price,
        atr=float(row_5m.get("atr", 0.0)),
        ema_fast=float(row_5m.get("ema_fast", 0.0)),
        ema_mid=float(row_5m.get("ema_mid", 0.0)),
        ema_slow=float(row_5m.get("ema_slow", 0.0)),
        kijun=float(row_5m.get("kijun", 0.0)),
    )
    if sig is not None:
        signals.append(sig)
signal = self._signal_blender.select(signals)
```

Also modify `BacktestDataPreparer` to compute EMA columns (14, 18, 24) on 5M data alongside Ichimoku indicators.

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_backtest.py -v
```

- [ ] **Step 5: Run challenge simulator on backtest results**

After the main loop completes, add:
```python
# After equity_curve and trades are finalized:
from src.backtesting.challenge_simulator import ChallengeSimulator
challenge_sim = ChallengeSimulator(
    account_size=self._initial_balance,
    phase_1_target=prop_firm_cfg.get("phase_1", {}).get("profit_target_pct", 8.0),
    phase_2_target=prop_firm_cfg.get("phase_2", {}).get("profit_target_pct", 5.0),
)
challenge_result = challenge_sim.run(
    trades=[{"r_multiple": t.get("r_multiple", 0), "risk_pct": t.get("risk_pct", 1.5),
             "day_index": (t.get("entry_time") - first_ts).days}
            for t in closed_trades],
    total_trading_days=(last_ts - first_ts).days,
)
```

Store `challenge_result` in `BacktestResult`.

- [ ] **Step 6: Commit**

```bash
git add src/backtesting/vectorbt_engine.py src/backtesting/multi_tf.py tests/test_backtest.py
git commit -m "feat: rewrite backtest engine for multi-strategy signal blending"
```

---

## Task 9: Retarget Optimization Loop

**Files:**
- Modify: `scripts/run_optimization_loop.py`
- Modify: `src/backtesting/results_exporter.py`

- [ ] **Step 1: Update results exporter to include challenge simulation**

In `src/backtesting/results_exporter.py`, add challenge simulation data to the exported JSON:

```python
# In export_run_report(), add after metrics extraction:
challenge = result.challenge_simulation if hasattr(result, "challenge_simulation") else None
if challenge:
    report["challenge_simulation"] = {
        "pass_rate": challenge.pass_rate,
        "rolling_pass_rate": challenge.rolling_pass_rate,
        "monte_carlo_pass_rate": challenge.monte_carlo_pass_rate,
        "total_windows": challenge.total_windows,
        "full_pass_count": challenge.full_pass_count,
        "phase_1_pass_count": challenge.phase_1_pass_count,
        "avg_days_phase_1": challenge.avg_days_phase_1,
        "avg_days_phase_2": challenge.avg_days_phase_2,
        "failure_breakdown": challenge.failure_breakdown,
        "funded_avg_monthly_return": challenge.avg_funded_monthly_return,
        "months_above_10pct": challenge.months_above_10pct,
    }
```

- [ ] **Step 2: Update optimization loop stopping condition**

In `run_optimization_loop.py`, replace Sharpe-based logic:

```python
# Replace line ~111:
# self._target_sharpe = float(opt.get("target_sharpe", 1.0))
self._target_pass_rate = float(opt.get("target_pass_rate", 0.50))

# Replace line ~218 (target check):
# if current_sharpe >= self._target_sharpe:
challenge_sim = result.challenge_simulation if hasattr(result, "challenge_simulation") else None
current_pass_rate = challenge_sim.pass_rate if challenge_sim else 0.0
if current_pass_rate >= self._target_pass_rate:
    stop_reason = "target_reached"
    break

# Replace line ~289 (keep/revert decision):
# if new_sharpe >= current_sharpe:
new_pass_rate = new_challenge_sim.pass_rate if new_challenge_sim else 0.0
if new_pass_rate >= current_pass_rate:
    verdict = "kept"
```

- [ ] **Step 3: Rewrite Claude prompt**

Replace `_build_claude_prompt()` (lines 372-475) to focus on challenge pass rate:

```python
lines = [
    "You are a quantitative trading strategy optimizer for a The5ers 2-Step prop firm challenge.",
    "",
    f"## Current Run: {run_id}",
    "### Challenge Simulation Results",
    f"- Pass Rate:              {challenge.pass_rate:.1%}" if challenge else "- Pass Rate: N/A",
    f"- Phase 1 Passes:         {challenge.phase_1_pass_count}/{challenge.total_windows}" if challenge else "",
    f"- Phase 2 Passes:         {challenge.full_pass_count}/{challenge.total_windows}" if challenge else "",
    f"- Avg Days Phase 1:       {challenge.avg_days_phase_1:.0f}" if challenge else "",
    f"- Avg Days Phase 2:       {challenge.avg_days_phase_2:.0f}" if challenge else "",
    "",
    "### Failure Breakdown",
]
if challenge and challenge.failure_breakdown:
    for reason, count in challenge.failure_breakdown.items():
        lines.append(f"- {reason}: {count}")
    biggest_fail = max(challenge.failure_breakdown, key=challenge.failure_breakdown.get)
    lines.append(f"")
    lines.append(f"**BIGGEST FAILURE MODE: {biggest_fail}** — prioritize fixing this.")

lines += [
    "",
    "### Secondary Metrics",
    f"- Sharpe Ratio:   {metrics.get('sharpe', 'N/A')}",
    f"- Win Rate:       {metrics.get('win_rate', 'N/A')}",
    f"- Trade Count:    {metrics.get('trade_count', 'N/A')}",
    f"- Max Drawdown:   {metrics.get('max_drawdown', 'N/A')}",
    "",
    "## Your Task",
    f"Improve the challenge pass rate. Target: >= {self._target_pass_rate:.0%}",
    f"Make AT MOST {self._max_changes} parameter changes.",
    "Edit ONLY config/strategy.yaml and config/edges.yaml.",
    "Focus on the biggest failure mode above.",
]
```

- [ ] **Step 4: Update _load_config to handle new config structure**

Update `_load_config()` to also extract `prop_firm`, `active_strategies`, and per-strategy configs from the new YAML structure.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_optimization_loop.py src/backtesting/results_exporter.py
git commit -m "feat: retarget optimization loop from Sharpe to challenge pass rate"
```

---

## Task 10: Live Optimization Dashboard

**Files:**
- Create: `src/backtesting/optimization_dashboard.py`
- Create: `src/backtesting/optimization_dashboard.html`

- [ ] **Step 1: Create dashboard server**

Follow the same pattern as `src/backtesting/live_dashboard.py` (stdlib HTTP server, no dependencies). The server:
- Runs on port 8502
- Serves `/` -> HTML dashboard
- Serves `/api/state` -> JSON with iteration data
- Polls `reports/opt_iter_*.json` every 2 seconds
- Reads `reports/loop_status.json` for current phase
- Reads `CLAUDE.md` for learnings

```python
# src/backtesting/optimization_dashboard.py
"""Live optimization loop dashboard.

Polls reports directory for iteration JSON files and serves a
real-time dashboard showing pass rate progression, Claude's changes,
and live backtest progress.
"""

from __future__ import annotations

import glob
import json
import logging
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class OptimizationDashboardServer:
    def __init__(self, port: int = 8502, reports_dir: str = "reports", auto_open: bool = True):
        self._port = port
        self._reports_dir = _REPO_ROOT / reports_dir
        self._auto_open = auto_open
        self._server = None
        self._thread = None

    def start(self):
        handler = _make_handler(self._reports_dir)
        self._server = HTTPServer(("127.0.0.1", self._port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        url = f"http://localhost:{self._port}"
        logger.info("Optimization dashboard at %s", url)
        if self._auto_open:
            webbrowser.open(url)

    def stop(self):
        if self._server:
            self._server.shutdown()
```

- [ ] **Step 2: Create dashboard HTML**

Create `src/backtesting/optimization_dashboard.html` - self-contained HTML/CSS/JS that:
- Polls `/api/state` every 2 seconds
- Renders an iteration progress table (color-coded: green=kept, red=reverted, pulsing=active)
- Shows pass rate chart (line chart across iterations)
- Shows failure breakdown bar chart
- Shows live backtest progress bar + mini equity curve
- Shows Claude's changes and learnings
- Uses the same dark theme as `dashboard_chart.html` (var(--bg): #0f0f23)

Table columns: Iter | Pass Rate | Ph1 | Ph2 | Trades | Fail Mode | Claude Changed | Verdict

- [ ] **Step 3: Integrate dashboard launch into optimization loop**

In `run_optimization_loop.py`, add at the start of `run()`:

```python
from src.backtesting.optimization_dashboard import OptimizationDashboardServer
dashboard = OptimizationDashboardServer(port=8502, reports_dir=str(self._exporter._reports_dir))
dashboard.start()
# ... at the end:
dashboard.stop()
```

Also write `reports/loop_status.json` at each phase transition:
```python
def _write_status(self, phase: str, iteration: int, detail: str = ""):
    status = {"phase": phase, "iteration": iteration, "detail": detail,
              "timestamp": datetime.now(timezone.utc).isoformat()}
    status_path = Path(self._exporter._reports_dir) / "loop_status.json"
    status_path.write_text(json.dumps(status), encoding="utf-8")
```

- [ ] **Step 4: Test manually**

```bash
source .venv/Scripts/activate
python -c "
from src.backtesting.optimization_dashboard import OptimizationDashboardServer
s = OptimizationDashboardServer(port=8502, auto_open=True)
s.start()
import time; time.sleep(5)
s.stop()
print('Dashboard started and stopped successfully')
"
```

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/optimization_dashboard.py src/backtesting/optimization_dashboard.html
git commit -m "feat: add live optimization dashboard (port 8502, auto-refresh)"
```

---

## Task 11: Integration Test - Full Pipeline

**Files:**
- Create: `tests/integration/test_the5ers_pipeline.py`

- [ ] **Step 1: Write end-to-end integration test**

```python
# tests/integration/test_the5ers_pipeline.py
"""End-to-end integration test for The5ers 2-Step pipeline."""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta


def _make_trending_gold_data(days: int = 60) -> pd.DataFrame:
    """Generate synthetic 1M gold data with trend + mean reversion."""
    np.random.seed(42)
    n_bars = days * 24 * 60  # 1M bars
    prices = [2000.0]
    for i in range(1, n_bars):
        # Trending with session patterns
        hour = (i // 60) % 24
        volatility = 0.3 if 6 <= hour < 14 else 0.1  # London volatile
        trend = 0.001 if (i // (24 * 60)) % 10 < 7 else -0.001  # 7 days up, 3 down
        prices.append(prices[-1] + np.random.normal(trend, volatility))

    dates = pd.date_range("2024-01-01", periods=n_bars, freq="1min", tz="UTC")
    df = pd.DataFrame({
        "open": prices,
        "high": [p + abs(np.random.normal(0, 0.5)) for p in prices],
        "low": [p - abs(np.random.normal(0, 0.5)) for p in prices],
        "close": prices,
        "volume": np.random.randint(100, 1000, n_bars),
    }, index=dates)
    df.index.name = "timestamp"
    return df


class TestFullPipeline:
    def test_backtest_produces_trades_with_multi_strategy(self):
        from src.backtesting.vectorbt_engine import IchimokuBacktester
        data = _make_trending_gold_data(days=30)
        config = {
            "active_strategies": ["asian_breakout", "ema_pullback"],
            "timeframes": ["15M", "5M"],
            "min_confluence_score": 1,
            "atr_stop_multiplier": 2.5,
            "edges": {},
        }
        bt = IchimokuBacktester(config=config, initial_balance=10_000.0)
        result = bt.run(data)
        # With session breakout + EMA, should get some trades in 30 days
        assert len(result.trades) > 0, f"Expected trades, got {len(result.trades)}"

    def test_challenge_simulation_returns_valid_result(self):
        from src.backtesting.challenge_simulator import ChallengeSimulator
        # Simulate a strategy with 55% win rate, 1:2 RR
        np.random.seed(42)
        trades = []
        for i in range(200):
            r = 2.0 if np.random.random() < 0.55 else -1.0
            trades.append({"r_multiple": r, "risk_pct": 1.5, "day_index": i // 3})
        sim = ChallengeSimulator()
        result = sim.run(trades, total_trading_days=200)
        assert 0.0 <= result.pass_rate <= 1.0
        assert result.total_windows > 0

    def test_multi_phase_tracker_full_progression(self):
        from src.backtesting.metrics import MultiPhasePropFirmTracker
        tracker = MultiPhasePropFirmTracker()
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        tracker.initialise(10_000.0, ts)

        # Simulate gradual profit: +$50/day for 20 days = +$1000 = +10%
        balance = 10_000.0
        for day in range(20):
            balance += 50.0
            tracker.update(ts + timedelta(days=day), balance)

        status = tracker.get_status()
        # Should have passed Phase 1 (8%) and be in Phase 2 or beyond
        assert status["phase_1_passed"] is True
```

- [ ] **Step 2: Run integration tests**

```bash
pytest tests/integration/test_the5ers_pipeline.py -v --timeout=120
```

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_the5ers_pipeline.py
git commit -m "test: add end-to-end integration tests for The5ers pipeline"
```

---

## Task 12: Update Existing Tests

**Files:**
- Modify: `tests/test_full_pipeline.py`
- Modify: `tests/test_monte_carlo.py`

- [ ] **Step 1: Update time_limit_days references**

In `tests/test_full_pipeline.py`, change any `time_limit_days=30` to `time_limit_days=0` (unlimited) where The5ers rules apply.

In `tests/test_monte_carlo.py`, update DD threshold assertions to match the new 4.5% circuit breaker.

- [ ] **Step 2: Run full test suite**

```bash
pytest tests/ -v --timeout=120 -x
```
Fix any failures from the refactored signal engine or updated config structure.

- [ ] **Step 3: Commit**

```bash
git add tests/
git commit -m "test: update existing tests for The5ers rules and multi-strategy engine"
```

---

## Execution Order & Parallelism

```
Wave 1 (parallel - no dependencies):
  Task 1: Config changes
  Task 2: MultiPhasePropFirmTracker
  Task 3: Ichimoku 15M simplification
  Task 4: Asian Range Breakout strategy
  Task 5: EMA Pullback State Machine
  Task 6: Signal Blender

Wave 2 (sequential - depends on Wave 1):
  Task 7: Challenge Simulator (depends on Task 2)
  Task 8: Rewrite Backtest Engine (depends on Tasks 3-6)

Wave 3 (parallel - depends on Wave 2):
  Task 9: Retarget Optimization Loop (depends on Tasks 7-8)
  Task 10: Live Dashboard (depends on Task 9 for status.json)

Wave 4 (final - depends on everything):
  Task 11: Integration tests
  Task 12: Update existing tests
```
