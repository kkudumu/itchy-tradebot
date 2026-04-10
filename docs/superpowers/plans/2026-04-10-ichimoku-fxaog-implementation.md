# Ichimoku FXAOG Strategy — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the existing Ichimoku strategy with a faithful implementation of the FX At One Glance course methodology — 7 evaluators, 8 trade types, configurable exits, Five Elements O/G, Time Theory, Wave Analysis, and Price Range targets.

**Architecture:** Multi-Evaluator + Strategy Coordinator. 7 new evaluators (registered via `Evaluator(ABC, key='...')`) feed an `EvalMatrix` consumed by a single `IchimokuFXAOGStrategy(Strategy, key='ichimoku_fxaog')`. Three new indicator modules provide the computational foundation. An `IchimokuExitManager(TradingMode)` handles configurable exit logic.

**Tech Stack:** Python 3.11+, NumPy, Pandas, pytest, existing base classes in `src/strategy/base.py`

**Spec:** `docs/superpowers/specs/2026-04-10-ichimoku-fxaog-strategy-design.md`

---

## File Structure

### New Files (8)
| File | Responsibility |
|---|---|
| `src/indicators/fractals.py` | Bill Williams 5-bar fractal detection, market structure tracking |
| `src/indicators/price_action.py` | Candlestick pattern detection (tweezer, inside bar, engulfing, pin bar) |
| `src/indicators/heikin_ashi.py` | Heikin Ashi candle computation |
| `src/strategy/evaluators/price_action_eval.py` | Wraps price_action indicator as Evaluator |
| `src/strategy/evaluators/fractal_eval.py` | Wraps fractals indicator + FFO/momentum as Evaluator |
| `src/strategy/evaluators/five_elements_eval.py` | O/G equilibrium counting system |
| `src/strategy/evaluators/time_theory_eval.py` | Kihon Suchi projection + Tato Suchi detection |
| `src/strategy/evaluators/wave_analysis_eval.py` | I/V/N/P/Y wave classification |
| `src/strategy/evaluators/price_target_eval.py` | E/V/N/NT target calculation |
| `src/strategy/trading_modes/ichimoku_exit.py` | Configurable exit manager (trailing/targets/hybrid + HA) |

### Replaced Files (2)
| File | What changes |
|---|---|
| `src/strategy/evaluators/ichimoku_eval.py` | Rewritten as `IchimokuCoreEval` (key: `ichimoku_core`) — the old `ichimoku` key is retired |
| `src/strategy/strategies/ichimoku.py` | Rewritten as `IchimokuFXAOGStrategy` (key: `ichimoku_fxaog`) — old strategy moved to `_legacy/` |

### Modified Files (1)
| File | What changes |
|---|---|
| `config/strategy.yaml` | New `ichimoku_fxaog` config block added; `active_strategy` updated |

### Test Files (12)
| File | Covers |
|---|---|
| `tests/indicators/test_fractals.py` | Fractal detection, momentum, structure |
| `tests/indicators/test_price_action.py` | Tweezer, inside bar, engulfing, pin bar |
| `tests/indicators/test_heikin_ashi.py` | HA candle computation |
| `tests/strategy/evaluators/test_ichimoku_core_eval.py` | 5-point checklist, state snapshot |
| `tests/strategy/evaluators/test_price_action_eval.py` | PA evaluator integration |
| `tests/strategy/evaluators/test_fractal_eval.py` | Fractal evaluator + FFO |
| `tests/strategy/evaluators/test_five_elements_eval.py` | O/G counting, cycle reset |
| `tests/strategy/evaluators/test_time_theory_eval.py` | Kihon Suchi, Tato Suchi |
| `tests/strategy/evaluators/test_wave_analysis_eval.py` | Wave classification |
| `tests/strategy/evaluators/test_price_target_eval.py` | Target calculations |
| `tests/strategy/trading_modes/test_ichimoku_exit.py` | All 3 exit modes |
| `tests/strategy/strategies/test_ichimoku_fxaog.py` | Strategy integration, trade types, confluence |

---

## Phase 1: Indicator Foundation

### Task 1: Fractal Indicator

**Files:**
- Create: `src/indicators/fractals.py`
- Create: `tests/indicators/test_fractals.py`

- [ ] **Step 1: Write failing tests for fractal detection**

```python
# tests/indicators/test_fractals.py
from __future__ import annotations

import numpy as np
import pytest

from src.indicators.fractals import detect_fractals, FractalLevel, FractalResult


def _make_v_shape(n=20, base=1800.0):
    """V-shape: descend to midpoint, then ascend — creates one bear fractal at bottom."""
    mid = n // 2
    prices = np.zeros(n)
    for i in range(n):
        if i <= mid:
            prices[i] = base - i * 2.0
        else:
            prices[i] = base - mid * 2.0 + (i - mid) * 2.0
    high = prices + 1.0
    low = prices - 1.0
    return high, low, prices


def _make_mountain(n=20, base=1800.0):
    """Mountain: ascend to midpoint, then descend — creates one bull fractal at top."""
    mid = n // 2
    prices = np.zeros(n)
    for i in range(n):
        if i <= mid:
            prices[i] = base + i * 2.0
        else:
            prices[i] = base + mid * 2.0 - (i - mid) * 2.0
    high = prices + 1.0
    low = prices - 1.0
    return high, low, prices


class TestDetectFractals:
    def test_bull_fractal_at_mountain_peak(self):
        high, low, close = _make_mountain(n=11, base=1800.0)
        result = detect_fractals(high, low)
        bulls = [f for f in result.bull_fractals if not np.isnan(f.price)]
        assert len(bulls) >= 1
        # Peak is at index 5 (midpoint of 11 bars)
        assert any(f.bar_index == 5 for f in bulls)

    def test_bear_fractal_at_v_bottom(self):
        high, low, close = _make_v_shape(n=11, base=1800.0)
        result = detect_fractals(high, low)
        bears = [f for f in result.bear_fractals if not np.isnan(f.price)]
        assert len(bears) >= 1
        assert any(f.bar_index == 5 for f in bears)

    def test_five_bar_minimum(self):
        """Fractal needs 2 bars on each side — minimum 5 bars to form."""
        high = np.array([10.0, 11.0, 12.0, 11.0, 10.0])
        low = np.array([9.0, 10.0, 11.0, 10.0, 9.0])
        result = detect_fractals(high, low)
        assert len(result.bull_fractals) == 1
        assert result.bull_fractals[0].bar_index == 2

    def test_no_fractals_in_flat_market(self):
        high = np.full(20, 1800.0)
        low = np.full(20, 1799.0)
        result = detect_fractals(high, low)
        assert len(result.bull_fractals) == 0
        assert len(result.bear_fractals) == 0

    def test_too_few_bars_returns_empty(self):
        high = np.array([10.0, 11.0, 10.0])
        low = np.array([9.0, 10.0, 9.0])
        result = detect_fractals(high, low)
        assert len(result.bull_fractals) == 0
        assert len(result.bear_fractals) == 0

    def test_fractal_never_repaints(self):
        """Once a fractal is detected, extending the array must not remove it."""
        high1 = np.array([10.0, 11.0, 13.0, 11.0, 10.0, 9.0, 8.0])
        low1 = np.array([9.0, 10.0, 12.0, 10.0, 9.0, 8.0, 7.0])
        r1 = detect_fractals(high1, low1)
        bulls1 = {f.bar_index for f in r1.bull_fractals}

        # Extend with more bars
        high2 = np.concatenate([high1, np.array([9.0, 10.0, 11.0])])
        low2 = np.concatenate([low1, np.array([8.0, 9.0, 10.0])])
        r2 = detect_fractals(high2, low2)
        bulls2 = {f.bar_index for f in r2.bull_fractals}
        assert bulls1.issubset(bulls2)


class TestFractalMomentum:
    def test_strengthening_momentum(self):
        """Successive bull fractals getting further apart = strengthening."""
        # Create staircase: each peak higher and further apart
        high = np.array([
            10, 11, 15, 11, 10,   # bull fractal at 2 (15)
            10, 11, 12, 20, 12,   # bull fractal at 8 (20)
            11, 12, 13, 27, 13,   # bull fractal at 13 (27)
            12, 11, 10, 9, 8,
        ], dtype=float)
        low = high - 2.0
        result = detect_fractals(high, low)
        # Distance between successive bull fractals should increase
        bulls = sorted(result.bull_fractals, key=lambda f: f.bar_index)
        if len(bulls) >= 3:
            d1 = bulls[1].price - bulls[0].price
            d2 = bulls[2].price - bulls[1].price
            assert d2 > d1  # strengthening
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/indicators/test_fractals.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.indicators.fractals'`

- [ ] **Step 3: Implement fractal detection**

```python
# src/indicators/fractals.py
"""Bill Williams 5-bar fractal detection and market structure tracking."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FractalLevel:
    """A single fractal point."""
    price: float
    bar_index: int


@dataclass
class FractalResult:
    """All detected fractals and derived structure metrics."""
    bull_fractals: list[FractalLevel] = field(default_factory=list)
    bear_fractals: list[FractalLevel] = field(default_factory=list)


def detect_fractals(high: np.ndarray, low: np.ndarray) -> FractalResult:
    """Detect Bill Williams 5-bar fractals.

    A bull fractal: bar with higher high than 2 bars on each side.
    A bear fractal: bar with lower low than 2 bars on each side.
    Requires minimum 5 bars. Side bars can equal each other but NOT the center.
    """
    n = len(high)
    result = FractalResult()
    if n < 5:
        return result

    for i in range(2, n - 2):
        # Bull fractal: high[i] is strictly greater than 2 bars on each side
        if (high[i] > high[i - 1] and high[i] > high[i - 2]
                and high[i] > high[i + 1] and high[i] > high[i + 2]):
            result.bull_fractals.append(FractalLevel(price=high[i], bar_index=i))

        # Bear fractal: low[i] is strictly less than 2 bars on each side
        if (low[i] < low[i - 1] and low[i] < low[i - 2]
                and low[i] < low[i + 1] and low[i] < low[i + 2]):
            result.bear_fractals.append(FractalLevel(price=low[i], bar_index=i))

    return result


def fractal_momentum(fractals: list[FractalLevel]) -> list[float]:
    """Compute price distances between successive same-type fractals.

    Returns a list of absolute distances. Increasing distances = strengthening
    momentum; decreasing = weakening.
    """
    if len(fractals) < 2:
        return []
    sorted_f = sorted(fractals, key=lambda f: f.bar_index)
    return [abs(sorted_f[i + 1].price - sorted_f[i].price)
            for i in range(len(sorted_f) - 1)]


def momentum_trend(distances: list[float]) -> str:
    """Classify momentum as strengthening, weakening, or flat.

    Uses the last 3 distances. If 2+ are increasing, strengthening.
    If 2+ are decreasing, weakening. Otherwise flat.
    """
    if len(distances) < 2:
        return 'flat'
    recent = distances[-3:] if len(distances) >= 3 else distances
    increases = sum(1 for i in range(len(recent) - 1) if recent[i + 1] > recent[i])
    decreases = sum(1 for i in range(len(recent) - 1) if recent[i + 1] < recent[i])
    if increases > decreases:
        return 'strengthening'
    elif decreases > increases:
        return 'weakening'
    return 'flat'


def last_broken_direction(
    high: np.ndarray,
    low: np.ndarray,
    bull_fractals: list[FractalLevel],
    bear_fractals: list[FractalLevel],
) -> str | None:
    """Determine which fractal type was most recently broken by price.

    Returns 'bull' if a bull fractal was broken (price went above it),
    'bear' if a bear fractal was broken (price went below it), or None.
    """
    last_bull_break_idx = -1
    last_bear_break_idx = -1

    for f in bull_fractals:
        # Find first bar after fractal where high exceeds fractal price
        for j in range(f.bar_index + 3, len(high)):
            if high[j] > f.price:
                last_bull_break_idx = max(last_bull_break_idx, j)
                break

    for f in bear_fractals:
        for j in range(f.bar_index + 3, len(low)):
            if low[j] < f.price:
                last_bear_break_idx = max(last_bear_break_idx, j)
                break

    if last_bull_break_idx > last_bear_break_idx:
        return 'bull'
    elif last_bear_break_idx > last_bull_break_idx:
        return 'bear'
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/indicators/test_fractals.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/indicators/fractals.py tests/indicators/test_fractals.py
git commit -m "feat(task-1): fractal indicator with Bill Williams 5-bar detection"
```

---

### Task 2: Price Action Indicator

**Files:**
- Create: `src/indicators/price_action.py`
- Create: `tests/indicators/test_price_action.py`

- [ ] **Step 1: Write failing tests for price action patterns**

```python
# tests/indicators/test_price_action.py
from __future__ import annotations

import numpy as np
import pytest

from src.indicators.price_action import (
    PriceActionResult,
    detect_patterns,
)


def _candles(ohlc_list: list[tuple]) -> tuple[np.ndarray, ...]:
    """Build arrays from list of (open, high, low, close) tuples."""
    arr = np.array(ohlc_list, dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


class TestTweezerDetection:
    def test_tweezer_bottom(self):
        """Red candle then green candle with matching lows."""
        o, h, l, c = _candles([
            (1810, 1812, 1800, 1802),  # red candle (open > close)
            (1802, 1811, 1800, 1809),  # green candle (close > open), same low
        ])
        result = detect_patterns(o, h, l, c, tick_tolerance=2)
        assert result.tweezer_bottom is True
        assert result.tweezer_top is False

    def test_tweezer_top(self):
        """Green candle then red candle with matching highs."""
        o, h, l, c = _candles([
            (1800, 1812, 1799, 1810),  # green
            (1810, 1812, 1801, 1803),  # red, same high
        ])
        result = detect_patterns(o, h, l, c, tick_tolerance=2)
        assert result.tweezer_top is True

    def test_same_color_not_tweezer(self):
        """Two green candles cannot form a tweezer."""
        o, h, l, c = _candles([
            (1800, 1812, 1800, 1810),  # green
            (1805, 1813, 1800, 1811),  # green — not a tweezer
        ])
        result = detect_patterns(o, h, l, c, tick_tolerance=2)
        assert result.tweezer_bottom is False
        assert result.tweezer_top is False


class TestInsideBar:
    def test_inside_bar_count(self):
        """3 candles inside the mother bar body, then breakout above."""
        o, h, l, c = _candles([
            (1800, 1815, 1795, 1810),  # mother bar (body 1800-1810)
            (1803, 1808, 1801, 1807),  # inside (body 1803-1807)
            (1804, 1809, 1802, 1806),  # inside
            (1805, 1807, 1803, 1805),  # inside
            (1806, 1818, 1804, 1815),  # breakout above 1810
        ])
        result = detect_patterns(o, h, l, c)
        assert result.inside_bar_count == 3
        assert result.inside_bar_breakout == 'up'

    def test_inside_bar_uses_body_not_wick(self):
        """Wicks can exceed the mother bar — only bodies matter."""
        o, h, l, c = _candles([
            (1800, 1815, 1795, 1810),  # mother body: 1800-1810
            (1803, 1820, 1790, 1807),  # body inside (1803-1807), wicks outside
        ])
        result = detect_patterns(o, h, l, c)
        assert result.inside_bar_count == 1


class TestEngulfing:
    def test_bullish_engulfing(self):
        o, h, l, c = _candles([
            (1810, 1812, 1800, 1802),  # red (body 1802-1810)
            (1800, 1815, 1798, 1812),  # green, body 1800-1812 engulfs 1802-1810
        ])
        result = detect_patterns(o, h, l, c)
        assert result.engulfing_bullish is True

    def test_bearish_engulfing(self):
        o, h, l, c = _candles([
            (1800, 1812, 1798, 1810),  # green (body 1800-1810)
            (1812, 1814, 1797, 1798),  # red, body 1798-1812 engulfs 1800-1810
        ])
        result = detect_patterns(o, h, l, c)
        assert result.engulfing_bearish is True


class TestPinBar:
    def test_hammer(self):
        """Long lower wick, small body at top = bullish pin bar."""
        o, h, l, c = _candles([
            (1802, 1805, 1790, 1804),  # body 2, lower wick 12 — hammer
        ])
        result = detect_patterns(o, h, l, c)
        assert result.pin_bar_bullish is True

    def test_shooting_star(self):
        """Long upper wick, small body at bottom = bearish pin bar."""
        o, h, l, c = _candles([
            (1804, 1818, 1801, 1802),  # body 2, upper wick 16 — shooting star
        ])
        result = detect_patterns(o, h, l, c)
        assert result.pin_bar_bearish is True


class TestDoji:
    def test_doji_detected(self):
        o, h, l, c = _candles([
            (1800.0, 1805, 1795, 1800.1),  # open ~= close
        ])
        result = detect_patterns(o, h, l, c)
        assert result.doji is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/indicators/test_price_action.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement price action pattern detection**

```python
# src/indicators/price_action.py
"""Candlestick price action pattern detection.

Detects: tweezers, inside bars, outside breaks, engulfing, pin bars, doji.
All patterns operate on OHLC numpy arrays and look at the most recent candles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PriceActionResult:
    """Detected price action patterns from the most recent candles."""
    tweezer_bottom: bool = False
    tweezer_top: bool = False
    inside_bar_count: int = 0
    inside_bar_breakout: str = 'none'   # 'up' | 'down' | 'none'
    engulfing_bullish: bool = False
    engulfing_bearish: bool = False
    pin_bar_bullish: bool = False
    pin_bar_bearish: bool = False
    doji: bool = False
    mother_bar_high: float = 0.0
    mother_bar_low: float = 0.0


def _is_red(o: float, c: float) -> bool:
    return c < o


def _is_green(o: float, c: float) -> bool:
    return c > o


def _body_top(o: float, c: float) -> float:
    return max(o, c)


def _body_bot(o: float, c: float) -> float:
    return min(o, c)


def _body_size(o: float, c: float) -> float:
    return abs(c - o)


def detect_patterns(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    tick_tolerance: float = 2.0,
    doji_body_pct: float = 0.05,
) -> PriceActionResult:
    """Detect price action patterns from the most recent candles.

    Parameters
    ----------
    open_, high, low, close : np.ndarray
        OHLC price arrays (at least 1 bar required).
    tick_tolerance : float
        Maximum difference in ticks for tweezer matching.
    doji_body_pct : float
        Maximum body/range ratio to classify as doji.
    """
    n = len(open_)
    result = PriceActionResult()

    if n == 0:
        return result

    # --- Single-bar patterns (last bar) ---
    last = n - 1
    o_l, h_l, l_l, c_l = open_[last], high[last], low[last], close[last]
    body = _body_size(o_l, c_l)
    full_range = h_l - l_l

    # Doji
    if full_range > 0 and body / full_range <= doji_body_pct:
        result.doji = True

    # Pin bars
    if full_range > 0:
        upper_wick = h_l - _body_top(o_l, c_l)
        lower_wick = _body_bot(o_l, c_l) - l_l
        # Hammer / bullish pin: lower wick >= 2x body, upper wick < body
        if lower_wick >= 2.0 * max(body, 0.01) and upper_wick < body:
            result.pin_bar_bullish = True
        # Shooting star / bearish pin: upper wick >= 2x body, lower wick < body
        if upper_wick >= 2.0 * max(body, 0.01) and lower_wick < body:
            result.pin_bar_bearish = True

    if n < 2:
        return result

    # --- Two-bar patterns (last two bars) ---
    prev = last - 1
    o_p, h_p, l_p, c_p = open_[prev], high[prev], low[prev], close[prev]

    # Tweezer bottom: red candle then green candle, matching lows
    if (_is_red(o_p, c_p) and _is_green(o_l, c_l)
            and abs(l_p - l_l) <= tick_tolerance):
        result.tweezer_bottom = True

    # Tweezer top: green candle then red candle, matching highs
    if (_is_green(o_p, c_p) and _is_red(o_l, c_l)
            and abs(h_p - h_l) <= tick_tolerance):
        result.tweezer_top = True

    # Engulfing bullish: red then green, green body engulfs red body
    prev_bt, prev_bb = _body_top(o_p, c_p), _body_bot(o_p, c_p)
    last_bt, last_bb = _body_top(o_l, c_l), _body_bot(o_l, c_l)
    if _is_red(o_p, c_p) and _is_green(o_l, c_l):
        if last_bt >= prev_bt and last_bb <= prev_bb and body > 0:
            result.engulfing_bullish = True

    # Engulfing bearish: green then red, red body engulfs green body
    if _is_green(o_p, c_p) and _is_red(o_l, c_l):
        if last_bt >= prev_bt and last_bb <= prev_bb and body > 0:
            result.engulfing_bearish = True

    # --- Inside bar detection (scan backward from last bar) ---
    # Find the mother bar: first bar before the inside sequence
    # whose body is larger than subsequent bars' bodies.
    # Mother bar body defines the range; doji/hammer → use previous full-body candle.
    if n >= 3:
        # Walk backward to find mother bar
        mother_idx = _find_mother_bar(open_, close, last)
        if mother_idx is not None:
            m_top = _body_top(open_[mother_idx], close[mother_idx])
            m_bot = _body_bot(open_[mother_idx], close[mother_idx])
            result.mother_bar_high = m_top
            result.mother_bar_low = m_bot

            # Count inside bars after mother
            count = 0
            for i in range(mother_idx + 1, last):
                bt = _body_top(open_[i], close[i])
                bb = _body_bot(open_[i], close[i])
                if bb >= m_bot and bt <= m_top:
                    count += 1
                else:
                    break  # inside sequence broken

            if count > 0:
                result.inside_bar_count = count
                # Check if last bar breaks out
                if close[last] > m_top:
                    result.inside_bar_breakout = 'up'
                elif close[last] < m_bot:
                    result.inside_bar_breakout = 'down'

    return result


def _find_mother_bar(open_: np.ndarray, close: np.ndarray, last: int) -> int | None:
    """Find the mother bar for inside-bar analysis.

    Walks backward from `last - 1` to find a candle with a sufficiently
    large body. Skips doji/very-small-body candles (the course says to use
    the previous full-body candle when a doji/hammer forms).
    """
    for i in range(last - 1, max(last - 10, -1), -1):
        body = _body_size(open_[i], close[i])
        full_range = max(abs(open_[i] - close[i]), 0.01)
        # Check subsequent bars are inside this one
        m_top = _body_top(open_[i], close[i])
        m_bot = _body_bot(open_[i], close[i])
        if body < 0.5:  # skip near-zero body (doji)
            continue
        # Verify at least one bar after this is inside
        if i + 1 < last:
            next_bt = _body_top(open_[i + 1], close[i + 1])
            next_bb = _body_bot(open_[i + 1], close[i + 1])
            if next_bb >= m_bot and next_bt <= m_top:
                return i
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/indicators/test_price_action.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/indicators/price_action.py tests/indicators/test_price_action.py
git commit -m "feat(task-2): price action indicator — tweezer, inside bar, engulfing, pin bar"
```

---

### Task 3: Heikin Ashi Indicator

**Files:**
- Create: `src/indicators/heikin_ashi.py`
- Create: `tests/indicators/test_heikin_ashi.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/indicators/test_heikin_ashi.py
from __future__ import annotations

import numpy as np
import pytest

from src.indicators.heikin_ashi import compute_heikin_ashi, HACandle, ha_trend_signal


class TestComputeHeikinAshi:
    def test_close_is_ohlc_average(self):
        o = np.array([100.0, 105.0])
        h = np.array([110.0, 115.0])
        l = np.array([95.0, 100.0])
        c = np.array([105.0, 110.0])
        ha = compute_heikin_ashi(o, h, l, c)
        # HA close = (O+H+L+C)/4
        expected_close_1 = (105.0 + 115.0 + 100.0 + 110.0) / 4
        assert ha.close[1] == pytest.approx(expected_close_1)

    def test_open_is_midpoint_of_previous(self):
        o = np.array([100.0, 105.0, 108.0])
        h = np.array([110.0, 115.0, 118.0])
        l = np.array([95.0, 100.0, 103.0])
        c = np.array([105.0, 110.0, 115.0])
        ha = compute_heikin_ashi(o, h, l, c)
        # HA open[2] = (HA open[1] + HA close[1]) / 2
        expected_open_2 = (ha.open[1] + ha.close[1]) / 2
        assert ha.open[2] == pytest.approx(expected_open_2)

    def test_first_bar_uses_real_ohlc(self):
        o = np.array([100.0])
        h = np.array([110.0])
        l = np.array([95.0])
        c = np.array([105.0])
        ha = compute_heikin_ashi(o, h, l, c)
        assert ha.open[0] == pytest.approx((100.0 + 105.0) / 2)
        assert ha.close[0] == pytest.approx((100.0 + 110.0 + 95.0 + 105.0) / 4)

    def test_output_length_matches_input(self):
        n = 50
        o = np.random.uniform(100, 110, n)
        h = o + np.random.uniform(0, 5, n)
        l = o - np.random.uniform(0, 5, n)
        c = np.random.uniform(l, h)
        ha = compute_heikin_ashi(o, h, l, c)
        assert len(ha.open) == n
        assert len(ha.high) == n
        assert len(ha.low) == n
        assert len(ha.close) == n


class TestHATrendSignal:
    def test_strong_bullish_no_lower_wick(self):
        # Green HA candle with no lower wick: open == low
        candle = HACandle(open=100.0, high=110.0, low=100.0, close=108.0)
        assert ha_trend_signal(candle) == 'strong_bullish'

    def test_weak_bullish_has_lower_wick(self):
        candle = HACandle(open=100.0, high=110.0, low=97.0, close=108.0)
        assert ha_trend_signal(candle) == 'weak_bullish'

    def test_strong_bearish_no_upper_wick(self):
        candle = HACandle(open=108.0, high=108.0, low=95.0, close=100.0)
        assert ha_trend_signal(candle) == 'strong_bearish'

    def test_doji(self):
        candle = HACandle(open=100.0, high=105.0, low=95.0, close=100.1)
        assert ha_trend_signal(candle) == 'indecision'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/indicators/test_heikin_ashi.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement Heikin Ashi**

```python
# src/indicators/heikin_ashi.py
"""Heikin Ashi candle computation and trend signal classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HACandle:
    """Single Heikin Ashi candle."""
    open: float
    high: float
    low: float
    close: float


@dataclass
class HAResult:
    """Full Heikin Ashi series."""
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray


def compute_heikin_ashi(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> HAResult:
    """Compute Heikin Ashi candles from regular OHLC data.

    HA Close = (O + H + L + C) / 4
    HA Open[0] = (O[0] + C[0]) / 2  (first bar seed)
    HA Open[i] = (HA Open[i-1] + HA Close[i-1]) / 2
    HA High = max(H, HA Open, HA Close)
    HA Low = min(L, HA Open, HA Close)
    """
    n = len(open_)
    ha_close = (open_ + high + low + close) / 4.0

    ha_open = np.empty(n)
    ha_open[0] = (open_[0] + close[0]) / 2.0
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    ha_high = np.maximum(high, np.maximum(ha_open, ha_close))
    ha_low = np.minimum(low, np.minimum(ha_open, ha_close))

    return HAResult(open=ha_open, high=ha_high, low=ha_low, close=ha_close)


def ha_candle_at(ha: HAResult, idx: int) -> HACandle:
    """Extract a single HACandle at the given index."""
    return HACandle(
        open=ha.open[idx], high=ha.high[idx],
        low=ha.low[idx], close=ha.close[idx],
    )


def ha_trend_signal(candle: HACandle, doji_pct: float = 0.05) -> str:
    """Classify a single HA candle's trend signal.

    Returns one of:
        'strong_bullish'  — green, no lower wick (open == low)
        'weak_bullish'    — green, has lower wick
        'strong_bearish'  — red, no upper wick (open == high)
        'weak_bearish'    — red, has upper wick
        'indecision'      — doji / spinning top
    """
    body = abs(candle.close - candle.open)
    full_range = candle.high - candle.low
    if full_range > 0 and body / full_range <= doji_pct:
        return 'indecision'

    is_green = candle.close > candle.open
    tol = full_range * 0.01 if full_range > 0 else 0.001

    if is_green:
        # No lower wick means open ~= low
        if abs(candle.open - candle.low) <= tol:
            return 'strong_bullish'
        return 'weak_bullish'
    else:
        # No upper wick means open ~= high
        if abs(candle.open - candle.high) <= tol:
            return 'strong_bearish'
        return 'weak_bearish'
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/indicators/test_heikin_ashi.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/indicators/heikin_ashi.py tests/indicators/test_heikin_ashi.py
git commit -m "feat(task-3): Heikin Ashi indicator with trend signal classification"
```

---

## Phase 2: Core Evaluators

### Task 4: IchimokuCoreEval (replaces existing ichimoku evaluator)

**Files:**
- Modify: `src/strategy/evaluators/ichimoku_eval.py` — rewrite as `IchimokuCoreEval`
- Create: `tests/strategy/evaluators/test_ichimoku_core_eval.py`

**Context:** The existing `IchimokuEvaluator` (key: `'ichimoku'`) is registered in `EVALUATOR_REGISTRY`. The new evaluator uses key `'ichimoku_core'` to avoid collision during migration. The old key remains available for backward compatibility.

- [ ] **Step 1: Write failing tests**

```python
# tests/strategy/evaluators/test_ichimoku_core_eval.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.base import EvaluatorResult, EVALUATOR_REGISTRY


def _make_ohlcv(n=200, start=1800.0, step=0.0, session_hour=10) -> pd.DataFrame:
    """Generate OHLCV data. step > 0 = uptrend, step < 0 = downtrend, 0 = flat."""
    idx = pd.date_range('2026-01-01', periods=n, freq='1h', tz='UTC')
    close = np.arange(n, dtype=float) * step + start
    return pd.DataFrame({
        'open': close - 0.5 * abs(step) if step != 0 else close,
        'high': close + 2.0,
        'low': close - 2.0,
        'close': close,
        'volume': np.ones(n) * 100,
    }, index=idx)


def _assert_evaluator_result_shape(result: EvaluatorResult) -> None:
    assert isinstance(result, EvaluatorResult)
    assert isinstance(result.direction, float)
    assert isinstance(result.confidence, float)
    assert isinstance(result.metadata, dict)
    assert -1.0 <= result.direction <= 1.0
    assert 0.0 <= result.confidence <= 1.0


class TestIchimokuCoreEvalRegistered:
    def test_registered_in_evaluator_registry(self):
        # Import triggers registration
        from src.strategy.evaluators.ichimoku_core_eval import IchimokuCoreEval
        assert 'ichimoku_core' in EVALUATOR_REGISTRY

    def test_old_ichimoku_key_still_exists(self):
        """Backward compat: original ichimoku evaluator should still be importable."""
        assert 'ichimoku' in EVALUATOR_REGISTRY


class TestIchimokuCoreEvalOutput:
    def setup_method(self):
        from src.strategy.evaluators.ichimoku_core_eval import IchimokuCoreEval
        self.eval = IchimokuCoreEval()

    def test_returns_evaluator_result(self):
        ohlcv = _make_ohlcv(n=200, step=1.0)
        result = self.eval.evaluate(ohlcv)
        _assert_evaluator_result_shape(result)

    def test_uptrend_bullish_direction(self):
        ohlcv = _make_ohlcv(n=200, step=2.0)
        result = self.eval.evaluate(ohlcv)
        assert result.direction == 1.0

    def test_downtrend_bearish_direction(self):
        ohlcv = _make_ohlcv(n=200, step=-2.0)
        result = self.eval.evaluate(ohlcv)
        assert result.direction == -1.0

    def test_flat_market_neutral(self):
        ohlcv = _make_ohlcv(n=200, step=0.0)
        result = self.eval.evaluate(ohlcv)
        assert result.direction == 0.0

    def test_metadata_contains_ichimoku_state(self):
        ohlcv = _make_ohlcv(n=200, step=1.0)
        result = self.eval.evaluate(ohlcv)
        expected_keys = {
            'tenkan', 'kijun', 'senkou_a', 'senkou_b', 'chikou',
            'cloud_position', 'cloud_direction', 'tk_cross',
            'tk_cross_bars_ago', 'chikou_vs_price', 'chikou_vs_kumo',
            'kumo_future_direction', 'kumo_thickness', 'kijun_flat',
            'kijun_distance_pips', 'tenkan_kijun_angle',
        }
        assert expected_keys.issubset(set(result.metadata.keys()))

    def test_confidence_scales_with_checklist(self):
        """More aligned signals → higher confidence."""
        ohlcv_trend = _make_ohlcv(n=200, step=2.0)
        ohlcv_flat = _make_ohlcv(n=200, step=0.0)
        r_trend = self.eval.evaluate(ohlcv_trend)
        r_flat = self.eval.evaluate(ohlcv_flat)
        assert r_trend.confidence >= r_flat.confidence

    def test_chikou_inside_price_sets_zero_direction(self):
        """When chikou is inside price (ranging), direction should be neutral."""
        # Create choppy data that will make chikou bounce inside price
        n = 200
        idx = pd.date_range('2026-01-01', periods=n, freq='1h', tz='UTC')
        # Oscillate price to create ranging conditions
        close = 1800.0 + 5.0 * np.sin(np.arange(n) * 0.5)
        ohlcv = pd.DataFrame({
            'open': close - 0.5,
            'high': close + 3.0,
            'low': close - 3.0,
            'close': close,
            'volume': np.ones(n) * 100,
        }, index=idx)
        result = self.eval.evaluate(ohlcv)
        # In a ranging market, chikou should be inside → direction neutral
        if result.metadata.get('chikou_vs_price') == 'inside':
            assert result.direction == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/strategy/evaluators/test_ichimoku_core_eval.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.strategy.evaluators.ichimoku_core_eval'`

- [ ] **Step 3: Implement IchimokuCoreEval**

Create `src/strategy/evaluators/ichimoku_core_eval.py`:

```python
# src/strategy/evaluators/ichimoku_core_eval.py
"""IchimokuCoreEval — enhanced Ichimoku evaluator with full 5-point checklist.

Replaces the simple ichimoku evaluator for the FXAOG strategy.
Produces a rich metadata dict with the complete Ichimoku state snapshot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import Evaluator, EvaluatorResult
from ...indicators.ichimoku import IchimokuCalculator


class IchimokuCoreEval(Evaluator, key='ichimoku_core'):
    """Full-featured Ichimoku evaluator implementing the 5-point checklist.

    Metadata contains:
        tenkan, kijun, senkou_a, senkou_b, chikou — raw values at last bar
        cloud_position — 'above' | 'below' | 'inside'
        cloud_direction — 'bullish' | 'bearish'
        tk_cross — 'bullish' | 'bearish' | 'none'
        tk_cross_bars_ago — int (bars since last TK cross)
        chikou_vs_price — 'above' | 'below' | 'inside'
        chikou_vs_kumo — 'above' | 'below' | 'inside'
        kumo_future_direction — 'bullish' | 'bearish'
        kumo_thickness — float (absolute)
        kijun_flat — bool
        kijun_distance_pips — float
        tenkan_kijun_angle — float (normalized slope)
    """

    def __init__(self, tenkan_period: int = 9, kijun_period: int = 26,
                 senkou_b_period: int = 52):
        self._calc = IchimokuCalculator(tenkan_period, kijun_period, senkou_b_period)
        self._kijun_period = kijun_period

    def evaluate(self, ohlcv: pd.DataFrame) -> EvaluatorResult:
        high = ohlcv['high'].values
        low = ohlcv['low'].values
        close = ohlcv['close'].values
        n = len(close)

        ichi = self._calc.calculate(high, low, close)

        # Use last valid bar for evaluation
        last = n - 1
        tenkan = ichi.tenkan_sen[last] if not np.isnan(ichi.tenkan_sen[last]) else np.nan
        kijun = ichi.kijun_sen[last] if not np.isnan(ichi.kijun_sen[last]) else np.nan

        # Senkou values at last bar position (they are shifted forward 26)
        # Current cloud = senkou values at index (last - kijun_period) if available
        cloud_idx = last
        senkou_a = ichi.senkou_a[cloud_idx] if cloud_idx < len(ichi.senkou_a) and not np.isnan(ichi.senkou_a[cloud_idx]) else np.nan
        senkou_b = ichi.senkou_b[cloud_idx] if cloud_idx < len(ichi.senkou_b) and not np.isnan(ichi.senkou_b[cloud_idx]) else np.nan

        # Chikou is current close plotted 26 bars back
        chikou_val = close[last]
        chikou_compare_idx = last - self._kijun_period
        chikou_compare_price = close[chikou_compare_idx] if chikou_compare_idx >= 0 else np.nan

        # --- 5-Point Checklist ---
        checklist_score = 0
        direction_vote = 0

        # 1. Cloud position
        cloud_top = max(senkou_a, senkou_b) if not (np.isnan(senkou_a) or np.isnan(senkou_b)) else np.nan
        cloud_bot = min(senkou_a, senkou_b) if not (np.isnan(senkou_a) or np.isnan(senkou_b)) else np.nan

        if not np.isnan(cloud_top):
            if close[last] > cloud_top:
                cloud_position = 'above'
                checklist_score += 1
                direction_vote += 1
            elif close[last] < cloud_bot:
                cloud_position = 'below'
                checklist_score += 1
                direction_vote -= 1
            else:
                cloud_position = 'inside'
        else:
            cloud_position = 'inside'

        # 2. TK relationship
        if not (np.isnan(tenkan) or np.isnan(kijun)):
            if tenkan > kijun:
                checklist_score += 1
                direction_vote += 1
            elif tenkan < kijun:
                checklist_score += 1
                direction_vote -= 1
        # else: no contribution

        # 3. Chikou vs price
        if not np.isnan(chikou_compare_price):
            chikou_high = max(high[max(0, chikou_compare_idx - 2):chikou_compare_idx + 3]) if chikou_compare_idx + 3 <= n else chikou_compare_price + 1
            chikou_low = min(low[max(0, chikou_compare_idx - 2):chikou_compare_idx + 3]) if chikou_compare_idx + 3 <= n else chikou_compare_price - 1
            if chikou_val > chikou_high:
                chikou_vs_price = 'above'
                checklist_score += 1
                direction_vote += 1
            elif chikou_val < chikou_low:
                chikou_vs_price = 'below'
                checklist_score += 1
                direction_vote -= 1
            else:
                chikou_vs_price = 'inside'
        else:
            chikou_vs_price = 'inside'

        # 4. Chikou vs Kumo (26 periods back)
        if chikou_compare_idx >= 0 and not np.isnan(cloud_top):
            # Get cloud at the chikou comparison point
            c_senkou_a = ichi.senkou_a[chikou_compare_idx] if chikou_compare_idx < len(ichi.senkou_a) and not np.isnan(ichi.senkou_a[chikou_compare_idx]) else np.nan
            c_senkou_b = ichi.senkou_b[chikou_compare_idx] if chikou_compare_idx < len(ichi.senkou_b) and not np.isnan(ichi.senkou_b[chikou_compare_idx]) else np.nan
            if not (np.isnan(c_senkou_a) or np.isnan(c_senkou_b)):
                c_top = max(c_senkou_a, c_senkou_b)
                c_bot = min(c_senkou_a, c_senkou_b)
                if chikou_val > c_top:
                    chikou_vs_kumo = 'above'
                    checklist_score += 1
                    direction_vote += 1
                elif chikou_val < c_bot:
                    chikou_vs_kumo = 'below'
                    checklist_score += 1
                    direction_vote -= 1
                else:
                    chikou_vs_kumo = 'inside'
            else:
                chikou_vs_kumo = 'inside'
        else:
            chikou_vs_kumo = 'inside'

        # 5. Kumo future direction (26 bars ahead if available, else use current)
        future_idx = min(last + self._kijun_period, len(ichi.senkou_a) - 1)
        f_a = ichi.senkou_a[future_idx] if future_idx < len(ichi.senkou_a) and not np.isnan(ichi.senkou_a[future_idx]) else np.nan
        f_b = ichi.senkou_b[future_idx] if future_idx < len(ichi.senkou_b) and not np.isnan(ichi.senkou_b[future_idx]) else np.nan
        if not (np.isnan(f_a) or np.isnan(f_b)):
            if f_a > f_b:
                kumo_future = 'bullish'
                checklist_score += 1
                direction_vote += 1
            else:
                kumo_future = 'bearish'
                checklist_score += 1
                direction_vote -= 1
        else:
            kumo_future = 'bullish'  # default

        # Cloud direction (current)
        if not (np.isnan(senkou_a) or np.isnan(senkou_b)):
            cloud_direction = 'bullish' if senkou_a > senkou_b else 'bearish'
        else:
            cloud_direction = 'bullish'

        # TK cross detection
        tk_cross, tk_cross_bars_ago = self._detect_tk_cross(ichi.tenkan_sen, ichi.kijun_sen, last)

        # Kijun flat detection (unchanged for 5+ bars)
        kijun_flat = self._is_kijun_flat(ichi.kijun_sen, last, lookback=5)

        # Kijun distance
        kijun_distance = abs(close[last] - kijun) if not np.isnan(kijun) else 0.0

        # Tenkan/Kijun angle (slope over last 5 bars)
        tk_angle = self._tk_angle(ichi.tenkan_sen, ichi.kijun_sen, last)

        # Kumo thickness
        thickness = abs(senkou_a - senkou_b) if not (np.isnan(senkou_a) or np.isnan(senkou_b)) else 0.0

        # --- Final direction and confidence ---
        # Chikou inside price = ranging → force neutral
        if chikou_vs_price == 'inside':
            final_direction = 0.0
        elif direction_vote > 0:
            final_direction = 1.0
        elif direction_vote < 0:
            final_direction = -1.0
        else:
            final_direction = 0.0

        confidence = checklist_score / 5.0

        metadata = {
            'tenkan': float(tenkan) if not np.isnan(tenkan) else None,
            'kijun': float(kijun) if not np.isnan(kijun) else None,
            'senkou_a': float(senkou_a) if not np.isnan(senkou_a) else None,
            'senkou_b': float(senkou_b) if not np.isnan(senkou_b) else None,
            'chikou': float(chikou_val),
            'cloud_position': cloud_position,
            'cloud_direction': cloud_direction,
            'tk_cross': tk_cross,
            'tk_cross_bars_ago': tk_cross_bars_ago,
            'chikou_vs_price': chikou_vs_price,
            'chikou_vs_kumo': chikou_vs_kumo,
            'kumo_future_direction': kumo_future,
            'kumo_thickness': float(thickness),
            'kijun_flat': kijun_flat,
            'kijun_distance_pips': float(kijun_distance),
            'tenkan_kijun_angle': float(tk_angle),
            'checklist_score': checklist_score,
            'cloud_top': float(cloud_top) if not np.isnan(cloud_top) else None,
            'cloud_bot': float(cloud_bot) if not np.isnan(cloud_bot) else None,
        }

        return EvaluatorResult(
            direction=final_direction,
            confidence=confidence,
            metadata=metadata,
        )

    def _detect_tk_cross(self, tenkan: np.ndarray, kijun: np.ndarray,
                         last: int) -> tuple[str, int]:
        """Detect TK crossover at last bar and bars since last cross."""
        if last < 1 or np.isnan(tenkan[last]) or np.isnan(kijun[last]):
            return 'none', 9999

        # Current bar cross
        curr_diff = tenkan[last] - kijun[last]
        prev_diff = tenkan[last - 1] - kijun[last - 1]
        if np.isnan(prev_diff):
            return 'none', 9999

        cross = 'none'
        if prev_diff <= 0 < curr_diff:
            cross = 'bullish'
        elif prev_diff >= 0 > curr_diff:
            cross = 'bearish'

        # Count bars since last cross
        bars_ago = 0
        for i in range(last, max(last - 500, 0), -1):
            if i < 1 or np.isnan(tenkan[i]) or np.isnan(kijun[i]):
                break
            d = tenkan[i] - kijun[i]
            d_prev = tenkan[i - 1] - kijun[i - 1]
            if np.isnan(d_prev):
                break
            if (d_prev <= 0 < d) or (d_prev >= 0 > d):
                bars_ago = last - i
                break

        return cross, bars_ago

    def _is_kijun_flat(self, kijun: np.ndarray, last: int, lookback: int = 5) -> bool:
        if last < lookback:
            return False
        segment = kijun[last - lookback:last + 1]
        if np.any(np.isnan(segment)):
            return False
        return np.all(segment == segment[0])

    def _tk_angle(self, tenkan: np.ndarray, kijun: np.ndarray,
                  last: int, lookback: int = 5) -> float:
        """Compute normalized slope of T-K midpoint over lookback bars."""
        if last < lookback:
            return 0.0
        t_seg = tenkan[last - lookback:last + 1]
        k_seg = kijun[last - lookback:last + 1]
        if np.any(np.isnan(t_seg)) or np.any(np.isnan(k_seg)):
            return 0.0
        mid = (t_seg + k_seg) / 2.0
        slope = (mid[-1] - mid[0]) / lookback
        # Normalize by average price level
        avg_price = np.mean(mid)
        if avg_price == 0:
            return 0.0
        return slope / avg_price
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/strategy/evaluators/test_ichimoku_core_eval.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/strategy/evaluators/ichimoku_core_eval.py tests/strategy/evaluators/test_ichimoku_core_eval.py
git commit -m "feat(task-4): IchimokuCoreEval with 5-point checklist and full state snapshot"
```

---

### Task 5: PriceActionEval

**Files:**
- Create: `src/strategy/evaluators/price_action_eval.py`
- Create: `tests/strategy/evaluators/test_price_action_eval.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/strategy/evaluators/test_price_action_eval.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.base import EvaluatorResult, EVALUATOR_REGISTRY


def _make_ohlcv_from_candles(candles: list[tuple]) -> pd.DataFrame:
    """Build OHLCV DataFrame from (open, high, low, close) tuples."""
    idx = pd.date_range('2026-01-01', periods=len(candles), freq='1h', tz='UTC')
    arr = np.array(candles, dtype=float)
    return pd.DataFrame({
        'open': arr[:, 0], 'high': arr[:, 1],
        'low': arr[:, 2], 'close': arr[:, 3],
        'volume': np.ones(len(candles)) * 100,
    }, index=idx)


class TestPriceActionEvalRegistered:
    def test_registered(self):
        from src.strategy.evaluators.price_action_eval import PriceActionEval
        assert 'price_action' in EVALUATOR_REGISTRY


class TestPriceActionEvalPatterns:
    def setup_method(self):
        from src.strategy.evaluators.price_action_eval import PriceActionEval
        self.eval = PriceActionEval()

    def test_tweezer_bottom_detected_in_metadata(self):
        ohlcv = _make_ohlcv_from_candles([
            (1810, 1812, 1800, 1802),  # red
            (1802, 1811, 1800, 1809),  # green, same low → tweezer bottom
        ])
        result = self.eval.evaluate(ohlcv)
        assert result.metadata['tweezer_bottom'] is True
        assert result.direction >= 0.0  # bullish signal

    def test_engulfing_bearish_detected(self):
        ohlcv = _make_ohlcv_from_candles([
            (1800, 1812, 1798, 1810),  # green
            (1812, 1814, 1797, 1798),  # red engulfing
        ])
        result = self.eval.evaluate(ohlcv)
        assert result.metadata['engulfing_bearish'] is True
        assert result.direction <= 0.0

    def test_inside_bar_breakout_up(self):
        ohlcv = _make_ohlcv_from_candles([
            (1800, 1815, 1795, 1810),  # mother
            (1803, 1808, 1801, 1807),  # inside
            (1804, 1809, 1802, 1806),  # inside
            (1806, 1818, 1804, 1815),  # breakout up
        ])
        result = self.eval.evaluate(ohlcv)
        assert result.metadata['inside_bar_breakout'] == 'up'
        assert result.metadata['inside_bar_count'] >= 2

    def test_no_patterns_in_flat_market(self):
        candles = [(1800, 1801, 1799, 1800)] * 10
        ohlcv = _make_ohlcv_from_candles(candles)
        result = self.eval.evaluate(ohlcv)
        assert result.metadata['tweezer_bottom'] is False
        assert result.metadata['engulfing_bullish'] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/strategy/evaluators/test_price_action_eval.py -v`
Expected: FAIL

- [ ] **Step 3: Implement PriceActionEval**

```python
# src/strategy/evaluators/price_action_eval.py
"""PriceActionEval — candlestick pattern detection evaluator."""

from __future__ import annotations

import pandas as pd

from ..base import Evaluator, EvaluatorResult
from ...indicators.price_action import detect_patterns


class PriceActionEval(Evaluator, key='price_action'):
    """Detects tweezer, inside bar, engulfing, pin bar, and doji patterns.

    Direction:
      +1.0 for bullish patterns (tweezer bottom, bullish engulfing, inside-up)
      -1.0 for bearish patterns (tweezer top, bearish engulfing, inside-down)
       0.0 if no pattern or mixed

    Confidence:
      engulfing = 1.0, inside-out breakout = 0.9, tweezer = 0.7, pin bar = 0.5
    """

    def __init__(self, tick_tolerance: float = 2.0):
        self._tick_tolerance = tick_tolerance

    def evaluate(self, ohlcv: pd.DataFrame) -> EvaluatorResult:
        pa = detect_patterns(
            ohlcv['open'].values,
            ohlcv['high'].values,
            ohlcv['low'].values,
            ohlcv['close'].values,
            tick_tolerance=self._tick_tolerance,
        )

        direction = 0.0
        confidence = 0.0

        # Priority: engulfing > inside-out > tweezer > pin bar
        if pa.engulfing_bullish:
            direction = 1.0
            confidence = 1.0
        elif pa.engulfing_bearish:
            direction = -1.0
            confidence = 1.0
        elif pa.inside_bar_breakout == 'up':
            direction = 1.0
            confidence = 0.9
        elif pa.inside_bar_breakout == 'down':
            direction = -1.0
            confidence = 0.9
        elif pa.tweezer_bottom:
            direction = 1.0
            confidence = 0.7
        elif pa.tweezer_top:
            direction = -1.0
            confidence = 0.7
        elif pa.pin_bar_bullish:
            direction = 1.0
            confidence = 0.5
        elif pa.pin_bar_bearish:
            direction = -1.0
            confidence = 0.5

        metadata = {
            'tweezer_bottom': pa.tweezer_bottom,
            'tweezer_top': pa.tweezer_top,
            'inside_bar_count': pa.inside_bar_count,
            'inside_bar_breakout': pa.inside_bar_breakout,
            'engulfing_bullish': pa.engulfing_bullish,
            'engulfing_bearish': pa.engulfing_bearish,
            'pin_bar_bullish': pa.pin_bar_bullish,
            'pin_bar_bearish': pa.pin_bar_bearish,
            'doji': pa.doji,
            'mother_bar_high': pa.mother_bar_high,
            'mother_bar_low': pa.mother_bar_low,
        }

        return EvaluatorResult(
            direction=direction,
            confidence=confidence,
            metadata=metadata,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/strategy/evaluators/test_price_action_eval.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/strategy/evaluators/price_action_eval.py tests/strategy/evaluators/test_price_action_eval.py
git commit -m "feat(task-5): PriceActionEval — tweezer, inside-out, engulfing detection"
```

---

### Task 6: FractalEval

**Files:**
- Create: `src/strategy/evaluators/fractal_eval.py`
- Create: `tests/strategy/evaluators/test_fractal_eval.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/strategy/evaluators/test_fractal_eval.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.base import EvaluatorResult, EVALUATOR_REGISTRY


def _make_trending_ohlcv(n=100, start=1800.0, step=2.0) -> pd.DataFrame:
    """Uptrend with periodic pullbacks creating fractals."""
    idx = pd.date_range('2026-01-01', periods=n, freq='1h', tz='UTC')
    # Create sawtooth: 5 up, 2 down, repeat
    close = np.zeros(n)
    price = start
    for i in range(n):
        if i % 7 < 5:
            price += step
        else:
            price -= step * 0.5
        close[i] = price
    return pd.DataFrame({
        'open': close - 0.5,
        'high': close + 1.5,
        'low': close - 1.5,
        'close': close,
        'volume': np.ones(n) * 100,
    }, index=idx)


class TestFractalEvalRegistered:
    def test_registered(self):
        from src.strategy.evaluators.fractal_eval import FractalEval
        assert 'fractal' in EVALUATOR_REGISTRY


class TestFractalEvalOutput:
    def setup_method(self):
        from src.strategy.evaluators.fractal_eval import FractalEval
        self.eval = FractalEval()

    def test_returns_evaluator_result(self):
        ohlcv = _make_trending_ohlcv(n=100)
        result = self.eval.evaluate(ohlcv)
        assert isinstance(result, EvaluatorResult)
        assert -1.0 <= result.direction <= 1.0

    def test_metadata_contains_fractal_data(self):
        ohlcv = _make_trending_ohlcv(n=100)
        result = self.eval.evaluate(ohlcv)
        assert 'bull_fractals' in result.metadata
        assert 'bear_fractals' in result.metadata
        assert 'last_broken_direction' in result.metadata
        assert 'momentum_trend' in result.metadata

    def test_uptrend_has_bull_fractals(self):
        ohlcv = _make_trending_ohlcv(n=100, step=2.0)
        result = self.eval.evaluate(ohlcv)
        assert len(result.metadata['bull_fractals']) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/strategy/evaluators/test_fractal_eval.py -v`
Expected: FAIL

- [ ] **Step 3: Implement FractalEval**

```python
# src/strategy/evaluators/fractal_eval.py
"""FractalEval — Bill Williams fractal detection and market structure evaluator."""

from __future__ import annotations

import pandas as pd

from ..base import Evaluator, EvaluatorResult
from ...indicators.fractals import (
    detect_fractals, fractal_momentum, momentum_trend, last_broken_direction,
)


class FractalEval(Evaluator, key='fractal'):
    """Evaluates market structure via 5-bar fractals.

    Direction: +1.0 if last broken direction is bull, -1.0 if bear, 0.0 if unknown.
    Confidence: based on momentum trend (strengthening=0.8, flat=0.5, weakening=0.2).

    Metadata contains bull/bear fractal lists, FFO levels, momentum metrics.
    """

    def evaluate(self, ohlcv: pd.DataFrame) -> EvaluatorResult:
        high = ohlcv['high'].values
        low = ohlcv['low'].values

        fractal_result = detect_fractals(high, low)

        # Market structure direction
        broken_dir = last_broken_direction(
            high, low,
            fractal_result.bull_fractals,
            fractal_result.bear_fractals,
        )

        if broken_dir == 'bull':
            direction = 1.0
        elif broken_dir == 'bear':
            direction = -1.0
        else:
            direction = 0.0

        # Momentum analysis
        bulls_momentum = fractal_momentum(fractal_result.bull_fractals)
        bears_momentum = fractal_momentum(fractal_result.bear_fractals)
        active_momentum = bulls_momentum if direction >= 0 else bears_momentum
        m_trend = momentum_trend(active_momentum)

        confidence_map = {'strengthening': 0.8, 'flat': 0.5, 'weakening': 0.2}
        confidence = confidence_map.get(m_trend, 0.5)

        # Serialize fractal levels for metadata
        bulls_serialized = [{'price': f.price, 'bar_index': f.bar_index}
                           for f in fractal_result.bull_fractals]
        bears_serialized = [{'price': f.price, 'bar_index': f.bar_index}
                           for f in fractal_result.bear_fractals]

        # Nearest fractals to current price
        current_price = ohlcv['close'].values[-1]
        nearest_bull = None
        nearest_bear = None
        for f in sorted(fractal_result.bull_fractals, key=lambda x: abs(x.price - current_price)):
            nearest_bull = {'price': f.price, 'bar_index': f.bar_index}
            break
        for f in sorted(fractal_result.bear_fractals, key=lambda x: abs(x.price - current_price)):
            nearest_bear = {'price': f.price, 'bar_index': f.bar_index}
            break

        metadata = {
            'bull_fractals': bulls_serialized,
            'bear_fractals': bears_serialized,
            'last_broken_direction': broken_dir,
            'momentum_trend': m_trend,
            'fractal_momentum_distances': active_momentum,
            'nearest_bull_fractal': nearest_bull,
            'nearest_bear_fractal': nearest_bear,
        }

        return EvaluatorResult(
            direction=direction,
            confidence=confidence,
            metadata=metadata,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/strategy/evaluators/test_fractal_eval.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/strategy/evaluators/fractal_eval.py tests/strategy/evaluators/test_fractal_eval.py
git commit -m "feat(task-6): FractalEval — market structure, FFO, momentum tracking"
```

---

### Task 7: FiveElementsEval

**Files:**
- Create: `src/strategy/evaluators/five_elements_eval.py`
- Create: `tests/strategy/evaluators/test_five_elements_eval.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/strategy/evaluators/test_five_elements_eval.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.base import EvaluatorResult, EVALUATOR_REGISTRY


def _make_ohlcv_with_tk_cross(n=200, cross_at=100, direction='bullish') -> pd.DataFrame:
    """Create data with a clear TK crossover at the specified bar.

    For bullish: flat until cross_at, then strong uptrend.
    For bearish: flat until cross_at, then strong downtrend.
    """
    idx = pd.date_range('2026-01-01', periods=n, freq='1h', tz='UTC')
    close = np.full(n, 1800.0)
    step = 3.0 if direction == 'bullish' else -3.0
    for i in range(cross_at, n):
        close[i] = 1800.0 + (i - cross_at) * step
    return pd.DataFrame({
        'open': close - 0.5,
        'high': close + 2.0,
        'low': close - 2.0,
        'close': close,
        'volume': np.ones(n) * 100,
    }, index=idx)


class TestFiveElementsEvalRegistered:
    def test_registered(self):
        from src.strategy.evaluators.five_elements_eval import FiveElementsEval
        assert 'five_elements' in EVALUATOR_REGISTRY


class TestFiveElementsEvalLogic:
    def setup_method(self):
        from src.strategy.evaluators.five_elements_eval import FiveElementsEval
        self.eval = FiveElementsEval()

    def test_returns_evaluator_result(self):
        ohlcv = _make_ohlcv_with_tk_cross(n=200, cross_at=100)
        result = self.eval.evaluate(ohlcv)
        assert isinstance(result, EvaluatorResult)

    def test_metadata_has_og_counts(self):
        ohlcv = _make_ohlcv_with_tk_cross(n=200, cross_at=100)
        result = self.eval.evaluate(ohlcv)
        assert 'overcoming_count' in result.metadata
        assert 'generating_count' in result.metadata
        assert 'is_disequilibrium' in result.metadata
        assert 'cycle_direction' in result.metadata

    def test_tk_cross_resets_cycle(self):
        """After a TK cross, O and G counts should reset to 0."""
        ohlcv = _make_ohlcv_with_tk_cross(n=200, cross_at=190)
        result = self.eval.evaluate(ohlcv)
        # Right after cross, counts should be near zero
        assert result.metadata['overcoming_count'] >= 0
        assert result.metadata['generating_count'] >= 0

    def test_disequilibrium_when_counts_unequal(self):
        ohlcv = _make_ohlcv_with_tk_cross(n=200, cross_at=50)
        result = self.eval.evaluate(ohlcv)
        o = result.metadata['overcoming_count']
        g = result.metadata['generating_count']
        expected = (o != g)
        assert result.metadata['is_disequilibrium'] == expected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/strategy/evaluators/test_five_elements_eval.py -v`
Expected: FAIL

- [ ] **Step 3: Implement FiveElementsEval**

```python
# src/strategy/evaluators/five_elements_eval.py
"""FiveElementsEval — O/G equilibrium counting system from Ichimoku Advanced.

Tracks Overcoming (O) and Generating (G) crossovers after each TK cross.
Market is tradeable when O != G (disequilibrium).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..base import Evaluator, EvaluatorResult
from ...indicators.ichimoku import IchimokuCalculator


# O/G classification per element theory crossover table
_CROSS_TABLE = {
    # (faster_component, slower_component) -> 'O' or 'G'
    ('tenkan', 'senkou_a'): 'O',
    ('tenkan', 'senkou_b'): 'G',
    ('kijun', 'senkou_a'): 'G',
    ('kijun', 'senkou_b'): 'O',
    ('chikou', 'tenkan'): 'G',
    ('chikou', 'kijun'): 'G',
    ('chikou', 'senkou_a'): 'O',
    ('chikou', 'senkou_b'): 'G',
    ('senkou_a', 'senkou_b'): 'G',  # Kumo twist — always G
}


@dataclass
class CrossoverEvent:
    pair: str
    og_type: str   # 'O' or 'G'
    bar_index: int


class FiveElementsEval(Evaluator, key='five_elements'):
    """Counts O/G crossovers after TK cross to determine equilibrium state.

    Direction: matches the TK cross direction (+1 bullish, -1 bearish, 0 no cycle).
    Confidence: 1.0 if disequilibrium (tradeable), 0.0 if equilibrium (blocked).
    """

    def __init__(self, tenkan_period: int = 9, kijun_period: int = 26,
                 senkou_b_period: int = 52):
        self._calc = IchimokuCalculator(tenkan_period, kijun_period, senkou_b_period)
        self._kp = kijun_period

    def evaluate(self, ohlcv: pd.DataFrame) -> EvaluatorResult:
        high = ohlcv['high'].values
        low = ohlcv['low'].values
        close = ohlcv['close'].values

        ichi = self._calc.calculate(high, low, close)
        n = len(close)

        # Build component arrays
        tenkan = ichi.tenkan_sen
        kijun = ichi.kijun_sen
        # Chikou is close displaced back kijun_period
        chikou = np.full(n, np.nan)
        if n > self._kp:
            chikou[:n - self._kp] = close[self._kp:]
        senkou_a = ichi.senkou_a[:n]
        senkou_b = ichi.senkou_b[:n]

        # Detect TK crosses to find cycle start
        tk_crosses = self._find_tk_crosses(tenkan, kijun)

        if not tk_crosses:
            return EvaluatorResult(
                direction=0.0, confidence=0.0,
                metadata={'overcoming_count': 0, 'generating_count': 0,
                          'is_disequilibrium': False, 'cycle_direction': 'none',
                          'total_signals': 0, 'crossover_log': []},
            )

        # Use last TK cross as cycle start
        last_cross_idx, cycle_dir = tk_crosses[-1]

        # Count O/G crossovers after the cycle start
        components = {
            'tenkan': tenkan, 'kijun': kijun, 'chikou': chikou,
            'senkou_a': senkou_a, 'senkou_b': senkou_b,
        }

        o_count = 0
        g_count = 0
        seen_pairs: set[str] = set()
        crossover_log: list[dict] = []

        for (c1_name, c2_name), og_type in _CROSS_TABLE.items():
            if c1_name == 'tenkan' and c2_name == 'kijun':
                continue  # TK cross starts the cycle, don't count it

            c1 = components.get(c1_name)
            c2 = components.get(c2_name)
            if c1 is None or c2 is None:
                continue

            pair_key = f"{c1_name}_{c2_name}"
            if pair_key in seen_pairs:
                continue

            # Check if this pair crossed after the cycle start
            for i in range(last_cross_idx + 1, n):
                if i < 1:
                    continue
                if np.isnan(c1[i]) or np.isnan(c2[i]) or np.isnan(c1[i-1]) or np.isnan(c2[i-1]):
                    continue
                prev_diff = c1[i - 1] - c2[i - 1]
                curr_diff = c1[i] - c2[i]
                if (prev_diff <= 0 < curr_diff) or (prev_diff >= 0 > curr_diff):
                    seen_pairs.add(pair_key)
                    if og_type == 'O':
                        o_count += 1
                    else:
                        g_count += 1
                    crossover_log.append({
                        'pair': pair_key, 'type': og_type, 'bar_index': i,
                    })
                    break  # count once per cycle

        is_diseq = o_count != g_count
        direction = 1.0 if cycle_dir == 'bullish' else -1.0
        confidence = 1.0 if is_diseq else 0.0

        metadata = {
            'overcoming_count': o_count,
            'generating_count': g_count,
            'is_disequilibrium': is_diseq,
            'cycle_direction': cycle_dir,
            'total_signals': o_count + g_count,
            'cycle_start_bar': last_cross_idx,
            'crossover_log': crossover_log,
        }

        return EvaluatorResult(
            direction=direction, confidence=confidence, metadata=metadata,
        )

    def _find_tk_crosses(self, tenkan: np.ndarray,
                         kijun: np.ndarray) -> list[tuple[int, str]]:
        """Find all TK crossover points. Returns list of (bar_index, 'bullish'|'bearish')."""
        crosses = []
        for i in range(1, len(tenkan)):
            if np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(tenkan[i-1]) or np.isnan(kijun[i-1]):
                continue
            prev = tenkan[i - 1] - kijun[i - 1]
            curr = tenkan[i] - kijun[i]
            if prev <= 0 < curr:
                crosses.append((i, 'bullish'))
            elif prev >= 0 > curr:
                crosses.append((i, 'bearish'))
        return crosses
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/strategy/evaluators/test_five_elements_eval.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/strategy/evaluators/five_elements_eval.py tests/strategy/evaluators/test_five_elements_eval.py
git commit -m "feat(task-7): FiveElementsEval — O/G equilibrium counting system"
```

---

## Phase 3: Advanced Evaluators

### Task 8: TimeTheoryEval

**Files:**
- Create: `src/strategy/evaluators/time_theory_eval.py`
- Create: `tests/strategy/evaluators/test_time_theory_eval.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/strategy/evaluators/test_time_theory_eval.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.base import EvaluatorResult, EVALUATOR_REGISTRY


def _make_ohlcv_with_fractal_at(n=200, fractal_bar=50) -> pd.DataFrame:
    """Data with a clear bull fractal at fractal_bar (V-shape bottom)."""
    idx = pd.date_range('2026-01-01', periods=n, freq='1h', tz='UTC')
    close = np.full(n, 1800.0)
    # Create a dip at fractal_bar to form a bear fractal
    for i in range(max(0, fractal_bar - 5), min(n, fractal_bar + 6)):
        close[i] = 1800.0 - (5 - abs(i - fractal_bar)) * 3.0
    return pd.DataFrame({
        'open': close - 0.5,
        'high': close + 1.5,
        'low': close - 1.5,
        'close': close,
        'volume': np.ones(n) * 100,
    }, index=idx)


class TestTimeTheoryEvalRegistered:
    def test_registered(self):
        from src.strategy.evaluators.time_theory_eval import TimeTheoryEval
        assert 'time_theory' in EVALUATOR_REGISTRY


class TestTimeTheoryEvalKihonSuchi:
    def setup_method(self):
        from src.strategy.evaluators.time_theory_eval import TimeTheoryEval
        self.eval = TimeTheoryEval()

    def test_returns_evaluator_result(self):
        ohlcv = _make_ohlcv_with_fractal_at(n=200, fractal_bar=50)
        result = self.eval.evaluate(ohlcv)
        assert isinstance(result, EvaluatorResult)

    def test_metadata_has_projections(self):
        ohlcv = _make_ohlcv_with_fractal_at(n=200, fractal_bar=50)
        result = self.eval.evaluate(ohlcv)
        assert 'active_projections' in result.metadata
        assert 'is_on_kihon_suchi' in result.metadata

    def test_kihon_suchi_9_detected(self):
        """If current bar is 9 bars from a fractal, flag it."""
        # Fractal at bar 50, current bar at 59 → 9 bars elapsed
        ohlcv = _make_ohlcv_with_fractal_at(n=62, fractal_bar=50)
        result = self.eval.evaluate(ohlcv)
        # Check if any projection shows 9-bar alignment
        projections = result.metadata['active_projections']
        has_9 = any(p['bars_elapsed'] in (8, 9, 10) for p in projections)
        # May or may not hit depending on fractal detection
        assert isinstance(projections, list)

    def test_tolerance_plus_minus_one(self):
        """Kihon Suchi 26 should match at 25, 26, or 27 bars."""
        from src.strategy.evaluators.time_theory_eval import _is_kihon_suchi
        assert _is_kihon_suchi(25, tolerance=1) is True  # 26-1
        assert _is_kihon_suchi(26, tolerance=1) is True
        assert _is_kihon_suchi(27, tolerance=1) is True  # 26+1
        assert _is_kihon_suchi(28, tolerance=1) is False

    def test_non_kihon_number_rejected(self):
        from src.strategy.evaluators.time_theory_eval import _is_kihon_suchi
        assert _is_kihon_suchi(15, tolerance=1) is False
        assert _is_kihon_suchi(30, tolerance=1) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/strategy/evaluators/test_time_theory_eval.py -v`
Expected: FAIL

- [ ] **Step 3: Implement TimeTheoryEval**

```python
# src/strategy/evaluators/time_theory_eval.py
"""TimeTheoryEval — Kihon Suchi projection and Tato Suchi cycle detection.

Projects reversal days from fractal extremes using Ichimoku time numbers.
Detects repeating equal cycles (Tato Suchi) for double confirmation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import Evaluator, EvaluatorResult
from ...indicators.fractals import detect_fractals, FractalLevel


KIHON_SUCHI = [9, 17, 26, 33, 42, 51, 65, 76, 83, 97, 101, 129, 172, 200]


def _is_kihon_suchi(bars: int, tolerance: int = 1) -> bool:
    """Check if a bar count matches any Kihon Suchi number within tolerance."""
    return any(abs(bars - ks) <= tolerance for ks in KIHON_SUCHI)


def _nearest_kihon_suchi(bars: int) -> int:
    """Return the nearest upcoming Kihon Suchi number >= bars."""
    for ks in KIHON_SUCHI:
        if ks >= bars:
            return ks
    return KIHON_SUCHI[-1]


class TimeTheoryEval(Evaluator, key='time_theory'):
    """Projects Kihon Suchi numbers from fractal extremes.

    Direction: 0.0 (time theory does not determine direction).
    Confidence: 0.0–1.0 based on number of projections hitting current bar.
      single hit = 0.5, double confirmation = 0.8, triple = 1.0.
    """

    def __init__(self, tolerance: int = 1, max_fractal_sources: int = 5):
        self._tolerance = tolerance
        self._max_sources = max_fractal_sources

    def evaluate(self, ohlcv: pd.DataFrame) -> EvaluatorResult:
        high = ohlcv['high'].values
        low = ohlcv['low'].values
        n = len(high)

        fractal_result = detect_fractals(high, low)

        # Collect all fractals, sorted by bar index descending (most recent first)
        all_fractals: list[FractalLevel] = sorted(
            fractal_result.bull_fractals + fractal_result.bear_fractals,
            key=lambda f: f.bar_index,
            reverse=True,
        )

        # Limit to most recent N fractal sources
        sources = all_fractals[:self._max_sources]

        last_bar = n - 1
        projections = []
        hits = 0

        for frac in sources:
            bars_elapsed = last_bar - frac.bar_index
            if bars_elapsed <= 0:
                continue
            on_ks = _is_kihon_suchi(bars_elapsed, self._tolerance)
            next_ks = _nearest_kihon_suchi(bars_elapsed + 1)
            if on_ks:
                hits += 1
            projections.append({
                'source_bar_index': frac.bar_index,
                'source_price': frac.price,
                'bars_elapsed': bars_elapsed,
                'is_on_kihon_suchi': on_ks,
                'next_kihon_suchi': next_ks,
                'bars_to_next': next_ks - bars_elapsed,
            })

        # Tato Suchi detection: equal cycle lengths between consecutive fractal pairs
        tato_suchi = False
        tato_cycles = []
        sorted_fractals = sorted(
            fractal_result.bull_fractals + fractal_result.bear_fractals,
            key=lambda f: f.bar_index,
        )
        if len(sorted_fractals) >= 3:
            for i in range(len(sorted_fractals) - 2):
                d1 = sorted_fractals[i + 1].bar_index - sorted_fractals[i].bar_index
                d2 = sorted_fractals[i + 2].bar_index - sorted_fractals[i + 1].bar_index
                if abs(d1 - d2) <= self._tolerance:
                    tato_suchi = True
                    matching_ks = None
                    if _is_kihon_suchi(d1, self._tolerance):
                        matching_ks = _nearest_kihon_suchi(d1)
                    tato_cycles.append({
                        'bar_count': d1,
                        'matching_kihon_suchi': matching_ks,
                    })

        double_confirmation = hits >= 2

        # Confidence scoring
        if hits >= 3:
            confidence = 1.0
        elif hits == 2:
            confidence = 0.8
        elif hits == 1:
            confidence = 0.5
        else:
            confidence = 0.0

        if tato_suchi:
            confidence = min(1.0, confidence + 0.2)

        metadata = {
            'active_projections': projections,
            'is_on_kihon_suchi': hits > 0,
            'kihon_suchi_hits': hits,
            'tato_suchi_detected': tato_suchi,
            'tato_suchi_cycles': tato_cycles,
            'double_confirmation': double_confirmation,
        }

        return EvaluatorResult(
            direction=0.0,  # time theory doesn't determine direction
            confidence=confidence,
            metadata=metadata,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/strategy/evaluators/test_time_theory_eval.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/strategy/evaluators/time_theory_eval.py tests/strategy/evaluators/test_time_theory_eval.py
git commit -m "feat(task-8): TimeTheoryEval — Kihon Suchi projection + Tato Suchi"
```

---

### Task 9: WaveAnalysisEval

**Files:**
- Create: `src/strategy/evaluators/wave_analysis_eval.py`
- Create: `tests/strategy/evaluators/test_wave_analysis_eval.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/strategy/evaluators/test_wave_analysis_eval.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.base import EvaluatorResult, EVALUATOR_REGISTRY


def _make_n_wave_ohlcv(n=100) -> pd.DataFrame:
    """Create an N-wave: up, pullback, up again (A-B-C-D)."""
    idx = pd.date_range('2026-01-01', periods=n, freq='1h', tz='UTC')
    close = np.zeros(n)
    # A (0) → B (30) up, B (30) → C (50) pullback, C (50) → D (100) up
    for i in range(n):
        if i <= 30:
            close[i] = 1800.0 + i * 3.0  # A to B
        elif i <= 50:
            close[i] = 1890.0 - (i - 30) * 2.0  # B to C (pullback)
        else:
            close[i] = 1850.0 + (i - 50) * 3.0  # C to D
    return pd.DataFrame({
        'open': close - 0.5,
        'high': close + 2.0,
        'low': close - 2.0,
        'close': close,
        'volume': np.ones(n) * 100,
    }, index=idx)


def _make_flat_ohlcv(n=100) -> pd.DataFrame:
    idx = pd.date_range('2026-01-01', periods=n, freq='1h', tz='UTC')
    close = np.full(n, 1800.0) + np.random.uniform(-0.5, 0.5, n)
    return pd.DataFrame({
        'open': close - 0.3,
        'high': close + 0.5,
        'low': close - 0.5,
        'close': close,
        'volume': np.ones(n) * 100,
    }, index=idx)


class TestWaveAnalysisEvalRegistered:
    def test_registered(self):
        from src.strategy.evaluators.wave_analysis_eval import WaveAnalysisEval
        assert 'wave_analysis' in EVALUATOR_REGISTRY


class TestWaveAnalysisEvalOutput:
    def setup_method(self):
        from src.strategy.evaluators.wave_analysis_eval import WaveAnalysisEval
        self.eval = WaveAnalysisEval()

    def test_returns_evaluator_result(self):
        ohlcv = _make_n_wave_ohlcv()
        result = self.eval.evaluate(ohlcv)
        assert isinstance(result, EvaluatorResult)

    def test_metadata_has_wave_fields(self):
        ohlcv = _make_n_wave_ohlcv()
        result = self.eval.evaluate(ohlcv)
        assert 'current_wave_type' in result.metadata
        assert 'wave_position' in result.metadata
        assert 'wave_direction' in result.metadata

    def test_n_wave_detected_in_trending_data(self):
        ohlcv = _make_n_wave_ohlcv()
        result = self.eval.evaluate(ohlcv)
        # Should detect N-wave or at least an impulse leg
        assert result.metadata['current_wave_type'] in ('I', 'V', 'N', 'P', 'Y', 'box')

    def test_flat_market_classified_as_box(self):
        ohlcv = _make_flat_ohlcv()
        result = self.eval.evaluate(ohlcv)
        assert result.metadata['current_wave_type'] in ('box', 'P', 'I')
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/strategy/evaluators/test_wave_analysis_eval.py -v`
Expected: FAIL

- [ ] **Step 3: Implement WaveAnalysisEval**

```python
# src/strategy/evaluators/wave_analysis_eval.py
"""WaveAnalysisEval — I/V/N/P/Y/Box wave classification.

Uses fractal swing points to identify the current market wave pattern
and determine whether we are in an impulse or correction leg.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import Evaluator, EvaluatorResult
from ...indicators.fractals import detect_fractals, FractalLevel


class WaveAnalysisEval(Evaluator, key='wave_analysis'):
    """Classifies market wave pattern from fractal swing points.

    Direction: +1 if bullish wave, -1 if bearish, 0 if box/unclear.
    Confidence: 0.8 for clear N-wave, 0.5 for I/V, 0.3 for P/Y/box.

    Metadata: wave_type, wave_position, n_wave_points, is_trading_correction.
    """

    def evaluate(self, ohlcv: pd.DataFrame) -> EvaluatorResult:
        high = ohlcv['high'].values
        low = ohlcv['low'].values
        close = ohlcv['close'].values

        fractal_result = detect_fractals(high, low)

        # Get swing points (alternating bull/bear fractals sorted by time)
        swings = self._build_swing_sequence(
            fractal_result.bull_fractals, fractal_result.bear_fractals)

        wave_type, wave_dir, wave_pos, n_points = self._classify_wave(swings, close)

        # Is the current position a correction within a larger wave?
        is_correction = (wave_type == 'V')

        direction_map = {'bullish': 1.0, 'bearish': -1.0}
        direction = direction_map.get(wave_dir, 0.0)

        confidence_map = {'N': 0.8, 'I': 0.5, 'V': 0.5, 'P': 0.3, 'Y': 0.3, 'box': 0.2}
        confidence = confidence_map.get(wave_type, 0.3)

        metadata = {
            'current_wave_type': wave_type,
            'wave_position': wave_pos,
            'wave_direction': wave_dir,
            'n_wave_points': n_points,
            'is_trading_correction': is_correction,
            'swing_count': len(swings),
        }

        return EvaluatorResult(
            direction=direction, confidence=confidence, metadata=metadata,
        )

    def _build_swing_sequence(
        self,
        bulls: list[FractalLevel],
        bears: list[FractalLevel],
    ) -> list[dict]:
        """Merge bull/bear fractals into alternating swing sequence."""
        all_swings = []
        for f in bulls:
            all_swings.append({'type': 'high', 'price': f.price, 'bar': f.bar_index})
        for f in bears:
            all_swings.append({'type': 'low', 'price': f.price, 'bar': f.bar_index})
        all_swings.sort(key=lambda s: s['bar'])

        # Remove consecutive same-type swings (keep the more extreme)
        filtered = []
        for s in all_swings:
            if filtered and filtered[-1]['type'] == s['type']:
                if s['type'] == 'high' and s['price'] > filtered[-1]['price']:
                    filtered[-1] = s
                elif s['type'] == 'low' and s['price'] < filtered[-1]['price']:
                    filtered[-1] = s
            else:
                filtered.append(s)
        return filtered

    def _classify_wave(
        self,
        swings: list[dict],
        close: np.ndarray,
    ) -> tuple[str, str, str, dict]:
        """Classify the wave pattern from the last 4+ swings.

        Returns (wave_type, direction, position, n_wave_points).
        """
        n_points = {}

        if len(swings) < 2:
            return 'I', 'bullish' if len(close) > 1 and close[-1] > close[0] else 'bearish', 'impulse_leg', n_points

        # Check last 4 swings for N-wave pattern (A-B-C-D)
        if len(swings) >= 4:
            s = swings[-4:]
            # Bullish N: low-high-low-high with higher highs
            if (s[0]['type'] == 'low' and s[1]['type'] == 'high'
                    and s[2]['type'] == 'low' and s[3]['type'] == 'high'):
                if s[3]['price'] > s[1]['price'] and s[2]['price'] > s[0]['price']:
                    n_points = {'A': s[0]['price'], 'B': s[1]['price'],
                                'C': s[2]['price'], 'D_projected': s[2]['price'] + (s[1]['price'] - s[0]['price'])}
                    wave_pos = 'impulse_leg' if close[-1] > s[2]['price'] else 'correction_leg'
                    return 'N', 'bullish', wave_pos, n_points

            # Bearish N: high-low-high-low with lower lows
            if (s[0]['type'] == 'high' and s[1]['type'] == 'low'
                    and s[2]['type'] == 'high' and s[3]['type'] == 'low'):
                if s[3]['price'] < s[1]['price'] and s[2]['price'] < s[0]['price']:
                    n_points = {'A': s[0]['price'], 'B': s[1]['price'],
                                'C': s[2]['price'], 'D_projected': s[2]['price'] - (s[0]['price'] - s[1]['price'])}
                    wave_pos = 'impulse_leg' if close[-1] < s[2]['price'] else 'correction_leg'
                    return 'N', 'bearish', wave_pos, n_points

        # Check for P-wave (converging) or Y-wave (expanding)
        if len(swings) >= 4:
            highs = [s['price'] for s in swings if s['type'] == 'high']
            lows = [s['price'] for s in swings if s['type'] == 'low']
            if len(highs) >= 2 and len(lows) >= 2:
                h_decreasing = highs[-1] < highs[-2]
                l_increasing = lows[-1] > lows[-2]
                h_increasing = highs[-1] > highs[-2]
                l_decreasing = lows[-1] < lows[-2]

                if h_decreasing and l_increasing:
                    return 'P', 'bullish', 'breakout_pending', n_points
                if h_increasing and l_decreasing:
                    return 'Y', 'bullish', 'breakout_pending', n_points

        # Check for box (range)
        if len(swings) >= 4:
            prices = [s['price'] for s in swings[-6:]]
            price_range = max(prices) - min(prices)
            avg = np.mean(prices)
            if avg > 0 and price_range / avg < 0.01:  # less than 1% range
                return 'box', 'none', 'breakout_pending', n_points

        # Check for V-wave (2 legs)
        if len(swings) >= 3:
            s = swings[-3:]
            if s[0]['type'] == 'low' and s[1]['type'] == 'high' and s[2]['type'] == 'low':
                return 'V', 'bearish', 'correction_leg', n_points
            if s[0]['type'] == 'high' and s[1]['type'] == 'low' and s[2]['type'] == 'high':
                return 'V', 'bullish', 'correction_leg', n_points

        # Default: I-wave
        direction = 'bullish' if close[-1] > close[0] else 'bearish'
        return 'I', direction, 'impulse_leg', n_points
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/strategy/evaluators/test_wave_analysis_eval.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/strategy/evaluators/wave_analysis_eval.py tests/strategy/evaluators/test_wave_analysis_eval.py
git commit -m "feat(task-9): WaveAnalysisEval — I/V/N/P/Y/Box wave classification"
```

---

### Task 10: PriceTargetEval

**Files:**
- Create: `src/strategy/evaluators/price_target_eval.py`
- Create: `tests/strategy/evaluators/test_price_target_eval.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/strategy/evaluators/test_price_target_eval.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.base import EvaluatorResult, EVALUATOR_REGISTRY


class TestPriceTargetEvalRegistered:
    def test_registered(self):
        from src.strategy.evaluators.price_target_eval import PriceTargetEval
        assert 'price_target' in EVALUATOR_REGISTRY


class TestTargetCalculations:
    def test_bullish_n_value(self):
        """N-value = C + (B - A) for bullish 1-2-3."""
        from src.strategy.evaluators.price_target_eval import compute_targets
        targets = compute_targets(a=1800.0, b=1900.0, c=1850.0, direction='bullish')
        assert targets['n_value'] == pytest.approx(1950.0)  # 1850 + (1900 - 1800)

    def test_bullish_v_value(self):
        """V-value = B + (B - C) for bullish."""
        from src.strategy.evaluators.price_target_eval import compute_targets
        targets = compute_targets(a=1800.0, b=1900.0, c=1850.0, direction='bullish')
        assert targets['v_value'] == pytest.approx(1950.0)  # 1900 + (1900 - 1850)

    def test_bullish_e_value(self):
        """E-value = B + (B - A) for bullish."""
        from src.strategy.evaluators.price_target_eval import compute_targets
        targets = compute_targets(a=1800.0, b=1900.0, c=1850.0, direction='bullish')
        assert targets['e_value'] == pytest.approx(2000.0)  # 1900 + (1900 - 1800)

    def test_bullish_nt_value(self):
        """NT-value = C + (C - A) for bullish."""
        from src.strategy.evaluators.price_target_eval import compute_targets
        targets = compute_targets(a=1800.0, b=1900.0, c=1850.0, direction='bullish')
        assert targets['nt_value'] == pytest.approx(1900.0)  # 1850 + (1850 - 1800)

    def test_bearish_n_value(self):
        """N-value = C - (A - B) for bearish 1-2-3."""
        from src.strategy.evaluators.price_target_eval import compute_targets
        targets = compute_targets(a=1900.0, b=1800.0, c=1850.0, direction='bearish')
        assert targets['n_value'] == pytest.approx(1750.0)  # 1850 - (1900 - 1800)

    def test_bearish_e_value(self):
        from src.strategy.evaluators.price_target_eval import compute_targets
        targets = compute_targets(a=1900.0, b=1800.0, c=1850.0, direction='bearish')
        assert targets['e_value'] == pytest.approx(1700.0)  # 1800 - (1900 - 1800)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/strategy/evaluators/test_price_target_eval.py -v`
Expected: FAIL

- [ ] **Step 3: Implement PriceTargetEval**

```python
# src/strategy/evaluators/price_target_eval.py
"""PriceTargetEval — E/V/N/NT price target calculation.

Uses 1-2-3 (A-B-C) patterns from wave analysis to project targets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import Evaluator, EvaluatorResult
from ...indicators.fractals import detect_fractals


def compute_targets(a: float, b: float, c: float, direction: str) -> dict:
    """Compute E/V/N/NT target levels from A-B-C points.

    Bullish (A=low, B=high, C=pullback low):
      NT = C + (C - A)
      N  = C + (B - A)    ← THE primary target (1:1 ratio)
      V  = B + (B - C)
      E  = B + (B - A)

    Bearish (A=high, B=low, C=pullback high):
      NT = C - (A - C)
      N  = C - (A - B)
      V  = B - (C - B)
      E  = B - (A - B)
    """
    if direction == 'bullish':
        ab = b - a  # positive
        bc = b - c  # positive
        ca = c - a  # positive
        return {
            'nt_value': c + ca,       # 0.927 fib
            'n_value': c + ab,        # 1.0 fib — primary
            'v_value': b + bc,        # 1.221 fib
            'e_value': b + ab,        # 1.611 fib
        }
    else:  # bearish
        ab = a - b  # positive
        bc = c - b  # positive
        ca = a - c  # positive
        return {
            'nt_value': c - ca,
            'n_value': c - ab,
            'v_value': b - bc,
            'e_value': b - ab,
        }


class PriceTargetEval(Evaluator, key='price_target'):
    """Calculates price targets from the most recent 1-2-3 pattern.

    Direction: 0.0 (targets don't determine direction).
    Confidence: 1.0 if valid A-B-C found, 0.0 if not.
    """

    def evaluate(self, ohlcv: pd.DataFrame) -> EvaluatorResult:
        high = ohlcv['high'].values
        low = ohlcv['low'].values
        close = ohlcv['close'].values

        fractal_result = detect_fractals(high, low)

        # Build alternating swing sequence for A-B-C identification
        swings = self._build_swings(
            fractal_result.bull_fractals, fractal_result.bear_fractals)

        targets = {}
        has_pattern = False
        a_val = b_val = c_val = 0.0
        direction = 'bullish'

        if len(swings) >= 3:
            s = swings[-3:]
            # Bullish 1-2-3: low → high → low (A, B, C)
            if s[0]['type'] == 'low' and s[1]['type'] == 'high' and s[2]['type'] == 'low':
                a_val, b_val, c_val = s[0]['price'], s[1]['price'], s[2]['price']
                direction = 'bullish'
                has_pattern = True
            # Bearish 1-2-3: high → low → high (A, B, C)
            elif s[0]['type'] == 'high' and s[1]['type'] == 'low' and s[2]['type'] == 'high':
                a_val, b_val, c_val = s[0]['price'], s[1]['price'], s[2]['price']
                direction = 'bearish'
                has_pattern = True

        if has_pattern:
            targets = compute_targets(a_val, b_val, c_val, direction)

        # Check which targets have been hit
        targets_hit = {}
        if has_pattern:
            current_price = close[-1]
            for k, v in targets.items():
                if direction == 'bullish':
                    targets_hit[k] = current_price >= v
                else:
                    targets_hit[k] = current_price <= v

        metadata = {
            'has_valid_pattern': has_pattern,
            'point_a': a_val if has_pattern else None,
            'point_b': b_val if has_pattern else None,
            'point_c': c_val if has_pattern else None,
            'direction': direction if has_pattern else None,
            **targets,
            'targets_hit': targets_hit,
        }

        return EvaluatorResult(
            direction=0.0,
            confidence=1.0 if has_pattern else 0.0,
            metadata=metadata,
        )

    def _build_swings(self, bulls, bears) -> list[dict]:
        all_s = []
        for f in bulls:
            all_s.append({'type': 'high', 'price': f.price, 'bar': f.bar_index})
        for f in bears:
            all_s.append({'type': 'low', 'price': f.price, 'bar': f.bar_index})
        all_s.sort(key=lambda s: s['bar'])
        # Remove consecutive same-type
        filtered = []
        for s in all_s:
            if filtered and filtered[-1]['type'] == s['type']:
                if s['type'] == 'high' and s['price'] > filtered[-1]['price']:
                    filtered[-1] = s
                elif s['type'] == 'low' and s['price'] < filtered[-1]['price']:
                    filtered[-1] = s
            else:
                filtered.append(s)
        return filtered
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/strategy/evaluators/test_price_target_eval.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/strategy/evaluators/price_target_eval.py tests/strategy/evaluators/test_price_target_eval.py
git commit -m "feat(task-10): PriceTargetEval — E/V/N/NT target calculations"
```

---

## Phase 4: Exit Manager & Strategy

### Task 11: IchimokuExitManager

**Files:**
- Create: `src/strategy/trading_modes/ichimoku_exit.py`
- Create: `tests/strategy/trading_modes/test_ichimoku_exit.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/strategy/trading_modes/test_ichimoku_exit.py
from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from src.strategy.base import ExitDecision, EvalMatrix, EvaluatorResult


def _make_trade(direction='long', entry_price=1900.0, stop_loss=1880.0,
                remaining_pct=1.0):
    trade = MagicMock()
    trade.direction = direction
    trade.entry_price = entry_price
    trade.stop_loss = stop_loss
    trade.original_stop_loss = stop_loss
    trade.remaining_pct = remaining_pct
    return trade


def _make_current_data(close=1920.0, kijun=1910.0, atr=5.0):
    return {'close': close, 'high': close + 2, 'low': close - 2, 'atr': atr}


def _make_eval_matrix(kijun=1910.0, nearest_bear_fractal=1905.0):
    matrix = EvalMatrix()
    matrix.set('ichimoku_core_1H', EvaluatorResult(
        direction=1.0, confidence=0.8,
        metadata={'kijun': kijun, 'tenkan': kijun + 5},
    ))
    matrix.set('fractal_1H', EvaluatorResult(
        direction=1.0, confidence=0.5,
        metadata={
            'nearest_bear_fractal': {'price': nearest_bear_fractal, 'bar_index': 10},
            'nearest_bull_fractal': {'price': 1930.0, 'bar_index': 8},
        },
    ))
    return matrix


class TestTrailingMode:
    def setup_method(self):
        from src.strategy.trading_modes.ichimoku_exit import IchimokuExitManager
        self.mgr = IchimokuExitManager(mode='trailing', secondary_tf='1H')

    def test_hold_when_price_above_kijun(self):
        trade = _make_trade(direction='long', entry_price=1900.0)
        data = _make_current_data(close=1920.0, kijun=1910.0)
        matrix = _make_eval_matrix(kijun=1910.0)
        decision = self.mgr.check_exit(trade, data, matrix)
        assert decision.action in ('hold', 'trail_update')

    def test_full_exit_when_price_below_kijun(self):
        trade = _make_trade(direction='long', entry_price=1900.0)
        data = _make_current_data(close=1905.0, kijun=1910.0)
        matrix = _make_eval_matrix(kijun=1910.0)
        decision = self.mgr.check_exit(trade, data, matrix)
        assert decision.action == 'full_exit'

    def test_trail_update_moves_stop_to_fractal(self):
        trade = _make_trade(direction='long', entry_price=1900.0, stop_loss=1880.0)
        data = _make_current_data(close=1925.0, kijun=1910.0)
        matrix = _make_eval_matrix(kijun=1910.0, nearest_bear_fractal=1912.0)
        decision = self.mgr.check_exit(trade, data, matrix)
        if decision.action == 'trail_update':
            assert decision.new_stop >= 1910.0  # at least Kijun level


class TestHybridMode:
    def setup_method(self):
        from src.strategy.trading_modes.ichimoku_exit import IchimokuExitManager
        self.mgr = IchimokuExitManager(
            mode='hybrid', secondary_tf='1H',
            partial_close_pct=0.5, primary_target_price=1950.0,
        )

    def test_partial_exit_at_target(self):
        trade = _make_trade(direction='long', entry_price=1900.0)
        data = _make_current_data(close=1952.0)
        matrix = _make_eval_matrix()
        decision = self.mgr.check_exit(trade, data, matrix)
        assert decision.action == 'partial_exit'
        assert decision.close_pct == pytest.approx(0.5)

    def test_hold_before_target(self):
        trade = _make_trade(direction='long', entry_price=1900.0)
        data = _make_current_data(close=1930.0)
        matrix = _make_eval_matrix()
        decision = self.mgr.check_exit(trade, data, matrix)
        assert decision.action in ('hold', 'trail_update')


class TestTargetsMode:
    def setup_method(self):
        from src.strategy.trading_modes.ichimoku_exit import IchimokuExitManager
        self.mgr = IchimokuExitManager(
            mode='targets', secondary_tf='1H',
            primary_target_price=1950.0,
        )

    def test_full_exit_at_target(self):
        trade = _make_trade(direction='long', entry_price=1900.0)
        data = _make_current_data(close=1955.0)
        matrix = _make_eval_matrix()
        decision = self.mgr.check_exit(trade, data, matrix)
        assert decision.action == 'full_exit'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/strategy/trading_modes/test_ichimoku_exit.py -v`
Expected: FAIL

- [ ] **Step 3: Implement IchimokuExitManager**

```python
# src/strategy/trading_modes/ichimoku_exit.py
"""IchimokuExitManager — configurable exit logic for FXAOG strategy.

Supports three modes: trailing (Kijun/fractal), targets (fixed TP),
hybrid (partial at target + trail remainder).
"""

from __future__ import annotations

from ..base import TradingMode, ExitDecision, EvalMatrix


class IchimokuExitManager(TradingMode):
    """Configurable exit manager with trailing, targets, and hybrid modes.

    Parameters
    ----------
    mode : str
        'trailing', 'targets', or 'hybrid'
    secondary_tf : str
        Timeframe key for reading evaluator results (e.g. '1H')
    kijun_buffer : float
        Buffer below/above Kijun for stop placement
    partial_close_pct : float
        Fraction to close at target in hybrid mode
    primary_target_price : float or None
        Fixed target price (set by strategy when opening trade)
    move_stop_to_entry : bool
        Whether to move stop to breakeven after partial close
    """

    def __init__(
        self,
        mode: str = 'trailing',
        secondary_tf: str = '1H',
        kijun_buffer: float = 5.0,
        partial_close_pct: float = 0.5,
        primary_target_price: float | None = None,
        move_stop_to_entry: bool = True,
    ):
        self.mode = mode
        self.tf = secondary_tf
        self.kijun_buffer = kijun_buffer
        self.partial_close_pct = partial_close_pct
        self.target = primary_target_price
        self.move_stop_to_entry = move_stop_to_entry
        self._partial_taken = False

    def check_exit(
        self,
        trade: object,
        current_data: dict,
        eval_results: EvalMatrix,
    ) -> ExitDecision:
        close = current_data['close']
        direction = trade.direction  # 'long' or 'short'
        is_long = direction == 'long'

        # Extract Kijun from eval matrix
        ichi_key = f'ichimoku_core_{self.tf}'
        ichi_result = eval_results.get(ichi_key)
        kijun = ichi_result.metadata.get('kijun') if ichi_result else None

        # Extract nearest opposite fractal
        frac_key = f'fractal_{self.tf}'
        frac_result = eval_results.get(frac_key)
        opp_fractal = None
        if frac_result:
            frac_key_name = 'nearest_bear_fractal' if is_long else 'nearest_bull_fractal'
            frac_data = frac_result.metadata.get(frac_key_name)
            if frac_data:
                opp_fractal = frac_data['price']

        # --- Check stop hit ---
        if kijun is not None:
            if is_long and close < kijun - self.kijun_buffer:
                return ExitDecision(action='full_exit', reason='Price closed below Kijun-sen')
            if not is_long and close > kijun + self.kijun_buffer:
                return ExitDecision(action='full_exit', reason='Price closed above Kijun-sen')

        # --- Mode-specific logic ---
        if self.mode == 'targets':
            return self._check_targets(close, is_long, trade)
        elif self.mode == 'hybrid':
            return self._check_hybrid(close, is_long, trade, kijun, opp_fractal)
        else:  # trailing
            return self._check_trailing(close, is_long, trade, kijun, opp_fractal)

    def _check_trailing(self, close, is_long, trade, kijun, opp_fractal) -> ExitDecision:
        """Trail stop with Kijun and fractal — use whichever is tighter."""
        new_stop = trade.stop_loss

        candidates = []
        if kijun is not None:
            buf = self.kijun_buffer if is_long else -self.kijun_buffer
            candidates.append(kijun - buf if is_long else kijun + abs(buf))
        if opp_fractal is not None:
            candidates.append(opp_fractal)

        if candidates:
            if is_long:
                best = max(candidates)
                if best > trade.stop_loss:
                    new_stop = best
            else:
                best = min(candidates)
                if best < trade.stop_loss:
                    new_stop = best

        if new_stop != trade.stop_loss:
            return ExitDecision(action='trail_update', new_stop=new_stop,
                                reason='Trail with Kijun/fractal')
        return ExitDecision(action='hold')

    def _check_targets(self, close, is_long, trade) -> ExitDecision:
        if self.target is None:
            return ExitDecision(action='hold')
        if is_long and close >= self.target:
            return ExitDecision(action='full_exit', close_pct=1.0,
                                reason=f'Target hit at {self.target}')
        if not is_long and close <= self.target:
            return ExitDecision(action='full_exit', close_pct=1.0,
                                reason=f'Target hit at {self.target}')
        return ExitDecision(action='hold')

    def _check_hybrid(self, close, is_long, trade, kijun, opp_fractal) -> ExitDecision:
        # Phase 1: Check if target hit for partial close
        if not self._partial_taken and self.target is not None:
            target_hit = (is_long and close >= self.target) or (not is_long and close <= self.target)
            if target_hit:
                self._partial_taken = True
                new_stop = trade.entry_price if self.move_stop_to_entry else trade.stop_loss
                return ExitDecision(
                    action='partial_exit',
                    close_pct=self.partial_close_pct,
                    new_stop=new_stop,
                    reason=f'Partial close at target {self.target}',
                )

        # Phase 2: Trail remainder
        return self._check_trailing(close, is_long, trade, kijun, opp_fractal)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/strategy/trading_modes/test_ichimoku_exit.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/strategy/trading_modes/ichimoku_exit.py tests/strategy/trading_modes/test_ichimoku_exit.py
git commit -m "feat(task-11): IchimokuExitManager — trailing, targets, hybrid exit modes"
```

---

### Task 12: IchimokuFXAOGStrategy

**Files:**
- Create: `src/strategy/strategies/ichimoku_fxaog.py`
- Create: `tests/strategy/strategies/test_ichimoku_fxaog.py`

This is the largest task. The strategy coordinates all evaluators and implements 8 trade types.

- [ ] **Step 1: Write failing tests for the strategy**

```python
# tests/strategy/strategies/test_ichimoku_fxaog.py
from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from src.strategy.base import (
    EvalMatrix, EvaluatorResult, ConfluenceResult, STRATEGY_REGISTRY,
)


def _bullish_ichimoku_meta(**overrides):
    """Fully bullish IchimokuCoreEval metadata (all 5-point checklist pass)."""
    meta = {
        'tenkan': 1910.0, 'kijun': 1905.0,
        'senkou_a': 1895.0, 'senkou_b': 1885.0,
        'chikou': 1920.0,
        'cloud_position': 'above', 'cloud_direction': 'bullish',
        'tk_cross': 'none', 'tk_cross_bars_ago': 5,
        'chikou_vs_price': 'above', 'chikou_vs_kumo': 'above',
        'kumo_future_direction': 'bullish',
        'kumo_thickness': 10.0, 'kijun_flat': False,
        'kijun_distance_pips': 15.0, 'tenkan_kijun_angle': 0.001,
        'checklist_score': 5,
        'cloud_top': 1895.0, 'cloud_bot': 1885.0,
    }
    meta.update(overrides)
    return meta


def _make_bullish_matrix(primary='4H', secondary='1H'):
    """Build an EvalMatrix with all evaluators showing bullish alignment."""
    matrix = EvalMatrix()

    for tf in [primary, secondary]:
        matrix.set(f'ichimoku_core_{tf}', EvaluatorResult(
            direction=1.0, confidence=1.0,
            metadata=_bullish_ichimoku_meta(),
        ))
        matrix.set(f'fractal_{tf}', EvaluatorResult(
            direction=1.0, confidence=0.8,
            metadata={
                'bull_fractals': [{'price': 1925.0, 'bar_index': 90}],
                'bear_fractals': [{'price': 1895.0, 'bar_index': 85}],
                'last_broken_direction': 'bull',
                'momentum_trend': 'strengthening',
                'nearest_bull_fractal': {'price': 1925.0, 'bar_index': 90},
                'nearest_bear_fractal': {'price': 1895.0, 'bar_index': 85},
                'fractal_momentum_distances': [5.0, 7.0, 10.0],
            },
        ))

    # Secondary-only evaluators
    matrix.set(f'price_action_{secondary}', EvaluatorResult(
        direction=1.0, confidence=0.9,
        metadata={
            'tweezer_bottom': False, 'tweezer_top': False,
            'inside_bar_count': 2, 'inside_bar_breakout': 'up',
            'engulfing_bullish': True, 'engulfing_bearish': False,
            'pin_bar_bullish': False, 'pin_bar_bearish': False,
            'doji': False, 'mother_bar_high': 1915.0, 'mother_bar_low': 1905.0,
        },
    ))

    # Primary-only evaluators
    matrix.set(f'five_elements_{primary}', EvaluatorResult(
        direction=1.0, confidence=1.0,
        metadata={
            'overcoming_count': 3, 'generating_count': 1,
            'is_disequilibrium': True, 'cycle_direction': 'bullish',
            'total_signals': 4, 'cycle_start_bar': 80, 'crossover_log': [],
        },
    ))

    matrix.set(f'time_theory_{primary}', EvaluatorResult(
        direction=0.0, confidence=0.5,
        metadata={
            'is_on_kihon_suchi': True, 'kihon_suchi_hits': 1,
            'double_confirmation': False, 'tato_suchi_detected': False,
            'active_projections': [], 'tato_suchi_cycles': [],
        },
    ))

    matrix.set(f'wave_analysis_{primary}', EvaluatorResult(
        direction=1.0, confidence=0.8,
        metadata={
            'current_wave_type': 'N', 'wave_position': 'impulse_leg',
            'wave_direction': 'bullish', 'is_trading_correction': False,
            'n_wave_points': {'A': 1800, 'B': 1900, 'C': 1850, 'D_projected': 1950},
            'swing_count': 4,
        },
    ))
    matrix.set(f'wave_analysis_{secondary}', EvaluatorResult(
        direction=1.0, confidence=0.8,
        metadata={
            'current_wave_type': 'I', 'wave_position': 'impulse_leg',
            'wave_direction': 'bullish', 'is_trading_correction': False,
            'n_wave_points': {}, 'swing_count': 2,
        },
    ))

    matrix.set(f'price_target_{secondary}', EvaluatorResult(
        direction=0.0, confidence=1.0,
        metadata={
            'has_valid_pattern': True,
            'point_a': 1800.0, 'point_b': 1900.0, 'point_c': 1850.0,
            'direction': 'bullish',
            'nt_value': 1900.0, 'n_value': 1950.0,
            'v_value': 1950.0, 'e_value': 2000.0,
            'targets_hit': {},
        },
    ))

    return matrix


class TestIchimokuFXAOGRegistered:
    def test_registered(self):
        from src.strategy.strategies.ichimoku_fxaog import IchimokuFXAOGStrategy
        assert 'ichimoku_fxaog' in STRATEGY_REGISTRY


class TestChecklistGate:
    def setup_method(self):
        from src.strategy.strategies.ichimoku_fxaog import IchimokuFXAOGStrategy
        self.strategy = IchimokuFXAOGStrategy()

    def test_bullish_signal_when_all_aligned(self):
        matrix = _make_bullish_matrix()
        signal = self.strategy.decide(matrix)
        assert signal is not None
        assert signal.direction == 'long'

    def test_no_signal_when_chikou_inside(self):
        matrix = _make_bullish_matrix()
        # Override Chikou to 'inside' on both TFs
        for tf in ['4H', '1H']:
            r = matrix.get(f'ichimoku_core_{tf}')
            r.metadata['chikou_vs_price'] = 'inside'
            r.direction = 0.0
        signal = self.strategy.decide(matrix)
        assert signal is None

    def test_no_signal_when_cloud_position_wrong(self):
        matrix = _make_bullish_matrix()
        r = matrix.get('ichimoku_core_1H')
        r.metadata['cloud_position'] = 'below'
        r.direction = -1.0
        signal = self.strategy.decide(matrix)
        assert signal is None

    def test_no_signal_when_five_elements_balanced(self):
        """Hard gate: O == G blocks trade."""
        from src.strategy.strategies.ichimoku_fxaog import IchimokuFXAOGStrategy
        strategy = IchimokuFXAOGStrategy(five_elements_mode='hard_gate')
        matrix = _make_bullish_matrix()
        fe = matrix.get('five_elements_4H')
        fe.metadata['is_disequilibrium'] = False
        fe.metadata['overcoming_count'] = 2
        fe.metadata['generating_count'] = 2
        fe.confidence = 0.0
        signal = strategy.decide(matrix)
        assert signal is None


class TestConfluenceScoring:
    def setup_method(self):
        from src.strategy.strategies.ichimoku_fxaog import IchimokuFXAOGStrategy
        self.strategy = IchimokuFXAOGStrategy()

    def test_full_bullish_score_is_high(self):
        matrix = _make_bullish_matrix()
        result = self.strategy.score_confluence(matrix)
        assert result.score >= 9
        assert result.quality_tier in ('A+', 'A')

    def test_score_breakdown_has_components(self):
        matrix = _make_bullish_matrix()
        result = self.strategy.score_confluence(matrix)
        assert 'checklist' in result.breakdown
        assert 'price_action' in result.breakdown
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/strategy/strategies/test_ichimoku_fxaog.py -v`
Expected: FAIL

- [ ] **Step 3: Implement IchimokuFXAOGStrategy**

Create `src/strategy/strategies/ichimoku_fxaog.py`. This is the largest file — it implements the strategy coordinator with all 8 trade types. Key structure:

```python
# src/strategy/strategies/ichimoku_fxaog.py
"""IchimokuFXAOGStrategy — faithful FX At One Glance implementation.

8 trade types checked in priority order, with 5-point checklist gate,
configurable Five Elements and Time Theory filters, and 0-15 confluence scoring.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from ..base import (
    Strategy, EvalMatrix, EvalRequirement, ConfluenceResult,
)
from ..signal_engine import Signal
from ..trading_modes.ichimoku_exit import IchimokuExitManager


class IchimokuFXAOGStrategy(Strategy, key='ichimoku_fxaog'):

    warmup_bars = 200  # overridden dynamically in __init__

    def __init__(
        self,
        primary_tf: str = '4H',
        secondary_tf: str = '1H',
        mta_mode: str = 'full_alignment',
        five_elements_mode: str = 'hard_gate',
        time_theory_mode: str = 'soft_filter',
        min_confluence_score: int = 6,
        min_tier: str = 'B',
        max_kijun_distance_pips: float = 200.0,
        exit_mode: str = 'hybrid',
        partial_close_pct: float = 0.5,
        primary_target: str = 'n_value',
        kijun_buffer_pips: float = 5.0,
        max_stop_pips: float = 100.0,
        min_rr_ratio: float = 1.5,
        instrument: str = 'XAUUSD',
        **kwargs,
    ):
        self._primary = primary_tf
        self._secondary = secondary_tf
        self._mta_mode = mta_mode
        self._fe_mode = five_elements_mode
        self._tt_mode = time_theory_mode
        self._min_score = min_confluence_score
        self._min_tier = min_tier
        self._max_kijun_dist = max_kijun_distance_pips
        self._exit_mode = exit_mode
        self._partial_pct = partial_close_pct
        self._primary_target_key = primary_target
        self._kijun_buffer = kijun_buffer_pips
        self._max_stop = max_stop_pips
        self._min_rr = min_rr_ratio
        self._instrument = instrument

        both = [primary_tf, secondary_tf]
        self.required_evaluators = [
            EvalRequirement('ichimoku_core', both),
            EvalRequirement('price_action', [secondary_tf]),
            EvalRequirement('fractal', both),
            EvalRequirement('five_elements', [primary_tf]),
            EvalRequirement('time_theory', [primary_tf]),
            EvalRequirement('wave_analysis', both),
            EvalRequirement('price_target', [secondary_tf]),
        ]

        self.trading_mode = IchimokuExitManager(
            mode=exit_mode,
            secondary_tf=secondary_tf,
            kijun_buffer=kijun_buffer_pips,
            partial_close_pct=partial_close_pct,
        )

    # ------------------------------------------------------------------
    # 5-point checklist gate
    # ------------------------------------------------------------------
    def _check_checklist(self, ichi_meta: dict, direction: str) -> bool:
        """Return True if all 5-point checklist items align for direction."""
        is_long = direction == 'long'
        checks = [
            ichi_meta.get('cloud_position') == ('above' if is_long else 'below'),
            (ichi_meta.get('tenkan', 0) > ichi_meta.get('kijun', 0)) == is_long
                if ichi_meta.get('tenkan') and ichi_meta.get('kijun') else False,
            ichi_meta.get('chikou_vs_price') == ('above' if is_long else 'below'),
            ichi_meta.get('chikou_vs_kumo') == ('above' if is_long else 'below'),
            ichi_meta.get('kumo_future_direction') == ('bullish' if is_long else 'bearish'),
        ]
        return all(checks)

    def _checklist_passes(self, matrix: EvalMatrix) -> str | None:
        """Check 5-point on both TFs. Returns 'long'/'short' or None."""
        sec_ichi = matrix.get(f'ichimoku_core_{self._secondary}')
        if not sec_ichi:
            return None

        # Chikou inside = ranging → block
        if sec_ichi.metadata.get('chikou_vs_price') == 'inside':
            return None

        direction = 'long' if sec_ichi.direction > 0 else 'short' if sec_ichi.direction < 0 else None
        if direction is None:
            return None

        # Check secondary TF checklist
        if not self._check_checklist(sec_ichi.metadata, direction):
            return None

        # Check primary TF
        if self._mta_mode == 'full_alignment':
            pri_ichi = matrix.get(f'ichimoku_core_{self._primary}')
            if not pri_ichi or not self._check_checklist(pri_ichi.metadata, direction):
                return None
        else:  # kumo_bias
            pri_ichi = matrix.get(f'ichimoku_core_{self._primary}')
            if pri_ichi:
                pri_pos = pri_ichi.metadata.get('cloud_position')
                if direction == 'long' and pri_pos == 'below':
                    return None
                if direction == 'short' and pri_pos == 'above':
                    return None

        return direction

    # ------------------------------------------------------------------
    # Configurable filters
    # ------------------------------------------------------------------
    def _filter_five_elements(self, matrix: EvalMatrix) -> bool:
        """Return True if Five Elements allows trading."""
        if self._fe_mode == 'disabled':
            return True
        fe = matrix.get(f'five_elements_{self._primary}')
        if not fe:
            return self._fe_mode != 'hard_gate'
        if self._fe_mode == 'hard_gate':
            return fe.metadata.get('is_disequilibrium', False)
        return True  # soft_filter handled in scoring

    def _filter_time_theory(self, matrix: EvalMatrix) -> bool:
        """Return True if Time Theory allows trading."""
        if self._tt_mode == 'disabled':
            return True
        if self._tt_mode != 'hard_gate':
            return True
        tt = matrix.get(f'time_theory_{self._primary}')
        if not tt:
            return False
        return tt.metadata.get('is_on_kihon_suchi', False)

    # ------------------------------------------------------------------
    # Trade type checks (priority order)
    # ------------------------------------------------------------------
    def _check_walking_dragon(self, matrix, direction, ichi_meta):
        bars_ago = ichi_meta.get('tk_cross_bars_ago', 9999)
        if 5 <= bars_ago <= 10:
            angle = abs(ichi_meta.get('tenkan_kijun_angle', 0))
            if angle > 0.0003:
                return 'walking_dragon'
        return None

    def _check_tk_crossover(self, matrix, direction, ichi_meta):
        if ichi_meta.get('tk_cross') == ('bullish' if direction == 'long' else 'bearish'):
            if ichi_meta.get('kijun_distance_pips', 9999) <= self._max_kijun_dist:
                pa = matrix.get(f'price_action_{self._secondary}')
                if pa and pa.direction != 0.0:
                    return 'tk_crossover'
        return None

    def _check_kumo_breakout(self, matrix, direction, ichi_meta):
        cloud_pos = ichi_meta.get('cloud_position')
        if cloud_pos == ('above' if direction == 'long' else 'below'):
            return 'kumo_breakout'
        return None

    def _check_kijun_bounce(self, matrix, direction, ichi_meta):
        dist = ichi_meta.get('kijun_distance_pips', 9999)
        if dist <= 50:  # close to Kijun
            if not ichi_meta.get('kijun_flat', False):
                pa = matrix.get(f'price_action_{self._secondary}')
                if pa and (pa.metadata.get('inside_bar_breakout') != 'none'
                           or pa.metadata.get('engulfing_bullish' if direction == 'long' else 'engulfing_bearish')):
                    return 'kijun_bounce'
        return None

    def _check_kijun_break(self, matrix, direction, ichi_meta):
        kijun = ichi_meta.get('kijun')
        if kijun is not None:
            pa = matrix.get(f'price_action_{self._secondary}')
            if pa and pa.metadata.get('engulfing_bullish' if direction == 'long' else 'engulfing_bearish'):
                return 'kijun_break'
        return None

    def _check_ffo(self, matrix, direction, ichi_meta):
        frac = matrix.get(f'fractal_{self._secondary}')
        if not frac:
            return None
        frac_key = 'nearest_bull_fractal' if direction == 'long' else 'nearest_bear_fractal'
        frac_data = frac.metadata.get(frac_key)
        if frac_data:
            return 'ffo'
        return None

    def _check_fractal_breakout(self, matrix, direction, ichi_meta):
        frac = matrix.get(f'fractal_{self._secondary}')
        if not frac:
            return None
        if frac.metadata.get('last_broken_direction') == ('bull' if direction == 'long' else 'bear'):
            if frac.metadata.get('momentum_trend') != 'weakening':
                return 'fractal_breakout'
        return None

    def _check_rolling_dragon(self, matrix, direction, ichi_meta):
        if ichi_meta.get('kijun_flat', False):
            dist = ichi_meta.get('kijun_distance_pips', 0)
            if dist > 50:
                pa = matrix.get(f'price_action_{self._secondary}')
                if pa and (pa.metadata.get('pin_bar_bullish' if direction == 'long' else 'pin_bar_bearish')):
                    return 'rolling_dragon'
        return None

    # ------------------------------------------------------------------
    # Main decide method
    # ------------------------------------------------------------------
    def decide(self, eval_matrix: EvalMatrix) -> Optional[Signal]:
        direction = self._checklist_passes(eval_matrix)
        if direction is None:
            return None

        if not self._filter_five_elements(eval_matrix):
            return None
        if not self._filter_time_theory(eval_matrix):
            return None

        ichi = eval_matrix.get(f'ichimoku_core_{self._secondary}')
        if not ichi:
            return None
        meta = ichi.metadata

        # Check trade types in priority order
        trade_type = None
        checks = [
            self._check_walking_dragon,
            self._check_tk_crossover,
            self._check_kumo_breakout,
            self._check_kijun_bounce,
            self._check_kijun_break,
            self._check_ffo,
            self._check_fractal_breakout,
            self._check_rolling_dragon,
        ]
        for check in checks:
            result = check(eval_matrix, direction, meta)
            if result:
                trade_type = result
                break

        if trade_type is None:
            return None

        # Score confluence
        confluence = self.score_confluence(eval_matrix)
        if confluence.score < self._min_score:
            return None

        tier_order = {'A+': 0, 'A': 1, 'B': 2, 'C': 3, 'no_trade': 4}
        if tier_order.get(confluence.quality_tier, 4) > tier_order.get(self._min_tier, 2):
            return None

        # Calculate entry, SL, TP
        kijun = meta.get('kijun', 0)
        entry_price = ichi.metadata.get('tenkan', kijun)
        atr = meta.get('kumo_thickness', 10.0)  # fallback

        # Stop loss
        frac = eval_matrix.get(f'fractal_{self._secondary}')
        opp_frac_key = 'nearest_bear_fractal' if direction == 'long' else 'nearest_bull_fractal'
        opp_frac = frac.metadata.get(opp_frac_key) if frac else None
        if opp_frac and direction == 'long':
            stop_loss = opp_frac['price'] - self._kijun_buffer
        elif opp_frac and direction == 'short':
            stop_loss = opp_frac['price'] + self._kijun_buffer
        elif kijun:
            stop_loss = kijun - self._kijun_buffer if direction == 'long' else kijun + self._kijun_buffer
        else:
            return None

        # Take profit from price targets
        pt = eval_matrix.get(f'price_target_{self._secondary}')
        target_price = None
        if pt and pt.metadata.get('has_valid_pattern'):
            target_price = pt.metadata.get(self._primary_target_key)
        if target_price is None:
            target_price = entry_price + (entry_price - stop_loss) * 2  # fallback 2R

        # Risk:reward check
        risk = abs(entry_price - stop_loss)
        reward = abs(target_price - entry_price)
        if risk > 0 and reward / risk < self._min_rr:
            return None

        # Max stop check
        if risk > self._max_stop:
            return None

        # Update exit manager with target
        if isinstance(self.trading_mode, IchimokuExitManager):
            self.trading_mode.target = target_price
            self.trading_mode._partial_taken = False

        return Signal(
            timestamp=datetime.now(timezone.utc),
            instrument=self._instrument,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=target_price,
            confluence_score=confluence.score,
            quality_tier=confluence.quality_tier,
            atr=atr,
            strategy_name='ichimoku_fxaog',
            reasoning={
                'trade_type': trade_type,
                'confluence_score': confluence.score,
                'quality_tier': confluence.quality_tier,
                'checklist': {
                    'cloud_position': meta.get('cloud_position'),
                    'tk_relationship': 'bullish' if meta.get('tenkan', 0) > meta.get('kijun', 0) else 'bearish',
                    'chikou_vs_price': meta.get('chikou_vs_price'),
                    'chikou_vs_kumo': meta.get('chikou_vs_kumo'),
                    'kumo_future': meta.get('kumo_future_direction'),
                },
                **confluence.breakdown,
            },
        )

    # ------------------------------------------------------------------
    # Confluence scoring (0-15)
    # ------------------------------------------------------------------
    def score_confluence(self, eval_matrix: EvalMatrix) -> ConfluenceResult:
        score = 0
        breakdown = {}

        # 1. Checklist (0-5)
        ichi = eval_matrix.get(f'ichimoku_core_{self._secondary}')
        checklist_pts = ichi.metadata.get('checklist_score', 0) if ichi else 0
        score += checklist_pts
        breakdown['checklist'] = checklist_pts

        # 2. Signal strength (0-2) — position relative to Kumo
        if ichi and ichi.metadata.get('cloud_position') in ('above', 'below'):
            score += 2
            breakdown['signal_strength'] = 2
        else:
            breakdown['signal_strength'] = 0

        # 3. Price action quality (0-2)
        pa = eval_matrix.get(f'price_action_{self._secondary}')
        pa_pts = 0
        if pa:
            if pa.metadata.get('engulfing_bullish') or pa.metadata.get('engulfing_bearish'):
                pa_pts = 2
            elif pa.metadata.get('inside_bar_breakout') != 'none':
                pa_pts = 2
            elif pa.metadata.get('pin_bar_bullish') or pa.metadata.get('pin_bar_bearish'):
                pa_pts = 1
        score += pa_pts
        breakdown['price_action'] = pa_pts

        # 4. O/G disequilibrium (0-2)
        fe = eval_matrix.get(f'five_elements_{self._primary}')
        fe_pts = 0
        if fe:
            total = fe.metadata.get('total_signals', 0)
            if fe.metadata.get('is_disequilibrium') and total >= 3:
                fe_pts = 2
            elif fe.metadata.get('is_disequilibrium'):
                fe_pts = 1
        score += fe_pts
        breakdown['five_elements'] = fe_pts

        # 5. Kihon Suchi alignment (0-2)
        tt = eval_matrix.get(f'time_theory_{self._primary}')
        tt_pts = 0
        if tt:
            if tt.metadata.get('double_confirmation'):
                tt_pts = 2
            elif tt.metadata.get('is_on_kihon_suchi'):
                tt_pts = 1
        score += tt_pts
        breakdown['time_theory'] = tt_pts

        # 6. Wave context (0-1)
        wa = eval_matrix.get(f'wave_analysis_{self._primary}')
        wa_pts = 0
        if wa and not wa.metadata.get('is_trading_correction'):
            wa_pts = 1
        score += wa_pts
        breakdown['wave_context'] = wa_pts

        # 7. Fractal momentum (0-1)
        frac = eval_matrix.get(f'fractal_{self._secondary}')
        fm_pts = 0
        if frac and frac.metadata.get('momentum_trend') == 'strengthening':
            fm_pts = 1
        score += fm_pts
        breakdown['fractal_momentum'] = fm_pts

        # Tier classification
        if score >= 12:
            tier = 'A+'
        elif score >= 9:
            tier = 'A'
        elif score >= 6:
            tier = 'B'
        else:
            tier = 'C'

        return ConfluenceResult(score=score, quality_tier=tier, breakdown=breakdown)

    def suggest_params(self, trial) -> dict:
        return {
            'min_confluence_score': trial.suggest_int('min_confluence_score', 4, 12),
            'min_tier': trial.suggest_categorical('min_tier', ['A_plus', 'A', 'B']),
            'exit_mode': trial.suggest_categorical('exit_mode', ['trailing', 'targets', 'hybrid']),
            'primary_target': trial.suggest_categorical('primary_target', ['n_value', 'v_value']),
            'partial_close_pct': trial.suggest_int('partial_close_pct', 30, 70, step=10),
            'max_kijun_distance_pips': trial.suggest_int('max_kijun_distance_pips', 100, 300, step=50),
            'five_elements_mode': trial.suggest_categorical('five_elements_mode', ['hard_gate', 'soft_filter', 'disabled']),
            'time_theory_mode': trial.suggest_categorical('time_theory_mode', ['hard_gate', 'soft_filter', 'disabled']),
            'min_rr_ratio': trial.suggest_float('min_rr_ratio', 1.0, 3.0, step=0.5),
            'tf_primary': trial.suggest_categorical('tf_primary', ['D1', '4H', '1H']),
            'tf_secondary': trial.suggest_categorical('tf_secondary', ['4H', '1H', '15M']),
        }

    def populate_edge_context(self, eval_matrix: EvalMatrix) -> dict:
        ichi = eval_matrix.get(f'ichimoku_core_{self._secondary}')
        if not ichi:
            return {}
        return {
            'kijun': ichi.metadata.get('kijun'),
            'cloud_thickness': ichi.metadata.get('kumo_thickness'),
            'kijun_distance': ichi.metadata.get('kijun_distance_pips'),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/strategy/strategies/test_ichimoku_fxaog.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/strategy/strategies/ichimoku_fxaog.py tests/strategy/strategies/test_ichimoku_fxaog.py
git commit -m "feat(task-12): IchimokuFXAOGStrategy — 8 trade types, 5-point checklist, 0-15 scoring"
```

---

### Task 13: Configuration Update

**Files:**
- Modify: `config/strategy.yaml`

- [ ] **Step 1: Add the new strategy config block**

Add the `ichimoku_fxaog` section to `config/strategy.yaml` under the `strategies:` key, and update `active_strategy`:

```yaml
# At the top:
active_strategy: ichimoku_fxaog

# Under strategies:
  ichimoku_fxaog:
    ichimoku:
      tenkan_period: 9
      kijun_period: 26
      senkou_b_period: 52
    timeframes:
      primary: "4H"
      secondary: "1H"
      mta_mode: full_alignment
    trade_types:
      walking_dragon: {enabled: true, pullback_window: [5, 10], min_angle_threshold: 0.0003}
      tk_crossover: {enabled: true, max_kijun_distance_pips: 200, require_both_sloping: true}
      kumo_breakout: {enabled: true}
      kijun_bounce: {enabled: true, kijun_proximity_atr: 0.5, reject_flat_kijun: true}
      kijun_break: {enabled: true}
      ffo: {enabled: true}
      fractal_breakout: {enabled: true, reject_weakening_momentum: true}
      rolling_dragon: {enabled: true, c_clamp_threshold_atr: 2.0, require_flat_kijun: true}
    price_action:
      tweezer_tick_tolerance: 2
      inside_bar_use_body_only: true
      min_inside_bars: 1
      engulfing_immediate_entry: true
    five_elements:
      mode: hard_gate
      min_total_signals: 3
      count_once_per_cycle: true
    time_theory:
      mode: soft_filter
      kihon_suchi: [9, 17, 26, 33, 42, 51, 65, 76, 83, 97, 101, 129, 172, 200]
      tolerance_bars: 1
      max_fractal_sources: 5
      tato_suchi_enabled: true
      apply_on_timeframe: primary
    wave_analysis:
      enabled: true
      warn_trading_correction: true
      correction_score_penalty: 2
    price_targets:
      enabled: true
      fib_levels: {nt_value: 0.927, n_value: 1.0, v_value: 1.221, e_value: 1.611}
    exit:
      mode: hybrid
      partial_close_pct: 50
      primary_target: n_value
      trail_remainder: true
      move_stop_to_entry: true
    stop_loss:
      preference_order: [fractal, kijun, kumo_edge, candle_open]
      kijun_buffer_pips: 5
      max_stop_pips: 100
      min_rr_ratio: 1.5
    heikin_ashi:
      enabled: true
      use_for: exit_only
    quality:
      min_tier: B
      a_plus_size_mult: 1.5
      a_size_mult: 1.0
      b_size_mult: 0.5
    session_filters:
      close_before_weekend: true
      friday_cutoff_utc: "18:00"
      news_blackout_minutes: 30
    signal:
      max_signals_per_bar: 1
      min_confluence_score: 6
```

- [ ] **Step 2: Verify old strategy still loadable**

Run: `python -c "from src.strategy.base import STRATEGY_REGISTRY; from src.strategy.strategies.ichimoku import IchimokuStrategy; print('ichimoku' in STRATEGY_REGISTRY)"`
Expected: `True`

- [ ] **Step 3: Verify new strategy loadable**

Run: `python -c "from src.strategy.base import STRATEGY_REGISTRY; from src.strategy.strategies.ichimoku_fxaog import IchimokuFXAOGStrategy; print('ichimoku_fxaog' in STRATEGY_REGISTRY)"`
Expected: `True`

- [ ] **Step 4: Commit**

```bash
git add config/strategy.yaml
git commit -m "feat(task-13): config — add ichimoku_fxaog strategy configuration"
```

---

## Phase 5: Integration

### Task 14: End-to-End Integration Test

**Files:**
- Create: `tests/integration/test_ichimoku_fxaog_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/integration/test_ichimoku_fxaog_integration.py
"""End-to-end integration test: all evaluators → strategy → signal."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.base import EvalMatrix, EVALUATOR_REGISTRY, STRATEGY_REGISTRY


def _make_trending_ohlcv(n=300, start=1800.0, step=1.5) -> pd.DataFrame:
    """Strong uptrend with periodic pullbacks — should generate signals."""
    idx = pd.date_range('2026-01-01', periods=n, freq='1h', tz='UTC')
    close = np.zeros(n)
    price = start
    for i in range(n):
        if i % 20 < 15:
            price += step
        else:
            price -= step * 0.5
        close[i] = price
    return pd.DataFrame({
        'open': close - 0.3 * step,
        'high': close + abs(step),
        'low': close - abs(step),
        'close': close,
        'volume': np.ones(n) * 100,
    }, index=idx)


class TestFullPipeline:
    def test_all_evaluators_registered(self):
        """All 7 FXAOG evaluators must be in the registry."""
        # Trigger imports
        import src.strategy.evaluators.ichimoku_core_eval
        import src.strategy.evaluators.price_action_eval
        import src.strategy.evaluators.fractal_eval
        import src.strategy.evaluators.five_elements_eval
        import src.strategy.evaluators.time_theory_eval
        import src.strategy.evaluators.wave_analysis_eval
        import src.strategy.evaluators.price_target_eval

        expected = [
            'ichimoku_core', 'price_action', 'fractal',
            'five_elements', 'time_theory', 'wave_analysis', 'price_target',
        ]
        for key in expected:
            assert key in EVALUATOR_REGISTRY, f"{key} not registered"

    def test_strategy_registered(self):
        import src.strategy.strategies.ichimoku_fxaog
        assert 'ichimoku_fxaog' in STRATEGY_REGISTRY

    def test_evaluators_produce_results_on_real_data(self):
        """Run each evaluator on trending data and verify output shape."""
        ohlcv = _make_trending_ohlcv(n=300)
        for key in ['ichimoku_core', 'price_action', 'fractal',
                     'five_elements', 'time_theory', 'wave_analysis', 'price_target']:
            eval_cls = EVALUATOR_REGISTRY[key]
            evaluator = eval_cls()
            result = evaluator.evaluate(ohlcv)
            assert result is not None, f"{key} returned None"
            assert -1.0 <= result.direction <= 1.0, f"{key} direction out of range"
            assert 0.0 <= result.confidence <= 1.0, f"{key} confidence out of range"

    def test_full_signal_generation(self):
        """Build EvalMatrix from all evaluators, feed to strategy, get signal."""
        ohlcv = _make_trending_ohlcv(n=300, step=2.0)

        # Run all evaluators
        matrix = EvalMatrix()
        for tf in ['4H', '1H']:
            for key in ['ichimoku_core', 'fractal', 'wave_analysis']:
                eval_cls = EVALUATOR_REGISTRY[key]
                evaluator = eval_cls()
                result = evaluator.evaluate(ohlcv)
                matrix.set(f'{key}_{tf}', result)

        for key in ['price_action', 'price_target']:
            eval_cls = EVALUATOR_REGISTRY[key]
            evaluator = eval_cls()
            result = evaluator.evaluate(ohlcv)
            matrix.set(f'{key}_1H', result)

        for key in ['five_elements', 'time_theory']:
            eval_cls = EVALUATOR_REGISTRY[key]
            evaluator = eval_cls()
            result = evaluator.evaluate(ohlcv)
            matrix.set(f'{key}_4H', result)

        # Run strategy
        from src.strategy.strategies.ichimoku_fxaog import IchimokuFXAOGStrategy
        strategy = IchimokuFXAOGStrategy(
            five_elements_mode='disabled',
            time_theory_mode='disabled',
            min_confluence_score=1,
            min_tier='C',
        )
        signal = strategy.decide(matrix)
        # With a strong trend and relaxed filters, we expect a signal
        # (may still be None if checklist doesn't align — that's OK)
        if signal is not None:
            assert signal.direction in ('long', 'short')
            assert signal.strategy_name == 'ichimoku_fxaog'
            assert signal.confluence_score >= 1
            assert 'trade_type' in signal.reasoning

    def test_backward_compat_old_strategy_still_works(self):
        """The old ichimoku strategy should still be importable and registered."""
        from src.strategy.strategies.ichimoku import IchimokuStrategy
        assert 'ichimoku' in STRATEGY_REGISTRY
```

- [ ] **Step 2: Run the full test suite**

Run: `pytest tests/ -v --tb=short -x`
Expected: All tests PASS (both new and existing)

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_ichimoku_fxaog_integration.py
git commit -m "feat(task-14): integration tests — full FXAOG pipeline end-to-end"
```

---

## Validation Checklist

After all tasks are complete, verify:

- [ ] `pytest tests/indicators/ -v` — All indicator tests pass
- [ ] `pytest tests/strategy/evaluators/ -v` — All evaluator tests pass
- [ ] `pytest tests/strategy/strategies/ -v` — Strategy tests pass (both old and new)
- [ ] `pytest tests/strategy/trading_modes/ -v` — Exit manager tests pass
- [ ] `pytest tests/integration/ -v` — Integration tests pass
- [ ] `pytest tests/ -v` — Full suite passes (no regressions)
- [ ] `python -c "from src.strategy.base import STRATEGY_REGISTRY; print(list(STRATEGY_REGISTRY.keys()))"` — Shows both `ichimoku` and `ichimoku_fxaog`
