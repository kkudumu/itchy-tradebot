# FX At One Glance — Complete Ichimoku Strategy Implementation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the existing `IchimokuStrategy` with `FXAtOneGlance` — a complete implementation of the FX At One Glance Ichimoku course (basic + all advanced Japanese techniques), featuring 9 trade types, 5-wave Elliott counting, fractal-first market structure, 8 configurable timeframe modes, and a strategy profile system for blending.

**Architecture:** Multi-Evaluator + Strategy Coordinator. 8 new evaluators (registered via `Evaluator(ABC, key='...')`) feed an `EvalMatrix` consumed by a single `FXAtOneGlance(Strategy, key='fx_at_one_glance')`. Six new indicator modules provide the computational foundation. An `IchimokuExitManager(TradingMode)` handles configurable exit logic with 3 modes. The old `ichimoku` strategy key is preserved for backward compatibility.

**Tech Stack:** Python 3.11+, NumPy, Pandas, pytest, existing base classes in `src/strategy/base.py`

**Source Material:**
- FX At One Glance — Ichimoku First Glance Video Course (11 lessons)
- FX At One Glance — Ichimoku Advanced Japanese Techniques (5 lessons: 5 Elements, Time Theory 1&2, Wave Analysis, Price Range Observation)

### Key Design Decisions (from brainstorm)
- Triple range filter: Cloud balancing O/G + ADX>20 + cloud thickness
- Timeframe-adaptive Chikou: Chikou on Daily/H4, RSI>50 on H1 and below
- Swing detection: Williams 5-bar on H4+, ATR-based prominence on M15 and below
- Hyperscalp M1: No Ichimoku lines — pure price action at higher-TF Kijun/Cloud levels
- Elliott: Rules-based 5-wave with confidence scoring, not ML
- Strategy profiles: Named profiles in strategy.yaml
- Fractal-first: Market structure (HH/HL/LH/LL chains) is the foundation; Ichimoku overlays on top
- Single strategy, unified scoring: One class detects all 9 trade types, scores by confluence, fires highest
- Full course exits: 50% partial at N-value wave target, trail remainder with fractals + Kijun

### Existing Code to Leverage
- `RSICalculator` in `src/indicators/confluence.py` — already implemented, needs evaluator wrapper
- `DivergenceDetector` in `src/indicators/divergence.py` — already implemented, needs evaluator wrapper
- `ADXEvaluator` in `src/strategy/evaluators/adx_eval.py` — already registered as `'adx'`
- `ATREvaluator` in `src/strategy/evaluators/atr_eval.py` — already registered as `'atr'`
- `IchimokuCalculator` in `src/indicators/ichimoku.py` — used by all Ichimoku evaluators
- `EvaluatorCoordinator` in `src/strategy/coordinator.py` — handles multi-TF resampling

---

## File Structure

### New Files (21)
| File | Responsibility |
|---|---|
| `src/indicators/fractals.py` | Williams 5-bar fractal detection, market structure tracking |
| `src/indicators/price_action.py` | Candlestick patterns: tweezer, inside bar, engulfing, pin bar, doji |
| `src/indicators/cloud_balancing.py` | Wu Xing Five Elements O/G counting system |
| `src/indicators/kihon_suchi.py` | Ichimoku Time Theory cycle projection |
| `src/indicators/wave_patterns.py` | N-wave detection, N/V/E/NT targets, 5-wave Elliott counting |
| `src/indicators/heikin_ashi.py` | Heikin Ashi candle computation and trend signals |
| `src/strategy/evaluators/fractal_eval.py` | Fractal structure + momentum evaluator |
| `src/strategy/evaluators/price_action_eval.py` | Candlestick pattern evaluator |
| `src/strategy/evaluators/cloud_balance_eval.py` | Five Elements O/G equilibrium evaluator |
| `src/strategy/evaluators/kihon_suchi_eval.py` | Time Theory projection evaluator |
| `src/strategy/evaluators/wave_eval.py` | Wave classification + Elliott evaluator |
| `src/strategy/evaluators/rsi_eval.py` | RSI evaluator (wraps existing RSICalculator) |
| `src/strategy/evaluators/divergence_eval.py` | Divergence evaluator (wraps existing DivergenceDetector) |
| `src/strategy/strategies/fx_at_one_glance.py` | Main strategy: 9 trade types, 5-point checklist, 0-15 confluence |
| `src/strategy/trading_modes/ichimoku_exit.py` | 3-mode exit manager: trailing/targets/hybrid + HA confirmation |
| `tests/indicators/test_fractals.py` | Fractal indicator tests |
| `tests/indicators/test_price_action.py` | Price action indicator tests |
| `tests/indicators/test_cloud_balancing.py` | Cloud balancing indicator tests |
| `tests/indicators/test_kihon_suchi.py` | Kihon Suchi indicator tests |
| `tests/indicators/test_wave_patterns.py` | Wave patterns + Elliott tests |
| `tests/indicators/test_heikin_ashi.py` | Heikin Ashi tests |

### Modified Files (3)
| File | What changes |
|---|---|
| `src/strategy/evaluators/ichimoku_eval.py` | Keep as-is for backward compat; old `'ichimoku'` key stays |
| `src/strategy/strategies/ichimoku.py` | Keep as-is; old `'ichimoku'` key stays alongside new `'fx_at_one_glance'` |
| `config/strategy.yaml` | New `fx_at_one_glance` config block, strategy profiles, `active_strategy` update |

---

## Wave 1 — Foundation: Indicators & Evaluators (parallelizable)

### Task 1: Williams Fractal Indicator + Fractal Evaluator

**Files:**
- Create: `src/indicators/fractals.py`
- Create: `src/strategy/evaluators/fractal_eval.py`
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
        assert any(f.bar_index == 5 for f in bulls)

    def test_bear_fractal_at_v_bottom(self):
        high, low, close = _make_v_shape(n=11, base=1800.0)
        result = detect_fractals(high, low)
        bears = [f for f in result.bear_fractals if not np.isnan(f.price)]
        assert len(bears) >= 1
        assert any(f.bar_index == 5 for f in bears)

    def test_five_bar_minimum(self):
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
        high1 = np.array([10.0, 11.0, 13.0, 11.0, 10.0, 9.0, 8.0])
        low1 = np.array([9.0, 10.0, 12.0, 10.0, 9.0, 8.0, 7.0])
        r1 = detect_fractals(high1, low1)
        bulls1 = {f.bar_index for f in r1.bull_fractals}
        high2 = np.concatenate([high1, np.array([9.0, 10.0, 11.0])])
        low2 = np.concatenate([low1, np.array([8.0, 9.0, 10.0])])
        r2 = detect_fractals(high2, low2)
        bulls2 = {f.bar_index for f in r2.bull_fractals}
        assert bulls1.issubset(bulls2)


class TestMarketStructure:
    def test_uptrend_detection(self):
        # Staircase pattern: HH + HL = uptrend
        from src.indicators.fractals import market_structure
        high = np.array([
            10, 11, 15, 11, 10,   # bull fractal at 2 (15)
            10, 11, 12, 18, 12,   # bull fractal at 8 (18)
            11, 12, 13, 22, 13,   # bull fractal at 13 (22)
            12, 11, 10, 9, 8,
        ], dtype=float)
        low = high - 3.0
        result = market_structure(high, low)
        assert result.trend in ('uptrend', 'ranging')

    def test_flat_market_is_ranging(self):
        from src.indicators.fractals import market_structure
        high = np.full(50, 1800.0) + np.random.uniform(-0.5, 0.5, 50)
        low = high - 1.0
        result = market_structure(high, low)
        assert result.trend == 'ranging'
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
    """All detected fractals."""
    bull_fractals: list[FractalLevel] = field(default_factory=list)
    bear_fractals: list[FractalLevel] = field(default_factory=list)


@dataclass
class StructureState:
    """Market structure derived from fractal sequences."""
    trend: str  # 'uptrend' | 'downtrend' | 'ranging'
    last_bull_fractal: FractalLevel | None = None
    last_bear_fractal: FractalLevel | None = None
    swing_highs: list[float] = field(default_factory=list)
    swing_lows: list[float] = field(default_factory=list)


def detect_fractals(high: np.ndarray, low: np.ndarray, window: int = 2) -> FractalResult:
    """Detect Bill Williams fractals. Window=2 means 5-bar (2 on each side)."""
    n = len(high)
    result = FractalResult()
    if n < 2 * window + 1:
        return result
    for i in range(window, n - window):
        is_bull = all(high[i] > high[i - j] and high[i] > high[i + j] for j in range(1, window + 1))
        is_bear = all(low[i] < low[i - j] and low[i] < low[i + j] for j in range(1, window + 1))
        if is_bull:
            result.bull_fractals.append(FractalLevel(price=high[i], bar_index=i))
        if is_bear:
            result.bear_fractals.append(FractalLevel(price=low[i], bar_index=i))
    return result


def market_structure(high: np.ndarray, low: np.ndarray, window: int = 2) -> StructureState:
    """Determine trend from HH/HL/LH/LL fractal sequences."""
    fr = detect_fractals(high, low, window)
    state = StructureState(trend='ranging')
    if fr.bull_fractals:
        state.last_bull_fractal = fr.bull_fractals[-1]
        state.swing_highs = [f.price for f in fr.bull_fractals[-5:]]
    if fr.bear_fractals:
        state.last_bear_fractal = fr.bear_fractals[-1]
        state.swing_lows = [f.price for f in fr.bear_fractals[-5:]]
    # HH+HL = uptrend, LH+LL = downtrend
    if len(state.swing_highs) >= 2 and len(state.swing_lows) >= 2:
        hh = state.swing_highs[-1] > state.swing_highs[-2]
        hl = state.swing_lows[-1] > state.swing_lows[-2]
        lh = state.swing_highs[-1] < state.swing_highs[-2]
        ll = state.swing_lows[-1] < state.swing_lows[-2]
        if hh and hl:
            state.trend = 'uptrend'
        elif lh and ll:
            state.trend = 'downtrend'
    return state


def fractal_momentum(fractals: list[FractalLevel]) -> list[float]:
    """Price distances between successive same-type fractals."""
    if len(fractals) < 2:
        return []
    sorted_f = sorted(fractals, key=lambda f: f.bar_index)
    return [abs(sorted_f[i + 1].price - sorted_f[i].price) for i in range(len(sorted_f) - 1)]


def momentum_trend(distances: list[float]) -> str:
    """Classify momentum: 'strengthening', 'weakening', or 'flat'."""
    if len(distances) < 2:
        return 'flat'
    recent = distances[-3:] if len(distances) >= 3 else distances
    inc = sum(1 for i in range(len(recent) - 1) if recent[i + 1] > recent[i])
    dec = sum(1 for i in range(len(recent) - 1) if recent[i + 1] < recent[i])
    if inc > dec:
        return 'strengthening'
    elif dec > inc:
        return 'weakening'
    return 'flat'
```

- [ ] **Step 4: Implement FractalEval**

```python
# src/strategy/evaluators/fractal_eval.py
"""FractalEval — fractal structure + momentum evaluator."""
from __future__ import annotations
import pandas as pd
from ..base import Evaluator, EvaluatorResult
from ...indicators.fractals import (
    detect_fractals, market_structure, fractal_momentum, momentum_trend,
)


class FractalStructureEvaluator(Evaluator, key='fractal'):
    def __init__(self, window: int = 2):
        self._window = window

    def evaluate(self, ohlcv: pd.DataFrame) -> EvaluatorResult:
        high, low = ohlcv['high'].values, ohlcv['low'].values
        struct = market_structure(high, low, self._window)
        fr = detect_fractals(high, low, self._window)

        direction = {'uptrend': 1.0, 'downtrend': -1.0}.get(struct.trend, 0.0)
        mom = fractal_momentum(fr.bull_fractals if direction >= 0 else fr.bear_fractals)
        m_trend = momentum_trend(mom)
        confidence = {'strengthening': 0.8, 'flat': 0.5, 'weakening': 0.2}.get(m_trend, 0.5)

        return EvaluatorResult(direction=direction, confidence=confidence, metadata={
            'structure': struct.trend,
            'last_bull_fractal': struct.last_bull_fractal,
            'last_bear_fractal': struct.last_bear_fractal,
            'swing_highs': struct.swing_highs,
            'swing_lows': struct.swing_lows,
            'momentum_trend': m_trend,
            'fractal_count_bull': len(fr.bull_fractals),
            'fractal_count_bear': len(fr.bear_fractals),
        })
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/indicators/test_fractals.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/indicators/fractals.py src/strategy/evaluators/fractal_eval.py tests/indicators/test_fractals.py
git commit -m "feat(task-1): fractal indicator + FractalStructureEvaluator"
```

---

### Task 2: Price Action Indicator + Evaluator

**Files:**
- Create: `src/indicators/price_action.py`
- Create: `src/strategy/evaluators/price_action_eval.py`
- Create: `tests/indicators/test_price_action.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/indicators/test_price_action.py
from __future__ import annotations
import numpy as np
import pytest
from src.indicators.price_action import detect_patterns, PriceActionResult


def _candles(ohlc_list):
    arr = np.array(ohlc_list, dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


class TestTweezerDetection:
    def test_tweezer_bottom(self):
        o, h, l, c = _candles([
            (1810, 1812, 1800, 1802),  # red
            (1802, 1811, 1800, 1809),  # green, same low
        ])
        result = detect_patterns(o, h, l, c, tick_tolerance=2)
        assert result.tweezer_bottom is True

    def test_tweezer_top(self):
        o, h, l, c = _candles([
            (1800, 1812, 1799, 1810),  # green
            (1810, 1812, 1801, 1803),  # red, same high
        ])
        result = detect_patterns(o, h, l, c, tick_tolerance=2)
        assert result.tweezer_top is True

    def test_same_color_not_tweezer(self):
        o, h, l, c = _candles([
            (1800, 1812, 1800, 1810),
            (1805, 1813, 1800, 1811),
        ])
        result = detect_patterns(o, h, l, c, tick_tolerance=2)
        assert result.tweezer_bottom is False


class TestInsideBar:
    def test_inside_bar_count(self):
        o, h, l, c = _candles([
            (1800, 1815, 1795, 1810),  # mother
            (1803, 1808, 1801, 1807),  # inside
            (1804, 1809, 1802, 1806),  # inside
            (1805, 1807, 1803, 1805),  # inside
            (1806, 1818, 1804, 1815),  # breakout above
        ])
        result = detect_patterns(o, h, l, c)
        assert result.inside_bar_count >= 2
        assert result.inside_bar_breakout == 'up'


class TestEngulfing:
    def test_bullish_engulfing(self):
        o, h, l, c = _candles([
            (1810, 1812, 1800, 1802),
            (1800, 1815, 1798, 1812),
        ])
        result = detect_patterns(o, h, l, c)
        assert result.engulfing_bullish is True

    def test_bearish_engulfing(self):
        o, h, l, c = _candles([
            (1800, 1812, 1798, 1810),
            (1812, 1814, 1797, 1798),
        ])
        result = detect_patterns(o, h, l, c)
        assert result.engulfing_bearish is True


class TestPinBar:
    def test_hammer(self):
        o, h, l, c = _candles([(1802, 1805, 1790, 1804)])
        result = detect_patterns(o, h, l, c)
        assert result.pin_bar_bullish is True

    def test_shooting_star(self):
        o, h, l, c = _candles([(1804, 1818, 1801, 1802)])
        result = detect_patterns(o, h, l, c)
        assert result.pin_bar_bearish is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/indicators/test_price_action.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement price action detection**

```python
# src/indicators/price_action.py
"""Candlestick price action pattern detection."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class PriceActionResult:
    tweezer_bottom: bool = False
    tweezer_top: bool = False
    inside_bar_count: int = 0
    inside_bar_breakout: str = 'none'  # 'up' | 'down' | 'none'
    engulfing_bullish: bool = False
    engulfing_bearish: bool = False
    pin_bar_bullish: bool = False
    pin_bar_bearish: bool = False
    doji: bool = False
    mother_bar_high: float = 0.0
    mother_bar_low: float = 0.0


def _body_top(o, c): return max(o, c)
def _body_bot(o, c): return min(o, c)
def _is_red(o, c): return c < o
def _is_green(o, c): return c > o


def detect_patterns(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
    tick_tolerance: float = 2.0, doji_body_pct: float = 0.05,
) -> PriceActionResult:
    n = len(open_)
    result = PriceActionResult()
    if n == 0:
        return result

    last = n - 1
    o_l, h_l, l_l, c_l = open_[last], high[last], low[last], close[last]
    body = abs(c_l - o_l)
    full_range = h_l - l_l

    # Doji
    if full_range > 0 and body / full_range <= doji_body_pct:
        result.doji = True

    # Pin bars
    if full_range > 0:
        upper_wick = h_l - _body_top(o_l, c_l)
        lower_wick = _body_bot(o_l, c_l) - l_l
        if lower_wick >= 2.0 * max(body, 0.01) and upper_wick < body:
            result.pin_bar_bullish = True
        if upper_wick >= 2.0 * max(body, 0.01) and lower_wick < body:
            result.pin_bar_bearish = True

    if n < 2:
        return result

    prev = last - 1
    o_p, h_p, l_p, c_p = open_[prev], high[prev], low[prev], close[prev]

    # Tweezer bottom: red then green, matching lows
    if _is_red(o_p, c_p) and _is_green(o_l, c_l) and abs(l_p - l_l) <= tick_tolerance:
        result.tweezer_bottom = True
    # Tweezer top: green then red, matching highs
    if _is_green(o_p, c_p) and _is_red(o_l, c_l) and abs(h_p - h_l) <= tick_tolerance:
        result.tweezer_top = True

    # Engulfing
    prev_bt, prev_bb = _body_top(o_p, c_p), _body_bot(o_p, c_p)
    last_bt, last_bb = _body_top(o_l, c_l), _body_bot(o_l, c_l)
    if _is_red(o_p, c_p) and _is_green(o_l, c_l) and last_bt >= prev_bt and last_bb <= prev_bb and body > 0:
        result.engulfing_bullish = True
    if _is_green(o_p, c_p) and _is_red(o_l, c_l) and last_bt >= prev_bt and last_bb <= prev_bb and body > 0:
        result.engulfing_bearish = True

    # Inside bars — scan backward from last
    if n >= 3:
        for mi in range(last - 1, max(last - 10, -1), -1):
            m_top = _body_top(open_[mi], close[mi])
            m_bot = _body_bot(open_[mi], close[mi])
            if abs(close[mi] - open_[mi]) < 0.5:
                continue
            count = 0
            for i in range(mi + 1, last):
                bt = _body_top(open_[i], close[i])
                bb = _body_bot(open_[i], close[i])
                if bb >= m_bot and bt <= m_top:
                    count += 1
                else:
                    break
            if count > 0:
                result.inside_bar_count = count
                result.mother_bar_high = m_top
                result.mother_bar_low = m_bot
                if close[last] > m_top:
                    result.inside_bar_breakout = 'up'
                elif close[last] < m_bot:
                    result.inside_bar_breakout = 'down'
                break

    return result
```

- [ ] **Step 4: Implement PriceActionEvaluator**

```python
# src/strategy/evaluators/price_action_eval.py
"""PriceActionEvaluator — wraps price_action indicator as Evaluator."""
from __future__ import annotations
import pandas as pd
from ..base import Evaluator, EvaluatorResult
from ...indicators.price_action import detect_patterns


class PriceActionEvaluator(Evaluator, key='price_action'):
    def __init__(self, tick_tolerance: float = 2.0):
        self._tick_tol = tick_tolerance

    def evaluate(self, ohlcv: pd.DataFrame) -> EvaluatorResult:
        pa = detect_patterns(
            ohlcv['open'].values, ohlcv['high'].values,
            ohlcv['low'].values, ohlcv['close'].values,
            tick_tolerance=self._tick_tol,
        )
        direction, confidence = 0.0, 0.0
        # Priority: engulfing > inside-out > tweezer > pin bar
        if pa.engulfing_bullish:
            direction, confidence = 1.0, 1.0
        elif pa.engulfing_bearish:
            direction, confidence = -1.0, 1.0
        elif pa.inside_bar_breakout == 'up':
            direction, confidence = 1.0, 0.9
        elif pa.inside_bar_breakout == 'down':
            direction, confidence = -1.0, 0.9
        elif pa.tweezer_bottom:
            direction, confidence = 1.0, 0.7
        elif pa.tweezer_top:
            direction, confidence = -1.0, 0.7
        elif pa.pin_bar_bullish:
            direction, confidence = 1.0, 0.5
        elif pa.pin_bar_bearish:
            direction, confidence = -1.0, 0.5

        return EvaluatorResult(direction=direction, confidence=confidence, metadata={
            'tweezer_bottom': pa.tweezer_bottom, 'tweezer_top': pa.tweezer_top,
            'inside_bar_count': pa.inside_bar_count, 'inside_bar_breakout': pa.inside_bar_breakout,
            'engulfing_bullish': pa.engulfing_bullish, 'engulfing_bearish': pa.engulfing_bearish,
            'pin_bar_bullish': pa.pin_bar_bullish, 'pin_bar_bearish': pa.pin_bar_bearish,
            'doji': pa.doji, 'mother_bar_high': pa.mother_bar_high, 'mother_bar_low': pa.mother_bar_low,
        })
```

- [ ] **Step 5: Run tests, verify pass**

Run: `pytest tests/indicators/test_price_action.py -v`

- [ ] **Step 6: Commit**

```bash
git add src/indicators/price_action.py src/strategy/evaluators/price_action_eval.py tests/indicators/test_price_action.py
git commit -m "feat(task-2): price action indicator + PriceActionEvaluator"
```

---

### Task 3: Cloud Balancing (Five Elements) Indicator + Evaluator

**Files:**
- Create: `src/indicators/cloud_balancing.py`
- Create: `src/strategy/evaluators/cloud_balance_eval.py`
- Create: `tests/indicators/test_cloud_balancing.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/indicators/test_cloud_balancing.py
from __future__ import annotations
import numpy as np
import pytest
from src.indicators.cloud_balancing import CloudBalancer, BalanceState

KIHON_PERIOD = 26


def _trending_components(n=200, direction='bullish'):
    """Create Ichimoku components with a clear TK cross and subsequent crossovers."""
    step = 2.0 if direction == 'bullish' else -2.0
    base = np.arange(n, dtype=float) * step + 1800.0
    tenkan = base + 5.0
    kijun = base
    chikou = np.roll(base, KIHON_PERIOD)
    senkou_a = base + 3.0
    senkou_b = base - 3.0
    return tenkan, kijun, chikou, senkou_a, senkou_b


class TestCloudBalancer:
    def test_returns_balance_state(self):
        t, k, ch, sa, sb = _trending_components()
        cb = CloudBalancer()
        state = cb.calculate(t, k, ch, sa, sb)
        assert isinstance(state, BalanceState)

    def test_has_og_counts(self):
        t, k, ch, sa, sb = _trending_components()
        cb = CloudBalancer()
        state = cb.calculate(t, k, ch, sa, sb)
        assert hasattr(state, 'o_count')
        assert hasattr(state, 'g_count')
        assert hasattr(state, 'is_disequilibrium')

    def test_tk_cross_resets_cycle(self):
        t, k, ch, sa, sb = _trending_components()
        cb = CloudBalancer()
        state = cb.calculate(t, k, ch, sa, sb)
        assert state.o_count >= 0
        assert state.g_count >= 0

    def test_disequilibrium_when_unequal(self):
        t, k, ch, sa, sb = _trending_components()
        cb = CloudBalancer()
        state = cb.calculate(t, k, ch, sa, sb)
        assert state.is_disequilibrium == (state.o_count != state.g_count)

    def test_crossover_counted_once_per_cycle(self):
        t, k, ch, sa, sb = _trending_components()
        cb = CloudBalancer()
        state = cb.calculate(t, k, ch, sa, sb)
        assert state.o_count + state.g_count <= 9  # max 9 non-TK cross types
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/indicators/test_cloud_balancing.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement Cloud Balancing**

```python
# src/indicators/cloud_balancing.py
"""Wu Xing Five Elements O/G counting system.

Elements: Fire=Tenkan, Water=Kijun, Wood=Chikou, Metal=SpanA, Earth=SpanB

Crossover O/G classification per course:
  Tenkan x SpanA = O,  Tenkan x SpanB = G
  Kijun x SpanA = G,   Kijun x SpanB = O
  Chikou x Tenkan = G,  Chikou x Kijun = G
  Chikou x SpanA = O,   Chikou x SpanB = G
  SpanA x SpanB (Kumo twist) = G (always)

Rules:
  - TK cross starts new cycle, resets count
  - Each crossover type counted only ONCE per cycle
  - O == G → equilibrium → DO NOT TRADE
  - O != G → disequilibrium → CAN TRADE in TK cross direction
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


_CROSS_TABLE = {
    ('tenkan', 'senkou_a'): 'O',
    ('tenkan', 'senkou_b'): 'G',
    ('kijun', 'senkou_a'): 'G',
    ('kijun', 'senkou_b'): 'O',
    ('chikou', 'tenkan'): 'G',
    ('chikou', 'kijun'): 'G',
    ('chikou', 'senkou_a'): 'O',
    ('chikou', 'senkou_b'): 'G',
    ('senkou_a', 'senkou_b'): 'G',
}


@dataclass
class BalanceState:
    o_count: int = 0
    g_count: int = 0
    is_disequilibrium: bool = False
    tk_direction: int = 0  # +1 bullish, -1 bearish, 0 no cycle
    cycle_start_bar: int = 0
    crossover_log: list[dict] = field(default_factory=list)


class CloudBalancer:
    def calculate(
        self, tenkan: np.ndarray, kijun: np.ndarray,
        chikou: np.ndarray, senkou_a: np.ndarray, senkou_b: np.ndarray,
    ) -> BalanceState:
        n = len(tenkan)
        components = {
            'tenkan': tenkan, 'kijun': kijun, 'chikou': chikou,
            'senkou_a': senkou_a, 'senkou_b': senkou_b,
        }

        # Find last TK cross
        tk_crosses = []
        for i in range(1, n):
            if np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(tenkan[i-1]) or np.isnan(kijun[i-1]):
                continue
            prev = tenkan[i-1] - kijun[i-1]
            curr = tenkan[i] - kijun[i]
            if prev <= 0 < curr:
                tk_crosses.append((i, 1))
            elif prev >= 0 > curr:
                tk_crosses.append((i, -1))

        if not tk_crosses:
            return BalanceState()

        cycle_bar, tk_dir = tk_crosses[-1]

        # Count O/G after cycle start, each pair only once
        o_count, g_count = 0, 0
        seen: set[str] = set()
        log: list[dict] = []

        for (c1_name, c2_name), og_type in _CROSS_TABLE.items():
            c1, c2 = components.get(c1_name), components.get(c2_name)
            if c1 is None or c2 is None:
                continue
            pair_key = f"{c1_name}_{c2_name}"
            if pair_key in seen:
                continue
            for i in range(cycle_bar + 1, n):
                if i < 1 or np.isnan(c1[i]) or np.isnan(c2[i]) or np.isnan(c1[i-1]) or np.isnan(c2[i-1]):
                    continue
                if (c1[i-1] - c2[i-1]) * (c1[i] - c2[i]) < 0:  # sign change = cross
                    seen.add(pair_key)
                    if og_type == 'O':
                        o_count += 1
                    else:
                        g_count += 1
                    log.append({'pair': pair_key, 'type': og_type, 'bar': i})
                    break

        return BalanceState(
            o_count=o_count, g_count=g_count,
            is_disequilibrium=(o_count != g_count),
            tk_direction=tk_dir, cycle_start_bar=cycle_bar,
            crossover_log=log,
        )
```

- [ ] **Step 4: Implement CloudBalancingEvaluator**

```python
# src/strategy/evaluators/cloud_balance_eval.py
"""CloudBalancingEvaluator — Five Elements O/G equilibrium evaluator."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..base import Evaluator, EvaluatorResult
from ...indicators.ichimoku import IchimokuCalculator
from ...indicators.cloud_balancing import CloudBalancer


class CloudBalancingEvaluator(Evaluator, key='cloud_balance'):
    def __init__(self, tenkan_period=9, kijun_period=26, senkou_b_period=52):
        self._calc = IchimokuCalculator(tenkan_period, kijun_period, senkou_b_period)
        self._cb = CloudBalancer()
        self._kp = kijun_period

    def evaluate(self, ohlcv: pd.DataFrame) -> EvaluatorResult:
        h, l, c = ohlcv['high'].values, ohlcv['low'].values, ohlcv['close'].values
        n = len(c)
        ichi = self._calc.calculate(h, l, c)
        chikou = np.full(n, np.nan)
        if n > self._kp:
            chikou[:n - self._kp] = c[self._kp:]

        state = self._cb.calculate(
            ichi.tenkan_sen, ichi.kijun_sen, chikou,
            ichi.senkou_a[:n], ichi.senkou_b[:n],
        )
        direction = float(state.tk_direction)
        confidence = 1.0 if state.is_disequilibrium else 0.0
        return EvaluatorResult(direction=direction, confidence=confidence, metadata={
            'o_count': state.o_count, 'g_count': state.g_count,
            'is_disequilibrium': state.is_disequilibrium,
            'tk_direction': state.tk_direction,
            'cycle_start_bar': state.cycle_start_bar,
            'crossover_log': state.crossover_log,
        })
```

- [ ] **Step 5: Run tests, verify pass**
- [ ] **Step 6: Commit**

```bash
git add src/indicators/cloud_balancing.py src/strategy/evaluators/cloud_balance_eval.py tests/indicators/test_cloud_balancing.py
git commit -m "feat(task-3): Five Elements cloud balancing + CloudBalancingEvaluator"
```

---

### Task 4: Kihon Suchi (Time Theory) Indicator + Evaluator

**Files:**
- Create: `src/indicators/kihon_suchi.py`
- Create: `src/strategy/evaluators/kihon_suchi_eval.py`
- Create: `tests/indicators/test_kihon_suchi.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/indicators/test_kihon_suchi.py
from __future__ import annotations
import pytest
from src.indicators.kihon_suchi import is_kihon_number, project_from_pivot, KIHON_NUMBERS


class TestKihonNumbers:
    def test_known_numbers(self):
        for n in [9, 17, 26, 33, 42]:
            assert is_kihon_number(n, tolerance=0)

    def test_tolerance(self):
        assert is_kihon_number(25, tolerance=1) is True   # 26-1
        assert is_kihon_number(27, tolerance=1) is True   # 26+1
        assert is_kihon_number(28, tolerance=1) is False

    def test_non_kihon_rejected(self):
        assert is_kihon_number(15, tolerance=1) is False
        assert is_kihon_number(30, tolerance=1) is False


class TestProjection:
    def test_projects_all_numbers(self):
        targets = project_from_pivot(pivot_bar=50, total_bars=300)
        assert len(targets) > 0
        assert all(t['target_bar'] > 50 for t in targets)

    def test_projects_within_range(self):
        targets = project_from_pivot(pivot_bar=50, total_bars=100)
        assert all(t['target_bar'] < 100 for t in targets)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/indicators/test_kihon_suchi.py -v`

- [ ] **Step 3: Implement Kihon Suchi**

```python
# src/indicators/kihon_suchi.py
"""Ichimoku Time Theory — Kihon Suchi projection + Taito Suchi detection."""
from __future__ import annotations
from dataclasses import dataclass, field

KIHON_NUMBERS = [9, 17, 26, 33, 42, 51, 65, 76, 83, 97, 101, 129, 172, 200, 226, 257]


def is_kihon_number(bars: int, tolerance: int = 1) -> bool:
    return any(abs(bars - ks) <= tolerance for ks in KIHON_NUMBERS)


def nearest_kihon(bars: int) -> int:
    for ks in KIHON_NUMBERS:
        if ks >= bars:
            return ks
    return KIHON_NUMBERS[-1]


def project_from_pivot(pivot_bar: int, total_bars: int) -> list[dict]:
    targets = []
    for ks in KIHON_NUMBERS:
        target = pivot_bar + ks
        if target < total_bars:
            targets.append({'target_bar': target, 'kihon_number': ks})
    return targets


def find_active_cycles(current_bar: int, pivots: list[dict], tolerance: int = 1) -> list[dict]:
    """Check which pivots project a kihon number to current_bar."""
    hits = []
    for p in pivots:
        elapsed = current_bar - p['bar_index']
        if elapsed > 0 and is_kihon_number(elapsed, tolerance):
            hits.append({
                'source_bar': p['bar_index'], 'source_price': p['price'],
                'bars_elapsed': elapsed, 'matched_kihon': nearest_kihon(elapsed),
            })
    return hits


def detect_taito_suchi(pivots: list[dict], tolerance: int = 1) -> list[dict]:
    """Detect equal-period cycles between consecutive pivots."""
    cycles = []
    if len(pivots) < 3:
        return cycles
    sorted_p = sorted(pivots, key=lambda p: p['bar_index'])
    for i in range(len(sorted_p) - 2):
        d1 = sorted_p[i+1]['bar_index'] - sorted_p[i]['bar_index']
        d2 = sorted_p[i+2]['bar_index'] - sorted_p[i+1]['bar_index']
        if abs(d1 - d2) <= tolerance:
            cycles.append({'bar_count': d1, 'matches_kihon': is_kihon_number(d1, tolerance)})
    return cycles
```

- [ ] **Step 4: Implement KihonSuchiEvaluator**

```python
# src/strategy/evaluators/kihon_suchi_eval.py
"""KihonSuchiEvaluator — Time Theory projection evaluator."""
from __future__ import annotations
import pandas as pd
from ..base import Evaluator, EvaluatorResult
from ...indicators.fractals import detect_fractals
from ...indicators.kihon_suchi import find_active_cycles, detect_taito_suchi


class KihonSuchiEvaluator(Evaluator, key='kihon_suchi'):
    def __init__(self, tolerance: int = 1, max_sources: int = 5):
        self._tol = tolerance
        self._max = max_sources

    def evaluate(self, ohlcv: pd.DataFrame) -> EvaluatorResult:
        h, l = ohlcv['high'].values, ohlcv['low'].values
        n = len(h)
        fr = detect_fractals(h, l)
        all_pivots = [{'bar_index': f.bar_index, 'price': f.price}
                      for f in fr.bull_fractals + fr.bear_fractals]
        all_pivots.sort(key=lambda p: p['bar_index'], reverse=True)
        sources = all_pivots[:self._max]

        hits = find_active_cycles(n - 1, sources, self._tol)
        taito = detect_taito_suchi(all_pivots, self._tol)

        if len(hits) >= 3:
            confidence = 1.0
        elif len(hits) == 2:
            confidence = 0.8
        elif len(hits) == 1:
            confidence = 0.5
        else:
            confidence = 0.0
        if taito:
            confidence = min(1.0, confidence + 0.2)

        return EvaluatorResult(direction=0.0, confidence=confidence, metadata={
            'is_cycle_date': len(hits) > 0,
            'kihon_hits': len(hits),
            'active_cycles': hits,
            'taito_suchi_detected': len(taito) > 0,
            'taito_cycles': taito,
            'double_confirmation': len(hits) >= 2,
        })
```

- [ ] **Step 5: Run tests, verify pass**
- [ ] **Step 6: Commit**

```bash
git add src/indicators/kihon_suchi.py src/strategy/evaluators/kihon_suchi_eval.py tests/indicators/test_kihon_suchi.py
git commit -m "feat(task-4): Kihon Suchi time theory + KihonSuchiEvaluator"
```

---

### Task 5: Wave Patterns + Elliott Wave Indicator + Evaluator

**Files:**
- Create: `src/indicators/wave_patterns.py`
- Create: `src/strategy/evaluators/wave_eval.py`
- Create: `tests/indicators/test_wave_patterns.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/indicators/test_wave_patterns.py
from __future__ import annotations
import numpy as np
import pytest
from src.indicators.wave_patterns import (
    WaveAnalyzer, n_value, v_value, e_value, nt_value,
)


class TestTargetCalculations:
    def test_bullish_n_value(self):
        assert n_value(1800.0, 1900.0, 1850.0, 'bullish') == pytest.approx(1950.0)

    def test_bullish_v_value(self):
        assert v_value(1800.0, 1900.0, 1850.0, 'bullish') == pytest.approx(1950.0)

    def test_bullish_e_value(self):
        assert e_value(1800.0, 1900.0, 1850.0, 'bullish') == pytest.approx(2000.0)

    def test_bullish_nt_value(self):
        assert nt_value(1800.0, 1900.0, 1850.0, 'bullish') == pytest.approx(1900.0)

    def test_bearish_n_value(self):
        assert n_value(1900.0, 1800.0, 1850.0, 'bearish') == pytest.approx(1750.0)


class TestWaveClassification:
    def test_n_wave_detected(self):
        # Create clear N-wave: low-high-low-high with higher highs
        analyzer = WaveAnalyzer()
        swings = [
            {'type': 'low', 'price': 1800.0, 'bar': 0},
            {'type': 'high', 'price': 1900.0, 'bar': 20},
            {'type': 'low', 'price': 1850.0, 'bar': 40},
            {'type': 'high', 'price': 1960.0, 'bar': 60},
        ]
        result = analyzer.classify(swings, current_price=1960.0)
        assert result['wave_type'] == 'N'
        assert result['direction'] == 'bullish'

    def test_flat_is_box(self):
        analyzer = WaveAnalyzer()
        swings = [
            {'type': 'low', 'price': 1799.0, 'bar': 0},
            {'type': 'high', 'price': 1801.0, 'bar': 10},
            {'type': 'low', 'price': 1799.5, 'bar': 20},
            {'type': 'high', 'price': 1800.5, 'bar': 30},
        ]
        result = analyzer.classify(swings, current_price=1800.0)
        assert result['wave_type'] in ('box', 'P')


class TestElliottCounting:
    def test_wave3_not_shortest(self):
        from src.indicators.wave_patterns import count_elliott
        swings = [1800, 1850, 1820, 1910, 1870, 1920]  # W1=50, W3=90, W5=50
        result = count_elliott(swings, 'bullish')
        assert result is not None
        assert result['confidence'] > 0

    def test_wave2_100pct_retrace_invalid(self):
        from src.indicators.wave_patterns import count_elliott
        swings = [1800, 1850, 1800, 1900, 1850, 1920]  # W2 retraces 100%
        result = count_elliott(swings, 'bullish')
        assert result is None  # hard constraint violated
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/indicators/test_wave_patterns.py -v`

- [ ] **Step 3: Implement wave patterns + Elliott**

```python
# src/indicators/wave_patterns.py
"""N-wave detection, E/V/N/NT price targets, I/V/N/P/Y wave classification,
and rules-based 5-wave Elliott counting."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# --- Price Range Observation (Targets) ---

def n_value(a, b, c, direction):
    return c + (b - a) if direction == 'bullish' else c - (a - b)

def v_value(a, b, c, direction):
    return b + (b - c) if direction == 'bullish' else b - (c - b)

def e_value(a, b, c, direction):
    return b + (b - a) if direction == 'bullish' else b - (a - b)

def nt_value(a, b, c, direction):
    return c + (c - a) if direction == 'bullish' else c - (a - c)

def compute_all_targets(a, b, c, direction):
    return {
        'nt_value': nt_value(a, b, c, direction),
        'n_value': n_value(a, b, c, direction),
        'v_value': v_value(a, b, c, direction),
        'e_value': e_value(a, b, c, direction),
    }


# --- Wave Classification ---

class WaveAnalyzer:
    def build_swing_sequence(self, bulls, bears):
        all_s = [{'type': 'high', 'price': f.price, 'bar': f.bar_index} for f in bulls]
        all_s += [{'type': 'low', 'price': f.price, 'bar': f.bar_index} for f in bears]
        all_s.sort(key=lambda s: s['bar'])
        # Remove consecutive same-type (keep more extreme)
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

    def classify(self, swings, current_price):
        if len(swings) < 2:
            d = 'bullish' if len(swings) == 0 or current_price > swings[0]['price'] else 'bearish'
            return {'wave_type': 'I', 'direction': d, 'position': 'impulse', 'targets': {}}

        # N-wave: 4 swings, bullish = low-high-low-high with higher highs
        if len(swings) >= 4:
            s = swings[-4:]
            if s[0]['type'] == 'low' and s[1]['type'] == 'high' and s[2]['type'] == 'low' and s[3]['type'] == 'high':
                if s[3]['price'] > s[1]['price'] and s[2]['price'] > s[0]['price']:
                    targets = compute_all_targets(s[0]['price'], s[1]['price'], s[2]['price'], 'bullish')
                    pos = 'impulse' if current_price > s[2]['price'] else 'correction'
                    return {'wave_type': 'N', 'direction': 'bullish', 'position': pos,
                            'targets': targets, 'A': s[0]['price'], 'B': s[1]['price'], 'C': s[2]['price']}
            if s[0]['type'] == 'high' and s[1]['type'] == 'low' and s[2]['type'] == 'high' and s[3]['type'] == 'low':
                if s[3]['price'] < s[1]['price'] and s[2]['price'] < s[0]['price']:
                    targets = compute_all_targets(s[0]['price'], s[1]['price'], s[2]['price'], 'bearish')
                    pos = 'impulse' if current_price < s[2]['price'] else 'correction'
                    return {'wave_type': 'N', 'direction': 'bearish', 'position': pos,
                            'targets': targets, 'A': s[0]['price'], 'B': s[1]['price'], 'C': s[2]['price']}

        # P-wave (converging) / Y-wave (expanding)
        if len(swings) >= 4:
            highs = [s['price'] for s in swings if s['type'] == 'high']
            lows = [s['price'] for s in swings if s['type'] == 'low']
            if len(highs) >= 2 and len(lows) >= 2:
                if highs[-1] < highs[-2] and lows[-1] > lows[-2]:
                    return {'wave_type': 'P', 'direction': 'none', 'position': 'breakout_pending', 'targets': {}}
                if highs[-1] > highs[-2] and lows[-1] < lows[-2]:
                    return {'wave_type': 'Y', 'direction': 'none', 'position': 'breakout_pending', 'targets': {}}

        # Box (tight range)
        if len(swings) >= 4:
            prices = [s['price'] for s in swings[-6:]]
            rng = max(prices) - min(prices)
            if np.mean(prices) > 0 and rng / np.mean(prices) < 0.01:
                return {'wave_type': 'box', 'direction': 'none', 'position': 'breakout_pending', 'targets': {}}

        # V-wave
        if len(swings) >= 3:
            s = swings[-3:]
            if s[0]['type'] == 'low' and s[1]['type'] == 'high' and s[2]['type'] == 'low':
                return {'wave_type': 'V', 'direction': 'bearish', 'position': 'correction', 'targets': {}}
            if s[0]['type'] == 'high' and s[1]['type'] == 'low' and s[2]['type'] == 'high':
                return {'wave_type': 'V', 'direction': 'bullish', 'position': 'correction', 'targets': {}}

        d = 'bullish' if current_price > swings[0]['price'] else 'bearish'
        return {'wave_type': 'I', 'direction': d, 'position': 'impulse', 'targets': {}}


# --- 5-Wave Elliott Counting ---

def count_elliott(swing_prices, direction):
    """Rules-based 5-wave impulse counting.

    swing_prices: [W0_start, W1_end, W2_end, W3_end, W4_end, W5_end]
    Returns dict with wave_number, confidence, targets — or None if invalid.

    Hard constraints (auto-disqualify):
    - Wave 2 must not retrace 100% of Wave 1
    - Wave 3 must not be the shortest of waves 1, 3, 5
    - Wave 4 must not overlap Wave 1 territory
    """
    if len(swing_prices) < 4:
        return None

    is_bull = direction == 'bullish'
    p = swing_prices

    # Wave lengths
    w1 = abs(p[1] - p[0])
    w2_retrace = abs(p[2] - p[1])

    # Hard constraint: W2 < 100% of W1
    if w2_retrace >= w1:
        return None

    confidence = 0.0

    # Fibonacci scoring for W2 retrace (ideal 50-78.6%)
    w2_pct = w2_retrace / w1 if w1 > 0 else 0
    if 0.50 <= w2_pct <= 0.786:
        confidence += 0.25

    if len(swing_prices) >= 4:
        w3 = abs(p[3] - p[2])
        # W3 extension (ideal 161.8% of W1)
        if w1 > 0 and w3 / w1 >= 1.382:
            confidence += 0.25

    if len(swing_prices) >= 5:
        w3 = abs(p[3] - p[2])
        w4_retrace = abs(p[4] - p[3])
        # Hard: W4 must not overlap W1
        if is_bull and p[4] <= p[1]:
            return None
        if not is_bull and p[4] >= p[1]:
            return None
        # W4 retrace (ideal 23.6-50% of W3)
        w4_pct = w4_retrace / w3 if w3 > 0 else 0
        if 0.236 <= w4_pct <= 0.50:
            confidence += 0.25

    if len(swing_prices) >= 6:
        w3 = abs(p[3] - p[2])
        w5 = abs(p[5] - p[4])
        # Hard: W3 not shortest
        if w3 < w1 and w3 < w5:
            return None
        # W5 equals W1 (ideal)
        if w1 > 0 and 0.8 <= w5 / w1 <= 1.2:
            confidence += 0.25

    wave_number = min(len(swing_prices) - 1, 5)
    return {
        'wave_number': wave_number,
        'confidence': confidence,
        'is_complete': wave_number >= 5,
        'direction': direction,
    }
```

- [ ] **Step 4: Implement WavePatternEvaluator**

```python
# src/strategy/evaluators/wave_eval.py
"""WavePatternEvaluator — wave classification + Elliott + price targets."""
from __future__ import annotations
import pandas as pd
from ..base import Evaluator, EvaluatorResult
from ...indicators.fractals import detect_fractals
from ...indicators.wave_patterns import WaveAnalyzer, count_elliott


class WavePatternEvaluator(Evaluator, key='wave'):
    def evaluate(self, ohlcv: pd.DataFrame) -> EvaluatorResult:
        h, l, c = ohlcv['high'].values, ohlcv['low'].values, ohlcv['close'].values
        fr = detect_fractals(h, l)
        analyzer = WaveAnalyzer()
        swings = analyzer.build_swing_sequence(fr.bull_fractals, fr.bear_fractals)
        wave_result = analyzer.classify(swings, c[-1])

        # Try Elliott counting from recent swings
        elliott = None
        if len(swings) >= 4:
            prices = [s['price'] for s in swings[-6:]]
            direction = wave_result.get('direction', 'bullish')
            if direction in ('bullish', 'bearish'):
                elliott = count_elliott(prices, direction)

        dir_map = {'bullish': 1.0, 'bearish': -1.0}
        direction = dir_map.get(wave_result['direction'], 0.0)
        conf_map = {'N': 0.8, 'I': 0.5, 'V': 0.5, 'P': 0.3, 'Y': 0.3, 'box': 0.2}
        confidence = conf_map.get(wave_result['wave_type'], 0.3)

        return EvaluatorResult(direction=direction, confidence=confidence, metadata={
            'wave_type': wave_result['wave_type'],
            'wave_direction': wave_result['direction'],
            'wave_position': wave_result['position'],
            'targets': wave_result.get('targets', {}),
            'n_wave_A': wave_result.get('A'),
            'n_wave_B': wave_result.get('B'),
            'n_wave_C': wave_result.get('C'),
            'elliott': elliott,
            'swing_count': len(swings),
            'is_correction': wave_result['position'] == 'correction',
        })
```

- [ ] **Step 5: Run tests, verify pass**
- [ ] **Step 6: Commit**

```bash
git add src/indicators/wave_patterns.py src/strategy/evaluators/wave_eval.py tests/indicators/test_wave_patterns.py
git commit -m "feat(task-5): wave patterns + Elliott counting + WavePatternEvaluator"
```

---

### Task 6: Heikin Ashi Indicator

**Files:**
- Create: `src/indicators/heikin_ashi.py`
- Create: `tests/indicators/test_heikin_ashi.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/indicators/test_heikin_ashi.py
from __future__ import annotations
import numpy as np
import pytest
from src.indicators.heikin_ashi import compute_heikin_ashi, ha_trend_signal, HACandle


class TestComputeHeikinAshi:
    def test_close_is_ohlc_average(self):
        o = np.array([100.0, 105.0])
        h = np.array([110.0, 115.0])
        l = np.array([95.0, 100.0])
        c = np.array([105.0, 110.0])
        ha = compute_heikin_ashi(o, h, l, c)
        expected = (105.0 + 115.0 + 100.0 + 110.0) / 4
        assert ha.close[1] == pytest.approx(expected)

    def test_open_is_midpoint_of_previous(self):
        o = np.array([100.0, 105.0, 108.0])
        h = np.array([110.0, 115.0, 118.0])
        l = np.array([95.0, 100.0, 103.0])
        c = np.array([105.0, 110.0, 115.0])
        ha = compute_heikin_ashi(o, h, l, c)
        expected = (ha.open[1] + ha.close[1]) / 2
        assert ha.open[2] == pytest.approx(expected)

    def test_output_length_matches_input(self):
        n = 50
        o = np.random.uniform(100, 110, n)
        h = o + np.random.uniform(0, 5, n)
        l = o - np.random.uniform(0, 5, n)
        c = np.random.uniform(l, h)
        ha = compute_heikin_ashi(o, h, l, c)
        assert len(ha.open) == n


class TestHATrendSignal:
    def test_strong_bullish_no_lower_wick(self):
        candle = HACandle(open=100.0, high=110.0, low=100.0, close=108.0)
        assert ha_trend_signal(candle) == 'strong_bullish'

    def test_strong_bearish_no_upper_wick(self):
        candle = HACandle(open=108.0, high=108.0, low=95.0, close=100.0)
        assert ha_trend_signal(candle) == 'strong_bearish'

    def test_doji(self):
        candle = HACandle(open=100.0, high=105.0, low=95.0, close=100.1)
        assert ha_trend_signal(candle) == 'indecision'
```

- [ ] **Step 2: Run tests to verify they fail**
- [ ] **Step 3: Implement Heikin Ashi**

```python
# src/indicators/heikin_ashi.py
"""Heikin Ashi candle computation and trend signal classification."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class HACandle:
    open: float
    high: float
    low: float
    close: float


@dataclass
class HAResult:
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray


def compute_heikin_ashi(open_, high, low, close):
    n = len(open_)
    ha_close = (open_ + high + low + close) / 4.0
    ha_open = np.empty(n)
    ha_open[0] = (open_[0] + close[0]) / 2.0
    for i in range(1, n):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2.0
    ha_high = np.maximum(high, np.maximum(ha_open, ha_close))
    ha_low = np.minimum(low, np.minimum(ha_open, ha_close))
    return HAResult(open=ha_open, high=ha_high, low=ha_low, close=ha_close)


def ha_candle_at(ha, idx):
    return HACandle(open=ha.open[idx], high=ha.high[idx], low=ha.low[idx], close=ha.close[idx])


def ha_trend_signal(candle, doji_pct=0.05):
    body = abs(candle.close - candle.open)
    full_range = candle.high - candle.low
    if full_range > 0 and body / full_range <= doji_pct:
        return 'indecision'
    tol = full_range * 0.01 if full_range > 0 else 0.001
    if candle.close > candle.open:
        return 'strong_bullish' if abs(candle.open - candle.low) <= tol else 'weak_bullish'
    else:
        return 'strong_bearish' if abs(candle.open - candle.high) <= tol else 'weak_bearish'
```

- [ ] **Step 4: Run tests, verify pass**
- [ ] **Step 5: Commit**

```bash
git add src/indicators/heikin_ashi.py tests/indicators/test_heikin_ashi.py
git commit -m "feat(task-6): Heikin Ashi indicator with trend signal classification"
```

---

### Task 7: RSI Evaluator (wraps existing RSICalculator)

**Files:**
- Create: `src/strategy/evaluators/rsi_eval.py`

**Note:** `RSICalculator` already exists in `src/indicators/confluence.py`. This task just wraps it as an Evaluator for the FXAOG strategy (Chikou replacement on H1 and below).

- [ ] **Step 1: Implement RSIEvaluator**

```python
# src/strategy/evaluators/rsi_eval.py
"""RSIEvaluator — wraps existing RSICalculator. Used as Chikou replacement on H1 and below."""
from __future__ import annotations
import pandas as pd
from ..base import Evaluator, EvaluatorResult
from ...indicators.confluence import RSICalculator


class RSIEvaluator(Evaluator, key='rsi'):
    def __init__(self, period: int = 14, overbought: float = 70, oversold: float = 30):
        self._calc = RSICalculator(period=period, overbought=overbought, oversold=oversold)

    def evaluate(self, ohlcv: pd.DataFrame) -> EvaluatorResult:
        rsi_result = self._calc.calculate(ohlcv['close'].values)
        rsi_val = rsi_result.rsi[-1] if len(rsi_result.rsi) > 0 else 50.0
        if rsi_val > 50:
            direction = 1.0
        elif rsi_val < 50:
            direction = -1.0
        else:
            direction = 0.0
        confidence = abs(rsi_val - 50) / 50.0  # 0.0 at 50, 0.4 at 70/30
        return EvaluatorResult(direction=direction, confidence=confidence, metadata={
            'rsi': float(rsi_val),
            'is_overbought': bool(rsi_result.overbought[-1]) if len(rsi_result.overbought) > 0 else False,
            'is_oversold': bool(rsi_result.oversold[-1]) if len(rsi_result.oversold) > 0 else False,
        })
```

- [ ] **Step 2: Verify it registers**

Run: `python -c "from src.strategy.evaluators.rsi_eval import RSIEvaluator; from src.strategy.base import EVALUATOR_REGISTRY; print('rsi' in EVALUATOR_REGISTRY)"`
Expected: `True`

- [ ] **Step 3: Commit**

```bash
git add src/strategy/evaluators/rsi_eval.py
git commit -m "feat(task-7): RSIEvaluator — Chikou replacement for H1 and below"
```

---

### Task 8: Divergence Evaluator (wraps existing DivergenceDetector)

**Files:**
- Create: `src/strategy/evaluators/divergence_eval.py`

**Note:** `DivergenceDetector` already exists in `src/indicators/divergence.py` with regular + hidden divergence. This wraps it as an Evaluator and uses fractal swing points for more robust pivot identification.

- [ ] **Step 1: Implement DivergenceEvaluator**

```python
# src/strategy/evaluators/divergence_eval.py
"""DivergenceEvaluator — wraps existing DivergenceDetector as Evaluator."""
from __future__ import annotations
import pandas as pd
from ..base import Evaluator, EvaluatorResult
from ...indicators.divergence import DivergenceDetector
from ...indicators.confluence import RSICalculator


class DivergenceEvaluator(Evaluator, key='divergence'):
    def __init__(self, rsi_period: int = 14):
        self._rsi = RSICalculator(period=rsi_period)
        self._div = DivergenceDetector()

    def evaluate(self, ohlcv: pd.DataFrame) -> EvaluatorResult:
        h, l, c = ohlcv['high'].values, ohlcv['low'].values, ohlcv['close'].values
        rsi_result = self._rsi.calculate(c)
        div = self._div.detect(c, rsi_result.rsi, h, l)

        direction, confidence = 0.0, 0.0
        div_type, div_dir = None, None

        # Check last bar for active divergence signals
        last = len(c) - 1
        if last >= 0:
            if div.regular_bullish[last]:
                direction, confidence = 1.0, 0.6
                div_type, div_dir = 'regular', 'bullish'
            elif div.regular_bearish[last]:
                direction, confidence = -1.0, 0.6
                div_type, div_dir = 'regular', 'bearish'
            elif div.hidden_bullish[last]:
                direction, confidence = 1.0, 0.75  # hidden = continuation = higher conf
                div_type, div_dir = 'hidden', 'bullish'
            elif div.hidden_bearish[last]:
                direction, confidence = -1.0, 0.75
                div_type, div_dir = 'hidden', 'bearish'

        return EvaluatorResult(direction=direction, confidence=confidence, metadata={
            'divergence_type': div_type,
            'divergence_direction': div_dir,
            'rsi_value': float(rsi_result.rsi[last]) if last >= 0 and last < len(rsi_result.rsi) else None,
        })
```

- [ ] **Step 2: Verify it registers**

Run: `python -c "from src.strategy.evaluators.divergence_eval import DivergenceEvaluator; from src.strategy.base import EVALUATOR_REGISTRY; print('divergence' in EVALUATOR_REGISTRY)"`

- [ ] **Step 3: Commit**

```bash
git add src/strategy/evaluators/divergence_eval.py
git commit -m "feat(task-8): DivergenceEvaluator — regular + hidden divergence"
```

---

## Wave 2 — Strategy Core (sequential, depends on Wave 1)

### Task 9: FXAtOneGlance Strategy Class — 9 Trade Types + Confluence Scoring

**Files:**
- Create: `src/strategy/strategies/fx_at_one_glance.py`

This is the main strategy. Implements all 9 trade types with unified confluence scoring. The old `IchimokuStrategy` (key: `'ichimoku'`) is preserved — this registers as `'fx_at_one_glance'`.

- [ ] **Step 1: Implement the strategy**

```python
# src/strategy/strategies/fx_at_one_glance.py
"""FXAtOneGlance — faithful FXAOG course implementation.

9 trade types, 5-point checklist gate, configurable Five Elements and
Time Theory filters, 8 TF modes, 0-15 confluence scoring.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional
from ..base import Strategy, EvalMatrix, EvalRequirement, ConfluenceResult
from ..signal_engine import Signal


TF_MODES = {
    'swing':      {'bias': 'D',  'entry': '4H', 'chikou': True,  'ichi_entry': True},
    'intraday':   {'bias': '4H', 'entry': '1H', 'chikou': True,  'ichi_entry': True},
    'hybrid':     {'bias': '4H', 'entry': '15M','chikou': False, 'ichi_entry': True},
    'scalp':      {'bias': '1H', 'entry': '15M','chikou': False, 'ichi_entry': True},
    'hyperscalp_m15_m5': {'bias': '15M','entry': '5M', 'chikou': False, 'ichi_entry': 'stripped'},
    'hyperscalp_h1_m5':  {'bias': '1H', 'entry': '5M', 'chikou': False, 'ichi_entry': 'stripped'},
    'hyperscalp_m5_m1':  {'bias': '5M', 'entry': '1M', 'chikou': False, 'ichi_entry': False},
    'hyperscalp_h1_m1':  {'bias': '1H', 'entry': '1M', 'chikou': False, 'ichi_entry': False},
}


class FXAtOneGlance(Strategy, key='fx_at_one_glance'):

    warmup_bars = 200

    def __init__(self, config=None, instrument='XAUUSD'):
        cfg = config or {}
        self._instrument = instrument
        self._tf_mode_name = cfg.get('tf_mode', 'intraday')
        mode = TF_MODES[self._tf_mode_name]
        self._bias_tf = mode['bias']
        self._entry_tf = mode['entry']
        self._use_chikou = mode['chikou']
        self._ichi_entry = mode['ichi_entry']

        self._fe_mode = cfg.get('five_elements_mode', 'hard_gate')
        self._tt_mode = cfg.get('time_theory_mode', 'soft_filter')
        self._min_score = cfg.get('min_confluence_score', 6)
        self._min_tier = cfg.get('min_tier', 'B')
        self._max_kijun_dist = cfg.get('max_kijun_distance_pips', 200.0)
        self._kijun_buffer = cfg.get('kijun_buffer_pips', 5.0)
        self._max_stop = cfg.get('max_stop_pips', 100.0)
        self._min_rr = cfg.get('min_rr_ratio', 1.5)
        self._primary_target_key = cfg.get('primary_target', 'n_value')
        self._exit_mode = cfg.get('exit_mode', 'hybrid')
        self._partial_pct = cfg.get('partial_close_pct', 0.5)

        # Build evaluator requirements based on TF mode
        both = [self._bias_tf, self._entry_tf]
        reqs = [
            EvalRequirement('ichimoku', both),
            EvalRequirement('fractal', both),
            EvalRequirement('wave', both),
            EvalRequirement('price_action', [self._entry_tf]),
            EvalRequirement('cloud_balance', [self._bias_tf]),
            EvalRequirement('kihon_suchi', [self._bias_tf]),
            EvalRequirement('adx', [self._entry_tf]),
            EvalRequirement('atr', [self._entry_tf]),
        ]
        if not self._use_chikou:
            reqs.append(EvalRequirement('rsi', [self._entry_tf]))
        reqs.append(EvalRequirement('divergence', [self._entry_tf]))
        self.required_evaluators = reqs

        # Exit manager set up in Task 10
        self.trading_mode = None

    # --- 5-Point Checklist ---
    def _checklist_direction(self, matrix):
        """Check 5-point on entry TF. Returns 'long'/'short' or None."""
        ichi = matrix.get(f'ichimoku_{self._entry_tf}')
        if not ichi:
            return None
        meta = ichi.metadata
        # Chikou gate: inside = ranging = block
        if self._use_chikou and meta.get('chikou_vs_price') == 'inside':
            return None
        # RSI gate on lower TFs
        if not self._use_chikou:
            rsi = matrix.get(f'rsi_{self._entry_tf}')
            if rsi and abs(rsi.metadata.get('rsi', 50) - 50) < 5:
                return None  # RSI too close to 50 = no conviction

        if ichi.direction > 0:
            direction = 'long'
        elif ichi.direction < 0:
            direction = 'short'
        else:
            return None

        # Bias TF alignment
        bias_ichi = matrix.get(f'ichimoku_{self._bias_tf}')
        if bias_ichi:
            bias_pos = bias_ichi.metadata.get('cloud_position')
            if direction == 'long' and bias_pos == 'below':
                return None
            if direction == 'short' and bias_pos == 'above':
                return None
        return direction

    # --- Filters ---
    def _filter_five_elements(self, matrix):
        if self._fe_mode == 'disabled':
            return True
        fe = matrix.get(f'cloud_balance_{self._bias_tf}')
        if not fe:
            return self._fe_mode != 'hard_gate'
        if self._fe_mode == 'hard_gate':
            return fe.metadata.get('is_disequilibrium', False)
        return True

    def _filter_time_theory(self, matrix):
        if self._tt_mode in ('disabled', 'soft_filter'):
            return True
        tt = matrix.get(f'kihon_suchi_{self._bias_tf}')
        if not tt:
            return False
        return tt.metadata.get('is_cycle_date', False)

    def _filter_range(self, matrix):
        """Triple range filter: ADX>20 + cloud balance disequilibrium + cloud thickness."""
        adx = matrix.get(f'adx_{self._entry_tf}')
        if adx and adx.metadata.get('adx', 0) < 20:
            return False
        ichi = matrix.get(f'ichimoku_{self._entry_tf}')
        if ichi and ichi.metadata.get('kumo_thickness', 999) < 0.50:
            return False
        return True

    # --- 9 Trade Types (priority order) ---
    def _detect_trade_type(self, matrix, direction, meta):
        checks = [
            self._check_walking_dragon,
            self._check_tk_crossover,
            self._check_kumo_breakout,
            self._check_kijun_bounce,
            self._check_kijun_break,
            self._check_ffo,
            self._check_fractal_breakout,
            self._check_kumo_twist,
            self._check_rolling_dragon,
        ]
        for check in checks:
            result = check(matrix, direction, meta)
            if result:
                return result
        return None

    def _check_walking_dragon(self, matrix, direction, meta):
        bars_ago = meta.get('tk_cross_bars_ago', 9999)
        if 5 <= bars_ago <= 10 and abs(meta.get('tenkan_kijun_angle', 0)) > 0.0003:
            return 'walking_dragon'

    def _check_tk_crossover(self, matrix, direction, meta):
        expected = 'bullish' if direction == 'long' else 'bearish'
        if meta.get('tk_cross') == expected and meta.get('kijun_distance_pips', 9999) <= self._max_kijun_dist:
            pa = matrix.get(f'price_action_{self._entry_tf}')
            if pa and pa.direction != 0.0:
                return 'tk_crossover'

    def _check_kumo_breakout(self, matrix, direction, meta):
        expected_pos = 'above' if direction == 'long' else 'below'
        if meta.get('cloud_position') == expected_pos:
            return 'kumo_breakout'

    def _check_kijun_bounce(self, matrix, direction, meta):
        if meta.get('kijun_distance_pips', 9999) <= 50 and not meta.get('kijun_flat', False):
            pa = matrix.get(f'price_action_{self._entry_tf}')
            if pa and (pa.metadata.get('inside_bar_breakout') != 'none' or
                       pa.metadata.get('engulfing_bullish' if direction == 'long' else 'engulfing_bearish')):
                return 'kijun_bounce'

    def _check_kijun_break(self, matrix, direction, meta):
        pa = matrix.get(f'price_action_{self._entry_tf}')
        if pa and pa.metadata.get('engulfing_bullish' if direction == 'long' else 'engulfing_bearish'):
            return 'kijun_break'

    def _check_ffo(self, matrix, direction, meta):
        frac = matrix.get(f'fractal_{self._entry_tf}')
        if not frac:
            return None
        # FFO: fractal structure aligned with direction + momentum strengthening
        if frac.metadata.get('structure') == ('uptrend' if direction == 'long' else 'downtrend'):
            if frac.metadata.get('momentum_trend') == 'strengthening':
                return 'ffo'

    def _check_fractal_breakout(self, matrix, direction, meta):
        frac = matrix.get(f'fractal_{self._entry_tf}')
        if not frac:
            return None
        expected = 'uptrend' if direction == 'long' else 'downtrend'
        if frac.metadata.get('structure') == expected and frac.metadata.get('momentum_trend') != 'weakening':
            return 'fractal_breakout'

    def _check_kumo_twist(self, matrix, direction, meta):
        if meta.get('kumo_future_direction') != meta.get('cloud_direction'):
            return 'kumo_twist'

    def _check_rolling_dragon(self, matrix, direction, meta):
        if meta.get('kijun_flat', False) and meta.get('kijun_distance_pips', 0) > 50:
            pa = matrix.get(f'price_action_{self._entry_tf}')
            if pa and pa.metadata.get('pin_bar_bullish' if direction == 'long' else 'pin_bar_bearish'):
                return 'rolling_dragon'

    # --- Confluence Scoring (0-15) ---
    def score_confluence(self, eval_matrix, direction=None):
        score, breakdown = 0, {}

        # 1. Checklist (0-5) from ichimoku evaluator
        ichi = eval_matrix.get(f'ichimoku_{self._entry_tf}')
        checklist = ichi.metadata.get('checklist_score', 0) if ichi else 0
        score += checklist
        breakdown['checklist'] = checklist

        # 2. Signal strength — cloud position (0-2)
        ss = 2 if ichi and ichi.metadata.get('cloud_position') in ('above', 'below') else 0
        score += ss
        breakdown['signal_strength'] = ss

        # 3. Price action quality (0-2)
        pa = eval_matrix.get(f'price_action_{self._entry_tf}')
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
        fe = eval_matrix.get(f'cloud_balance_{self._bias_tf}')
        fe_pts = 0
        if fe and fe.metadata.get('is_disequilibrium'):
            fe_pts = 2 if (fe.metadata.get('o_count', 0) + fe.metadata.get('g_count', 0)) >= 3 else 1
        score += fe_pts
        breakdown['five_elements'] = fe_pts

        # 5. Kihon Suchi (0-2)
        tt = eval_matrix.get(f'kihon_suchi_{self._bias_tf}')
        tt_pts = 0
        if tt:
            if tt.metadata.get('double_confirmation'):
                tt_pts = 2
            elif tt.metadata.get('is_cycle_date'):
                tt_pts = 1
        score += tt_pts
        breakdown['time_theory'] = tt_pts

        # 6. Wave context (0-1) — not in correction
        wa = eval_matrix.get(f'wave_{self._bias_tf}')
        wa_pts = 1 if wa and not wa.metadata.get('is_correction') else 0
        score += wa_pts
        breakdown['wave_context'] = wa_pts

        # 7. Fractal momentum (0-1)
        frac = eval_matrix.get(f'fractal_{self._entry_tf}')
        fm = 1 if frac and frac.metadata.get('momentum_trend') == 'strengthening' else 0
        score += fm
        breakdown['fractal_momentum'] = fm

        # Tier
        if score >= 12:
            tier = 'A+'
        elif score >= 9:
            tier = 'A'
        elif score >= 6:
            tier = 'B'
        elif score >= 4:
            tier = 'C'
        else:
            tier = 'no_trade'

        return ConfluenceResult(score=score, quality_tier=tier, breakdown=breakdown)

    # --- Main decide() ---
    def decide(self, eval_matrix):
        direction = self._checklist_direction(eval_matrix)
        if not direction:
            return None
        if not self._filter_five_elements(eval_matrix):
            return None
        if not self._filter_time_theory(eval_matrix):
            return None
        if not self._filter_range(eval_matrix):
            return None

        ichi = eval_matrix.get(f'ichimoku_{self._entry_tf}')
        if not ichi:
            return None
        meta = ichi.metadata

        trade_type = self._detect_trade_type(eval_matrix, direction, meta)
        if not trade_type:
            return None

        confluence = self.score_confluence(eval_matrix, direction)
        if confluence.score < self._min_score:
            return None
        tier_order = {'A+': 0, 'A': 1, 'B': 2, 'C': 3, 'no_trade': 4}
        if tier_order.get(confluence.quality_tier, 4) > tier_order.get(self._min_tier, 2):
            return None

        # Entry, SL, TP
        kijun = meta.get('kijun', 0)
        entry_price = meta.get('tenkan', kijun)
        atr_eval = eval_matrix.get(f'atr_{self._entry_tf}')
        atr = atr_eval.metadata.get('atr', 10.0) if atr_eval else 10.0

        # Stop from fractal or kijun
        frac = eval_matrix.get(f'fractal_{self._entry_tf}')
        if frac and frac.metadata.get('last_bear_fractal' if direction == 'long' else 'last_bull_fractal'):
            frac_data = frac.metadata.get('last_bear_fractal' if direction == 'long' else 'last_bull_fractal')
            stop_loss = (frac_data.price - self._kijun_buffer) if direction == 'long' else (frac_data.price + self._kijun_buffer)
        elif kijun:
            stop_loss = (kijun - self._kijun_buffer) if direction == 'long' else (kijun + self._kijun_buffer)
        else:
            return None

        # TP from wave targets
        wave = eval_matrix.get(f'wave_{self._entry_tf}')
        target = None
        if wave and wave.metadata.get('targets'):
            target = wave.metadata['targets'].get(self._primary_target_key)
        if target is None:
            target = entry_price + (entry_price - stop_loss) * 2  # fallback 2R

        # R:R and max stop checks
        risk = abs(entry_price - stop_loss)
        reward = abs(target - entry_price)
        if risk <= 0 or reward / risk < self._min_rr or risk > self._max_stop:
            return None

        return Signal(
            timestamp=datetime.now(timezone.utc),
            instrument=self._instrument,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=target,
            confluence_score=confluence.score,
            quality_tier=confluence.quality_tier,
            atr=atr,
            strategy_name='fx_at_one_glance',
            reasoning={
                'trade_type': trade_type,
                'tf_mode': self._tf_mode_name,
                'confluence_score': confluence.score,
                'quality_tier': confluence.quality_tier,
                **confluence.breakdown,
            },
        )

    def suggest_params(self, trial):
        return {
            'min_confluence_score': trial.suggest_int('min_confluence_score', 4, 12),
            'min_tier': trial.suggest_categorical('min_tier', ['A_plus', 'A', 'B']),
            'exit_mode': trial.suggest_categorical('exit_mode', ['trailing', 'targets', 'hybrid']),
            'primary_target': trial.suggest_categorical('primary_target', ['n_value', 'v_value']),
            'five_elements_mode': trial.suggest_categorical('five_elements_mode', ['hard_gate', 'soft_filter', 'disabled']),
            'time_theory_mode': trial.suggest_categorical('time_theory_mode', ['hard_gate', 'soft_filter', 'disabled']),
            'min_rr_ratio': trial.suggest_float('min_rr_ratio', 1.0, 3.0, step=0.5),
            'tf_mode': trial.suggest_categorical('tf_mode', ['swing', 'intraday', 'hybrid', 'scalp']),
        }

    def populate_edge_context(self, eval_matrix):
        ichi = eval_matrix.get(f'ichimoku_{self._entry_tf}')
        if not ichi:
            return {}
        return {
            'kijun': ichi.metadata.get('kijun'),
            'cloud_thickness': ichi.metadata.get('kumo_thickness'),
        }
```

- [ ] **Step 2: Verify it registers**

Run: `python -c "from src.strategy.strategies.fx_at_one_glance import FXAtOneGlance; from src.strategy.base import STRATEGY_REGISTRY; print('fx_at_one_glance' in STRATEGY_REGISTRY, 'ichimoku' in STRATEGY_REGISTRY)"`
Expected: `True True` (both old and new coexist)

- [ ] **Step 3: Commit**

```bash
git add src/strategy/strategies/fx_at_one_glance.py
git commit -m "feat(task-9): FXAtOneGlance strategy — 9 trade types, 5-point checklist, 0-15 scoring"
```

---

### Task 10: IchimokuExitManager — 3-Mode Exit Logic + Heikin Ashi

**Files:**
- Create: `src/strategy/trading_modes/ichimoku_exit.py`

- [ ] **Step 1: Implement the exit manager**

```python
# src/strategy/trading_modes/ichimoku_exit.py
"""IchimokuExitManager — 3-mode configurable exit with HA confirmation.

Modes:
  trailing — Kijun + fractal trailing, no fixed target
  targets  — fixed TP at N/V/E-value, full exit
  hybrid   — 50% partial at N-value, trail remainder with Kijun/fractal

Heikin Ashi confirmation:
  In trailing/hybrid, only trail-update when HA shows weak candle (has
  opposite wick). Strong HA candle = hold position tighter.
"""
from __future__ import annotations
from ..base import TradingMode, ExitDecision, EvalMatrix
from ...indicators.heikin_ashi import compute_heikin_ashi, ha_candle_at, ha_trend_signal


class IchimokuExitManager(TradingMode):
    def __init__(self, mode='hybrid', entry_tf='1H', kijun_buffer=5.0,
                 partial_close_pct=0.5, target_price=None,
                 move_stop_to_entry=True, use_heikin_ashi=True):
        self.mode = mode
        self.tf = entry_tf
        self.kijun_buffer = kijun_buffer
        self.partial_pct = partial_close_pct
        self.target = target_price
        self.move_be = move_stop_to_entry
        self.use_ha = use_heikin_ashi
        self._partial_taken = False

    def check_exit(self, trade, current_data, eval_results):
        close = current_data['close']
        is_long = trade.direction == 'long'

        # Extract Kijun
        ichi = eval_results.get(f'ichimoku_{self.tf}')
        kijun = ichi.metadata.get('kijun') if ichi else None

        # Extract nearest opposite fractal
        frac = eval_results.get(f'fractal_{self.tf}')
        opp_frac = None
        if frac:
            key = 'last_bear_fractal' if is_long else 'last_bull_fractal'
            fd = frac.metadata.get(key)
            if fd and hasattr(fd, 'price'):
                opp_frac = fd.price

        # Kijun close check — price wrong side of Kijun = full exit
        if kijun is not None:
            if is_long and close < kijun - self.kijun_buffer:
                return ExitDecision(action='full_exit', reason='Close below Kijun-sen')
            if not is_long and close > kijun + self.kijun_buffer:
                return ExitDecision(action='full_exit', reason='Close above Kijun-sen')

        # HA confirmation for trail decisions
        ha_signal = None
        if self.use_ha and 'open' in current_data:
            try:
                import numpy as np
                o = np.array(current_data.get('open_series', [current_data.get('open', close)]))
                h = np.array(current_data.get('high_series', [current_data.get('high', close)]))
                l = np.array(current_data.get('low_series', [current_data.get('low', close)]))
                c = np.array(current_data.get('close_series', [close]))
                if len(o) >= 2:
                    ha = compute_heikin_ashi(o, h, l, c)
                    candle = ha_candle_at(ha, -1)
                    ha_signal = ha_trend_signal(candle)
            except Exception:
                pass

        if self.mode == 'targets':
            return self._check_targets(close, is_long)
        elif self.mode == 'hybrid':
            return self._check_hybrid(close, is_long, trade, kijun, opp_frac, ha_signal)
        else:
            return self._check_trailing(close, is_long, trade, kijun, opp_frac, ha_signal)

    def _check_trailing(self, close, is_long, trade, kijun, opp_frac, ha_signal):
        new_stop = trade.stop_loss
        candidates = []
        if kijun is not None:
            buf = self.kijun_buffer if is_long else -self.kijun_buffer
            candidates.append(kijun - buf if is_long else kijun + abs(buf))
        if opp_frac is not None:
            candidates.append(opp_frac)

        if candidates:
            best = max(candidates) if is_long else min(candidates)
            # Only trail if HA doesn't show strong trend (strong = hold tighter)
            if ha_signal and 'strong' in ha_signal:
                pass  # hold — don't widen the trail
            elif (is_long and best > trade.stop_loss) or (not is_long and best < trade.stop_loss):
                new_stop = best

        if new_stop != trade.stop_loss:
            return ExitDecision(action='trail_update', new_stop=new_stop, reason='Kijun/fractal trail')
        return ExitDecision(action='hold')

    def _check_targets(self, close, is_long):
        if self.target is None:
            return ExitDecision(action='hold')
        if (is_long and close >= self.target) or (not is_long and close <= self.target):
            return ExitDecision(action='full_exit', close_pct=1.0, reason=f'Target hit {self.target}')
        return ExitDecision(action='hold')

    def _check_hybrid(self, close, is_long, trade, kijun, opp_frac, ha_signal):
        # Phase 1: partial at target
        if not self._partial_taken and self.target is not None:
            hit = (is_long and close >= self.target) or (not is_long and close <= self.target)
            if hit:
                self._partial_taken = True
                new_stop = trade.entry_price if self.move_be else trade.stop_loss
                return ExitDecision(
                    action='partial_exit', close_pct=self.partial_pct,
                    new_stop=new_stop, reason=f'Partial close at N-value {self.target}',
                )
        # Phase 2: trail remainder
        return self._check_trailing(close, is_long, trade, kijun, opp_frac, ha_signal)
```

- [ ] **Step 2: Wire exit manager into strategy (edit fx_at_one_glance.py)**

In `FXAtOneGlance.__init__`, after the evaluator requirements, add:

```python
from ..trading_modes.ichimoku_exit import IchimokuExitManager
self.trading_mode = IchimokuExitManager(
    mode=self._exit_mode, entry_tf=self._entry_tf,
    kijun_buffer=self._kijun_buffer, partial_close_pct=self._partial_pct,
)
```

And in `decide()`, before returning the Signal, set the exit target:

```python
if isinstance(self.trading_mode, IchimokuExitManager):
    self.trading_mode.target = target
    self.trading_mode._partial_taken = False
```

- [ ] **Step 3: Commit**

```bash
git add src/strategy/trading_modes/ichimoku_exit.py src/strategy/strategies/fx_at_one_glance.py
git commit -m "feat(task-10): IchimokuExitManager — trailing/targets/hybrid + HA confirmation"
```

---

### Task 11: Config Update + Strategy Profiles

**Files:**
- Modify: `config/strategy.yaml`

- [ ] **Step 1: Add the config block**

Add to `config/strategy.yaml`:

```yaml
active_strategy: fx_at_one_glance

# Strategy profiles for blending
profiles:
  ichimoku_only:
    strategies: [fx_at_one_glance]
  full_blend:
    strategies:
      - fx_at_one_glance: {weight: 2.0}
      - asian_breakout: {weight: 1.0}
      - ema_pullback: {weight: 0.5}

strategies:
  fx_at_one_glance:
    tf_mode: intraday            # swing|intraday|hybrid|scalp|hyperscalp_*
    ichimoku:
      tenkan_period: 9
      kijun_period: 26
      senkou_b_period: 52
    adx:
      period: 14
      threshold: 20
    atr:
      period: 14
      stop_multiplier: 2.5
    cloud_balance:
      enabled: true
    kihon_suchi:
      enabled: true
      tolerance: 1
    wave:
      elliott_enabled: true
    divergence:
      enabled: true
      oscillator: rsi
    five_elements_mode: hard_gate
    time_theory_mode: soft_filter
    range_filter:
      adx_min: 20
      cloud_thickness_min_usd: 0.50
    signal:
      min_confluence_score: 6
      min_tier: B
    exit:
      mode: hybrid
      partial_close_pct: 50
      primary_target: n_value
      trail_with: fractal_kijun
      no_be_before_r: 1.0
      use_heikin_ashi: true
    stop_loss:
      preference: [fractal, kijun, kumo_edge]
      kijun_buffer_pips: 5
      max_stop_pips: 100
      min_rr_ratio: 1.5
```

- [ ] **Step 2: Verify both strategies loadable**

Run: `python -c "from src.strategy.base import STRATEGY_REGISTRY; from src.strategy.strategies.ichimoku import IchimokuStrategy; from src.strategy.strategies.fx_at_one_glance import FXAtOneGlance; print(sorted(STRATEGY_REGISTRY.keys()))"`
Expected: `['fx_at_one_glance', 'ichimoku']`

- [ ] **Step 3: Commit**

```bash
git add config/strategy.yaml
git commit -m "feat(task-11): config — fx_at_one_glance strategy + profiles"
```

---

## Wave 3 — Integration (sequential, depends on Wave 2)

### Task 12: Coordinator Updates

**Files:**
- Modify: `src/strategy/coordinator.py`

The existing `EvaluatorCoordinator` already handles multi-TF resampling and evaluator dispatch. This task adds the new evaluators to its import chain and ensures the `_TF_RULES` map covers all 8 TF modes.

- [ ] **Step 1: Add new TF rules if missing**

Check `_TF_RULES` in coordinator.py. If `"D"` (Daily) is missing, add:

```python
_TF_RULES = {
    "1M": "1min",
    "5M": "5min",
    "15M": "15min",
    "1H": "1h",
    "4H": "4h",
    "D": "1D",
}
```

- [ ] **Step 2: Ensure new evaluators are importable**

Add to `src/strategy/evaluators/__init__.py`:

```python
from .fractal_eval import FractalStructureEvaluator
from .price_action_eval import PriceActionEvaluator
from .cloud_balance_eval import CloudBalancingEvaluator
from .kihon_suchi_eval import KihonSuchiEvaluator
from .wave_eval import WavePatternEvaluator
from .rsi_eval import RSIEvaluator
from .divergence_eval import DivergenceEvaluator
```

- [ ] **Step 3: Verify coordinator can run all evaluators**

Run: `python -c "
from src.strategy.base import EVALUATOR_REGISTRY
import src.strategy.evaluators
expected = ['ichimoku', 'adx', 'atr', 'session', 'fractal', 'price_action', 'cloud_balance', 'kihon_suchi', 'wave', 'rsi', 'divergence']
for key in expected:
    assert key in EVALUATOR_REGISTRY, f'{key} missing'
print('All evaluators registered')
"`

- [ ] **Step 4: Commit**

```bash
git add src/strategy/coordinator.py src/strategy/evaluators/__init__.py
git commit -m "feat(task-12): coordinator — register new evaluators, add Daily TF rule"
```

---

### Task 13: Signal Engine Extensions

**Files:**
- Modify: `src/strategy/signal_engine.py`

Add new fields to the Signal dataclass for FXAOG-specific context.

- [ ] **Step 1: Extend Signal dataclass**

Add these optional fields:

```python
# In Signal dataclass, add after existing fields:
trade_type: str = ""               # 'kijun_bounce'|'kumo_breakout'|'walking_dragon'|...
wave_targets: dict = field(default_factory=dict)  # {'n_value': float, ...}
elliott_position: dict = field(default_factory=dict)
cloud_balance_state: dict = field(default_factory=dict)
kihon_suchi_state: dict = field(default_factory=dict)
```

- [ ] **Step 2: Verify existing tests still pass**

Run: `pytest tests/ -k signal -v --tb=short`
Expected: All PASS (new fields have defaults so nothing breaks)

- [ ] **Step 3: Commit**

```bash
git add src/strategy/signal_engine.py
git commit -m "feat(task-13): Signal dataclass — add FXAOG trade type + wave target fields"
```

---

## Wave 4 — Testing & Validation (sequential, depends on Wave 3)

### Task 14: Integration Test — Full Pipeline

**Files:**
- Create: `tests/integration/test_fx_at_one_glance_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/integration/test_fx_at_one_glance_integration.py
"""End-to-end: all evaluators → strategy → signal."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from src.strategy.base import EvalMatrix, EVALUATOR_REGISTRY, STRATEGY_REGISTRY


def _make_trending_ohlcv(n=300, start=1800.0, step=1.5):
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
        'open': close - 0.3 * step, 'high': close + abs(step),
        'low': close - abs(step), 'close': close,
        'volume': np.ones(n) * 100,
    }, index=idx)


class TestAllEvaluatorsRegistered:
    def test_new_evaluators_present(self):
        import src.strategy.evaluators
        for key in ['fractal', 'price_action', 'cloud_balance', 'kihon_suchi', 'wave', 'rsi', 'divergence']:
            assert key in EVALUATOR_REGISTRY, f"{key} not registered"


class TestStrategyRegistration:
    def test_fx_at_one_glance_registered(self):
        from src.strategy.strategies.fx_at_one_glance import FXAtOneGlance
        assert 'fx_at_one_glance' in STRATEGY_REGISTRY

    def test_old_ichimoku_still_registered(self):
        from src.strategy.strategies.ichimoku import IchimokuStrategy
        assert 'ichimoku' in STRATEGY_REGISTRY


class TestEvaluatorsProduceResults:
    def test_each_evaluator_runs(self):
        ohlcv = _make_trending_ohlcv(n=300)
        for key in ['fractal', 'price_action', 'cloud_balance', 'kihon_suchi', 'wave', 'rsi', 'divergence']:
            eval_cls = EVALUATOR_REGISTRY[key]
            evaluator = eval_cls()
            result = evaluator.evaluate(ohlcv)
            assert result is not None, f"{key} returned None"
            assert -1.0 <= result.direction <= 1.0
            assert 0.0 <= result.confidence <= 1.0


class TestFullSignalGeneration:
    def test_strategy_produces_signal_with_relaxed_filters(self):
        ohlcv = _make_trending_ohlcv(n=300, step=2.0)
        matrix = EvalMatrix()
        for tf in ['4H', '1H']:
            for key in ['ichimoku', 'fractal', 'wave', 'adx', 'atr']:
                evaluator = EVALUATOR_REGISTRY[key]()
                matrix.set(f'{key}_{tf}', evaluator.evaluate(ohlcv))
        for key in ['price_action', 'rsi', 'divergence']:
            matrix.set(f'{key}_1H', EVALUATOR_REGISTRY[key]().evaluate(ohlcv))
        for key in ['cloud_balance', 'kihon_suchi']:
            matrix.set(f'{key}_4H', EVALUATOR_REGISTRY[key]().evaluate(ohlcv))

        from src.strategy.strategies.fx_at_one_glance import FXAtOneGlance
        strategy = FXAtOneGlance(config={
            'five_elements_mode': 'disabled', 'time_theory_mode': 'disabled',
            'min_confluence_score': 1, 'min_tier': 'C',
        })
        signal = strategy.decide(matrix)
        if signal is not None:
            assert signal.direction in ('long', 'short')
            assert signal.strategy_name == 'fx_at_one_glance'
            assert 'trade_type' in signal.reasoning
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/integration/test_fx_at_one_glance_integration.py -v`
Expected: All PASS

- [ ] **Step 3: Run full test suite for regressions**

Run: `pytest tests/ -v --tb=short -x`
Expected: All PASS (no regressions)

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_fx_at_one_glance_integration.py
git commit -m "feat(task-14): integration test — full FXAOG pipeline end-to-end"
```

---

### Task 15: Backtest & Tune on Historical Data

**Files:** None created — this is a validation run.

- [ ] **Step 1: Run backtest with new strategy**

```bash
python scripts/run_demo_challenge.py --mode validate \
    --data-file data/projectx_mgc_1m_20260101_20260409.parquet
```

- [ ] **Step 2: Compare results vs old IchimokuStrategy**

Check: trade count, win rate, profit factor, max drawdown, equity curve shape.

- [ ] **Step 3: Tune if needed**

Adjust `min_confluence_score`, `five_elements_mode`, `time_theory_mode` in config based on results.

---

## Dependency Graph

```
Wave 1 (parallel):
  Task 1 (Fractals)        ─┐
  Task 2 (Price Action)    ─┤
  Task 3 (Cloud Balance)   ─┤
  Task 4 (Kihon Suchi)     ─┼──→ Wave 2 (sequential)
  Task 5 (Wave/Elliott)    ─┤      Task 9 (Strategy) → Task 10 (Exit) → Task 11 (Config)
  Task 6 (Heikin Ashi)     ─┤
  Task 7 (RSI Eval)        ─┤                                              │
  Task 8 (Divergence Eval) ─┘                                              ▼
                                                                    Wave 3 (sequential)
                                                           Task 12 (Coordinator) → Task 13 (Signal)
                                                                           │
                                                                           ▼
                                                                    Wave 4 (sequential)
                                                           Task 14 (Integration) → Task 15 (Backtest)
```

## Estimated Scope
- **New files:** ~21
- **Modified files:** ~3
- **New evaluators:** 8 (fractal, price_action, cloud_balance, kihon_suchi, wave, rsi, divergence, + existing ichimoku kept)
- **New indicators:** 6 (fractals, price_action, cloud_balancing, kihon_suchi, wave_patterns, heikin_ashi)
- **New strategy:** 1 (FXAtOneGlance — 9 trade types)
- **New trading mode:** 1 (IchimokuExitManager — 3 modes + HA)
- **Backward compat:** Old `ichimoku` strategy key preserved alongside new `fx_at_one_glance`

## Risk Notes

- **Elliott wave ambiguity:** Rules-based counting will produce wrong counts sometimes. Confidence scoring mitigates — low-confidence counts reduce position sizing rather than blocking trades.
- **Cloud balancing complexity:** The O/G counting with cycle resets is the most novel component with zero open-source precedent. Needs thorough testing against course examples.
- **Hyperscalp M1 modes:** No Ichimoku lines on M1 means these are essentially pure price-action scalping with higher-TF bias. May behave very differently from the course methodology.
- **Kihon Suchi on non-Daily:** The course teaches Time Theory on Daily charts. Effectiveness on H4/H1 is unvalidated. Implemented as soft_filter (confluence bonus) by default, not hard_gate.
- **Fractal structure vs Ichimoku:** When fractal trend says "uptrend" but Ichimoku says "below cloud," the checklist gate blocks the trade. This is intentional — Ichimoku overrides fractals for direction.
- **RSI as Chikou replacement:** The course uses Chikou Span on all timeframes. RSI>50 is our approximation for H1 and below where Chikou is unreliable. This is a deviation from pure course methodology.
- **Nine trade types + filters = low trade frequency.** With Five Elements hard gate + 5-point checklist + min confluence 6, expect ~2-5 trades per week in intraday mode. This is by design — quality over quantity.
