"""Bill Williams 5-bar fractal detection and market structure tracking."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class FractalLevel:
    price: float
    bar_index: int


@dataclass
class FractalResult:
    bull_fractals: list[FractalLevel] = field(default_factory=list)
    bear_fractals: list[FractalLevel] = field(default_factory=list)


@dataclass
class StructureState:
    trend: str  # 'uptrend' | 'downtrend' | 'ranging'
    last_bull_fractal: FractalLevel | None = None
    last_bear_fractal: FractalLevel | None = None
    swing_highs: list[float] = field(default_factory=list)
    swing_lows: list[float] = field(default_factory=list)


def detect_fractals(high: np.ndarray, low: np.ndarray, window: int = 2) -> FractalResult:
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
    fr = detect_fractals(high, low, window)
    state = StructureState(trend='ranging')
    if fr.bull_fractals:
        state.last_bull_fractal = fr.bull_fractals[-1]
        state.swing_highs = [f.price for f in fr.bull_fractals[-5:]]
    if fr.bear_fractals:
        state.last_bear_fractal = fr.bear_fractals[-1]
        state.swing_lows = [f.price for f in fr.bear_fractals[-5:]]
    if len(state.swing_highs) >= 3 and len(state.swing_lows) >= 3:
        # Require two consecutive HH+HL pairs for uptrend, LH+LL pairs for downtrend
        highs = state.swing_highs[-3:]
        lows = state.swing_lows[-3:]
        all_hh = all(highs[i + 1] > highs[i] for i in range(len(highs) - 1))
        all_hl = all(lows[i + 1] > lows[i] for i in range(len(lows) - 1))
        all_lh = all(highs[i + 1] < highs[i] for i in range(len(highs) - 1))
        all_ll = all(lows[i + 1] < lows[i] for i in range(len(lows) - 1))
        if all_hh and all_hl:
            state.trend = 'uptrend'
        elif all_lh and all_ll:
            state.trend = 'downtrend'
    return state


def fractal_momentum(fractals: list[FractalLevel]) -> list[float]:
    if len(fractals) < 2:
        return []
    sorted_f = sorted(fractals, key=lambda f: f.bar_index)
    return [abs(sorted_f[i + 1].price - sorted_f[i].price) for i in range(len(sorted_f) - 1)]


def momentum_trend(distances: list[float]) -> str:
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
