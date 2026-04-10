"""Candlestick price action pattern detection."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class PriceActionResult:
    tweezer_bottom: bool = False
    tweezer_top: bool = False
    inside_bar_count: int = 0
    inside_bar_breakout: str = 'none'
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

    if full_range > 0 and body / full_range <= doji_body_pct:
        result.doji = True

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

    if _is_red(o_p, c_p) and _is_green(o_l, c_l) and abs(l_p - l_l) <= tick_tolerance:
        result.tweezer_bottom = True
    if _is_green(o_p, c_p) and _is_red(o_l, c_l) and abs(h_p - h_l) <= tick_tolerance:
        result.tweezer_top = True

    prev_bt, prev_bb = _body_top(o_p, c_p), _body_bot(o_p, c_p)
    last_bt, last_bb = _body_top(o_l, c_l), _body_bot(o_l, c_l)
    if _is_red(o_p, c_p) and _is_green(o_l, c_l) and last_bt >= prev_bt and last_bb <= prev_bb and body > 0:
        result.engulfing_bullish = True
    if _is_green(o_p, c_p) and _is_red(o_l, c_l) and last_bt >= prev_bt and last_bb <= prev_bb and body > 0:
        result.engulfing_bearish = True

    if n >= 3:
        best_count = 0
        best_m_top = 0.0
        best_m_bot = 0.0
        search_start = max(last - 10, 0)
        for mi in range(search_start, last):
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
            if count > best_count:
                best_count = count
                best_m_top = m_top
                best_m_bot = m_bot
        if best_count > 0:
            result.inside_bar_count = best_count
            result.mother_bar_high = best_m_top
            result.mother_bar_low = best_m_bot
            if close[last] > best_m_top:
                result.inside_bar_breakout = 'up'
            elif close[last] < best_m_bot:
                result.inside_bar_breakout = 'down'

    return result
