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
