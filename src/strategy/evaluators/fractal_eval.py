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
