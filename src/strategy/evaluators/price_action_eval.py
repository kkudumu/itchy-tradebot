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
