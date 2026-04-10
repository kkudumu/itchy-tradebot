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
