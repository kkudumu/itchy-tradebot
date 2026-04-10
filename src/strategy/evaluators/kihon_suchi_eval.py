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
