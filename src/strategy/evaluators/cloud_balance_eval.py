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
