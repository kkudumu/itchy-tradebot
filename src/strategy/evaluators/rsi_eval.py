"""RSIEvaluator — wraps existing RSICalculator. Used as Chikou replacement on H1 and below."""
from __future__ import annotations

import pandas as pd

from ..base import Evaluator, EvaluatorResult
from ...indicators.confluence import RSICalculator


class RSIEvaluator(Evaluator, key='rsi'):
    """Wraps :class:`RSICalculator` and normalises output into an :class:`EvaluatorResult`.

    Direction is bullish when RSI > 50 and bearish when RSI < 50.
    Confidence scales linearly from 0 at RSI == 50 to 1.0 at RSI == 0 or 100.

    Parameters
    ----------
    period:
        RSI lookback period. Default: 14.
    overbought:
        RSI level above which the market is considered overbought. Default: 70.
    oversold:
        RSI level below which the market is considered oversold. Default: 30.
    """

    def __init__(
        self,
        period: int = 14,
        overbought: float = 70.0,
        oversold: float = 30.0,
    ) -> None:
        self._calc = RSICalculator(period=period, overbought=overbought, oversold=oversold)

    def evaluate(self, ohlcv: pd.DataFrame) -> EvaluatorResult:
        """Compute RSI and return a normalised direction/confidence result.

        Parameters
        ----------
        ohlcv:
            OHLCV DataFrame with at minimum a ``close`` column.

        Returns
        -------
        EvaluatorResult
            direction: +1.0 (RSI > 50), -1.0 (RSI < 50), 0.0 (RSI == 50).
            confidence: abs(rsi - 50) / 50, clamped to [0, 1].
            metadata: ``rsi``, ``is_overbought``, ``is_oversold``.
        """
        rsi_result = self._calc.calculate(ohlcv['close'].values)

        rsi_val = float(rsi_result.rsi[-1]) if len(rsi_result.rsi) > 0 else 50.0

        if rsi_val > 50:
            direction = 1.0
        elif rsi_val < 50:
            direction = -1.0
        else:
            direction = 0.0

        confidence = abs(rsi_val - 50) / 50.0

        is_overbought = bool(rsi_result.overbought[-1]) if len(rsi_result.overbought) > 0 else False
        is_oversold = bool(rsi_result.oversold[-1]) if len(rsi_result.oversold) > 0 else False

        return EvaluatorResult(
            direction=direction,
            confidence=confidence,
            metadata={
                'rsi': rsi_val,
                'is_overbought': is_overbought,
                'is_oversold': is_oversold,
            },
        )
