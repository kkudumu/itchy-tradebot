"""ADX evaluator — wraps ADXCalculator to measure trend direction and strength."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.confluence import ADXCalculator
from src.strategy.base import Evaluator, EvaluatorResult


class ADXEvaluator(Evaluator, key='adx'):
    """Evaluate ADX trend direction and strength at the last bar.

    Direction is determined by the dominant directional indicator:
    - +1.0 when +DI > -DI  (bullish pressure dominant)
    - -1.0 when -DI >= +DI (bearish pressure dominant)

    Confidence is the raw ADX value normalised to [0, 1] by dividing by 100.
    """

    def __init__(self) -> None:
        self._calc = ADXCalculator(period=14, threshold=28.0)

    def evaluate(self, ohlcv: pd.DataFrame) -> EvaluatorResult:
        """Compute ADX signal at the last bar.

        Parameters
        ----------
        ohlcv:
            OHLCV DataFrame with at minimum columns ``high``, ``low``, ``close``.

        Returns
        -------
        EvaluatorResult
            direction (+1 or -1), normalised confidence, and ADX metadata.
        """
        high = ohlcv['high'].to_numpy(dtype=np.float64)
        low = ohlcv['low'].to_numpy(dtype=np.float64)
        close = ohlcv['close'].to_numpy(dtype=np.float64)

        result = self._calc.calculate(high, low, close)

        idx = len(close) - 1
        adx_val = float(result.adx[idx])
        plus_di_val = float(result.plus_di[idx])
        minus_di_val = float(result.minus_di[idx])
        is_trending_val = bool(result.is_trending[idx])

        direction = 1.0 if plus_di_val > minus_di_val else -1.0

        # Normalise ADX to [0, 1]; cap at 1.0 in case ADX exceeds 100
        confidence = min(adx_val / 100.0, 1.0) if not np.isnan(adx_val) else 0.0

        metadata = {
            'adx':         adx_val,
            'plus_di':     plus_di_val,
            'minus_di':    minus_di_val,
            'is_trending': is_trending_val,
        }

        return EvaluatorResult(
            direction=direction,
            confidence=confidence,
            metadata=metadata,
        )
