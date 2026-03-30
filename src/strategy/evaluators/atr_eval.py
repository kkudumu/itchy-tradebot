"""ATR evaluator — wraps ATRCalculator to measure normalised volatility."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.confluence import ATRCalculator
from src.strategy.base import Evaluator, EvaluatorResult

_ROLLING_WINDOW = 100


class ATREvaluator(Evaluator, key='atr'):
    """Evaluate ATR-based volatility at the last bar.

    ATR is non-directional, so direction is always 0.0.

    Confidence is the min-max normalised ATR value over the last 100 bars,
    representing how elevated current volatility is relative to recent history:
    - 0.0 → ATR at its lowest point in the 100-bar window
    - 1.0 → ATR at its highest point in the 100-bar window
    - 0.5 → fallback when there is no valid range (all-NaN or flat)
    """

    def __init__(self) -> None:
        self._calc = ATRCalculator(period=14)

    def evaluate(self, ohlcv: pd.DataFrame) -> EvaluatorResult:
        """Compute rolling-normalised ATR confidence at the last bar.

        Parameters
        ----------
        ohlcv:
            OHLCV DataFrame with at minimum columns ``high``, ``low``, ``close``.

        Returns
        -------
        EvaluatorResult
            direction=0.0, normalised confidence in [0,1], and raw ATR value.
        """
        high = ohlcv['high'].to_numpy(dtype=np.float64)
        low = ohlcv['low'].to_numpy(dtype=np.float64)
        close = ohlcv['close'].to_numpy(dtype=np.float64)

        atr_array = self._calc.calculate(high, low, close)

        idx = len(atr_array) - 1
        atr_val = float(atr_array[idx])

        # Rolling-normalise: use up to the last _ROLLING_WINDOW valid values
        window_start = max(0, idx + 1 - _ROLLING_WINDOW)
        window = atr_array[window_start: idx + 1]
        valid = window[~np.isnan(window)]

        if len(valid) >= 2:
            rolling_min = float(np.min(valid))
            rolling_max = float(np.max(valid))
            spread = rolling_max - rolling_min
            if spread > 0 and not np.isnan(atr_val):
                confidence = float(np.clip((atr_val - rolling_min) / spread, 0.0, 1.0))
            else:
                confidence = 0.5
        else:
            confidence = 0.5

        return EvaluatorResult(
            direction=0.0,
            confidence=confidence,
            metadata={'atr': atr_val},
        )
