"""Ichimoku evaluator — wraps IchimokuCalculator and IchimokuSignals."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.indicators.ichimoku import IchimokuCalculator
from src.indicators.signals import IchimokuSignals
from src.strategy.base import Evaluator, EvaluatorResult


class IchimokuEvaluator(Evaluator, key='ichimoku'):
    """Evaluate Ichimoku Kinko Hyo confluence on a single timeframe.

    Returns a normalised direction (−1/0/+1) and a confidence score
    built from four independent signal dimensions:

    - TK cross alignment with cloud direction      (+0.25)
    - Cloud position (price vs cloud) alignment    (+0.25)
    - Chikou confirmation alignment                (+0.25)
    - Non-trivial cloud thickness (thickness > 0)  (+0.25)
    """

    def __init__(self) -> None:
        self._calc = IchimokuCalculator(
            tenkan_period=9,
            kijun_period=26,
            senkou_b_period=52,
        )
        self._sig = IchimokuSignals()

    def evaluate(self, ohlcv: pd.DataFrame) -> EvaluatorResult:
        """Compute Ichimoku signal state at the last bar.

        Parameters
        ----------
        ohlcv:
            OHLCV DataFrame indexed by timestamp with at minimum columns
            ``high``, ``low``, ``close``.

        Returns
        -------
        EvaluatorResult
            direction, confidence, and raw component metadata.
        """
        high = ohlcv['high'].to_numpy(dtype=np.float64)
        low = ohlcv['low'].to_numpy(dtype=np.float64)
        close = ohlcv['close'].to_numpy(dtype=np.float64)

        result = self._calc.calculate(high, low, close)

        idx = len(close) - 1
        state = self._sig.signal_state_at(
            idx,
            result.tenkan_sen,
            result.kijun_sen,
            close,
            result.senkou_a,
            result.senkou_b,
            result.chikou_span,
        )

        # Map cloud_direction (int) → float direction
        direction = float(state.cloud_direction)  # already -1, 0, or 1

        # Confidence: accumulate 0.25 per aligned dimension
        confidence = 0.0
        if direction != 0.0:
            # 1. TK cross matches direction
            if state.tk_cross != 0 and float(state.tk_cross) == direction:
                confidence += 0.25
            # 2. Cloud position matches direction
            if state.cloud_position != 0 and float(state.cloud_position) == direction:
                confidence += 0.25
            # 3. Chikou confirms direction
            if state.chikou_confirmed != 0 and float(state.chikou_confirmed) == direction:
                confidence += 0.25
        else:
            # Neutral cloud — check partial alignments still
            if state.tk_cross != 0:
                confidence += 0.25
            if state.cloud_position != 0:
                confidence += 0.25
            if state.chikou_confirmed != 0:
                confidence += 0.25

        # 4. Non-trivial cloud thickness
        if not math.isnan(state.cloud_thickness) and state.cloud_thickness > 0:
            confidence += 0.25

        confidence = min(confidence, 1.0)

        # Retrieve scalar values safely (NaN → None is not needed; keep float)
        def _safe(arr: np.ndarray, i: int) -> float:
            val = arr[i]
            return float(val)  # may be nan — that's fine for metadata

        metadata = {
            'tenkan':            _safe(result.tenkan_sen, idx),
            'kijun':             _safe(result.kijun_sen, idx),
            'senkou_a':          _safe(result.senkou_a, idx),
            'senkou_b':          _safe(result.senkou_b, idx),
            'chikou':            _safe(result.chikou_span, idx),
            'cloud_thickness':   state.cloud_thickness,
            'cloud_direction':   state.cloud_direction,
            'tk_cross':          state.tk_cross,
            'cloud_position':    state.cloud_position,
            'chikou_confirmed':  state.chikou_confirmed,
        }

        return EvaluatorResult(
            direction=direction,
            confidence=confidence,
            metadata=metadata,
        )
