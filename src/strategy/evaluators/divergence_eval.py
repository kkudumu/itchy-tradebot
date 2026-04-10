"""DivergenceEvaluator — wraps existing DivergenceDetector as Evaluator."""
from __future__ import annotations

import pandas as pd

from ..base import Evaluator, EvaluatorResult
from ...indicators.divergence import DivergenceDetector
from ...indicators.confluence import RSICalculator


class DivergenceEvaluator(Evaluator, key='divergence'):
    """Evaluate RSI divergence (regular and hidden) against price.

    Confidence values
    -----------------
    Regular divergence (reversal signal):  0.60
    Hidden divergence  (continuation):     0.75

    Hidden divergence scores higher because it confirms an existing trend
    rather than predicting a reversal, making it statistically more reliable
    in trending markets such as XAU/USD.

    Parameters
    ----------
    rsi_period:
        Lookback period for RSI calculation. Default: 14.
    lookback:
        Half-window for swing-point detection in :class:`DivergenceDetector`.
        Default: 5.
    min_bars_between:
        Minimum bar separation between two swing points. Default: 5.
    max_bars_between:
        Maximum bar separation between two swing points. Default: 50.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        lookback: int = 5,
        min_bars_between: int = 5,
        max_bars_between: int = 50,
    ) -> None:
        self._rsi = RSICalculator(period=rsi_period)
        self._div = DivergenceDetector(
            lookback=lookback,
            min_bars_between=min_bars_between,
            max_bars_between=max_bars_between,
        )

    def evaluate(self, ohlcv: pd.DataFrame) -> EvaluatorResult:
        """Detect divergence on the most recent bar.

        Parameters
        ----------
        ohlcv:
            OHLCV DataFrame with at minimum ``high``, ``low``, ``close``
            columns.

        Returns
        -------
        EvaluatorResult
            direction  = +1.0 (bullish) / -1.0 (bearish) / 0.0 (none)
            confidence = 0.60 (regular) / 0.75 (hidden) / 0.0 (none)
            metadata keys:
                divergence_type      — 'regular' | 'hidden' | None
                divergence_direction — 'bullish' | 'bearish' | None
                rsi_value            — float RSI at the last bar, or None
        """
        h = ohlcv['high'].values
        l = ohlcv['low'].values
        c = ohlcv['close'].values

        rsi_result = self._rsi.calculate(c)
        div = self._div.detect(c, rsi_result.rsi, h, l)

        direction: float = 0.0
        confidence: float = 0.0
        div_type: str | None = None
        div_dir: str | None = None

        last = len(c) - 1
        if last >= 0:
            if div.regular_bullish[last]:
                direction, confidence = 1.0, 0.6
                div_type, div_dir = 'regular', 'bullish'
            elif div.regular_bearish[last]:
                direction, confidence = -1.0, 0.6
                div_type, div_dir = 'regular', 'bearish'
            elif div.hidden_bullish[last]:
                direction, confidence = 1.0, 0.75
                div_type, div_dir = 'hidden', 'bullish'
            elif div.hidden_bearish[last]:
                direction, confidence = -1.0, 0.75
                div_type, div_dir = 'hidden', 'bearish'

        rsi_value: float | None = None
        if last >= 0 and last < len(rsi_result.rsi):
            raw = rsi_result.rsi[last]
            import math
            if not math.isnan(raw):
                rsi_value = float(raw)

        return EvaluatorResult(
            direction=direction,
            confidence=confidence,
            metadata={
                'divergence_type': div_type,
                'divergence_direction': div_dir,
                'rsi_value': rsi_value,
            },
        )
