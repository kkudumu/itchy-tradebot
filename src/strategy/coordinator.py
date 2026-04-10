"""
EvaluatorCoordinator — resamples 1M data and feeds evaluators per-timeframe.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .base import (
    EvalMatrix,
    EvalRequirement,
    EvaluatorResult,
    EVALUATOR_REGISTRY,
)


# Timeframe resample rules (matching existing MTFAnalyzer convention)
_TF_RULES: dict[str, str] = {
    "1M":  "1min",
    "5M":  "5min",
    "15M": "15min",
    "1H":  "1h",
    "4H":  "4h",
    "D":   "1D",
}


def _resample_ohlcv(df_1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample 1-minute OHLCV to the requested frequency.

    Uses label='left', closed='left' convention (bar named by open time).
    """
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    available = {k: v for k, v in agg.items() if k in df_1m.columns}
    resampled = df_1m.resample(rule, label="left", closed="left").agg(available)
    return resampled.dropna(how="all")


class EvaluatorCoordinator:
    """Coordinate evaluator execution across multiple timeframes.

    Parameters
    ----------
    required_evaluators:
        List of EvalRequirement declaring which evaluators to run on which TFs.
    warmup_bars:
        Minimum number of 1M bars before the coordinator can produce results.
        This prevents signals during the warm-up period.
    """

    def __init__(
        self,
        required_evaluators: list[EvalRequirement],
        warmup_bars: int = 0,
    ) -> None:
        self._requirements = required_evaluators
        self._warmup_bars = warmup_bars

        # Instantiate evaluator instances from registry
        self._evaluators: dict[str, object] = {}
        for req in required_evaluators:
            if req.evaluator_name not in EVALUATOR_REGISTRY:
                raise ValueError(
                    f"Evaluator '{req.evaluator_name}' not found in registry. "
                    f"Available: {list(EVALUATOR_REGISTRY.keys())}"
                )
            if req.evaluator_name not in self._evaluators:
                cls = EVALUATOR_REGISTRY[req.evaluator_name]
                self._evaluators[req.evaluator_name] = cls()

        # Collect all required timeframes
        self._required_tfs: set[str] = set()
        for req in required_evaluators:
            for tf in req.timeframes:
                self._required_tfs.add(tf)

    def evaluate(
        self,
        data_1m: pd.DataFrame,
        current_bar: int = -1,
    ) -> Optional[EvalMatrix]:
        """Resample 1M data, feed evaluators, collect results.

        Parameters
        ----------
        data_1m:
            1-minute OHLCV DataFrame with DatetimeIndex (UTC).
        current_bar:
            Bar index to evaluate up to. -1 = all available data.
            When positive, slices data_1m[:current_bar+1] to prevent
            lookahead.

        Returns
        -------
        EvalMatrix or None
            None if warmup period not satisfied.
        """
        # Slice data up to current_bar to prevent lookahead
        if current_bar >= 0:
            data = data_1m.iloc[:current_bar + 1]
        else:
            data = data_1m

        # Check warmup
        if len(data) < self._warmup_bars:
            return None

        # Resample to each required timeframe
        tf_data: dict[str, pd.DataFrame] = {}
        for tf in self._required_tfs:
            if tf not in _TF_RULES:
                raise ValueError(f"Unknown timeframe: {tf}. Valid: {list(_TF_RULES.keys())}")
            resampled = _resample_ohlcv(data, _TF_RULES[tf])

            # CRITICAL: Shift by 1 bar to prevent lookahead
            # This means indicator values only become available on the NEXT bar
            # after the higher-TF bar closes
            resampled = resampled.shift(1)
            resampled = resampled.dropna(how="all")

            if len(resampled) < 2:
                return None  # Not enough data after shift

            tf_data[tf] = resampled

        # Feed each evaluator its required timeframes
        matrix = EvalMatrix()

        for req in self._requirements:
            evaluator = self._evaluators[req.evaluator_name]
            for tf in req.timeframes:
                if tf not in tf_data or len(tf_data[tf]) == 0:
                    continue

                result = evaluator.evaluate(tf_data[tf])
                key = f"{req.evaluator_name}_{tf}"
                matrix.set(key, result)

        return matrix
