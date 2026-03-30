"""Session evaluator — wraps SessionIdentifier to label active trading sessions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.sessions import SessionIdentifier
from src.strategy.base import Evaluator, EvaluatorResult


class SessionEvaluator(Evaluator, key='session'):
    """Evaluate the current trading session at the last bar.

    Session is non-directional, so direction is always 0.0.

    Confidence:
    - 1.0 if the last bar falls in an active trading session (London or NY)
    - 0.0 otherwise (Asian session or off-hours)
    """

    def __init__(self) -> None:
        self._ident = SessionIdentifier()

    def evaluate(self, ohlcv: pd.DataFrame) -> EvaluatorResult:
        """Identify the current trading session at the last bar.

        Parameters
        ----------
        ohlcv:
            OHLCV DataFrame indexed by timestamp (datetime-like index).
            The index is converted to numpy datetime64 for session detection.

        Returns
        -------
        EvaluatorResult
            direction=0.0, confidence 0.0 or 1.0, and session label metadata.
        """
        # Convert the DataFrame index to numpy datetime64[s] array
        timestamps = np.asarray(ohlcv.index, dtype='datetime64[s]')

        labels = self._ident.identify(timestamps)
        active_mask = self._ident.is_active_session(timestamps)

        idx = len(timestamps) - 1
        session_label = str(labels[idx])
        is_active = bool(active_mask[idx])

        confidence = 1.0 if is_active else 0.0

        return EvaluatorResult(
            direction=0.0,
            confidence=confidence,
            metadata={
                'session':   session_label,
                'is_active': is_active,
            },
        )
