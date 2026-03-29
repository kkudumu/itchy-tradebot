"""
3-state Hidden Markov Model for market regime classification.

Uses log returns and rolling volatility computed from 1-minute candle data
(resampled to 5-minute bars) as the observation features for a Gaussian HMM
with 3 hidden states: BULL, BEAR, and SIDEWAYS.

Key diagnostic capability
~~~~~~~~~~~~~~~~~~~~~~~~~~
``diagnose()`` cross-references regime change detection with strategy win-rate
to distinguish "the market changed" (adapt parameters) from "the strategy is
broken" (halt or fundamentally redesign).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional import — graceful fallback if hmmlearn is unavailable
# ---------------------------------------------------------------------------
try:
    from hmmlearn.hmm import GaussianHMM as _GaussianHMM

    _HMMLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HMMLEARN_AVAILABLE = False
    _GaussianHMM = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Public enums & dataclasses
# ---------------------------------------------------------------------------


class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"


class DiagnosisResult(Enum):
    NORMAL = "normal"
    REGIME_SHIFT = "regime_shift"
    REGIME_SHIFT_SEVERE = "regime_shift_severe"
    STRATEGY_BROKEN = "strategy_broken"


@dataclass
class RegimeState:
    """Snapshot of the current market regime as assessed by RegimeDetector."""

    regime: MarketRegime
    confidence: float  # 0.0 – 1.0
    changed: bool
    diagnosis: Optional[DiagnosisResult] = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_BARS_FOR_FEATURES = 20  # rolling vol window = 20; need exactly that many valid rows
_ROLLING_VOL_WINDOW = 20
_RESAMPLE_RULE = "5min"

# Thresholds for the simple fallback classifier (no HMM)
_BULL_RETURN_THRESHOLD = 0.0001   # > 0.01% mean 5M return
_BEAR_RETURN_THRESHOLD = -0.0001  # < -0.01% mean 5M return


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resample_to_5m(candles_1m: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-minute OHLCV data to 5-minute bars."""
    ohlcv = candles_1m.resample(_RESAMPLE_RULE).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return ohlcv.dropna(subset=["close"])


def _build_features(candles_5m: pd.DataFrame) -> np.ndarray:
    """Return (N, 2) feature array of [log_return, rolling_volatility].

    Rows with NaN are dropped; the result may be shorter than the input.
    """
    close = candles_5m["close"]
    log_ret = np.log(close / close.shift(1))
    rolling_vol = log_ret.rolling(_ROLLING_VOL_WINDOW).std()

    features = pd.DataFrame({"ret": log_ret, "vol": rolling_vol}).dropna()
    return features.values.astype(np.float64)


def _map_states_to_regimes(model: "_GaussianHMM") -> dict[int, MarketRegime]:  # type: ignore[type-arg]
    """Assign each HMM state to a MarketRegime based on mean return.

    State with the highest mean return → BULL
    State with the lowest mean return → BEAR
    Middle state → SIDEWAYS
    """
    # means_ shape: (n_components, n_features); column 0 is log return
    mean_returns = model.means_[:, 0]
    sorted_indices = np.argsort(mean_returns)  # ascending

    mapping: dict[int, MarketRegime] = {}
    mapping[int(sorted_indices[0])] = MarketRegime.BEAR
    mapping[int(sorted_indices[-1])] = MarketRegime.BULL
    # Middle state(s) → SIDEWAYS
    for idx in sorted_indices[1:-1]:
        mapping[int(idx)] = MarketRegime.SIDEWAYS
    # Edge case: only 2 states (shouldn't happen with n_states=3 but be safe)
    if len(sorted_indices) == 2:
        mapping[int(sorted_indices[1])] = MarketRegime.BULL
    return mapping


def _fallback_classify(features: np.ndarray) -> tuple[MarketRegime, float]:
    """Simple threshold-based classifier used when HMM is unavailable or fails."""
    mean_ret = float(np.mean(features[:, 0]))
    if mean_ret > _BULL_RETURN_THRESHOLD:
        return MarketRegime.BULL, 0.6
    if mean_ret < _BEAR_RETURN_THRESHOLD:
        return MarketRegime.BEAR, 0.6
    return MarketRegime.SIDEWAYS, 0.6


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class RegimeDetector:
    """3-state Gaussian HMM for market regime classification.

    Uses log returns and rolling volatility (from 5M-resampled 1M candles) as
    observation features.  After fitting, maps each hidden state to BULL /
    BEAR / SIDEWAYS by comparing mean returns.

    Key diagnostic
    ~~~~~~~~~~~~~~
    ``diagnose(rolling_win_rate, baseline_win_rate)`` cross-references regime
    change detection with strategy win-rate to distinguish:
    - Market changed (adapt parameters) → REGIME_SHIFT / REGIME_SHIFT_SEVERE
    - Strategy broken (halt) → STRATEGY_BROKEN
    - Everything fine → NORMAL
    """

    def __init__(self, n_states: int = 3, lookback: int = 100) -> None:
        if n_states < 2:
            raise ValueError("n_states must be >= 2")
        if lookback < _MIN_BARS_FOR_FEATURES:
            raise ValueError(f"lookback must be >= {_MIN_BARS_FOR_FEATURES}")

        self._n_states = n_states
        self._lookback = lookback
        self._model: Optional["_GaussianHMM"] = None  # type: ignore[type-arg]
        self._regime_map: dict[int, MarketRegime] = {}
        self._previous_regime: Optional[MarketRegime] = None
        self._current_regime: Optional[MarketRegime] = None
        self._current_confidence: float = 0.0
        self._is_fitted: bool = False
        self._using_fallback: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, candles_1m: pd.DataFrame) -> None:
        """Train the HMM on historical 1-minute candle data.

        Parameters
        ----------
        candles_1m:
            DataFrame with a DatetimeIndex and columns ``open``, ``high``,
            ``low``, ``close``, ``volume``.  Must have at least
            ``_MIN_BARS_FOR_FEATURES`` valid 5M bars after resampling.

        Raises
        ------
        ValueError
            If there is insufficient data to build features.
        """
        self._validate_candles(candles_1m)

        candles_5m = _resample_to_5m(candles_1m)
        features = _build_features(candles_5m)

        if len(features) < _MIN_BARS_FOR_FEATURES:
            raise ValueError(
                f"Insufficient data: need at least {_MIN_BARS_FOR_FEATURES} valid "
                f"5M bars with non-NaN features, got {len(features)}."
            )

        if not _HMMLEARN_AVAILABLE:
            # Mark as fitted in fallback mode; no model object
            self._using_fallback = True
            self._is_fitted = True
            return

        model = _GaussianHMM(
            n_components=self._n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model.fit(features)
            except Exception:
                # Fall back to threshold-based classifier
                self._using_fallback = True
                self._is_fitted = True
                return

        self._model = model
        self._regime_map = _map_states_to_regimes(model)
        self._using_fallback = False
        self._is_fitted = True

    def update(self, recent_data: pd.DataFrame) -> RegimeState:
        """Classify the current regime from recent candle data.

        Parameters
        ----------
        recent_data:
            Recent 1-minute candles (same format as ``fit``).  The last
            ``lookback`` rows are used.

        Returns
        -------
        RegimeState
            Current regime, confidence, and whether it changed.

        Raises
        ------
        RuntimeError
            If called before ``fit()``.
        """
        if not self._is_fitted:
            raise RuntimeError("RegimeDetector must be fitted before calling update().")

        self._validate_candles(recent_data)

        # Resample first, then cap at lookback 5M bars so the rolling-vol
        # window always has enough rows regardless of input granularity.
        candles_5m = _resample_to_5m(recent_data)
        candles_5m = candles_5m.iloc[-self._lookback :]
        features = _build_features(candles_5m)

        if len(features) < _MIN_BARS_FOR_FEATURES:
            # Reuse last known regime if possible
            regime = self._current_regime or MarketRegime.SIDEWAYS
            state = RegimeState(regime=regime, confidence=0.0, changed=False)
            return state

        # Classify
        if self._using_fallback or self._model is None:
            regime, confidence = _fallback_classify(features)
        else:
            try:
                regime, confidence = self._hmm_classify(features)
            except Exception:
                # Degenerate covariance or other numerical issue — use fallback
                regime, confidence = _fallback_classify(features)

        # Detect change
        self._previous_regime = self._current_regime
        self._current_regime = regime
        self._current_confidence = confidence
        changed = (
            self._previous_regime is not None
            and self._previous_regime != self._current_regime
        )

        return RegimeState(regime=regime, confidence=confidence, changed=changed)

    def current_regime(self) -> Optional[MarketRegime]:
        """Return the most recently classified regime, or None if not yet set."""
        return self._current_regime

    def regime_changed(self) -> bool:
        """Return True if the current regime differs from the previous one."""
        if self._previous_regime is None or self._current_regime is None:
            return False
        return self._previous_regime != self._current_regime

    def diagnose(
        self,
        rolling_win_rate: float,
        baseline_win_rate: float,
    ) -> DiagnosisResult:
        """Diagnose whether a performance drop is due to market change or strategy failure.

        Parameters
        ----------
        rolling_win_rate:
            Recent observed win rate (0.0 – 1.0).
        baseline_win_rate:
            Historical baseline win rate when strategy was performing well (0.0 – 1.0).

        Returns
        -------
        DiagnosisResult
            - NORMAL: no regime change and win rate ≥ 80 % of baseline
            - REGIME_SHIFT: regime changed and win rate ≥ 50 % of baseline
            - REGIME_SHIFT_SEVERE: regime changed and win rate < 50 % of baseline
            - STRATEGY_BROKEN: no regime change and win rate < 50 % of baseline
        """
        if baseline_win_rate <= 0:
            raise ValueError("baseline_win_rate must be > 0")

        changed = self.regime_changed()
        ratio = rolling_win_rate / baseline_win_rate

        if not changed:
            if ratio >= 0.80:
                return DiagnosisResult.NORMAL
            return DiagnosisResult.STRATEGY_BROKEN

        # Regime changed
        if ratio >= 0.50:
            return DiagnosisResult.REGIME_SHIFT
        return DiagnosisResult.REGIME_SHIFT_SEVERE

    def get_state(self) -> dict:
        """Return a full state dictionary suitable for dashboard/logging."""
        return {
            "is_fitted": self._is_fitted,
            "using_fallback": self._using_fallback,
            "n_states": self._n_states,
            "lookback": self._lookback,
            "current_regime": self._current_regime.value if self._current_regime else None,
            "previous_regime": self._previous_regime.value if self._previous_regime else None,
            "confidence": self._current_confidence,
            "regime_changed": self.regime_changed(),
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def using_fallback(self) -> bool:
        return self._using_fallback

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _hmm_classify(self, features: np.ndarray) -> tuple[MarketRegime, float]:
        """Run HMM prediction and return (regime, confidence)."""
        assert self._model is not None

        hidden_states = self._model.predict(features)
        last_state = int(hidden_states[-1])

        # Confidence from posterior probability of current state
        try:
            posteriors = self._model.predict_proba(features)
            confidence = float(posteriors[-1, last_state])
        except Exception:
            confidence = 0.5

        regime = self._regime_map.get(last_state, MarketRegime.SIDEWAYS)
        return regime, confidence

    @staticmethod
    def _validate_candles(df: pd.DataFrame) -> None:
        """Raise ValueError if the DataFrame lacks required columns or index."""
        if df is None or len(df) == 0:
            raise ValueError("candles DataFrame must be non-empty.")
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"candles DataFrame missing columns: {missing}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("candles DataFrame must have a DatetimeIndex.")
