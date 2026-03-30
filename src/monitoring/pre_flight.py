"""
PreFlightDiagnostic — pre-backtest/pre-live validation for the signal pipeline.

Samples bars from the dataset, runs the full signal engine cascade, and
reports per-filter pass/fail diagnostics.  If zero signals are found on the
initial scan the diagnostic automatically engages the AdaptiveRelaxer, tier
by tier, until signals appear or the budget is exhausted.

Usage
-----
>>> from src.monitoring.pre_flight import PreFlightDiagnostic
>>> result = PreFlightDiagnostic(signal_engine, edge_manager).run(candles_1m)
>>> if not result.passed:
...     print(result.message)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from src.edges.manager import EdgeManager
from src.monitoring.adaptive_relaxer import AdaptiveRelaxer
from src.monitoring.funnel_tracker import SignalFunnelTracker
from src.strategy.signal_engine import Signal, SignalEngine

logger = logging.getLogger(__name__)

# Minimum number of bars in a window before we run scan() — ensures indicators
# have time to warm up before we trust the output.
_WARMUP_BARS = 300


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PreFlightResult:
    """Result of a pre-flight diagnostic check.

    Attributes
    ----------
    passed:
        True if at least one signal was found (possibly after relaxation).
    signals_found:
        Number of signals generated in the sample.
    sample_size:
        Number of 1-minute bars in the sample that was scanned.
    relaxation_applied:
        True if the AdaptiveRelaxer was invoked (zero signals on first scan).
    relaxation_tier:
        The final relaxation tier reached.  0 = no relaxation applied.
    funnel_report:
        Per-filter pass/fail/rate data from the SignalFunnelTracker.
    aborted:
        True when all tiers were exhausted and still zero signals found.
    message:
        Human-readable summary of the diagnostic outcome.
    relaxed_config:
        Snapshot of the relaxed parameter values when auto-relaxation was
        applied and signals were found.  ``None`` when no relaxation was
        needed or when the diagnostic aborted.
    """

    passed: bool
    signals_found: int
    sample_size: int
    relaxation_applied: bool
    relaxation_tier: int
    funnel_report: dict
    aborted: bool
    message: str
    relaxed_config: Optional[dict] = None


# ---------------------------------------------------------------------------
# Main diagnostic class
# ---------------------------------------------------------------------------


class PreFlightDiagnostic:
    """Pre-backtest/pre-live validation for the signal pipeline.

    Samples a slice of the provided candles, runs ``signal_engine.scan()``
    across it while recording per-filter events in a
    :class:`~src.monitoring.funnel_tracker.SignalFunnelTracker`, and returns a
    :class:`PreFlightResult`.

    If zero signals are found on the initial scan the diagnostic engages the
    :class:`~src.monitoring.adaptive_relaxer.AdaptiveRelaxer`, relaxing one
    tier at a time and re-scanning after each step until signals appear or all
    tiers are exhausted.

    Parameters
    ----------
    signal_engine:
        The :class:`~src.strategy.signal_engine.SignalEngine` instance to test.
    edge_manager:
        The :class:`~src.edges.manager.EdgeManager` instance — passed to the
        :class:`~src.monitoring.adaptive_relaxer.AdaptiveRelaxer` when
        auto-relaxation is needed.
    sample_size:
        Number of 1-minute bars to use for the sample.  Defaults to 5 000.
        If the dataset is smaller the entire dataset is used.
    config:
        Optional config dict forwarded to :class:`AdaptiveRelaxer`.  Falls
        back to built-in defaults when ``None``.
    """

    def __init__(
        self,
        signal_engine: SignalEngine,
        edge_manager: EdgeManager,
        sample_size: int = 5000,
        config: dict | None = None,
    ) -> None:
        self._signal_engine = signal_engine
        self._edge_manager = edge_manager
        self._sample_size = sample_size
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, candles_1m: pd.DataFrame) -> PreFlightResult:
        """Run the pre-flight diagnostic against a 1-minute candle dataframe.

        Steps
        -----
        1. Extract a sample from the middle of the dataset (skips warmup data).
        2. Scan every 5-minute bar through the signal engine.
        3. Track pass/fail per bar via :class:`SignalFunnelTracker`.
        4. If zero signals found engage the :class:`AdaptiveRelaxer` tier by
           tier, re-scanning after each relaxation step.
        5. Return a :class:`PreFlightResult` capturing all findings.

        Parameters
        ----------
        candles_1m:
            Full 1-minute OHLCV dataframe (any length).

        Returns
        -------
        PreFlightResult
        """
        if candles_1m is None or len(candles_1m) == 0:
            return PreFlightResult(
                passed=False,
                signals_found=0,
                sample_size=0,
                relaxation_applied=False,
                relaxation_tier=0,
                funnel_report={},
                aborted=True,
                message="Aborted: empty candle dataframe supplied to PreFlightDiagnostic.",
            )

        sample = self._sample_data(candles_1m)
        tracker = SignalFunnelTracker(window_size=max(2000, len(sample) // 5 + 1))

        logger.info(
            "PreFlightDiagnostic: scanning %d bars (sampled from %d total).",
            len(sample),
            len(candles_1m),
        )

        signals = self._scan_sample(sample, tracker)
        funnel = tracker.get_funnel_report()

        # ------------------------------------------------------------------
        # Happy path: signals found on the first scan
        # ------------------------------------------------------------------
        if signals > 0:
            msg = (
                f"Pre-flight passed: {signals} signal(s) found in "
                f"{len(sample)}-bar sample.  No relaxation needed."
            )
            logger.info(msg)
            return PreFlightResult(
                passed=True,
                signals_found=signals,
                sample_size=len(sample),
                relaxation_applied=False,
                relaxation_tier=0,
                funnel_report=funnel,
                aborted=False,
                message=msg,
            )

        # ------------------------------------------------------------------
        # Zero signals — engage auto-relaxation loop
        # ------------------------------------------------------------------
        logger.warning(
            "PreFlightDiagnostic: zero signals in initial scan — starting auto-relaxation."
        )
        relaxed_signals, final_tier, aborted, relaxed_cfg = self._auto_relax_loop(sample, tracker)
        funnel = tracker.get_funnel_report()

        if not aborted and relaxed_signals > 0:
            msg = (
                f"Pre-flight passed after relaxation: {relaxed_signals} signal(s) found "
                f"at tier {final_tier} in {len(sample)}-bar sample."
            )
            logger.info(msg)
            return PreFlightResult(
                passed=True,
                signals_found=relaxed_signals,
                sample_size=len(sample),
                relaxation_applied=True,
                relaxation_tier=final_tier,
                funnel_report=funnel,
                aborted=False,
                message=msg,
                relaxed_config=relaxed_cfg,
            )

        # All tiers exhausted — abort
        bottleneck = funnel.get("bottleneck") or "unknown"
        msg = (
            f"Pre-flight ABORTED: zero signals after exhausting all relaxation tiers "
            f"(reached tier {final_tier}).  "
            f"Bottleneck filter: '{bottleneck}'.  "
            f"Check data quality and strategy parameters."
        )
        logger.error(msg)
        return PreFlightResult(
            passed=False,
            signals_found=0,
            sample_size=len(sample),
            relaxation_applied=True,
            relaxation_tier=final_tier,
            funnel_report=funnel,
            aborted=True,
            message=msg,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_data(self, candles_1m: pd.DataFrame) -> pd.DataFrame:
        """Extract a sample slice from the middle of the dataset.

        Starts at 25% into the dataset so that indicator lookback periods
        (typically 52–78 bars on higher timeframes) have already warmed up.
        If the dataset is smaller than ``sample_size`` the whole dataset is
        returned.

        Parameters
        ----------
        candles_1m:
            Full 1-minute candle dataframe.

        Returns
        -------
        pd.DataFrame
            Slice of at most ``self._sample_size`` bars.
        """
        total = len(candles_1m)
        if total <= self._sample_size:
            return candles_1m

        start_idx = total // 4
        end_idx = min(start_idx + self._sample_size, total)
        return candles_1m.iloc[start_idx:end_idx]

    def _scan_sample(
        self,
        sample: pd.DataFrame,
        tracker: SignalFunnelTracker,
    ) -> int:
        """Scan every 5-minute bar boundary through the signal engine.

        Because ``signal_engine.scan()`` returns ``Optional[Signal]`` rather
        than a detailed per-filter breakdown, we record a single synthetic
        ``"full_pipeline"`` filter event per bar (passed if a signal was
        generated, failed otherwise).  This is sufficient for drought
        detection and signal-rate diagnostics.

        Parameters
        ----------
        sample:
            1-minute candle slice to scan.
        tracker:
            :class:`SignalFunnelTracker` to record events into.

        Returns
        -------
        int
            Number of signal-generating bars found in the sample.
        """
        signal_count = 0
        step = 5  # advance one 5M bar at a time through the 1M data
        total_bars = len(sample)

        for bar_idx in range(0, total_bars, step):
            window = sample.iloc[: bar_idx + 1]
            if len(window) < _WARMUP_BARS:
                continue

            signal: Optional[Signal] = self._signal_engine.scan(
                data_1m=window, current_bar=-1
            )

            passed = signal is not None
            virtual_bar = bar_idx // step
            tracker.record_filter(virtual_bar, "full_pipeline", passed)
            tracker.record_bar_complete(virtual_bar, signal_generated=passed)

            if passed:
                signal_count += 1

        return signal_count

    def _auto_relax_loop(
        self,
        sample: pd.DataFrame,
        tracker: SignalFunnelTracker,
    ) -> tuple[int, int, bool, Optional[dict]]:
        """Relax one tier at a time and re-scan until signals appear.

        Parameters
        ----------
        sample:
            1-minute candle slice (same as used in the initial scan).
        tracker:
            :class:`SignalFunnelTracker` to reset and re-record into.

        Returns
        -------
        tuple[int, int, bool, Optional[dict]]
            ``(signals_found, final_tier, aborted, relaxed_config)``

            - ``signals_found``: count of signals after the last scan
            - ``final_tier``: relaxation tier reached (from
              ``relaxer.get_state().current_tier``)
            - ``aborted``: ``True`` if budget was exhausted with zero signals
            - ``relaxed_config``: snapshot of relaxed parameters, or ``None``
              if aborted
        """
        relaxer = AdaptiveRelaxer(self._edge_manager, self._config)

        while not relaxer.is_budget_exhausted():
            applied = relaxer.relax_next_tier()
            if not applied:
                # No more tiers available
                break

            tracker.reset()
            signals = self._scan_sample(sample, tracker)

            state = relaxer.get_state()
            logger.info(
                "PreFlightDiagnostic: after tier %d relaxation, found %d signal(s).",
                state.current_tier,
                signals,
            )

            if signals > 0:
                return signals, state.current_tier, False, relaxer.capture_current_config()

        state = relaxer.get_state()
        return 0, state.current_tier, True, None
