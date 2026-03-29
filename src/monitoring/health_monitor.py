"""
StrategyHealthMonitor — top-level orchestrator for all health monitoring sub-systems.

Composes SignalFunnelTracker, AdaptiveRelaxer, PreFlightDiagnostic, and
RegimeDetector into a single class that implements the health state machine.

State Machine
-------------
NORMAL → DROUGHT_DETECTED → RELAXING → MONITORING_RELAXED → TIGHTENING → NORMAL
                                                            ↓
Any state → HALTED (on budget exhaustion or strategy_broken diagnosis)

Called on every 5-minute bar by the backtester or live engine.  Identical code
path for both modes; 'live' mode may use tighter parameters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd

from src.edges.manager import EdgeManager
from src.strategy.signal_engine import Signal, SignalEngine
from src.monitoring.funnel_tracker import SignalFunnelTracker
from src.monitoring.adaptive_relaxer import AdaptiveRelaxer
from src.monitoring.pre_flight import PreFlightDiagnostic, PreFlightResult
from src.monitoring.regime_detector import RegimeDetector, DiagnosisResult, MarketRegime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State machine enumerations
# ---------------------------------------------------------------------------


class HealthState(Enum):
    """State machine states for the StrategyHealthMonitor."""

    NORMAL = "normal"
    DROUGHT_DETECTED = "drought_detected"
    RELAXING = "relaxing"
    MONITORING_RELAXED = "monitoring_relaxed"
    TIGHTENING = "tightening"
    HALTED = "halted"


# ---------------------------------------------------------------------------
# Status dataclass (dashboard / reporting)
# ---------------------------------------------------------------------------


@dataclass
class HealthStatus:
    """Full health report for dashboard integration.

    Attributes
    ----------
    state:
        Current state machine state.
    funnel_report:
        Per-filter pass/fail/rate snapshot from SignalFunnelTracker.
    relaxation_tier:
        Current relaxation tier (0 = no relaxation applied).
    relaxation_budget_used:
        Cumulative budget consumed (0.0–MAX_BUDGET fraction).
    regime:
        Current MarketRegime value string (e.g. 'bull'), or None.
    regime_confidence:
        Confidence for the current regime classification (0.0–1.0).
    diagnosis:
        Last DiagnosisResult value string, or None.
    rolling_win_rate:
        Win rate over the last N trades.
    baseline_win_rate:
        Expected win rate from optimization.
    is_drought:
        True if a signal drought is currently detected.
    bottleneck_filter:
        Name of the filter with the highest rejection rate, or None.
    bars_processed:
        Total bars processed since construction.
    trades_closed:
        Total trades closed since construction.
    message:
        Human-readable summary of the current state.
    """

    state: HealthState
    funnel_report: dict
    relaxation_tier: int
    relaxation_budget_used: float
    regime: Optional[str]           # MarketRegime.value or None
    regime_confidence: float
    diagnosis: Optional[str]        # DiagnosisResult.value or None
    rolling_win_rate: float
    baseline_win_rate: float
    is_drought: bool
    bottleneck_filter: Optional[str]
    bars_processed: int
    trades_closed: int
    message: str


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


class StrategyHealthMonitor:
    """Top-level orchestrator composing all health monitoring sub-systems.

    State Machine
    -------------
    NORMAL → DROUGHT_DETECTED → RELAXING → MONITORING_RELAXED → TIGHTENING → NORMAL
                                                                ↓
    Any state → HALTED (on budget exhaustion or strategy_broken diagnosis)

    The monitor is called on every bar by the backtester/live engine.
    Same code runs in both modes. Live mode uses tighter parameters.

    Parameters
    ----------
    signal_engine:
        The signal engine instance.
    edge_manager:
        The edge manager instance.
    config:
        Optional config dict (from health_monitor.yaml).
    mode:
        'backtest' or 'live'. Live uses tighter relaxation steps.
    drought_window:
        Number of bars to check for drought. Default 500.
    baseline_win_rate:
        Expected win rate from optimization. Default 0.55.
    """

    def __init__(
        self,
        signal_engine: SignalEngine,
        edge_manager: EdgeManager,
        config: dict | None = None,
        mode: str = "backtest",
        drought_window: int = 500,
        baseline_win_rate: float = 0.55,
    ) -> None:
        self._signal_engine = signal_engine
        self._edge_manager = edge_manager
        self._config = config or {}
        self._mode = mode
        self._drought_window = drought_window
        self._baseline_win_rate = baseline_win_rate

        # Sub-systems
        self._funnel = SignalFunnelTracker(window_size=max(2000, drought_window * 4))
        self._relaxer = AdaptiveRelaxer(edge_manager, config)
        self._regime = RegimeDetector(
            n_states=self._config.get("regime", {}).get("hmm_states", 3),
        )

        # State machine
        self._state = HealthState.NORMAL
        self._bar_count = 0
        self._trades_closed = 0

        # Win rate tracking
        self._win_rate_window: int = 20       # Track over last 20 trades
        self._trade_results: list[bool] = []  # True=win, False=loss

        # Latest regime confidence (from last update() call)
        self._regime_confidence: float = 0.0

        # Latest diagnosis string (for reporting)
        self._last_diagnosis: Optional[str] = None

        # State transition log
        self._transitions: list[dict] = []

    # ------------------------------------------------------------------
    # Pre-flight
    # ------------------------------------------------------------------

    def pre_flight(self, candles_1m: pd.DataFrame) -> PreFlightResult:
        """Run pre-flight diagnostic and train the regime detector.

        Call ONCE before the main loop starts.

        Steps:

        1. Run PreFlightDiagnostic (auto-relaxes if zero signals).
        2. Fit the RegimeDetector on the candle data.
        3. Return PreFlightResult.

        Parameters
        ----------
        candles_1m:
            Full 1-minute OHLCV dataframe used for scanning and HMM training.

        Returns
        -------
        PreFlightResult
        """
        diag = PreFlightDiagnostic(
            self._signal_engine,
            self._edge_manager,
            config=self._config,
        )
        result = diag.run(candles_1m)

        # Train regime detector — best-effort; failure is non-fatal
        try:
            self._regime.fit(candles_1m)
        except Exception as exc:
            logger.warning("RegimeDetector training failed: %s", exc)

        if result.aborted:
            self._transition_to(
                HealthState.HALTED,
                "pre-flight aborted: no signals found after exhausting all relaxation tiers",
            )

        return result

    # ------------------------------------------------------------------
    # Per-bar hook
    # ------------------------------------------------------------------

    def on_bar(self, bar_idx: int, signal: Optional[Signal] = None) -> None:
        """Called every 5M bar by the backtester/live engine.

        Parameters
        ----------
        bar_idx:
            Current bar index (monotonically increasing).
        signal:
            Signal generated this bar, or None if no signal was generated.
        """
        self._bar_count += 1
        self._relaxer.on_bar()

        # Record in funnel tracker
        passed = signal is not None
        self._funnel.record_filter(bar_idx, "full_pipeline", passed)
        self._funnel.record_bar_complete(bar_idx, signal_generated=passed)

        # Halted state: do nothing further
        if self._state == HealthState.HALTED:
            return

        # ---- NORMAL -------------------------------------------------------
        if self._state == HealthState.NORMAL:
            if self._funnel.is_drought(window=self._drought_window):
                self._transition_to(
                    HealthState.DROUGHT_DETECTED,
                    f"drought: 0 signals in last {self._drought_window} bars",
                )

        # ---- DROUGHT_DETECTED ---------------------------------------------
        elif self._state == HealthState.DROUGHT_DETECTED:
            applied = self._relaxer.relax_next_tier()
            if applied:
                self._transition_to(
                    HealthState.RELAXING,
                    f"relaxing tier {self._relaxer.get_state().current_tier}",
                )
            elif self._relaxer.is_budget_exhausted():
                self._transition_to(
                    HealthState.HALTED,
                    "budget exhausted during drought — no further relaxation possible",
                )

        # ---- RELAXING -----------------------------------------------------
        elif self._state == HealthState.RELAXING:
            if signal is not None:
                # Signals found under relaxed config
                self._transition_to(
                    HealthState.MONITORING_RELAXED,
                    "signals found under relaxed config",
                )
            elif self._funnel.is_drought(window=self._drought_window):
                # Still in drought — try the next tier
                applied = self._relaxer.relax_next_tier()
                if not applied and self._relaxer.is_budget_exhausted():
                    self._transition_to(
                        HealthState.HALTED,
                        "all tiers exhausted — still no signals",
                    )

        # ---- MONITORING_RELAXED -------------------------------------------
        elif self._state == HealthState.MONITORING_RELAXED:
            # Tightening is triggered by on_trade_closed() via AdaptiveRelaxer.
            # Check if the relaxer has auto-tightened back to base.
            state = self._relaxer.get_state()
            if state.current_tier == 0:
                self._transition_to(
                    HealthState.NORMAL,
                    "auto-tightened back to base config",
                )

        # ---- TIGHTENING ---------------------------------------------------
        elif self._state == HealthState.TIGHTENING:
            state = self._relaxer.get_state()
            if state.current_tier == 0:
                self._transition_to(HealthState.NORMAL, "tightening complete")
            else:
                self._transition_to(
                    HealthState.MONITORING_RELAXED,
                    f"partial tighten — still at tier {state.current_tier}",
                )

    # ------------------------------------------------------------------
    # Trade hooks
    # ------------------------------------------------------------------

    def on_trade_closed(self, won: bool, trade_data: dict | None = None) -> None:
        """Called when a trade closes.

        Records the result, updates the rolling win rate, informs the
        AdaptiveRelaxer (which may auto-tighten on consecutive losses), and
        runs the regime diagnosis once enough trades have accumulated.

        Parameters
        ----------
        won:
            True if the trade was profitable.
        trade_data:
            Optional dict with trade details (unused currently, reserved for
            future regime-aware analysis).
        """
        self._trades_closed += 1
        self._trade_results.append(won)

        # Keep only the last N trades for rolling win rate
        if len(self._trade_results) > self._win_rate_window:
            self._trade_results = self._trade_results[-self._win_rate_window:]

        # Inform relaxer — may auto-tighten on consecutive losses
        self._relaxer.on_trade_closed(won)

        # Check if relaxer auto-tightened back to tier 0 while we were in
        # MONITORING_RELAXED state
        state = self._relaxer.get_state()
        if self._state == HealthState.MONITORING_RELAXED and state.current_tier == 0:
            self._transition_to(
                HealthState.NORMAL,
                "auto-tightened after consecutive losses",
            )

        # Run diagnosis once sufficient trade history exists
        if len(self._trade_results) >= 10:
            self._run_diagnosis()

    # ------------------------------------------------------------------
    # Status reporting
    # ------------------------------------------------------------------

    def get_status(self) -> HealthStatus:
        """Return a full health report for dashboard integration.

        Returns
        -------
        HealthStatus
        """
        relaxer_state = self._relaxer.get_state()
        current_regime = self._regime.current_regime()

        return HealthStatus(
            state=self._state,
            funnel_report=self._funnel.get_funnel_report(),
            relaxation_tier=relaxer_state.current_tier,
            relaxation_budget_used=relaxer_state.budget_used,
            regime=current_regime.value if current_regime else None,
            regime_confidence=self._regime_confidence,
            diagnosis=self._last_diagnosis,
            rolling_win_rate=self._rolling_win_rate(),
            baseline_win_rate=self._baseline_win_rate,
            is_drought=self._funnel.is_drought(window=self._drought_window),
            bottleneck_filter=self._funnel.bottleneck_filter(),
            bars_processed=self._bar_count,
            trades_closed=self._trades_closed,
            message=self._state_message(),
        )

    @property
    def state(self) -> HealthState:
        """Current state machine state."""
        return self._state

    @property
    def is_halted(self) -> bool:
        """True when the monitor is in HALTED state."""
        return self._state == HealthState.HALTED

    @property
    def transitions(self) -> list[dict]:
        """List of state transition records (from, to, bar_idx, reason)."""
        return list(self._transitions)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rolling_win_rate(self) -> float:
        """Compute win rate over the last _win_rate_window trades."""
        if not self._trade_results:
            return 0.0
        return sum(1 for w in self._trade_results if w) / len(self._trade_results)

    def _run_diagnosis(self) -> None:
        """Run regime diagnosis and potentially transition to HALTED."""
        if self._state == HealthState.HALTED:
            return

        win_rate = self._rolling_win_rate()
        try:
            diagnosis = self._regime.diagnose(win_rate, self._baseline_win_rate)
        except (ValueError, Exception) as exc:
            logger.debug("Diagnosis failed: %s", exc)
            return

        self._last_diagnosis = diagnosis.value

        if diagnosis == DiagnosisResult.STRATEGY_BROKEN:
            self._transition_to(
                HealthState.HALTED,
                (
                    f"strategy broken: rolling win rate {win_rate:.1%} vs "
                    f"baseline {self._baseline_win_rate:.1%}"
                ),
            )

    def _transition_to(self, new_state: HealthState, reason: str) -> None:
        """Record a state transition and update the internal state."""
        old_state = self._state
        self._state = new_state
        self._transitions.append(
            {
                "from": old_state.value,
                "to": new_state.value,
                "bar_idx": self._bar_count,
                "reason": reason,
            }
        )
        logger.info(
            "HealthMonitor: %s → %s (%s)",
            old_state.value,
            new_state.value,
            reason,
        )

    def _state_message(self) -> str:
        """Human-readable message for the current state."""
        messages: dict[HealthState, str] = {
            HealthState.NORMAL: "Operating normally",
            HealthState.DROUGHT_DETECTED: "Signal drought detected — preparing relaxation",
            HealthState.RELAXING: (
                f"Relaxing filters (tier {self._relaxer.get_state().current_tier})"
            ),
            HealthState.MONITORING_RELAXED: "Monitoring trade quality under relaxed config",
            HealthState.TIGHTENING: "Tightening filters back toward base config",
            HealthState.HALTED: "HALTED — manual intervention required",
        }
        return messages.get(self._state, "Unknown state")
