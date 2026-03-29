"""
Tests for StrategyHealthMonitor — 35+ unit and integration tests.

Test categories:
1.  Initialization — starts in NORMAL, sub-systems created
2.  Pre-flight — calls PreFlightDiagnostic, trains RegimeDetector, handles abort
3.  NORMAL → DROUGHT_DETECTED — when is_drought returns True
4.  DROUGHT_DETECTED → RELAXING — when relaxer applies a tier
5.  RELAXING → MONITORING_RELAXED — when signal found under relaxed config
6.  MONITORING_RELAXED → NORMAL — when relaxer auto-tightens to tier 0
7.  Any state → HALTED — budget exhaustion or strategy_broken
8.  HALTED state — on_bar does nothing, is_halted returns True
9.  on_trade_closed — records trades, updates win rate, feeds relaxer
10. Auto-tighten flow — 3 consecutive losses trigger tighten, transitions back
11. Diagnosis — strategy_broken triggers HALTED
12. get_status — returns HealthStatus with all fields populated
13. Pre-flight abort — failed pre-flight transitions to HALTED
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from unittest.mock import MagicMock, patch, PropertyMock
import pytest
import pandas as pd
import numpy as np

from src.monitoring.health_monitor import (
    StrategyHealthMonitor,
    HealthState,
    HealthStatus,
)
from src.monitoring.adaptive_relaxer import RelaxationState
from src.monitoring.pre_flight import PreFlightResult
from src.monitoring.regime_detector import DiagnosisResult, MarketRegime, RegimeState


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_candles(n: int = 200) -> pd.DataFrame:
    """Create a minimal 1-minute candle dataframe for tests."""
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    price = 2000.0 + np.cumsum(np.random.default_rng(42).normal(0, 0.5, n))
    df = pd.DataFrame(
        {
            "open": price,
            "high": price + 0.3,
            "low": price - 0.3,
            "close": price,
            "volume": 100.0,
        },
        index=idx,
    )
    return df


def _make_signal() -> MagicMock:
    """Create a minimal mock Signal object."""
    sig = MagicMock()
    sig.direction = "long"
    sig.entry_price = 2000.0
    return sig


class MockFunnelTracker:
    """Controllable mock for SignalFunnelTracker."""

    def __init__(self, is_drought_val: bool = False):
        self.is_drought_val = is_drought_val
        self.recorded_bars: list[bool] = []
        self.recorded_filters: list[tuple] = []

    def record_filter(self, bar_idx: int, name: str, passed: bool) -> None:
        self.recorded_filters.append((bar_idx, name, passed))

    def record_bar_complete(self, bar_idx: int, signal_generated: bool) -> None:
        self.recorded_bars.append(signal_generated)

    def is_drought(self, window: int = 500) -> bool:
        return self.is_drought_val

    def bottleneck_filter(self) -> Optional[str]:
        return "5m_entry"

    def rolling_signal_rate(self, window: int = 500) -> float:
        return 0.01

    def get_funnel_report(self) -> dict:
        return {"total_bars": len(self.recorded_bars)}


class MockRelaxerState:
    """Mutable state holder for MockAdaptiveRelaxer."""

    def __init__(self):
        self.current_tier = 0
        self.budget_used = 0.0
        self.bars_since_last_relax = 0
        self.consecutive_losses = 0
        self.base_config: dict = {}
        self.is_halted = False


class MockAdaptiveRelaxer:
    """Controllable mock for AdaptiveRelaxer."""

    def __init__(
        self,
        relax_returns: bool = True,
        budget_exhausted: bool = False,
        initial_tier: int = 0,
    ):
        self._relax_returns = relax_returns
        self._state = MockRelaxerState()
        self._state.is_halted = budget_exhausted
        self._state.current_tier = initial_tier
        self._on_bar_called = 0
        self._on_trade_closed_calls: list[bool] = []

    def relax_next_tier(self) -> bool:
        if self._state.is_halted:
            return False
        if self._relax_returns:
            self._state.current_tier += 1
            self._state.budget_used += 0.05
        return self._relax_returns

    def tighten_all(self) -> None:
        self._state.current_tier = 0
        self._state.budget_used = 0.0
        self._state.consecutive_losses = 0

    def tighten_one_tier(self) -> None:
        if self._state.current_tier > 0:
            self._state.current_tier -= 1

    def on_trade_closed(self, won: bool) -> None:
        self._on_trade_closed_calls.append(won)
        if not won:
            self._state.consecutive_losses += 1
            if self._state.consecutive_losses >= 3 and self._state.current_tier > 0:
                # Auto-tighten back to 0 on 3+ consecutive losses
                self._state.current_tier = 0
                self._state.consecutive_losses = 0
        else:
            self._state.consecutive_losses = 0

    def on_bar(self) -> None:
        self._on_bar_called += 1
        self._state.bars_since_last_relax += 1

    def get_state(self) -> RelaxationState:
        s = RelaxationState(
            current_tier=self._state.current_tier,
            budget_used=self._state.budget_used,
            bars_since_last_relax=self._state.bars_since_last_relax,
            consecutive_losses=self._state.consecutive_losses,
            base_config=self._state.base_config,
            is_halted=self._state.is_halted,
        )
        return s

    def is_budget_exhausted(self) -> bool:
        return self._state.is_halted


class MockRegimeDetector:
    """Controllable mock for RegimeDetector."""

    def __init__(
        self,
        diagnosis: DiagnosisResult = DiagnosisResult.NORMAL,
        current_regime: Optional[MarketRegime] = MarketRegime.SIDEWAYS,
        fit_raises: bool = False,
    ):
        self._diagnosis = diagnosis
        self._current_regime = current_regime
        self._fit_raises = fit_raises
        self.fit_called = False

    def fit(self, candles_1m: pd.DataFrame) -> None:
        self.fit_called = True
        if self._fit_raises:
            raise RuntimeError("HMM fit failed")

    def update(self, recent_data: pd.DataFrame) -> RegimeState:
        return RegimeState(
            regime=self._current_regime or MarketRegime.SIDEWAYS,
            confidence=0.75,
            changed=False,
        )

    def current_regime(self) -> Optional[MarketRegime]:
        return self._current_regime

    def regime_changed(self) -> bool:
        return False

    def diagnose(self, rolling_win_rate: float, baseline_win_rate: float) -> DiagnosisResult:
        return self._diagnosis

    def get_state(self) -> dict:
        return {
            "is_fitted": True,
            "current_regime": self._current_regime.value if self._current_regime else None,
        }


def _make_monitor(
    funnel: Optional[MockFunnelTracker] = None,
    relaxer: Optional[MockAdaptiveRelaxer] = None,
    regime: Optional[MockRegimeDetector] = None,
    drought_window: int = 500,
    baseline_win_rate: float = 0.55,
) -> StrategyHealthMonitor:
    """Create a StrategyHealthMonitor with mocked sub-systems.

    Bypasses real sub-system construction entirely by patching all three
    internal classes so the constructor never calls real EdgeManager methods.
    """
    signal_engine = MagicMock()
    edge_manager = MagicMock()

    _funnel = funnel or MockFunnelTracker()
    _relaxer = relaxer or MockAdaptiveRelaxer()
    _regime = regime or MockRegimeDetector()

    with patch("src.monitoring.health_monitor.SignalFunnelTracker", return_value=_funnel), \
         patch("src.monitoring.health_monitor.AdaptiveRelaxer", return_value=_relaxer), \
         patch("src.monitoring.health_monitor.RegimeDetector", return_value=_regime):
        monitor = StrategyHealthMonitor(
            signal_engine=signal_engine,
            edge_manager=edge_manager,
            drought_window=drought_window,
            baseline_win_rate=baseline_win_rate,
        )

    return monitor


# ---------------------------------------------------------------------------
# 1. Initialization tests
# ---------------------------------------------------------------------------


class TestInitialization:
    def test_starts_in_normal_state(self):
        monitor = _make_monitor()
        assert monitor.state == HealthState.NORMAL

    def test_is_halted_false_initially(self):
        monitor = _make_monitor()
        assert monitor.is_halted is False

    def test_bar_count_starts_at_zero(self):
        monitor = _make_monitor()
        assert monitor._bar_count == 0

    def test_trades_closed_starts_at_zero(self):
        monitor = _make_monitor()
        assert monitor._trades_closed == 0

    def test_trade_results_empty_initially(self):
        monitor = _make_monitor()
        assert monitor._trade_results == []

    def test_transitions_empty_initially(self):
        monitor = _make_monitor()
        assert monitor.transitions == []

    def test_sub_systems_created(self):
        """Sub-systems are wired up (non-None) after construction."""
        monitor = _make_monitor()
        assert monitor._funnel is not None
        assert monitor._relaxer is not None
        assert monitor._regime is not None

    def test_custom_drought_window(self):
        monitor = _make_monitor(drought_window=200)
        assert monitor._drought_window == 200

    def test_custom_baseline_win_rate(self):
        monitor = _make_monitor(baseline_win_rate=0.60)
        assert monitor._baseline_win_rate == 0.60


# ---------------------------------------------------------------------------
# 2. Pre-flight tests
# ---------------------------------------------------------------------------


class TestPreFlight:
    def _passing_result(self) -> PreFlightResult:
        return PreFlightResult(
            passed=True,
            signals_found=5,
            sample_size=1000,
            relaxation_applied=False,
            relaxation_tier=0,
            funnel_report={},
            aborted=False,
            message="ok",
        )

    def _aborting_result(self) -> PreFlightResult:
        return PreFlightResult(
            passed=False,
            signals_found=0,
            sample_size=1000,
            relaxation_applied=True,
            relaxation_tier=5,
            funnel_report={},
            aborted=True,
            message="aborted",
        )

    def test_pre_flight_passes_transitions_to_normal(self):
        monitor = _make_monitor(regime=MockRegimeDetector())
        candles = _make_candles(300)
        with patch("src.monitoring.health_monitor.PreFlightDiagnostic") as mock_pfd:
            mock_pfd.return_value.run.return_value = self._passing_result()
            result = monitor.pre_flight(candles)
        assert result.passed is True
        assert monitor.state == HealthState.NORMAL

    def test_pre_flight_aborted_transitions_to_halted(self):
        monitor = _make_monitor(regime=MockRegimeDetector())
        candles = _make_candles(300)
        with patch("src.monitoring.health_monitor.PreFlightDiagnostic") as mock_pfd:
            mock_pfd.return_value.run.return_value = self._aborting_result()
            result = monitor.pre_flight(candles)
        assert result.aborted is True
        assert monitor.state == HealthState.HALTED
        assert monitor.is_halted is True

    def test_pre_flight_trains_regime_detector(self):
        regime_mock = MockRegimeDetector()
        monitor = _make_monitor(regime=regime_mock)
        candles = _make_candles(300)
        with patch("src.monitoring.health_monitor.PreFlightDiagnostic") as mock_pfd:
            mock_pfd.return_value.run.return_value = self._passing_result()
            monitor.pre_flight(candles)
        assert regime_mock.fit_called is True

    def test_pre_flight_regime_fit_failure_non_fatal(self):
        """If RegimeDetector.fit() raises, pre-flight still succeeds."""
        regime_mock = MockRegimeDetector(fit_raises=True)
        monitor = _make_monitor(regime=regime_mock)
        candles = _make_candles(300)
        with patch("src.monitoring.health_monitor.PreFlightDiagnostic") as mock_pfd:
            mock_pfd.return_value.run.return_value = self._passing_result()
            result = monitor.pre_flight(candles)
        assert result.passed is True
        assert monitor.state == HealthState.NORMAL

    def test_pre_flight_returns_pre_flight_result(self):
        monitor = _make_monitor(regime=MockRegimeDetector())
        candles = _make_candles(300)
        expected = self._passing_result()
        with patch("src.monitoring.health_monitor.PreFlightDiagnostic") as mock_pfd:
            mock_pfd.return_value.run.return_value = expected
            result = monitor.pre_flight(candles)
        assert result is expected


# ---------------------------------------------------------------------------
# 3. NORMAL → DROUGHT_DETECTED
# ---------------------------------------------------------------------------


class TestNormalToDrought:
    def test_drought_triggers_state_change(self):
        funnel = MockFunnelTracker(is_drought_val=True)
        monitor = _make_monitor(funnel=funnel)
        monitor.on_bar(0)
        assert monitor.state == HealthState.DROUGHT_DETECTED

    def test_no_drought_stays_normal(self):
        funnel = MockFunnelTracker(is_drought_val=False)
        monitor = _make_monitor(funnel=funnel)
        for i in range(10):
            monitor.on_bar(i)
        assert monitor.state == HealthState.NORMAL

    def test_transition_recorded(self):
        funnel = MockFunnelTracker(is_drought_val=True)
        monitor = _make_monitor(funnel=funnel)
        monitor.on_bar(0)
        assert len(monitor.transitions) == 1
        t = monitor.transitions[0]
        assert t["from"] == "normal"
        assert t["to"] == "drought_detected"

    def test_bar_count_increments(self):
        funnel = MockFunnelTracker(is_drought_val=False)
        monitor = _make_monitor(funnel=funnel)
        monitor.on_bar(0)
        monitor.on_bar(1)
        monitor.on_bar(2)
        assert monitor._bar_count == 3

    def test_relaxer_on_bar_called(self):
        relaxer = MockAdaptiveRelaxer()
        monitor = _make_monitor(relaxer=relaxer)
        monitor.on_bar(0)
        assert relaxer._on_bar_called == 1


# ---------------------------------------------------------------------------
# 4. DROUGHT_DETECTED → RELAXING
# ---------------------------------------------------------------------------


class TestDroughtToRelaxing:
    def _drought_monitor(self) -> StrategyHealthMonitor:
        monitor = _make_monitor()
        monitor._state = HealthState.DROUGHT_DETECTED
        return monitor

    def test_relax_applied_transitions_to_relaxing(self):
        relaxer = MockAdaptiveRelaxer(relax_returns=True)
        monitor = self._drought_monitor()
        monitor._relaxer = relaxer
        monitor.on_bar(0)
        assert monitor.state == HealthState.RELAXING

    def test_relax_blocked_budget_exhausted_transitions_to_halted(self):
        relaxer = MockAdaptiveRelaxer(relax_returns=False, budget_exhausted=True)
        monitor = self._drought_monitor()
        monitor._relaxer = relaxer
        monitor.on_bar(0)
        assert monitor.state == HealthState.HALTED

    def test_relax_blocked_no_budget_issue_stays_in_drought(self):
        """Cooldown active: relax blocked but budget not exhausted — stay in DROUGHT."""
        relaxer = MockAdaptiveRelaxer(relax_returns=False, budget_exhausted=False)
        monitor = self._drought_monitor()
        monitor._relaxer = relaxer
        monitor.on_bar(0)
        assert monitor.state == HealthState.DROUGHT_DETECTED

    def test_relaxing_tier_recorded_in_transition(self):
        relaxer = MockAdaptiveRelaxer(relax_returns=True)
        monitor = self._drought_monitor()
        monitor._relaxer = relaxer
        monitor.on_bar(0)
        last_t = monitor.transitions[-1]
        assert last_t["to"] == "relaxing"
        assert "tier" in last_t["reason"]


# ---------------------------------------------------------------------------
# 5. RELAXING → MONITORING_RELAXED
# ---------------------------------------------------------------------------


class TestRelaxingToMonitoringRelaxed:
    def _relaxing_monitor(self) -> StrategyHealthMonitor:
        monitor = _make_monitor()
        monitor._state = HealthState.RELAXING
        return monitor

    def test_signal_found_transitions_to_monitoring_relaxed(self):
        funnel = MockFunnelTracker(is_drought_val=False)
        monitor = self._relaxing_monitor()
        monitor._funnel = funnel
        sig = _make_signal()
        monitor.on_bar(1, signal=sig)
        assert monitor.state == HealthState.MONITORING_RELAXED

    def test_no_signal_no_drought_stays_relaxing(self):
        funnel = MockFunnelTracker(is_drought_val=False)
        monitor = self._relaxing_monitor()
        monitor._funnel = funnel
        monitor.on_bar(1, signal=None)
        assert monitor.state == HealthState.RELAXING

    def test_no_signal_still_drought_tries_next_tier(self):
        funnel = MockFunnelTracker(is_drought_val=True)
        relaxer = MockAdaptiveRelaxer(relax_returns=True)
        monitor = self._relaxing_monitor()
        monitor._funnel = funnel
        monitor._relaxer = relaxer
        monitor.on_bar(1, signal=None)
        # Relaxed a tier — stays in RELAXING (no signal yet)
        assert monitor.state == HealthState.RELAXING

    def test_no_signal_drought_budget_exhausted_transitions_to_halted(self):
        funnel = MockFunnelTracker(is_drought_val=True)
        relaxer = MockAdaptiveRelaxer(relax_returns=False, budget_exhausted=True)
        monitor = self._relaxing_monitor()
        monitor._funnel = funnel
        monitor._relaxer = relaxer
        monitor.on_bar(1, signal=None)
        assert monitor.state == HealthState.HALTED


# ---------------------------------------------------------------------------
# 6. MONITORING_RELAXED → NORMAL
# ---------------------------------------------------------------------------


class TestMonitoringRelaxedToNormal:
    def _monitoring_relaxed_monitor(
        self, relaxer: Optional[MockAdaptiveRelaxer] = None
    ) -> StrategyHealthMonitor:
        monitor = _make_monitor()
        monitor._state = HealthState.MONITORING_RELAXED
        if relaxer:
            monitor._relaxer = relaxer
        return monitor

    def test_tier_0_relaxer_transitions_to_normal(self):
        relaxer = MockAdaptiveRelaxer(initial_tier=0)
        monitor = self._monitoring_relaxed_monitor(relaxer=relaxer)
        monitor.on_bar(5)
        assert monitor.state == HealthState.NORMAL

    def test_tier_nonzero_stays_in_monitoring_relaxed(self):
        relaxer = MockAdaptiveRelaxer(initial_tier=2)
        monitor = self._monitoring_relaxed_monitor(relaxer=relaxer)
        monitor.on_bar(5)
        assert monitor.state == HealthState.MONITORING_RELAXED

    def test_auto_tighten_via_on_trade_closed(self):
        """3 consecutive losses auto-tighten the relaxer to tier 0."""
        relaxer = MockAdaptiveRelaxer(initial_tier=2)
        monitor = self._monitoring_relaxed_monitor(relaxer=relaxer)
        # 3 losses should auto-tighten
        monitor.on_trade_closed(won=False)
        monitor.on_trade_closed(won=False)
        monitor.on_trade_closed(won=False)
        assert monitor.state == HealthState.NORMAL


# ---------------------------------------------------------------------------
# 7. Any state → HALTED
# ---------------------------------------------------------------------------


class TestTransitionToHalted:
    def test_budget_exhausted_in_drought_detected(self):
        relaxer = MockAdaptiveRelaxer(relax_returns=False, budget_exhausted=True)
        monitor = _make_monitor(relaxer=relaxer)
        monitor._state = HealthState.DROUGHT_DETECTED
        monitor.on_bar(0)
        assert monitor.state == HealthState.HALTED

    def test_budget_exhausted_in_relaxing_with_drought(self):
        funnel = MockFunnelTracker(is_drought_val=True)
        relaxer = MockAdaptiveRelaxer(relax_returns=False, budget_exhausted=True)
        monitor = _make_monitor(funnel=funnel, relaxer=relaxer)
        monitor._state = HealthState.RELAXING
        monitor.on_bar(0)
        assert monitor.state == HealthState.HALTED

    def test_strategy_broken_diagnosis_triggers_halt(self):
        regime = MockRegimeDetector(diagnosis=DiagnosisResult.STRATEGY_BROKEN)
        monitor = _make_monitor(regime=regime)
        # Feed 10+ trade losses to trigger diagnosis
        for _ in range(10):
            monitor._trade_results.append(False)
        monitor._run_diagnosis()
        assert monitor.state == HealthState.HALTED

    def test_halted_transition_recorded(self):
        relaxer = MockAdaptiveRelaxer(relax_returns=False, budget_exhausted=True)
        monitor = _make_monitor(relaxer=relaxer)
        monitor._state = HealthState.DROUGHT_DETECTED
        monitor.on_bar(0)
        last_t = monitor.transitions[-1]
        assert last_t["to"] == "halted"


# ---------------------------------------------------------------------------
# 8. HALTED state
# ---------------------------------------------------------------------------


class TestHaltedState:
    def test_is_halted_property(self):
        monitor = _make_monitor()
        monitor._state = HealthState.HALTED
        assert monitor.is_halted is True

    def test_on_bar_does_nothing_when_halted(self):
        funnel = MockFunnelTracker(is_drought_val=True)
        relaxer = MockAdaptiveRelaxer(relax_returns=True)
        monitor = _make_monitor(funnel=funnel, relaxer=relaxer)
        monitor._state = HealthState.HALTED
        transitions_before = len(monitor.transitions)
        monitor.on_bar(0)
        # No new state transitions (bar count and funnel recording still happen)
        assert monitor.state == HealthState.HALTED
        assert len(monitor.transitions) == transitions_before

    def test_halted_state_is_normal_false(self):
        monitor = _make_monitor()
        assert monitor.is_halted is False
        monitor._state = HealthState.HALTED
        assert monitor.is_halted is True

    def test_diagnosis_skipped_when_halted(self):
        """_run_diagnosis should not change state if already HALTED."""
        regime = MockRegimeDetector(diagnosis=DiagnosisResult.STRATEGY_BROKEN)
        monitor = _make_monitor(regime=regime)
        monitor._state = HealthState.HALTED
        for _ in range(15):
            monitor._trade_results.append(False)
        monitor._run_diagnosis()
        # Still halted, no additional transitions
        assert monitor.state == HealthState.HALTED

    def test_on_bar_increments_bar_count_even_when_halted(self):
        monitor = _make_monitor()
        monitor._state = HealthState.HALTED
        monitor.on_bar(0)
        assert monitor._bar_count == 1


# ---------------------------------------------------------------------------
# 9. on_trade_closed tests
# ---------------------------------------------------------------------------


class TestOnTradeClosed:
    def test_trades_closed_increments(self):
        monitor = _make_monitor()
        monitor.on_trade_closed(won=True)
        assert monitor._trades_closed == 1
        monitor.on_trade_closed(won=False)
        assert monitor._trades_closed == 2

    def test_win_results_recorded(self):
        monitor = _make_monitor()
        monitor.on_trade_closed(won=True)
        monitor.on_trade_closed(won=False)
        assert monitor._trade_results == [True, False]

    def test_rolling_window_capped(self):
        monitor = _make_monitor()
        monitor._win_rate_window = 5
        for _ in range(10):
            monitor.on_trade_closed(won=True)
        assert len(monitor._trade_results) == 5

    def test_win_rate_computed_correctly(self):
        monitor = _make_monitor()
        monitor.on_trade_closed(won=True)
        monitor.on_trade_closed(won=True)
        monitor.on_trade_closed(won=False)
        assert abs(monitor._rolling_win_rate() - 2 / 3) < 1e-9

    def test_on_trade_closed_feeds_relaxer(self):
        relaxer = MockAdaptiveRelaxer()
        monitor = _make_monitor(relaxer=relaxer)
        monitor.on_trade_closed(won=True)
        assert relaxer._on_trade_closed_calls == [True]
        monitor.on_trade_closed(won=False)
        assert relaxer._on_trade_closed_calls == [True, False]

    def test_empty_trade_results_win_rate_zero(self):
        monitor = _make_monitor()
        assert monitor._rolling_win_rate() == 0.0


# ---------------------------------------------------------------------------
# 10. Auto-tighten flow
# ---------------------------------------------------------------------------


class TestAutoTightenFlow:
    def test_three_consecutive_losses_from_monitoring_relaxed(self):
        """3 consecutive losses should tighten the relaxer and transition back."""
        relaxer = MockAdaptiveRelaxer(initial_tier=2)
        monitor = _make_monitor(relaxer=relaxer)
        monitor._state = HealthState.MONITORING_RELAXED
        monitor.on_trade_closed(won=False)
        monitor.on_trade_closed(won=False)
        monitor.on_trade_closed(won=False)
        # After 3 losses the MockAdaptiveRelaxer auto-tightens to tier 0
        assert relaxer._state.current_tier == 0
        assert monitor.state == HealthState.NORMAL

    def test_win_resets_consecutive_loss_count(self):
        relaxer = MockAdaptiveRelaxer(initial_tier=2)
        monitor = _make_monitor(relaxer=relaxer)
        monitor._state = HealthState.MONITORING_RELAXED
        monitor.on_trade_closed(won=False)
        monitor.on_trade_closed(won=False)
        monitor.on_trade_closed(won=True)   # win resets counter
        monitor.on_trade_closed(won=False)
        # Consecutive count reset — still in monitoring relaxed (only 1 loss after reset)
        assert relaxer._state.current_tier == 2  # not auto-tightened
        assert monitor.state == HealthState.MONITORING_RELAXED

    def test_diagnosis_not_called_with_fewer_than_10_trades(self):
        """Diagnosis should only run after 10+ trade results."""
        regime = MockRegimeDetector(diagnosis=DiagnosisResult.STRATEGY_BROKEN)
        monitor = _make_monitor(regime=regime)
        for _ in range(9):
            monitor.on_trade_closed(won=False)
        # Fewer than 10 trades — no diagnosis, no halt
        assert monitor.state != HealthState.HALTED

    def test_diagnosis_runs_at_10_trades(self):
        """At 10 trades, diagnosis is triggered."""
        regime = MockRegimeDetector(diagnosis=DiagnosisResult.STRATEGY_BROKEN)
        monitor = _make_monitor(regime=regime)
        for _ in range(10):
            monitor.on_trade_closed(won=False)
        assert monitor.state == HealthState.HALTED


# ---------------------------------------------------------------------------
# 11. Diagnosis tests
# ---------------------------------------------------------------------------


class TestDiagnosis:
    def test_normal_diagnosis_no_halt(self):
        regime = MockRegimeDetector(diagnosis=DiagnosisResult.NORMAL)
        monitor = _make_monitor(regime=regime)
        monitor._trade_results = [True] * 15
        monitor._run_diagnosis()
        assert monitor.state == HealthState.NORMAL

    def test_regime_shift_no_halt(self):
        regime = MockRegimeDetector(diagnosis=DiagnosisResult.REGIME_SHIFT)
        monitor = _make_monitor(regime=regime)
        monitor._trade_results = [True] * 10 + [False] * 5
        monitor._run_diagnosis()
        assert monitor.state == HealthState.NORMAL

    def test_regime_shift_severe_no_halt(self):
        regime = MockRegimeDetector(diagnosis=DiagnosisResult.REGIME_SHIFT_SEVERE)
        monitor = _make_monitor(regime=regime)
        monitor._trade_results = [False] * 15
        monitor._run_diagnosis()
        assert monitor.state == HealthState.NORMAL

    def test_strategy_broken_halts(self):
        regime = MockRegimeDetector(diagnosis=DiagnosisResult.STRATEGY_BROKEN)
        monitor = _make_monitor(regime=regime)
        monitor._trade_results = [False] * 15
        monitor._run_diagnosis()
        assert monitor.state == HealthState.HALTED
        assert monitor.is_halted is True

    def test_diagnosis_exception_is_non_fatal(self):
        """If diagnose() raises, we should not crash."""
        regime = MockRegimeDetector()
        regime.diagnose = MagicMock(side_effect=ValueError("bad baseline"))
        monitor = _make_monitor(regime=regime)
        monitor._trade_results = [False] * 15
        monitor._run_diagnosis()
        assert monitor.state == HealthState.NORMAL

    def test_last_diagnosis_stored(self):
        regime = MockRegimeDetector(diagnosis=DiagnosisResult.REGIME_SHIFT)
        monitor = _make_monitor(regime=regime)
        monitor._trade_results = [True] * 12
        monitor._run_diagnosis()
        assert monitor._last_diagnosis == DiagnosisResult.REGIME_SHIFT.value


# ---------------------------------------------------------------------------
# 12. get_status tests
# ---------------------------------------------------------------------------


class TestGetStatus:
    def test_get_status_returns_health_status(self):
        monitor = _make_monitor()
        status = monitor.get_status()
        assert isinstance(status, HealthStatus)

    def test_status_state_matches(self):
        monitor = _make_monitor()
        monitor._state = HealthState.RELAXING
        status = monitor.get_status()
        assert status.state == HealthState.RELAXING

    def test_status_bars_processed(self):
        funnel = MockFunnelTracker(is_drought_val=False)
        monitor = _make_monitor(funnel=funnel)
        monitor.on_bar(0)
        monitor.on_bar(1)
        status = monitor.get_status()
        assert status.bars_processed == 2

    def test_status_trades_closed(self):
        monitor = _make_monitor()
        monitor.on_trade_closed(won=True)
        monitor.on_trade_closed(won=False)
        status = monitor.get_status()
        assert status.trades_closed == 2

    def test_status_relaxation_tier(self):
        relaxer = MockAdaptiveRelaxer(initial_tier=3)
        monitor = _make_monitor(relaxer=relaxer)
        status = monitor.get_status()
        assert status.relaxation_tier == 3

    def test_status_regime_none_when_not_fitted(self):
        regime = MockRegimeDetector(current_regime=None)
        monitor = _make_monitor(regime=regime)
        status = monitor.get_status()
        assert status.regime is None

    def test_status_regime_value_string(self):
        regime = MockRegimeDetector(current_regime=MarketRegime.BULL)
        monitor = _make_monitor(regime=regime)
        status = monitor.get_status()
        assert status.regime == "bull"

    def test_status_rolling_win_rate(self):
        monitor = _make_monitor()
        monitor._trade_results = [True, True, False]  # 2/3 = 0.667
        status = monitor.get_status()
        assert abs(status.rolling_win_rate - 2 / 3) < 1e-9

    def test_status_baseline_win_rate(self):
        monitor = _make_monitor(baseline_win_rate=0.60)
        status = monitor.get_status()
        assert status.baseline_win_rate == 0.60

    def test_status_is_drought(self):
        funnel = MockFunnelTracker(is_drought_val=True)
        monitor = _make_monitor(funnel=funnel)
        status = monitor.get_status()
        assert status.is_drought is True

    def test_status_bottleneck_filter(self):
        funnel = MockFunnelTracker()
        monitor = _make_monitor(funnel=funnel)
        status = monitor.get_status()
        assert status.bottleneck_filter == "5m_entry"

    def test_status_message_not_empty(self):
        monitor = _make_monitor()
        status = monitor.get_status()
        assert len(status.message) > 0

    def test_status_message_changes_with_state(self):
        monitor = _make_monitor()
        normal_msg = monitor.get_status().message
        monitor._state = HealthState.HALTED
        halted_msg = monitor.get_status().message
        assert normal_msg != halted_msg


# ---------------------------------------------------------------------------
# 13. Pre-flight abort → HALTED
# ---------------------------------------------------------------------------


class TestPreFlightAbort:
    def test_abort_sets_halted_state(self):
        monitor = _make_monitor(regime=MockRegimeDetector())
        candles = _make_candles(300)
        abort_result = PreFlightResult(
            passed=False,
            signals_found=0,
            sample_size=500,
            relaxation_applied=True,
            relaxation_tier=5,
            funnel_report={},
            aborted=True,
            message="aborted: no signals",
        )
        with patch("src.monitoring.health_monitor.PreFlightDiagnostic") as mock_pfd:
            mock_pfd.return_value.run.return_value = abort_result
            monitor.pre_flight(candles)
        assert monitor.state == HealthState.HALTED
        assert monitor.is_halted is True

    def test_abort_transition_in_log(self):
        monitor = _make_monitor(regime=MockRegimeDetector())
        candles = _make_candles(300)
        abort_result = PreFlightResult(
            passed=False,
            signals_found=0,
            sample_size=500,
            relaxation_applied=True,
            relaxation_tier=5,
            funnel_report={},
            aborted=True,
            message="aborted",
        )
        with patch("src.monitoring.health_monitor.PreFlightDiagnostic") as mock_pfd:
            mock_pfd.return_value.run.return_value = abort_result
            monitor.pre_flight(candles)
        assert len(monitor.transitions) == 1
        assert monitor.transitions[0]["to"] == "halted"
        assert "pre-flight" in monitor.transitions[0]["reason"]


# ---------------------------------------------------------------------------
# 14. TIGHTENING state tests
# ---------------------------------------------------------------------------


class TestTighteningState:
    def test_tightening_to_tier_0_transitions_normal(self):
        relaxer = MockAdaptiveRelaxer(initial_tier=0)
        monitor = _make_monitor(relaxer=relaxer)
        monitor._state = HealthState.TIGHTENING
        monitor.on_bar(5)
        assert monitor.state == HealthState.NORMAL

    def test_tightening_partial_tighten_goes_to_monitoring_relaxed(self):
        relaxer = MockAdaptiveRelaxer(initial_tier=2)
        monitor = _make_monitor(relaxer=relaxer)
        monitor._state = HealthState.TIGHTENING
        monitor.on_bar(5)
        assert monitor.state == HealthState.MONITORING_RELAXED


# ---------------------------------------------------------------------------
# 15. Integration-style test with real EdgeManager
# ---------------------------------------------------------------------------


class TestIntegrationWithRealEdgeManager:
    """Lightweight integration test using the real EdgeManager from config."""

    def _make_real_monitor(self) -> StrategyHealthMonitor:
        import yaml
        with open(
            "C:/Users/kkudu/Documents/Code/itchy-tradebot/.claude/worktrees/agent-acbf879c/config/edges.yaml"
        ) as f:
            edge_cfg = yaml.safe_load(f)
        from src.edges.manager import EdgeManager
        em = EdgeManager(edge_cfg)
        signal_engine = MagicMock()
        return StrategyHealthMonitor(
            signal_engine=signal_engine,
            edge_manager=em,
            drought_window=500,
        )

    def test_real_monitor_starts_normal(self):
        monitor = self._make_real_monitor()
        assert monitor.state == HealthState.NORMAL

    def test_real_monitor_on_bar_no_crash(self):
        monitor = self._make_real_monitor()
        for i in range(5):
            monitor.on_bar(i)
        assert monitor._bar_count == 5

    def test_real_monitor_get_status_no_crash(self):
        monitor = self._make_real_monitor()
        status = monitor.get_status()
        assert isinstance(status, HealthStatus)
        assert status.state == HealthState.NORMAL


# ---------------------------------------------------------------------------
# 16. Transitions property
# ---------------------------------------------------------------------------


class TestTransitions:
    def test_transitions_list_returned(self):
        monitor = _make_monitor()
        assert isinstance(monitor.transitions, list)

    def test_transitions_list_is_copy(self):
        """transitions returns a copy — mutating it should not affect internals."""
        monitor = _make_monitor()
        t = monitor.transitions
        t.append({"fake": True})
        assert len(monitor.transitions) == 0

    def test_transition_fields(self):
        funnel = MockFunnelTracker(is_drought_val=True)
        monitor = _make_monitor(funnel=funnel)
        monitor.on_bar(42)
        t = monitor.transitions[0]
        assert "from" in t
        assert "to" in t
        assert "bar_idx" in t
        assert "reason" in t

    def test_multiple_transitions_ordered(self):
        """Walk through NORMAL → DROUGHT → RELAXING and check order."""
        funnel = MockFunnelTracker(is_drought_val=True)
        relaxer = MockAdaptiveRelaxer(relax_returns=True)
        monitor = _make_monitor(funnel=funnel, relaxer=relaxer)
        monitor.on_bar(0)   # → DROUGHT_DETECTED
        monitor.on_bar(1)   # → RELAXING
        assert monitor.transitions[0]["to"] == "drought_detected"
        assert monitor.transitions[1]["to"] == "relaxing"
