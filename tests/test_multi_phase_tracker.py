"""Tests for MultiPhasePropFirmTracker - The5ers 2-Step High Stakes."""
import datetime as dt
from src.backtesting.metrics import MultiPhasePropFirmTracker


def _utc(year, month, day, hour=0):
    return dt.datetime(year, month, day, hour, tzinfo=dt.timezone.utc)


class TestPhase1:
    def test_starts_in_phase_1(self):
        tracker = MultiPhasePropFirmTracker()
        tracker.initialise(10_000.0, _utc(2024, 1, 1))
        status = tracker.get_status()
        assert status["phase"] == "phase_1_active"
        assert status["profit_pct"] == 0.0

    def test_phase_1_passes_at_8_percent(self):
        tracker = MultiPhasePropFirmTracker()
        tracker.initialise(10_000.0, _utc(2024, 1, 1))
        tracker.update(_utc(2024, 1, 15), 10_800.0)
        status = tracker.get_status()
        assert status["phase"] == "phase_2_active"
        assert status["phase_1_passed"] is True

    def test_phase_1_fails_on_total_dd(self):
        tracker = MultiPhasePropFirmTracker()
        tracker.initialise(10_000.0, _utc(2024, 1, 1))
        tracker.update(_utc(2024, 1, 2), 8_900.0)  # -11% total DD
        status = tracker.get_status()
        assert status["phase"] == "failed_phase_1"
        assert "total_dd" in status["failure_reason"]

    def test_phase_1_fails_on_daily_dd(self):
        tracker = MultiPhasePropFirmTracker(phase_1_daily_loss_pct=5.0)
        tracker.initialise(10_000.0, _utc(2024, 1, 1))
        tracker.update(_utc(2024, 1, 1, 23), 10_000.0)
        tracker.update(_utc(2024, 1, 2, 12), 9_400.0)
        status = tracker.get_status()
        assert status["phase"] == "failed_phase_1"
        assert "daily_dd" in status["failure_reason"]


class TestPhase2:
    def test_phase_2_resets_balance_to_10k(self):
        tracker = MultiPhasePropFirmTracker()
        tracker.initialise(10_000.0, _utc(2024, 1, 1))
        tracker.update(_utc(2024, 1, 15), 10_800.0)
        status = tracker.get_status()
        assert status["phase"] == "phase_2_active"
        assert status["phase_balance"] == 10_000.0

    def test_phase_2_passes_at_5_percent(self):
        tracker = MultiPhasePropFirmTracker()
        tracker.initialise(10_000.0, _utc(2024, 1, 1))
        tracker.update(_utc(2024, 1, 15), 10_800.0)
        tracker.update(_utc(2024, 2, 1), 10_500.0)
        status = tracker.get_status()
        assert status["phase"] == "funded_active"

    def test_phase_2_fails_on_total_dd(self):
        tracker = MultiPhasePropFirmTracker()
        tracker.initialise(10_000.0, _utc(2024, 1, 1))
        tracker.update(_utc(2024, 1, 15), 10_800.0)
        tracker.update(_utc(2024, 2, 1), 8_900.0)
        status = tracker.get_status()
        assert status["phase"] == "failed_phase_2"


class TestFunded:
    def test_funded_tracks_monthly_returns(self):
        tracker = MultiPhasePropFirmTracker()
        tracker.initialise(10_000.0, _utc(2024, 1, 1))
        tracker.update(_utc(2024, 1, 10), 10_800.0)  # Phase 1 pass
        tracker.update(_utc(2024, 1, 20), 10_500.0)  # Phase 2 pass
        # Funded months
        tracker.update(_utc(2024, 2, 28), 11_000.0)
        tracker.update(_utc(2024, 3, 31), 11_500.0)
        tracker.update(_utc(2024, 4, 30), 12_200.0)
        status = tracker.get_status()
        assert status["phase"] == "funded_active"
        assert len(status["funded_monthly_returns"]) == 3
        assert status["funded_monthly_returns"][0] > 9.0  # ~10%


class TestDailyDDFormula:
    def test_daily_dd_uses_max_of_equity_and_balance(self):
        """The5ers: daily_dd_limit = 5% * MAX(prev_close_equity, prev_close_balance)"""
        tracker = MultiPhasePropFirmTracker(phase_1_daily_loss_pct=5.0)
        tracker.initialise(10_000.0, _utc(2024, 1, 1))
        tracker.update(_utc(2024, 1, 1, 21), 10_300.0)
        tracker.update(_utc(2024, 1, 2, 12), 9_750.0)
        status = tracker.get_status()
        assert status["phase"] == "failed_phase_1"


class TestPhaseTransitionHardReset:
    def test_dd_tracking_resets_between_phases(self):
        tracker = MultiPhasePropFirmTracker()
        tracker.initialise(10_000.0, _utc(2024, 1, 1))
        tracker.update(_utc(2024, 1, 5), 9_500.0)
        tracker.update(_utc(2024, 1, 15), 10_800.0)
        status = tracker.get_status()
        assert status["max_total_dd_pct"] == 0.0
        assert status["phase"] == "phase_2_active"
