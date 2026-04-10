"""End-to-end futures position sizing test (plan Task 5).

Wires ``AdaptivePositionSizer`` with a ``FuturesContractSizer`` and
verifies the integrated risk→contract conversion produces sane
contract counts at several account sizes and risk percentages.
"""

from __future__ import annotations

import pytest

from src.risk.instrument_sizer import FuturesContractSizer
from src.risk.position_sizer import AdaptivePositionSizer


MGC = dict(tick_size=0.10, tick_value_usd=1.0, max_contracts=50)


def _mk_sizer(balance: float, risk_pct: float = 1.0) -> AdaptivePositionSizer:
    return AdaptivePositionSizer(
        initial_balance=balance,
        initial_risk_pct=risk_pct,
        reduced_risk_pct=risk_pct * 0.5,
        phase_threshold_pct=4.0,
        instrument_sizer=FuturesContractSizer(**MGC),
    )


class TestEndToEndFuturesSizing:
    def test_50k_account_1pct_risk_5dollar_stop(self) -> None:
        """$50K × 1% = $500 risk. $5 stop = $50/contract → 10 contracts."""
        sizer = _mk_sizer(50_000.0, risk_pct=1.0)
        result = sizer.calculate_position_size(
            account_equity=50_000.0,
            atr=5.0,
            atr_multiplier=1.0,
            point_value=1.0,  # unused when instrument_sizer is set
        )
        assert result.lot_size == 10
        assert result.risk_amount == pytest.approx(500.0)

    def test_50k_account_small_risk_tiny_stop_capped_at_max(self) -> None:
        """$50K × 2% = $1000 risk. $1 stop = $10/contract → 100 → cap at 50."""
        sizer = _mk_sizer(50_000.0, risk_pct=2.0)
        result = sizer.calculate_position_size(
            account_equity=50_000.0,
            atr=1.0,
            atr_multiplier=1.0,
            point_value=1.0,
        )
        assert result.lot_size == 50  # max_contracts cap

    def test_small_account_tiny_risk_returns_zero(self) -> None:
        """$10K × 0.5% = $50 risk. $50 stop = $500/contract → 0.1 → 0."""
        sizer = _mk_sizer(10_000.0, risk_pct=0.5)
        result = sizer.calculate_position_size(
            account_equity=10_000.0,
            atr=50.0,
            atr_multiplier=1.0,
            point_value=1.0,
        )
        assert result.lot_size == 0

    def test_phase_switch_halves_risk(self) -> None:
        """Hitting +4% profit flips to reduced_risk_pct."""
        sizer = _mk_sizer(50_000.0, risk_pct=1.0)
        # Aggressive: $500 risk → 10 contracts at $5 stop
        result1 = sizer.calculate_position_size(
            account_equity=50_000.0, atr=5.0, atr_multiplier=1.0, point_value=1.0,
        )
        assert result1.phase == "aggressive"
        assert result1.lot_size == 10

        # Push balance past the 4% threshold → protective phase
        sizer.update_balance(52_500.0)  # +5%
        result2 = sizer.calculate_position_size(
            account_equity=52_500.0, atr=5.0, atr_multiplier=1.0, point_value=1.0,
        )
        assert result2.phase == "protective"
        # 0.5% of 52500 = 262.5 risk → $50/contract → 5 contracts (round down)
        assert result2.lot_size == 5

    def test_forex_path_unchanged_when_no_instrument_sizer(self) -> None:
        """Backward compat: no instrument_sizer means legacy forex formula."""
        sizer = AdaptivePositionSizer(
            initial_balance=10_000.0,
            initial_risk_pct=1.5,
            reduced_risk_pct=0.75,
            # no instrument_sizer passed
        )
        result = sizer.calculate_position_size(
            account_equity=10_000.0,
            atr=0.5,
            atr_multiplier=1.5,
            point_value=1.0,
        )
        # risk_amount = 150, stop_distance = 0.75, raw_lot = 150/0.75 = 200
        # Clamped to max_lot = 10.0
        assert result.lot_size == 10.0
