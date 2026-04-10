"""Tests for the InstrumentSizer protocol + ForexLotSizer + FuturesContractSizer (plan Task 5)."""

from __future__ import annotations

import pytest

from src.config.profile import InstrumentClass
from src.risk.instrument_sizer import (
    ForexLotSizer,
    FuturesContractSizer,
    InstrumentSizer,
    sizer_for_instrument,
)


# ---------------------------------------------------------------------------
# ForexLotSizer
# ---------------------------------------------------------------------------


class TestForexLotSizer:
    def _make(self) -> ForexLotSizer:
        # XAU/USD spot: pip_size 0.01, $1/pip/lot (gold on The5ers)
        return ForexLotSizer(pip_size=0.01, pip_value_per_lot=1.0)

    def test_50_dollar_risk_50_pip_stop_returns_one_lot(self) -> None:
        sizer = self._make()
        # 50 pip stop on $50 risk → 1.0 lot
        assert sizer.size_for_risk(risk_usd=50.0, stop_distance_price=0.50) == pytest.approx(1.0)

    def test_double_risk_doubles_lot(self) -> None:
        sizer = self._make()
        assert sizer.size_for_risk(risk_usd=100.0, stop_distance_price=0.50) == pytest.approx(2.0)

    def test_half_stop_doubles_lot(self) -> None:
        sizer = self._make()
        assert sizer.size_for_risk(risk_usd=50.0, stop_distance_price=0.25) == pytest.approx(2.0)

    def test_zero_risk_returns_min_lot(self) -> None:
        sizer = self._make()
        assert sizer.size_for_risk(risk_usd=0.0, stop_distance_price=0.50) == sizer.min_lot_size

    def test_clamp_to_max_lot(self) -> None:
        sizer = ForexLotSizer(pip_size=0.01, pip_value_per_lot=1.0, max_lot_size=5.0)
        # $500 risk, 1 pip stop → raw 500 lots → clamp to 5
        assert sizer.size_for_risk(risk_usd=500.0, stop_distance_price=0.01) == 5.0

    def test_implements_protocol(self) -> None:
        sizer = self._make()
        assert isinstance(sizer, InstrumentSizer)


# ---------------------------------------------------------------------------
# FuturesContractSizer — plan's Test cases in Step 5.6
# ---------------------------------------------------------------------------


class TestFuturesContractSizer:
    def _mgc(self) -> FuturesContractSizer:
        return FuturesContractSizer(tick_size=0.10, tick_value_usd=1.0, max_contracts=50)

    def test_50_dollar_risk_5_dollar_stop_returns_1_contract(self) -> None:
        sizer = self._mgc()
        # $5 stop = 50 ticks = $50/contract. $50 risk → 1 contract.
        assert sizer.size_for_risk(risk_usd=50.0, stop_distance_price=5.0) == 1

    def test_500_dollar_risk_5_dollar_stop_returns_10_contracts(self) -> None:
        sizer = self._mgc()
        assert sizer.size_for_risk(risk_usd=500.0, stop_distance_price=5.0) == 10

    def test_50_dollar_risk_50_dollar_stop_returns_0_contracts(self) -> None:
        sizer = self._mgc()
        # $50 stop = 500 ticks = $500/contract. $50 risk buys nothing.
        # Plan spec: return 0 so the engine skips the trade with a clean rejection.
        assert sizer.size_for_risk(risk_usd=50.0, stop_distance_price=50.0) == 0

    def test_cap_at_max_contracts(self) -> None:
        sizer = self._mgc()
        # $500 risk, $1 stop = 10 ticks = $10/contract → raw 50 contracts
        # (which exactly hits the max). Try $1000 to overshoot.
        assert sizer.size_for_risk(risk_usd=1_000.0, stop_distance_price=1.0) == 50

    def test_rounds_down_not_up(self) -> None:
        sizer = self._mgc()
        # $149 risk, $5 stop → $50/contract → raw 2.98 → 2 contracts (round down)
        assert sizer.size_for_risk(risk_usd=149.0, stop_distance_price=5.0) == 2

    def test_zero_risk_returns_zero(self) -> None:
        sizer = self._mgc()
        assert sizer.size_for_risk(risk_usd=0.0, stop_distance_price=5.0) == 0

    def test_negative_stop_returns_zero(self) -> None:
        sizer = self._mgc()
        assert sizer.size_for_risk(risk_usd=500.0, stop_distance_price=-1.0) == 0

    def test_min_contracts_enforced(self) -> None:
        # A contract cost above risk budget → skip (return 0)
        sizer = FuturesContractSizer(tick_size=0.10, tick_value_usd=1.0, min_contracts=1)
        # $5 risk, $10 stop = $100/contract → 0.05 contracts → below min → 0
        assert sizer.size_for_risk(risk_usd=5.0, stop_distance_price=10.0) == 0


# ---------------------------------------------------------------------------
# sizer_for_instrument factory
# ---------------------------------------------------------------------------


class _FakeInst:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestSizerForInstrument:
    def test_futures_instrument_returns_contract_sizer(self) -> None:
        inst = _FakeInst(
            class_=InstrumentClass.FUTURES,
            tick_size=0.10,
            tick_value_usd=1.0,
        )
        sizer = sizer_for_instrument(inst)
        assert isinstance(sizer, FuturesContractSizer)
        assert sizer.tick_size == 0.10
        assert sizer.tick_value_usd == 1.0

    def test_forex_instrument_returns_lot_sizer(self) -> None:
        inst = _FakeInst(
            class_=InstrumentClass.FOREX,
            pip_size=0.0001,
            pip_value_per_lot=10.0,
        )
        sizer = sizer_for_instrument(inst)
        assert isinstance(sizer, ForexLotSizer)
        assert sizer.pip_size == 0.0001
        assert sizer.pip_value_per_lot == 10.0

    def test_defaults_to_forex_when_class_missing(self) -> None:
        inst = _FakeInst(pip_value_usd=1.0, tick_size=0.01)
        sizer = sizer_for_instrument(inst)
        assert isinstance(sizer, ForexLotSizer)

    def test_futures_falls_back_to_tick_value_when_tick_value_usd_missing(self) -> None:
        inst = _FakeInst(
            class_=InstrumentClass.FUTURES,
            tick_size=0.25,
            tick_value=12.50,  # legacy field
        )
        sizer = sizer_for_instrument(inst)
        assert isinstance(sizer, FuturesContractSizer)
        assert sizer.tick_value_usd == 12.50

    def test_string_class_accepted(self) -> None:
        inst = _FakeInst(class_="futures", tick_size=0.10, tick_value_usd=1.0)
        assert isinstance(sizer_for_instrument(inst), FuturesContractSizer)


# ---------------------------------------------------------------------------
# Real project config round-trip
# ---------------------------------------------------------------------------


class TestFactoryWithRealConfig:
    def test_xauusd_from_config_returns_futures_sizer(self) -> None:
        from src.config.loader import load_config

        cfg = load_config()
        inst = cfg.instruments.get("XAUUSD")
        assert inst is not None
        sizer = sizer_for_instrument(inst)
        assert isinstance(sizer, FuturesContractSizer)
        assert sizer.tick_size == pytest.approx(0.10)
        assert sizer.tick_value_usd == pytest.approx(1.0)
