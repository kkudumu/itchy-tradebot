"""Tests for the discovery profile adapter (plan Task 19)."""

from __future__ import annotations

import pytest

from src.config.profile import InstrumentClass
from src.discovery.profile_adapter import (
    ProfileAdapter,
    adapt_codegen,
    adapt_objective,
    make_adapter,
)


class _FakeInst:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestMakeAdapter:
    def test_forex_instrument_returns_forex_profile(self) -> None:
        inst = _FakeInst(class_=InstrumentClass.FOREX, pip_size=0.01)
        adapter = make_adapter(inst)
        assert adapter.instrument_class == InstrumentClass.FOREX
        assert adapter.objective_name == "pass_rate"
        assert adapter.pip_size == 0.01

    def test_futures_instrument_returns_futures_profile(self) -> None:
        inst = _FakeInst(class_=InstrumentClass.FUTURES, tick_size=0.10, tick_value_usd=1.0)
        adapter = make_adapter(inst)
        assert adapter.instrument_class == InstrumentClass.FUTURES
        assert adapter.objective_name == "topstep_combine_pass_score"
        assert adapter.tick_size == 0.10
        assert adapter.tick_value_usd == 1.0

    def test_string_class_accepted(self) -> None:
        inst = _FakeInst(class_="futures", tick_size=0.25, tick_value_usd=12.50)
        adapter = make_adapter(inst)
        assert adapter.instrument_class == InstrumentClass.FUTURES

    def test_falls_back_to_forex_when_class_missing(self) -> None:
        inst = _FakeInst()
        adapter = make_adapter(inst)
        assert adapter.instrument_class == InstrumentClass.FOREX


class TestAdaptObjective:
    def test_futures_profile_returns_topstep_objective(self) -> None:
        adapter = ProfileAdapter(instrument_class=InstrumentClass.FUTURES)
        adapter.objective_name = "topstep_combine_pass_score"
        fn = adapt_objective(adapter)
        # Verify it's the actual topstep scorer by calling it on a
        # passed-run fixture
        result = {
            "prop_firm": {
                "active_tracker": {
                    "status": "passed",
                    "initial_balance": 50_000.0,
                    "current_balance": 53_000.0,
                    "profit_target_usd": 3_000.0,
                }
            }
        }
        assert fn(result) == 1.0

    def test_forex_profile_returns_base_objective(self) -> None:
        adapter = ProfileAdapter(instrument_class=InstrumentClass.FOREX)
        base = lambda r: 0.75  # noqa: E731
        fn = adapt_objective(adapter, base_objective=base)
        assert fn({}) == 0.75

    def test_no_base_objective_returns_zero_fn(self) -> None:
        adapter = ProfileAdapter(instrument_class=InstrumentClass.FOREX)
        fn = adapt_objective(adapter)
        assert fn({}) == 0.0


class TestAdaptCodegen:
    def test_forex_template_returned_verbatim(self) -> None:
        adapter = ProfileAdapter(instrument_class=InstrumentClass.FOREX)
        template = "pip_value = 0.0001\nmin_range_pips = 10\n"
        assert adapt_codegen(adapter, template) == template

    def test_futures_template_rewrites_pip_value(self) -> None:
        adapter = ProfileAdapter(
            instrument_class=InstrumentClass.FUTURES, tick_size=0.10
        )
        template = "pip_value = 0.0001\nmin_range_pips = 10\n"
        result = adapt_codegen(adapter, template)
        assert "pip_value = 0.1" in result

    def test_futures_without_tick_size_returns_verbatim(self) -> None:
        adapter = ProfileAdapter(
            instrument_class=InstrumentClass.FUTURES, tick_size=None
        )
        template = "pip_value = 0.0001\n"
        assert adapt_codegen(adapter, template) == template
