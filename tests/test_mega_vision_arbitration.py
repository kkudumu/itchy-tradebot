"""Tests for the mega-vision Arbitrator (plan Task 25)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pytest

from src.mega_vision.arbitration import Arbitrator


@dataclass
class FakeSignal:
    strategy_name: str
    direction: str = "long"


def _native(strategies: List[str]) -> List[FakeSignal]:
    return [FakeSignal(strategy_name=s) for s in strategies]


class TestArbitrator:
    def test_disabled_mode_returns_native_unchanged(self):
        arb = Arbitrator(mode="disabled")
        native = _native(["sss", "ichimoku"])
        result = arb.arbitrate(None, native)
        assert [s.strategy_name for s in result] == ["sss", "ichimoku"]

    def test_shadow_mode_returns_native_regardless_of_pick(self):
        arb = Arbitrator(mode="shadow")
        native = _native(["sss", "ichimoku"])
        pick = {"strategy_picks": ["sss"], "fallback": False}
        result = arb.arbitrate(pick, native)
        # Shadow NEVER filters — full native list returned
        assert len(result) == 2

    def test_authority_mode_filters_by_picks(self):
        arb = Arbitrator(mode="authority")
        native = _native(["sss", "ichimoku", "ema_pullback"])
        pick = {"strategy_picks": ["sss", "ema_pullback"], "fallback": False}
        result = arb.arbitrate(pick, native)
        names = [s.strategy_name for s in result]
        assert names == ["sss", "ema_pullback"]

    def test_authority_with_empty_picks_returns_nothing(self):
        arb = Arbitrator(mode="authority")
        native = _native(["sss", "ichimoku"])
        pick = {"strategy_picks": [], "fallback": False}
        result = arb.arbitrate(pick, native)
        assert result == []

    def test_fallback_pick_returns_full_native_in_authority(self):
        arb = Arbitrator(mode="authority")
        native = _native(["sss", "ichimoku"])
        pick = {"strategy_picks": None, "fallback": True, "reasoning": "kill_switch"}
        result = arb.arbitrate(pick, native)
        assert len(result) == 2

    def test_none_pick_returns_native_in_shadow(self):
        arb = Arbitrator(mode="shadow")
        native = _native(["sss"])
        result = arb.arbitrate(None, native)
        assert len(result) == 1

    def test_mode_property(self):
        arb = Arbitrator(mode="authority")
        assert arb.mode == "authority"
