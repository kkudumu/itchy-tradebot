"""Tests for MegaStrategyAgent (plan Task 25)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.mega_vision.agent import MegaStrategyAgent, MegaVisionConfig


UTC = timezone.utc


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class _FakeCtx:
    def __init__(self):
        self.context_builder = MagicMock()
        self.context_builder.current_market_state.return_value = {}
        self.telemetry_collector = None
        self.trade_memory = None
        self.performance_buckets = None
        self.regime_detector = None
        self.screenshot_provider = None
        self.current_ts = None
        self.current_candles = None
        self.last_pick = None


class FakeTracker:
    def __init__(self, status="pending"):
        self.status = status


class TestMegaVisionConfig:
    def test_from_dict_defaults(self):
        cfg = MegaVisionConfig.from_dict(None)
        assert cfg.mode == "disabled"
        assert cfg.subscription_mode is True

    def test_from_dict_overrides(self):
        cfg = MegaVisionConfig.from_dict(
            {
                "mode": "shadow",
                "shadow_model": "claude-sonnet-4-6",
                "cost_budget_usd": 10.0,
                "subscription_mode": False,
            }
        )
        assert cfg.mode == "shadow"
        assert cfg.shadow_model == "claude-sonnet-4-6"
        assert cfg.cost_budget_usd == 10.0
        assert cfg.subscription_mode is False


class TestMegaStrategyAgentDisabled:
    def test_disabled_mode_returns_fallback_immediately(self):
        agent = MegaStrategyAgent(
            config=MegaVisionConfig(mode="disabled"),
            ctx=_FakeCtx(),
            prop_firm_tracker=FakeTracker(),
        )
        result = _run(agent.decide(datetime.now(UTC), []))
        assert result["fallback"] is True
        assert "disabled" in result["reasoning"]


class TestKillSwitch:
    def test_kill_switch_returns_fallback(self, monkeypatch):
        monkeypatch.setenv("MEGA_VISION_KILL_SWITCH", "1")
        agent = MegaStrategyAgent(
            config=MegaVisionConfig(mode="shadow"),
            ctx=_FakeCtx(),
            prop_firm_tracker=FakeTracker(),
        )
        result = _run(agent.decide(datetime.now(UTC), []))
        assert result["fallback"] is True
        assert "kill_switch" in result["reasoning"]


class TestCostBudget:
    def test_exhausted_budget_returns_fallback(self):
        agent = MegaStrategyAgent(
            config=MegaVisionConfig(
                mode="shadow", subscription_mode=False, cost_budget_usd=0.0
            ),
            ctx=_FakeCtx(),
            prop_firm_tracker=FakeTracker(),
        )
        # Force over budget
        agent.cost_tracker.total_cost_usd = 1.0
        result = _run(agent.decide(datetime.now(UTC), []))
        assert result["fallback"] is True
        assert "budget" in result["reasoning"]


class TestSdkNotInstalled:
    def test_no_sdk_returns_fallback(self):
        # Force the agent's view of the SDK to False
        with patch("src.mega_vision.agent._HAS_SDK", False):
            agent = MegaStrategyAgent(
                config=MegaVisionConfig(mode="shadow"),
                ctx=_FakeCtx(),
                prop_firm_tracker=FakeTracker(),
            )
            result = _run(agent.decide(datetime.now(UTC), []))
            assert result["fallback"] is True
            assert "sdk_not_installed" in result["reasoning"]


class TestPromptLoading:
    def test_loads_system_prompt_from_disk(self, tmp_path):
        system_md = tmp_path / "mega_vision_system.md"
        system_md.write_text("test system prompt", encoding="utf-8")
        (tmp_path / "mega_vision_user_template.md").write_text(
            "test user template {timestamp_utc}", encoding="utf-8"
        )
        agent = MegaStrategyAgent(
            config=MegaVisionConfig(mode="shadow"),
            ctx=_FakeCtx(),
            prop_firm_tracker=FakeTracker(),
            prompts_dir=tmp_path,
        )
        assert agent._load_system_prompt() == "test system prompt"

    def test_falls_back_to_default_when_prompt_missing(self, tmp_path):
        agent = MegaStrategyAgent(
            config=MegaVisionConfig(mode="shadow"),
            ctx=_FakeCtx(),
            prop_firm_tracker=FakeTracker(),
            prompts_dir=tmp_path / "doesnotexist",
        )
        system = agent._load_system_prompt()
        assert "Mega-Strategy" in system or "mega" in system.lower()
