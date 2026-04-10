"""Tests for the mega-vision trading tools (plan Task 24)."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.mega_vision.mcp_server import make_trading_mcp_server
from src.mega_vision.tools import make_tools


UTC = timezone.utc


class _FakeCtx:
    """Minimal context stub for exercising the tools."""

    def __init__(
        self,
        telemetry=None,
        trade_memory=None,
        performance_buckets=None,
        regime_detector=None,
        screenshot_provider=None,
    ):
        self.telemetry_collector = telemetry
        self.trade_memory = trade_memory
        self.performance_buckets = performance_buckets
        self.regime_detector = regime_detector
        self.screenshot_provider = screenshot_provider
        self.context_builder = MagicMock()
        self.context_builder.current_market_state.return_value = {
            "timestamp_utc": "2026-04-09T14:00:00Z",
            "bar_count": 100,
        }
        self.current_ts = datetime(2026, 4, 9, 14, 0, tzinfo=UTC)
        self.current_candles = None
        self.last_pick: dict | None = None


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestMakeTools:
    def test_returns_seven_tools(self):
        ctx = _FakeCtx()
        tools = make_tools(ctx)
        assert len(tools) == 7

    def test_tool_names_match_plan(self):
        ctx = _FakeCtx()
        tools = make_tools(ctx)
        # When the SDK isn't installed, our fallback decorator attaches
        # tool_name to the function. With the SDK, the @tool decorator
        # wraps differently — either way the function list order matches.
        expected = [
            "get_market_state",
            "get_recent_telemetry",
            "get_strategy_performance_buckets",
            "get_recent_trades",
            "get_regime_tag",
            "view_chart_screenshot",
            "record_strategy_pick",
        ]
        actual = [getattr(t, "tool_name", t.__name__) for t in tools]
        assert actual == expected


class TestGetMarketState:
    def test_returns_text_content_block(self):
        ctx = _FakeCtx()
        tools = make_tools(ctx)
        get_market_state = tools[0]
        result = _run(get_market_state({}))
        assert result["content"][0]["type"] == "text"
        payload = json.loads(result["content"][0]["text"])
        assert "timestamp_utc" in payload
        assert payload["bar_count"] == 100


class TestGetRecentTelemetry:
    def test_no_telemetry_returns_error(self):
        ctx = _FakeCtx(telemetry=None)
        tools = make_tools(ctx)
        result = _run(tools[1]({}))
        payload = json.loads(result["content"][0]["text"])
        assert payload.get("error") == "no telemetry"

    def test_with_telemetry_returns_summary(self):
        telemetry = MagicMock()
        telemetry.summary.return_value = {"total_events": 42}
        ctx = _FakeCtx(telemetry=telemetry)
        tools = make_tools(ctx)
        result = _run(tools[1]({}))
        payload = json.loads(result["content"][0]["text"])
        assert payload["total_events"] == 42


class TestGetStrategyPerformanceBuckets:
    def test_with_buckets_returns_them(self):
        perf = MagicMock()
        perf.get_buckets.return_value = {"sss": {"ny": {}}}
        ctx = _FakeCtx(performance_buckets=perf)
        tools = make_tools(ctx)
        result = _run(tools[2]({"strategy_name": "sss"}))
        payload = json.loads(result["content"][0]["text"])
        assert "sss" in payload


class TestGetRecentTrades:
    def test_returns_from_memory(self):
        memory = MagicMock()
        memory.query_recent.return_value = [{"trade_id": "1"}, {"trade_id": "2"}]
        ctx = _FakeCtx(trade_memory=memory)
        tools = make_tools(ctx)
        result = _run(tools[3]({"n": 5, "strategy_name": "sss"}))
        payload = json.loads(result["content"][0]["text"])
        assert len(payload) == 2


class TestGetRegimeTag:
    def test_no_detector_returns_unknown(self):
        ctx = _FakeCtx()
        tools = make_tools(ctx)
        result = _run(tools[4]({}))
        payload = json.loads(result["content"][0]["text"])
        assert payload["regime"] == "unknown"


class TestViewChartScreenshot:
    def test_no_provider_returns_error(self):
        ctx = _FakeCtx()
        tools = make_tools(ctx)
        result = _run(tools[5]({}))
        payload = json.loads(result["content"][0]["text"])
        assert "disabled" in payload.get("error", "").lower()

    def test_with_provider_returns_image_block(self, tmp_path: Path):
        png_path = tmp_path / "chart.png"
        png_path.write_bytes(b"\x89PNG\r\n\x1a\ntestimage")
        screenshot_provider = MagicMock()
        screenshot_provider.render_for_decision.return_value = png_path
        ctx = _FakeCtx(screenshot_provider=screenshot_provider)
        tools = make_tools(ctx)
        result = _run(tools[5]({}))
        assert result["content"][0]["type"] == "image"
        assert result["content"][0]["source"]["media_type"] == "image/png"


class TestRecordStrategyPick:
    def test_mutates_ctx_last_pick(self):
        ctx = _FakeCtx()
        tools = make_tools(ctx)
        _run(
            tools[6](
                {
                    "strategy_picks": ["sss", "ichimoku"],
                    "confidence": 0.75,
                    "reasoning": "strong regime alignment",
                }
            )
        )
        assert ctx.last_pick is not None
        assert ctx.last_pick["strategy_picks"] == ["sss", "ichimoku"]
        assert ctx.last_pick["confidence"] == 0.75
        assert "regime" in ctx.last_pick["reasoning"]

    def test_empty_picks_accepted(self):
        """Picking empty list is valid — means 'skip this signal'."""
        ctx = _FakeCtx()
        tools = make_tools(ctx)
        _run(tools[6]({"strategy_picks": [], "confidence": 0.9, "reasoning": "skip"}))
        assert ctx.last_pick is not None
        assert ctx.last_pick["strategy_picks"] == []


class TestMcpServer:
    def test_make_trading_mcp_server_returns_server_object(self):
        ctx = _FakeCtx()
        server = make_trading_mcp_server(ctx)
        # With SDK: a real MCPServer object; without: our _FakeMcpServer
        # Both should have a ``name`` attribute / key-path
        assert server is not None
