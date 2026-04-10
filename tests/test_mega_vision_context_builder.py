"""Tests for ContextBuilder + ScreenshotProvider (plan Task 23)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.mega_vision.context_builder import ContextBuilder, ContextBundle
from src.mega_vision.screenshot_provider import (
    ScreenshotProvider,
    load_image_as_content_block,
)


UTC = timezone.utc


@pytest.fixture
def ts():
    return datetime(2026, 4, 9, 15, 0, tzinfo=UTC)


@pytest.fixture
def recent_bars():
    return pd.DataFrame(
        {
            "open": [2000.0] * 60,
            "high": [2001.0] * 60,
            "low": [1999.0] * 60,
            "close": [2000.5] * 60,
            "volume": [100] * 60,
        }
    )


class TestContextBuilder:
    def test_build_with_no_collaborators(self, ts, recent_bars):
        builder = ContextBuilder()
        bundle = builder.build(
            ts=ts,
            candidate_signals=[],
            current_state={"recent_bars": recent_bars},
        )
        assert isinstance(bundle, ContextBundle)
        assert bundle.timestamp_utc == ts
        assert bundle.recent_bars is recent_bars
        assert bundle.candidate_signals == []

    def test_build_populates_telemetry_summary(self, ts, recent_bars):
        telemetry = MagicMock()
        telemetry.summary.return_value = {"total_events": 42}
        builder = ContextBuilder(telemetry_collector=telemetry)
        bundle = builder.build(ts, [], {"recent_bars": recent_bars})
        assert bundle.recent_telemetry_summary == {"total_events": 42}

    def test_build_populates_performance_buckets(self, ts, recent_bars):
        perf = MagicMock()
        perf.get_buckets.return_value = {"sss": {}}
        builder = ContextBuilder(performance_buckets=perf)
        bundle = builder.build(ts, [], {"recent_bars": recent_bars})
        assert "sss" in bundle.performance_buckets

    def test_build_populates_recent_trades(self, ts, recent_bars):
        memory = MagicMock()
        memory.query_recent.return_value = [{"trade_id": "1"}, {"trade_id": "2"}]
        builder = ContextBuilder(trade_memory=memory)
        bundle = builder.build(ts, [], {"recent_bars": recent_bars})
        assert len(bundle.recent_trades) == 2

    def test_build_populates_screenshot_path(self, ts, recent_bars, tmp_path: Path):
        screenshot_provider = MagicMock()
        fake_path = tmp_path / "fake.png"
        fake_path.write_bytes(b"\x89PNG\r\n\x1a\n")  # PNG header
        screenshot_provider.render_for_decision.return_value = fake_path
        builder = ContextBuilder(screenshot_provider=screenshot_provider)
        bundle = builder.build(ts, [], {"recent_bars": recent_bars})
        assert bundle.screenshot_path == str(fake_path)

    def test_build_carries_prop_firm_state(self, ts, recent_bars):
        builder = ContextBuilder()
        bundle = builder.build(
            ts,
            [],
            {
                "recent_bars": recent_bars,
                "prop_firm_state": {"status": "pending", "mll": 48_000.0},
            },
        )
        assert bundle.prop_firm_state["status"] == "pending"

    def test_tool_mutable_state_set(self, ts, recent_bars):
        builder = ContextBuilder()
        builder.build(ts, [], {"recent_bars": recent_bars})
        assert builder.current_ts == ts
        assert builder.current_candles is recent_bars

    def test_collaborator_exceptions_swallowed(self, ts, recent_bars):
        """A misbehaving collaborator shouldn't crash the builder."""
        telemetry = MagicMock()
        telemetry.summary.side_effect = RuntimeError("kaboom")
        builder = ContextBuilder(telemetry_collector=telemetry)
        bundle = builder.build(ts, [], {"recent_bars": recent_bars})
        assert bundle.recent_telemetry_summary == {}

    def test_to_summary_dict(self, ts, recent_bars):
        builder = ContextBuilder()
        bundle = builder.build(ts, [], {"recent_bars": recent_bars})
        summary = bundle.to_summary_dict()
        assert "timestamp_utc" in summary
        assert summary["candidate_signal_count"] == 0


class TestScreenshotProvider:
    def test_render_with_no_capture_returns_none(self, ts, recent_bars):
        provider = ScreenshotProvider(screenshot_capture=None)
        assert provider.render_for_decision(ts, recent_bars) is None

    def test_render_delegates_to_capture(self, ts, recent_bars, tmp_path: Path):
        capture = MagicMock()
        fake_path = tmp_path / "out.png"
        fake_path.write_bytes(b"\x89PNG")
        capture.capture.return_value = fake_path
        provider = ScreenshotProvider(screenshot_capture=capture)
        result = provider.render_for_decision(ts, recent_bars)
        assert result == fake_path
        capture.capture.assert_called_once()

    def test_load_image_as_content_block(self, tmp_path: Path):
        png = tmp_path / "img.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\ntestdata")
        block = load_image_as_content_block(png)
        assert block["type"] == "image"
        assert block["source"]["type"] == "base64"
        assert block["source"]["media_type"] == "image/png"
        import base64

        decoded = base64.standard_b64decode(block["source"]["data"])
        assert decoded == b"\x89PNG\r\n\x1a\ntestdata"
