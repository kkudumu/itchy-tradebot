"""Tests for dashboard visualization helpers + screenshot capture (Task 21)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.backtesting.dashboard_visualizations import (
    render_daily_loss_gauge,
    render_mll_gauge,
    render_pattern_histogram,
    render_per_strategy_panel,
    render_session_distribution,
    render_telemetry_summary_panel,
    render_top_rejection_stages,
)
from src.backtesting.screenshot_capture import ScreenshotCapture, ScreenshotConfig


def _fixture_summary():
    return {
        "total_events": 100,
        "per_strategy": {
            "sss": {"generated": 60, "entered": 5, "rejected": 20, "entry_rate_pct": 8.33},
            "ichimoku": {"generated": 40, "entered": 2, "rejected": 10, "entry_rate_pct": 5.0},
        },
        "per_session": {"asian": 20, "london": 50, "ny": 30},
        "per_pattern": {"cbc": 30, "fifty_tap": 20, "tk_cross": 10},
        "top_rejection_stages": {"edge.spread": 15, "learning.pre_trade": 10, "in_trade": 5},
    }


class TestPerStrategyPanel:
    def test_renders_strategy_rows(self):
        html = render_per_strategy_panel(_fixture_summary())
        assert "sss" in html
        assert "ichimoku" in html
        assert "60" in html  # generated count
        assert "8.33%" in html

    def test_empty_returns_placeholder(self):
        html = render_per_strategy_panel({"per_strategy": {}})
        assert "No strategy telemetry" in html


class TestTopRejectionStages:
    def test_renders_svg_bars(self):
        html = render_top_rejection_stages(_fixture_summary())
        assert "<svg" in html
        assert "edge.spread" in html
        assert "15" in html

    def test_empty_returns_placeholder(self):
        html = render_top_rejection_stages({"top_rejection_stages": {}})
        assert "No rejections" in html


class TestSessionDistribution:
    def test_renders_stacked_bar(self):
        html = render_session_distribution(_fixture_summary())
        assert "asian" in html
        assert "london" in html
        assert "ny" in html

    def test_empty_returns_placeholder(self):
        html = render_session_distribution({"per_session": {}})
        assert "No session data" in html


class TestPatternHistogram:
    def test_renders_bars(self):
        html = render_pattern_histogram(_fixture_summary())
        assert "cbc" in html
        assert "fifty_tap" in html
        assert "30" in html

    def test_empty_returns_placeholder(self):
        html = render_pattern_histogram({"per_pattern": {}})
        assert "No patterns" in html


class TestMllGauge:
    def test_topstep_state_renders_gauge(self):
        state = {
            "active_tracker": {
                "style": "topstep_combine_dollar",
                "current_balance": 51_500.0,
                "mll": 49_000.0,
                "initial_balance": 50_000.0,
            }
        }
        html = render_mll_gauge(state)
        assert "$2,500" in html  # distance = 51500 - 49000
        assert "DIST → MLL" in html

    def test_non_topstep_returns_empty(self):
        state = {"status": "passed"}
        assert render_mll_gauge(state) == ""

    def test_none_returns_empty(self):
        assert render_mll_gauge({}) == ""


class TestDailyLossGauge:
    def test_renders_when_topstep(self):
        state = {
            "active_tracker": {
                "style": "topstep_combine_dollar",
                "daily_pnl": -250.0,
            }
        }
        html = render_daily_loss_gauge(state)
        assert "$250" in html
        assert "DAILY LOSS USED" in html


class TestComposedPanel:
    def test_full_render_concatenates(self):
        html = render_telemetry_summary_panel(_fixture_summary())
        assert "Per-strategy funnel" in html
        assert "Top rejection stages" in html
        assert "Session distribution" in html
        assert "Pattern histogram" in html

    def test_with_topstep_state_includes_gauges(self):
        state = {
            "active_tracker": {
                "style": "topstep_combine_dollar",
                "current_balance": 51_000.0,
                "mll": 49_000.0,
                "initial_balance": 50_000.0,
                "daily_pnl": -100.0,
            }
        }
        html = render_telemetry_summary_panel(_fixture_summary(), prop_firm_state=state)
        assert "TopstepX status" in html
        assert "DIST → MLL" in html


class TestScreenshotCapture:
    def test_disabled_returns_none(self, tmp_path: Path):
        cfg = ScreenshotConfig(enabled=False)
        cap = ScreenshotCapture(cfg, run_id="test")
        # When disabled, capture returns None without touching disk
        result = cap.capture(
            candles=None,
            timestamp=None,  # type: ignore[arg-type]
            strategy_name="sss",
            event_type="signal_generated",
        )
        assert result is None

    def test_disabled_every_signal_returns_none(self, tmp_path: Path):
        cfg = ScreenshotConfig(enabled=True, every_signal=False, out_dir=str(tmp_path))
        cap = ScreenshotCapture(cfg, run_id="test")
        result = cap.capture(
            candles=None,
            timestamp=None,  # type: ignore[arg-type]
            strategy_name="sss",
            event_type="signal_generated",
        )
        assert result is None

    def test_on_bar_end_cadence(self, tmp_path: Path):
        cfg = ScreenshotConfig(
            enabled=True, every_n_bars=5, every_signal=False, out_dir=str(tmp_path)
        )
        cap = ScreenshotCapture(cfg, run_id="test")
        results = [cap.on_bar_end() for _ in range(12)]
        # Triggers at bar 5 and bar 10
        assert results == [False] * 4 + [True] + [False] * 4 + [True] + [False] * 2

    def test_config_from_dict(self):
        cfg = ScreenshotConfig.from_dict(
            {"enabled": True, "every_signal": False, "every_n_bars": 3}
        )
        assert cfg.enabled is True
        assert cfg.every_signal is False
        assert cfg.every_n_bars == 3
