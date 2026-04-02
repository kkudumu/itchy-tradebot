# tests/test_event_proximity.py
"""Tests for EventProximityFilter edge filter."""

import pytest
from datetime import datetime, timezone


def _make_context(timestamp):
    """Build a minimal EdgeContext for testing."""
    from src.edges.base import EdgeContext

    return EdgeContext(
        timestamp=timestamp,
        day_of_week=timestamp.weekday(),
        close_price=2650.0,
        high_price=2655.0,
        low_price=2645.0,
        spread=0.30,
        session="london",
        adx=30.0,
        atr=5.0,
    )


class TestEventProximityFilter:
    def test_blocks_entry_near_nfp(self):
        """Trade 2 hours before NFP should be blocked."""
        from src.macro.event_proximity import EventProximityFilter

        config = {
            "enabled": True,
            "params": {
                "hours_before": 4,
                "hours_after": 2,
            },
        }
        edge = EventProximityFilter(config)

        # NFP Jan 2025 = Jan 3 at 13:30 UTC; check at 11:30 (2h before)
        ctx = _make_context(datetime(2025, 1, 3, 11, 30, tzinfo=timezone.utc))
        result = edge.should_allow(ctx)

        assert result.allowed is False
        assert "NFP" in result.reason or "Non-Farm" in result.reason

    def test_allows_entry_far_from_events(self):
        """Trade on a quiet Tuesday should be allowed."""
        from src.macro.event_proximity import EventProximityFilter

        config = {
            "enabled": True,
            "params": {
                "hours_before": 4,
                "hours_after": 2,
            },
        }
        edge = EventProximityFilter(config)

        # Feb 18 2025 is a Tuesday with no nearby events
        ctx = _make_context(datetime(2025, 2, 18, 10, 0, tzinfo=timezone.utc))
        result = edge.should_allow(ctx)

        assert result.allowed is True

    def test_disabled_filter_always_allows(self):
        """Disabled filter should pass through."""
        from src.macro.event_proximity import EventProximityFilter

        config = {"enabled": False, "params": {}}
        edge = EventProximityFilter(config)

        ctx = _make_context(datetime(2025, 1, 3, 13, 30, tzinfo=timezone.utc))
        result = edge.should_allow(ctx)

        assert result.allowed is True

    def test_blocks_entry_near_fomc(self):
        """Trade near FOMC decision should be blocked."""
        from src.macro.event_proximity import EventProximityFilter

        config = {
            "enabled": True,
            "params": {"hours_before": 4, "hours_after": 2},
        }
        edge = EventProximityFilter(config)

        # FOMC Jan 29, 2025 at 19:00 UTC; check at 16:00 (3h before)
        ctx = _make_context(datetime(2025, 1, 29, 16, 0, tzinfo=timezone.utc))
        result = edge.should_allow(ctx)

        assert result.allowed is False
        assert "FOMC" in result.reason

    def test_blocks_entry_near_cpi(self):
        """Trade near CPI release should be blocked."""
        from src.macro.event_proximity import EventProximityFilter

        config = {
            "enabled": True,
            "params": {"hours_before": 4, "hours_after": 2},
        }
        edge = EventProximityFilter(config)

        # CPI ~Jan 12, 2025 (Sunday -> Monday Jan 13) at 13:30 UTC
        # Check at 12:00 (1.5h before)
        ctx = _make_context(datetime(2025, 1, 13, 12, 0, tzinfo=timezone.utc))
        result = edge.should_allow(ctx)

        assert result.allowed is False
        assert "CPI" in result.reason

    def test_configurable_window(self):
        """Wider window should block more trades."""
        from src.macro.event_proximity import EventProximityFilter

        # Narrow window: 1h before, 1h after
        narrow_config = {
            "enabled": True,
            "params": {"hours_before": 1, "hours_after": 1},
        }
        narrow_edge = EventProximityFilter(narrow_config)

        # Wide window: 8h before, 4h after
        wide_config = {
            "enabled": True,
            "params": {"hours_before": 8, "hours_after": 4},
        }
        wide_edge = EventProximityFilter(wide_config)

        # 6 hours before NFP Jan 3, 2025
        ctx = _make_context(datetime(2025, 1, 3, 7, 30, tzinfo=timezone.utc))

        narrow_result = narrow_edge.should_allow(ctx)
        wide_result = wide_edge.should_allow(ctx)

        assert narrow_result.allowed is True  # 6h before, outside 1h window
        assert wide_result.allowed is False   # 6h before, inside 8h window
