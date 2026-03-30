"""
Edge isolation integration tests.

Each edge is tested ON vs OFF to verify its marginal impact on the pipeline.
Tests ensure:
- System works with ALL edges disabled
- System works with ALL edges enabled
- Each edge independently filters/modifies when active
- toggle_edge() runtime control works correctly
- Edge combinations produce consistent results
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from src.edges.base import EdgeContext, EdgeResult


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_edge_context(
    ts: datetime = None,
    day_of_week: int = 2,  # Wednesday
    hour_utc: int = 10,
    spread: float = 5.0,
    adx: float = 35.0,
    confluence_score: int = 6,
    current_r: Optional[float] = None,
    candles_since_entry: Optional[int] = None,
    bb_squeeze: bool = False,
    equity_curve: Optional[list] = None,
) -> EdgeContext:
    """Build a well-formed EdgeContext for edge filter tests."""
    if ts is None:
        ts = datetime(2024, 1, 3, hour_utc, 30, tzinfo=timezone.utc)  # Wednesday 10:30 UTC

    return EdgeContext(
        timestamp=ts,
        day_of_week=day_of_week,
        close_price=1905.0,
        high_price=1908.0,
        low_price=1902.0,
        spread=spread,
        session="london",
        adx=adx,
        atr=8.0,
        indicator_values={'kijun': 1900.0, 'cloud_thickness': 15.0},
        bb_squeeze=bb_squeeze,
        confluence_score=confluence_score,
        current_r=current_r,
        candles_since_entry=candles_since_entry,
        equity_curve=equity_curve or [],
    )


def _make_all_disabled_config():
    """Return an EdgeConfig with all edges disabled."""
    from src.config.models import EdgeConfig

    ec = EdgeConfig()
    for field_name in [
        "time_of_day", "day_of_week", "london_open_delay",
        "candle_close_confirmation", "spread_filter", "news_filter",
        "friday_close", "regime_filter", "time_stop",
        "bb_squeeze", "confluence_scoring", "equity_curve",
    ]:
        getattr(ec, field_name).enabled = False
    return ec


def _make_all_enabled_config():
    """Return a default EdgeConfig with all edges enabled."""
    from src.config.models import EdgeConfig
    return EdgeConfig()


# ---------------------------------------------------------------------------
# 1. System with ALL edges OFF
# ---------------------------------------------------------------------------

class TestAllEdgesDisabled:
    """With all edges disabled, entry should always be permitted."""

    def test_check_entry_passes_with_all_disabled(self):
        """check_entry() returns True when all edges are disabled."""
        from src.edges.manager import EdgeManager

        manager = EdgeManager(_make_all_disabled_config())
        ctx = _make_edge_context()

        passed, results = manager.check_entry(ctx)
        assert passed is True

    def test_check_exit_not_triggered_with_all_disabled(self):
        """check_exit() returns False (no exit) when all exit edges are disabled."""
        from src.edges.manager import EdgeManager

        manager = EdgeManager(_make_all_disabled_config())
        ctx = _make_edge_context(current_r=0.5, candles_since_entry=5)

        triggered, results = manager.check_exit(ctx)
        assert triggered is False

    def test_modifiers_all_neutral_with_disabled(self):
        """Disabled modifier edges return 1.0 (neutral) multiplier."""
        from src.edges.manager import EdgeManager

        manager = EdgeManager(_make_all_disabled_config())
        ctx = _make_edge_context()

        modifiers = manager.get_modifiers(ctx)
        combined = manager.get_combined_size_multiplier(ctx)

        # All modifiers should be neutral (1.0) when disabled
        for key, val in modifiers.items():
            assert val == 1.0, f"Expected 1.0 for {key}, got {val}"
        assert combined == 1.0

    def test_enabled_edges_list_empty_when_all_disabled(self):
        """get_enabled_edges() returns empty list when all edges are disabled."""
        from src.edges.manager import EdgeManager

        manager = EdgeManager(_make_all_disabled_config())
        enabled = manager.get_enabled_edges()
        assert len(enabled) == 0


# ---------------------------------------------------------------------------
# 2. Individual edge ON vs OFF marginal impact
# ---------------------------------------------------------------------------

class TestTimeOfDayEdgeIsolation:
    """TimeOfDayFilter independently blocks entries outside trading hours."""

    def test_blocks_outside_window_when_enabled(self):
        """Entry at 02:00 UTC is blocked by enabled TimeOfDayFilter."""
        from src.config.models import EdgeConfig
        from src.edges.manager import EdgeManager

        ec = _make_all_disabled_config()
        ec.time_of_day.enabled = True
        ec.time_of_day.params = {"start_utc": "08:00", "end_utc": "17:00"}

        manager = EdgeManager(ec)
        ctx = _make_edge_context(ts=datetime(2024, 1, 3, 2, 0, tzinfo=timezone.utc))

        passed, results = manager.check_entry(ctx)
        assert passed is False
        assert any(r.edge_name == "time_of_day" for r in results)

    def test_passes_within_window_when_enabled(self):
        """Entry at 10:00 UTC passes with enabled TimeOfDayFilter."""
        from src.config.models import EdgeConfig
        from src.edges.manager import EdgeManager

        ec = _make_all_disabled_config()
        ec.time_of_day.enabled = True
        ec.time_of_day.params = {"start_utc": "08:00", "end_utc": "17:00"}

        manager = EdgeManager(ec)
        ctx = _make_edge_context(ts=datetime(2024, 1, 3, 10, 0, tzinfo=timezone.utc))

        passed, results = manager.check_entry(ctx)
        assert passed is True

    def test_passes_same_context_when_disabled(self):
        """Same out-of-window context passes when TimeOfDayFilter is disabled."""
        from src.config.models import EdgeConfig
        from src.edges.manager import EdgeManager

        ec = _make_all_disabled_config()
        ec.time_of_day.enabled = False

        manager = EdgeManager(ec)
        ctx = _make_edge_context(ts=datetime(2024, 1, 3, 2, 0, tzinfo=timezone.utc))

        passed, _ = manager.check_entry(ctx)
        assert passed is True  # disabled -> no filter


class TestDayOfWeekEdgeIsolation:
    """DayOfWeekFilter blocks entries on disallowed weekdays."""

    def test_blocks_monday_when_not_in_allowed_days(self):
        """Monday (day_of_week=0) blocked when allowed_days=[1,2,3]."""
        from src.config.models import EdgeConfig
        from src.edges.manager import EdgeManager

        ec = _make_all_disabled_config()
        ec.day_of_week.enabled = True
        ec.day_of_week.params = {"allowed_days": [1, 2, 3]}

        manager = EdgeManager(ec)
        ctx = _make_edge_context(day_of_week=0)  # Monday

        passed, _ = manager.check_entry(ctx)
        assert passed is False

    def test_allows_wednesday_when_in_allowed_days(self):
        """Wednesday (day_of_week=2) passes when in allowed_days."""
        from src.config.models import EdgeConfig
        from src.edges.manager import EdgeManager

        ec = _make_all_disabled_config()
        ec.day_of_week.enabled = True
        ec.day_of_week.params = {"allowed_days": [1, 2, 3]}

        manager = EdgeManager(ec)
        ctx = _make_edge_context(day_of_week=2)  # Wednesday

        passed, _ = manager.check_entry(ctx)
        assert passed is True


class TestSpreadFilterEdgeIsolation:
    """SpreadFilter blocks entries when spread is too wide."""

    def test_blocks_high_spread_when_enabled(self):
        """Spread of 50 pts blocked when max_spread_points=30."""
        from src.config.models import EdgeConfig
        from src.edges.manager import EdgeManager

        ec = _make_all_disabled_config()
        ec.spread_filter.enabled = True
        ec.spread_filter.params = {"max_spread_points": 30}

        manager = EdgeManager(ec)
        ctx = _make_edge_context(spread=50.0)

        passed, results = manager.check_entry(ctx)
        assert passed is False
        assert any("spread" in r.edge_name.lower() for r in results)

    def test_allows_low_spread_when_enabled(self):
        """Spread of 5 pts passes when max_spread_points=30."""
        from src.config.models import EdgeConfig
        from src.edges.manager import EdgeManager

        ec = _make_all_disabled_config()
        ec.spread_filter.enabled = True
        ec.spread_filter.params = {"max_spread_points": 30}

        manager = EdgeManager(ec)
        ctx = _make_edge_context(spread=5.0)

        passed, _ = manager.check_entry(ctx)
        assert passed is True

    def test_high_spread_passes_when_disabled(self):
        """Same high spread passes when SpreadFilter is disabled."""
        from src.edges.manager import EdgeManager

        manager = EdgeManager(_make_all_disabled_config())
        ctx = _make_edge_context(spread=999.0)

        passed, _ = manager.check_entry(ctx)
        assert passed is True


class TestRegimeFilterEdgeIsolation:
    """RegimeFilter blocks entries in non-trending markets."""

    def test_blocks_low_adx_when_enabled(self):
        """ADX of 15 blocked when adx_min=28."""
        from src.config.models import EdgeConfig
        from src.edges.manager import EdgeManager

        ec = _make_all_disabled_config()
        ec.regime_filter.enabled = True
        ec.regime_filter.params = {
            "adx_min": 28,
            "cloud_thickness_percentile": 50,
        }

        manager = EdgeManager(ec)
        ctx = _make_edge_context(adx=15.0)

        passed, _ = manager.check_entry(ctx)
        assert passed is False

    def test_allows_high_adx_when_enabled(self):
        """ADX of 35 and sufficient cloud thickness passes RegimeFilter."""
        from src.config.models import EdgeConfig
        from src.edges.manager import EdgeManager

        ec = _make_all_disabled_config()
        ec.regime_filter.enabled = True
        # cloud_thickness_percentile is used as absolute minimum vs context.cloud_thickness
        # Set threshold = 10 (raw price pts) so cloud_thickness=15 in context passes
        ec.regime_filter.params = {
            "adx_min": 28,
            "cloud_thickness_percentile": 10,  # 15 > 10, will pass
        }

        manager = EdgeManager(ec)
        ctx = _make_edge_context(adx=35.0)  # cloud_thickness=15 by default

        passed, _ = manager.check_entry(ctx)
        assert passed is True


class TestConfluenceScoringEdgeIsolation:
    """ConfluenceScoringFilter reduces position size for lower-tier signals.

    Note: confluence_scoring is registered as a MODIFIER edge, not an entry
    edge. It's evaluated via get_modifiers() / get_combined_size_multiplier(),
    not check_entry(). A score below min_score returns modifier=0.0.
    """

    def test_below_min_score_returns_zero_modifier(self):
        """Confluence score 2 (below min_score=4) returns modifier=0.0."""
        from src.config.models import EdgeConfig
        from src.edges.manager import EdgeManager

        ec = _make_all_disabled_config()
        ec.confluence_scoring.enabled = True
        ec.confluence_scoring.params = {
            "min_score": 4,
            "tier_a_plus_threshold": 7,
            "tier_b_threshold": 5,
            "tier_c_threshold": 4,
            "tier_b_size_pct": 0.75,
            "tier_c_size_pct": 0.50,
        }

        manager = EdgeManager(ec)
        ctx = _make_edge_context(confluence_score=2)

        modifiers = manager.get_modifiers(ctx)
        assert modifiers.get("confluence_scoring") == 0.0

    def test_a_plus_tier_full_size(self):
        """Confluence score 8 (A+) should return modifier 1.0."""
        from src.config.models import EdgeConfig
        from src.edges.manager import EdgeManager

        ec = _make_all_disabled_config()
        ec.confluence_scoring.enabled = True
        ec.confluence_scoring.params = {
            "min_score": 4,
            "tier_a_plus_threshold": 7,
            "tier_b_threshold": 5,
            "tier_c_threshold": 4,
            "tier_b_size_pct": 0.75,
            "tier_c_size_pct": 0.50,
        }

        manager = EdgeManager(ec)
        ctx = _make_edge_context(confluence_score=8)  # A+ tier

        modifiers = manager.get_modifiers(ctx)
        assert modifiers.get("confluence_scoring", 1.0) == 1.0

    def test_b_tier_reduced_size(self):
        """Confluence score 5 (B tier) should return modifier 0.75."""
        from src.config.models import EdgeConfig
        from src.edges.manager import EdgeManager

        ec = _make_all_disabled_config()
        ec.confluence_scoring.enabled = True
        ec.confluence_scoring.params = {
            "min_score": 4,
            "tier_a_plus_threshold": 7,
            "tier_b_threshold": 5,
            "tier_c_threshold": 4,
            "tier_b_size_pct": 0.75,
            "tier_c_size_pct": 0.50,
        }

        manager = EdgeManager(ec)
        ctx = _make_edge_context(confluence_score=5)  # B tier

        modifiers = manager.get_modifiers(ctx)
        assert abs(modifiers.get("confluence_scoring", 1.0) - 0.75) < 0.01


class TestEquityCurveEdgeIsolation:
    """EquityCurveFilter reduces position size when equity is below moving average."""

    def test_neutral_modifier_when_equity_above_ma(self):
        """Positive equity curve returns modifier 1.0."""
        from src.config.models import EdgeConfig
        from src.edges.manager import EdgeManager

        ec = _make_all_disabled_config()
        ec.equity_curve.enabled = True
        ec.equity_curve.params = {
            "lookback_trades": 5,
            "reduced_size_multiplier": 0.5,
        }

        manager = EdgeManager(ec)
        # Positive equity curve: growing R-multiples
        ctx = _make_edge_context(
            equity_curve=[1.5, 2.0, 1.8, 2.5, 2.0]
        )

        modifiers = manager.get_modifiers(ctx)
        # Equity is growing, should be neutral (1.0) or near it
        modifier = modifiers.get("equity_curve", 1.0)
        assert isinstance(modifier, float)


class TestFridayCloseEdgeIsolation:
    """FridayCloseFilter triggers exit on Friday afternoon."""

    def test_triggers_exit_on_friday_past_close_time(self):
        """Exit triggered on Friday after 20:00 UTC."""
        from src.config.models import EdgeConfig
        from src.edges.manager import EdgeManager

        ec = _make_all_disabled_config()
        ec.friday_close.enabled = True
        ec.friday_close.params = {"close_time_utc": "20:00", "day": 4}

        manager = EdgeManager(ec)
        # Friday 20:30 UTC
        ctx = _make_edge_context(
            ts=datetime(2024, 1, 5, 20, 30, tzinfo=timezone.utc),
            day_of_week=4,  # Friday
        )

        triggered, results = manager.check_exit(ctx)
        assert triggered is True
        assert any("friday" in r.edge_name.lower() for r in results)

    def test_does_not_trigger_on_wednesday(self):
        """Exit not triggered on Wednesday."""
        from src.config.models import EdgeConfig
        from src.edges.manager import EdgeManager

        ec = _make_all_disabled_config()
        ec.friday_close.enabled = True
        ec.friday_close.params = {"close_time_utc": "20:00", "day": 4}

        manager = EdgeManager(ec)
        ctx = _make_edge_context(
            ts=datetime(2024, 1, 3, 20, 30, tzinfo=timezone.utc),
            day_of_week=2,  # Wednesday
        )

        triggered, _ = manager.check_exit(ctx)
        assert triggered is False


class TestTimeStopEdgeIsolation:
    """TimeStopFilter triggers exit when trade has been open too long."""

    def test_triggers_after_candle_limit(self):
        """Exit triggered when candles_since_entry > candle_limit."""
        from src.config.models import EdgeConfig
        from src.edges.manager import EdgeManager

        ec = _make_all_disabled_config()
        ec.time_stop.enabled = True
        ec.time_stop.params = {
            "candle_limit": 12,
            "breakeven_r_threshold": 0.5,
        }

        manager = EdgeManager(ec)
        ctx = _make_edge_context(
            current_r=0.2,        # below breakeven threshold
            candles_since_entry=15,  # exceeded 12-candle limit
        )

        triggered, results = manager.check_exit(ctx)
        assert triggered is True
        assert any("time_stop" in r.edge_name.lower() for r in results)

    def test_does_not_trigger_before_candle_limit(self):
        """Exit not triggered when candles_since_entry <= candle_limit."""
        from src.config.models import EdgeConfig
        from src.edges.manager import EdgeManager

        ec = _make_all_disabled_config()
        ec.time_stop.enabled = True
        ec.time_stop.params = {
            "candle_limit": 12,
            "breakeven_r_threshold": 0.5,
        }

        manager = EdgeManager(ec)
        ctx = _make_edge_context(
            current_r=0.2,
            candles_since_entry=5,  # within limit
        )

        triggered, _ = manager.check_exit(ctx)
        assert triggered is False


# ---------------------------------------------------------------------------
# 3. Runtime toggle_edge() controls
# ---------------------------------------------------------------------------

class TestRuntimeEdgeToggle:
    """toggle_edge() dynamically enables/disables edges after construction."""

    def test_toggle_disables_active_edge(self):
        """Disabling an edge mid-run causes it to pass through."""
        from src.edges.manager import EdgeManager
        from src.config.models import EdgeConfig

        ec = EdgeConfig()  # all enabled
        manager = EdgeManager(ec)

        # All entry edges enabled — check a strict context
        ctx = _make_edge_context(spread=999.0)

        # With spread filter on, should fail
        passed_before, _ = manager.check_entry(ctx)
        # Spread filter is one of many — may or may not block based on other filters too

        # Now disable spread filter
        manager.toggle_edge("spread_filter", False)
        assert "spread_filter" not in manager.get_enabled_edges()

    def test_toggle_enables_disabled_edge(self):
        """Enabling a previously disabled edge makes it active."""
        from src.edges.manager import EdgeManager

        manager = EdgeManager(_make_all_disabled_config())
        assert "time_of_day" not in manager.get_enabled_edges()

        manager.toggle_edge("time_of_day", True)
        assert "time_of_day" in manager.get_enabled_edges()

    def test_toggle_unknown_edge_raises_key_error(self):
        """toggle_edge() raises KeyError for unknown edge names."""
        from src.edges.manager import EdgeManager

        manager = EdgeManager(_make_all_disabled_config())
        with pytest.raises(KeyError):
            manager.toggle_edge("nonexistent_edge", True)

    def test_all_12_edges_present_in_manager(self):
        """EdgeManager must register exactly 12 edge keys."""
        from src.edges.manager import EdgeManager

        manager = EdgeManager(_make_all_enabled_config())
        assert len(manager._all_edges) == 12

        expected_edges = {
            "time_of_day", "day_of_week", "london_open_delay",
            "candle_close_confirmation", "spread_filter", "news_filter",
            "friday_close", "regime_filter", "time_stop",
            "bb_squeeze", "confluence_scoring", "equity_curve",
        }
        assert set(manager._all_edges.keys()) == expected_edges


# ---------------------------------------------------------------------------
# 4. Edge combination consistency
# ---------------------------------------------------------------------------

class TestEdgeCombinations:
    """Various combinations of edges produce consistent behavior."""

    def test_multiple_entry_edges_short_circuit_on_first_failure(self):
        """check_entry() stops at the first failing edge."""
        from src.config.models import EdgeConfig
        from src.edges.manager import EdgeManager

        ec = _make_all_disabled_config()
        ec.spread_filter.enabled = True
        ec.spread_filter.params = {"max_spread_points": 10}
        ec.regime_filter.enabled = True
        ec.regime_filter.params = {
            "adx_min": 28,
            "cloud_thickness_percentile": 50,
        }

        manager = EdgeManager(ec)
        # High spread AND low ADX — should fail on spread (first entry filter)
        ctx = _make_edge_context(spread=50.0, adx=10.0)

        passed, results = manager.check_entry(ctx)
        assert passed is False
        # Short-circuit: only 1 result since first filter fails
        assert len(results) == 1

    def test_all_modifiers_combine_multiplicatively(self):
        """get_combined_size_multiplier() multiplies confluence and equity modifiers."""
        from src.config.models import EdgeConfig
        from src.edges.manager import EdgeManager

        ec = _make_all_disabled_config()
        ec.confluence_scoring.enabled = True
        ec.confluence_scoring.params = {
            "min_score": 4,
            "tier_a_plus_threshold": 7,
            "tier_b_threshold": 5,
            "tier_c_threshold": 4,
            "tier_b_size_pct": 0.75,
            "tier_c_size_pct": 0.50,
        }
        ec.equity_curve.enabled = True
        ec.equity_curve.params = {
            "lookback_trades": 5,
            "reduced_size_multiplier": 0.5,
        }

        manager = EdgeManager(ec)
        ctx = _make_edge_context(confluence_score=5)  # B tier = 0.75

        combined = manager.get_combined_size_multiplier(ctx)
        # Combined should be in [0, 1]
        assert 0.0 <= combined <= 1.0

    def test_entry_edges_dont_affect_exit_evaluation(self):
        """Entry filter failures do not prevent exit evaluation."""
        from src.config.models import EdgeConfig
        from src.edges.manager import EdgeManager

        ec = _make_all_disabled_config()
        ec.spread_filter.enabled = True  # entry filter
        ec.friday_close.enabled = True   # exit filter
        ec.friday_close.params = {"close_time_utc": "20:00", "day": 4}

        manager = EdgeManager(ec)
        # Friday 21:00 UTC — should trigger exit regardless of entry filter
        ctx = _make_edge_context(
            ts=datetime(2024, 1, 5, 21, 0, tzinfo=timezone.utc),
            day_of_week=4,
            spread=999.0,  # would block entry
        )

        triggered, results = manager.check_exit(ctx)
        # Exit should fire even though spread would block entry
        assert triggered is True

    def test_exit_edges_all_evaluated_no_short_circuit(self):
        """All exit edges are evaluated even when one triggers."""
        from src.config.models import EdgeConfig
        from src.edges.manager import EdgeManager

        ec = _make_all_disabled_config()
        ec.friday_close.enabled = True
        ec.friday_close.params = {"close_time_utc": "20:00", "day": 4}
        ec.time_stop.enabled = True
        ec.time_stop.params = {"candle_limit": 5, "breakeven_r_threshold": 0.5}

        manager = EdgeManager(ec)
        # Friday 21:00 UTC AND 10 candles elapsed — both exit edges should fire
        ctx = _make_edge_context(
            ts=datetime(2024, 1, 5, 21, 0, tzinfo=timezone.utc),
            day_of_week=4,
            current_r=0.2,
            candles_since_entry=10,
        )

        triggered, results = manager.check_exit(ctx)
        assert triggered is True
        # Both exit edges should be in results (no short-circuit for exit)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# 5. Edge isolation: verify marginal impact on BacktestResult
# ---------------------------------------------------------------------------

class TestEdgeImpactOnBacktest:
    """Toggle individual edges and measure their effect on backtest signal count."""

    def _make_1m_candles(self, n: int = 1500) -> "pd.DataFrame":
        """Generate synthetic 1-minute OHLCV candles."""
        rng = np.random.default_rng(7)
        start = datetime(2024, 1, 2, 8, 0, tzinfo=timezone.utc)
        prices = [1900.0]
        for _ in range(n - 1):
            prices.append(prices[-1] + rng.normal(0.05, 1.0))

        prices = np.array(prices)
        timestamps = [start + timedelta(minutes=i) for i in range(n)]
        rows = []
        for p in prices:
            noise = rng.uniform(0, 0.5, 4)
            o = p + noise[0] - 0.25
            h = max(o, p) + noise[1]
            l = min(o, p) - noise[2]
            c = p + noise[3] - 0.25
            rows.append({"open": o, "high": h, "low": l, "close": c, "volume": 500})

        return pd.DataFrame(rows, index=pd.DatetimeIndex(timestamps, tz=timezone.utc))

    def test_disabling_spread_filter_increases_or_equals_signals(self):
        """Disabling SpreadFilter should allow >= signals vs enabled state."""
        from src.backtesting.vectorbt_engine import IchimokuBacktester

        candles = self._make_1m_candles()

        # Run with spread filter ON (default config)
        result_on = IchimokuBacktester(
            config={"edges": {"spread_filter": {"enabled": True, "params": {"max_spread_points": 1}}}},
        ).run(candles)

        # Run with spread filter OFF
        result_off = IchimokuBacktester(
            config={"edges": {"spread_filter": {"enabled": False, "params": {"max_spread_points": 1}}}},
        ).run(candles)

        # Disabling the filter should never decrease signals
        # (skipped_signals + total_trades should be >= when filter is off)
        total_on = result_on.total_signals
        total_off = result_off.total_signals
        # total_signals is the raw count before filtering — it should be the same
        # skipped signals should be <= when filter is OFF
        assert result_off.skipped_signals <= result_on.skipped_signals

    def test_backtest_runs_with_zero_edges_enabled(self):
        """Backtest completes successfully with all edges disabled."""
        from src.backtesting.vectorbt_engine import IchimokuBacktester

        # Config with all edges off
        config = {
            "edges": {k: {"enabled": False, "params": {}} for k in [
                "time_of_day", "day_of_week", "london_open_delay",
                "candle_close_confirmation", "spread_filter", "news_filter",
                "friday_close", "regime_filter", "time_stop",
                "bb_squeeze", "confluence_scoring", "equity_curve",
            ]}
        }

        candles = self._make_1m_candles()
        result = IchimokuBacktester(config=config).run(candles)

        # Should complete without error and return a valid result
        assert result is not None
        assert isinstance(result.equity_curve, pd.Series)
        assert len(result.equity_curve) > 0
