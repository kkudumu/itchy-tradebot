"""
Tests for AdaptiveRelaxer + edge setter API.

Covers:
- Edge setter methods (RegimeFilter, ConfluenceScoringFilter,
  TimeOfDayFilter, DayOfWeekFilter, LondonOpenDelayFilter)
- EdgeManager.get_edge() and set_edge_param()
- 5-tier relaxation ladder
- Shield function (hard floors, ceilings, velocity limits)
- Budget tracking (accumulates, halts at 30%)
- Tightening refunds budget
- tighten_all() reverts to base config in one step
- Cooldown enforcement
- Consecutive loss tracking and auto-tighten
- Edge cases: budget exhaustion, rapid tighten after relax, boundaries
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from datetime import datetime

from src.edges.regime_filter import RegimeFilter
from src.edges.confluence_scoring import ConfluenceScoringFilter
from src.edges.time_of_day import TimeOfDayFilter
from src.edges.day_of_week import DayOfWeekFilter
from src.edges.london_open_delay import LondonOpenDelayFilter
from src.edges.manager import EdgeManager
from src.monitoring.adaptive_relaxer import (
    AdaptiveRelaxer,
    RelaxationState,
    ShieldBounds,
    SHIELD_BOUNDS,
    DEFAULT_TIERS,
    TierConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def minimal_edge_config() -> dict:
    """Minimal EdgeManager config with sensible defaults for all edges."""
    return {
        "time_of_day": {
            "enabled": True,
            "params": {"start_utc": "08:00", "end_utc": "17:00"},
        },
        "day_of_week": {
            "enabled": True,
            "params": {"allowed_days": [1, 2, 3]},
        },
        "london_open_delay": {
            "enabled": True,
            "params": {"london_open_utc": "08:00", "delay_minutes": 30},
        },
        "regime_filter": {
            "enabled": True,
            "params": {"adx_min": 28, "cloud_thickness_percentile": 5.0},
        },
        "confluence_scoring": {
            "enabled": True,
            "params": {"min_score": 4},
        },
        "candle_close_confirmation": {"enabled": False, "params": {}},
        "spread_filter": {"enabled": False, "params": {}},
        "news_filter": {"enabled": False, "params": {}},
        "friday_close": {"enabled": False, "params": {}},
        "time_stop": {"enabled": False, "params": {}},
        "bb_squeeze": {"enabled": False, "params": {}},
        "equity_curve": {"enabled": False, "params": {}},
    }


@pytest.fixture
def edge_manager():
    return EdgeManager(minimal_edge_config())


@pytest.fixture
def relaxer(edge_manager):
    return AdaptiveRelaxer(edge_manager)


# ---------------------------------------------------------------------------
# RegimeFilter setter tests
# ---------------------------------------------------------------------------

class TestRegimeFilterSetters:
    def test_get_adx_min_returns_default(self):
        rf = RegimeFilter({"params": {"adx_min": 28}})
        assert rf.get_adx_min() == 28.0

    def test_set_adx_min_updates_value(self):
        rf = RegimeFilter({"params": {"adx_min": 28}})
        rf.set_adx_min(22.0)
        assert rf.get_adx_min() == 22.0

    def test_set_adx_min_coerces_to_float(self):
        rf = RegimeFilter({"params": {}})
        rf.set_adx_min(20)
        assert isinstance(rf.get_adx_min(), float)

    def test_set_adx_min_affects_should_allow(self):
        """Lowering ADX min should allow weaker-trend bar."""
        rf = RegimeFilter({"params": {"adx_min": 28, "cloud_thickness_percentile": 0.0}})
        ctx = MagicMock()
        ctx.adx = 24.0
        ctx.cloud_thickness = 10.0
        # Before: ADX 24 < 28 → blocked
        result_before = rf.should_allow(ctx)
        assert not result_before.allowed
        # After lowering to 22: ADX 24 ≥ 22 → allowed
        rf.set_adx_min(22.0)
        result_after = rf.should_allow(ctx)
        assert result_after.allowed


# ---------------------------------------------------------------------------
# ConfluenceScoringFilter setter tests
# ---------------------------------------------------------------------------

class TestConfluenceScoringFilterSetters:
    def test_get_min_score_returns_default(self):
        cs = ConfluenceScoringFilter({"params": {"min_score": 4}})
        assert cs.get_min_score() == 4

    def test_set_min_score_updates_value(self):
        cs = ConfluenceScoringFilter({"params": {"min_score": 4}})
        cs.set_min_score(3)
        assert cs.get_min_score() == 3

    def test_set_min_score_coerces_to_int(self):
        cs = ConfluenceScoringFilter({"params": {}})
        cs.set_min_score(3.0)
        assert isinstance(cs.get_min_score(), int)

    def test_set_min_score_affects_should_allow(self):
        cs = ConfluenceScoringFilter({"params": {"min_score": 4}})
        ctx = MagicMock()
        ctx.confluence_score = 3
        # Before: score 3 < 4 → blocked
        assert not cs.should_allow(ctx).allowed
        # After lowering to 3: score 3 ≥ 3 → allowed
        cs.set_min_score(3)
        assert cs.should_allow(ctx).allowed


# ---------------------------------------------------------------------------
# TimeOfDayFilter setter tests
# ---------------------------------------------------------------------------

class TestTimeOfDayFilterSetters:
    def test_get_window_returns_defaults(self):
        tod = TimeOfDayFilter({"params": {"start_utc": "08:00", "end_utc": "17:00"}})
        assert tod.get_window() == (480, 1020)

    def test_set_window_updates_values(self):
        tod = TimeOfDayFilter({"params": {}})
        tod.set_window(360, 1140)
        assert tod.get_window() == (360, 1140)

    def test_set_window_affects_should_allow(self):
        tod = TimeOfDayFilter({"params": {"start_utc": "08:00", "end_utc": "17:00"}})
        ctx = MagicMock()
        ctx.timestamp = datetime(2024, 1, 2, 6, 30)  # 06:30 UTC — outside base window
        assert not tod.should_allow(ctx).allowed
        # Widen window to 06:00–19:00
        tod.set_window(360, 1140)
        assert tod.should_allow(ctx).allowed

    def test_set_window_coerces_to_int(self):
        tod = TimeOfDayFilter({"params": {}})
        tod.set_window(360.5, 1140.5)
        start, end = tod.get_window()
        assert isinstance(start, int)
        assert isinstance(end, int)


# ---------------------------------------------------------------------------
# DayOfWeekFilter setter tests
# ---------------------------------------------------------------------------

class TestDayOfWeekFilterSetters:
    def test_get_allowed_days_returns_defaults(self):
        dow = DayOfWeekFilter({"params": {"allowed_days": [1, 2, 3]}})
        assert dow.get_allowed_days() == {1, 2, 3}

    def test_set_allowed_days_updates_value(self):
        dow = DayOfWeekFilter({"params": {}})
        dow.set_allowed_days({0, 1, 2, 3, 4})
        assert dow.get_allowed_days() == {0, 1, 2, 3, 4}

    def test_get_allowed_days_returns_copy(self):
        dow = DayOfWeekFilter({"params": {"allowed_days": [1, 2, 3]}})
        days = dow.get_allowed_days()
        days.add(0)
        # Original should be unchanged
        assert 0 not in dow.get_allowed_days()

    def test_set_allowed_days_affects_should_allow(self):
        dow = DayOfWeekFilter({"params": {"allowed_days": [1, 2, 3]}})
        ctx = MagicMock()
        ctx.day_of_week = 0  # Monday
        assert not dow.should_allow(ctx).allowed
        dow.set_allowed_days({0, 1, 2, 3, 4})
        assert dow.should_allow(ctx).allowed


# ---------------------------------------------------------------------------
# LondonOpenDelayFilter setter tests
# ---------------------------------------------------------------------------

class TestLondonOpenDelayFilterSetters:
    def test_set_enabled_disables_filter(self):
        lod = LondonOpenDelayFilter({"enabled": True, "params": {}})
        assert lod.enabled is True
        lod.set_enabled(False)
        assert lod.enabled is False

    def test_disabled_filter_allows_all(self):
        lod = LondonOpenDelayFilter({"enabled": True, "params": {}})
        ctx = MagicMock()
        ctx.timestamp = datetime(2024, 1, 2, 8, 15)  # In blackout window
        assert not lod.should_allow(ctx).allowed
        lod.set_enabled(False)
        assert lod.should_allow(ctx).allowed

    def test_set_enabled_re_enables_filter(self):
        lod = LondonOpenDelayFilter({"enabled": False, "params": {}})
        lod.set_enabled(True)
        assert lod.enabled is True


# ---------------------------------------------------------------------------
# EdgeManager.get_edge() and set_edge_param() tests
# ---------------------------------------------------------------------------

class TestEdgeManagerGetEdge:
    def test_get_edge_returns_instance(self, edge_manager):
        edge = edge_manager.get_edge("regime_filter")
        assert isinstance(edge, RegimeFilter)

    def test_get_edge_unknown_raises_key_error(self, edge_manager):
        with pytest.raises(KeyError, match="Unknown edge"):
            edge_manager.get_edge("nonexistent_edge")

    def test_get_edge_all_known_edges(self, edge_manager):
        for name in ["time_of_day", "day_of_week", "london_open_delay",
                     "regime_filter", "confluence_scoring"]:
            edge = edge_manager.get_edge(name)
            assert edge is not None


class TestEdgeManagerSetEdgeParam:
    def test_set_edge_param_regime_adx_min(self, edge_manager):
        edge_manager.set_edge_param("regime_filter", "adx_min", 22.0)
        rf = edge_manager.get_edge("regime_filter")
        assert rf.get_adx_min() == 22.0  # type: ignore[attr-defined]

    def test_set_edge_param_confluence_min_score(self, edge_manager):
        edge_manager.set_edge_param("confluence_scoring", "min_score", 3)
        cs = edge_manager.get_edge("confluence_scoring")
        assert cs.get_min_score() == 3  # type: ignore[attr-defined]

    def test_set_edge_param_day_of_week(self, edge_manager):
        edge_manager.set_edge_param("day_of_week", "allowed_days", {0, 1, 2, 3, 4})
        dow = edge_manager.get_edge("day_of_week")
        assert dow.get_allowed_days() == {0, 1, 2, 3, 4}  # type: ignore[attr-defined]

    def test_set_edge_param_unknown_edge_raises_key_error(self, edge_manager):
        with pytest.raises(KeyError):
            edge_manager.set_edge_param("nonexistent", "some_param", 1.0)

    def test_set_edge_param_unknown_param_raises_attribute_error(self, edge_manager):
        with pytest.raises(AttributeError, match="set_nonexistent"):
            edge_manager.set_edge_param("regime_filter", "nonexistent", 1.0)


# ---------------------------------------------------------------------------
# Shield function tests
# ---------------------------------------------------------------------------

class TestShieldFunction:
    def test_shield_clamp_below_floor(self, relaxer):
        result = relaxer.shield_clamp("adx_min", 5.0)
        assert result == SHIELD_BOUNDS["adx_min"].hard_floor

    def test_shield_clamp_above_ceiling(self, relaxer):
        result = relaxer.shield_clamp("adx_min", 100.0)
        assert result == SHIELD_BOUNDS["adx_min"].hard_ceiling

    def test_shield_clamp_within_bounds(self, relaxer):
        result = relaxer.shield_clamp("adx_min", 25.0)
        assert result == 25.0

    def test_shield_clamp_at_floor(self, relaxer):
        result = relaxer.shield_clamp("adx_min", 18.0)
        assert result == 18.0

    def test_shield_clamp_at_ceiling(self, relaxer):
        result = relaxer.shield_clamp("adx_min", 35.0)
        assert result == 35.0

    def test_shield_clamp_unknown_param_passthrough(self, relaxer):
        result = relaxer.shield_clamp("unknown_param", 99.9)
        assert result == 99.9

    def test_shield_step_respects_max_step_downward(self, relaxer):
        """Going from 28 to 18 in one step should be clamped to max_step=3."""
        result = relaxer.shield_step("adx_min", current=28.0, proposed=18.0)
        assert result == 25.0  # 28 - 3 = 25

    def test_shield_step_respects_max_step_upward(self, relaxer):
        result = relaxer.shield_step("adx_min", current=22.0, proposed=32.0)
        assert result == 25.0  # 22 + 3 = 25

    def test_shield_step_within_step_limit(self, relaxer):
        result = relaxer.shield_step("adx_min", current=28.0, proposed=26.0)
        assert result == 26.0  # 2 < max_step=3, no clamping needed

    def test_shield_step_clamps_to_hard_floor(self, relaxer):
        """Even a small step shouldn't go below hard_floor."""
        result = relaxer.shield_step("adx_min", current=19.0, proposed=17.0)
        assert result == 18.0  # floor wins

    def test_shield_step_min_score(self, relaxer):
        result = relaxer.shield_step("min_score", current=4.0, proposed=2.0)
        assert result == 3.0  # 4 - 1 = 3 (max_step=1)

    def test_shield_step_unknown_param_passthrough(self, relaxer):
        result = relaxer.shield_step("unknown", current=10.0, proposed=5.0)
        assert result == 5.0


# ---------------------------------------------------------------------------
# Relaxation tier tests
# ---------------------------------------------------------------------------

class TestRelaxationTiers:
    def test_initial_state_no_relaxation(self, relaxer):
        state = relaxer.get_state()
        assert state.current_tier == 0
        assert state.budget_used == 0.0
        assert not state.is_halted

    def test_tier1_day_of_week_applied(self, relaxer, edge_manager):
        result = relaxer.relax_next_tier()
        assert result is True
        dow = edge_manager.get_edge("day_of_week")
        # Should now include Mon and Fri
        assert 0 in dow.get_allowed_days()  # type: ignore[attr-defined]
        assert 4 in dow.get_allowed_days()  # type: ignore[attr-defined]

    def test_tier1_increments_current_tier(self, relaxer):
        relaxer.relax_next_tier()
        assert relaxer.get_state().current_tier == 1

    def test_tier1_costs_budget(self, relaxer):
        relaxer.relax_next_tier()
        assert relaxer.get_state().budget_used == pytest.approx(0.05)

    def test_tier2_time_of_day_applied(self, relaxer, edge_manager):
        relaxer.relax_next_tier()  # Tier 1
        # Bypass cooldown
        relaxer._state.bars_since_last_relax = 200
        relaxer.relax_next_tier()  # Tier 2
        tod = edge_manager.get_edge("time_of_day")
        start, end = tod.get_window()  # type: ignore[attr-defined]
        assert start == 360   # 06:00
        assert end == 1140    # 19:00

    def test_tier3_london_open_delay_disabled(self, relaxer, edge_manager):
        relaxer.relax_next_tier()  # Tier 1
        relaxer._state.bars_since_last_relax = 200
        relaxer.relax_next_tier()  # Tier 2
        relaxer._state.bars_since_last_relax = 200
        relaxer.relax_next_tier()  # Tier 3
        lod = edge_manager.get_edge("london_open_delay")
        assert not lod.enabled

    def test_tier4_regime_filter_adx_lowered(self, relaxer, edge_manager):
        for _ in range(3):
            relaxer.relax_next_tier()
            relaxer._state.bars_since_last_relax = 200
        relaxer.relax_next_tier()  # Tier 4
        rf = edge_manager.get_edge("regime_filter")
        # ADX should be at most 25 (28 - 3 = 25, clamped by velocity)
        assert rf.get_adx_min() <= 28.0  # type: ignore[attr-defined]
        assert rf.get_adx_min() >= 18.0  # type: ignore[attr-defined]

    def test_tier5_confluence_scoring_lowered(self, relaxer, edge_manager):
        for _ in range(4):
            relaxer.relax_next_tier()
            relaxer._state.bars_since_last_relax = 200
        relaxer.relax_next_tier()  # Tier 5
        cs = edge_manager.get_edge("confluence_scoring")
        assert cs.get_min_score() == 3  # type: ignore[attr-defined]

    def test_cannot_exceed_5_tiers(self, relaxer):
        for _ in range(5):
            relaxer.relax_next_tier()
            relaxer._state.bars_since_last_relax = 200
        result = relaxer.relax_next_tier()
        assert result is False

    def test_budget_accumulates_across_tiers(self, relaxer):
        expected_budget = 0.0
        for i, tier in enumerate(DEFAULT_TIERS):
            relaxer._state.bars_since_last_relax = 200
            relaxer.relax_next_tier()
            expected_budget += tier.budget_cost

        # Total = 0.05 + 0.08 + 0.04 + 0.08 + 0.05 = 0.30
        assert relaxer.get_state().budget_used == pytest.approx(expected_budget)


# ---------------------------------------------------------------------------
# Budget exhaustion tests
# ---------------------------------------------------------------------------

class TestBudgetExhaustion:
    def test_budget_halts_at_30_percent(self, edge_manager):
        """Relaxer should halt when total budget hits 30%."""
        relaxer = AdaptiveRelaxer(edge_manager)
        # Apply all tiers (total = 0.30 exactly)
        for _ in range(5):
            relaxer._state.bars_since_last_relax = 200
            relaxer.relax_next_tier()
        assert relaxer.is_budget_exhausted()

    def test_halted_relaxer_returns_false(self, edge_manager):
        relaxer = AdaptiveRelaxer(edge_manager)
        relaxer._state.is_halted = True
        result = relaxer.relax_next_tier()
        assert result is False

    def test_custom_low_budget_halts_early(self, edge_manager):
        """With max_budget=0.06 only tier1 (0.05) should apply."""
        relaxer = AdaptiveRelaxer(edge_manager, config={"relaxation": {"max_budget": 0.06}})
        result1 = relaxer.relax_next_tier()
        assert result1 is True
        relaxer._state.bars_since_last_relax = 200
        result2 = relaxer.relax_next_tier()
        assert result2 is False
        assert relaxer.is_budget_exhausted()


# ---------------------------------------------------------------------------
# Cooldown enforcement tests
# ---------------------------------------------------------------------------

class TestCooldownEnforcement:
    def test_first_relax_has_no_cooldown(self, relaxer):
        """First tier can be applied at bar 0."""
        result = relaxer.relax_next_tier()
        assert result is True

    def test_second_relax_blocked_before_cooldown(self, relaxer):
        relaxer.relax_next_tier()  # Tier 1
        relaxer._state.bars_since_last_relax = 100  # < 200
        result = relaxer.relax_next_tier()
        assert result is False

    def test_second_relax_allowed_after_cooldown(self, relaxer):
        relaxer.relax_next_tier()  # Tier 1
        relaxer._state.bars_since_last_relax = 200  # Exactly at cooldown
        result = relaxer.relax_next_tier()
        assert result is True

    def test_on_bar_increments_counter(self, relaxer):
        relaxer.relax_next_tier()
        relaxer.on_bar()
        relaxer.on_bar()
        assert relaxer.get_state().bars_since_last_relax == 2

    def test_relax_resets_bar_counter(self, relaxer):
        relaxer.relax_next_tier()
        relaxer._state.bars_since_last_relax = 200
        relaxer.relax_next_tier()
        assert relaxer.get_state().bars_since_last_relax == 0

    def test_custom_cooldown_from_config(self, edge_manager):
        relaxer = AdaptiveRelaxer(edge_manager, config={"relaxation": {"cooldown_bars": 50}})
        relaxer.relax_next_tier()
        relaxer._state.bars_since_last_relax = 50
        result = relaxer.relax_next_tier()
        assert result is True


# ---------------------------------------------------------------------------
# Tightening tests
# ---------------------------------------------------------------------------

class TestTightening:
    def test_tighten_all_reverts_to_base(self, relaxer, edge_manager):
        relaxer.relax_next_tier()  # Tier 1: day_of_week expanded
        relaxer.tighten_all()
        dow = edge_manager.get_edge("day_of_week")
        # Should be back to {1, 2, 3}
        assert dow.get_allowed_days() == {1, 2, 3}  # type: ignore[attr-defined]

    def test_tighten_all_resets_tier_counter(self, relaxer):
        relaxer.relax_next_tier()
        relaxer.tighten_all()
        assert relaxer.get_state().current_tier == 0

    def test_tighten_all_resets_budget(self, relaxer):
        relaxer.relax_next_tier()
        relaxer.tighten_all()
        assert relaxer.get_state().budget_used == 0.0

    def test_tighten_all_clears_halt(self, relaxer):
        relaxer._state.is_halted = True
        relaxer._state.current_tier = 3
        relaxer._state.budget_used = 0.25
        relaxer.tighten_all()
        assert not relaxer.get_state().is_halted

    def test_tighten_all_noop_when_no_relaxation(self, relaxer):
        # Should not raise
        relaxer.tighten_all()
        assert relaxer.get_state().current_tier == 0

    def test_tighten_one_tier_decrements_counter(self, relaxer):
        relaxer.relax_next_tier()
        relaxer.tighten_one_tier()
        assert relaxer.get_state().current_tier == 0

    def test_tighten_one_tier_refunds_budget(self, relaxer):
        relaxer.relax_next_tier()
        budget_after_relax = relaxer.get_state().budget_used
        relaxer.tighten_one_tier()
        assert relaxer.get_state().budget_used < budget_after_relax

    def test_tighten_one_tier_reverts_day_of_week(self, relaxer, edge_manager):
        relaxer.relax_next_tier()
        relaxer.tighten_one_tier()
        dow = edge_manager.get_edge("day_of_week")
        assert dow.get_allowed_days() == {1, 2, 3}  # type: ignore[attr-defined]

    def test_tighten_one_tier_reverts_london_open_delay(self, relaxer, edge_manager):
        # Apply 3 tiers to reach london_open_delay
        for _ in range(3):
            relaxer.relax_next_tier()
            relaxer._state.bars_since_last_relax = 200
        lod = edge_manager.get_edge("london_open_delay")
        assert not lod.enabled
        relaxer.tighten_one_tier()
        assert lod.enabled

    def test_tighten_one_tier_noop_at_base(self, relaxer):
        relaxer.tighten_one_tier()
        assert relaxer.get_state().current_tier == 0

    def test_tighten_clears_halt_flag(self, relaxer):
        relaxer._state.is_halted = True
        relaxer._state.current_tier = 1
        relaxer._state.budget_used = 0.30
        relaxer.tighten_one_tier()
        assert not relaxer.get_state().is_halted

    def test_tighten_all_2_tiers(self, relaxer, edge_manager):
        relaxer.relax_next_tier()  # Tier 1
        relaxer._state.bars_since_last_relax = 200
        relaxer.relax_next_tier()  # Tier 2
        relaxer.tighten_all()
        # time_of_day should be back to 08:00-17:00
        tod = edge_manager.get_edge("time_of_day")
        start, end = tod.get_window()  # type: ignore[attr-defined]
        assert start == 480   # 08:00
        assert end == 1020    # 17:00


# ---------------------------------------------------------------------------
# Consecutive loss / auto-tighten tests
# ---------------------------------------------------------------------------

class TestConsecutiveLossAutoTighten:
    def test_win_resets_consecutive_losses(self, relaxer):
        relaxer._state.consecutive_losses = 2
        relaxer.on_trade_closed(won=True)
        assert relaxer.get_state().consecutive_losses == 0

    def test_loss_increments_consecutive_losses(self, relaxer):
        relaxer.on_trade_closed(won=False)
        assert relaxer.get_state().consecutive_losses == 1

    def test_3_consecutive_losses_triggers_auto_tighten(self, relaxer, edge_manager):
        relaxer.relax_next_tier()  # Apply tier 1
        assert relaxer.get_state().current_tier == 1
        # 3 consecutive losses
        relaxer.on_trade_closed(won=False)
        relaxer.on_trade_closed(won=False)
        relaxer.on_trade_closed(won=False)
        # Should have auto-tightened (up to tighten_speed=3 tiers, but only 1 applied)
        assert relaxer.get_state().current_tier == 0

    def test_auto_tighten_resets_consecutive_loss_counter(self, relaxer):
        relaxer.relax_next_tier()
        relaxer.on_trade_closed(won=False)
        relaxer.on_trade_closed(won=False)
        relaxer.on_trade_closed(won=False)
        assert relaxer.get_state().consecutive_losses == 0

    def test_2_losses_do_not_trigger_auto_tighten(self, relaxer):
        relaxer.relax_next_tier()
        relaxer.on_trade_closed(won=False)
        relaxer.on_trade_closed(won=False)
        assert relaxer.get_state().current_tier == 1  # Still at tier 1

    def test_custom_loss_threshold(self, edge_manager):
        relaxer = AdaptiveRelaxer(
            edge_manager,
            config={"relaxation": {"consecutive_loss_threshold": 2}},
        )
        relaxer.relax_next_tier()
        relaxer.on_trade_closed(won=False)
        relaxer.on_trade_closed(won=False)
        assert relaxer.get_state().current_tier == 0  # Auto-tightened after 2 losses

    def test_auto_tighten_limited_to_applied_tiers(self, relaxer, edge_manager):
        """tighten_speed=3 but only 1 tier applied — should only revert 1."""
        relaxer.relax_next_tier()
        for _ in range(3):
            relaxer.on_trade_closed(won=False)
        assert relaxer.get_state().current_tier == 0
        # Should not go negative
        assert relaxer.get_state().budget_used >= 0.0


# ---------------------------------------------------------------------------
# Base config capture tests
# ---------------------------------------------------------------------------

class TestBaseConfigCapture:
    def test_base_config_captures_regime_adx(self, relaxer):
        assert relaxer._state.base_config["regime_filter__adx_min"] == 28.0

    def test_base_config_captures_confluence_score(self, relaxer):
        assert relaxer._state.base_config["confluence_scoring__min_score"] == 4

    def test_base_config_captures_day_of_week(self, relaxer):
        assert relaxer._state.base_config["day_of_week__allowed_days"] == {1, 2, 3}

    def test_base_config_captures_time_window(self, relaxer):
        assert relaxer._state.base_config["time_of_day__start_minutes"] == 480
        assert relaxer._state.base_config["time_of_day__end_minutes"] == 1020

    def test_base_config_captures_london_enabled(self, relaxer):
        assert relaxer._state.base_config["london_open_delay__enabled"] is True


# ---------------------------------------------------------------------------
# Tier 4 regime_filter velocity clamping test
# ---------------------------------------------------------------------------

class TestTier4VelocityClamping:
    def test_adx_step_clamped_to_3(self, edge_manager):
        """ADX goes from 28 to 22 but velocity limit is 3, so result is 25."""
        relaxer = AdaptiveRelaxer(edge_manager)
        for i in range(3):
            relaxer.relax_next_tier()
            relaxer._state.bars_since_last_relax = 200
        relaxer.relax_next_tier()  # Tier 4
        rf = edge_manager.get_edge("regime_filter")
        # The tier config says relaxed=22, but from 28, max step is 3 → 25
        assert rf.get_adx_min() == pytest.approx(25.0)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# get_state() returns copy test
# ---------------------------------------------------------------------------

class TestGetStateCopy:
    def test_get_state_returns_copy(self, relaxer):
        state = relaxer.get_state()
        state.current_tier = 99
        assert relaxer.get_state().current_tier != 99


# ---------------------------------------------------------------------------
# repr tests
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_includes_tier_info(self, relaxer):
        r = repr(relaxer)
        assert "tier=" in r
        assert "budget=" in r
        assert "halted=" in r

    def test_edge_manager_repr_still_works(self, edge_manager):
        r = repr(edge_manager)
        assert "EdgeManager" in r


# ---------------------------------------------------------------------------
# DEFAULT_TIERS integrity tests
# ---------------------------------------------------------------------------

class TestDefaultTiers:
    def test_five_default_tiers(self):
        assert len(DEFAULT_TIERS) == 5

    def test_total_budget_is_30_percent(self):
        total = sum(t.budget_cost for t in DEFAULT_TIERS)
        assert total == pytest.approx(0.30)

    def test_tier_names_unique(self):
        names = [t.name for t in DEFAULT_TIERS]
        assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# SHIELD_BOUNDS integrity tests
# ---------------------------------------------------------------------------

class TestShieldBoundsConstants:
    def test_adx_floor_less_than_ceiling(self):
        b = SHIELD_BOUNDS["adx_min"]
        assert b.hard_floor < b.hard_ceiling

    def test_min_score_floor_less_than_ceiling(self):
        b = SHIELD_BOUNDS["min_score"]
        assert b.hard_floor < b.hard_ceiling

    def test_time_start_bounds(self):
        b = SHIELD_BOUNDS["time_start"]
        assert b.hard_floor == 300   # 05:00
        assert b.hard_ceiling == 540 # 09:00

    def test_time_end_bounds(self):
        b = SHIELD_BOUNDS["time_end"]
        assert b.hard_floor == 960    # 16:00
        assert b.hard_ceiling == 1260 # 21:00
