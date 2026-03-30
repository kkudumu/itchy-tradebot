"""
AdaptiveRelaxer — bounded parameter adjustment engine.

Relaxes edge filters in tiered order when trade drought is detected.
Re-tightens when relaxed signals produce consecutive losses.

Design principles:
- Shield function: hard floors/ceilings + velocity limits on all params
- Relaxation budget: 30% max cumulative relaxation
- Asymmetric rates: relax one tier at a time (200-bar cooldown),
  tighten 2-3x faster (multiple tiers at once on loss signal)
- 5-tier ladder from softest (day_of_week) to hardest (confluence_scoring)
- Core filters (4H cloud, 1H confirmation, 15M signal) are NEVER relaxed
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

from src.edges.manager import EdgeManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shield bounds — hard limits and velocity caps for each tunable parameter
# ---------------------------------------------------------------------------

@dataclass
class ShieldBounds:
    """Hard limits and max step size for a single tunable parameter.

    Attributes
    ----------
    param_name:
        Name of the parameter (for logging and lookup).
    hard_floor:
        Absolute minimum — value will never go below this.
    hard_ceiling:
        Absolute maximum — value will never go above this.
    max_step_size:
        Maximum change allowed in a single relaxation or tighten step.
    """

    param_name: str
    hard_floor: float
    hard_ceiling: float
    max_step_size: float


# Default shield bounds table
SHIELD_BOUNDS: dict[str, ShieldBounds] = {
    "adx_min": ShieldBounds(
        param_name="adx_min",
        hard_floor=18.0,
        hard_ceiling=35.0,
        max_step_size=3.0,
    ),
    "min_score": ShieldBounds(
        param_name="min_score",
        hard_floor=2.0,
        hard_ceiling=6.0,
        max_step_size=1.0,
    ),
    "time_start": ShieldBounds(
        param_name="time_start",
        hard_floor=300.0,   # 05:00 UTC
        hard_ceiling=540.0,  # 09:00 UTC
        max_step_size=60.0,
    ),
    "time_end": ShieldBounds(
        param_name="time_end",
        hard_floor=960.0,    # 16:00 UTC
        hard_ceiling=1260.0, # 21:00 UTC
        max_step_size=60.0,
    ),
}


# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------

@dataclass
class TierConfig:
    """Configuration for a single relaxation tier.

    Attributes
    ----------
    name:
        Human-readable tier name.
    edge:
        Edge name in EdgeManager (e.g. 'day_of_week').
    action:
        How to apply this tier: 'set_allowed_days', 'set_window',
        'disable', 'set_adx_min', 'set_min_score'.
    relaxed_value:
        Value or values to apply when relaxing (None for 'disable').
    budget_cost:
        Fraction of total budget this tier consumes (0.0–1.0).
    """

    name: str
    edge: str
    action: str
    relaxed_value: Any
    budget_cost: float


# Default 5-tier ladder (softest → hardest)
DEFAULT_TIERS: list[TierConfig] = [
    TierConfig(
        name="day_of_week",
        edge="day_of_week",
        action="set_allowed_days",
        relaxed_value={0, 1, 2, 3, 4},  # Mon–Fri
        budget_cost=0.05,
    ),
    TierConfig(
        name="time_of_day",
        edge="time_of_day",
        action="set_window",
        relaxed_value=(360, 1140),  # 06:00–19:00 UTC
        budget_cost=0.08,
    ),
    TierConfig(
        name="london_open_delay",
        edge="london_open_delay",
        action="disable",
        relaxed_value=None,
        budget_cost=0.04,
    ),
    TierConfig(
        name="regime_filter",
        edge="regime_filter",
        action="set_adx_min",
        relaxed_value=22.0,
        budget_cost=0.08,
    ),
    TierConfig(
        name="confluence_scoring",
        edge="confluence_scoring",
        action="set_min_score",
        relaxed_value=3,
        budget_cost=0.05,
    ),
]


# ---------------------------------------------------------------------------
# Relaxation state
# ---------------------------------------------------------------------------

@dataclass
class RelaxationState:
    """Snapshot of the current relaxation state.

    Attributes
    ----------
    current_tier:
        0 = no relaxation applied; 1–5 = tiers applied so far.
    budget_used:
        Cumulative budget consumed as a fraction (0.0–1.0, max 0.30).
    bars_since_last_relax:
        Bars elapsed since the most recent relaxation step.
    consecutive_losses:
        Number of consecutive losing trades since last tighten.
    base_config:
        Snapshot of original parameter values — used to revert on tighten_all().
    is_halted:
        True when budget has hit MAX_BUDGET and no further relaxation is allowed.
    """

    current_tier: int = 0
    budget_used: float = 0.0
    bars_since_last_relax: int = 0
    consecutive_losses: int = 0
    base_config: dict = field(default_factory=dict)
    is_halted: bool = False


# ---------------------------------------------------------------------------
# AdaptiveRelaxer
# ---------------------------------------------------------------------------

class AdaptiveRelaxer:
    """Bounded parameter adjustment engine for drought-driven relaxation.

    Relaxes edge filters one tier at a time when trade drought is detected,
    re-tightens when relaxed signals produce losses. All adjustments are
    constrained by the shield function (hard floors + velocity limits) and
    a cumulative 30% budget cap.

    Parameters
    ----------
    edge_manager:
        Live EdgeManager instance whose parameters will be adjusted.
    config:
        Optional config dict (from health_monitor.yaml relaxation section).
        Falls back to built-in defaults when None.

    Examples
    --------
    >>> relaxer = AdaptiveRelaxer(edge_manager)
    >>> relaxer.relax_next_tier()   # Called by DroughtDetector
    True
    >>> relaxer.on_trade_closed(won=False)
    >>> relaxer.on_bar()
    """

    COOLDOWN_BARS = 200
    MAX_BUDGET = 0.30
    CONSECUTIVE_LOSS_THRESHOLD = 3

    def __init__(
        self,
        edge_manager: EdgeManager,
        config: dict | None = None,
    ) -> None:
        self._em = edge_manager
        self._config = config or {}

        relaxation_cfg = self._config.get("relaxation", {})

        self._cooldown_bars: int = int(
            relaxation_cfg.get("cooldown_bars", self.COOLDOWN_BARS)
        )
        self._max_budget: float = float(
            relaxation_cfg.get("max_budget", self.MAX_BUDGET)
        )
        self._loss_threshold: int = int(
            relaxation_cfg.get("consecutive_loss_threshold", self.CONSECUTIVE_LOSS_THRESHOLD)
        )
        self._tighten_speed: int = int(
            relaxation_cfg.get("tighten_speed_multiplier", 3)
        )

        # Build tiers from config or use defaults
        self._tiers: list[TierConfig] = self._build_tiers(
            relaxation_cfg.get("tiers", None)
        )

        # Build shield bounds from config or use defaults
        self._shield: dict[str, ShieldBounds] = self._build_shield(
            relaxation_cfg.get("shield", None)
        )

        # Capture base config before any relaxation
        self._state = RelaxationState(
            base_config=self._capture_base_config(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def relax_next_tier(self) -> bool:
        """Apply the next relaxation tier if conditions permit.

        Conditions checked:
        - Not halted (budget exhausted)
        - All tiers not already applied
        - Cooldown has elapsed since last relaxation

        Returns
        -------
        bool
            True if a tier was applied; False if blocked by cooldown,
            budget exhaustion, or all tiers already active.
        """
        if self._state.is_halted:
            logger.debug("AdaptiveRelaxer: halted — budget exhausted")
            return False

        if self._state.current_tier >= len(self._tiers):
            logger.debug("AdaptiveRelaxer: all tiers already applied")
            return False

        if (
            self._state.current_tier > 0
            and self._state.bars_since_last_relax < self._cooldown_bars
        ):
            logger.debug(
                "AdaptiveRelaxer: cooldown active (%d/%d bars)",
                self._state.bars_since_last_relax,
                self._cooldown_bars,
            )
            return False

        next_tier_idx = self._state.current_tier
        tier = self._tiers[next_tier_idx]

        # Check budget
        new_budget = self._state.budget_used + tier.budget_cost
        if new_budget > self._max_budget:
            logger.warning(
                "AdaptiveRelaxer: tier '%s' would exceed budget (%.1f%% > %.1f%%)",
                tier.name,
                new_budget * 100,
                self._max_budget * 100,
            )
            self._state.is_halted = True
            return False

        # Apply the tier
        self._apply_tier(tier)
        self._state.current_tier += 1
        self._state.budget_used = new_budget
        self._state.bars_since_last_relax = 0
        self._state.consecutive_losses = 0

        if self._state.budget_used >= self._max_budget:
            self._state.is_halted = True

        logger.info(
            "AdaptiveRelaxer: applied tier %d '%s' (budget %.1f%%)",
            self._state.current_tier,
            tier.name,
            self._state.budget_used * 100,
        )
        return True

    def tighten_all(self) -> None:
        """Revert all parameters to base config immediately.

        Resets budget, tier counter, and consecutive loss counter.
        """
        if self._state.current_tier == 0:
            return

        # Revert each applied tier in reverse order
        for i in range(self._state.current_tier - 1, -1, -1):
            self._revert_tier(self._tiers[i])

        old_tier = self._state.current_tier
        self._state.current_tier = 0
        self._state.budget_used = 0.0
        self._state.consecutive_losses = 0
        self._state.is_halted = False

        logger.info(
            "AdaptiveRelaxer: tightened all — reverted %d tiers to base config",
            old_tier,
        )

    def tighten_one_tier(self) -> None:
        """Revert the most recently applied tier.

        Tightening is faster than relaxation (no cooldown enforced).
        """
        if self._state.current_tier == 0:
            return

        tier_idx = self._state.current_tier - 1
        tier = self._tiers[tier_idx]
        self._revert_tier(tier)

        self._state.current_tier -= 1
        self._state.budget_used = max(
            0.0, self._state.budget_used - tier.budget_cost
        )
        # Re-enable tightening if we were halted
        if self._state.is_halted and self._state.budget_used < self._max_budget:
            self._state.is_halted = False

        logger.info(
            "AdaptiveRelaxer: tightened one tier — reverted '%s' (budget %.1f%%)",
            tier.name,
            self._state.budget_used * 100,
        )

    def on_trade_closed(self, won: bool) -> None:
        """Record a trade result and auto-tighten on consecutive losses.

        Parameters
        ----------
        won:
            True if the trade was profitable; False for a loss.
        """
        if won:
            self._state.consecutive_losses = 0
        else:
            self._state.consecutive_losses += 1
            if self._state.consecutive_losses >= self._loss_threshold:
                tiers_to_revert = min(
                    self._tighten_speed,
                    self._state.current_tier,
                )
                logger.warning(
                    "AdaptiveRelaxer: %d consecutive losses — auto-tightening %d tier(s)",
                    self._state.consecutive_losses,
                    tiers_to_revert,
                )
                for _ in range(tiers_to_revert):
                    if self._state.current_tier > 0:
                        self.tighten_one_tier()
                self._state.consecutive_losses = 0

    def on_bar(self) -> None:
        """Advance the bar counter. Call once per closed bar."""
        self._state.bars_since_last_relax += 1

    def get_state(self) -> RelaxationState:
        """Return a copy of the current relaxation state."""
        return copy.copy(self._state)

    def is_budget_exhausted(self) -> bool:
        """True when the relaxation budget has reached its maximum."""
        return self._state.is_halted

    def capture_current_config(self) -> dict:
        """Snapshot current EdgeManager parameter values.

        Returns the same dict shape as ``RelaxationState.base_config`` but
        reflecting the *current* (possibly relaxed) parameter values.
        """
        return self._capture_base_config()

    def shield_clamp(self, param: str, value: float) -> float:
        """Clamp a parameter value to its shield bounds.

        If the parameter has no registered bounds, returns ``value`` unchanged.

        Parameters
        ----------
        param:
            Parameter name (key in SHIELD_BOUNDS / config shield section).
        value:
            Proposed new value.

        Returns
        -------
        float
            Value clamped to [hard_floor, hard_ceiling].
        """
        bounds = self._shield.get(param)
        if bounds is None:
            return value
        return max(bounds.hard_floor, min(bounds.hard_ceiling, float(value)))

    def shield_step(self, param: str, current: float, proposed: float) -> float:
        """Clamp a parameter change to the shield velocity limit.

        Applies both the velocity limit (max_step_size) and the hard floor/
        ceiling in a single call.

        Parameters
        ----------
        param:
            Parameter name.
        current:
            Current parameter value.
        proposed:
            Desired new value before velocity clamping.

        Returns
        -------
        float
            New value respecting both step limit and hard bounds.
        """
        bounds = self._shield.get(param)
        if bounds is None:
            return proposed

        delta = proposed - current
        clamped_delta = max(-bounds.max_step_size, min(bounds.max_step_size, delta))
        new_value = current + clamped_delta
        return max(bounds.hard_floor, min(bounds.hard_ceiling, new_value))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_tier(self, tier: TierConfig) -> None:
        """Apply a tier's relaxation to the edge manager."""
        action = tier.action
        edge_name = tier.edge

        if action == "disable":
            self._em.toggle_edge(edge_name, enabled=False)

        elif action == "set_allowed_days":
            days = set(tier.relaxed_value)
            self._em.set_edge_param(edge_name, "allowed_days", days)

        elif action == "set_window":
            start, end = tier.relaxed_value
            edge = self._em.get_edge(edge_name)
            edge.set_window(start, end)  # type: ignore[attr-defined]

        elif action == "set_adx_min":
            current = self._em.get_edge(edge_name).get_adx_min()  # type: ignore[attr-defined]
            clamped = self.shield_step("adx_min", current, float(tier.relaxed_value))
            self._em.set_edge_param(edge_name, "adx_min", clamped)

        elif action == "set_min_score":
            current = float(self._em.get_edge(edge_name).get_min_score())  # type: ignore[attr-defined]
            clamped = self.shield_step("min_score", current, float(tier.relaxed_value))
            self._em.set_edge_param(edge_name, "min_score", int(clamped))

        else:
            logger.warning("AdaptiveRelaxer: unknown action '%s' for tier '%s'", action, tier.name)

    def _revert_tier(self, tier: TierConfig) -> None:
        """Revert a tier by restoring base config values."""
        action = tier.action
        edge_name = tier.edge
        base = self._state.base_config

        if action == "disable":
            # Restore original enabled state
            original_enabled = base.get(f"{edge_name}__enabled", True)
            self._em.toggle_edge(edge_name, enabled=original_enabled)

        elif action == "set_allowed_days":
            original_days = base.get(f"{edge_name}__allowed_days", {1, 2, 3})
            self._em.set_edge_param(edge_name, "allowed_days", set(original_days))

        elif action == "set_window":
            original_start = base.get(f"{edge_name}__start_minutes", 480)  # 08:00
            original_end = base.get(f"{edge_name}__end_minutes", 1020)     # 17:00
            edge = self._em.get_edge(edge_name)
            edge.set_window(original_start, original_end)  # type: ignore[attr-defined]

        elif action == "set_adx_min":
            original = base.get(f"{edge_name}__adx_min", 28.0)
            self._em.set_edge_param(edge_name, "adx_min", original)

        elif action == "set_min_score":
            original = base.get(f"{edge_name}__min_score", 4)
            self._em.set_edge_param(edge_name, "min_score", original)

        else:
            logger.warning(
                "AdaptiveRelaxer: unknown action '%s' when reverting tier '%s'",
                action, tier.name,
            )

    def _capture_base_config(self) -> dict:
        """Snapshot current parameter values from all relaxable edges.

        These values are used as the revert targets for tighten operations.
        """
        base: dict[str, Any] = {}

        try:
            dow = self._em.get_edge("day_of_week")
            base["day_of_week__allowed_days"] = set(
                dow.get_allowed_days()  # type: ignore[attr-defined]
            )
        except (KeyError, AttributeError):
            base["day_of_week__allowed_days"] = {1, 2, 3}

        try:
            tod = self._em.get_edge("time_of_day")
            start, end = tod.get_window()  # type: ignore[attr-defined]
            base["time_of_day__start_minutes"] = start
            base["time_of_day__end_minutes"] = end
        except (KeyError, AttributeError):
            base["time_of_day__start_minutes"] = 480   # 08:00
            base["time_of_day__end_minutes"] = 1020    # 17:00

        try:
            lod = self._em.get_edge("london_open_delay")
            base["london_open_delay__enabled"] = lod.enabled
        except KeyError:
            base["london_open_delay__enabled"] = True

        try:
            rf = self._em.get_edge("regime_filter")
            base["regime_filter__adx_min"] = rf.get_adx_min()  # type: ignore[attr-defined]
        except (KeyError, AttributeError):
            base["regime_filter__adx_min"] = 28.0

        try:
            cs = self._em.get_edge("confluence_scoring")
            base["confluence_scoring__min_score"] = cs.get_min_score()  # type: ignore[attr-defined]
        except (KeyError, AttributeError):
            base["confluence_scoring__min_score"] = 4

        return base

    def _build_tiers(self, tiers_cfg: list[dict] | None) -> list[TierConfig]:
        """Build tier list from config or fall back to defaults."""
        if tiers_cfg is None:
            return list(DEFAULT_TIERS)

        tiers = []
        for t in tiers_cfg:
            rv = t.get("relaxed_value")
            # Convert list to set for allowed_days, tuple for window
            if t.get("action") == "set_allowed_days" and isinstance(rv, list):
                rv = set(rv)
            elif t.get("action") == "set_window" and isinstance(rv, list):
                rv = tuple(rv)

            tiers.append(
                TierConfig(
                    name=t["name"],
                    edge=t["edge"],
                    action=t["action"],
                    relaxed_value=rv,
                    budget_cost=float(t.get("budget_cost", 0.05)),
                )
            )
        return tiers

    def _build_shield(self, shield_cfg: dict | None) -> dict[str, ShieldBounds]:
        """Build shield bounds from config or fall back to defaults."""
        if shield_cfg is None:
            return dict(SHIELD_BOUNDS)

        bounds = {}
        for param_name, cfg in shield_cfg.items():
            bounds[param_name] = ShieldBounds(
                param_name=param_name,
                hard_floor=float(cfg.get("hard_floor", 0.0)),
                hard_ceiling=float(cfg.get("hard_ceiling", 9999.0)),
                max_step_size=float(cfg.get("max_step", 1.0)),
            )
        return bounds

    def __repr__(self) -> str:
        state = self._state
        return (
            f"AdaptiveRelaxer("
            f"tier={state.current_tier}/{len(self._tiers)}, "
            f"budget={state.budget_used:.1%}, "
            f"halted={state.is_halted})"
        )
