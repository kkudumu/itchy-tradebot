"""Profile-aware adapter for the discovery loop.

Lets the discovery orchestrator target either the forex/the5ers pct
objective or the futures/topstep dollar objective without carrying
the profile-specific logic inside orchestrator.py itself. The
adapter is a thin layer — it doesn't re-implement any tracker logic,
just selects which scoring function to apply to a backtest result
based on the active instrument's class.

Used by:
  * ``src/discovery/orchestrator.py`` — reads the active profile at
    the top of each discovery run and routes objectives through here.
  * ``src/discovery/codegen.py`` (if it exists) — generated strategy
    stubs call into :func:`adapt_codegen` for profile-specific unit
    conversion hints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from src.config.profile import InstrumentClass


@dataclass
class ProfileAdapter:
    """Profile metadata bundle used by the discovery loop.

    Call :func:`make_adapter` rather than constructing directly — the
    factory reads the active instrument config and wires up the
    right scoring function.
    """

    instrument_class: InstrumentClass
    tick_size: Optional[float] = None
    tick_value_usd: Optional[float] = None
    pip_size: Optional[float] = None
    objective_name: str = "pass_rate"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_adapter(instrument_config: Any) -> ProfileAdapter:
    """Build a :class:`ProfileAdapter` from an instrument config object.

    Accepts anything with ``class_``/``instrument_class`` plus the
    relevant tick/pip metadata — works with both
    :class:`InstrumentOverride` and simple dicts.
    """
    cls = getattr(instrument_config, "class_", None) or getattr(
        instrument_config, "instrument_class", None
    )
    if isinstance(cls, str):
        try:
            cls = InstrumentClass(cls)
        except ValueError:
            cls = InstrumentClass.FOREX
    if cls is None:
        cls = InstrumentClass.FOREX

    tick_size = _maybe_float(
        getattr(instrument_config, "tick_size", None)
        if hasattr(instrument_config, "tick_size")
        else None
    )
    tick_value_usd = _maybe_float(
        getattr(instrument_config, "tick_value_usd", None)
        or getattr(instrument_config, "tick_value", None)
    )
    pip_size = _maybe_float(getattr(instrument_config, "pip_size", None))

    objective_name = (
        "topstep_combine_pass_score"
        if cls == InstrumentClass.FUTURES
        else "pass_rate"
    )

    return ProfileAdapter(
        instrument_class=cls,
        tick_size=tick_size,
        tick_value_usd=tick_value_usd,
        pip_size=pip_size,
        objective_name=objective_name,
    )


def adapt_objective(
    profile: ProfileAdapter,
    base_objective: Optional[Callable[[Any], float]] = None,
) -> Callable[[Any], float]:
    """Return the scoring function matching the profile's objective.

    When *profile.objective_name* is ``"topstep_combine_pass_score"``,
    returns the topstep objective from :mod:`src.optimization.objectives`.
    Otherwise returns *base_objective* unchanged (or a no-op when
    ``None``).
    """
    if profile.objective_name == "topstep_combine_pass_score":
        from src.optimization.objectives import topstep_combine_pass_score

        return topstep_combine_pass_score
    if base_objective is not None:
        return base_objective
    return lambda result: 0.0


def adapt_codegen(profile: ProfileAdapter, strategy_template: str) -> str:
    """Rewrite a generated strategy template with profile-aware units.

    The forex path returns the template verbatim. The futures path
    substitutes ``pip_size`` references with tick-based equivalents
    so generated strategies run correctly on MGC without manual fixes.
    """
    if profile.instrument_class != InstrumentClass.FUTURES:
        return strategy_template

    template = strategy_template
    if profile.tick_size is not None:
        # Replace hardcoded `pip_value = 0.0001` (forex convention) with
        # `pip_value = <tick_size>` so the generated code aligns with
        # the instrument's native price scale.
        template = template.replace(
            "pip_value = 0.0001",
            f"pip_value = {profile.tick_size:g}",
        )
    return template


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
