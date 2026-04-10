"""Instrument-class profile abstraction.

A *profile* is a bundle of class-wide defaults (commission, slippage, session
windows, daily reset rules) that apply to every instrument of a given class
(forex or futures). Per-instrument settings in ``config/instruments.yaml``
override profile defaults.

The profile drives:
  * Commission/slippage cost model selection (lots vs contracts)
  * Daily reset hour + timezone for prop firm day rollover
  * Session open/close times in local time
  * The ``price_distance(instrument, pips)`` helper used by strategies to
    convert pip-denominated thresholds to real price deltas without having
    to hardcode whether they're running on forex or futures data.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class InstrumentClass(str, Enum):
    """Asset class that drives profile selection.

    ``str`` subclass so YAML can deserialize directly from string values
    like ``"forex"`` or ``"futures"``.
    """

    FOREX = "forex"
    FUTURES = "futures"


class ProfileConfig(BaseModel):
    """Class-wide defaults loaded from ``config/profiles/<class>.yaml``.

    All fields have defaults so a missing profile file falls back to
    a forex-like profile. Fields are a superset of what either class
    needs; irrelevant fields on the opposite class are simply unused.

    ``strategy_overrides`` carries per-strategy parameter patches that
    are merged on top of the base ``config/strategy.yaml`` values when
    this profile is active. This is how forex and futures get
    independent strategy tuning without duplicating the entire
    strategy config file.
    """

    instrument_class: InstrumentClass
    # Commission model — forex uses per-lot, futures uses per-contract round trip
    default_commission_per_lot: float = 0.0
    default_commission_per_contract_round_trip: float = 0.0
    # Slippage model
    default_slippage_pips: float = 0.0
    default_slippage_ticks: int = 0
    # Day-boundary settings (prop firm daily reset)
    daily_reset_hour: int = 0
    daily_reset_tz: str = "UTC"
    # Session window in local time (daily_reset_tz)
    session_open_local: str = "00:00"
    session_close_local: str = "23:59"
    # Daily maintenance window minutes (e.g. futures 1h daily halt at 16:00 CT)
    maintenance_window_minutes: int = 0
    # Per-strategy parameter overrides — merged on top of the base
    # strategy.yaml values when this profile is the active one.
    # Example: {"sss": {"min_swing_pips": 2.0}, "asian_breakout": {"min_range_pips": 5}}
    strategy_overrides: dict[str, Any] = Field(default_factory=dict)


_DEFAULT_PROFILE_DIR = Path(__file__).resolve().parents[2] / "config" / "profiles"


def load_profile(
    instrument_class: InstrumentClass | str,
    profile_dir: Path | None = None,
) -> ProfileConfig:
    """Load the profile YAML for the given instrument class.

    Returns defaults keyed to the class if the YAML file is missing,
    so unit tests and fresh workspaces don't need to ship profile files.
    """
    if isinstance(instrument_class, str):
        instrument_class = InstrumentClass(instrument_class)

    directory = profile_dir or _DEFAULT_PROFILE_DIR
    path = directory / f"{instrument_class.value}.yaml"
    if not path.exists():
        return ProfileConfig(instrument_class=instrument_class)

    with path.open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    # Ensure the class from disk matches what was requested
    data.setdefault("instrument_class", instrument_class.value)
    return ProfileConfig.model_validate(data)


def price_distance(instrument: Any, pips: float) -> float:
    """Convert a pip count to a real price delta for *instrument*.

    Conventions:
      * **Forex**: one "pip" = ``pip_size`` price units. For XAU/USD spot
        with a 0.01 pip size, ``price_distance(inst, 10)`` = 0.10.
      * **Futures**: one "pip" is interpreted as 10 ticks, so a strategy
        threshold of ``0.5 pips`` is preserved across the forex→futures
        port. For MGC with a 0.10 tick size, ``price_distance(inst, 0.5)``
        = 0.50 (half a dollar of price movement).

    The instrument parameter can be anything with ``class_``/``instrument_class``
    plus ``tick_size``/``pip_size`` attributes — it doesn't have to be a
    specific Pydantic model, which keeps this helper decoupled from the
    config module's concrete types.
    """
    cls = getattr(instrument, "class_", None) or getattr(
        instrument, "instrument_class", None
    )
    if isinstance(cls, str):
        try:
            cls = InstrumentClass(cls)
        except ValueError:
            cls = None

    if cls == InstrumentClass.FUTURES:
        tick_size = _float_or_none(getattr(instrument, "tick_size", None))
        if tick_size is None or tick_size <= 0:
            raise ValueError(
                f"price_distance: futures instrument requires tick_size > 0 (got {tick_size!r})"
            )
        # One "pip" is 10 ticks for futures so that forex-tuned strategy
        # thresholds port over without re-scaling.
        return tick_size * float(pips) * 10.0

    # Default: forex
    pip_size = _float_or_none(getattr(instrument, "pip_size", None))
    if pip_size is None:
        # Fall back to tick_size if pip_size wasn't set explicitly
        pip_size = _float_or_none(getattr(instrument, "tick_size", None))
    if pip_size is None or pip_size <= 0:
        raise ValueError(
            f"price_distance: forex instrument requires pip_size > 0 (got {pip_size!r})"
        )
    return pip_size * float(pips)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
