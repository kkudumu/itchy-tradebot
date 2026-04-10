"""Mega-Vision trading agent package.

Implements the Claude Agent SDK based strategy selector that runs on
top of the deterministic signal blender. See
``docs/mega_vision_design.md`` for the architectural rationale and
``docs/superpowers/plans/2026-04-09-futures-profile-and-topstepx.md``
(Tasks 22-27) for the implementation plan.
"""

from .trade_memory import TradeMemory, TradeRecord
from .performance_buckets import PerformanceBuckets

__all__ = [
    "TradeMemory",
    "TradeRecord",
    "PerformanceBuckets",
]
