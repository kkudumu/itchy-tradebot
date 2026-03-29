"""
Strategy health monitoring package.

Provides adaptive parameter relaxation when trade drought is detected,
with shield function (hard floors, velocity limits) and budget tracking.
"""

from .adaptive_relaxer import AdaptiveRelaxer, RelaxationState, ShieldBounds

__all__ = ["AdaptiveRelaxer", "RelaxationState", "ShieldBounds"]
