"""
src/monitoring — Strategy health monitoring components.

Exports
-------
SignalFunnelTracker
    Per-filter pass/fail/rate metrics tracker.
AdaptiveRelaxer
    Bounded parameter adjustment engine with shield function.
RegimeDetector
    3-state HMM-based market regime classification.
"""

from .funnel_tracker import SignalFunnelTracker, FilterEvent
from .adaptive_relaxer import AdaptiveRelaxer, RelaxationState, ShieldBounds
from .regime_detector import (
    DiagnosisResult,
    MarketRegime,
    RegimeDetector,
    RegimeState,
)

__all__ = [
    "SignalFunnelTracker",
    "FilterEvent",
    "AdaptiveRelaxer",
    "RelaxationState",
    "ShieldBounds",
    "DiagnosisResult",
    "MarketRegime",
    "RegimeDetector",
    "RegimeState",
]
