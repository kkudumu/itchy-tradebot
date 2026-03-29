"""
src/monitoring — Strategy Health Monitor package.

Exposes the primary monitoring components:
  - RegimeDetector: 3-state HMM-based market regime classification
"""

from .regime_detector import (
    DiagnosisResult,
    MarketRegime,
    RegimeDetector,
    RegimeState,
)

__all__ = [
    "DiagnosisResult",
    "MarketRegime",
    "RegimeDetector",
    "RegimeState",
]
