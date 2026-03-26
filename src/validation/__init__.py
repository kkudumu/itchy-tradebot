"""Pre-challenge validation pipeline for the XAU/USD Ichimoku trading strategy.

This package provides a full go/no-go assessment before submitting to a prop
firm challenge.  The pipeline flows as follows:

    WalkForwardAnalyzer → OOS metrics → 25% haircut → ThresholdChecker
    → MonteCarloSimulator → ValidationReportGenerator

Usage::

    from src.validation.go_nogo import GoNoGoValidator

    validator = GoNoGoValidator(data=df)
    result = validator.run_full_validation()
    print(result.final_verdict)  # 'GO', 'NO-GO', or 'BORDERLINE'
"""

from src.validation.go_nogo import FullValidationResult, GoNoGoValidator
from src.validation.threshold_checker import (
    ThresholdChecker,
    ThresholdResult,
    ValidationResult,
)
from src.validation.report import ValidationReportGenerator

__all__ = [
    "GoNoGoValidator",
    "FullValidationResult",
    "ThresholdChecker",
    "ThresholdResult",
    "ValidationResult",
    "ValidationReportGenerator",
]
