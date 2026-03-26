"""Optuna-based optimization pipeline for the XAU/USD Ichimoku strategy.

Modules
-------
objectives
    Custom Optuna objective functions for single- and multi-objective
    optimization aligned with The5ers prop firm challenge constraints.
optuna_runner
    Main optimizer entry point — creates Optuna studies, applies TPE/
    NSGA-II samplers, attaches MedianPruner, and integrates PostgreSQL
    storage backends.
walk_forward
    Rolling walk-forward analysis: 12-month in-sample optimization
    followed by 3-month out-of-sample validation per window.
overfit_detector
    Statistical overfitting checks — Deflated Sharpe Ratio (Bailey &
    Lopez de Prado 2014), Walk-Forward Efficiency, and plateau test.
edge_tester
    Edge isolation framework — measures each edge's marginal contribution
    to out-of-sample performance and flags non-contributing edges for
    removal.
"""

from src.optimization.objectives import MultiObjective, PropFirmObjective
from src.optimization.optuna_runner import OptunaOptimizer
from src.optimization.overfit_detector import OverfitDetector, OverfitReport
from src.optimization.walk_forward import WFResult, WalkForwardAnalyzer
from src.optimization.edge_tester import EdgeImpact, EdgeIsolationTester

__all__ = [
    "PropFirmObjective",
    "MultiObjective",
    "OptunaOptimizer",
    "WalkForwardAnalyzer",
    "WFResult",
    "OverfitDetector",
    "OverfitReport",
    "EdgeIsolationTester",
    "EdgeImpact",
]
