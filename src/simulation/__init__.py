"""
Monte Carlo simulation package for prop firm challenge modelling.

Provides fat-tailed trade distributions, block bootstrapping, and
10,000-iteration pass-rate estimation with daily drawdown enforcement.
"""

from src.simulation.monte_carlo import (
    ChallengeOutcome,
    MCResult,
    MonteCarloSimulator,
)
from src.simulation.distributions import BlockBootstrapper, TradeDistribution
from src.simulation.visualizer import MCVisualizer

__all__ = [
    "MonteCarloSimulator",
    "MCResult",
    "ChallengeOutcome",
    "TradeDistribution",
    "BlockBootstrapper",
    "MCVisualizer",
]
