"""Macro regime detection and multi-asset correlation.

Synthesizes DXY from FX components, fetches SPX/US10Y,
classifies daily regimes, and provides event proximity filtering.
"""

from src.macro.dxy_synthesizer import compute_dxy_from_rates, compute_dxy_series

__all__ = [
    "compute_dxy_from_rates",
    "compute_dxy_series",
]
