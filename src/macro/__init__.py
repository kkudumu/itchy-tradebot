"""Macro regime detection and multi-asset correlation.

Synthesizes DXY from FX components, fetches SPX/US10Y,
classifies daily regimes, and provides event proximity filtering.
"""

from src.macro.dxy_synthesizer import compute_dxy_from_rates, compute_dxy_series
from src.macro.regime_classifier import RegimeClassifier, RegimeLabel
from src.macro.econ_calendar import EconCalendar
from src.macro.event_proximity import EventProximityFilter
from src.macro.trade_tagger import TradeTagger
from src.macro.market_data import build_macro_panel, fetch_daily_macro

__all__ = [
    "compute_dxy_from_rates",
    "compute_dxy_series",
    "RegimeClassifier",
    "RegimeLabel",
    "EconCalendar",
    "EventProximityFilter",
    "TradeTagger",
    "build_macro_panel",
    "fetch_daily_macro",
]
