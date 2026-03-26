"""
Edge optimization modules for XAU/USD Ichimoku trading agent.

Each module is an independently toggleable filter that refines entry quality
or manages open trade risk. Edges are chained by EdgeManager into a pipeline.
"""

from .base import EdgeContext, EdgeFilter, EdgeResult
from .manager import EdgeManager

from .time_of_day import TimeOfDayFilter
from .day_of_week import DayOfWeekFilter
from .london_open_delay import LondonOpenDelayFilter
from .candle_close_confirmation import CandleCloseConfirmationFilter
from .spread_filter import SpreadFilter
from .news_filter import NewsFilter
from .friday_close import FridayCloseFilter
from .regime_filter import RegimeFilter
from .time_stop import TimeStopFilter
from .bb_squeeze import BBSqueezeAmplifier
from .confluence_scoring import ConfluenceScoringFilter
from .equity_curve import EquityCurveFilter

__all__ = [
    "EdgeContext",
    "EdgeFilter",
    "EdgeResult",
    "EdgeManager",
    "TimeOfDayFilter",
    "DayOfWeekFilter",
    "LondonOpenDelayFilter",
    "CandleCloseConfirmationFilter",
    "SpreadFilter",
    "NewsFilter",
    "FridayCloseFilter",
    "RegimeFilter",
    "TimeStopFilter",
    "BBSqueezeAmplifier",
    "ConfluenceScoringFilter",
    "EquityCurveFilter",
]
