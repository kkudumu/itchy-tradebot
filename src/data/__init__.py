"""
Historical data pipeline: download, normalize, and load OHLCV data.

Public surface
--------------
DukascopyDownloader  — fetch .bi5 tick data from Dukascopy servers
DataNormalizer       — aggregate ticks to 1-minute bars, detect gaps, validate
TimescaleLoader      — bulk insert into candles_1m and verify continuous aggregates
"""

from .downloader import DukascopyDownloader
from .loader import TimescaleLoader
from .normalizer import DataNormalizer

__all__ = [
    "DukascopyDownloader",
    "DataNormalizer",
    "TimescaleLoader",
]
