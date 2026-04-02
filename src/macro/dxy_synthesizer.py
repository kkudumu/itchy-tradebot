"""DXY (US Dollar Index) synthesis from 6 FX component pairs.

The US Dollar Index is a geometrically-weighted index of the dollar
against 6 major currencies. The official ICE formula:

    DXY = 50.14348112
        * EURUSD^(-0.576)
        * USDJPY^(0.136)
        * GBPUSD^(-0.119)
        * USDCAD^(0.091)
        * USDSEK^(0.042)
        * USDCHF^(0.036)

Data source: Dukascopy via the existing DukascopyDownloader, which already
supports configurable instrument and point_divisor (100_000.0 for FX pairs).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ICE DXY formula weights (exponents)
_DXY_CONSTANT = 50.14348112
_DXY_WEIGHTS: Dict[str, float] = {
    "EURUSD": -0.576,
    "USDJPY":  0.136,
    "GBPUSD": -0.119,
    "USDCAD":  0.091,
    "USDSEK":  0.042,
    "USDCHF":  0.036,
}

# All 6 pairs required for DXY computation
REQUIRED_FX_PAIRS = list(_DXY_WEIGHTS.keys())

# Dukascopy point divisor for standard FX pairs
FX_POINT_DIVISOR = 100_000.0

# Dukascopy point divisor for JPY pairs (3 decimal places)
JPY_POINT_DIVISOR = 1_000.0


def compute_dxy_from_rates(
    eurusd: float,
    usdjpy: float,
    gbpusd: float,
    usdcad: float,
    usdsek: float,
    usdchf: float,
) -> float:
    """Compute DXY from individual FX spot rates.

    Parameters
    ----------
    eurusd, usdjpy, gbpusd, usdcad, usdsek, usdchf:
        Spot rates for each component pair.

    Returns
    -------
    float: DXY index value.
    """
    return (
        _DXY_CONSTANT
        * (eurusd ** _DXY_WEIGHTS["EURUSD"])
        * (usdjpy ** _DXY_WEIGHTS["USDJPY"])
        * (gbpusd ** _DXY_WEIGHTS["GBPUSD"])
        * (usdcad ** _DXY_WEIGHTS["USDCAD"])
        * (usdsek ** _DXY_WEIGHTS["USDSEK"])
        * (usdchf ** _DXY_WEIGHTS["USDCHF"])
    )


def compute_dxy_series(
    fx_data: Dict[str, pd.DataFrame],
) -> pd.Series:
    """Compute a DXY time series from daily FX close prices.

    Parameters
    ----------
    fx_data:
        Dict mapping pair name (e.g. "EURUSD") to a DataFrame with
        a DatetimeIndex and a "close" column.

    Returns
    -------
    pd.Series with name="DXY" and the same DatetimeIndex.

    Raises
    ------
    ValueError: If any of the 6 required FX pairs are missing.
    """
    missing = [p for p in REQUIRED_FX_PAIRS if p not in fx_data]
    if missing:
        raise ValueError(f"Missing FX pairs for DXY computation: {missing}")

    # Align all pairs to the same date index via outer join + forward fill
    aligned = pd.DataFrame()
    for pair in REQUIRED_FX_PAIRS:
        df = fx_data[pair]
        aligned[pair] = df["close"]

    aligned = aligned.ffill().bfill()

    # Vectorised DXY computation
    dxy = np.full(len(aligned), _DXY_CONSTANT)
    for pair, weight in _DXY_WEIGHTS.items():
        dxy = dxy * (aligned[pair].values ** weight)

    result = pd.Series(dxy, index=aligned.index, name="DXY")
    logger.info(
        "Computed DXY series: %d days, range %.2f – %.2f",
        len(result), result.min(), result.max(),
    )
    return result


def download_fx_daily_closes(
    start_date: str,
    end_date: str,
    cache_dir: str = "data/fx_cache",
) -> Dict[str, pd.DataFrame]:
    """Download daily close prices for all 6 DXY component pairs.

    Uses the existing DukascopyDownloader for each pair, aggregates
    ticks to daily close prices, and caches results as parquet files.

    Parameters
    ----------
    start_date: ISO date string (e.g. "2024-01-01").
    end_date: ISO date string (e.g. "2025-12-31").
    cache_dir: Directory for parquet cache files.

    Returns
    -------
    Dict mapping pair name to DataFrame with DatetimeIndex and "close" column.
    """
    from datetime import datetime, timezone
    from pathlib import Path

    from src.data.downloader import DukascopyDownloader
    from src.data.normalizer import DataNormalizer

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    normalizer = DataNormalizer()
    fx_data: Dict[str, pd.DataFrame] = {}

    # Dukascopy symbols and their point divisors
    pair_config = {
        "EURUSD": FX_POINT_DIVISOR,
        "USDJPY": JPY_POINT_DIVISOR,
        "GBPUSD": FX_POINT_DIVISOR,
        "USDCAD": FX_POINT_DIVISOR,
        "USDSEK": FX_POINT_DIVISOR,
        "USDCHF": FX_POINT_DIVISOR,
    }

    start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)

    for pair, divisor in pair_config.items():
        cache_file = cache_path / f"{pair}_{start_date}_{end_date}_daily.parquet"

        if cache_file.exists():
            logger.info("Loading cached %s from %s", pair, cache_file)
            daily = pd.read_parquet(cache_file)
        else:
            logger.info("Downloading %s tick data from Dukascopy...", pair)
            dl = DukascopyDownloader(instrument=pair, point_divisor=divisor)
            ticks = dl.download_range(start_dt, end_dt)

            if ticks.empty:
                logger.warning("No tick data for %s — skipping", pair)
                continue

            ohlcv_1m = normalizer.ticks_to_1m_ohlcv(ticks)
            ohlcv_1m = ohlcv_1m.set_index("timestamp")

            # Resample to daily using last close of each day
            daily = ohlcv_1m["close"].resample("1D").last().dropna()
            daily = daily.to_frame(name="close")

            daily.to_parquet(cache_file)
            logger.info("Cached %s daily closes (%d days) to %s", pair, len(daily), cache_file)

        fx_data[pair] = daily

    return fx_data
