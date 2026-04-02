"""Fetch daily macro data (SPX, US10Y) via yfinance.

Provides build_macro_panel() which merges DXY, SPX, and US10Y
daily closes and their percentage changes into a single DataFrame
for regime classification.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Yahoo Finance tickers for macro instruments
_TICKERS = {
    "spx": "^GSPC",       # S&P 500
    "us10y": "^TNX",       # US 10-Year Treasury Yield
}


def fetch_daily_macro(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch daily SPX and US10Y data from yfinance.

    Parameters
    ----------
    start_date: ISO date string (e.g. "2024-01-01").
    end_date: ISO date string (e.g. "2025-12-31").

    Returns
    -------
    DataFrame with DatetimeIndex and columns:
        spx_close, us10y_close, spx_pct_change, us10y_pct_change
    """
    if yf is None:
        logger.error("yfinance not installed — pip install yfinance")
        return pd.DataFrame()

    tickers = list(_TICKERS.values())

    try:
        raw = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        logger.error("yfinance download failed: %s", exc)
        return pd.DataFrame()

    if raw.empty:
        logger.warning("No data returned from yfinance for %s", tickers)
        return pd.DataFrame()

    # yfinance returns MultiIndex columns: (price_field, ticker)
    result = pd.DataFrame(index=raw.index)

    for name, ticker in _TICKERS.items():
        try:
            result[f"{name}_close"] = raw[("Close", ticker)]
        except KeyError:
            logger.warning("Missing data for %s (%s)", name, ticker)
            result[f"{name}_close"] = float("nan")

    # Compute daily percentage changes
    for name in _TICKERS:
        result[f"{name}_pct_change"] = result[f"{name}_close"].pct_change()

    # Ensure UTC timezone on index
    if result.index.tz is None:
        result.index = result.index.tz_localize("UTC")
    else:
        result.index = result.index.tz_convert("UTC")

    logger.info(
        "Fetched macro data: %d days, SPX %.0f-%.0f, US10Y %.2f-%.2f",
        len(result),
        result["spx_close"].min() if len(result) else 0,
        result["spx_close"].max() if len(result) else 0,
        result["us10y_close"].min() if len(result) else 0,
        result["us10y_close"].max() if len(result) else 0,
    )
    return result


def build_macro_panel(
    dxy_series: pd.Series,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Merge DXY, SPX, and US10Y into a single daily panel.

    Parameters
    ----------
    dxy_series: Daily DXY values (from compute_dxy_series).
    start_date: ISO date string for SPX/US10Y fetch.
    end_date: ISO date string for SPX/US10Y fetch.

    Returns
    -------
    DataFrame with columns: dxy_close, dxy_pct_change, spx_close,
    spx_pct_change, us10y_close, us10y_pct_change.
    """
    macro = fetch_daily_macro(start_date, end_date)

    panel = pd.DataFrame(index=dxy_series.index)
    panel["dxy_close"] = dxy_series
    panel["dxy_pct_change"] = dxy_series.pct_change()

    if not macro.empty:
        # Align macro data to DXY index via reindex + ffill
        macro_aligned = macro.reindex(panel.index, method="ffill")
        for col in macro_aligned.columns:
            panel[col] = macro_aligned[col]
    else:
        for name in _TICKERS:
            panel[f"{name}_close"] = float("nan")
            panel[f"{name}_pct_change"] = float("nan")

    return panel
