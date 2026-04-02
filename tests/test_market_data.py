# tests/test_market_data.py
"""Tests for macro market data fetching (SPX, US10Y)."""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone


def _mock_yfinance_download(tickers, start, end, **kwargs):
    """Build a fake yfinance response DataFrame."""
    dates = pd.date_range(start, end, freq="B", tz="UTC")[:5]
    if isinstance(tickers, str):
        tickers = [tickers]

    # yfinance returns MultiIndex columns: (price_field, ticker)
    data = {}
    for ticker in tickers:
        if ticker == "^GSPC":
            data[("Close", ticker)] = [4800.0, 4810.0, 4790.0, 4820.0, 4830.0][:len(dates)]
        elif ticker == "^TNX":
            data[("Close", ticker)] = [4.25, 4.30, 4.28, 4.35, 4.32][:len(dates)]
        else:
            data[("Close", ticker)] = [100.0] * len(dates)

    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


class TestFetchDailyMacro:
    @patch("src.macro.market_data.yf")
    def test_returns_dataframe_with_expected_columns(self, mock_yf):
        from src.macro.market_data import fetch_daily_macro

        mock_yf.download.return_value = _mock_yfinance_download(
            ["^GSPC", "^TNX"], "2025-01-06", "2025-01-10"
        )
        df = fetch_daily_macro("2025-01-06", "2025-01-10")

        assert isinstance(df, pd.DataFrame)
        assert "spx_close" in df.columns
        assert "us10y_close" in df.columns
        assert "spx_pct_change" in df.columns
        assert "us10y_pct_change" in df.columns
        assert len(df) > 0

    @patch("src.macro.market_data.yf")
    def test_pct_changes_are_computed(self, mock_yf):
        from src.macro.market_data import fetch_daily_macro

        mock_yf.download.return_value = _mock_yfinance_download(
            ["^GSPC", "^TNX"], "2025-01-06", "2025-01-10"
        )
        df = fetch_daily_macro("2025-01-06", "2025-01-10")

        # First row pct_change is NaN, rest should be numeric
        assert pd.isna(df["spx_pct_change"].iloc[0])
        assert not pd.isna(df["spx_pct_change"].iloc[1])

    @patch("src.macro.market_data.yf")
    def test_handles_empty_response(self, mock_yf):
        from src.macro.market_data import fetch_daily_macro

        mock_yf.download.return_value = pd.DataFrame()
        df = fetch_daily_macro("2025-01-06", "2025-01-10")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestBuildMacroPanel:
    @patch("src.macro.market_data.yf")
    def test_merges_dxy_with_spx_us10y(self, mock_yf):
        from src.macro.market_data import build_macro_panel

        mock_yf.download.return_value = _mock_yfinance_download(
            ["^GSPC", "^TNX"], "2025-01-06", "2025-01-10"
        )

        dates = pd.date_range("2025-01-06", periods=5, freq="B", tz="UTC")
        dxy_series = pd.Series(
            [103.0, 103.5, 104.0, 103.8, 103.2],
            index=dates,
            name="DXY",
        )
        panel = build_macro_panel(dxy_series, "2025-01-06", "2025-01-10")

        assert "dxy_close" in panel.columns
        assert "dxy_pct_change" in panel.columns
        assert "spx_close" in panel.columns
        assert "us10y_close" in panel.columns
        assert len(panel) == 5
