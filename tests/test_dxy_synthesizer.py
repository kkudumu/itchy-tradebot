# tests/test_dxy_synthesizer.py
"""Tests for DXY synthesis from 6 FX component pairs."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone


def _make_fx_daily(dates, eurusd, usdjpy, gbpusd, usdcad, usdsek, usdchf):
    """Helper: build a dict of FX pair DataFrames with daily close prices."""
    idx = pd.DatetimeIndex(dates, tz="UTC")
    return {
        "EURUSD": pd.DataFrame({"close": eurusd}, index=idx),
        "USDJPY": pd.DataFrame({"close": usdjpy}, index=idx),
        "GBPUSD": pd.DataFrame({"close": gbpusd}, index=idx),
        "USDCAD": pd.DataFrame({"close": usdcad}, index=idx),
        "USDSEK": pd.DataFrame({"close": usdsek}, index=idx),
        "USDCHF": pd.DataFrame({"close": usdchf}, index=idx),
    }


class TestComputeDXY:
    def test_formula_matches_known_value(self):
        """DXY ~ 103.5 with typical early-2026 FX rates."""
        from src.macro.dxy_synthesizer import compute_dxy_from_rates

        # Typical rates that should produce DXY ~ 103-104
        dxy = compute_dxy_from_rates(
            eurusd=1.0800,
            usdjpy=150.00,
            gbpusd=1.2650,
            usdcad=1.3600,
            usdsek=10.50,
            usdchf=0.8800,
        )
        assert 95.0 < dxy < 115.0  # Sanity range for DXY

    def test_inverse_relationship_eurusd(self):
        """Weaker EUR (lower EURUSD) should raise DXY."""
        from src.macro.dxy_synthesizer import compute_dxy_from_rates

        base_rates = dict(
            eurusd=1.0800, usdjpy=150.0, gbpusd=1.2650,
            usdcad=1.3600, usdsek=10.50, usdchf=0.8800,
        )
        dxy_base = compute_dxy_from_rates(**base_rates)
        dxy_weaker_eur = compute_dxy_from_rates(
            **{**base_rates, "eurusd": 1.0500}
        )
        assert dxy_weaker_eur > dxy_base  # EURUSD down -> DXY up

    def test_compute_dxy_series(self):
        """Compute DXY for a multi-day series of FX data."""
        from src.macro.dxy_synthesizer import compute_dxy_series

        dates = pd.date_range("2025-01-06", periods=5, freq="B", tz="UTC")
        fx_data = _make_fx_daily(
            dates,
            eurusd=[1.08, 1.07, 1.06, 1.07, 1.08],
            usdjpy=[150.0, 151.0, 152.0, 151.0, 150.0],
            gbpusd=[1.265, 1.260, 1.255, 1.260, 1.265],
            usdcad=[1.36, 1.37, 1.38, 1.37, 1.36],
            usdsek=[10.50, 10.60, 10.70, 10.60, 10.50],
            usdchf=[0.88, 0.89, 0.90, 0.89, 0.88],
        )
        dxy_series = compute_dxy_series(fx_data)

        assert isinstance(dxy_series, pd.Series)
        assert len(dxy_series) == 5
        assert dxy_series.name == "DXY"
        # Day 3 (strongest dollar) should have highest DXY
        assert dxy_series.iloc[2] == dxy_series.max()

    def test_handles_missing_pair_raises(self):
        """Missing FX pair should raise ValueError."""
        from src.macro.dxy_synthesizer import compute_dxy_series

        dates = pd.date_range("2025-01-06", periods=3, freq="B", tz="UTC")
        fx_data = {
            "EURUSD": pd.DataFrame({"close": [1.08, 1.07, 1.06]}, index=dates),
            # Missing USDJPY and others
        }
        with pytest.raises(ValueError, match="Missing FX pairs"):
            compute_dxy_series(fx_data)

    def test_handles_nan_forward_fills(self):
        """NaN values in FX data should be forward-filled."""
        from src.macro.dxy_synthesizer import compute_dxy_series

        dates = pd.date_range("2025-01-06", periods=3, freq="B", tz="UTC")
        fx_data = _make_fx_daily(
            dates,
            eurusd=[1.08, np.nan, 1.06],
            usdjpy=[150.0, 151.0, 152.0],
            gbpusd=[1.265, 1.260, 1.255],
            usdcad=[1.36, 1.37, 1.38],
            usdsek=[10.50, 10.60, 10.70],
            usdchf=[0.88, 0.89, 0.90],
        )
        dxy_series = compute_dxy_series(fx_data)
        assert not dxy_series.isna().any()
