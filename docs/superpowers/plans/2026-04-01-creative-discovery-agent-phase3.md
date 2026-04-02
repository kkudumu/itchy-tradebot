# Creative Pattern Discovery Agent (Phase 3: Macro Regime Detection + Multi-Asset Correlation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add macro regime context to the discovery engine by synthesizing DXY from 6 FX component pairs (via the existing Dukascopy downloader), fetching SPX/US10Y daily data from yfinance, classifying each trading day into one of 5 regimes (risk_on, risk_off, dollar_driven, inflation_fear, mixed), tagging every trade with its regime, and filtering trades near high-impact economic events (NFP, FOMC, CPI).

**Architecture:** DXY is computed from 6 FX pairs downloaded via the existing `DukascopyDownloader` (which already supports configurable `instrument` and `point_divisor`). Daily SPX/US10Y data comes from yfinance. A `RegimeClassifier` assigns each day a regime label based on DXY/SPX/US10Y daily percentage moves. The `FeatureVectorBuilder` fills reserved dims 59-63 with DXY % change, SPX % change, US10Y % change, regime one-hot, and event proximity. An `EconCalendar` generates deterministic dates for NFP, FOMC, and CPI. An `EventProximityFilter` (EdgeFilter subclass) vetoes entries within N hours of high-impact events.

**Tech Stack:** Existing `DukascopyDownloader` (6 FX pairs at `point_divisor=100_000.0`), yfinance (^GSPC, ^TNX), numpy/pandas, existing EdgeFilter ABC, existing FeatureVectorBuilder.

**Phases overview (this is Phase 3 of 5):**
- Phase 1: XGBoost/SHAP analysis + hypothesis loop + knowledge base
- Phase 2: PatternPy chart patterns + selective screenshots + Claude visual analysis
- **Phase 3 (this plan):** Macro regime (DXY synthesis, SPX, US10Y, econ calendar)
- Phase 4: LLM-generated EdgeFilter code with AST/test/backtest safety
- Phase 5: Full orchestrator tying phases 1-4 into the 30-day rolling challenge loop

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/macro/__init__.py` | Package init, public exports |
| `src/macro/dxy_synthesizer.py` | Download 6 FX pairs via Dukascopy, compute DXY daily close |
| `src/macro/market_data.py` | Fetch SPX + US10Y daily data via yfinance |
| `src/macro/regime_classifier.py` | Classify each day as risk_on / risk_off / dollar_driven / inflation_fear / mixed |
| `src/macro/econ_calendar.py` | Deterministic high-impact event date generator (NFP, FOMC, CPI) |
| `src/macro/event_proximity.py` | EdgeFilter that vetoes trades within N hours of events |
| `src/macro/trade_tagger.py` | Tag trades with regime + event proximity metadata |
| `tests/test_dxy_synthesizer.py` | Tests for DXY formula and daily aggregation |
| `tests/test_market_data.py` | Tests for SPX/US10Y fetching |
| `tests/test_regime_classifier.py` | Tests for regime classification logic |
| `tests/test_econ_calendar.py` | Tests for NFP/FOMC/CPI date generation |
| `tests/test_event_proximity.py` | Tests for EdgeFilter event blocking |
| `tests/test_trade_tagger.py` | Tests for regime tagging trades |
| `tests/test_macro_integration.py` | End-to-end macro pipeline integration test |

---

### Task 1: DXY Synthesizer — FX Pair Daily Closes

**Files:**
- Create: `src/macro/__init__.py`
- Create: `src/macro/dxy_synthesizer.py`
- Test: `tests/test_dxy_synthesizer.py`

- [ ] **Step 1: Write failing test for compute_dxy**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dxy_synthesizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.macro'`

- [ ] **Step 3: Create package init and dxy_synthesizer**

```python
# src/macro/__init__.py
"""Macro regime detection and multi-asset correlation.

Synthesizes DXY from FX components, fetches SPX/US10Y,
classifies daily regimes, and provides event proximity filtering.
"""

from src.macro.dxy_synthesizer import compute_dxy_from_rates, compute_dxy_series
from src.macro.regime_classifier import RegimeClassifier, RegimeLabel

__all__ = [
    "compute_dxy_from_rates",
    "compute_dxy_series",
    "RegimeClassifier",
    "RegimeLabel",
]
```

```python
# src/macro/dxy_synthesizer.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dxy_synthesizer.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/macro/__init__.py src/macro/dxy_synthesizer.py tests/test_dxy_synthesizer.py
git commit -m "feat: add DXY synthesizer from 6 FX component pairs (Task 1)"
```

---

### Task 2: Market Data Fetcher (SPX + US10Y via yfinance)

**Files:**
- Create: `src/macro/market_data.py`
- Test: `tests/test_market_data.py`

- [ ] **Step 1: Write failing test for fetch_daily_macro**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_market_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.macro.market_data'`

- [ ] **Step 3: Implement market_data**

```python
# src/macro/market_data.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pip install yfinance && pytest tests/test_market_data.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/macro/market_data.py tests/test_market_data.py
git commit -m "feat: add SPX/US10Y daily data fetcher via yfinance (Task 2)"
```

---

### Task 3: Regime Classifier

**Files:**
- Create: `src/macro/regime_classifier.py`
- Test: `tests/test_regime_classifier.py`

- [ ] **Step 1: Write failing test for classify_regime**

```python
# tests/test_regime_classifier.py
"""Tests for daily macro regime classification."""

import numpy as np
import pandas as pd
import pytest


class TestRegimeLabel:
    def test_all_labels_are_strings(self):
        from src.macro.regime_classifier import RegimeLabel

        for label in RegimeLabel:
            assert isinstance(label.value, str)

    def test_five_regimes_exist(self):
        from src.macro.regime_classifier import RegimeLabel

        assert len(RegimeLabel) == 5


class TestClassifySingleDay:
    def test_risk_on(self):
        """SPX up, DXY down, US10Y stable -> risk_on."""
        from src.macro.regime_classifier import classify_single_day, RegimeLabel

        regime = classify_single_day(
            dxy_pct=-0.5, spx_pct=1.2, us10y_pct=0.01
        )
        assert regime == RegimeLabel.RISK_ON

    def test_risk_off(self):
        """SPX down, DXY up, US10Y down (flight to safety) -> risk_off."""
        from src.macro.regime_classifier import classify_single_day, RegimeLabel

        regime = classify_single_day(
            dxy_pct=0.8, spx_pct=-1.5, us10y_pct=-0.3
        )
        assert regime == RegimeLabel.RISK_OFF

    def test_dollar_driven(self):
        """DXY strong move, SPX mixed -> dollar_driven."""
        from src.macro.regime_classifier import classify_single_day, RegimeLabel

        regime = classify_single_day(
            dxy_pct=1.2, spx_pct=0.1, us10y_pct=0.1
        )
        assert regime == RegimeLabel.DOLLAR_DRIVEN

    def test_inflation_fear(self):
        """US10Y spikes, SPX down -> inflation_fear."""
        from src.macro.regime_classifier import classify_single_day, RegimeLabel

        regime = classify_single_day(
            dxy_pct=0.3, spx_pct=-0.8, us10y_pct=3.0
        )
        assert regime == RegimeLabel.INFLATION_FEAR

    def test_mixed(self):
        """Small moves in all -> mixed."""
        from src.macro.regime_classifier import classify_single_day, RegimeLabel

        regime = classify_single_day(
            dxy_pct=0.05, spx_pct=-0.05, us10y_pct=0.02
        )
        assert regime == RegimeLabel.MIXED


class TestRegimeClassifier:
    def _make_panel(self, n=20, seed=42):
        """Generate a synthetic macro panel."""
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2025-01-06", periods=n, freq="B", tz="UTC")
        return pd.DataFrame({
            "dxy_close": 103.0 + np.cumsum(rng.normal(0, 0.3, n)),
            "dxy_pct_change": rng.normal(0, 0.5, n),
            "spx_close": 4800.0 + np.cumsum(rng.normal(0, 20, n)),
            "spx_pct_change": rng.normal(0, 0.8, n),
            "us10y_close": 4.25 + np.cumsum(rng.normal(0, 0.05, n)),
            "us10y_pct_change": rng.normal(0, 1.0, n),
        }, index=dates)

    def test_classify_panel_returns_series(self):
        from src.macro.regime_classifier import RegimeClassifier

        classifier = RegimeClassifier()
        panel = self._make_panel()
        regimes = classifier.classify(panel)

        assert isinstance(regimes, pd.Series)
        assert len(regimes) == len(panel)
        assert regimes.name == "regime"

    def test_all_values_are_valid_labels(self):
        from src.macro.regime_classifier import RegimeClassifier, RegimeLabel

        classifier = RegimeClassifier()
        panel = self._make_panel(n=50)
        regimes = classifier.classify(panel)

        valid_labels = {label.value for label in RegimeLabel}
        for regime in regimes:
            assert regime in valid_labels

    def test_get_regime_for_date(self):
        from src.macro.regime_classifier import RegimeClassifier

        classifier = RegimeClassifier()
        panel = self._make_panel()
        regimes = classifier.classify(panel)

        target_date = panel.index[5]
        regime = classifier.get_regime_for_date(regimes, target_date)
        assert isinstance(regime, str)

    def test_get_regime_for_missing_date_returns_mixed(self):
        from src.macro.regime_classifier import RegimeClassifier

        classifier = RegimeClassifier()
        panel = self._make_panel()
        regimes = classifier.classify(panel)

        missing_date = pd.Timestamp("2020-01-01", tz="UTC")
        regime = classifier.get_regime_for_date(regimes, missing_date)
        assert regime == "mixed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_regime_classifier.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.macro.regime_classifier'`

- [ ] **Step 3: Implement RegimeClassifier**

```python
# src/macro/regime_classifier.py
"""Daily macro regime classifier.

Classifies each trading day into one of 5 regimes based on the
daily percentage moves of DXY, SPX, and US10Y:

    risk_on:        SPX rallying, DXY weakening (risk appetite)
    risk_off:       SPX falling, DXY strengthening (flight to safety)
    dollar_driven:  DXY moving strongly, SPX relatively flat
    inflation_fear: US10Y spiking, SPX weak (rate shock)
    mixed:          No clear macro theme (small moves)

Thresholds are calibrated for daily percentage changes:
- "Strong" move: |pct_change| > 0.5% for DXY/SPX, > 1.5% for US10Y
- "Spike" move: |pct_change| > 2.0% for US10Y

Gold correlation patterns:
- risk_off + dollar_driven -> bearish for gold (strong USD headwind)
- risk_off + DXY_down -> bullish for gold (safe haven bid)
- inflation_fear -> bullish for gold (inflation hedge)
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class RegimeLabel(Enum):
    """Macro regime classification labels."""
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    DOLLAR_DRIVEN = "dollar_driven"
    INFLATION_FEAR = "inflation_fear"
    MIXED = "mixed"


# Thresholds for regime classification (daily pct change)
_DXY_STRONG = 0.5     # |DXY pct change| > 0.5% = strong dollar move
_SPX_STRONG = 0.5     # |SPX pct change| > 0.5% = strong equity move
_US10Y_STRONG = 1.5   # |US10Y pct change| > 1.5% = notable yield move
_US10Y_SPIKE = 2.0    # |US10Y pct change| > 2.0% = yield spike


def classify_single_day(
    dxy_pct: float,
    spx_pct: float,
    us10y_pct: float,
) -> RegimeLabel:
    """Classify a single day's macro regime.

    Parameters
    ----------
    dxy_pct: DXY daily percentage change.
    spx_pct: SPX daily percentage change.
    us10y_pct: US10Y daily percentage change.

    Returns
    -------
    RegimeLabel for the day.
    """
    dxy_abs = abs(dxy_pct)
    spx_abs = abs(spx_pct)
    us10y_abs = abs(us10y_pct)

    # Inflation fear: yields spike sharply while equities drop
    if us10y_abs >= _US10Y_SPIKE and spx_pct < 0:
        return RegimeLabel.INFLATION_FEAR

    # Risk-on: equities rising, dollar weakening
    if spx_pct > _SPX_STRONG and dxy_pct < 0:
        return RegimeLabel.RISK_ON

    # Risk-off: equities falling, dollar strengthening
    if spx_pct < -_SPX_STRONG and dxy_pct > 0:
        return RegimeLabel.RISK_OFF

    # Dollar-driven: DXY moving strongly, equities relatively flat
    if dxy_abs >= _DXY_STRONG and spx_abs < _SPX_STRONG:
        return RegimeLabel.DOLLAR_DRIVEN

    # Inflation fear (secondary): yields notable + equities weak
    if us10y_abs >= _US10Y_STRONG and spx_pct < -_SPX_STRONG:
        return RegimeLabel.INFLATION_FEAR

    # Mixed: no clear dominant theme
    return RegimeLabel.MIXED


class RegimeClassifier:
    """Classify daily macro regimes from a macro panel DataFrame.

    Parameters
    ----------
    dxy_strong: Threshold for strong DXY daily move (default 0.5%).
    spx_strong: Threshold for strong SPX daily move (default 0.5%).
    us10y_spike: Threshold for US10Y yield spike (default 2.0%).
    """

    def __init__(
        self,
        dxy_strong: float = _DXY_STRONG,
        spx_strong: float = _SPX_STRONG,
        us10y_spike: float = _US10Y_SPIKE,
    ) -> None:
        self._dxy_strong = dxy_strong
        self._spx_strong = spx_strong
        self._us10y_spike = us10y_spike

    def classify(self, panel: pd.DataFrame) -> pd.Series:
        """Classify each day in the macro panel.

        Parameters
        ----------
        panel:
            DataFrame with columns: dxy_pct_change, spx_pct_change,
            us10y_pct_change.

        Returns
        -------
        pd.Series of regime label strings with name="regime".
        """
        regimes = []
        for _, row in panel.iterrows():
            dxy_pct = float(row.get("dxy_pct_change", 0) or 0)
            spx_pct = float(row.get("spx_pct_change", 0) or 0)
            us10y_pct = float(row.get("us10y_pct_change", 0) or 0)

            label = classify_single_day(dxy_pct, spx_pct, us10y_pct)
            regimes.append(label.value)

        result = pd.Series(regimes, index=panel.index, name="regime")
        logger.info(
            "Classified %d days: %s",
            len(result),
            result.value_counts().to_dict(),
        )
        return result

    @staticmethod
    def get_regime_for_date(
        regime_series: pd.Series,
        target_date: pd.Timestamp,
    ) -> str:
        """Look up the regime for a specific date.

        If the exact date is missing, uses the most recent prior date
        (forward-fill semantics). Falls back to "mixed" if no data exists.

        Parameters
        ----------
        regime_series: Output of classify().
        target_date: The date to look up.

        Returns
        -------
        Regime label string.
        """
        if target_date in regime_series.index:
            return str(regime_series.loc[target_date])

        # Find nearest prior date
        prior = regime_series.index[regime_series.index <= target_date]
        if len(prior) > 0:
            return str(regime_series.loc[prior[-1]])

        return RegimeLabel.MIXED.value
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_regime_classifier.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/macro/regime_classifier.py tests/test_regime_classifier.py
git commit -m "feat: add daily macro regime classifier with 5 regimes (Task 3)"
```

---

### Task 4: Economic Calendar (Deterministic Date Generator)

**Files:**
- Create: `src/macro/econ_calendar.py`
- Test: `tests/test_econ_calendar.py`

- [ ] **Step 1: Write failing test for EconCalendar**

```python
# tests/test_econ_calendar.py
"""Tests for deterministic economic calendar date generator."""

import pytest
from datetime import datetime, timezone, timedelta


class TestNFPDates:
    def test_nfp_is_first_friday(self):
        """NFP is released on the first Friday of each month."""
        from src.macro.econ_calendar import EconCalendar

        cal = EconCalendar()
        nfp_dates = cal.get_nfp_dates(2025)

        assert len(nfp_dates) == 12
        for dt in nfp_dates:
            assert dt.weekday() == 4  # Friday
            assert dt.day <= 7  # First 7 days = first week

    def test_nfp_jan_2025(self):
        """NFP for January 2025 should be Friday Jan 3."""
        from src.macro.econ_calendar import EconCalendar

        cal = EconCalendar()
        nfp_dates = cal.get_nfp_dates(2025)
        assert nfp_dates[0].day == 3
        assert nfp_dates[0].month == 1


class TestFOMCDates:
    def test_fomc_count(self):
        """FOMC meets 8 times per year."""
        from src.macro.econ_calendar import EconCalendar

        cal = EconCalendar()
        fomc_dates = cal.get_fomc_dates(2025)
        assert len(fomc_dates) == 8

    def test_fomc_dates_are_wednesdays(self):
        """FOMC decisions are announced on Wednesdays."""
        from src.macro.econ_calendar import EconCalendar

        cal = EconCalendar()
        fomc_dates = cal.get_fomc_dates(2025)
        for dt in fomc_dates:
            assert dt.weekday() == 2  # Wednesday

    def test_fomc_months_match_schedule(self):
        """FOMC meets in Jan, Mar, May, Jun, Jul, Sep, Nov, Dec."""
        from src.macro.econ_calendar import EconCalendar

        cal = EconCalendar()
        fomc_dates = cal.get_fomc_dates(2025)
        months = sorted(set(dt.month for dt in fomc_dates))
        assert months == [1, 3, 5, 6, 7, 9, 11, 12]


class TestCPIDates:
    def test_cpi_count(self):
        """CPI is released monthly -> 12 dates per year."""
        from src.macro.econ_calendar import EconCalendar

        cal = EconCalendar()
        cpi_dates = cal.get_cpi_dates(2025)
        assert len(cpi_dates) == 12

    def test_cpi_falls_around_12th(self):
        """CPI is typically released around the 10th-15th of the month."""
        from src.macro.econ_calendar import EconCalendar

        cal = EconCalendar()
        cpi_dates = cal.get_cpi_dates(2025)
        for dt in cpi_dates:
            assert 8 <= dt.day <= 16  # ~12th +/- a few days for weekends


class TestGetAllEvents:
    def test_returns_sorted_events(self):
        from src.macro.econ_calendar import EconCalendar

        cal = EconCalendar()
        events = cal.get_all_events(2025)

        assert len(events) > 0
        timestamps = [e.timestamp for e in events]
        assert timestamps == sorted(timestamps)

    def test_events_have_correct_impact(self):
        from src.macro.econ_calendar import EconCalendar

        cal = EconCalendar()
        events = cal.get_all_events(2025)

        for event in events:
            assert event.impact == "red"  # All are high-impact

    def test_events_for_date_range(self):
        from src.macro.econ_calendar import EconCalendar

        cal = EconCalendar()
        start = datetime(2025, 3, 1, tzinfo=timezone.utc)
        end = datetime(2025, 3, 31, tzinfo=timezone.utc)
        events = cal.get_events_in_range(start, end)

        # March 2025 should have: 1 NFP + 1 FOMC + 1 CPI = 3 events
        assert len(events) >= 2  # At minimum NFP + CPI


class TestIsNearEvent:
    def test_within_window_returns_true(self):
        from src.macro.econ_calendar import EconCalendar

        cal = EconCalendar()
        # Get NFP date for Jan 2025 (Jan 3)
        nfp = cal.get_nfp_dates(2025)[0]
        nfp_time = nfp.replace(hour=13, minute=30)  # 8:30 AM ET = 13:30 UTC

        check_time = nfp_time - timedelta(hours=1)  # 1 hour before
        is_near, event = cal.is_near_event(check_time, hours_before=4, hours_after=2)
        assert is_near is True
        assert event is not None

    def test_outside_window_returns_false(self):
        from src.macro.econ_calendar import EconCalendar

        cal = EconCalendar()
        # A random Tuesday far from any event
        check_time = datetime(2025, 2, 18, 10, 0, tzinfo=timezone.utc)
        is_near, event = cal.is_near_event(check_time, hours_before=4, hours_after=2)
        assert is_near is False
        assert event is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_econ_calendar.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement EconCalendar**

```python
# src/macro/econ_calendar.py
"""Deterministic economic calendar for high-impact US events.

Generates dates for NFP (Non-Farm Payrolls), FOMC (Federal Reserve
decisions), and CPI (Consumer Price Index) releases using calendar
rules rather than external API calls.

Event timing (all UTC):
    NFP:  First Friday of month, 13:30 UTC (8:30 AM ET)
    FOMC: See schedule below, 19:00 UTC (2:00 PM ET)
    CPI:  ~12th of month (adjusted for weekends), 13:30 UTC (8:30 AM ET)

FOMC schedule: 8 meetings per year. The months are fixed; specific
dates vary by year but follow a pattern of Tue-Wed meetings where
the decision is announced on Wednesday.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Reuse NewsEvent from the edges.news_filter module
from src.edges.news_filter import NewsEvent


# FOMC meeting months (decision day months)
_FOMC_MONTHS = [1, 3, 5, 6, 7, 9, 11, 12]

# FOMC approximate decision day within month (3rd Wednesday as default)
# For precise dates, a hardcoded schedule per year is used.
_FOMC_SCHEDULES: Dict[int, List[Tuple[int, int]]] = {
    # (month, day) for the Wednesday decision announcement
    2024: [(1, 31), (3, 20), (5, 1), (6, 12), (7, 31), (9, 18), (11, 7), (12, 18)],
    2025: [(1, 29), (3, 19), (5, 7), (6, 18), (7, 30), (9, 17), (11, 5), (12, 17)],
    2026: [(1, 28), (3, 18), (5, 6), (6, 17), (7, 29), (9, 16), (11, 4), (12, 16)],
}


def _first_friday(year: int, month: int) -> datetime:
    """Return the first Friday of the given month as a UTC datetime."""
    dt = datetime(year, month, 1, tzinfo=timezone.utc)
    # weekday(): 0=Mon, 4=Fri
    days_until_friday = (4 - dt.weekday()) % 7
    return dt + timedelta(days=days_until_friday)


def _nearest_weekday(year: int, month: int, target_day: int) -> datetime:
    """Return the nearest weekday to target_day in the given month.

    If target_day falls on Saturday, use Friday (target_day - 1).
    If target_day falls on Sunday, use Monday (target_day + 1).
    """
    dt = datetime(year, month, target_day, tzinfo=timezone.utc)
    if dt.weekday() == 5:  # Saturday -> Friday
        dt -= timedelta(days=1)
    elif dt.weekday() == 6:  # Sunday -> Monday
        dt += timedelta(days=1)
    return dt


def _third_wednesday(year: int, month: int) -> datetime:
    """Return the third Wednesday of the given month."""
    dt = datetime(year, month, 1, tzinfo=timezone.utc)
    # Find first Wednesday
    days_until_wed = (2 - dt.weekday()) % 7
    first_wed = dt + timedelta(days=days_until_wed)
    # Third Wednesday = first + 14 days
    return first_wed + timedelta(days=14)


class EconCalendar:
    """Deterministic generator for high-impact US economic event dates.

    Generates NFP, FOMC, and CPI dates for a given year using calendar
    rules. All events are tagged with "red" impact level for use with
    the existing NewsFilter/NewsEvent infrastructure.
    """

    def get_nfp_dates(self, year: int) -> List[datetime]:
        """Get all 12 NFP release dates for a year.

        NFP = first Friday of each month at 13:30 UTC.
        """
        dates = []
        for month in range(1, 13):
            nfp = _first_friday(year, month)
            nfp = nfp.replace(hour=13, minute=30)
            dates.append(nfp)
        return dates

    def get_fomc_dates(self, year: int) -> List[datetime]:
        """Get all 8 FOMC decision dates for a year.

        Uses hardcoded schedules for known years, falls back to
        third-Wednesday heuristic for other years. Time: 19:00 UTC.
        """
        if year in _FOMC_SCHEDULES:
            dates = []
            for month, day in _FOMC_SCHEDULES[year]:
                dt = datetime(year, month, day, 19, 0, tzinfo=timezone.utc)
                dates.append(dt)
            return dates

        # Fallback: third Wednesday of each FOMC month
        dates = []
        for month in _FOMC_MONTHS:
            dt = _third_wednesday(year, month)
            dt = dt.replace(hour=19, minute=0)
            dates.append(dt)
        return dates

    def get_cpi_dates(self, year: int) -> List[datetime]:
        """Get all 12 CPI release dates for a year.

        CPI ~ 12th of month (adjusted for weekends) at 13:30 UTC.
        """
        dates = []
        for month in range(1, 13):
            cpi = _nearest_weekday(year, month, 12)
            cpi = cpi.replace(hour=13, minute=30)
            dates.append(cpi)
        return dates

    def get_all_events(self, year: int) -> List[NewsEvent]:
        """Get all high-impact events for a year, sorted chronologically.

        Returns NewsEvent objects compatible with the existing
        src.edges.news_filter.NewsFilter infrastructure.
        """
        events: List[NewsEvent] = []

        for dt in self.get_nfp_dates(year):
            events.append(NewsEvent(
                timestamp=dt,
                title=f"US Non-Farm Payrolls ({dt.strftime('%b %Y')})",
                impact="red",
            ))

        for dt in self.get_fomc_dates(year):
            events.append(NewsEvent(
                timestamp=dt,
                title=f"FOMC Rate Decision ({dt.strftime('%b %Y')})",
                impact="red",
            ))

        for dt in self.get_cpi_dates(year):
            events.append(NewsEvent(
                timestamp=dt,
                title=f"US CPI ({dt.strftime('%b %Y')})",
                impact="red",
            ))

        events.sort(key=lambda e: e.timestamp)
        logger.info("Generated %d high-impact events for %d", len(events), year)
        return events

    def get_events_in_range(
        self,
        start: datetime,
        end: datetime,
    ) -> List[NewsEvent]:
        """Get events within a date range (may span year boundaries).

        Parameters
        ----------
        start: Range start (inclusive).
        end: Range end (inclusive).
        """
        years = set(range(start.year, end.year + 1))
        all_events = []
        for year in years:
            all_events.extend(self.get_all_events(year))

        return [e for e in all_events if start <= e.timestamp <= end]

    def is_near_event(
        self,
        timestamp: datetime,
        hours_before: int = 4,
        hours_after: int = 2,
    ) -> Tuple[bool, Optional[NewsEvent]]:
        """Check if a timestamp is near any high-impact event.

        Parameters
        ----------
        timestamp: The time to check (UTC).
        hours_before: Blackout window before event.
        hours_after: Blackout window after event.

        Returns
        -------
        (is_near, nearest_event): Whether within window, and which event.
        """
        # Check events for the year of the timestamp
        events = self.get_all_events(timestamp.year)

        for event in events:
            window_start = event.timestamp - timedelta(hours=hours_before)
            window_end = event.timestamp + timedelta(hours=hours_after)
            if window_start <= timestamp <= window_end:
                return True, event

        return False, None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_econ_calendar.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/macro/econ_calendar.py tests/test_econ_calendar.py
git commit -m "feat: add deterministic economic calendar (NFP, FOMC, CPI) (Task 4)"
```

---

### Task 5: Event Proximity EdgeFilter

**Files:**
- Create: `src/macro/event_proximity.py`
- Test: `tests/test_event_proximity.py`

- [ ] **Step 1: Write failing test for EventProximityFilter**

```python
# tests/test_event_proximity.py
"""Tests for EventProximityFilter edge filter."""

import pytest
from datetime import datetime, timezone


def _make_context(timestamp):
    """Build a minimal EdgeContext for testing."""
    from src.edges.base import EdgeContext

    return EdgeContext(
        timestamp=timestamp,
        day_of_week=timestamp.weekday(),
        close_price=2650.0,
        high_price=2655.0,
        low_price=2645.0,
        spread=0.30,
        session="london",
        adx=30.0,
        atr=5.0,
    )


class TestEventProximityFilter:
    def test_blocks_entry_near_nfp(self):
        """Trade 2 hours before NFP should be blocked."""
        from src.macro.event_proximity import EventProximityFilter

        config = {
            "enabled": True,
            "params": {
                "hours_before": 4,
                "hours_after": 2,
            },
        }
        edge = EventProximityFilter(config)

        # NFP Jan 2025 = Jan 3 at 13:30 UTC; check at 11:30 (2h before)
        ctx = _make_context(datetime(2025, 1, 3, 11, 30, tzinfo=timezone.utc))
        result = edge.should_allow(ctx)

        assert result.allowed is False
        assert "NFP" in result.reason or "Non-Farm" in result.reason

    def test_allows_entry_far_from_events(self):
        """Trade on a quiet Tuesday should be allowed."""
        from src.macro.event_proximity import EventProximityFilter

        config = {
            "enabled": True,
            "params": {
                "hours_before": 4,
                "hours_after": 2,
            },
        }
        edge = EventProximityFilter(config)

        # Feb 18 2025 is a Tuesday with no nearby events
        ctx = _make_context(datetime(2025, 2, 18, 10, 0, tzinfo=timezone.utc))
        result = edge.should_allow(ctx)

        assert result.allowed is True

    def test_disabled_filter_always_allows(self):
        """Disabled filter should pass through."""
        from src.macro.event_proximity import EventProximityFilter

        config = {"enabled": False, "params": {}}
        edge = EventProximityFilter(config)

        ctx = _make_context(datetime(2025, 1, 3, 13, 30, tzinfo=timezone.utc))
        result = edge.should_allow(ctx)

        assert result.allowed is True

    def test_blocks_entry_near_fomc(self):
        """Trade near FOMC decision should be blocked."""
        from src.macro.event_proximity import EventProximityFilter

        config = {
            "enabled": True,
            "params": {"hours_before": 4, "hours_after": 2},
        }
        edge = EventProximityFilter(config)

        # FOMC Jan 29, 2025 at 19:00 UTC; check at 16:00 (3h before)
        ctx = _make_context(datetime(2025, 1, 29, 16, 0, tzinfo=timezone.utc))
        result = edge.should_allow(ctx)

        assert result.allowed is False
        assert "FOMC" in result.reason

    def test_blocks_entry_near_cpi(self):
        """Trade near CPI release should be blocked."""
        from src.macro.event_proximity import EventProximityFilter

        config = {
            "enabled": True,
            "params": {"hours_before": 4, "hours_after": 2},
        }
        edge = EventProximityFilter(config)

        # CPI ~Jan 12, 2025 (Sunday -> Monday Jan 13) at 13:30 UTC
        # Check at 12:00 (1.5h before)
        ctx = _make_context(datetime(2025, 1, 13, 12, 0, tzinfo=timezone.utc))
        result = edge.should_allow(ctx)

        assert result.allowed is False
        assert "CPI" in result.reason

    def test_configurable_window(self):
        """Wider window should block more trades."""
        from src.macro.event_proximity import EventProximityFilter

        # Narrow window: 1h before, 1h after
        narrow_config = {
            "enabled": True,
            "params": {"hours_before": 1, "hours_after": 1},
        }
        narrow_edge = EventProximityFilter(narrow_config)

        # Wide window: 8h before, 4h after
        wide_config = {
            "enabled": True,
            "params": {"hours_before": 8, "hours_after": 4},
        }
        wide_edge = EventProximityFilter(wide_config)

        # 6 hours before NFP Jan 3, 2025
        ctx = _make_context(datetime(2025, 1, 3, 7, 30, tzinfo=timezone.utc))

        narrow_result = narrow_edge.should_allow(ctx)
        wide_result = wide_edge.should_allow(ctx)

        assert narrow_result.allowed is True  # 6h before, outside 1h window
        assert wide_result.allowed is False   # 6h before, inside 8h window
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_event_proximity.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement EventProximityFilter**

```python
# src/macro/event_proximity.py
"""Event proximity edge filter.

Blocks entries within N hours of high-impact economic events (NFP,
FOMC, CPI) using the deterministic EconCalendar.

Unlike the existing news_filter (which requires an injected calendar
implementation), this filter is self-contained: it generates event
dates deterministically and requires no external data source.

Config keys (via params):
    hours_before: int — Blackout window before event. Default 4.
    hours_after:  int — Blackout window after event. Default 2.
"""

from __future__ import annotations

import logging

from src.edges.base import EdgeContext, EdgeFilter, EdgeResult
from src.macro.econ_calendar import EconCalendar

logger = logging.getLogger(__name__)


class EventProximityFilter(EdgeFilter):
    """Block entries within configurable hours of high-impact US events.

    Parameters
    ----------
    config:
        Edge config dict with 'enabled' and 'params' keys.
    """

    def __init__(self, config: dict) -> None:
        super().__init__("event_proximity", config)
        params = config.get("params", {})
        self._hours_before: int = int(params.get("hours_before", 4))
        self._hours_after: int = int(params.get("hours_after", 2))
        self._calendar = EconCalendar()

    def should_allow(self, context: EdgeContext) -> EdgeResult:
        if not self.enabled:
            return self._disabled_result()

        is_near, event = self._calendar.is_near_event(
            context.timestamp,
            hours_before=self._hours_before,
            hours_after=self._hours_after,
        )

        if is_near and event is not None:
            return EdgeResult(
                allowed=False,
                edge_name=self.name,
                reason=(
                    f"Bar at {context.timestamp.strftime('%Y-%m-%d %H:%M')} UTC "
                    f"is within {self._hours_before}h pre / {self._hours_after}h post "
                    f"blackout for '{event.title}' at "
                    f"{event.timestamp.strftime('%H:%M')} UTC"
                ),
            )

        return EdgeResult(
            allowed=True,
            edge_name=self.name,
            reason="No high-impact events within blackout window",
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_event_proximity.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/macro/event_proximity.py tests/test_event_proximity.py
git commit -m "feat: add event proximity EdgeFilter for NFP/FOMC/CPI (Task 5)"
```

---

### Task 6: Trade Tagger (Regime + Event Proximity Metadata)

**Files:**
- Create: `src/macro/trade_tagger.py`
- Test: `tests/test_trade_tagger.py`

- [ ] **Step 1: Write failing test for tag_trades**

```python
# tests/test_trade_tagger.py
"""Tests for trade regime/event tagging."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone
from unittest.mock import patch


def _make_trades(n=10, seed=42):
    """Generate trades with timestamps."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-06", periods=n, freq="B", tz="UTC")
    trades = []
    for i in range(n):
        trades.append({
            "r_multiple": float(rng.choice([-1.0, 1.5])),
            "entry_time": dates[i].to_pydatetime(),
            "context": {
                "adx_value": float(rng.uniform(10, 40)),
                "session": "london",
            },
        })
    return trades


def _make_regime_series(dates, seed=42):
    """Generate a mock regime series."""
    rng = np.random.default_rng(seed)
    labels = ["risk_on", "risk_off", "dollar_driven", "inflation_fear", "mixed"]
    return pd.Series(
        rng.choice(labels, len(dates)),
        index=dates,
        name="regime",
    )


class TestTradeTagger:
    def test_tags_trades_with_regime(self):
        from src.macro.trade_tagger import TradeTagger

        trades = _make_trades()
        dates = pd.date_range("2025-01-01", periods=30, freq="B", tz="UTC")
        regimes = _make_regime_series(dates)

        tagger = TradeTagger(regime_series=regimes)
        tagged = tagger.tag_trades(trades)

        assert len(tagged) == len(trades)
        for trade in tagged:
            assert "regime" in trade
            assert trade["regime"] in [
                "risk_on", "risk_off", "dollar_driven",
                "inflation_fear", "mixed",
            ]

    def test_tags_trades_with_event_proximity(self):
        from src.macro.trade_tagger import TradeTagger

        # Trade on NFP day (Jan 3, 2025 is first Friday)
        trades = [{
            "r_multiple": 1.5,
            "entry_time": datetime(2025, 1, 3, 12, 0, tzinfo=timezone.utc),
            "context": {"session": "london"},
        }]

        dates = pd.date_range("2025-01-01", periods=10, freq="B", tz="UTC")
        regimes = _make_regime_series(dates)

        tagger = TradeTagger(regime_series=regimes)
        tagged = tagger.tag_trades(trades, hours_before=4, hours_after=2)

        assert tagged[0]["near_event"] is True
        assert "NFP" in tagged[0].get("nearest_event", "") or \
               "Non-Farm" in tagged[0].get("nearest_event", "")

    def test_quiet_day_not_near_event(self):
        from src.macro.trade_tagger import TradeTagger

        trades = [{
            "r_multiple": -1.0,
            "entry_time": datetime(2025, 2, 18, 10, 0, tzinfo=timezone.utc),
            "context": {"session": "london"},
        }]

        dates = pd.date_range("2025-01-01", periods=60, freq="B", tz="UTC")
        regimes = _make_regime_series(dates)

        tagger = TradeTagger(regime_series=regimes)
        tagged = tagger.tag_trades(trades)

        assert tagged[0]["near_event"] is False

    def test_preserves_existing_trade_fields(self):
        from src.macro.trade_tagger import TradeTagger

        trades = [{
            "r_multiple": 2.0,
            "entry_time": datetime(2025, 1, 6, 10, 0, tzinfo=timezone.utc),
            "context": {"adx_value": 35.0},
            "custom_field": "preserved",
        }]

        dates = pd.date_range("2025-01-01", periods=30, freq="B", tz="UTC")
        regimes = _make_regime_series(dates)

        tagger = TradeTagger(regime_series=regimes)
        tagged = tagger.tag_trades(trades)

        assert tagged[0]["custom_field"] == "preserved"
        assert tagged[0]["r_multiple"] == 2.0

    def test_regime_stats(self):
        from src.macro.trade_tagger import TradeTagger

        trades = _make_trades(n=20)
        dates = pd.date_range("2025-01-01", periods=30, freq="B", tz="UTC")
        regimes = _make_regime_series(dates)

        tagger = TradeTagger(regime_series=regimes)
        tagged = tagger.tag_trades(trades)
        stats = tagger.compute_regime_stats(tagged)

        assert isinstance(stats, dict)
        # At least one regime should appear
        assert len(stats) > 0
        for regime, info in stats.items():
            assert "count" in info
            assert "win_rate" in info
            assert "avg_r" in info
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trade_tagger.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement TradeTagger**

```python
# src/macro/trade_tagger.py
"""Tag backtest trades with macro regime and event proximity metadata.

Enriches trade dicts with:
    - regime: daily macro regime label (risk_on, risk_off, etc.)
    - near_event: bool — whether entry was within N hours of NFP/FOMC/CPI
    - nearest_event: str — name of the nearest event (if near_event=True)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from src.macro.econ_calendar import EconCalendar
from src.macro.regime_classifier import RegimeClassifier, RegimeLabel

logger = logging.getLogger(__name__)


class TradeTagger:
    """Enrich trade dicts with macro regime and event proximity tags.

    Parameters
    ----------
    regime_series:
        Pre-computed regime series (from RegimeClassifier.classify()).
        If None, all trades get regime="mixed".
    """

    def __init__(
        self,
        regime_series: Optional[pd.Series] = None,
    ) -> None:
        self._regimes = regime_series
        self._calendar = EconCalendar()
        self._classifier = RegimeClassifier()

    def tag_trades(
        self,
        trades: List[Dict[str, Any]],
        hours_before: int = 4,
        hours_after: int = 2,
    ) -> List[Dict[str, Any]]:
        """Tag each trade with regime and event proximity.

        Parameters
        ----------
        trades:
            List of trade dicts. Each should have 'entry_time' (datetime).
        hours_before:
            Event blackout window before event (hours).
        hours_after:
            Event blackout window after event (hours).

        Returns
        -------
        List of trade dicts with added 'regime', 'near_event',
        'nearest_event' keys. Original dicts are not modified.
        """
        tagged: List[Dict[str, Any]] = []

        for trade in trades:
            enriched = dict(trade)  # shallow copy

            entry_time = trade.get("entry_time")
            if entry_time is None:
                enriched["regime"] = RegimeLabel.MIXED.value
                enriched["near_event"] = False
                enriched["nearest_event"] = ""
                tagged.append(enriched)
                continue

            # Ensure timezone-aware
            if isinstance(entry_time, datetime) and entry_time.tzinfo is None:
                entry_time = entry_time.replace(tzinfo=timezone.utc)

            # Regime tag
            if self._regimes is not None:
                regime = self._classifier.get_regime_for_date(
                    self._regimes, pd.Timestamp(entry_time)
                )
            else:
                regime = RegimeLabel.MIXED.value
            enriched["regime"] = regime

            # Event proximity tag
            is_near, event = self._calendar.is_near_event(
                entry_time,
                hours_before=hours_before,
                hours_after=hours_after,
            )
            enriched["near_event"] = is_near
            enriched["nearest_event"] = event.title if event else ""

            tagged.append(enriched)

        logger.info(
            "Tagged %d trades: %d near events, regimes: %s",
            len(tagged),
            sum(1 for t in tagged if t.get("near_event")),
            pd.Series([t.get("regime", "unknown") for t in tagged]).value_counts().to_dict(),
        )
        return tagged

    @staticmethod
    def compute_regime_stats(
        tagged_trades: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Compute per-regime trade statistics.

        Returns
        -------
        Dict mapping regime label to:
            count: number of trades
            win_rate: fraction of winning trades (r_multiple > 0)
            avg_r: average R-multiple
            near_event_count: trades near high-impact events
        """
        from collections import defaultdict

        buckets: Dict[str, List[float]] = defaultdict(list)
        event_counts: Dict[str, int] = defaultdict(int)

        for trade in tagged_trades:
            regime = trade.get("regime", "mixed")
            r = trade.get("r_multiple")
            if r is not None:
                buckets[regime].append(float(r))
            if trade.get("near_event"):
                event_counts[regime] += 1

        stats: Dict[str, Dict[str, Any]] = {}
        for regime, r_values in buckets.items():
            wins = sum(1 for r in r_values if r > 0)
            stats[regime] = {
                "count": len(r_values),
                "win_rate": round(wins / len(r_values), 4) if r_values else 0.0,
                "avg_r": round(sum(r_values) / len(r_values), 4) if r_values else 0.0,
                "near_event_count": event_counts.get(regime, 0),
            }

        return stats
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_trade_tagger.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/macro/trade_tagger.py tests/test_trade_tagger.py
git commit -m "feat: add trade tagger with regime + event proximity metadata (Task 6)"
```

---

### Task 7: Feature Vector Builder — Fill Reserved Dims 59-63

**Files:**
- Modify: `src/learning/feature_vector.py`
- Modify: `src/discovery/xgb_analyzer.py` (FEATURE_NAMES update)
- Test: `tests/test_feature_vector_macro.py`

- [ ] **Step 1: Write failing test for macro dims**

```python
# tests/test_feature_vector_macro.py
"""Tests for macro regime dimensions (59-63) in FeatureVectorBuilder."""

import numpy as np
import pytest


class TestMacroFeatureDims:
    def test_dxy_pct_change_dim59(self):
        """Dim 59 should encode DXY daily pct change."""
        from src.learning.feature_vector import FeatureVectorBuilder

        builder = FeatureVectorBuilder()
        ctx = {"dxy_pct_change": 0.5}  # 0.5% DXY move
        vec = builder.build(ctx)

        # Normalised to [0, 1] from range [-2%, 2%]
        assert 0.0 <= vec[59] <= 1.0
        assert vec[59] > 0.5  # Positive DXY change -> above midpoint

    def test_spx_pct_change_dim60(self):
        """Dim 60 should encode SPX daily pct change."""
        from src.learning.feature_vector import FeatureVectorBuilder

        builder = FeatureVectorBuilder()
        ctx = {"spx_pct_change": -1.0}  # 1% SPX drop
        vec = builder.build(ctx)

        assert 0.0 <= vec[60] <= 1.0
        assert vec[60] < 0.5  # Negative SPX change -> below midpoint

    def test_us10y_pct_change_dim61(self):
        """Dim 61 should encode US10Y daily pct change."""
        from src.learning.feature_vector import FeatureVectorBuilder

        builder = FeatureVectorBuilder()
        ctx = {"us10y_pct_change": 2.0}  # 2% yield spike
        vec = builder.build(ctx)

        assert 0.0 <= vec[61] <= 1.0
        assert vec[61] > 0.5  # Positive yield change -> above midpoint

    def test_regime_one_hot_dim62(self):
        """Dim 62 should encode regime as ordinal 0-1."""
        from src.learning.feature_vector import FeatureVectorBuilder

        builder = FeatureVectorBuilder()

        # risk_on = 0.0, risk_off = 0.25, dollar_driven = 0.5,
        # inflation_fear = 0.75, mixed = 1.0
        for regime, expected_range in [
            ("risk_on", (0.0, 0.1)),
            ("risk_off", (0.2, 0.3)),
            ("dollar_driven", (0.4, 0.6)),
            ("inflation_fear", (0.7, 0.8)),
            ("mixed", (0.9, 1.0)),
        ]:
            vec = builder.build({"macro_regime": regime})
            assert expected_range[0] <= vec[62] <= expected_range[1], (
                f"regime={regime}, vec[62]={vec[62]}"
            )

    def test_event_proximity_dim63(self):
        """Dim 63 should encode event proximity (0=far, 1=imminent)."""
        from src.learning.feature_vector import FeatureVectorBuilder

        builder = FeatureVectorBuilder()

        # Far from event
        vec_far = builder.build({"hours_to_event": 24.0})
        # Near event
        vec_near = builder.build({"hours_to_event": 1.0})

        assert vec_far[63] < vec_near[63]  # Nearer = higher value

    def test_no_macro_data_defaults_to_zero(self):
        """Missing macro fields should default to 0.0 (midpoint or zero)."""
        from src.learning.feature_vector import FeatureVectorBuilder

        builder = FeatureVectorBuilder()
        vec = builder.build({})

        # Dims 59-63 should all be 0.0 when no macro data provided
        for dim in range(59, 64):
            assert vec[dim] == 0.0, f"dim {dim} = {vec[dim]}, expected 0.0"

    def test_overall_vector_still_64_dims(self):
        """Vector dimension must remain 64."""
        from src.learning.feature_vector import FeatureVectorBuilder

        builder = FeatureVectorBuilder()
        vec = builder.build({
            "dxy_pct_change": 0.3,
            "spx_pct_change": -0.5,
            "us10y_pct_change": 1.2,
            "macro_regime": "risk_off",
            "hours_to_event": 3.0,
        })
        assert len(vec) == 64
        assert vec.min() >= 0.0
        assert vec.max() <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_feature_vector_macro.py -v`
Expected: FAIL — dims 59-63 are all 0.0 regardless of input (reserved slots)

- [ ] **Step 3: Modify FeatureVectorBuilder to fill dims 59-63**

In `src/learning/feature_vector.py`, add a regime ordinal map and replace the reserved block:

Add to the ordinal lookup tables section (after `_VOL_REGIME_MAP`):

```python
# macro regime ordinal encoding
_REGIME_MAP: Dict[str, float] = {
    "risk_on":        0.0,
    "risk_off":       0.25,
    "dollar_driven":  0.5,
    "inflation_fear": 0.75,
    "mixed":          1.0,
}
```

Replace the `# dims 59–63: reserved -> 0.0` block in the `build()` method:

```python
        # dim 59: dxy_pct_change — DXY daily % change, normalised [-2%, 2%] -> [0, 1]
        dxy_pct = float(context.get("dxy_pct_change") or 0.0)
        vec[59] = self._normalize_continuous(dxy_pct, -2.0, 2.0)

        # dim 60: spx_pct_change — SPX daily % change, normalised [-3%, 3%] -> [0, 1]
        spx_pct = float(context.get("spx_pct_change") or 0.0)
        vec[60] = self._normalize_continuous(spx_pct, -3.0, 3.0)

        # dim 61: us10y_pct_change — US10Y yield daily % change, normalised [-5%, 5%] -> [0, 1]
        us10y_pct = float(context.get("us10y_pct_change") or 0.0)
        vec[61] = self._normalize_continuous(us10y_pct, -5.0, 5.0)

        # dim 62: macro_regime — ordinal encoding of daily regime label
        regime_str = str(context.get("macro_regime") or "").lower().strip()
        vec[62] = _REGIME_MAP.get(regime_str, 0.0)

        # dim 63: event_proximity — hours to nearest event, inverted [0=far, 1=imminent]
        # Normalised: 0h = 1.0 (imminent), 24h+ = 0.0 (far)
        hours_to_event = float(context.get("hours_to_event") or 0.0)
        if hours_to_event > 0:
            vec[63] = self._normalize_continuous(24.0 - hours_to_event, 0.0, 24.0)
        # else: 0.0 (no event data)
```

- [ ] **Step 4: Update FEATURE_NAMES in xgb_analyzer.py**

Replace the reserved names in `src/discovery/xgb_analyzer.py` FEATURE_NAMES:

```python
    # Regime (55-63)
    "trend_strength", "vol_regime", "spread_norm", "daily_range_vs_atr",
    "dxy_pct_change", "spx_pct_change", "us10y_pct_change", "macro_regime", "event_proximity",
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_feature_vector_macro.py -v`
Expected: All 7 tests PASS

- [ ] **Step 6: Run existing tests to verify no regressions**

Run: `pytest tests/ -v --tb=short`
Expected: All existing tests still PASS

- [ ] **Step 7: Commit**

```bash
git add src/learning/feature_vector.py src/discovery/xgb_analyzer.py tests/test_feature_vector_macro.py
git commit -m "feat: fill feature vector dims 59-63 with macro regime data (Task 7)"
```

---

### Task 8: Wire Into Edge Manager + Config

**Files:**
- Modify: `src/edges/manager.py`
- Modify: `config/edges.yaml`
- Modify: `src/macro/__init__.py`

- [ ] **Step 1: Register EventProximityFilter in EdgeManager**

In `src/edges/manager.py`, add the import and registry entry:

```python
# Add to imports
from src.macro.event_proximity import EventProximityFilter
```

Add to `_REGISTRY`:

```python
    "event_proximity":           (EventProximityFilter,            "entry"),
```

- [ ] **Step 2: Add config to edges.yaml**

Append to `config/edges.yaml`:

```yaml
# 13. Event proximity — block entries near NFP, FOMC, CPI releases
event_proximity:
  enabled: false
  params:
    hours_before: 4
    hours_after: 2
```

Add to `strategy_profiles.sss`:

```yaml
    event_proximity:
      enabled: true
      hours_before: 4
      hours_after: 2
```

- [ ] **Step 3: Update src/macro/__init__.py with full exports**

```python
# src/macro/__init__.py
"""Macro regime detection and multi-asset correlation.

Synthesizes DXY from FX components, fetches SPX/US10Y,
classifies daily regimes, and provides event proximity filtering.
"""

from src.macro.dxy_synthesizer import compute_dxy_from_rates, compute_dxy_series
from src.macro.regime_classifier import RegimeClassifier, RegimeLabel
from src.macro.econ_calendar import EconCalendar
from src.macro.event_proximity import EventProximityFilter
from src.macro.trade_tagger import TradeTagger

__all__ = [
    "compute_dxy_from_rates",
    "compute_dxy_series",
    "RegimeClassifier",
    "RegimeLabel",
    "EconCalendar",
    "EventProximityFilter",
    "TradeTagger",
]
```

- [ ] **Step 4: Run full edge manager test suite**

Run: `pytest tests/ -v --tb=short -k "edge or manager"`
Expected: All tests PASS, new event_proximity filter loads correctly

- [ ] **Step 5: Commit**

```bash
git add src/edges/manager.py config/edges.yaml src/macro/__init__.py
git commit -m "feat: wire EventProximityFilter into EdgeManager + config (Task 8)"
```

---

### Task 9: Integration Test — Full Macro Pipeline

**Files:**
- Test: `tests/test_macro_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_macro_integration.py
"""Integration test: full macro pipeline from FX data to tagged trades."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock


class TestMacroPipelineIntegration:
    def _make_fx_data(self, n=20, seed=42):
        """Generate synthetic daily FX data for all 6 pairs."""
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2025-01-06", periods=n, freq="B", tz="UTC")
        return {
            "EURUSD": pd.DataFrame({"close": 1.08 + np.cumsum(rng.normal(0, 0.002, n))}, index=dates),
            "USDJPY": pd.DataFrame({"close": 150.0 + np.cumsum(rng.normal(0, 0.5, n))}, index=dates),
            "GBPUSD": pd.DataFrame({"close": 1.265 + np.cumsum(rng.normal(0, 0.002, n))}, index=dates),
            "USDCAD": pd.DataFrame({"close": 1.36 + np.cumsum(rng.normal(0, 0.003, n))}, index=dates),
            "USDSEK": pd.DataFrame({"close": 10.5 + np.cumsum(rng.normal(0, 0.05, n))}, index=dates),
            "USDCHF": pd.DataFrame({"close": 0.88 + np.cumsum(rng.normal(0, 0.002, n))}, index=dates),
        }

    def _make_trades(self, dates, n=10, seed=42):
        """Generate trades with entry times matching given dates."""
        rng = np.random.default_rng(seed)
        trades = []
        for i in range(min(n, len(dates))):
            trades.append({
                "r_multiple": float(rng.choice([-1.0, 1.5, 2.0, -0.5])),
                "entry_time": dates[i].to_pydatetime().replace(hour=10, minute=0),
                "context": {
                    "adx_value": float(rng.uniform(15, 40)),
                    "atr_value": float(rng.uniform(3, 8)),
                    "session": "london",
                },
            })
        return trades

    def test_dxy_to_regime_to_tagged_trades(self):
        """Full pipeline: FX data -> DXY -> regime -> tagged trades."""
        from src.macro.dxy_synthesizer import compute_dxy_series
        from src.macro.regime_classifier import RegimeClassifier
        from src.macro.trade_tagger import TradeTagger

        fx_data = self._make_fx_data(n=20)
        dates = list(fx_data["EURUSD"].index)

        # Step 1: Compute DXY
        dxy = compute_dxy_series(fx_data)
        assert len(dxy) == 20
        assert 90 < dxy.mean() < 120  # Reasonable DXY range

        # Step 2: Build mock macro panel (skip yfinance in test)
        panel = pd.DataFrame({
            "dxy_close": dxy,
            "dxy_pct_change": dxy.pct_change(),
            "spx_close": 4800.0 + np.cumsum(np.random.default_rng(42).normal(0, 20, 20)),
            "spx_pct_change": np.random.default_rng(42).normal(0, 0.8, 20),
            "us10y_close": 4.25 + np.cumsum(np.random.default_rng(42).normal(0, 0.05, 20)),
            "us10y_pct_change": np.random.default_rng(42).normal(0, 1.0, 20),
        }, index=dxy.index)

        # Step 3: Classify regimes
        classifier = RegimeClassifier()
        regimes = classifier.classify(panel)
        assert len(regimes) == 20
        assert all(r in ["risk_on", "risk_off", "dollar_driven", "inflation_fear", "mixed"]
                   for r in regimes)

        # Step 4: Tag trades
        trades = self._make_trades(pd.DatetimeIndex(dates), n=10)
        tagger = TradeTagger(regime_series=regimes)
        tagged = tagger.tag_trades(trades)

        assert len(tagged) == 10
        for trade in tagged:
            assert "regime" in trade
            assert "near_event" in trade
            assert isinstance(trade["near_event"], bool)

        # Step 5: Compute regime stats
        stats = tagger.compute_regime_stats(tagged)
        assert isinstance(stats, dict)
        total = sum(s["count"] for s in stats.values())
        assert total == 10

    def test_feature_vector_receives_macro_data(self):
        """FeatureVectorBuilder correctly encodes macro context."""
        from src.learning.feature_vector import FeatureVectorBuilder

        builder = FeatureVectorBuilder()
        context = {
            "adx_value": 30.0,
            "atr_value": 5.0,
            "session": "london",
            "dxy_pct_change": 0.8,
            "spx_pct_change": -1.2,
            "us10y_pct_change": 2.5,
            "macro_regime": "risk_off",
            "hours_to_event": 3.0,
        }
        vec = builder.build(context)

        assert len(vec) == 64
        assert vec[59] > 0.5   # Positive DXY change
        assert vec[60] < 0.5   # Negative SPX change
        assert vec[61] > 0.5   # Positive yield change
        assert 0.2 <= vec[62] <= 0.3  # risk_off ordinal
        assert vec[63] > 0.0   # Near-ish event

    def test_econ_calendar_consistency(self):
        """Calendar events are consistent and properly dated."""
        from src.macro.econ_calendar import EconCalendar

        cal = EconCalendar()
        events_2025 = cal.get_all_events(2025)

        # 12 NFP + 8 FOMC + 12 CPI = 32 events
        assert len(events_2025) == 32

        # All events should be in 2025
        for event in events_2025:
            assert event.timestamp.year == 2025
            assert event.impact == "red"

        # Events should be chronologically sorted
        timestamps = [e.timestamp for e in events_2025]
        assert timestamps == sorted(timestamps)

    def test_event_proximity_filter_integration(self):
        """EventProximityFilter works when registered in EdgeManager."""
        from src.edges.base import EdgeContext
        from src.macro.event_proximity import EventProximityFilter

        config = {
            "enabled": True,
            "params": {"hours_before": 4, "hours_after": 2},
        }
        edge = EventProximityFilter(config)

        # NFP day: Jan 3, 2025, 13:30 UTC
        ctx_nfp = EdgeContext(
            timestamp=datetime(2025, 1, 3, 12, 0, tzinfo=timezone.utc),
            day_of_week=4,
            close_price=2650.0,
            high_price=2655.0,
            low_price=2645.0,
            spread=0.30,
            session="london",
            adx=30.0,
            atr=5.0,
        )
        result = edge.should_allow(ctx_nfp)
        assert result.allowed is False

        # Quiet day
        ctx_quiet = EdgeContext(
            timestamp=datetime(2025, 2, 18, 10, 0, tzinfo=timezone.utc),
            day_of_week=1,
            close_price=2650.0,
            high_price=2655.0,
            low_price=2645.0,
            spread=0.30,
            session="london",
            adx=30.0,
            atr=5.0,
        )
        result = edge.should_allow(ctx_quiet)
        assert result.allowed is True
```

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/test_macro_integration.py tests/test_dxy_synthesizer.py tests/test_market_data.py tests/test_regime_classifier.py tests/test_econ_calendar.py tests/test_event_proximity.py tests/test_trade_tagger.py tests/test_feature_vector_macro.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_macro_integration.py
git commit -m "feat: add macro pipeline integration tests (Task 9)"
```

---

### Task 10: Knowledge Base — Persist Regime Findings

**Files:**
- Modify: `src/discovery/knowledge_base.py`
- Test: `tests/test_knowledge_base_macro.py`

- [ ] **Step 1: Write failing tests for regime persistence**

```python
# tests/test_knowledge_base_macro.py
"""Tests for regime finding persistence in the knowledge base."""

import json
import pytest
from pathlib import Path


class TestRegimePersistence:
    def _make_kb(self, tmp_path):
        from src.discovery.knowledge_base import KnowledgeBase
        return KnowledgeBase(base_dir=str(tmp_path / "agent_knowledge"))

    def test_save_and_load_regime_stats(self, tmp_path):
        kb = self._make_kb(tmp_path)
        stats = {
            "risk_on": {"count": 15, "win_rate": 0.53, "avg_r": 0.42},
            "risk_off": {"count": 8, "win_rate": 0.25, "avg_r": -0.31},
            "dollar_driven": {"count": 12, "win_rate": 0.33, "avg_r": -0.05},
            "inflation_fear": {"count": 5, "win_rate": 0.60, "avg_r": 0.85},
            "mixed": {"count": 20, "win_rate": 0.35, "avg_r": 0.10},
        }
        kb.save_regime_stats(stats, window_id="w_003")
        loaded = kb.load_regime_stats("w_003")

        assert loaded["risk_on"]["win_rate"] == 0.53
        assert loaded["inflation_fear"]["count"] == 5

    def test_save_and_load_macro_panel_summary(self, tmp_path):
        kb = self._make_kb(tmp_path)
        summary = {
            "window_id": "w_003",
            "date_range": "2025-01-06 to 2025-01-31",
            "dxy_mean": 103.5,
            "dxy_std": 0.8,
            "regime_distribution": {
                "risk_on": 5,
                "risk_off": 3,
                "mixed": 12,
            },
            "events_in_window": 3,
        }
        kb.save_macro_summary(summary, window_id="w_003")
        loaded = kb.load_macro_summary("w_003")

        assert loaded["dxy_mean"] == 103.5
        assert loaded["events_in_window"] == 3

    def test_list_regime_stats_across_windows(self, tmp_path):
        kb = self._make_kb(tmp_path)
        kb.save_regime_stats({"risk_on": {"count": 10}}, window_id="w_001")
        kb.save_regime_stats({"risk_off": {"count": 5}}, window_id="w_002")

        all_stats = kb.list_regime_stats()
        assert len(all_stats) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_knowledge_base_macro.py -v`
Expected: FAIL with `AttributeError: 'KnowledgeBase' object has no attribute 'save_regime_stats'`

- [ ] **Step 3: Add regime persistence methods to KnowledgeBase**

Append to `src/discovery/knowledge_base.py`:

```python
    # -- Regime Stats --

    def save_regime_stats(self, stats: Dict[str, Any], window_id: str) -> Path:
        """Persist per-regime trade statistics for a window."""
        regime_dir = self._base / "regime_stats"
        regime_dir.mkdir(parents=True, exist_ok=True)
        path = regime_dir / f"{window_id}.json"
        path.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
        return path

    def load_regime_stats(self, window_id: str) -> Dict[str, Any]:
        """Load regime stats for a specific window."""
        path = self._base / "regime_stats" / f"{window_id}.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def list_regime_stats(self) -> List[Dict[str, Any]]:
        """List all saved regime stats across windows."""
        regime_dir = self._base / "regime_stats"
        if not regime_dir.exists():
            return []
        results = []
        for p in sorted(regime_dir.glob("*.json")):
            stats = json.loads(p.read_text(encoding="utf-8"))
            stats["_window_id"] = p.stem
            results.append(stats)
        return results

    # -- Macro Panel Summaries --

    def save_macro_summary(self, summary: Dict[str, Any], window_id: str) -> Path:
        """Persist macro panel summary for a window."""
        macro_dir = self._base / "macro_summaries"
        macro_dir.mkdir(parents=True, exist_ok=True)
        path = macro_dir / f"{window_id}.json"
        path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        return path

    def load_macro_summary(self, window_id: str) -> Dict[str, Any]:
        """Load macro panel summary for a specific window."""
        path = self._base / "macro_summaries" / f"{window_id}.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_knowledge_base_macro.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discovery/knowledge_base.py tests/test_knowledge_base_macro.py
git commit -m "feat: add regime stats + macro summary persistence to knowledge base (Task 10)"
```

---

## Summary

| Task | Component | Files | Tests |
|------|-----------|-------|-------|
| 1 | DXY synthesizer | `dxy_synthesizer.py` | 5 tests |
| 2 | Market data fetcher | `market_data.py` | 4 tests |
| 3 | Regime classifier | `regime_classifier.py` | 9 tests |
| 4 | Economic calendar | `econ_calendar.py` | 10 tests |
| 5 | Event proximity filter | `event_proximity.py` | 6 tests |
| 6 | Trade tagger | `trade_tagger.py` | 5 tests |
| 7 | Feature vector macro dims | `feature_vector.py` | 7 tests |
| 8 | EdgeManager + config wiring | `manager.py`, `edges.yaml` | existing tests |
| 9 | Integration test | -- | 4 tests |
| 10 | Knowledge base regime persistence | `knowledge_base.py` | 3 tests |

**Total: 10 tasks, 53 tests, 7 new source files, 3 modified source files, 1 modified config file.**

**Dependencies to install:** `pip install yfinance`

**pyproject.toml addition:** Add `"yfinance>=0.2"` to `[project.dependencies]`.

Phase 4-5 plans will be written separately once Phase 3 is validated.
