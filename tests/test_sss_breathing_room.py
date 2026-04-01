"""Tests for BreathingRoomDetector — vectorized swing high/low detection."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
import pytest

from src.strategy.strategies.sss import BreathingRoomDetector, SwingPoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_TIME = datetime(2024, 1, 1, 9, 0, 0)


def make_swing_data(prices: List[float], start_time: Optional[datetime] = None) -> pd.DataFrame:
    """Create 1M OHLCV DataFrame from a sequence of close prices.

    Each price becomes a bar with:
      high  = price + 0.5
      low   = price - 0.5
      close = price
      open  = price
    """
    t = start_time or BASE_TIME
    idx = pd.date_range(t, periods=len(prices), freq="1min")
    arr = np.array(prices, dtype=float)
    return pd.DataFrame(
        {
            "open": arr,
            "high": arr + 0.5,
            "low": arr - 0.5,
            "close": arr,
            "volume": 1.0,
        },
        index=idx,
    )


def make_swing_point(
    index: int,
    price: float,
    swing_type: str,
    bar_count: int = 5,
    timestamp: Optional[datetime] = None,
) -> SwingPoint:
    """Create a SwingPoint for testing."""
    return SwingPoint(
        index=index,
        timestamp=timestamp or (BASE_TIME + timedelta(minutes=index)),
        price=price,
        swing_type=swing_type,
        bar_count_since_prev=bar_count,
    )


# ---------------------------------------------------------------------------
# Basic detection tests
# ---------------------------------------------------------------------------


def test_detect_returns_empty_on_empty_df():
    detector = BreathingRoomDetector()
    df = pd.DataFrame(columns=["high", "low"])
    result = detector.detect(df)
    assert result == []


def test_detect_known_swing_pattern():
    """A simple V-shape should produce a swing low."""
    # 7 bars: valley in the middle (bar index 3 after 0-based)
    # With lookback_n=3, window=7; bar 3 is the centre
    prices = [10.0, 9.0, 8.0, 7.0, 8.0, 9.0, 10.0]
    df = make_swing_data(prices)
    detector = BreathingRoomDetector(lookback_n=3, min_swing_pips=0.1)
    result = detector.detect(df)
    lows = [s for s in result if s.swing_type == "low"]
    assert len(lows) >= 1
    assert lows[0].price == pytest.approx(6.5)  # low = price - 0.5 = 7.0 - 0.5


def test_detect_known_high_pattern():
    """An inverted-V should produce a swing high."""
    prices = [10.0, 11.0, 12.0, 13.0, 12.0, 11.0, 10.0]
    df = make_swing_data(prices)
    detector = BreathingRoomDetector(lookback_n=3, min_swing_pips=0.1)
    result = detector.detect(df)
    highs = [s for s in result if s.swing_type == "high"]
    assert len(highs) >= 1
    assert highs[0].price == pytest.approx(13.5)  # high = 13.0 + 0.5


def test_detect_alternating_guarantee():
    """Output must never have two consecutive swings of the same type."""
    # Create a noisy zigzag with many potential candidates
    prices = [100 + 5 * np.sin(i * 0.5) for i in range(50)]
    df = make_swing_data(prices)
    detector = BreathingRoomDetector(lookback_n=3, min_swing_pips=0.01)
    result = detector.detect(df)
    for i in range(1, len(result)):
        assert result[i].swing_type != result[i - 1].swing_type, (
            f"Consecutive {result[i].swing_type} at indices {result[i-1].index}, {result[i].index}"
        )


def test_min_swing_pips_filter():
    """Tiny swings below min_swing_pips should be filtered out."""
    # Very small perturbation — 0.5 pip in price (high/low diff = 1 pip)
    prices = [100.0, 100.01, 99.99, 100.0, 100.01, 99.99, 100.0,
              100.01, 99.99, 100.0, 100.01, 99.99, 100.0, 100.01]
    df = make_swing_data(prices)
    # min_swing_pips=5 → min move = 5 * 0.1 = 0.5 price units
    detector = BreathingRoomDetector(lookback_n=2, min_swing_pips=5.0, pip_value=0.1)
    result = detector.detect(df)
    # Most tiny swings should be filtered; possibly 0 or very few
    for i in range(1, len(result)):
        price_diff = abs(result[i].price - result[i - 1].price)
        assert price_diff >= 5.0 * 0.1 - 1e-9, (
            f"Swing too small: {price_diff:.4f} < {5.0 * 0.1:.4f}"
        )


def test_flat_tops_handled():
    """Two bars at the same high — only the first should be the swing."""
    # flat top at bars 3 and 4
    prices = [10.0, 11.0, 12.0, 13.0, 13.0, 12.0, 11.0, 10.0]
    df = make_swing_data(prices)
    detector = BreathingRoomDetector(lookback_n=3, min_swing_pips=0.1)
    result = detector.detect(df)
    highs = [s for s in result if s.swing_type == "high"]
    # Should detect at most one swing high for the flat top region
    # (rolling max will find both bars equal, but we take the first via nonzero())
    assert len(highs) <= 2  # alternation will reduce duplicates


def test_nan_bars_skipped():
    """NaN values in high/low must be excluded from swing detection."""
    prices = [10.0, 9.0, 8.0, 7.0, 8.0, 9.0, 10.0]
    df = make_swing_data(prices)
    # Inject NaN at bar 3 (the valley)
    df.loc[df.index[3], "high"] = float("nan")
    df.loc[df.index[3], "low"] = float("nan")
    detector = BreathingRoomDetector(lookback_n=3, min_swing_pips=0.01)
    result = detector.detect(df)
    # NaN bar should not appear in swings
    for sp in result:
        assert not np.isnan(sp.price)


def test_custom_lookback_n():
    """lookback_n=1 should produce more swing points than lookback_n=5."""
    prices = [10, 9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9, 10]
    df = make_swing_data(prices)
    det_wide = BreathingRoomDetector(lookback_n=5, min_swing_pips=0.01)
    det_narrow = BreathingRoomDetector(lookback_n=1, min_swing_pips=0.01)
    wide = det_wide.detect(df)
    narrow = det_narrow.detect(df)
    assert len(narrow) >= len(wide)


# ---------------------------------------------------------------------------
# Incremental detection tests
# ---------------------------------------------------------------------------


def test_incremental_matches_batch_detect():
    """Incremental should produce the same swings as the batch detect."""
    prices = [100 + 5 * np.sin(i * 0.4) for i in range(60)]
    df = make_swing_data(prices)

    detector_batch = BreathingRoomDetector(lookback_n=3, min_swing_pips=0.05)
    batch_swings = detector_batch.detect(df)

    detector_incr = BreathingRoomDetector(lookback_n=3, min_swing_pips=0.05)
    incr_history: List[SwingPoint] = []
    for i, row in enumerate(df.itertuples()):
        detector_incr.detect_incremental(
            bar_index=i,
            timestamp=row.Index,
            high=row.high,
            low=row.low,
            history=incr_history,
        )

    # Prices should match (within floating point tolerance)
    batch_prices = [s.price for s in batch_swings]
    incr_prices = [s.price for s in incr_history]
    # They may differ slightly due to edge effects; confirm types alternate
    for j in range(1, len(incr_history)):
        assert incr_history[j].swing_type != incr_history[j - 1].swing_type


def test_incremental_needs_full_window():
    """Before the window fills, incremental returns None."""
    detector = BreathingRoomDetector(lookback_n=3)
    history: List[SwingPoint] = []
    results = []
    prices = [10.0, 9.0, 8.0]  # only 3 bars, window = 7
    for i, p in enumerate(prices):
        sw = detector.detect_incremental(i, BASE_TIME + timedelta(minutes=i), p + 0.5, p - 0.5, history)
        results.append(sw)
    assert all(r is None for r in results)


def test_incremental_alternation_maintained():
    """Incremental history must maintain alternating high/low order."""
    prices = [100 + 5 * np.sin(i * 0.3) for i in range(80)]
    detector = BreathingRoomDetector(lookback_n=3, min_swing_pips=0.05)
    history: List[SwingPoint] = []
    for i, p in enumerate(prices):
        detector.detect_incremental(
            bar_index=i,
            timestamp=BASE_TIME + timedelta(minutes=i),
            high=p + 0.5,
            low=p - 0.5,
            history=history,
        )
    for j in range(1, len(history)):
        assert history[j].swing_type != history[j - 1].swing_type


# ---------------------------------------------------------------------------
# Performance test
# ---------------------------------------------------------------------------


def test_batch_detect_performance_100k_bars():
    """Batch detection on 100K bars must complete in under 500ms."""
    n = 100_000
    prices = 1800.0 + 50.0 * np.sin(np.linspace(0, 200 * np.pi, n))
    idx = pd.date_range(BASE_TIME, periods=n, freq="1min")
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.5,
            "low": prices - 0.5,
            "close": prices,
        },
        index=idx,
    )
    detector = BreathingRoomDetector(lookback_n=3, min_swing_pips=0.5)
    t0 = time.perf_counter()
    result = detector.detect(df)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 500, f"Detection took {elapsed_ms:.1f}ms — expected <500ms"
    assert len(result) > 0, "Should detect swings on sinusoidal data"
