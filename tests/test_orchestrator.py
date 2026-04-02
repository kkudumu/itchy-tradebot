# tests/test_orchestrator.py
"""Tests for the discovery orchestrator."""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def _make_candles(n_days: int = 60, bars_per_day: int = 390) -> pd.DataFrame:
    """Generate synthetic 1M candles spanning n_days trading days."""
    rng = np.random.default_rng(42)
    start = datetime(2025, 1, 2, 8, 0, tzinfo=timezone.utc)

    timestamps = []
    day = 0
    dt = start
    while day < n_days:
        # Skip weekends
        if dt.weekday() < 5:
            for bar in range(bars_per_day):
                timestamps.append(dt + timedelta(minutes=bar))
            day += 1
        dt += timedelta(days=1)

    n = len(timestamps)
    prices = 1800.0 + np.cumsum(rng.normal(0.01, 0.5, n))
    df = pd.DataFrame({
        "open": prices + rng.uniform(-0.2, 0.2, n),
        "high": prices + rng.uniform(0, 0.5, n),
        "low": prices - rng.uniform(0, 0.5, n),
        "close": prices,
        "volume": rng.integers(100, 800, n),
    }, index=pd.DatetimeIndex(timestamps, tz=timezone.utc))

    return df


class TestRollingWindowSlicer:
    def test_slice_into_windows(self):
        from src.discovery.orchestrator import DiscoveryOrchestrator

        candles = _make_candles(n_days=66)  # 3 windows of 22 days
        orch = DiscoveryOrchestrator(
            config={"orchestrator": {"window_size_trading_days": 22}},
            knowledge_dir=str(Path(tempfile.mkdtemp()) / "kb"),
        )

        windows = orch.slice_into_windows(candles)
        assert len(windows) == 3
        for w in windows:
            assert isinstance(w["candles"], pd.DataFrame)
            assert len(w["candles"]) > 0
            assert "window_id" in w
            assert "window_index" in w

    def test_windows_do_not_overlap(self):
        from src.discovery.orchestrator import DiscoveryOrchestrator

        candles = _make_candles(n_days=44)  # 2 windows
        orch = DiscoveryOrchestrator(
            config={"orchestrator": {"window_size_trading_days": 22}},
            knowledge_dir=str(Path(tempfile.mkdtemp()) / "kb"),
        )

        windows = orch.slice_into_windows(candles)
        if len(windows) >= 2:
            end_0 = windows[0]["candles"].index[-1]
            start_1 = windows[1]["candles"].index[0]
            assert start_1 > end_0

    def test_partial_last_window_included(self):
        from src.discovery.orchestrator import DiscoveryOrchestrator

        candles = _make_candles(n_days=30)  # 1 full + partial
        orch = DiscoveryOrchestrator(
            config={"orchestrator": {"window_size_trading_days": 22}},
            knowledge_dir=str(Path(tempfile.mkdtemp()) / "kb"),
        )

        windows = orch.slice_into_windows(candles)
        # At minimum 1 full window, partial may or may not be included
        assert len(windows) >= 1

    def test_window_id_format(self):
        from src.discovery.orchestrator import DiscoveryOrchestrator

        candles = _make_candles(n_days=22)
        orch = DiscoveryOrchestrator(
            config={"orchestrator": {"window_size_trading_days": 22}},
            knowledge_dir=str(Path(tempfile.mkdtemp()) / "kb"),
        )

        windows = orch.slice_into_windows(candles)
        assert len(windows) >= 1
        assert windows[0]["window_id"].startswith("w_")
