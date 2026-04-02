"""Full rolling-window orchestrator for the Creative Pattern Discovery Agent.

Ties Phases 1-4 together into a 30-day (22 trading day) rolling challenge
loop with layered memory, walk-forward validation, and dashboard integration.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DiscoveryOrchestrator:
    """Rolling-window discovery orchestrator.

    Slices historical data into 22-trading-day windows, runs backtest
    per window, invokes post-backtest discovery phases, applies validated
    changes, and tracks challenge pass/fail.

    Parameters
    ----------
    config:
        Discovery configuration dict (from config/discovery.yaml).
    knowledge_dir:
        Base directory for the layered memory system.
    edges_yaml_path:
        Path to config/edges.yaml for long-term absorption.
    data_file:
        Path to the historical data parquet file.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        knowledge_dir: str = "reports/agent_knowledge",
        edges_yaml_path: str = "config/edges.yaml",
        data_file: Optional[str] = None,
    ) -> None:
        self._config = config or {}
        self._orch_cfg = self._config.get("orchestrator", {})
        self._disc_cfg = self._config.get("discovery", {})
        self._val_cfg = self._config.get("validation", {})
        self._challenge_cfg = self._config.get("challenge", {})
        self._report_cfg = self._config.get("reporting", {})

        self._window_size = int(self._orch_cfg.get("window_size_trading_days", 22))
        self._max_windows = int(self._orch_cfg.get("max_windows", 12))
        self._strategy_name = str(self._orch_cfg.get("strategy_name", "sss"))

        self._knowledge_dir = knowledge_dir
        self._edges_yaml_path = edges_yaml_path
        self._data_file = data_file

    # ------------------------------------------------------------------
    # Data slicing
    # ------------------------------------------------------------------

    def slice_into_windows(
        self,
        candles: pd.DataFrame,
        min_bars_per_window: int = 100,
    ) -> List[Dict[str, Any]]:
        """Slice candle data into non-overlapping rolling windows.

        Each window spans ``window_size_trading_days`` trading days.
        A trading day is defined as a calendar date (UTC) with at least
        one bar of data.

        Parameters
        ----------
        candles:
            1-minute OHLCV DataFrame with UTC DatetimeIndex.
        min_bars_per_window:
            Windows with fewer bars than this are discarded.

        Returns
        -------
        List of dicts, each with:
            - window_id: str (e.g. "w_000")
            - window_index: int
            - candles: pd.DataFrame slice
            - start_date: datetime
            - end_date: datetime
            - trading_days: int
        """
        # Get unique trading days (calendar dates with data)
        dates = candles.index.normalize().unique().sort_values()
        trading_days = dates.tolist()

        windows: List[Dict[str, Any]] = []
        idx = 0
        window_index = 0

        while idx < len(trading_days) and window_index < self._max_windows:
            end_idx = min(idx + self._window_size, len(trading_days))
            window_dates = trading_days[idx:end_idx]

            if len(window_dates) < 5:  # skip tiny remnants
                break

            # Slice candles for this window
            start_dt = window_dates[0]
            end_dt = window_dates[-1] + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            window_candles = candles.loc[start_dt:end_dt]

            if len(window_candles) >= min_bars_per_window:
                windows.append({
                    "window_id": f"w_{window_index:03d}",
                    "window_index": window_index,
                    "candles": window_candles,
                    "start_date": window_dates[0],
                    "end_date": window_dates[-1],
                    "trading_days": len(window_dates),
                })
                window_index += 1

            idx = end_idx

        logger.info(
            "Sliced %d trading days into %d windows of %d days",
            len(trading_days), len(windows), self._window_size,
        )
        return windows
