"""Selective trade screenshot capture for visual analysis.

After each 30-day backtest window, selects the most informative trades
for Claude visual analysis: worst losers (what went wrong?), best winners
(what pattern to repeat?), and near-SL survivors (lucky or skilled?).

Generates annotated mplfinance charts with entry/exit markers and
optional pattern overlays.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ScreenshotSelector:
    """Selects and captures trade screenshots for visual analysis.

    Parameters
    ----------
    n_worst:
        Number of worst losers to capture. Default 5.
    n_best:
        Number of best winners to capture. Default 5.
    near_sl_threshold:
        R-multiple threshold for "near stop-loss" detection. A winning
        trade that dipped below -(1.0 - threshold)R during its life is
        considered a near-SL exit. Default 0.2 (dipped below -0.8R).
    bars_before:
        Number of bars before entry to include in chart. Default 60.
    bars_after:
        Number of bars after exit to include in chart. Default 20.
    """

    def __init__(
        self,
        n_worst: int = 5,
        n_best: int = 5,
        near_sl_threshold: float = 0.2,
        bars_before: int = 60,
        bars_after: int = 20,
    ) -> None:
        self._n_worst = n_worst
        self._n_best = n_best
        self._near_sl_thresh = near_sl_threshold
        self._bars_before = bars_before
        self._bars_after = bars_after

    def select_trades(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select the most informative trades for screenshot capture.

        Returns copies of trade dicts with a '_selection_reason' field
        indicating why each trade was selected.
        """
        if not trades:
            return []

        # Sort for worst losers (lowest R first)
        sorted_by_r = sorted(trades, key=lambda t: float(t.get("r_multiple", 0)))
        losers = [t for t in sorted_by_r if float(t.get("r_multiple", 0)) < 0]
        winners = [t for t in reversed(sorted_by_r) if float(t.get("r_multiple", 0)) > 0]

        selected: Dict[int, Dict[str, Any]] = {}  # keyed by id(trade) to prevent dupes

        # Worst losers
        for t in losers[:self._n_worst]:
            entry = {**t, "_selection_reason": "worst_loser"}
            selected[id(t)] = entry

        # Best winners
        for t in winners[:self._n_best]:
            if id(t) not in selected:
                selected[id(t)] = {**t, "_selection_reason": "best_winner"}

        # Near-SL exits: winning trades that nearly hit their stop
        sl_cutoff = -(1.0 - self._near_sl_thresh)
        for t in trades:
            if id(t) in selected:
                continue
            r = float(t.get("r_multiple", 0))
            min_r = float(t.get("min_r_during_trade", 0))
            if r > 0 and min_r < sl_cutoff:
                selected[id(t)] = {**t, "_selection_reason": "near_sl_exit"}

        return list(selected.values())

    def generate_chart(
        self,
        trade: Dict[str, Any],
        df_5m: pd.DataFrame,
        output_dir: str,
        patterns: Optional[List] = None,
        instrument: str = "XAUUSD",
    ) -> Optional[str]:
        """Generate an annotated mplfinance chart for a selected trade.

        Parameters
        ----------
        trade: Trade dict with entry_bar_idx, exit_bar_idx, direction, etc.
        df_5m: Full 5M OHLCV DataFrame from the backtest.
        output_dir: Directory to save PNG files.
        patterns: Optional list of ChartPattern objects to overlay.
        instrument: Instrument symbol for the chart title.

        Returns
        -------
        Path to saved PNG, or None on failure.
        """
        try:
            import mplfinance as mpf
        except ImportError:
            logger.warning("mplfinance not installed -- skipping chart generation")
            return None

        entry_idx = int(trade.get("entry_bar_idx", 0))
        exit_idx = int(trade.get("exit_bar_idx", entry_idx + 30))

        start = max(0, entry_idx - self._bars_before)
        end = min(len(df_5m), exit_idx + self._bars_after)
        chunk = df_5m.iloc[start:end].copy()

        if chunk.empty or len(chunk) < 5:
            return None

        # Ensure DatetimeIndex
        if not isinstance(chunk.index, pd.DatetimeIndex):
            chunk.index = pd.DatetimeIndex(chunk.index)

        # Rename tick_volume if needed
        if "tick_volume" in chunk.columns and "volume" not in chunk.columns:
            chunk = chunk.rename(columns={"tick_volume": "volume"})

        # Entry/exit markers
        entry_local = entry_idx - start
        exit_local = exit_idx - start
        entry_local = max(0, min(entry_local, len(chunk) - 1))
        exit_local = max(0, min(exit_local, len(chunk) - 1))

        entry_marker = np.full(len(chunk), np.nan)
        exit_marker = np.full(len(chunk), np.nan)
        entry_marker[entry_local] = chunk.iloc[entry_local]["low"] * 0.999
        exit_marker[exit_local] = chunk.iloc[exit_local]["high"] * 1.001

        apds = [
            mpf.make_addplot(entry_marker, type="scatter", markersize=120, marker="^", color="green"),
            mpf.make_addplot(exit_marker, type="scatter", markersize=120, marker="v", color="red"),
        ]

        # Build title
        r = float(trade.get("r_multiple", 0))
        reason = trade.get("_selection_reason", "")
        direction = trade.get("direction", "")
        title = f"{instrument} 5M -- {direction.upper()} ({r:+.2f}R) [{reason}]"

        # Output path
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        entry_time = trade.get("entry_time")
        ts_str = str(entry_time).replace(" ", "_").replace(":", "")[:15] if entry_time else "unknown"
        filename = f"{reason}_{r:+.2f}R_{ts_str}.png"
        file_path = out_path / filename

        try:
            mpf.plot(
                chunk,
                type="candle",
                style="charles",
                title=title,
                volume=("volume" in chunk.columns),
                addplot=apds,
                savefig=str(file_path),
                figsize=(14.0, 8.0),
            )
            import matplotlib.pyplot as plt
            plt.close("all")
            return str(file_path)
        except Exception as exc:
            logger.debug("Chart generation failed: %s", exc)
            return None
