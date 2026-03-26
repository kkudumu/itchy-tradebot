"""Rolling walk-forward analysis for Ichimoku strategy optimization.

Design
------
The analyzer slides a window through the full dataset:

  |<-- in-sample (IS) 12 mo -->|<-- out-of-sample (OOS) 3 mo -->|
                              |<-- IS 12 mo -->|<-- OOS 3 mo -->|
                                              ...

For each window:
1. Optimize on the IS slice (Optuna, *n_trials* per window).
2. Evaluate the best parameters on the OOS slice (single backtest).
3. Record IS and OOS Sharpe ratios, trades, and prop firm metrics.

After all windows are processed, Walk-Forward Efficiency (WFE) is
computed as::

    WFE = mean(OOS Sharpe) / mean(IS Sharpe)

A WFE > 0.5 indicates that the strategy generalises reasonably well.
Values below 0.5 suggest overfitting to the in-sample period.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Default window lengths expressed in calendar months.
_DEFAULT_IS_MONTHS: int = 12
_DEFAULT_OOS_MONTHS: int = 3

# Minimum bars required in a slice before running a backtest.
_MIN_BARS: int = 100


@dataclass
class WFWindow:
    """Results for a single walk-forward window."""

    window_index: int
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp

    # Sharpe ratios from in-sample optimization and OOS evaluation.
    is_sharpe: float
    oos_sharpe: float

    # Best parameters found during IS optimization.
    best_params: dict = field(default_factory=dict)

    # OOS metrics dict from PerformanceMetrics.calculate().
    oos_metrics: dict = field(default_factory=dict)

    # Closed trade list from the OOS backtest.
    oos_trades: list = field(default_factory=list)

    # Prop firm constraint status on the OOS slice.
    oos_prop_firm: dict = field(default_factory=dict)


@dataclass
class WFResult:
    """Aggregate results from a complete walk-forward run.

    Attributes
    ----------
    windows:
        Per-window results in chronological order.
    wfe:
        Walk-Forward Efficiency = mean(OOS Sharpe) / mean(IS Sharpe).
        A value > 0.5 indicates acceptable out-of-sample generalisation.
    oos_trades:
        Concatenated OOS trade list from all windows.
    oos_metrics:
        Aggregated OOS metrics: mean Sharpe, mean return, mean win rate,
        total trade count, and per-window breakdown.
    is_sharpes:
        List of IS Sharpe values, one per window.
    oos_sharpes:
        List of OOS Sharpe values, one per window.
    """

    windows: List[WFWindow]
    wfe: float
    oos_trades: list
    oos_metrics: dict
    is_sharpes: List[float]
    oos_sharpes: List[float]


class WalkForwardAnalyzer:
    """Rolling walk-forward optimizer and validator.

    Parameters
    ----------
    is_months:
        Length of the in-sample optimization window in calendar months.
        Default: 12.
    oos_months:
        Length of the out-of-sample validation window in calendar months.
        Default: 3.
    initial_balance:
        Starting account equity for every backtest run.  Default: 10 000.
    """

    def __init__(
        self,
        is_months: int = _DEFAULT_IS_MONTHS,
        oos_months: int = _DEFAULT_OOS_MONTHS,
        initial_balance: float = 10_000.0,
    ) -> None:
        self._is_months = is_months
        self._oos_months = oos_months
        self._initial_balance = initial_balance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        data: pd.DataFrame,
        n_trials: int = 200,
        storage: Optional[str] = None,
    ) -> WFResult:
        """Execute the full rolling walk-forward analysis.

        Parameters
        ----------
        data:
            1-minute OHLCV ``DataFrame`` with a UTC DatetimeIndex.
        n_trials:
            Number of Optuna trials per IS window.  Default: 200.
        storage:
            Optional PostgreSQL URL for trial persistence.

        Returns
        -------
        WFResult
            Aggregate statistics and per-window detail.
        """
        windows_def = self._build_windows(data)
        if not windows_def:
            logger.warning(
                "No walk-forward windows could be constructed from the "
                "available data (%d bars, IS=%d mo, OOS=%d mo).",
                len(data),
                self._is_months,
                self._oos_months,
            )
            return WFResult(
                windows=[],
                wfe=0.0,
                oos_trades=[],
                oos_metrics={},
                is_sharpes=[],
                oos_sharpes=[],
            )

        wf_windows: List[WFWindow] = []

        for idx, (is_start, is_end, oos_start, oos_end) in enumerate(windows_def):
            logger.info(
                "Walk-forward window %d/%d — IS: %s→%s, OOS: %s→%s",
                idx + 1,
                len(windows_def),
                is_start.date(),
                is_end.date(),
                oos_start.date(),
                oos_end.date(),
            )

            is_slice = data.loc[is_start:is_end]
            oos_slice = data.loc[oos_start:oos_end]

            if len(is_slice) < _MIN_BARS or len(oos_slice) < _MIN_BARS:
                logger.warning(
                    "Window %d skipped — insufficient bars (IS=%d, OOS=%d).",
                    idx + 1,
                    len(is_slice),
                    len(oos_slice),
                )
                continue

            # Optimize on the IS slice.
            best_params, is_sharpe = self._optimize_is(
                is_slice, n_trials, storage, window_idx=idx
            )

            # Evaluate on the OOS slice using the best IS parameters.
            oos_sharpe, oos_metrics, oos_trades, oos_prop = self._evaluate_oos(
                oos_slice, best_params
            )

            wf_windows.append(
                WFWindow(
                    window_index=idx,
                    is_start=is_start,
                    is_end=is_end,
                    oos_start=oos_start,
                    oos_end=oos_end,
                    is_sharpe=is_sharpe,
                    oos_sharpe=oos_sharpe,
                    best_params=best_params,
                    oos_metrics=oos_metrics,
                    oos_trades=oos_trades,
                    oos_prop_firm=oos_prop,
                )
            )

        return self._aggregate_results(wf_windows)

    def walk_forward_efficiency(
        self, is_sharpe: List[float], oos_sharpe: List[float]
    ) -> float:
        """Compute Walk-Forward Efficiency.

        Parameters
        ----------
        is_sharpe:
            List of IS Sharpe ratios, one per window.
        oos_sharpe:
            List of OOS Sharpe ratios, one per window.

        Returns
        -------
        float
            WFE = mean(OOS Sharpe) / mean(IS Sharpe).  Returns 0.0 when
            either list is empty or the IS mean is zero.
        """
        if not is_sharpe or not oos_sharpe:
            return 0.0

        mean_is = sum(is_sharpe) / len(is_sharpe)
        mean_oos = sum(oos_sharpe) / len(oos_sharpe)

        if mean_is == 0.0:
            return 0.0

        return mean_oos / mean_is

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_windows(
        self, data: pd.DataFrame
    ) -> List[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Generate (is_start, is_end, oos_start, oos_end) tuples.

        The window slides forward by *oos_months* on each iteration so
        consecutive windows share no out-of-sample data.
        """
        if data.empty:
            return []

        data_start = data.index[0]
        data_end = data.index[-1]

        window_size = pd.DateOffset(months=self._is_months)
        oos_size = pd.DateOffset(months=self._oos_months)

        windows = []
        is_start = data_start

        while True:
            is_end = is_start + window_size
            oos_start = is_end
            oos_end = oos_start + oos_size

            if oos_end > data_end:
                break

            windows.append((is_start, is_end, oos_start, oos_end))
            is_start = is_start + oos_size  # slide by one OOS block

        return windows

    def _optimize_is(
        self,
        is_data: pd.DataFrame,
        n_trials: int,
        storage: Optional[str],
        window_idx: int,
    ) -> tuple[dict, float]:
        """Run Optuna on the IS slice and return (best_params, is_sharpe)."""
        from src.optimization.optuna_runner import OptunaOptimizer

        optimizer = OptunaOptimizer(
            data=is_data,
            initial_balance=self._initial_balance,
        )
        study_name = f"wf_window_{window_idx}"
        study = optimizer.optimize_single(
            n_trials=n_trials,
            storage=storage,
            study_name=study_name,
        )

        best_params = optimizer.get_best_params(study)

        # Re-evaluate IS performance to record the IS Sharpe.
        _, is_sharpe, _, _ = self._evaluate_oos(is_data, best_params)

        return best_params, is_sharpe

    def _evaluate_oos(
        self, data: pd.DataFrame, params: dict
    ) -> tuple[float, dict, list, dict]:
        """Run one backtest on *data* with *params*.

        Returns
        -------
        (oos_sharpe, metrics, trades, prop_firm_dict)
        """
        from src.backtesting.vectorbt_engine import IchimokuBacktester

        try:
            bt = IchimokuBacktester(
                config=params,
                initial_balance=self._initial_balance,
            )
            result = bt.run(data, log_trades=False)
            sharpe = float(result.metrics.get("sharpe_ratio") or 0.0)
            import math
            if not math.isfinite(sharpe):
                sharpe = 0.0
            return sharpe, result.metrics, result.trades, result.prop_firm
        except Exception as exc:
            logger.debug("OOS evaluation failed: %s", exc)
            return 0.0, {}, [], {}

    def _aggregate_results(self, windows: List[WFWindow]) -> WFResult:
        """Build the final ``WFResult`` from all completed windows."""
        if not windows:
            return WFResult(
                windows=[],
                wfe=0.0,
                oos_trades=[],
                oos_metrics={},
                is_sharpes=[],
                oos_sharpes=[],
            )

        is_sharpes = [w.is_sharpe for w in windows]
        oos_sharpes = [w.oos_sharpe for w in windows]
        wfe = self.walk_forward_efficiency(is_sharpes, oos_sharpes)

        all_trades = []
        for w in windows:
            all_trades.extend(w.oos_trades)

        # Aggregate key OOS metrics across windows.
        n = len(windows)
        agg_metrics = {
            "mean_oos_sharpe": sum(oos_sharpes) / n,
            "mean_is_sharpe": sum(is_sharpes) / n,
            "wfe": wfe,
            "total_oos_trades": len(all_trades),
            "mean_oos_return_pct": (
                sum(w.oos_metrics.get("total_return_pct", 0.0) for w in windows) / n
            ),
            "mean_oos_win_rate": (
                sum(w.oos_metrics.get("win_rate", 0.0) for w in windows) / n
            ),
            "n_windows": n,
            "windows": [
                {
                    "index": w.window_index,
                    "is_sharpe": w.is_sharpe,
                    "oos_sharpe": w.oos_sharpe,
                    "oos_return_pct": w.oos_metrics.get("total_return_pct", 0.0),
                    "oos_trades": len(w.oos_trades),
                }
                for w in windows
            ],
        }

        logger.info(
            "Walk-forward complete: %d windows, WFE=%.3f, "
            "mean_OOS_sharpe=%.3f, total_OOS_trades=%d",
            n,
            wfe,
            agg_metrics["mean_oos_sharpe"],
            agg_metrics["total_oos_trades"],
        )

        return WFResult(
            windows=windows,
            wfe=wfe,
            oos_trades=all_trades,
            oos_metrics=agg_metrics,
            is_sharpes=is_sharpes,
            oos_sharpes=oos_sharpes,
        )
