"""Custom Optuna objective functions for prop firm challenge optimization.

Objective design
----------------
Single-objective (PropFirmObjective):
    Maximise a composite score that rewards high Sharpe ratio while
    penalising breaches of The5ers daily/total drawdown limits and
    insufficient return.  Hard-failing strategies (0 trades, total DD
    breach) return 0.0 so Optuna's pruner discards them quickly.

Multi-objective (MultiObjective):
    Returns (Sortino, -max_dd, Calmar) for NSGA-II so the Pareto front
    balances return quality, drawdown safety, and risk-adjusted return.

Parameter space
---------------
All Ichimoku periods are derived from a single ``ichimoku_scale`` float
that preserves the canonical 1:3:6 ratio (tenkan:kijun:senkou_b).
Additional parameters cover ADX threshold, ATR stop multiplier, take-
profit R-multiple, Kijun trail start, minimum confluence, and position
risk percentages.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np
import optuna
import pandas as pd

if TYPE_CHECKING:
    from src.backtesting.vectorbt_engine import IchimokuBacktester

logger = logging.getLogger(__name__)

# Base Ichimoku periods — scale factor multiplies all three simultaneously
# to preserve the 1:3:6 ratio.
_BASE_TENKAN: int = 9
_BASE_KIJUN: int = 26
_BASE_SENKOU_B: int = 52


def _safe_float(value: object, default: float = 0.0) -> float:
    """Return a finite float from *value*, or *default* on NaN/None/inf."""
    if value is None:
        return default
    try:
        f = float(value)  # type: ignore[arg-type]
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


# =============================================================================
# PropFirmObjective
# =============================================================================


class PropFirmObjective:
    """Single-objective Optuna objective targeting prop firm passage.

    Parameters
    ----------
    backtester:
        An instantiated ``IchimokuBacktester``.  A new one is constructed
        with each trial's parameters so the instance here is used only as
        a template to copy configuration defaults.
    data:
        1-minute OHLCV ``DataFrame`` (UTC DatetimeIndex) used for every
        backtest run.
    initial_balance:
        Starting account equity forwarded to the backtester.
    """

    def __init__(
        self,
        backtester: "IchimokuBacktester",
        data: pd.DataFrame,
        initial_balance: float = 10_000.0,
    ) -> None:
        self._backtester = backtester
        self._data = data
        self._initial_balance = initial_balance

    # ------------------------------------------------------------------
    # Optuna callable
    # ------------------------------------------------------------------

    def __call__(self, trial: optuna.Trial) -> float:
        """Evaluate one parameter combination and return a composite score.

        Score logic
        -----------
        1. Run the backtest with the trial's parameters.
        2. If no trades executed, return 0.0 (unviable strategy).
        3. Apply prop firm penalty rules:
           - ``max_daily_dd > 5 %``   → multiply score by 0.5
           - ``max_total_dd > 10 %``  → return 0.0 (hard fail)
           - ``total_return < 8 %``   → multiply score by (return / 8)
           - ``win_rate < 0.45``      → multiply score by 0.8
        4. Return ``sharpe_ratio * multiplier`` (clamped to [0, inf)).
        """
        params = self.suggest_params(trial)
        result = self._run_backtest(params)

        if result is None:
            return 0.0

        metrics = result.metrics
        prop = result.prop_firm

        n_trades = _safe_float(metrics.get("total_trades"), 0.0)
        if n_trades == 0:
            return 0.0

        sharpe = _safe_float(metrics.get("sharpe_ratio"), 0.0)
        if sharpe <= 0.0:
            return 0.0

        max_daily_dd = _safe_float(prop.get("max_daily_dd_pct"), 0.0)
        max_total_dd = _safe_float(prop.get("max_total_dd_pct"), 0.0)
        total_return = _safe_float(metrics.get("total_return_pct"), 0.0)
        win_rate = _safe_float(metrics.get("win_rate"), 0.0)

        # Hard failure — total DD breach disqualifies the strategy outright.
        if max_total_dd > 10.0:
            return 0.0

        multiplier = 1.0

        # Soft penalty — daily DD approaching the 5 % limit.
        if max_daily_dd > 5.0:
            multiplier *= 0.5

        # Insufficient return penalty — scale down proportionally.
        if total_return < 8.0:
            if total_return <= 0.0:
                return 0.0
            multiplier *= total_return / 8.0

        # Low win-rate penalty.
        if win_rate < 0.45:
            multiplier *= 0.8

        score = sharpe * multiplier
        return max(0.0, score)

    # ------------------------------------------------------------------
    # Parameter space
    # ------------------------------------------------------------------

    def suggest_params(self, trial: optuna.Trial) -> dict:
        """Define and sample the parameter search space for one trial.

        All Ichimoku periods derive from a single ``ichimoku_scale`` factor
        applied to the canonical 9/26/52 base periods so the 1:3:6 ratio
        is preserved throughout the search.

        Returns
        -------
        dict
            Parameter dict suitable for passing directly to
            ``IchimokuBacktester.__init__`` as *config*.
        """
        # Single scale factor — preserves 1:3:6 Ichimoku period ratio.
        scale = trial.suggest_float("ichimoku_scale", 0.7, 1.3)
        tenkan = max(3, round(_BASE_TENKAN * scale))
        kijun = max(9, round(_BASE_KIJUN * scale))
        senkou_b = max(18, round(_BASE_SENKOU_B * scale))

        adx_threshold = trial.suggest_int("adx_threshold", 20, 40)
        atr_stop_mult = trial.suggest_float("atr_stop_mult", 1.0, 2.5)
        tp_r_multiple = trial.suggest_float("tp_r_multiple", 1.5, 3.0)
        kijun_trail_start_r = trial.suggest_float("kijun_trail_start_r", 1.0, 2.5)
        min_confluence = trial.suggest_int("min_confluence", 3, 6)
        risk_initial = trial.suggest_float("risk_initial", 0.5, 2.0)
        risk_reduced = trial.suggest_float("risk_reduced", 0.25, 1.5)

        return {
            "tenkan_period": tenkan,
            "kijun_period": kijun,
            "senkou_b_period": senkou_b,
            "adx_threshold": adx_threshold,
            "atr_stop_multiplier": atr_stop_mult,
            "tp_r_multiple": tp_r_multiple,
            "kijun_trail_start_r": kijun_trail_start_r,
            "min_confluence_score": min_confluence,
            "initial_risk_pct": risk_initial,
            "reduced_risk_pct": risk_reduced,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_backtest(self, params: dict):
        """Construct a fresh backtester with *params* and run it.

        Returns the ``BacktestResult`` or ``None`` on unhandled exception.
        """
        from src.backtesting.vectorbt_engine import IchimokuBacktester

        try:
            backtester = IchimokuBacktester(
                config=params,
                initial_balance=self._initial_balance,
            )
            return backtester.run(self._data, log_trades=False)
        except Exception as exc:
            logger.debug("Trial backtest failed: %s", exc)
            return None


# =============================================================================
# MultiObjective
# =============================================================================


class MultiObjective:
    """NSGA-II multi-objective: Sortino, drawdown safety, Calmar.

    Returns three objectives for Pareto-front optimisation:
    - Maximise Sortino ratio (return quality with downside focus)
    - Maximise drawdown safety (negative of max total DD → minimise DD)
    - Maximise Calmar ratio (annualised return per unit of max DD)

    Parameters
    ----------
    backtester:
        ``IchimokuBacktester`` instance used as a configuration template.
    data:
        1-minute OHLCV ``DataFrame`` (UTC DatetimeIndex).
    initial_balance:
        Starting account equity.
    """

    def __init__(
        self,
        backtester: "IchimokuBacktester",
        data: pd.DataFrame,
        initial_balance: float = 10_000.0,
    ) -> None:
        # Delegate parameter sampling and backtest execution to
        # PropFirmObjective so the parameter space stays consistent.
        self._single = PropFirmObjective(backtester, data, initial_balance)

    def __call__(self, trial: optuna.Trial) -> tuple[float, float, float]:
        """Return (sortino, -max_dd_pct, calmar) for NSGA-II.

        Zero-trade or failed backtests return (0.0, 0.0, 0.0) — the worst
        possible Pareto rank — so the sampler avoids this region of the
        search space.
        """
        params = self._single.suggest_params(trial)
        result = self._single._run_backtest(params)

        if result is None:
            return 0.0, 0.0, 0.0

        metrics = result.metrics
        prop = result.prop_firm
        n_trades = _safe_float(metrics.get("total_trades"), 0.0)
        if n_trades == 0:
            return 0.0, 0.0, 0.0

        sortino = _safe_float(metrics.get("sortino_ratio"), 0.0)
        max_total_dd = _safe_float(prop.get("max_total_dd_pct"), 0.0)
        calmar = _safe_float(metrics.get("calmar_ratio"), 0.0)

        # Clamp to non-negative values to avoid sign confusion in the Pareto
        # front (Optuna maximises all objectives).
        sortino = max(0.0, sortino)
        dd_safety = max(0.0, 20.0 - max_total_dd)  # higher = safer, max 20
        calmar = max(0.0, calmar)

        return sortino, dd_safety, calmar
