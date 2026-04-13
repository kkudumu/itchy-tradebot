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
When a ``strategy_key`` is provided the parameter space is delegated to
``strategy.suggest_params(trial)`` via ``STRATEGY_REGISTRY``.  When no
strategy is registered for the key the class falls back to the original
Ichimoku-specific parameter space (backward compatible).

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


def topstep_combine_pass_score(result: object) -> float:
    """Score a backtest result against TopstepX Combine pass rules.

    Used as an Optuna objective when ``prop_firm.style ==
    "topstep_combine_dollar"``. Returns a continuous value in roughly
    [-2, 1] that Optuna can maximize:

    * +1.0 if the run passed (status=="passed" from the active tracker)
    * A scaled final-balance term in [-1, 1] plus style-specific
      penalties when the run failed:
        -0.5 for an MLL breach
        -0.3 for a daily loss breach
        -0.2 for a consistency failure

    The result parameter is anything with a ``prop_firm`` dict — a
    ``BacktestResult`` from :class:`IchimokuBacktester`, or a raw dict
    shaped the same way.
    """
    prop = getattr(result, "prop_firm", None)
    if prop is None and isinstance(result, dict):
        prop = result.get("prop_firm")
    if not prop:
        return -1.0

    # Find the active topstep snapshot — it lives under "active_tracker"
    # when the legacy pct view is also present.
    active = prop.get("active_tracker") if isinstance(prop, dict) else None
    snap = active or prop

    status = str(snap.get("status") or "").lower()
    if status == "passed":
        return 1.0

    initial = float(snap.get("initial_balance") or 0.0)
    current = float(snap.get("current_balance") or 0.0)
    # Profit target — try the active tracker first, fall back to
    # config-style fields or a sensible default.
    target = float(
        snap.get("profit_target_usd")
        or snap.get("distance_to_target", 0.0) + (current - initial)
        or 3_000.0
    )
    if target <= 0:
        target = 3_000.0

    balance_score = (current - initial) / target
    balance_score = max(-1.0, min(1.0, balance_score))

    penalty = 0.0
    if status.startswith("failed_mll"):
        penalty -= 0.5
    elif status.startswith("failed_daily_loss"):
        penalty -= 0.3
    elif status.startswith("failed_consistency"):
        penalty -= 0.2

    return balance_score + penalty


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
        a template to copy configuration defaults.  May be ``None`` when
        *strategy_key* is provided and ``StrategyBacktester`` is available.
    data:
        1-minute OHLCV ``DataFrame`` (UTC DatetimeIndex) used for every
        backtest run.
    initial_balance:
        Starting account equity forwarded to the backtester.
    strategy_key:
        Key into ``STRATEGY_REGISTRY`` that selects the strategy whose
        ``suggest_params`` defines the Optuna search space and whose
        ``StrategyBacktester`` runs the backtest.  Defaults to
        ``'ichimoku'`` for backward compatibility.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10_000.0,
        strategy_key: str = 'ichimoku',
        backtester: "IchimokuBacktester" = None,
        base_config: dict | None = None,
    ) -> None:
        self._backtester = backtester
        self._data = data
        self._initial_balance = initial_balance
        self._strategy_key = strategy_key
        self._base_config = base_config or {}

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

        # Topstep-style dollar combines should optimize against the
        # combine pass objective directly instead of the generic
        # sharpe/return heuristic used by the legacy pct-based flows.
        prop_cfg = self._base_config.get("prop_firm", {}) if isinstance(self._base_config, dict) else {}
        prop_style = str(prop_cfg.get("style") or "").strip().lower()
        metrics = result.metrics
        prop = result.prop_firm
        n_trades = _safe_float(metrics.get("total_trades"), 0.0)

        if prop_style == "topstep_combine_dollar":
            if n_trades <= 0:
                return -2.0
            score = topstep_combine_pass_score(result)
            if n_trades < 5:
                return score - 0.5
            return score + min(n_trades, 20.0) / 100.0

        if n_trades == 0:
            return 0.0

        sharpe = _safe_float(metrics.get("sharpe_ratio"), 0.0)
        max_daily_dd = _safe_float(prop.get("max_daily_dd_pct"), 0.0)
        max_total_dd = _safe_float(prop.get("max_total_dd_pct"), 0.0)
        total_return = _safe_float(metrics.get("total_return_pct"), 0.0)
        win_rate = _safe_float(metrics.get("win_rate"), 0.0)
        expectancy = _safe_float(metrics.get("expectancy"), 0.0)

        # Hard failure — total DD breach disqualifies the strategy outright.
        if max_total_dd > 10.0:
            return 0.0

        # Composite score: blend return, sharpe, and win rate for gradient.
        # This gives Optuna signal even for losing strategies so it can
        # learn which direction to optimize (unlike the old sharpe-only gate).
        score = 0.0

        # Return component (dominant): range roughly [-10, +10]
        score += total_return * 0.5

        # Sharpe component: reward positive sharpe
        score += max(sharpe, -2.0) * 2.0

        # Win rate bonus
        score += (win_rate - 0.30) * 10.0  # +0 at 30%, +2 at 50%

        # Trade count bonus (enough trades for significance)
        if n_trades >= 20:
            score += 1.0

        # Drawdown penalties
        if max_daily_dd > 5.0:
            score *= 0.5
        if max_total_dd > 8.0:
            score *= 0.5

        return score

    # ------------------------------------------------------------------
    # Parameter space
    # ------------------------------------------------------------------

    def suggest_params(self, trial: optuna.Trial) -> dict:
        """Define and sample the parameter search space for one trial.

        Delegates to ``strategy.suggest_params(trial)`` when the strategy
        registered under ``self._strategy_key`` overrides that method and
        returns a non-empty dict.  Falls back to the original Ichimoku-
        specific parameter space when no strategy is found or the strategy
        returns an empty dict (backward compatible).

        All Ichimoku fallback periods derive from a single ``ichimoku_scale``
        factor applied to the canonical 9/26/52 base periods so the 1:3:6
        ratio is preserved throughout the search.

        Returns
        -------
        dict
            Parameter dict suitable for passing directly to a backtester
            as *config*.
        """
        from src.strategy.base import STRATEGY_REGISTRY

        strategy_cls = STRATEGY_REGISTRY.get(self._strategy_key)
        if strategy_cls is not None:
            strategy = strategy_cls()
            params = strategy.suggest_params(trial)
            if params:
                return params

        # Fallback: original Ichimoku-specific params (backward compat).
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

        Merges the trial's *params* on top of any *base_config* provided at
        construction time.  This ensures that edges, risk, exit, and prop
        firm config are preserved across optimization trials while the
        strategy-specific search space varies per trial.

        Returns the ``BacktestResult`` or ``None`` on unhandled exception.
        """
        from src.backtesting.vectorbt_engine import IchimokuBacktester

        # Deep-merge base config with trial params (trial wins on conflict).
        # For strategies, merge at the per-strategy level so unoptimized
        # params (warmup_bars, spread_multiplier, etc.) are preserved.
        import copy
        merged = copy.deepcopy(self._base_config)
        for k, v in params.items():
            if k == "strategies" and isinstance(v, dict) and "strategies" in merged:
                for strat_name, strat_params in v.items():
                    if strat_name in merged["strategies"] and isinstance(strat_params, dict):
                        merged["strategies"][strat_name] = {
                            **merged["strategies"][strat_name],
                            **strat_params,
                        }
                    else:
                        merged["strategies"][strat_name] = strat_params
            else:
                merged[k] = v

        try:
            # Use IchimokuBacktester directly — it already handles all
            # registered strategies (SSS, Asian Breakout, etc.) via its
            # multi-strategy loop.  StrategyBacktester requires a full
            # Strategy ABC implementation, which Optuna adapters lack.
            backtester = IchimokuBacktester(
                config=merged,
                initial_balance=self._initial_balance,
            )
            return backtester.run(self._data, log_trades=False, enable_learning=True)
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
        May be ``None`` when *strategy_key* is provided and
        ``StrategyBacktester`` is available.
    data:
        1-minute OHLCV ``DataFrame`` (UTC DatetimeIndex).
    initial_balance:
        Starting account equity.
    strategy_key:
        Key into ``STRATEGY_REGISTRY`` forwarded to the underlying
        ``PropFirmObjective``.  Defaults to ``'ichimoku'``.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10_000.0,
        strategy_key: str = 'ichimoku',
        backtester: "IchimokuBacktester" = None,
        base_config: dict | None = None,
    ) -> None:
        # Delegate parameter sampling and backtest execution to
        # PropFirmObjective so the parameter space stays consistent.
        self._single = PropFirmObjective(
            data=data,
            initial_balance=initial_balance,
            strategy_key=strategy_key,
            backtester=backtester,
            base_config=base_config,
        )

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
