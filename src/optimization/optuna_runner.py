"""Main Optuna optimization runner.

Provides ``OptunaOptimizer`` which manages Optuna studies for both single-
objective (TPE + MedianPruner) and multi-objective (NSGA-II) optimization
of the Ichimoku strategy against prop firm constraints.

Storage
-------
Pass a PostgreSQL connection URL as *storage* to persist trials across
sessions::

    storage = "postgresql://trader:password@localhost/trading"
    optimizer.optimize_single(storage=storage)

When *storage* is ``None`` an in-memory study is created (useful for
unit tests and quick local runs).

Trial budget
------------
The default of 300 trials balances exploration coverage with wall-clock
time.  For the multi-objective NSGA-II study, 300 trials are also the
default because population diversity is handled by the sampler itself
rather than by trial count alone.
"""

from __future__ import annotations

import logging
from typing import Optional

import optuna
import pandas as pd

from src.optimization.objectives import MultiObjective, PropFirmObjective

# Register SSS parameter space in STRATEGY_REGISTRY so that
# OptunaOptimizer(strategy_key='sss') resolves suggest_params correctly.
import src.strategy.strategies.sss.optuna_params  # noqa: F401

logger = logging.getLogger(__name__)

# Silence Optuna's verbose per-trial logging by default.
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaOptimizer:
    """High-level interface to Optuna studies for strategy optimization.

    Parameters
    ----------
    data:
        1-minute OHLCV ``DataFrame`` (UTC DatetimeIndex).  All optimization
        runs use the same dataset; walk-forward slicing is handled by
        ``WalkForwardAnalyzer``.
    config:
        Optional base strategy configuration dict forwarded to the
        backtester constructor.
    initial_balance:
        Starting account equity used for every backtest.  Default: 10 000.
    strategy_key:
        Key into ``STRATEGY_REGISTRY`` that selects the strategy whose
        ``suggest_params`` defines the Optuna search space.  Defaults to
        ``'ichimoku'`` for backward compatibility.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config: Optional[dict] = None,
        initial_balance: float = 10_000.0,
        strategy_key: str = 'ichimoku',
    ) -> None:
        self._data = data
        self._config = config or {}
        self._initial_balance = initial_balance
        self._strategy_key = strategy_key

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize_single(
        self,
        n_trials: int = 300,
        storage: Optional[str] = None,
        study_name: str = "ichimoku_single",
        n_jobs: int = 1,
    ) -> optuna.Study:
        """Run single-objective optimization maximising the prop firm score.

        Uses the TPE sampler with a ``MedianPruner`` that discards trials
        whose intermediate Sharpe drops below the median of completed trials
        after 20 warm-up trials.

        Parameters
        ----------
        n_trials:
            Number of trials to evaluate.  Recommended: 300–500.
        storage:
            Optional PostgreSQL URL for persistent trial storage.
            Example: ``"postgresql://user:pass@host/db"``
        study_name:
            Logical name used by Optuna for the study.  The same name
            retrieves an existing study from *storage*.
        n_jobs:
            Parallel trial workers.  Keep at 1 for reproducible results;
            increase only when the backtester is thread-safe.

        Returns
        -------
        optuna.Study
            Completed study with ``study.best_trial`` populated.
        """
        sampler = optuna.samplers.TPESampler(seed=42)
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=20,
            n_warmup_steps=10,
        )

        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=True,
        )

        objective = PropFirmObjective(
            data=self._data,
            initial_balance=self._initial_balance,
            strategy_key=self._strategy_key,
        )

        logger.info(
            "Starting single-objective study '%s' — %d trials, storage=%s",
            study_name,
            n_trials,
            "postgresql" if storage else "in-memory",
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=False,
        )

        logger.info(
            "Study complete — best score=%.4f, params=%s",
            study.best_value,
            study.best_params,
        )
        return study

    def optimize_multi(
        self,
        n_trials: int = 300,
        storage: Optional[str] = None,
        study_name: str = "ichimoku_multi",
        n_jobs: int = 1,
    ) -> optuna.Study:
        """Run NSGA-II multi-objective optimization.

        Optimises three objectives simultaneously:
        - Sortino ratio (higher is better)
        - Drawdown safety (20 - max_dd_pct, higher is better)
        - Calmar ratio (higher is better)

        Parameters
        ----------
        n_trials:
            Number of trials for the NSGA-II population search.
        storage:
            Optional PostgreSQL URL for persistent trial storage.
        study_name:
            Logical study name.
        n_jobs:
            Parallel trial workers.

        Returns
        -------
        optuna.Study
            Completed study; use ``study.best_trials`` to access the
            Pareto-optimal set.
        """
        sampler = optuna.samplers.NSGAIISampler(seed=42)

        study = optuna.create_study(
            study_name=study_name,
            directions=["maximize", "maximize", "maximize"],
            sampler=sampler,
            storage=storage,
            load_if_exists=True,
        )

        objective = MultiObjective(
            data=self._data,
            initial_balance=self._initial_balance,
            strategy_key=self._strategy_key,
        )

        logger.info(
            "Starting NSGA-II multi-objective study '%s' — %d trials",
            study_name,
            n_trials,
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=False,
        )

        logger.info(
            "NSGA-II study complete — %d Pareto-optimal trials",
            len(study.best_trials),
        )
        return study

    def get_best_params(self, study: optuna.Study) -> dict:
        """Extract and validate the best parameter set from a completed study.

        For single-objective studies, returns the params from
        ``study.best_trial``.

        For multi-objective studies, selects the Pareto-optimal trial that
        maximises the composite score ``sortino + dd_safety + calmar``.

        Parameters
        ----------
        study:
            Completed Optuna study (single or multi-objective).

        Returns
        -------
        dict
            Parameter dict ready for use as the *config* argument to
            ``IchimokuBacktester``.
        """
        # Detect multi-objective study by number of directions.
        if len(study.directions) > 1:
            return self._best_pareto_params(study)

        trial = study.best_trial
        return self._scale_ichimoku_params(trial.params)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scale_ichimoku_params(raw_params: dict) -> dict:
        """Convert ``ichimoku_scale`` in *raw_params* to period integers.

        Optuna stores the raw sampled values.  The backtester expects
        integer period keys.  This method mirrors the logic in
        ``PropFirmObjective.suggest_params``.
        """
        params = dict(raw_params)
        scale = params.pop("ichimoku_scale", 1.0)
        params["tenkan_period"] = max(3, round(9 * scale))
        params["kijun_period"] = max(9, round(26 * scale))
        params["senkou_b_period"] = max(18, round(52 * scale))
        # Rename optimizer keys to backtester config keys where they differ.
        if "atr_stop_mult" in params:
            params["atr_stop_multiplier"] = params.pop("atr_stop_mult")
        if "min_confluence" in params:
            params["min_confluence_score"] = params.pop("min_confluence")
        if "risk_initial" in params:
            params["initial_risk_pct"] = params.pop("risk_initial")
        if "risk_reduced" in params:
            params["reduced_risk_pct"] = params.pop("risk_reduced")
        return params

    @staticmethod
    def _best_pareto_params(study: optuna.Study) -> dict:
        """Select the best Pareto trial by composite score sum."""
        best_trial = None
        best_score = -float("inf")
        for trial in study.best_trials:
            score = sum(v for v in trial.values if v is not None)
            if score > best_score:
                best_score = score
                best_trial = trial
        if best_trial is None:
            raise ValueError("No completed Pareto trials found in study.")
        return OptunaOptimizer._scale_ichimoku_params(best_trial.params)
