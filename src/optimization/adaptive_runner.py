"""Core adaptive optimization loop.

Ties together the context embedder, experience store, signal persister,
guardrails, data manager, and backtesting engine into a single
instrument-cycling optimization pipeline.

Usage
-----
    from src.database.connection import DatabasePool
    from src.optimization.adaptive_runner import AdaptiveRunner

    pool = DatabasePool(config)
    pool.initialise()
    runner = AdaptiveRunner(db_pool=pool, trials_per_epoch=50)

    # One epoch across all instruments:
    results = runner.run_once()

    # Or filter to a single instrument:
    results = runner.run_once(instrument_filter="MGC")

    # Run forever (cycles instruments indefinitely):
    runner.run_forever()
"""

from __future__ import annotations

import copy
import json
import logging
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import yaml

from src.config.loader import ConfigLoader
from src.optimization.context_embedder import embed_market, embed_outcome, embed_params
from src.optimization.data_manager import DataManager
from src.optimization.experience_store import ExperienceStore
from src.optimization.guardrails import (
    check_consecutive_passes,
    check_permutation_significance,
)
from src.optimization.objectives import topstep_combine_pass_score
from src.optimization.signal_persister import SignalPersister

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

INITIAL_BALANCE = 50_000.0

# Mapping from optimizer instrument symbols (optimizer_instruments.yaml)
# to the canonical instrument symbols used in instruments.yaml and by
# the backtesting engine.
_OPTIMIZER_TO_ENGINE_SYMBOL: dict[str, str] = {
    "MGC": "XAUUSD",
    "MCL": "MCLOIL",
    "MNQ": "MNQ",
    "MYM": "MYM",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flat_params_from_config(params: dict) -> dict[str, Any]:
    """Convert a nested strategy config dict to the flat Optuna param names.

    This is used to seed warm-start trials via ``study.enqueue_trial()``.
    Returns the flat dict keyed by the short Optuna param names (e.g.
    ``"ri"``, ``"rr"``, ``"tp"``).  Only includes params that Optuna's
    ``_suggest_params`` would create.
    """
    flat: dict[str, Any] = {}
    risk = params.get("risk", {})
    exit_ = params.get("exit", {})
    strats = params.get("strategies", {})
    sss = strats.get("sss", {})
    ichi = strats.get("ichimoku", {})
    ichi_ichi = ichi.get("ichimoku", {})
    ichi_atr = ichi.get("atr", {})
    ichi_adx = ichi.get("adx", {})
    ichi_sig = ichi.get("signal", {})
    ab = strats.get("asian_breakout", {})
    ep = strats.get("ema_pullback", {})

    try:
        flat["ri"] = float(risk.get("initial_risk_pct", 0.5))
        flat["rr"] = float(risk.get("reduced_risk_pct", 0.75))
        flat["cb"] = float(risk.get("daily_circuit_breaker_pct", 4.5))
        flat["mc"] = int(risk.get("max_concurrent_positions", 3))
        flat["tp"] = float(exit_.get("tp_r_multiple", 1.5))
        flat["be"] = float(exit_.get("breakeven_threshold_r", 1.0))

        flat["ss"] = float(sss.get("min_swing_pips", 0.5))
        flat["sst"] = float(sss.get("min_stop_pips", 10.0))
        flat["sc"] = int(sss.get("min_confluence_score", 2))
        flat["sr"] = float(sss.get("rr_ratio", 2.0))
        flat["se"] = str(sss.get("entry_mode", "cbc_only"))
        flat["sp"] = float(sss.get("spread_multiplier", 2.0))

        tenkan = int(ichi_ichi.get("tenkan_period", 9))
        flat["is"] = round(tenkan / 9.0, 2)
        flat["ia"] = float(ichi_atr.get("stop_multiplier", 2.5))
        flat["id"] = int(ichi_adx.get("threshold", 20))
        flat["ic"] = int(ichi_sig.get("min_confluence_score", 1))

        flat["am"] = float(ab.get("min_range_pips", 3.0))
        flat["ax"] = float(ab.get("max_range_pips", 80.0))
        flat["ar"] = float(ab.get("rr_ratio", 2.0))

        flat["ea"] = float(ep.get("min_ema_angle_deg", 2.0))
        flat["ep"] = int(ep.get("pullback_candles_max", 20))
        flat["er"] = float(ep.get("rr_ratio", 2.0))
    except (TypeError, ValueError):
        pass  # skip incompatible param sets

    return flat


# ---------------------------------------------------------------------------
# AdaptiveRunner
# ---------------------------------------------------------------------------


class AdaptiveRunner:
    """Orchestrates the adaptive optimization loop across instruments.

    Parameters
    ----------
    db_pool:
        A ``DatabasePool`` whose ``get_cursor`` context manager yields a
        psycopg2 cursor (auto-commit on clean exit).
    data_manager:
        Optional ``DataManager`` instance.  If ``None`` a default one is
        created pointing at ``<project_root>/data/``.
    trials_per_epoch:
        Number of Optuna trials per instrument per epoch.
    """

    def __init__(
        self,
        db_pool,
        data_manager: DataManager | None = None,
        trials_per_epoch: int = 50,
    ) -> None:
        self._pool = db_pool
        self._data_manager = data_manager or DataManager()
        self._trials_per_epoch = trials_per_epoch

        self._store = ExperienceStore(db_pool)
        self._persister = SignalPersister(db_pool)
        self._epoch = 0

        # Instrument-level status tracking
        self._status: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_forever(self) -> None:
        """Cycle through instruments forever."""
        logger.info("AdaptiveRunner entering run_forever loop")
        while True:
            try:
                self.run_once()
            except KeyboardInterrupt:
                logger.info("AdaptiveRunner interrupted — shutting down")
                break
            except Exception:
                logger.exception("AdaptiveRunner epoch %d failed — sleeping 60s", self._epoch)
                time.sleep(60)

    def run_once(self, instrument_filter: str | None = None) -> dict[str, Any]:
        """Run one epoch across all (or filtered) instruments.

        Parameters
        ----------
        instrument_filter:
            If provided, only optimise the instrument whose ``symbol``
            matches this string (case-insensitive).

        Returns
        -------
        dict mapping optimizer symbol to result dict.
        """
        self._epoch += 1
        logger.info("=== Epoch %d ===", self._epoch)

        instruments = self._data_manager.load_instruments()
        if instrument_filter:
            filt = instrument_filter.upper()
            instruments = [i for i in instruments if i["symbol"].upper() == filt]

        results: dict[str, Any] = {}
        for inst in instruments:
            sym = inst["symbol"]
            try:
                result = self.optimize_instrument(inst)
                results[sym] = result
                self._status[sym] = result
            except Exception:
                tb = traceback.format_exc()
                logger.exception("Failed to optimize %s", sym)
                err = {"symbol": sym, "error": str(tb), "epoch": self._epoch}
                results[sym] = err
                self._status[sym] = err

        logger.info("Epoch %d complete — %d instruments processed", self._epoch, len(results))
        return results

    def optimize_instrument(self, instrument: dict) -> dict[str, Any]:
        """Full optimization for one instrument.

        Steps
        -----
        1. Load data via DataManager
        2. Build base config for the instrument
        3. Embed market context (20-dim)
        4. Query experience store for warm-starts (successes + failures)
        5. Run Optuna optimization (N trials)
           - Each trial: suggest_params -> backtest -> persist to DB
        6. If best trial passed combine:
           a. Check 3 consecutive passes
           b. If all pass: run permutation test (20 permutations)
           c. If significant: save proven config
        7. Return status dict
        """
        sym = instrument["symbol"]
        engine_sym = _OPTIMIZER_TO_ENGINE_SYMBOL.get(sym, sym)
        logger.info("--- Optimizing %s (engine: %s) ---", sym, engine_sym)

        # 1. Load data
        data = self._data_manager.get_data(instrument)
        if data is None or data.empty:
            msg = f"No data available for {sym}"
            logger.warning(msg)
            return {"symbol": sym, "status": "no_data", "epoch": self._epoch}

        data_start = data.index.min().to_pydatetime()
        data_end = data.index.max().to_pydatetime()
        logger.info("Data range: %s to %s  (%d bars)", data_start, data_end, len(data))

        # 2. Build base config
        base_config = self._build_base_config(engine_sym)

        # 3. Embed market context
        inst_cfg = self._get_instrument_config(engine_sym)
        market_embedding = embed_market(
            data,
            tick_size=inst_cfg.get("tick_size", 0.1),
            tick_value_usd=inst_cfg.get("tick_value_usd", 1.0),
            contract_size=inst_cfg.get("contract_size", 10),
            point_value=inst_cfg.get("tick_value_usd", 1.0),
        )

        # 4. Query experience store for warm-starts
        successes = self._store.find_similar_successes(
            market_embedding, instrument=sym, limit=5,
        )
        failures = self._store.find_similar_failures(
            market_embedding, min_similarity=0.8, limit=10,
        )
        logger.info(
            "Warm-start: %d successes, %d failures found for %s",
            len(successes), len(failures), sym,
        )

        # 5. Run Optuna optimization
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42 + self._epoch),
        )

        # Enqueue warm-start trials from past successes
        for row in successes:
            try:
                past_params = row.get("full_params", {})
                if isinstance(past_params, str):
                    past_params = json.loads(past_params)
                flat = _flat_params_from_config(past_params)
                if flat:
                    study.enqueue_trial(flat)
                    logger.debug("Enqueued warm-start trial from run %s", row.get("run_id"))
            except Exception:
                logger.debug("Skipped incompatible warm-start trial", exc_info=True)

        # Track best result across trials
        best_result: dict[str, Any] | None = None
        best_score: float = -999.0
        best_params: dict[str, Any] | None = None

        def objective(trial: optuna.Trial) -> float:
            nonlocal best_result, best_score, best_params

            params = self._suggest_params(trial, base_config)
            result = self._backtest_fn(data, params, engine_sym, INITIAL_BALANCE)

            if result is None:
                # Persist failure
                outcome = {"total_trades": 0, "passed": False, "error": "backtest_returned_none"}
                self._store.persist_trial(
                    instrument=sym,
                    data_start=data_start,
                    data_end=data_end,
                    market_embedding=market_embedding,
                    params_embedding=embed_params(params),
                    outcome_embedding=embed_outcome(outcome),
                    full_params=params,
                    active_strategies=params.get("active_strategies", []),
                    outcome=outcome,
                    passed_combine=False,
                    epoch=self._epoch,
                )
                return -1.0

            score = topstep_combine_pass_score(result)
            passed = result.get("passed", False)

            # Persist trial
            outcome = result
            self._store.persist_trial(
                instrument=sym,
                data_start=data_start,
                data_end=data_end,
                market_embedding=market_embedding,
                params_embedding=embed_params(params),
                outcome_embedding=embed_outcome(outcome),
                full_params=params,
                active_strategies=params.get("active_strategies", []),
                outcome=outcome,
                passed_combine=passed,
                epoch=self._epoch,
            )

            if score > best_score:
                best_score = score
                best_result = result
                best_params = copy.deepcopy(params)

            logger.info(
                "Trial %d: score=%.4f  passed=%s  trades=%d",
                trial.number,
                score,
                passed,
                result.get("total_trades", 0),
            )
            return score

        # Suppress Optuna's verbose logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=self._trials_per_epoch)

        # 6. Post-optimization validation
        status_dict: dict[str, Any] = {
            "symbol": sym,
            "engine_symbol": engine_sym,
            "epoch": self._epoch,
            "n_trials": self._trials_per_epoch,
            "best_score": best_score,
            "best_passed": best_result is not None and best_result.get("passed", False),
            "total_trials_in_db": self._store.count_trials(instrument=sym),
            "data_start": str(data_start),
            "data_end": str(data_end),
            "data_bars": len(data),
        }

        if best_result is not None and best_result.get("passed", False) and best_params is not None:
            logger.info("Best trial for %s PASSED — running guardrails", sym)

            # 6a. Check 3 consecutive passes
            consec = check_consecutive_passes(
                data=data,
                config=best_params,
                instrument=engine_sym,
                backtest_fn=self._backtest_fn,
                n_required=3,
                initial_balance=INITIAL_BALANCE,
            )
            status_dict["consecutive_passes"] = consec["all_passed"]

            if consec["all_passed"]:
                # 6b. Permutation test
                real_return = best_result.get("total_return_pct", 0.0)
                perm = check_permutation_significance(
                    real_return=real_return,
                    data=data,
                    config=best_params,
                    instrument=engine_sym,
                    backtest_fn=self._backtest_fn,
                    n_permutations=20,
                    initial_balance=INITIAL_BALANCE,
                )
                status_dict["permutation_p_value"] = perm["p_value"]
                status_dict["permutation_significant"] = perm["significant"]

                if perm["significant"]:
                    # 6c. Save proven config
                    run_id = self._store.persist_trial(
                        instrument=sym,
                        data_start=data_start,
                        data_end=data_end,
                        market_embedding=market_embedding,
                        params_embedding=embed_params(best_params),
                        outcome_embedding=embed_outcome(best_result),
                        full_params=best_params,
                        active_strategies=best_params.get("active_strategies", []),
                        outcome=best_result,
                        passed_combine=True,
                        passed_permutation=True,
                        epoch=self._epoch,
                    )
                    self._store.save_proven_config(
                        instrument=sym,
                        run_id=run_id,
                        params=best_params,
                        win_rate=best_result.get("win_rate", 0.0),
                        total_return_pct=real_return,
                        p_value=perm["p_value"],
                        combine_passes=3,
                        data_start=data_start,
                        data_end=data_end,
                    )
                    status_dict["proven_config_saved"] = True
                    logger.info("PROVEN CONFIG saved for %s (p=%.4f)", sym, perm["p_value"])
                else:
                    status_dict["proven_config_saved"] = False
                    logger.info(
                        "Permutation test NOT significant for %s (p=%.4f)",
                        sym, perm["p_value"],
                    )
            else:
                status_dict["proven_config_saved"] = False
                logger.info("Consecutive passes FAILED for %s", sym)
        else:
            status_dict["consecutive_passes"] = False
            status_dict["proven_config_saved"] = False
            if best_result is not None:
                logger.info(
                    "Best trial for %s did NOT pass (score=%.4f)", sym, best_score
                )
            else:
                logger.info("No valid backtest results for %s", sym)

        return status_dict

    def status(self) -> dict[str, Any]:
        """Return the latest status of all instruments."""
        return {
            "epoch": self._epoch,
            "instruments": dict(self._status),
        }

    # ------------------------------------------------------------------
    # Config building
    # ------------------------------------------------------------------

    def _build_base_config(self, engine_symbol: str) -> dict[str, Any]:
        """Build the base strategy config dict for the backtesting engine.

        Uses the existing ConfigLoader + raw strategy.yaml to construct a
        config dict matching what ``run_demo_challenge.py`` passes to
        ``IchimokuBacktester``.

        Parameters
        ----------
        engine_symbol:
            The symbol in instruments.yaml (e.g. ``"XAUUSD"``, ``"MNQ"``).
        """
        loader = ConfigLoader(config_dir=str(_PROJECT_ROOT / "config"))
        app_config = loader.load()

        with (_PROJECT_ROOT / "config" / "strategy.yaml").open() as f:
            raw_strat = yaml.safe_load(f) or {}

        ss = app_config.strategy.model_dump()
        cfg: dict[str, Any] = {}

        if hasattr(app_config, "edges"):
            cfg["edges"] = app_config.edges.model_dump()

        cfg["active_strategies"] = raw_strat.get("active_strategies", ["ichimoku"])
        cfg["strategies"] = ss.get("strategies", {})

        for key in ("risk", "exit", "prop_firm"):
            if key in ss:
                cfg[key] = ss[key]

        # Instrument-specific config
        inst = app_config.instruments.get(engine_symbol)
        if inst is not None:
            cfg["instrument_class"] = inst.class_.value
            cfg["instrument"] = {
                "symbol": inst.symbol,
                "class": inst.class_.value,
                "tick_size": inst.tick_size,
                "tick_value_usd": inst.tick_value_usd,
                "contract_size": inst.contract_size,
                "commission_per_contract_round_trip": inst.commission_per_contract_round_trip,
                "session_open_ct": inst.session_open_ct,
                "session_close_ct": inst.session_close_ct,
                "daily_reset_hour_ct": inst.daily_reset_hour_ct,
            }
        else:
            logger.warning(
                "No instrument config found for %s in instruments.yaml", engine_symbol
            )

        cfg["prop_firm"] = cfg.get("prop_firm", {})
        return cfg

    def _get_instrument_config(self, engine_symbol: str) -> dict[str, Any]:
        """Return a flat dict of instrument metadata for the embedder."""
        loader = ConfigLoader(config_dir=str(_PROJECT_ROOT / "config"))
        app_config = loader.load()
        inst = app_config.instruments.get(engine_symbol)
        if inst is None:
            return {}
        return {
            "tick_size": inst.tick_size or 0.1,
            "tick_value_usd": inst.tick_value_usd or 1.0,
            "contract_size": inst.contract_size or 10,
        }

    # ------------------------------------------------------------------
    # Param suggestion
    # ------------------------------------------------------------------

    @staticmethod
    def _suggest_params(trial: optuna.Trial, base: dict) -> dict:
        """Suggest a full config dict from an Optuna trial."""
        params = copy.deepcopy(base)

        # Ensure nested dicts exist
        params.setdefault("risk", {})
        params.setdefault("exit", {})
        params.setdefault("strategies", {})
        params["strategies"].setdefault("sss", {})
        params["strategies"].setdefault("ichimoku", {})
        params["strategies"]["ichimoku"].setdefault("ichimoku", {})
        params["strategies"]["ichimoku"].setdefault("atr", {})
        params["strategies"]["ichimoku"].setdefault("adx", {})
        params["strategies"]["ichimoku"].setdefault("signal", {})
        params["strategies"].setdefault("asian_breakout", {})
        params["strategies"].setdefault("ema_pullback", {})

        # Risk
        params["risk"]["initial_risk_pct"] = trial.suggest_float("ri", 0.1, 2.0, step=0.1)
        params["risk"]["reduced_risk_pct"] = trial.suggest_float("rr", 0.1, 2.0, step=0.1)
        params["risk"]["daily_circuit_breaker_pct"] = trial.suggest_float("cb", 1.5, 5.0, step=0.5)
        params["risk"]["max_concurrent_positions"] = trial.suggest_int("mc", 1, 5)

        # Exit
        params["exit"]["tp_r_multiple"] = trial.suggest_float("tp", 1.0, 3.0, step=0.25)
        params["exit"]["breakeven_threshold_r"] = trial.suggest_float("be", 0.5, 1.5, step=0.25)

        # SSS
        params["strategies"]["sss"]["min_swing_pips"] = trial.suggest_float("ss", 0.01, 2.0, step=0.01)
        params["strategies"]["sss"]["min_stop_pips"] = trial.suggest_float("sst", 0.02, 5.0, step=0.02)
        params["strategies"]["sss"]["min_confluence_score"] = trial.suggest_int("sc", 0, 4)
        params["strategies"]["sss"]["rr_ratio"] = trial.suggest_float("sr", 1.5, 3.0, step=0.25)
        params["strategies"]["sss"]["entry_mode"] = trial.suggest_categorical(
            "se", ["cbc_only", "fifty_tap", "combined"]
        )
        params["strategies"]["sss"]["spread_multiplier"] = trial.suggest_float("sp", 0.5, 3.0, step=0.5)

        # Ichimoku
        scale = trial.suggest_float("is", 0.7, 1.3, step=0.05)
        params["strategies"]["ichimoku"]["ichimoku"]["tenkan_period"] = max(3, round(9 * scale))
        params["strategies"]["ichimoku"]["ichimoku"]["kijun_period"] = max(9, round(26 * scale))
        params["strategies"]["ichimoku"]["ichimoku"]["senkou_b_period"] = max(18, round(52 * scale))
        params["strategies"]["ichimoku"]["atr"]["stop_multiplier"] = trial.suggest_float(
            "ia", 1.0, 3.0, step=0.25
        )
        params["strategies"]["ichimoku"]["adx"]["threshold"] = trial.suggest_int("id", 10, 35)
        params["strategies"]["ichimoku"]["signal"]["min_confluence_score"] = trial.suggest_int(
            "ic", 1, 5
        )

        # Asian Breakout
        params["strategies"]["asian_breakout"]["min_range_pips"] = trial.suggest_float(
            "am", 0.01, 5.0, step=0.01
        )
        params["strategies"]["asian_breakout"]["max_range_pips"] = trial.suggest_float(
            "ax", 1.0, 100.0, step=1.0
        )
        params["strategies"]["asian_breakout"]["rr_ratio"] = trial.suggest_float(
            "ar", 1.5, 3.0, step=0.25
        )

        # EMA Pullback
        params["strategies"]["ema_pullback"]["min_ema_angle_deg"] = trial.suggest_float(
            "ea", 0.5, 10.0, step=0.5
        )
        params["strategies"]["ema_pullback"]["pullback_candles_max"] = trial.suggest_int("ep", 5, 30)
        params["strategies"]["ema_pullback"]["rr_ratio"] = trial.suggest_float(
            "er", 1.5, 3.0, step=0.25
        )

        return params

    # ------------------------------------------------------------------
    # Backtest function
    # ------------------------------------------------------------------

    @staticmethod
    def _backtest_fn(
        data: pd.DataFrame,
        config: dict,
        instrument: str,
        initial_balance: float,
    ) -> dict | None:
        """Run a single backtest and return a flat metrics dict (or None).

        Parameters
        ----------
        data:
            1-minute OHLCV DataFrame with UTC DatetimeIndex.
        config:
            Full strategy config dict.
        instrument:
            Engine-level instrument symbol (e.g. ``"XAUUSD"``).
        initial_balance:
            Starting account balance.

        Returns
        -------
        dict with keys like ``total_trades``, ``win_rate``,
        ``total_return_pct``, ``passed``, ``prop_firm``, etc.
        Returns ``None`` on unrecoverable errors.
        """
        from src.backtesting.vectorbt_engine import IchimokuBacktester

        try:
            prop = config.get("prop_firm", {})
            p1 = prop.get("phase_1", {})

            bt = IchimokuBacktester(
                config=config,
                initial_balance=initial_balance,
                prop_firm_profit_target_pct=float(p1.get("profit_target_pct", 8.0)),
                prop_firm_max_daily_dd_pct=float(p1.get("daily_loss_pct", 5.0)),
                prop_firm_max_total_dd_pct=float(p1.get("max_loss_pct", 10.0)),
                prop_firm_time_limit_days=int(p1.get("time_limit_days", 30)),
            )
            result = bt.run(
                candles_1m=data,
                instrument=instrument,
                log_trades=False,
                enable_learning=False,
            )
        except Exception:
            logger.exception("Backtest failed for %s", instrument)
            return None

        # Flatten BacktestResult into a plain dict for the experience store
        metrics = result.metrics if hasattr(result, "metrics") else {}
        prop_firm = result.prop_firm if hasattr(result, "prop_firm") else {}

        # Determine pass status from the prop firm tracker
        active = prop_firm.get("active_tracker") if isinstance(prop_firm, dict) else None
        snap = active or prop_firm
        passed = str(snap.get("status", "")).lower() == "passed"

        # Compute derived fields
        total_trades = len(result.trades) if hasattr(result, "trades") else 0
        win_count = sum(
            1 for t in (result.trades or [])
            if (t.get("pnl_usd") or t.get("pnl", 0)) > 0
        ) if hasattr(result, "trades") else 0
        win_rate = win_count / total_trades if total_trades > 0 else 0.0

        final_balance = float(snap.get("current_balance", initial_balance))
        total_return_pct = ((final_balance - initial_balance) / initial_balance) * 100.0

        out: dict[str, Any] = {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_return_pct": total_return_pct,
            "final_balance": final_balance,
            "passed": passed,
            "prop_firm": prop_firm,
        }

        # Merge in engine-computed metrics if available
        if isinstance(metrics, dict):
            for k in (
                "profit_factor", "sharpe_ratio", "max_drawdown_pct",
                "avg_r_multiple", "best_trade_r", "worst_trade_r",
                "avg_trade_duration_bars",
            ):
                if k in metrics:
                    out[k] = metrics[k]

        # TopstepX-specific fields from the active tracker snapshot
        if isinstance(snap, dict):
            for k in (
                "distance_to_target", "best_day_profit",
                "consistency_ratio",
            ):
                if k in snap:
                    out[k] = snap[k]

        return out
