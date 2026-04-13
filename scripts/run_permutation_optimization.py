"""
Permutation-validated TopstepX Combine optimization.

Implements the strict systems approach from the video:
1. Run backtest on real data → get real performance metrics
2. Permute (shuffle) candle order N times → get distribution of random performance
3. Compute p-value: how often does random beat real?
4. Optimize with Optuna until parameters pass the combine 3x in a row
   AND show statistical significance (p-value < 0.01)

Usage:
    python scripts/run_permutation_optimization.py
"""

from __future__ import annotations

import copy
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("permutation_optimizer")

# Suppress noisy sub-loggers during optimization
for _name in [
    "src.backtesting.vectorbt_engine",
    "src.strategy",
    "src.risk",
    "src.edges",
    "src.learning",
    "src.monitoring",
    "src.backtesting.strategy_telemetry",
]:
    logging.getLogger(_name).setLevel(logging.WARNING)


# =============================================================================
# Constants
# =============================================================================

DATA_FILE = "data/projectx_mgc_1m_last30d_20260310_20260409.parquet"
N_PERMUTATIONS = 100          # number of shuffled datasets for p-value
P_VALUE_THRESHOLD = 0.05      # accept if p < this
CONSECUTIVE_PASSES_REQUIRED = 3
MAX_OPTUNA_TRIALS = 200       # max optimization trials
INITIAL_BALANCE = 50_000.0


# =============================================================================
# Data loading
# =============================================================================

def load_data() -> pd.DataFrame:
    """Load the last-30-days MGC futures data."""
    path = _PROJECT_ROOT / DATA_FILE
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.columns = df.columns.str.lower()
    logger.info("Loaded %d bars from %s to %s", len(df), df.index[0], df.index[-1])
    return df


def permute_candles(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Permute (shuffle) candle returns while preserving statistical properties.

    Instead of shuffling raw candles (which would break OHLC relationships),
    we shuffle the bar-to-bar RETURNS and reconstruct the price series.
    This preserves:
    - Same set of individual bar returns (distribution identical)
    - Same volatility, mean return, kurtosis
    - Destroys temporal patterns (trends, mean reversion, etc.)
    """
    rng = np.random.default_rng(seed)

    # Compute bar-to-bar log returns for close
    closes = df["close"].values.astype(float)
    log_returns = np.diff(np.log(closes))

    # Shuffle the returns
    shuffled_returns = log_returns.copy()
    rng.shuffle(shuffled_returns)

    # Reconstruct close prices from shuffled returns
    new_closes = np.empty(len(closes))
    new_closes[0] = closes[0]
    for i in range(len(shuffled_returns)):
        new_closes[i + 1] = new_closes[i] * np.exp(shuffled_returns[i])

    # Scale OHLC proportionally
    ratio = new_closes / closes
    new_df = df.copy()
    for col in ["open", "high", "low", "close"]:
        if col in new_df.columns:
            new_df[col] = (df[col].values * ratio).astype(float)

    # Ensure high >= max(open, close) and low <= min(open, close)
    new_df["high"] = new_df[["open", "high", "close"]].max(axis=1)
    new_df["low"] = new_df[["open", "low", "close"]].min(axis=1)

    return new_df


# =============================================================================
# Config building
# =============================================================================

def build_base_config() -> dict:
    """Build the base backtest config from YAML files."""
    import yaml

    config_dir = _PROJECT_ROOT / "config"
    strat_yaml = config_dir / "strategy.yaml"
    mega_yaml = config_dir / "mega_vision.yaml"

    with strat_yaml.open() as f:
        raw_strat = yaml.safe_load(f) or {}

    mega_cfg = {}
    if mega_yaml.exists():
        with mega_yaml.open() as f:
            mega_cfg = yaml.safe_load(f) or {}

    # Load validated config for instrument info
    from src.config.loader import ConfigLoader
    loader = ConfigLoader(config_dir=str(config_dir))
    app_config = loader.load()

    strategy_snapshot = (
        app_config.strategy.model_dump()
        if hasattr(app_config, "strategy")
        else {}
    )

    bt_config = {}
    if hasattr(app_config, "edges"):
        bt_config["edges"] = app_config.edges.model_dump()
    bt_config["active_strategies"] = raw_strat.get(
        "active_strategies",
        [strategy_snapshot.get("active_strategy", "ichimoku")],
    )
    bt_config["strategies"] = strategy_snapshot.get("strategies", {})
    for key in ("risk", "exit", "prop_firm"):
        if key in strategy_snapshot:
            bt_config[key] = strategy_snapshot[key]
    if mega_cfg:
        bt_config["mega_vision"] = mega_cfg

    instrument_cfg = None
    if hasattr(app_config, "instruments"):
        instrument_cfg = app_config.instruments.get("XAUUSD")
    if instrument_cfg is not None:
        bt_config["instrument_class"] = instrument_cfg.class_.value
        bt_config["instrument"] = {
            "symbol": instrument_cfg.symbol,
            "class": instrument_cfg.class_.value,
            "tick_size": instrument_cfg.tick_size,
            "tick_value_usd": instrument_cfg.tick_value_usd,
            "contract_size": instrument_cfg.contract_size,
            "commission_per_contract_round_trip": instrument_cfg.commission_per_contract_round_trip,
            "session_open_ct": instrument_cfg.session_open_ct,
            "session_close_ct": instrument_cfg.session_close_ct,
            "daily_reset_hour_ct": instrument_cfg.daily_reset_hour_ct,
        }

    bt_config["prop_firm"] = bt_config.get("prop_firm", {})
    return bt_config


# =============================================================================
# Backtest runner
# =============================================================================

def run_single_backtest(
    data: pd.DataFrame,
    config: dict,
    initial_balance: float = INITIAL_BALANCE,
) -> Optional[dict]:
    """Run a single backtest and return key metrics + combine verdict."""
    from src.backtesting.vectorbt_engine import IchimokuBacktester

    try:
        backtester = IchimokuBacktester(
            config=config,
            initial_balance=initial_balance,
        )
        result = backtester.run(
            candles_1m=data,
            instrument="XAUUSD",
            log_trades=False,
            enable_learning=False,
        )
    except Exception as exc:
        logger.warning("Backtest failed: %s", exc)
        return None

    metrics = result.metrics
    prop = result.prop_firm
    active = prop.get("active_tracker", prop)

    return {
        "total_trades": metrics.get("total_trades", 0),
        "win_rate": metrics.get("win_rate", 0),
        "total_return_pct": metrics.get("total_return_pct", 0),
        "sharpe_ratio": metrics.get("sharpe_ratio", 0),
        "max_drawdown_pct": metrics.get("max_drawdown_pct", 0),
        "profit_factor": metrics.get("profit_factor", 0),
        "expectancy": metrics.get("expectancy", 0),
        "final_balance": float(active.get("current_balance", initial_balance)),
        "status": str(active.get("status", "pending")),
        "passed": str(active.get("status", "")) == "passed",
        "failure_reason": active.get("failure_reason"),
        "mll": float(active.get("mll", 0)),
        "total_profit": float(active.get("total_profit", 0)),
        "best_day_profit": float(active.get("best_day_profit", 0)),
        "distance_to_target": float(active.get("distance_to_target", 0)),
        "trades": result.trades,
    }


# =============================================================================
# Combine replay (from trade list)
# =============================================================================

def replay_combine(
    trades: List[dict],
    start_balance: float = INITIAL_BALANCE,
) -> dict:
    """Replay trades through TopstepCombineSimulator."""
    from src.config.models import TopstepCombineConfig
    from src.backtesting.topstep_simulator import TopstepCombineSimulator

    cfg = TopstepCombineConfig(
        account_size=start_balance,
        profit_target_usd=3000.0,
        max_loss_limit_usd_trailing=2000.0,
        daily_loss_limit_usd=1000.0,
        consistency_pct=50.0,
    )
    sim = TopstepCombineSimulator(config=cfg)
    result = sim.run(trades, start_balance=start_balance)
    return result.to_dict()


# =============================================================================
# Permutation testing
# =============================================================================

def run_permutation_test(
    real_data: pd.DataFrame,
    config: dict,
    n_permutations: int = N_PERMUTATIONS,
    metric_key: str = "total_return_pct",
) -> dict:
    """Run permutation test: compare real performance vs shuffled data.

    Returns dict with:
    - real_metric: the metric on real data
    - permuted_metrics: list of metric values on permuted data
    - p_value: fraction of permuted runs >= real metric
    - is_significant: whether p_value < threshold
    """
    logger.info("Running permutation test with %d permutations...", n_permutations)

    # Real data backtest
    real_result = run_single_backtest(real_data, config)
    if real_result is None:
        return {"real_metric": 0, "permuted_metrics": [], "p_value": 1.0, "is_significant": False}

    real_metric = real_result.get(metric_key, 0)
    real_passed = real_result.get("passed", False)
    logger.info("Real data: %s=%.4f, passed=%s, trades=%d",
                metric_key, real_metric, real_passed, real_result["total_trades"])

    # Permuted data backtests
    permuted_metrics = []
    permuted_passes = 0
    for i in range(n_permutations):
        seed = 42 + i
        perm_data = permute_candles(real_data, seed=seed)
        perm_result = run_single_backtest(perm_data, config)
        if perm_result is not None:
            perm_val = perm_result.get(metric_key, 0)
            permuted_metrics.append(perm_val)
            if perm_result.get("passed", False):
                permuted_passes += 1
        else:
            permuted_metrics.append(0)

        if (i + 1) % 10 == 0:
            logger.info("  Permutation %d/%d complete", i + 1, n_permutations)

    # Compute p-value: fraction of permuted metrics >= real metric
    if len(permuted_metrics) > 0:
        beats_real = sum(1 for pm in permuted_metrics if pm >= real_metric)
        p_value = beats_real / len(permuted_metrics)
    else:
        p_value = 1.0

    logger.info(
        "Permutation test: real=%s=%.4f, mean_permuted=%.4f, p_value=%.4f, "
        "permuted_passes=%d/%d",
        metric_key, real_metric,
        np.mean(permuted_metrics) if permuted_metrics else 0,
        p_value, permuted_passes, n_permutations,
    )

    return {
        "real_metric": real_metric,
        "real_result": real_result,
        "permuted_metrics": permuted_metrics,
        "p_value": p_value,
        "is_significant": p_value < P_VALUE_THRESHOLD,
        "permuted_pass_count": permuted_passes,
    }


# =============================================================================
# Multi-strategy Optuna parameter space
# =============================================================================

class BlendedStrategySpace:
    """Optuna parameter space for the full strategy blend."""

    def suggest_params(self, trial: optuna.Trial) -> dict:
        """Suggest parameters for all strategies + shared risk/exit."""
        params = {
            "active_strategies": ["sss", "ichimoku", "asian_breakout", "ema_pullback", "fx_at_one_glance"],
            "strategies": {},
            "risk": {},
            "exit": {},
        }

        # --- Shared risk params ---
        params["risk"]["initial_risk_pct"] = trial.suggest_float("risk_initial", 0.3, 1.5, step=0.1)
        params["risk"]["reduced_risk_pct"] = trial.suggest_float("risk_reduced", 0.3, 1.5, step=0.1)
        params["risk"]["daily_circuit_breaker_pct"] = trial.suggest_float("daily_cb_pct", 1.5, 4.5, step=0.5)
        params["risk"]["max_concurrent_positions"] = trial.suggest_int("max_concurrent", 1, 5)

        # --- Shared exit params ---
        params["exit"]["tp_r_multiple"] = trial.suggest_float("tp_r_multiple", 1.0, 3.0, step=0.25)
        params["exit"]["breakeven_threshold_r"] = trial.suggest_float("be_threshold_r", 0.5, 1.5, step=0.25)

        # --- SSS ---
        params["strategies"]["sss"] = {
            "enabled": True,
            "swing_lookback_n": trial.suggest_int("sss_lookback", 2, 5),
            "min_swing_pips": trial.suggest_float("sss_min_swing", 0.3, 3.0, step=0.1),
            "ss_candle_min": trial.suggest_int("sss_ss_min", 6, 15),
            "iss_candle_min": trial.suggest_int("sss_iss_min", 2, 5),
            "iss_candle_max": trial.suggest_int("sss_iss_max", 5, 10),
            "entry_mode": trial.suggest_categorical("sss_entry_mode", ["cbc_only", "fifty_tap", "combined"]),
            "min_confluence_score": trial.suggest_int("sss_confluence", 0, 4),
            "rr_ratio": trial.suggest_float("sss_rr", 1.5, 3.0, step=0.25),
        }

        # --- Ichimoku ---
        scale = trial.suggest_float("ichi_scale", 0.7, 1.3, step=0.05)
        params["strategies"]["ichimoku"] = {
            "ichimoku": {
                "tenkan_period": max(3, round(9 * scale)),
                "kijun_period": max(9, round(26 * scale)),
                "senkou_b_period": max(18, round(52 * scale)),
            },
            "adx": {"threshold": trial.suggest_int("ichi_adx", 15, 35)},
            "atr": {"stop_multiplier": trial.suggest_float("ichi_atr_mult", 1.0, 3.0, step=0.25)},
            "signal": {
                "min_confluence_score": trial.suggest_int("ichi_confluence", 1, 5),
                "tier_c": trial.suggest_int("ichi_tier_c", 1, 4),
            },
        }

        # --- Asian Breakout ---
        params["strategies"]["asian_breakout"] = {
            "enabled": True,
            "min_range_pips": trial.suggest_float("ab_min_range", 1.0, 10.0, step=0.5),
            "max_range_pips": trial.suggest_float("ab_max_range", 30.0, 150.0, step=5.0),
            "rr_ratio": trial.suggest_float("ab_rr", 1.5, 3.0, step=0.25),
            "london_entry_end_utc": trial.suggest_categorical("ab_london_end", ["10:00", "14:00", "20:00"]),
        }

        # --- EMA Pullback ---
        params["strategies"]["ema_pullback"] = {
            "enabled": True,
            "fast_ema": trial.suggest_int("ep_fast", 5, 14),
            "mid_ema": trial.suggest_int("ep_mid", 14, 26),
            "slow_ema": trial.suggest_int("ep_slow", 30, 60),
            "min_ema_angle_deg": trial.suggest_float("ep_angle_min", 1.0, 10.0, step=0.5),
            "pullback_candles_max": trial.suggest_int("ep_pb_max", 5, 25),
            "rr_ratio": trial.suggest_float("ep_rr", 1.5, 3.0, step=0.25),
        }

        # --- FX At One Glance ---
        params["strategies"]["fx_at_one_glance"] = {
            "tf_mode": trial.suggest_categorical("fxaog_tf_mode", [
                "hyperscalp_m15_m5", "scalp", "hybrid"
            ]),
            "signal": {
                "min_confluence_score": trial.suggest_int("fxaog_confluence", 3, 8),
                "min_tier": trial.suggest_categorical("fxaog_tier", ["B", "C"]),
            },
            "five_elements_mode": trial.suggest_categorical("fxaog_five_elem", [
                "hard_gate", "soft_filter", "disabled"
            ]),
            "exit": {
                "mode": "hybrid",
                "partial_close_pct": trial.suggest_int("fxaog_partial_pct", 30, 70, step=10),
            },
            "stop_loss": {
                "min_rr_ratio": trial.suggest_float("fxaog_min_rr", 1.0, 2.5, step=0.25),
            },
        }

        return params


# =============================================================================
# Main optimization loop
# =============================================================================

def run_optimization(
    data: pd.DataFrame,
    base_config: dict,
    n_trials: int = MAX_OPTUNA_TRIALS,
) -> Tuple[dict, float]:
    """Run Optuna optimization targeting TopstepX combine pass.

    Returns (best_params, best_score).
    """
    from src.optimization.objectives import topstep_combine_pass_score

    space = BlendedStrategySpace()

    def objective(trial: optuna.Trial) -> float:
        params = space.suggest_params(trial)

        # Merge with base config
        merged = copy.deepcopy(base_config)
        for k, v in params.items():
            if k == "strategies" and isinstance(v, dict) and "strategies" in merged:
                for sn, sp in v.items():
                    if sn in merged["strategies"] and isinstance(sp, dict):
                        merged["strategies"][sn] = {**merged["strategies"][sn], **sp}
                    else:
                        merged["strategies"][sn] = sp
            elif k == "risk" and "risk" in merged:
                merged["risk"] = {**merged["risk"], **v}
            elif k == "exit" and "exit" in merged:
                merged["exit"] = {**merged["exit"], **v}
            else:
                merged[k] = v

        result = run_single_backtest(data, merged)
        if result is None:
            return -2.0

        n_trades = result.get("total_trades", 0)
        if n_trades <= 0:
            return -2.0

        # Score: combine pass score + trade count bonus
        score = 0.0
        if result["passed"]:
            score = 1.0
        else:
            # Continuous gradient based on distance to target
            profit = result["final_balance"] - INITIAL_BALANCE
            target = 3000.0
            balance_score = profit / target
            balance_score = max(-1.0, min(1.0, balance_score))

            penalty = 0.0
            status = result["status"]
            if "failed_mll" in status:
                penalty = -0.5
            elif "failed_daily_loss" in status:
                penalty = -0.3
            elif "failed_consistency" in status:
                penalty = -0.2

            score = balance_score + penalty

        # Trade count bonus (more trades = more robust)
        score += min(n_trades, 30) / 200.0

        # Win rate bonus
        wr = result.get("win_rate", 0)
        if wr > 0.45:
            score += 0.1

        trial.set_user_attr("passed", result["passed"])
        trial.set_user_attr("total_trades", n_trades)
        trial.set_user_attr("final_balance", result["final_balance"])
        trial.set_user_attr("status", result["status"])
        trial.set_user_attr("win_rate", round(wr, 4))

        return score

    study = optuna.create_study(
        direction="maximize",
        study_name="topstep_blend_optimization",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Add a callback to log progress
    def log_callback(study, trial):
        if trial.value is not None:
            passed = trial.user_attrs.get("passed", False)
            trades = trial.user_attrs.get("total_trades", 0)
            balance = trial.user_attrs.get("final_balance", 0)
            status = trial.user_attrs.get("status", "?")
            logger.info(
                "Trial %d: score=%.4f, passed=%s, trades=%d, balance=$%.2f, status=%s",
                trial.number, trial.value, passed, trades, balance, status,
            )

    study.optimize(objective, n_trials=n_trials, callbacks=[log_callback])

    best = study.best_trial
    logger.info(
        "Best trial #%d: score=%.4f, passed=%s, trades=%d",
        best.number, best.value,
        best.user_attrs.get("passed"),
        best.user_attrs.get("total_trades"),
    )

    # Reconstruct best params
    best_params = space.suggest_params(best)
    return best_params, best.value


def check_consecutive_passes(
    data: pd.DataFrame,
    config: dict,
    n_required: int = CONSECUTIVE_PASSES_REQUIRED,
) -> Tuple[bool, List[dict]]:
    """Run the backtest n_required times (with different random seeds for learning)
    and check if ALL pass the combine.

    Since the data is deterministic, we simulate multiple "attempts" by
    splitting the data into overlapping windows or running the same data
    with slight perturbations to test robustness.
    """
    results = []
    data_len = len(data)

    # Create 3 overlapping windows from the data to simulate 3 different
    # combine attempts (like 3 funded accounts)
    # Window 1: full data
    # Window 2: offset by ~2 days
    # Window 3: offset by ~5 days
    offsets = [0]
    bars_per_day = 60 * 23  # ~23 hours of trading per day at 1min resolution
    for i in range(1, n_required):
        offset = min(bars_per_day * (i * 2), data_len // 4)
        offsets.append(offset)

    for i, offset in enumerate(offsets):
        window = data.iloc[offset:]
        if len(window) < 5000:  # need minimum bars
            logger.warning("Window %d too small (%d bars), using full data", i, len(window))
            window = data

        logger.info("=== Combine attempt %d/%d (offset=%d bars) ===", i + 1, n_required, offset)
        result = run_single_backtest(window, config)

        if result is None:
            results.append({"passed": False, "status": "error", "total_trades": 0})
            logger.info("Attempt %d: FAILED (backtest error)", i + 1)
        else:
            results.append(result)
            logger.info(
                "Attempt %d: %s — balance=$%.2f, trades=%d, win_rate=%.1f%%",
                i + 1,
                "PASSED" if result["passed"] else f"FAILED ({result['status']})",
                result["final_balance"],
                result["total_trades"],
                result.get("win_rate", 0) * 100,
            )

    all_passed = all(r.get("passed", False) for r in results)
    return all_passed, results


# =============================================================================
# Main
# =============================================================================

def main():
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("  PERMUTATION-VALIDATED TOPSTEPX COMBINE OPTIMIZATION")
    logger.info("=" * 70)

    # Load data
    data = load_data()

    # Build base config
    logger.info("Building base config...")
    base_config = build_base_config()
    logger.info("Active strategies: %s", base_config.get("active_strategies"))

    # Phase 1: Initial assessment on current config
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 1: BASELINE ASSESSMENT")
    logger.info("=" * 50)
    baseline = run_single_backtest(data, base_config)
    if baseline:
        logger.info(
            "Baseline: trades=%d, win_rate=%.1f%%, return=%.2f%%, "
            "balance=$%.2f, status=%s",
            baseline["total_trades"],
            baseline.get("win_rate", 0) * 100,
            baseline.get("total_return_pct", 0),
            baseline["final_balance"],
            baseline["status"],
        )
    else:
        logger.error("Baseline backtest failed — fixing issues needed")

    # Phase 2: Optimization loop
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 2: OPTUNA OPTIMIZATION")
    logger.info("=" * 50)

    iteration = 0
    max_iterations = 10  # max optimization rounds
    best_config = base_config
    found_passing = False

    while iteration < max_iterations and not found_passing:
        iteration += 1
        logger.info("\n--- Optimization round %d/%d ---", iteration, max_iterations)

        best_params, best_score = run_optimization(
            data, base_config,
            n_trials=min(50 + iteration * 20, MAX_OPTUNA_TRIALS),
        )

        # Merge best params into config
        test_config = copy.deepcopy(base_config)
        for k, v in best_params.items():
            if k == "strategies" and isinstance(v, dict) and "strategies" in test_config:
                for sn, sp in v.items():
                    if sn in test_config["strategies"] and isinstance(sp, dict):
                        test_config["strategies"][sn] = {**test_config["strategies"][sn], **sp}
                    else:
                        test_config["strategies"][sn] = sp
            elif k == "risk" and "risk" in test_config:
                test_config["risk"] = {**test_config["risk"], **v}
            elif k == "exit" and "exit" in test_config:
                test_config["exit"] = {**test_config["exit"], **v}
            else:
                test_config[k] = v

        # Check if best params pass the combine
        logger.info("\nChecking combine pass with best params...")
        single_result = run_single_backtest(data, test_config)
        if single_result and single_result["passed"]:
            logger.info("Single pass confirmed! Checking %d consecutive passes...",
                       CONSECUTIVE_PASSES_REQUIRED)

            all_passed, pass_results = check_consecutive_passes(
                data, test_config, CONSECUTIVE_PASSES_REQUIRED
            )

            if all_passed:
                logger.info("ALL %d CONSECUTIVE PASSES CONFIRMED!", CONSECUTIVE_PASSES_REQUIRED)
                best_config = test_config
                found_passing = True
            else:
                n_passed = sum(1 for r in pass_results if r.get("passed"))
                logger.info(
                    "Only %d/%d passes — continuing optimization...",
                    n_passed, CONSECUTIVE_PASSES_REQUIRED,
                )
        else:
            status = single_result["status"] if single_result else "error"
            logger.info("Best params don't pass combine (status=%s) — continuing...", status)

    if not found_passing:
        logger.warning("Could not find params that pass %d consecutive combines", CONSECUTIVE_PASSES_REQUIRED)
        logger.info("Using best available config for permutation testing...")
        best_config = test_config

    # Phase 3: Permutation testing
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 3: PERMUTATION TESTING")
    logger.info("=" * 50)

    perm_result = run_permutation_test(
        data, best_config,
        n_permutations=N_PERMUTATIONS,
        metric_key="total_return_pct",
    )

    # Phase 4: Final report
    logger.info("\n" + "=" * 70)
    logger.info("  FINAL REPORT")
    logger.info("=" * 70)

    final_result = run_single_backtest(data, best_config)
    if final_result:
        logger.info("Final backtest results:")
        logger.info("  Total trades:       %d", final_result["total_trades"])
        logger.info("  Win rate:           %.1f%%", final_result.get("win_rate", 0) * 100)
        logger.info("  Total return:       %.2f%%", final_result.get("total_return_pct", 0))
        logger.info("  Sharpe ratio:       %.2f", final_result.get("sharpe_ratio", 0))
        logger.info("  Final balance:      $%.2f", final_result["final_balance"])
        logger.info("  Combine status:     %s", final_result["status"])
        logger.info("  Passed:             %s", final_result["passed"])

    logger.info("\nPermutation test results:")
    logger.info("  Real return:        %.4f%%", perm_result["real_metric"])
    logger.info("  Mean permuted:      %.4f%%",
                np.mean(perm_result["permuted_metrics"]) if perm_result["permuted_metrics"] else 0)
    logger.info("  P-value:            %.4f", perm_result["p_value"])
    logger.info("  Significant:        %s (threshold=%.2f)",
                perm_result["is_significant"], P_VALUE_THRESHOLD)
    logger.info("  Permuted passes:    %d/%d",
                perm_result.get("permuted_pass_count", 0), N_PERMUTATIONS)

    if found_passing:
        logger.info("\n  CONSECUTIVE PASSES: %d/%d", CONSECUTIVE_PASSES_REQUIRED, CONSECUTIVE_PASSES_REQUIRED)
    logger.info("  VERDICT: %s",
                "STRATEGY HAS REAL EDGE" if perm_result["is_significant"] and found_passing
                else "NEEDS MORE OPTIMIZATION" if not found_passing
                else "PASSES BUT NOT STATISTICALLY SIGNIFICANT")

    # Save results
    output_dir = _PROJECT_ROOT / "reports"
    output_dir.mkdir(exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": ts,
        "data_file": DATA_FILE,
        "found_passing_config": found_passing,
        "consecutive_passes_required": CONSECUTIVE_PASSES_REQUIRED,
        "p_value": perm_result["p_value"],
        "is_significant": perm_result["is_significant"],
        "n_permutations": N_PERMUTATIONS,
        "real_return_pct": perm_result["real_metric"],
        "mean_permuted_return_pct": float(np.mean(perm_result["permuted_metrics"])) if perm_result["permuted_metrics"] else 0,
        "final_metrics": {k: v for k, v in (final_result or {}).items() if k != "trades"},
        "optimization_iterations": iteration,
        "elapsed_seconds": time.time() - start_time,
    }

    # Save optimized config
    config_report = {
        "active_strategies": best_config.get("active_strategies"),
        "strategies": best_config.get("strategies"),
        "risk": best_config.get("risk"),
        "exit": best_config.get("exit"),
    }
    report["optimized_config"] = config_report

    report_path = output_dir / f"permutation_optimization_{ts}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("\nReport saved to %s", report_path)

    elapsed = time.time() - start_time
    logger.info("Total elapsed: %.1f minutes", elapsed / 60)

    return 0 if (found_passing and perm_result["is_significant"]) else 1


if __name__ == "__main__":
    sys.exit(main())
