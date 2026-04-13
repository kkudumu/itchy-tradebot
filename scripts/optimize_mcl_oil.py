"""Optuna optimization for MCL Micro Crude Oil on TopstepX Combine."""

from __future__ import annotations

import copy
import json
import logging
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING, format="%(levelname)-8s %(name)s - %(message)s")
optuna.logging.set_verbosity(optuna.logging.WARNING)

INITIAL_BALANCE = 50_000.0
DATA_FILE = "data/projectx_mcl_1m_20260101_20260411.parquet"
INSTRUMENT = "MCLOIL"


def load_data():
    data = pd.read_parquet(_PROJECT_ROOT / DATA_FILE)
    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    data.columns = data.columns.str.lower()
    print(f"MCL data: {len(data)} bars, {data.index[0]} to {data.index[-1]}")
    return data


def build_base_config():
    from src.config.loader import ConfigLoader
    loader = ConfigLoader(config_dir=str(_PROJECT_ROOT / "config"))
    app_config = loader.load()

    strat_yaml = _PROJECT_ROOT / "config" / "strategy.yaml"
    with strat_yaml.open() as f:
        raw_strat = yaml.safe_load(f) or {}

    strategy_snapshot = app_config.strategy.model_dump()
    cfg = {}
    if hasattr(app_config, "edges"):
        cfg["edges"] = app_config.edges.model_dump()
    cfg["active_strategies"] = raw_strat.get("active_strategies", ["ichimoku"])
    cfg["strategies"] = strategy_snapshot.get("strategies", {})
    for key in ("risk", "exit", "prop_firm"):
        if key in strategy_snapshot:
            cfg[key] = strategy_snapshot[key]

    instrument_cfg = app_config.instruments.get(INSTRUMENT)
    if instrument_cfg is None:
        raise RuntimeError(f"Instrument {INSTRUMENT} not found in instruments.yaml")
    cfg["instrument_class"] = instrument_cfg.class_.value
    cfg["instrument"] = {
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
    cfg["prop_firm"] = cfg.get("prop_firm", {})
    return cfg


def run_backtest(data, config, instrument=INSTRUMENT):
    from src.backtesting.vectorbt_engine import IchimokuBacktester
    try:
        bt = IchimokuBacktester(config=config, initial_balance=INITIAL_BALANCE)
        result = bt.run(candles_1m=data, instrument=instrument, log_trades=False, enable_learning=False)
        m = result.metrics
        prop = result.prop_firm
        active = prop.get("active_tracker", prop)
        return {
            "total_trades": m.get("total_trades", 0),
            "win_rate": m.get("win_rate", 0),
            "total_return_pct": m.get("total_return_pct", 0),
            "sharpe_ratio": m.get("sharpe_ratio", 0),
            "max_drawdown_pct": m.get("max_drawdown_pct", 0),
            "profit_factor": m.get("profit_factor", 0),
            "final_balance": float(active.get("current_balance", INITIAL_BALANCE)),
            "status": str(active.get("status", "pending")),
            "passed": str(active.get("status", "")) == "passed",
            "failure_reason": active.get("failure_reason"),
            "trades": result.trades,
        }
    except Exception as e:
        logging.warning("Backtest failed: %s", e)
        return None


def suggest_mcl_params(trial, base_config):
    """MCL-specific parameter space."""
    params = copy.deepcopy(base_config)

    # Risk params
    params["risk"]["initial_risk_pct"] = trial.suggest_float("risk_init", 0.3, 2.0, step=0.1)
    params["risk"]["reduced_risk_pct"] = trial.suggest_float("risk_red", 0.3, 2.0, step=0.1)
    params["risk"]["daily_circuit_breaker_pct"] = trial.suggest_float("daily_cb", 1.5, 4.5, step=0.5)
    params["risk"]["max_concurrent_positions"] = trial.suggest_int("max_conc", 1, 5)

    # Exit params
    params["exit"]["tp_r_multiple"] = trial.suggest_float("tp_r", 1.0, 3.0, step=0.25)
    params["exit"]["breakeven_threshold_r"] = trial.suggest_float("be_r", 0.5, 1.5, step=0.25)

    # SSS — oil prices are ~$70-100 (much smaller than gold $5000+)
    params["strategies"]["sss"]["swing_lookback_n"] = trial.suggest_int("sss_lb", 2, 5)
    params["strategies"]["sss"]["min_swing_pips"] = trial.suggest_float("sss_sw", 0.01, 1.0, step=0.01)
    params["strategies"]["sss"]["min_confluence_score"] = trial.suggest_int("sss_conf", 0, 4)
    params["strategies"]["sss"]["min_stop_pips"] = trial.suggest_float("sss_stop", 0.05, 3.0, step=0.05)
    params["strategies"]["sss"]["rr_ratio"] = trial.suggest_float("sss_rr", 1.5, 3.0, step=0.25)
    params["strategies"]["sss"]["entry_mode"] = trial.suggest_categorical("sss_mode", ["cbc_only", "fifty_tap", "combined"])
    params["strategies"]["sss"]["spread_multiplier"] = trial.suggest_float("sss_spread", 1.0, 3.0, step=0.5)

    # Ichimoku
    scale = trial.suggest_float("ichi_scale", 0.7, 1.3, step=0.05)
    params["strategies"]["ichimoku"]["ichimoku"]["tenkan_period"] = max(3, round(9 * scale))
    params["strategies"]["ichimoku"]["ichimoku"]["kijun_period"] = max(9, round(26 * scale))
    params["strategies"]["ichimoku"]["ichimoku"]["senkou_b_period"] = max(18, round(52 * scale))
    params["strategies"]["ichimoku"]["atr"]["stop_multiplier"] = trial.suggest_float("ichi_atr", 1.0, 3.0, step=0.25)
    params["strategies"]["ichimoku"]["adx"]["threshold"] = trial.suggest_int("ichi_adx", 15, 30)
    params["strategies"]["ichimoku"]["signal"]["min_confluence_score"] = trial.suggest_int("ichi_conf", 1, 4)
    params["strategies"]["ichimoku"]["signal"]["tier_c"] = trial.suggest_int("ichi_tc", 1, 3)

    # Asian Breakout — oil ranges are much smaller than gold
    params["strategies"]["asian_breakout"]["min_range_pips"] = trial.suggest_float("ab_min", 0.01, 1.0, step=0.01)
    params["strategies"]["asian_breakout"]["max_range_pips"] = trial.suggest_float("ab_max", 1.0, 15.0, step=0.5)
    params["strategies"]["asian_breakout"]["rr_ratio"] = trial.suggest_float("ab_rr", 1.5, 3.0, step=0.25)

    # EMA Pullback
    params["strategies"]["ema_pullback"]["min_ema_angle_deg"] = trial.suggest_float("ep_angle", 0.5, 5.0, step=0.5)
    params["strategies"]["ema_pullback"]["pullback_candles_max"] = trial.suggest_int("ep_pb", 5, 25)
    params["strategies"]["ema_pullback"]["rr_ratio"] = trial.suggest_float("ep_rr", 1.5, 3.0, step=0.25)

    # FX At One Glance
    params["strategies"]["fx_at_one_glance"]["signal"]["min_confluence_score"] = trial.suggest_int("fxaog_conf", 3, 7)
    params["strategies"]["fx_at_one_glance"]["five_elements_mode"] = trial.suggest_categorical(
        "fxaog_5e", ["hard_gate", "soft_filter", "disabled"]
    )
    params["strategies"]["fx_at_one_glance"]["stop_loss"]["min_rr_ratio"] = trial.suggest_float(
        "fxaog_rr", 1.0, 2.5, step=0.25
    )

    return params


def permute_candles(df, seed):
    """Shuffle bar-to-bar log returns, reconstruct prices."""
    rng = np.random.default_rng(seed)
    closes = df["close"].values.astype(float)
    log_returns = np.diff(np.log(closes))
    shuffled = log_returns.copy()
    rng.shuffle(shuffled)
    new_closes = np.empty(len(closes))
    new_closes[0] = closes[0]
    for i in range(len(shuffled)):
        new_closes[i + 1] = new_closes[i] * np.exp(shuffled[i])
    ratio = new_closes / closes
    new_df = df.copy()
    for col in ["open", "high", "low", "close"]:
        if col in new_df.columns:
            new_df[col] = (df[col].values * ratio).astype(float)
    new_df["high"] = new_df[["open", "high", "close"]].max(axis=1)
    new_df["low"] = new_df[["open", "low", "close"]].min(axis=1)
    return new_df


def main():
    start_time = time.time()
    print("=" * 60)
    print("  MCL MICRO CRUDE OIL — TOPSTEPX COMBINE OPTIMIZATION")
    print("=" * 60, flush=True)

    data = load_data()
    base_config = build_base_config()
    print(f"Active strategies: {base_config['active_strategies']}", flush=True)

    # Phase 1: Optimization
    print("\n--- PHASE 1: OPTUNA OPTIMIZATION ---", flush=True)

    def objective(trial):
        params = suggest_mcl_params(trial, base_config)
        result = run_backtest(data, params)
        if not result or result["total_trades"] == 0:
            return -2.0

        if result["passed"]:
            score = 1.0
        else:
            profit = result["final_balance"] - INITIAL_BALANCE
            score = max(-1.0, min(1.0, profit / 3000.0))
            if "failed_mll" in result["status"]:
                score -= 0.5
            elif "failed_daily_loss" in result["status"]:
                score -= 0.3

        score += min(result["total_trades"], 30) / 200.0
        if result["win_rate"] > 0.4:
            score += 0.1

        trial.set_user_attr("passed", result["passed"])
        trial.set_user_attr("trades", result["total_trades"])
        trial.set_user_attr("balance", result["final_balance"])
        trial.set_user_attr("status", result["status"])
        trial.set_user_attr("wr", round(result["win_rate"], 3))
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

    best_passing = None
    n_trials = 200
    for i in range(n_trials):
        study.optimize(objective, n_trials=1)
        t = study.trials[-1]
        p = t.user_attrs.get("passed", False)

        if (i + 1) % 10 == 0 or p:
            print(
                f"Trial {i+1}: score={t.value:.3f}, passed={p}, "
                f"trades={t.user_attrs.get('trades', 0)}, "
                f"bal=${t.user_attrs.get('balance', 0):.0f}, "
                f"wr={t.user_attrs.get('wr', 0):.1%}, "
                f"status={t.user_attrs.get('status', '?')}",
                flush=True,
            )

        if p and best_passing is None:
            best_passing = i
            print(f"*** FIRST PASSING CONFIG AT TRIAL {i+1}! ***", flush=True)

        # Stop 40 trials after first pass for refinement
        if best_passing is not None and i >= best_passing + 40:
            print("Found passing config + 40 refinement trials. Stopping.", flush=True)
            break

    best = study.best_trial
    print(f"\n=== BEST TRIAL #{best.number} ===")
    print(f"Score: {best.value:.4f}")
    print(f"Passed: {best.user_attrs.get('passed')}")
    print(f"Trades: {best.user_attrs.get('trades')}")
    print(f"Balance: ${best.user_attrs.get('balance', 0):.2f}")
    print(f"Win rate: {best.user_attrs.get('wr', 0):.1%}")
    print(f"Status: {best.user_attrs.get('status')}", flush=True)

    # Reconstruct best config
    best_config = suggest_mcl_params(best, base_config)

    # Phase 2: Check 3 consecutive passes
    print("\n--- PHASE 2: 3 CONSECUTIVE COMBINE PASSES ---", flush=True)
    bars_per_day = 60 * 23
    offsets = [0, bars_per_day * 2, bars_per_day * 5]
    pass_results = []
    for i, offset in enumerate(offsets):
        window = data.iloc[offset:]
        if len(window) < 5000:
            window = data
        result = run_backtest(window, best_config)
        passed = result["passed"] if result else False
        pass_results.append(result or {"passed": False, "status": "error"})
        print(
            f"Attempt {i+1}: {'PASSED' if passed else 'FAILED'} — "
            f"trades={result['total_trades'] if result else 0}, "
            f"bal=${result['final_balance'] if result else 0:.2f}, "
            f"status={result['status'] if result else 'error'}",
            flush=True,
        )

    all_passed = all(r.get("passed", False) for r in pass_results)
    print(f"\nAll 3 passed: {all_passed}", flush=True)

    # Phase 3: Permutation test
    print("\n--- PHASE 3: PERMUTATION TEST (20 permutations) ---", flush=True)
    real_result = run_backtest(data, best_config)
    real_return = real_result["total_return_pct"] if real_result else 0
    print(f"Real return: {real_return:.2f}%", flush=True)

    perm_returns = []
    perm_passes = 0
    for i in range(20):
        perm_data = permute_candles(data, seed=42 + i)
        pr = run_backtest(perm_data, best_config)
        if pr:
            perm_returns.append(pr["total_return_pct"])
            if pr["passed"]:
                perm_passes += 1
            print(f"  Perm {i+1}/20: return={pr['total_return_pct']:.2f}%, passed={pr['passed']}", flush=True)
        else:
            perm_returns.append(0)

    beats = sum(1 for pr in perm_returns if pr >= real_return)
    p_value = beats / len(perm_returns) if perm_returns else 1.0

    print(f"\n=== PERMUTATION RESULTS ===")
    print(f"Real return:     {real_return:.2f}%")
    print(f"Mean permuted:   {np.mean(perm_returns):.2f}%")
    print(f"P-value:         {p_value:.4f}")
    print(f"Significant:     {p_value < 0.05}")
    print(f"Permuted passes: {perm_passes}/20", flush=True)

    # Save report
    elapsed = time.time() - start_time
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    report = {
        "instrument": INSTRUMENT,
        "timestamp": ts,
        "optimization_trials": len(study.trials),
        "best_trial": best.number,
        "best_score": best.value,
        "best_params": best.params,
        "consecutive_passes": all_passed,
        "pass_results": [{k: v for k, v in r.items() if k != "trades"} for r in pass_results],
        "real_return_pct": real_return,
        "mean_permuted_pct": float(np.mean(perm_returns)),
        "p_value": p_value,
        "significant": p_value < 0.05,
        "permuted_passes": perm_passes,
        "elapsed_minutes": elapsed / 60,
    }
    report_path = _PROJECT_ROOT / "reports" / f"mcl_optimization_{ts}.json"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")

    print(f"\n{'='*60}")
    print(f"  VERDICT: {'STRATEGY HAS REAL EDGE' if all_passed and p_value < 0.05 else 'NEEDS MORE WORK'}")
    print(f"  Elapsed: {elapsed/60:.1f} minutes")
    print(f"{'='*60}", flush=True)

    return 0 if all_passed and p_value < 0.05 else 1


if __name__ == "__main__":
    sys.exit(main())
