"""Fast MCL optimization — no FXAOG coordinator for speed."""
from __future__ import annotations
import copy, json, logging, sys, time, warnings
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, optuna, pandas as pd, yaml

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
logging.basicConfig(level=logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)

data = pd.read_parquet(_ROOT / "data/projectx_mcl_1m_20260101_20260411.parquet")
if data.index.tz is None:
    data.index = data.index.tz_localize("UTC")
data.columns = data.columns.str.lower()
print(f"MCL: {len(data)} bars, ${data['close'].min():.2f}-${data['close'].max():.2f}", flush=True)

from src.config.loader import ConfigLoader
loader = ConfigLoader(config_dir=str(_ROOT / "config"))
app_config = loader.load()
with (_ROOT / "config/strategy.yaml").open() as f:
    raw_strat = yaml.safe_load(f) or {}
ss = app_config.strategy.model_dump()
base = {}
if hasattr(app_config, "edges"):
    base["edges"] = app_config.edges.model_dump()
base["active_strategies"] = ["sss", "ichimoku", "asian_breakout", "ema_pullback"]
base["strategies"] = ss.get("strategies", {})
for k in ("risk", "exit", "prop_firm"):
    if k in ss:
        base[k] = ss[k]
inst = app_config.instruments.get("MCLOIL")
base["instrument_class"] = inst.class_.value
base["instrument"] = {
    "symbol": inst.symbol, "class": inst.class_.value,
    "tick_size": inst.tick_size, "tick_value_usd": inst.tick_value_usd,
    "contract_size": inst.contract_size,
    "commission_per_contract_round_trip": inst.commission_per_contract_round_trip,
    "session_open_ct": inst.session_open_ct, "session_close_ct": inst.session_close_ct,
    "daily_reset_hour_ct": inst.daily_reset_hour_ct,
}
base["prop_firm"] = base.get("prop_firm", {})


def run_bt(config):
    from src.backtesting.vectorbt_engine import IchimokuBacktester
    try:
        bt = IchimokuBacktester(config=config, initial_balance=50000.0)
        result = bt.run(candles_1m=data, instrument="MCLOIL", log_trades=False, enable_learning=False)
        m = result.metrics
        prop = result.prop_firm
        active = prop.get("active_tracker", prop)
        return {
            "total_trades": m.get("total_trades", 0),
            "win_rate": m.get("win_rate", 0),
            "total_return_pct": m.get("total_return_pct", 0),
            "final_balance": float(active.get("current_balance", 50000)),
            "status": str(active.get("status", "pending")),
            "passed": str(active.get("status", "")) == "passed",
            "pipeline": m.get("pipeline_counts", {}),
            "trades": result.trades,
        }
    except Exception as e:
        logging.warning("BT fail: %s", e)
        return None


# Diagnostic first
print("\n--- Diagnostic: default params ---", flush=True)
diag = run_bt(base)
if diag:
    print(f"Trades: {diag['total_trades']}, WR: {diag['win_rate']*100:.1f}%, "
          f"Ret: {diag['total_return_pct']:.2f}%, Bal: ${diag['final_balance']:.0f}, "
          f"Status: {diag['status']}", flush=True)
    print(f"Pipeline: {diag['pipeline']}", flush=True)
else:
    print("DIAGNOSTIC FAILED", flush=True)


def objective(trial):
    params = copy.deepcopy(base)

    # Oil needs lower risk % — contracts are small but many get filled
    params["risk"]["initial_risk_pct"] = trial.suggest_float("ri", 0.1, 0.8, step=0.05)
    params["risk"]["reduced_risk_pct"] = trial.suggest_float("rr", 0.1, 0.8, step=0.05)
    params["risk"]["daily_circuit_breaker_pct"] = trial.suggest_float("cb", 2.0, 5.0, step=0.5)
    params["risk"]["max_concurrent_positions"] = trial.suggest_int("mc", 1, 3)
    params["risk"]["max_lot_size"] = trial.suggest_int("ml", 5, 30)

    params["exit"]["tp_r_multiple"] = trial.suggest_float("tp", 1.0, 3.0, step=0.25)
    params["exit"]["breakeven_threshold_r"] = trial.suggest_float("be", 0.5, 1.5, step=0.25)

    # SSS — oil ticks are $0.01 vs gold $0.10, wider stops needed
    params["strategies"]["sss"]["min_swing_pips"] = trial.suggest_float("ss", 0.01, 0.3, step=0.01)
    params["strategies"]["sss"]["min_stop_pips"] = trial.suggest_float("sst", 0.05, 2.0, step=0.05)
    params["strategies"]["sss"]["min_confluence_score"] = trial.suggest_int("sc", 0, 3)
    params["strategies"]["sss"]["rr_ratio"] = trial.suggest_float("sr", 1.5, 3.0, step=0.25)
    params["strategies"]["sss"]["entry_mode"] = trial.suggest_categorical("se", ["cbc_only", "fifty_tap", "combined"])
    params["strategies"]["sss"]["spread_multiplier"] = trial.suggest_float("sp", 1.0, 3.0, step=0.5)
    params["strategies"]["sss"]["fifty_tap_tolerance_pips"] = trial.suggest_float("ft", 0.01, 0.3, step=0.01)

    # Ichimoku
    params["strategies"]["ichimoku"]["atr"]["stop_multiplier"] = trial.suggest_float("ia", 1.0, 3.0, step=0.25)
    params["strategies"]["ichimoku"]["adx"]["threshold"] = trial.suggest_int("id", 12, 30)
    params["strategies"]["ichimoku"]["signal"]["min_confluence_score"] = trial.suggest_int("ic", 1, 4)
    params["strategies"]["ichimoku"]["signal"]["tier_c"] = trial.suggest_int("it", 1, 3)

    # Asian Breakout — oil ranges
    params["strategies"]["asian_breakout"]["min_range_pips"] = trial.suggest_float("am", 0.01, 0.5, step=0.01)
    params["strategies"]["asian_breakout"]["max_range_pips"] = trial.suggest_float("ax", 0.5, 10.0, step=0.5)
    params["strategies"]["asian_breakout"]["rr_ratio"] = trial.suggest_float("ar", 1.5, 3.0, step=0.25)

    # EMA Pullback
    params["strategies"]["ema_pullback"]["min_ema_angle_deg"] = trial.suggest_float("ea", 0.5, 5.0, step=0.5)
    params["strategies"]["ema_pullback"]["pullback_candles_max"] = trial.suggest_int("ep", 5, 30)
    params["strategies"]["ema_pullback"]["rr_ratio"] = trial.suggest_float("er", 1.5, 3.0, step=0.25)

    result = run_bt(params)
    if not result or result["total_trades"] == 0:
        return -2.0

    if result["passed"]:
        score = 1.0
    else:
        profit = result["final_balance"] - 50000
        score = max(-1.0, min(1.0, profit / 3000.0))
        if "failed_mll" in result["status"]:
            score -= 0.5
        elif "failed_daily" in result["status"]:
            score -= 0.3

    score += min(result["total_trades"], 30) / 200.0
    if result["win_rate"] > 0.4:
        score += 0.1

    trial.set_user_attr("p", result["passed"])
    trial.set_user_attr("t", result["total_trades"])
    trial.set_user_attr("b", result["final_balance"])
    trial.set_user_attr("s", result["status"])
    trial.set_user_attr("w", round(result["win_rate"], 3))
    return score


print("\n--- OPTUNA OPTIMIZATION (no FXAOG for speed) ---", flush=True)
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

best_passing = None
for i in range(150):
    study.optimize(objective, n_trials=1)
    t = study.trials[-1]
    p = t.user_attrs.get("p", False)

    if (i + 1) % 5 == 0 or p:
        print(
            f"Trial {i+1}: score={t.value:.3f}, passed={p}, "
            f"trades={t.user_attrs.get('t', 0)}, "
            f"bal=${t.user_attrs.get('b', 0):.0f}, "
            f"wr={t.user_attrs.get('w', 0):.1%}, "
            f"status={t.user_attrs.get('s', '?')}",
            flush=True,
        )

    if p and best_passing is None:
        best_passing = i
        print(f"*** FIRST PASS AT TRIAL {i+1}! ***", flush=True)

    if best_passing is not None and i >= best_passing + 30:
        print("Stopping — 30 refinement trials after first pass.", flush=True)
        break

best = study.best_trial
print(f"\n=== BEST TRIAL #{best.number} ===", flush=True)
print(f"Score: {best.value:.4f}", flush=True)
print(f"Passed: {best.user_attrs.get('p')}", flush=True)
print(f"Trades: {best.user_attrs.get('t')}", flush=True)
print(f"Balance: ${best.user_attrs.get('b', 0):.2f}", flush=True)
print(f"Win rate: {best.user_attrs.get('w', 0):.1%}", flush=True)
print(f"Status: {best.user_attrs.get('s')}", flush=True)
print(f"Params: {json.dumps(best.params, indent=2)}", flush=True)

# Check 3 consecutive passes with best config
best_config = copy.deepcopy(base)
# Apply best params
bp = best.params
best_config["risk"]["initial_risk_pct"] = bp["ri"]
best_config["risk"]["reduced_risk_pct"] = bp["rr"]
best_config["risk"]["daily_circuit_breaker_pct"] = bp["cb"]
best_config["risk"]["max_concurrent_positions"] = bp["mc"]
best_config["risk"]["max_lot_size"] = bp.get("ml", 20)
best_config["exit"]["tp_r_multiple"] = bp["tp"]
best_config["exit"]["breakeven_threshold_r"] = bp["be"]
best_config["strategies"]["sss"]["min_swing_pips"] = bp["ss"]
best_config["strategies"]["sss"]["min_stop_pips"] = bp["sst"]
best_config["strategies"]["sss"]["min_confluence_score"] = bp["sc"]
best_config["strategies"]["sss"]["rr_ratio"] = bp["sr"]
best_config["strategies"]["sss"]["entry_mode"] = bp["se"]
best_config["strategies"]["sss"]["spread_multiplier"] = bp["sp"]
best_config["strategies"]["sss"]["fifty_tap_tolerance_pips"] = bp["ft"]
best_config["strategies"]["ichimoku"]["atr"]["stop_multiplier"] = bp["ia"]
best_config["strategies"]["ichimoku"]["adx"]["threshold"] = bp["id"]
best_config["strategies"]["ichimoku"]["signal"]["min_confluence_score"] = bp["ic"]
best_config["strategies"]["ichimoku"]["signal"]["tier_c"] = bp["it"]
best_config["strategies"]["asian_breakout"]["min_range_pips"] = bp["am"]
best_config["strategies"]["asian_breakout"]["max_range_pips"] = bp["ax"]
best_config["strategies"]["asian_breakout"]["rr_ratio"] = bp["ar"]
best_config["strategies"]["ema_pullback"]["min_ema_angle_deg"] = bp["ea"]
best_config["strategies"]["ema_pullback"]["pullback_candles_max"] = bp["ep"]
best_config["strategies"]["ema_pullback"]["rr_ratio"] = bp["er"]

print("\n--- 3 CONSECUTIVE COMBINE PASSES ---", flush=True)
bars_per_day = 60 * 23
offsets = [0, bars_per_day * 3, bars_per_day * 7]
all_pass = True
for i, off in enumerate(offsets):
    window = data.iloc[off:]
    if len(window) < 5000:
        window = data
    r = run_bt(best_config) if off == 0 else None
    if off > 0:
        from src.backtesting.vectorbt_engine import IchimokuBacktester
        try:
            bt = IchimokuBacktester(config=best_config, initial_balance=50000.0)
            result = bt.run(candles_1m=window, instrument="MCLOIL", log_trades=False, enable_learning=False)
            m = result.metrics
            prop = result.prop_firm
            active = prop.get("active_tracker", prop)
            r = {
                "total_trades": m.get("total_trades", 0),
                "win_rate": m.get("win_rate", 0),
                "final_balance": float(active.get("current_balance", 50000)),
                "status": str(active.get("status", "pending")),
                "passed": str(active.get("status", "")) == "passed",
            }
        except:
            r = {"passed": False, "status": "error", "total_trades": 0, "final_balance": 50000, "win_rate": 0}
    passed = r["passed"] if r else False
    if not passed:
        all_pass = False
    print(
        f"Attempt {i+1}: {'PASSED' if passed else 'FAILED'} — "
        f"trades={r['total_trades'] if r else 0}, "
        f"bal=${r['final_balance'] if r else 0:.0f}, "
        f"wr={r['win_rate']*100 if r else 0:.1f}%, "
        f"status={r['status'] if r else '?'}",
        flush=True,
    )

print(f"\nAll 3 passed: {all_pass}", flush=True)

# Permutation test
print("\n--- PERMUTATION TEST (20 permutations) ---", flush=True)
real_r = run_bt(best_config)
real_ret = real_r["total_return_pct"] if real_r else 0
print(f"Real return: {real_ret:.2f}%", flush=True)

perm_returns = []
perm_passes = 0
for i in range(20):
    rng = np.random.default_rng(42 + i)
    closes = data["close"].values.astype(float)
    lr = np.diff(np.log(closes))
    shuffled = lr.copy()
    rng.shuffle(shuffled)
    nc = np.empty(len(closes))
    nc[0] = closes[0]
    for j in range(len(shuffled)):
        nc[j + 1] = nc[j] * np.exp(shuffled[j])
    ratio = nc / closes
    perm = data.copy()
    for col in ["open", "high", "low", "close"]:
        perm[col] = (data[col].values * ratio).astype(float)
    perm["high"] = perm[["open", "high", "close"]].max(axis=1)
    perm["low"] = perm[["open", "low", "close"]].min(axis=1)

    pr = run_bt(best_config)  # use perm data
    # Actually need to run on permuted data
    from src.backtesting.vectorbt_engine import IchimokuBacktester
    try:
        bt2 = IchimokuBacktester(config=best_config, initial_balance=50000.0)
        res2 = bt2.run(candles_1m=perm, instrument="MCLOIL", log_trades=False, enable_learning=False)
        pm = res2.metrics
        pprop = res2.prop_firm
        pactive = pprop.get("active_tracker", pprop)
        pret = pm.get("total_return_pct", 0)
        ppassed = str(pactive.get("status", "")) == "passed"
    except:
        pret = 0
        ppassed = False

    perm_returns.append(pret)
    if ppassed:
        perm_passes += 1
    print(f"  Perm {i+1}/20: return={pret:.2f}%, passed={ppassed}", flush=True)

beats = sum(1 for pr in perm_returns if pr >= real_ret)
p_value = beats / len(perm_returns) if perm_returns else 1.0

print(f"\n=== PERMUTATION RESULTS ===")
print(f"Real return:     {real_ret:.2f}%")
print(f"Mean permuted:   {np.mean(perm_returns):.2f}%")
print(f"P-value:         {p_value:.4f}")
print(f"Significant:     {p_value < 0.05}")
print(f"Permuted passes: {perm_passes}/20", flush=True)

# Save report
ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
report = {
    "instrument": "MCLOIL",
    "timestamp": ts,
    "best_params": best.params,
    "best_score": best.value,
    "consecutive_passes": all_pass,
    "real_return_pct": real_ret,
    "p_value": p_value,
    "significant": p_value < 0.05,
    "permuted_passes": perm_passes,
    "n_trials": len(study.trials),
}
rpath = _ROOT / "reports" / f"mcl_optimization_{ts}.json"
rpath.parent.mkdir(exist_ok=True)
with open(rpath, "w") as f:
    json.dump(report, f, indent=2, default=str)
print(f"\nReport: {rpath}", flush=True)
