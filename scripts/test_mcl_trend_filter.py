"""Test MCL with trend direction filter: 3 consecutive passes + permutation test."""
from __future__ import annotations
import copy, json, logging, sys, time, warnings
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd, yaml

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
logging.basicConfig(level=logging.WARNING)

INITIAL_BALANCE = 50_000.0

data = pd.read_parquet(_ROOT / "data/projectx_mcl_1m_20260101_20260411.parquet")
if data.index.tz is None:
    data.index = data.index.tz_localize("UTC")
data.columns = data.columns.str.lower()
print(f"MCL: {len(data)} bars", flush=True)

from src.config.loader import ConfigLoader
loader = ConfigLoader(config_dir=str(_ROOT / "config"))
app_config = loader.load()
with (_ROOT / "config/strategy.yaml").open() as f:
    raw = yaml.safe_load(f) or {}
ss = app_config.strategy.model_dump()
cfg = {}
if hasattr(app_config, "edges"):
    cfg["edges"] = app_config.edges.model_dump()
cfg["active_strategies"] = ["sss", "ichimoku", "asian_breakout", "ema_pullback"]
cfg["strategies"] = ss.get("strategies", {})
for k in ("risk", "exit", "prop_firm"):
    if k in ss:
        cfg[k] = ss[k]

# Best MCL params from v1 optimization
cfg["risk"]["initial_risk_pct"] = 0.6
cfg["risk"]["reduced_risk_pct"] = 1.8
cfg["risk"]["daily_circuit_breaker_pct"] = 3.5
cfg["risk"]["max_concurrent_positions"] = 4
cfg["risk"]["max_lot_size"] = 20
cfg["exit"]["tp_r_multiple"] = 1.0
cfg["exit"]["breakeven_threshold_r"] = 1.5
cfg["strategies"]["sss"]["min_swing_pips"] = 0.42
cfg["strategies"]["sss"]["min_stop_pips"] = 0.68
cfg["strategies"]["sss"]["min_confluence_score"] = 2
cfg["strategies"]["sss"]["rr_ratio"] = 1.5
cfg["strategies"]["sss"]["entry_mode"] = "combined"
cfg["strategies"]["sss"]["spread_multiplier"] = 3.0
cfg["strategies"]["sss"]["fifty_tap_tolerance_pips"] = 0.27
cfg["strategies"]["ichimoku"]["atr"]["stop_multiplier"] = 3.0
cfg["strategies"]["ichimoku"]["adx"]["threshold"] = 23
cfg["strategies"]["ichimoku"]["signal"]["min_confluence_score"] = 2
cfg["strategies"]["ichimoku"]["signal"]["tier_c"] = 3
cfg["strategies"]["asian_breakout"]["min_range_pips"] = 0.3
cfg["strategies"]["asian_breakout"]["max_range_pips"] = 9.5
cfg["strategies"]["asian_breakout"]["rr_ratio"] = 3.0
cfg["strategies"]["ema_pullback"]["min_ema_angle_deg"] = 1.0
cfg["strategies"]["ema_pullback"]["pullback_candles_max"] = 19
cfg["strategies"]["ema_pullback"]["rr_ratio"] = 2.0

inst = app_config.instruments.get("MCLOIL")
cfg["instrument_class"] = inst.class_.value
cfg["instrument"] = {
    "symbol": inst.symbol, "class": inst.class_.value,
    "tick_size": inst.tick_size, "tick_value_usd": inst.tick_value_usd,
    "contract_size": inst.contract_size,
    "commission_per_contract_round_trip": inst.commission_per_contract_round_trip,
    "session_open_ct": inst.session_open_ct, "session_close_ct": inst.session_close_ct,
    "daily_reset_hour_ct": inst.daily_reset_hour_ct,
}
cfg["prop_firm"] = cfg.get("prop_firm", {})


def run_bt(d, c):
    from src.backtesting.vectorbt_engine import IchimokuBacktester
    try:
        bt = IchimokuBacktester(config=c, initial_balance=INITIAL_BALANCE)
        r = bt.run(candles_1m=d, instrument="MCLOIL", log_trades=False, enable_learning=False)
        m = r.metrics
        p = r.prop_firm
        a = p.get("active_tracker", p)
        return {
            "total_trades": m.get("total_trades", 0),
            "win_rate": m.get("win_rate", 0),
            "total_return_pct": m.get("total_return_pct", 0),
            "final_balance": float(a.get("current_balance", INITIAL_BALANCE)),
            "status": str(a.get("status", "pending")),
            "passed": str(a.get("status", "")) == "passed",
            "pipeline": m.get("pipeline_counts", {}),
        }
    except Exception as e:
        logging.warning("BT fail: %s", e)
        return None


# --- 3 CONSECUTIVE PASSES ---
print("\n=== 3 CONSECUTIVE COMBINE PASSES ===", flush=True)
bars_per_day = 60 * 23
offsets = [0, bars_per_day * 3, bars_per_day * 7]
all_pass = True
for i, off in enumerate(offsets):
    window = data.iloc[off:]
    if len(window) < 5000:
        window = data
    r = run_bt(window, cfg)
    passed = r["passed"] if r else False
    if not passed:
        all_pass = False
    print(
        f"Attempt {i+1}: {'PASSED' if passed else 'FAILED'} - "
        f"trades={r['total_trades'] if r else 0}, "
        f"bal=${r['final_balance'] if r else 0:.0f}, "
        f"wr={r['win_rate']*100 if r else 0:.1f}%, "
        f"status={r['status'] if r else '?'}, "
        f"edge_filtered={r['pipeline'].get('signals_filtered_edge', 0) if r else 0}",
        flush=True,
    )

print(f"\nAll 3 passed: {all_pass}", flush=True)

# --- PERMUTATION TEST ---
print("\n=== PERMUTATION TEST (20 permutations) ===", flush=True)
real_r = run_bt(data, cfg)
real_ret = real_r["total_return_pct"] if real_r else 0
print(f"Real return: {real_ret:.2f}%, trades={real_r['total_trades'] if real_r else 0}, "
      f"edge_filtered={real_r['pipeline'].get('signals_filtered_edge', 0) if real_r else 0}", flush=True)

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

    pr = run_bt(perm, cfg)
    if pr:
        perm_returns.append(pr["total_return_pct"])
        if pr["passed"]:
            perm_passes += 1
        print(f"  Perm {i+1}/20: return={pr['total_return_pct']:.2f}%, "
              f"trades={pr['total_trades']}, passed={pr['passed']}", flush=True)
    else:
        perm_returns.append(0)
        print(f"  Perm {i+1}/20: FAILED", flush=True)

beats = sum(1 for pr in perm_returns if pr >= real_ret)
p_value = beats / len(perm_returns) if perm_returns else 1.0

print(f"\n=== RESULTS ===")
print(f"Real return:     {real_ret:.2f}%")
print(f"Mean permuted:   {np.mean(perm_returns):.2f}%")
print(f"P-value:         {p_value:.4f}")
print(f"Significant:     {p_value < 0.05}")
print(f"Permuted passes: {perm_passes}/20")
print(f"All 3 passed:    {all_pass}", flush=True)
