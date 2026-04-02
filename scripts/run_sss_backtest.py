"""
SSS Strategy Backtest & Optimization Runner.

Runs the Shit Show Sequence strategy through:
  1. Baseline backtest on real XAU/USD data
  2. Optuna optimization (7-param search space)
  3. Best-params backtest
  4. Dashboard generation (static HTML + live optimization dashboard)

Usage
-----
    python scripts/run_sss_backtest.py --data-file data/xauusd_1m_2023_2025.parquet
    python scripts/run_sss_backtest.py --data-file data/xauusd_1m_2023_2025.parquet --n-trials 100
    python scripts/run_sss_backtest.py --data-file data/xauusd_1m_2023_2025.parquet --skip-optimize
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_sss_backtest")


# =============================================================================
# CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SSS Strategy backtest & optimization runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-file", type=str, required=True,
                    help="Path to 1-minute OHLCV parquet/CSV file.")
    p.add_argument("--initial-balance", type=float, default=10_000.0,
                    help="Starting account balance.")
    p.add_argument("--n-trials", type=int, default=50,
                    help="Optuna optimization trials.")
    p.add_argument("--skip-optimize", action="store_true",
                    help="Skip Optuna optimization, just run baseline backtest.")
    p.add_argument("--output-dir", type=str, default="reports",
                    help="Directory for output reports.")
    p.add_argument("--max-bars", type=int, default=None,
                    help="Limit input data to N 1M bars (for quick testing).")
    p.add_argument("--dashboard-port", type=int, default=8501,
                    help="Port for the optimization dashboard.")
    return p


# =============================================================================
# Data loading
# =============================================================================

def load_data(path: str, max_bars: Optional[int] = None) -> pd.DataFrame:
    """Load 1M OHLCV data from parquet or CSV."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    logger.info("Loading data from %s", path)
    if p.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=True, index_col=0)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    df.columns = df.columns.str.lower()
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if max_bars:
        df = df.iloc[:max_bars]

    logger.info("Loaded %d bars: %s to %s", len(df), df.index[0], df.index[-1])
    return df


# =============================================================================
# Config builder
# =============================================================================

def build_sss_config(config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Build a config dict for SSS-only backtesting from YAML files."""
    config_dir = config_dir or (_PROJECT_ROOT / "config")

    # Load raw YAML (bypasses Pydantic to get all fields)
    strategy_yaml = config_dir / "strategy.yaml"
    edges_yaml = config_dir / "edges.yaml"

    raw_strat = {}
    if strategy_yaml.exists():
        with strategy_yaml.open() as f:
            raw_strat = yaml.safe_load(f) or {}

    raw_edges = {}
    if edges_yaml.exists():
        with edges_yaml.open() as f:
            raw_edges = yaml.safe_load(f) or {}

    # Build config targeting SSS only
    config: Dict[str, Any] = {
        "active_strategies": ["sss"],
        "strategies": raw_strat.get("strategies", {}),
        "edges": raw_edges,
        "risk": raw_strat.get("risk", {}),
        "exit": raw_strat.get("exit", {}),
        "prop_firm": raw_strat.get("prop_firm", {}),
    }

    # Flatten some risk params to top-level for backward compat
    risk = config.get("risk", {})
    config["initial_risk_pct"] = risk.get("initial_risk_pct", 0.5)
    config["reduced_risk_pct"] = risk.get("reduced_risk_pct", 0.75)
    config["phase_threshold_pct"] = risk.get("phase_threshold_pct", 4.0)

    # Flatten exit params
    exit_cfg = config.get("exit", {})
    config["tp_r_multiple"] = exit_cfg.get("tp_r_multiple", 2.0)
    config["kijun_trail_start_r"] = exit_cfg.get("kijun_trail_start_r", 1.5)
    config["higher_tf_kijun_start_r"] = exit_cfg.get("higher_tf_kijun_start_r", 3.0)
    config["atr_stop_multiplier"] = raw_strat.get("strategies", {}).get(
        "ichimoku", {}
    ).get("atr", {}).get("stop_multiplier", 2.5)

    return config


# =============================================================================
# Run backtest
# =============================================================================

def run_backtest(
    data: pd.DataFrame,
    config: Dict[str, Any],
    initial_balance: float = 10_000.0,
    label: str = "baseline",
    live_dashboard=None,
) -> Any:
    """Run a single SSS backtest and return the BacktestResult."""
    from src.backtesting.vectorbt_engine import IchimokuBacktester

    prop = config.get("prop_firm", {})
    p1 = prop.get("phase_1", {})

    backtester = IchimokuBacktester(
        config=config,
        initial_balance=initial_balance,
        prop_firm_profit_target_pct=float(p1.get("profit_target_pct", 8.0)),
        prop_firm_max_daily_dd_pct=float(p1.get("daily_loss_pct", 5.0)),
        prop_firm_max_total_dd_pct=float(p1.get("max_loss_pct", 10.0)),
        prop_firm_time_limit_days=int(p1.get("time_limit_days", 0)) or 365,
    )

    # Wire edge manager to dashboard for runtime config access
    if live_dashboard is not None:
        live_dashboard._edge_manager = backtester.edge_manager

    logger.info("Running %s backtest on %d 1M bars ...", label, len(data))
    t0 = time.monotonic()
    result = backtester.run(
        data, instrument="XAUUSD", log_trades=False, enable_learning=True,
        live_dashboard=live_dashboard,
    )
    elapsed = time.monotonic() - t0
    logger.info("%s backtest complete in %.1fs — %d trades", label, elapsed, len(result.trades))

    return result


def print_results(result: Any, label: str = "BACKTEST", initial_balance: float = 10_000.0):
    """Print backtest results to stdout."""
    m = result.metrics
    pf = result.prop_firm

    print(f"\n{'=' * 60}")
    print(f"  {label} RESULTS — XAUUSD SSS Strategy")
    print(f"{'=' * 60}")
    print(f"  Total trades:       {m.get('total_trades', 0)}")
    print(f"  Win rate:           {m.get('win_rate', 0):.1%}")
    print(f"  Total return:       {m.get('total_return_pct', 0):.2f}%")
    print(f"  Sharpe ratio:       {m.get('sharpe_ratio', 0):.2f}")
    print(f"  Sortino ratio:      {m.get('sortino_ratio', 0):.2f}")
    print(f"  Max drawdown:       {m.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Profit factor:      {m.get('profit_factor') or 'N/A'}")
    print(f"  Expectancy (R):     {m.get('expectancy', 0):.3f}")
    print(f"  Avg win (R):        {m.get('avg_win_r', 0):.2f}")
    print(f"  Avg loss (R):       {m.get('avg_loss_r', 0):.2f}")
    print()
    print(f"  Prop firm status:   {pf.get('status', 'N/A')}")
    print(f"  Prop firm profit:   {pf.get('profit_pct', 0):.2f}%")
    print(f"  Max daily DD:       {pf.get('max_daily_dd_pct', 0):.2f}%")
    print(f"  Max total DD:       {pf.get('max_total_dd_pct', 0):.2f}%")
    print(f"  Signals generated:  {result.total_signals}")
    print(f"  Signals skipped:    {result.skipped_signals}")
    print(f"{'=' * 60}\n")


# =============================================================================
# Optimization
# =============================================================================

def run_optimization(
    data: pd.DataFrame,
    config: Dict[str, Any],
    initial_balance: float = 10_000.0,
    n_trials: int = 50,
) -> Dict[str, Any]:
    """Run Optuna single-objective optimization for SSS strategy."""
    from src.optimization.optuna_runner import OptunaOptimizer

    logger.info("Starting SSS optimization — %d trials ...", n_trials)
    t0 = time.monotonic()

    optimizer = OptunaOptimizer(
        data=data,
        config=config,
        initial_balance=initial_balance,
        strategy_key="sss",
    )

    study = optimizer.optimize_single(
        n_trials=n_trials,
        study_name="sss_optimization",
        n_jobs=1,
    )

    elapsed = time.monotonic() - t0
    logger.info("Optimization complete in %.1fs", elapsed)

    best = study.best_trial
    logger.info("Best trial #%d — score=%.4f", best.number, best.value)
    logger.info("Best params: %s", best.params)

    return {
        "best_score": best.value,
        "best_params": best.params,
        "best_trial_number": best.number,
        "n_trials": len(study.trials),
        "elapsed_seconds": elapsed,
        "study": study,
    }


# =============================================================================
# Report export
# =============================================================================

def export_report(
    result: Any,
    config: Dict[str, Any],
    run_id: str,
    output_dir: str = "reports",
    optimization_info: Optional[Dict] = None,
) -> Path:
    """Export backtest results as a JSON report."""
    from src.backtesting.results_exporter import ResultsExporter

    exporter = ResultsExporter(reports_dir=output_dir)

    # Load previous runs for delta comparison
    previous = exporter.list_runs() if hasattr(exporter, 'list_runs') else []

    path = exporter.export_run_report(
        result=result,
        config=config,
        run_id=run_id,
    )

    # Append optimization info to the report
    if optimization_info:
        try:
            with open(path) as f:
                report = json.load(f)
            report["optimization"] = {
                "best_score": optimization_info.get("best_score"),
                "best_params": optimization_info.get("best_params"),
                "best_trial_number": optimization_info.get("best_trial_number"),
                "n_trials": optimization_info.get("n_trials"),
                "elapsed_seconds": optimization_info.get("elapsed_seconds"),
            }
            with open(path, "w") as f:
                json.dump(report, f, indent=2, default=str)
        except Exception as exc:
            logger.warning("Could not append optimization info: %s", exc)

    logger.info("Report saved to %s", path)
    return path


# =============================================================================
# Dashboard
# =============================================================================

def launch_dashboard(
    result: Any,
    output_dir: str = "reports",
    initial_balance: float = 10_000.0,
    port: int = 8501,
) -> str:
    """Generate static HTML dashboard and launch optimization dashboard server."""
    # 1. Static HTML dashboard (self-contained, auto-opens)
    dash_path = None
    try:
        from src.backtesting.dashboard import BacktestDashboard
        dashboard = BacktestDashboard(title="XAU/USD SSS Strategy Backtest")
        dash_path = dashboard.save_and_open(
            result=result,
            output_dir=output_dir,
            initial_balance=initial_balance,
            learning_phase="enabled",
            learning_skipped=0,
            instrument="XAUUSD",
            auto_open=True,
        )
        logger.info("Static dashboard: %s", dash_path)
    except Exception as exc:
        logger.warning("Static dashboard generation failed: %s", exc)

    # 2. Optimization dashboard server (live, serves JSON reports)
    try:
        from src.backtesting.optimization_dashboard import OptimizationDashboardServer
        opt_server = OptimizationDashboardServer(
            port=port,
            reports_dir=output_dir,
            auto_open=True,
        )
        opt_server.start()
        logger.info("Optimization dashboard: http://localhost:%d", port)
        return f"http://localhost:{port}"
    except Exception as exc:
        logger.warning("Optimization dashboard failed: %s", exc)
        return dash_path or ""


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

    # 1. Load data
    data = load_data(args.data_file, max_bars=args.max_bars)

    # 2. Build SSS config
    config = build_sss_config()
    logger.info("SSS config: %s", config.get("strategies", {}).get("sss", {}))

    # Start live dashboard server (real-time equity, trades, charts)
    live_server = None
    try:
        from src.backtesting.live_dashboard import LiveDashboardServer
        live_server = LiveDashboardServer(
            port=args.dashboard_port, auto_open=True,
            app_config=config,
            config_dir=str(_PROJECT_ROOT / "config"),
        )
        live_server.start()
        logger.info("Live dashboard: http://localhost:%d", args.dashboard_port)
    except Exception as exc:
        logger.warning("Could not start live dashboard: %s", exc)

    # ── Phase 1: Baseline backtest ──
    print("\n" + "=" * 60)
    print("  PHASE 1: BASELINE SSS BACKTEST")
    print("=" * 60)

    baseline_result = run_backtest(
        data, config, args.initial_balance, label="baseline",
        live_dashboard=live_server,
    )
    print_results(baseline_result, label="BASELINE", initial_balance=args.initial_balance)

    # Export baseline report
    baseline_report = export_report(
        baseline_result, config, run_id=f"sss_baseline_{ts}", output_dir=args.output_dir
    )

    if args.skip_optimize:
        # Just show dashboard and exit
        print("\n  Skipping optimization (--skip-optimize)\n")
        dash_url = launch_dashboard(
            baseline_result, args.output_dir, args.initial_balance, args.dashboard_port
        )
        if dash_url:
            print(f"  Dashboard: {dash_url}")
            print("  Press Ctrl+C to stop.\n")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        return 0

    # ── Phase 2: Optuna optimization ──
    print("\n" + "=" * 60)
    print(f"  PHASE 2: SSS OPTIMIZATION ({args.n_trials} trials)")
    print("=" * 60 + "\n")

    opt_result = run_optimization(data, config, args.initial_balance, args.n_trials)

    print(f"\n  Best score:  {opt_result['best_score']:.4f}")
    print(f"  Best params: {json.dumps(opt_result['best_params'], indent=4)}")

    # ── Phase 3: Best-params backtest ──
    print("\n" + "=" * 60)
    print("  PHASE 3: OPTIMIZED SSS BACKTEST")
    print("=" * 60)

    # Merge best params into config
    optimized_config = dict(config)
    best_params = opt_result["best_params"]
    sss_overrides = {}
    for k, v in best_params.items():
        # Strip sss_ prefix from param names
        clean_key = k.replace("sss_", "")
        sss_overrides[clean_key] = v

    optimized_config["strategies"] = dict(config.get("strategies", {}))
    optimized_config["strategies"]["sss"] = {
        **config.get("strategies", {}).get("sss", {}),
        **sss_overrides,
    }

    optimized_result = run_backtest(
        data, optimized_config, args.initial_balance, label="optimized",
        live_dashboard=live_server,
    )
    print_results(optimized_result, label="OPTIMIZED", initial_balance=args.initial_balance)

    # Export optimized report
    opt_report = export_report(
        optimized_result, optimized_config,
        run_id=f"sss_optimized_{ts}",
        output_dir=args.output_dir,
        optimization_info=opt_result,
    )

    # ── Phase 4: Comparison ──
    bm = baseline_result.metrics
    om = optimized_result.metrics
    print("\n" + "=" * 60)
    print("  BASELINE vs OPTIMIZED COMPARISON")
    print("=" * 60)
    for key in ("total_trades", "win_rate", "total_return_pct", "sharpe_ratio",
                "max_drawdown_pct", "profit_factor", "expectancy"):
        bv = bm.get(key, 0) or 0
        ov = om.get(key, 0) or 0
        if isinstance(bv, float):
            delta = ov - bv
            arrow = "+" if delta > 0 else ""
            print(f"  {key:20s}  {bv:>10.2f}  ->  {ov:>10.2f}  ({arrow}{delta:.2f})")
        else:
            print(f"  {key:20s}  {bv:>10}  ->  {ov:>10}")
    print("=" * 60)

    # ── Phase 5: Dashboard ──
    print("\n  Launching dashboard ...\n")
    dash_url = launch_dashboard(
        optimized_result, args.output_dir, args.initial_balance, args.dashboard_port
    )
    if dash_url:
        print(f"  Dashboard: {dash_url}")
        print("  Press Ctrl+C to stop.\n")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
