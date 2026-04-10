"""
Demo challenge CLI orchestrator for the XAU/USD Ichimoku trading agent.

Orchestrates the full pipeline:
    load config -> prepare data -> run backtest -> optimize -> validate -> report

Usage examples
--------------
# Run a backtest on a local data file:
    python scripts/run_demo_challenge.py --mode backtest --data-file xauusd_1m.parquet

# Run a backtest with custom config:
    python scripts/run_demo_challenge.py --mode backtest --data-file data.parquet \\
        --config config/my_config.yaml

# Run validation pipeline (walk-forward + Monte Carlo + go/no-go):
    python scripts/run_demo_challenge.py --mode validate --data-file data.parquet \\
        --wf-trials 100 --mc-sims 5000

# Live mode (requires MT5 on Windows, mocked in backtest):
    python scripts/run_demo_challenge.py --mode live \\
        --mt5-login 12345 --mt5-password mypassword --mt5-server The5ers-Demo

Modes
-----
backtest  : Load data, run full backtest, compute metrics, generate report.
validate  : Backtest + walk-forward analysis + Monte Carlo + go/no-go verdict.
live      : Connect to MT5, run the signal engine on live data. (Windows only)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Make project root importable when running from scripts/ directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_demo_challenge")


# =============================================================================
# CLI argument definition
# =============================================================================


def _load_local_env() -> None:
    env_path = _PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="XAU/USD Ichimoku demo challenge orchestrator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Operating mode
    p.add_argument(
        "--mode",
        choices=["backtest", "validate", "live"],
        default="backtest",
        help="Pipeline mode: backtest, validate, or live.",
    )

    # Config file
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the configuration directory (default: project config/).",
    )

    # Data source (required for backtest and validate modes)
    data_group = p.add_mutually_exclusive_group()
    data_group.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to a 1-minute OHLCV CSV or Parquet file.",
    )
    data_group.add_argument(
        "--synthetic-data",
        action="store_true",
        default=False,
        help="Use synthetic data for quick testing (backtest mode only).",
    )
    p.add_argument(
        "--data-source",
        choices=["file", "synthetic", "projectx"],
        default="file",
        help="Data source for backtest/validate modes.",
    )

    # Backtest options
    p.add_argument(
        "--initial-balance",
        type=float,
        default=10_000.0,
        help="Starting account balance for backtest/optimize.",
    )
    p.add_argument(
        "--instrument",
        type=str,
        default="XAUUSD",
        help="Instrument symbol.",
    )

    # Validation options
    p.add_argument(
        "--wf-trials",
        type=int,
        default=100,
        help="Optuna trials per walk-forward window (validate mode).",
    )
    p.add_argument(
        "--mc-sims",
        type=int,
        default=5000,
        help="Monte Carlo simulations (validate mode).",
    )
    p.add_argument(
        "--haircut",
        type=float,
        default=25.0,
        help="Percentage haircut applied to OOS metrics.",
    )

    # Live mode options
    p.add_argument(
        "--mt5-login",
        type=int,
        default=None,
        help="MT5 account login (live mode, Windows only).",
    )
    p.add_argument(
        "--mt5-password",
        type=str,
        default=None,
        help="MT5 account password (live mode).",
    )
    p.add_argument(
        "--mt5-server",
        type=str,
        default=None,
        help="MT5 broker server name (live mode).",
    )
    p.add_argument(
        "--provider",
        choices=["projectx", "mt5"],
        default=None,
        help="Execution provider for live mode. Defaults to config/provider.yaml.",
    )
    p.add_argument(
        "--projectx-username",
        type=str,
        default=None,
        help="ProjectX username override. Falls back to env/config.",
    )
    p.add_argument(
        "--projectx-api-key",
        type=str,
        default=None,
        help="ProjectX API key override. Falls back to env/config.",
    )
    p.add_argument(
        "--projectx-account-id",
        type=int,
        default=None,
        help="ProjectX account id override for live trading.",
    )
    p.add_argument(
        "--projectx-contract-id",
        type=str,
        default=None,
        help="ProjectX contract id override for live/historical use.",
    )
    p.add_argument(
        "--projectx-symbol-id",
        type=str,
        default=None,
        help="ProjectX symbol id override for live/historical use.",
    )
    p.add_argument(
        "--projectx-start",
        type=str,
        default=None,
        help="UTC ISO timestamp for ProjectX historical start, e.g. 2025-01-01T00:00:00Z.",
    )
    p.add_argument(
        "--projectx-end",
        type=str,
        default=None,
        help="UTC ISO timestamp for ProjectX historical end, e.g. 2025-03-01T00:00:00Z.",
    )

    # Output
    p.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory for output reports and logs.",
    )
    p.add_argument(
        "--log-trades",
        action="store_true",
        default=False,
        help="Persist completed trades to database (requires DB connection).",
    )
    p.add_argument(
        "--persist-trades",
        action="store_true",
        default=False,
        help="Persist backtest trades to pgvector database.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )

    return p


# =============================================================================
# Data loading helpers
# =============================================================================


def _load_data_file(path: str) -> pd.DataFrame:
    """Load 1-minute OHLCV data from CSV or Parquet."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    logger.info("Loading data from %s", path)
    if p.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif p.suffix.lower() in (".csv", ".txt"):
        df = pd.read_csv(path, parse_dates=["time"])
        if "time" in df.columns:
            df = df.set_index("time")
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")

    # Ensure UTC DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    # Normalise column names to lowercase
    df.columns = df.columns.str.lower()

    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info("Loaded %d bars from %s to %s", len(df), df.index[0], df.index[-1])
    return df


def _generate_synthetic_data(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic 1-minute OHLCV data for quick testing."""
    rng = np.random.default_rng(seed)
    start = datetime(2020, 1, 2, 8, 0, tzinfo=timezone.utc)
    from datetime import timedelta

    prices = [1800.0]
    for _ in range(n - 1):
        prices.append(prices[-1] + rng.normal(0.02, 1.5))

    prices = np.array(prices)
    timestamps = [start + timedelta(minutes=i) for i in range(n)]

    rows = []
    for p in prices:
        noise = rng.uniform(0, 0.5, 4)
        o = p + noise[0] - 0.25
        h = max(o, p) + noise[1]
        l = min(o, p) - noise[2]
        c = p + noise[3] - 0.25
        rows.append({"open": o, "high": h, "low": l, "close": c, "volume": int(rng.integers(100, 800))})

    df = pd.DataFrame(rows, index=pd.DatetimeIndex(timestamps, tz=timezone.utc))
    logger.info("Generated %d synthetic 1M bars", len(df))
    return df


def _parse_utc_timestamp(value: str) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _load_projectx_data(args: argparse.Namespace, app_config) -> pd.DataFrame:
    """Load historical bars from ProjectX for backtest/validate modes."""
    if not args.projectx_start or not args.projectx_end:
        raise ValueError("ProjectX historical loading requires --projectx-start and --projectx-end")

    from src.providers import ProjectXHistoricalDataLoader, build_projectx_stack

    projectx_cfg = app_config.provider.projectx.model_copy(deep=True)
    if args.projectx_contract_id:
        projectx_cfg.default_contract_id = args.projectx_contract_id
    if args.projectx_symbol_id:
        projectx_cfg.default_symbol_id = args.projectx_symbol_id

    _, market_provider, _, _ = build_projectx_stack(
        config=projectx_cfg,
        instruments=app_config.instruments,
        username=args.projectx_username,
        api_key=args.projectx_api_key,
    )
    loader = ProjectXHistoricalDataLoader(market_provider)
    start_dt = _parse_utc_timestamp(args.projectx_start)
    end_dt = _parse_utc_timestamp(args.projectx_end)
    logger.info("Loading ProjectX data for %s from %s to %s", args.instrument, start_dt, end_dt)
    return loader.load_range(
        instrument=args.instrument,
        timeframe="1M",
        start_time=start_dt,
        end_time=end_dt,
    )


# =============================================================================
# Pipeline: BACKTEST mode
# =============================================================================


def run_backtest(args: argparse.Namespace) -> int:
    """Execute the full backtest pipeline and print results.

    Returns 0 on success, 1 on failure.
    """
    logger.info("=== BACKTEST MODE ===")

    # Load config
    try:
        from src.config.loader import ConfigLoader
        config_dir = args.config or None
        loader = ConfigLoader(config_dir=config_dir)
        app_config = loader.load()
        logger.info("Config loaded from %s", loader.config_dir)
    except Exception as exc:
        logger.error("Config loading failed: %s", exc)
        return 1

    # Load data
    try:
        data_source = args.data_source
        if args.synthetic_data:
            data_source = "synthetic"
        elif args.data_file and args.data_source == "file":
            data_source = "file"

        if data_source == "synthetic":
            candles = _generate_synthetic_data(seed=args.seed or 42)
        elif data_source == "projectx":
            candles = _load_projectx_data(args, app_config)
        elif args.data_file:
            candles = _load_data_file(args.data_file)
        else:
            logger.error("Backtest mode requires --data-file, --synthetic-data, or --data-source projectx")
            return 1
    except Exception as exc:
        logger.error("Data loading failed: %s", exc)
        return 1

    # Start live dashboard server
    live_server = None
    try:
        from src.backtesting.live_dashboard import LiveDashboardServer
        live_server = LiveDashboardServer(
            port=8501, auto_open=True,
            app_config=app_config.model_dump() if hasattr(app_config, 'model_dump') else {},
            config_dir=str(loader.config_dir),
        )
        live_server.start()
        logger.info("Live dashboard: %s", live_server.url)
    except Exception as exc:
        logger.warning("Could not start live dashboard: %s", exc)

    # Run backtest
    logger.info("Running backtest on %d bars ...", len(candles))
    try:
        from src.backtesting.vectorbt_engine import IchimokuBacktester

        # Build full config dict for the backtester from raw YAML.
        # The Pydantic model doesn't capture active_strategies or prop_firm,
        # so we load the raw strategy YAML for those fields.
        import yaml as _yaml
        _raw_strat = {}
        _strat_yaml = Path(loader.config_dir) / "strategy.yaml"
        if _strat_yaml.exists():
            with _strat_yaml.open() as _f:
                _raw_strat = _yaml.safe_load(_f) or {}

        bt_config = {}
        if hasattr(app_config, "edges"):
            bt_config["edges"] = app_config.edges.model_dump()
        bt_config["active_strategies"] = _raw_strat.get(
            "active_strategies", [_raw_strat.get("active_strategy", "ichimoku")]
        )
        bt_config["strategies"] = _raw_strat.get("strategies", {})
        for key in ("risk", "exit", "prop_firm"):
            if key in _raw_strat:
                bt_config[key] = _raw_strat[key]

        prop = bt_config.get("prop_firm", {})
        p1 = prop.get("phase_1", {})
        bt_config["prop_firm"] = prop

        backtester = IchimokuBacktester(
            config=bt_config,
            initial_balance=args.initial_balance,
            prop_firm_profit_target_pct=float(p1.get("profit_target_pct", 8.0)),
            prop_firm_max_daily_dd_pct=float(p1.get("daily_loss_pct", 5.0)),
            prop_firm_max_total_dd_pct=float(p1.get("max_loss_pct", 10.0)),
            prop_firm_time_limit_days=int(p1.get("time_limit_days", 30)),
        )
        # Wire edge manager to dashboard for runtime config access
        if live_server is not None:
            live_server._edge_manager = backtester.edge_manager

        result = backtester.run(
            candles_1m=candles,
            instrument=args.instrument,
            log_trades=args.log_trades,
            enable_learning=True,
            live_dashboard=live_server,
        )
    except Exception as exc:
        logger.exception("Backtest failed: %s", exc)
        return 1

    # Display results
    m = result.metrics
    pf = result.prop_firm
    learning_phase = backtester.learning_engine.get_phase()

    print("\n" + "=" * 60)
    print(f"  BACKTEST RESULTS — {args.instrument}")
    print("=" * 60)
    print(f"  Total trades:       {m.get('total_trades', 0)}")
    print(f"  Win rate:           {m.get('win_rate', 0):.1%}")
    print(f"  Total return:       {m.get('total_return_pct', 0):.2f}%")
    print(f"  Sharpe ratio:       {m.get('sharpe_ratio', 0):.2f}")
    print(f"  Max drawdown:       {m.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Profit factor:      {m.get('profit_factor') or 'N/A'}")
    print(f"  Expectancy (R):     {m.get('expectancy', 0):.3f}")
    print(f"  Skipped signals:    {result.skipped_signals}")
    print(f"  Learning phase:     {learning_phase}")
    print()
    print(f"  Prop firm status:   {pf.get('status', 'N/A')}")
    print(f"  Prop firm profit:   {pf.get('profit_pct', 0):.2f}%")
    print(f"  Max daily DD:       {pf.get('max_daily_dd_pct', 0):.2f}%")
    print(f"  Max total DD:       {pf.get('max_total_dd_pct', 0):.2f}%")
    print("=" * 60)

    # Save equity curve to output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    equity_path = output_dir / f"equity_curve_{ts}.csv"
    result.equity_curve.to_csv(equity_path)
    logger.info("Equity curve saved to %s", equity_path)

    # Persist trades to pgvector if requested
    if args.persist_trades:
        try:
            from src.database.connection import DatabasePool
            from src.backtesting.trade_persistence import TradePersistence
            from src.learning.embeddings import EmbeddingEngine

            db_pool = DatabasePool()
            db_pool.initialise()
            persistence = TradePersistence(db_pool=db_pool, embedding_engine=EmbeddingEngine())
            run_id = f"demo_{ts}"
            config_dict = app_config.model_dump() if hasattr(app_config, "model_dump") else {}
            n_persisted = persistence.persist_run(
                run_id=run_id,
                trades=result.trades,
                config_snapshot=config_dict,
                metrics=result.metrics,
            )
            logger.info("Persisted %d trades with run_id=%s", n_persisted, run_id)
        except Exception as exc:
            logger.warning("Trade persistence failed (continuing): %s", exc)

    # Generate static dashboard report (saved to disk, opens only if no live dashboard)
    try:
        from src.backtesting.dashboard import BacktestDashboard
        dashboard = BacktestDashboard()
        dash_path = dashboard.save_and_open(
            result=result,
            output_dir=args.output_dir,
            initial_balance=args.initial_balance,
            learning_phase=learning_phase,
            learning_skipped=0,
            instrument=args.instrument,
            auto_open=(live_server is None),  # only auto-open if no live dashboard
        )
        print(f"\n  Dashboard report: {dash_path}")
    except Exception as exc:
        logger.warning("Dashboard generation failed (matplotlib may not be installed): %s", exc)

    if live_server is not None:
        print(f"  Live dashboard still running at {live_server.url}")
        print("  Press Ctrl+C to stop the server.")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            live_server.stop()

    return 0


# =============================================================================
# Pipeline: VALIDATE mode
# =============================================================================


def run_validate(args: argparse.Namespace) -> int:
    """Execute the full validation pipeline.

    Steps: walk-forward -> Monte Carlo -> go/no-go -> report.
    Returns 0 on success, 1 on failure.
    """
    logger.info("=== VALIDATE MODE ===")

    try:
        from src.config.loader import ConfigLoader

        config_dir = args.config or None
        app_config = ConfigLoader(config_dir=config_dir).load()
    except Exception as exc:
        logger.error("Config loading failed: %s", exc)
        return 1

    # Load data
    try:
        data_source = args.data_source
        if args.synthetic_data:
            data_source = "synthetic"

        if data_source == "synthetic":
            candles = _generate_synthetic_data(seed=args.seed or 42)
        elif data_source == "projectx":
            candles = _load_projectx_data(args, app_config)
        elif args.data_file:
            candles = _load_data_file(args.data_file)
        else:
            logger.error("Validate mode requires --data-file, --synthetic-data, or --data-source projectx")
            return 1
    except Exception as exc:
        logger.error("Data loading failed: %s", exc)
        return 1

    # Run validation
    logger.info(
        "Running validation pipeline (wf_trials=%d, mc_sims=%d) ...",
        args.wf_trials,
        args.mc_sims,
    )
    try:
        from src.validation.go_nogo import GoNoGoValidator

        validator = GoNoGoValidator(
            data=candles,
            initial_balance=args.initial_balance,
            haircut_pct=args.haircut,
        )
        validation_result = validator.run_full_validation(
            n_wf_trials=args.wf_trials,
            n_mc_sims=args.mc_sims,
            seed=args.seed,
        )
    except Exception as exc:
        logger.exception("Validation failed: %s", exc)
        return 1

    # Print verdict
    vr = validation_result.validation_result
    mc = validation_result.monte_carlo

    print("\n" + "=" * 60)
    print("  PRE-CHALLENGE VALIDATION REPORT")
    print("=" * 60)
    print(f"  Final verdict:      {validation_result.final_verdict}")
    print(f"  OOS trades:         {validation_result.n_oos_trades}")
    print(f"  WFE:                {validation_result.overfit_report.wfe:.2f}")
    print(f"  DSR:                {validation_result.overfit_report.dsr:.4f}")
    print(f"  MC pass rate:       {mc.pass_rate:.1f}%")
    print(f"  MC avg days:        {mc.avg_days:.1f}")
    print()
    print("  Recommendations:")
    for rec in validation_result.recommendations:
        print(f"    - {rec}")
    print("=" * 60)

    # Persist report
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"validation_{ts}.txt"
    with open(report_path, "w") as fh:
        fh.write(f"Validation report: {ts}\n")
        fh.write(f"Verdict: {validation_result.final_verdict}\n")
        for rec in validation_result.recommendations:
            fh.write(f"- {rec}\n")
    logger.info("Report saved to %s", report_path)

    # Return non-zero exit code for NO-GO verdict (useful in CI pipelines)
    return 0 if validation_result.final_verdict in ("GO", "BORDERLINE") else 1


# =============================================================================
# Pipeline: LIVE mode
# =============================================================================


def run_live(args: argparse.Namespace) -> int:
    """Connect to the configured provider and run the live signal engine."""
    logger.info("=== LIVE MODE ===")

    # Build components
    try:
        from src.config.loader import ConfigLoader
        config_dir = args.config or None
        app_config = ConfigLoader(config_dir=config_dir).load()
    except Exception as exc:
        logger.error("Config loading failed: %s", exc)
        return 1

    provider_name = args.provider or app_config.provider.provider or "projectx"
    market_data_provider = None
    execution_provider = None
    account_provider = None
    point_value = 1.0
    bridge = None

    # Connect provider
    try:
        if provider_name == "mt5":
            if not args.mt5_login or not args.mt5_password or not args.mt5_server:
                logger.error(
                    "MT5 live mode requires --mt5-login, --mt5-password, and --mt5-server"
                )
                return 1

            from src.execution.mt5_bridge import MT5Bridge
            from src.execution.order_manager import OrderManager
            from src.execution.account_monitor import AccountMonitor
            from src.providers import MT5AccountProvider, MT5ExecutionProvider, MT5MarketDataProvider

            bridge = MT5Bridge(
                login=args.mt5_login,
                password=args.mt5_password,
                server=args.mt5_server,
            )
            if not bridge.connect():
                logger.error("MT5 connection failed")
                return 1

            market_data_provider = MT5MarketDataProvider(bridge)
            execution_provider = MT5ExecutionProvider(OrderManager(bridge=bridge))
            account_provider = MT5AccountProvider(AccountMonitor(bridge=bridge))
            spec = market_data_provider.get_contract_spec(args.instrument)
            point_value = float(spec.point_value or point_value)
            logger.info("MT5 connected to %s", args.mt5_server)
        else:
            from src.providers import build_projectx_stack

            projectx_cfg = app_config.provider.projectx.model_copy(deep=True)
            if args.projectx_account_id is not None:
                projectx_cfg.account_id = args.projectx_account_id
            if args.projectx_contract_id:
                projectx_cfg.default_contract_id = args.projectx_contract_id
            if args.projectx_symbol_id:
                projectx_cfg.default_symbol_id = args.projectx_symbol_id
            projectx_cfg.live = True

            _, market_data_provider, execution_provider, account_provider = build_projectx_stack(
                config=projectx_cfg,
                instruments=app_config.instruments,
                username=args.projectx_username,
                api_key=args.projectx_api_key,
            )
            spec = market_data_provider.get_contract_spec(args.instrument)
            point_value = float(spec.point_value or point_value)
            logger.info("ProjectX live provider ready for %s (%s)", args.instrument, spec.contract_id)
    except Exception as exc:
        logger.exception("Provider connection error: %s", exc)
        return 1

    # Build engine
    try:
        from src.strategy.signal_engine import SignalEngine
        from src.edges.manager import EdgeManager
        from src.risk.position_sizer import AdaptivePositionSizer
        from src.risk.circuit_breaker import DailyCircuitBreaker
        from src.risk.exit_manager import HybridExitManager
        from src.risk.trade_manager import TradeManager
        from src.learning.embeddings import EmbeddingEngine
        from src.learning.similarity import SimilaritySearch
        from src.zones.manager import ZoneManager
        from src.engine.decision_engine import DecisionEngine

        signal_engine = SignalEngine(instrument=args.instrument)
        edge_manager = EdgeManager(app_config.edges)
        sizer = AdaptivePositionSizer(initial_balance=args.initial_balance)
        breaker = DailyCircuitBreaker(max_daily_loss_pct=2.0)
        exit_mgr = HybridExitManager()
        trade_mgr = TradeManager(
            position_sizer=sizer,
            circuit_breaker=breaker,
            exit_manager=exit_mgr,
        )
        embedding_engine = EmbeddingEngine()
        similarity = SimilaritySearch(db_pool=None)
        zone_manager = ZoneManager()

        engine = DecisionEngine(
            config={"instrument": args.instrument, "point_value": point_value},
            signal_engine=signal_engine,
            edge_manager=edge_manager,
            trade_manager=trade_mgr,
            similarity_search=similarity,
            embedding_engine=embedding_engine,
            zone_manager=zone_manager,
            market_data_provider=market_data_provider,
            execution_provider=execution_provider,
            account_provider=account_provider,
            mt5_bridge=bridge,
        )

        logger.info("Live engine initialised — starting scan loop")
        engine.start()

    except KeyboardInterrupt:
        logger.info("Shutdown requested by user (Ctrl+C)")
    except Exception as exc:
        logger.exception("Live engine error: %s", exc)
        return 1
    finally:
        if bridge is not None:
            try:
                bridge.disconnect()
            except Exception:
                pass

    return 0


# =============================================================================
# Entry point
# =============================================================================


def main() -> int:
    _load_local_env()
    parser = _build_parser()
    args = parser.parse_args()

    logger.info(
        "Starting demo challenge — mode=%s instrument=%s balance=%.0f",
        args.mode,
        args.instrument,
        args.initial_balance,
    )

    if args.mode == "backtest":
        return run_backtest(args)
    elif args.mode == "validate":
        return run_validate(args)
    elif args.mode == "live":
        return run_live(args)
    else:
        logger.error("Unknown mode: %s", args.mode)
        return 1


if __name__ == "__main__":
    sys.exit(main())
