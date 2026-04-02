"""
Creative Pattern Discovery Agent -- rolling window orchestrator CLI.

Runs the full discovery loop: slice data into 22-trading-day windows,
run backtest per window, invoke SHAP/pattern/regime/codegen phases,
validate edges via walk-forward, and report findings.

Usage
-----
    python scripts/run_discovery_loop.py \\
        --data-file data/xauusd_1m.parquet \\
        --max-windows 12 \\
        --strategy sss

    python scripts/run_discovery_loop.py \\
        --data-file data.parquet \\
        --enable-claude \\
        --window-size 30
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

# Make project root importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_discovery_loop")

_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "discovery.yaml"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Creative Pattern Discovery Agent -- rolling window loop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--data-file",
        type=str,
        required=True,
        help="Path to 1-minute OHLCV Parquet or CSV file.",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to discovery.yaml config (default: config/discovery.yaml).",
    )
    p.add_argument(
        "--max-windows",
        type=int,
        default=12,
        help="Maximum number of rolling windows to process.",
    )
    p.add_argument(
        "--window-size",
        type=int,
        default=22,
        help="Trading days per window.",
    )
    p.add_argument(
        "--strategy",
        type=str,
        default="sss",
        help="Target strategy name.",
    )
    p.add_argument(
        "--enable-claude",
        action="store_true",
        default=False,
        help="Enable Claude CLI for hypothesis generation.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="reports/discovery",
        help="Directory for discovery reports.",
    )

    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config) if args.config else _DEFAULT_CONFIG_PATH
    config = {}
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        logger.info("Loaded config from %s", config_path)

    # CLI overrides
    config.setdefault("orchestrator", {})
    config["orchestrator"]["window_size_trading_days"] = args.window_size
    config["orchestrator"]["max_windows"] = args.max_windows
    config["orchestrator"]["strategy_name"] = args.strategy
    config.setdefault("reporting", {})
    config["reporting"]["reports_dir"] = args.output_dir

    # Load data
    data_path = Path(args.data_file)
    if not data_path.exists():
        logger.error("Data file not found: %s", data_path)
        return 1

    import pandas as pd

    logger.info("Loading data from %s", data_path)
    if data_path.suffix.lower() in (".parquet", ".pq"):
        candles = pd.read_parquet(data_path)
    else:
        candles = pd.read_csv(data_path, parse_dates=["time"])
        if "time" in candles.columns:
            candles = candles.set_index("time")

    if not isinstance(candles.index, pd.DatetimeIndex):
        logger.error("Data must have a DatetimeIndex")
        return 1
    if candles.index.tz is None:
        candles.index = candles.index.tz_localize("UTC")

    candles.columns = candles.columns.str.lower()
    logger.info("Loaded %d bars from %s to %s", len(candles), candles.index[0], candles.index[-1])

    # Load base strategy config
    strategy_yaml = _PROJECT_ROOT / "config" / "strategy.yaml"
    base_config = {}
    if strategy_yaml.exists():
        base_config = yaml.safe_load(strategy_yaml.read_text(encoding="utf-8")) or {}

    edges_yaml = _PROJECT_ROOT / "config" / "edges.yaml"
    if edges_yaml.exists():
        edges = yaml.safe_load(edges_yaml.read_text(encoding="utf-8")) or {}
        base_config["edges"] = edges

    # Run discovery loop
    from src.discovery.orchestrator import DiscoveryOrchestrator

    orch = DiscoveryOrchestrator(
        config=config,
        knowledge_dir=str(_PROJECT_ROOT / "reports" / "agent_knowledge"),
        edges_yaml_path=str(edges_yaml),
        data_file=str(data_path),
    )

    logger.info(
        "Starting discovery loop: strategy=%s, windows=%d, window_size=%d",
        args.strategy, args.max_windows, args.window_size,
    )

    summary = orch.run(
        candles=candles,
        base_config=base_config,
        enable_claude=args.enable_claude,
    )

    # Print results
    print("\n" + "=" * 60)
    print("  DISCOVERY LOOP COMPLETE")
    print("=" * 60)
    sr = summary.get("summary_report", {})
    print(f"  Windows processed:    {summary['windows_processed']}")
    print(f"  Phase 1 pass rate:    {sr.get('phase_1_pass_rate', 0):.1%}")
    print(f"  Phase 2 pass rate:    {sr.get('phase_2_pass_rate', 0):.1%}")
    print(f"  Edges discovered:     {sr.get('edges_discovered', 0)}")
    print(f"  Edges validated:      {sr.get('edges_validated', 0)}")
    print(f"  Edges absorbed:       {sr.get('edges_absorbed', 0)}")
    print(f"  Config changes:       {sr.get('total_config_changes', 0)}")
    print(f"  Pending edges:        {summary.get('pending_edges', 0)}")
    print(f"  Reports dir:          {args.output_dir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
