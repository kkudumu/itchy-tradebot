"""CLI entry point for the adaptive strategy optimizer."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Backtest screenshots are rendered from worker threads; force a headless
# backend before any matplotlib import chain is triggered.
os.environ.setdefault("MPLBACKEND", "Agg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

# Suppress noisy sub-loggers
for name in ["src.backtesting.vectorbt_engine", "src.strategy", "src.risk",
             "src.edges", "src.learning", "src.monitoring",
             "src.backtesting.strategy_telemetry"]:
    logging.getLogger(name).setLevel(logging.WARNING)

logger = logging.getLogger("adaptive_optimizer")


def _load_env():
    env_path = _PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def main():
    parser = argparse.ArgumentParser(description="Adaptive Strategy Optimizer")
    parser.add_argument("--instrument", type=str, default=None,
                       help="Optimize a single instrument (e.g., MGC, MCL)")
    parser.add_argument("--once", action="store_true",
                       help="Run one epoch only (no looping)")
    parser.add_argument("--status", action="store_true",
                       help="Show status of all instruments")
    parser.add_argument("--trials", type=int, default=50,
                       help="Optuna trials per instrument per epoch")
    args = parser.parse_args()

    _load_env()

    from src.database.connection import DatabasePool
    pool = DatabasePool()
    pool.initialise()

    from src.optimization.adaptive_runner import AdaptiveRunner
    runner = AdaptiveRunner(db_pool=pool, trials_per_epoch=args.trials)

    try:
        if args.status:
            status = runner.status()
            print(json.dumps(status, indent=2, default=str))
            return 0

        if args.once:
            results = runner.run_once(instrument_filter=args.instrument)
            for symbol, result in results.items():
                logger.info("%s: %s", symbol, json.dumps(result, default=str))
            return 0

        logger.info("Starting continuous adaptive optimization loop...")
        runner.run_forever()
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    finally:
        pool.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
