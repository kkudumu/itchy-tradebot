#!/usr/bin/env python3
"""
Historical data download and load pipeline.

Downloads XAU/USD (or any configured instrument) tick data from Dukascopy,
aggregates it to 1-minute OHLCV bars, validates the data, and bulk-inserts
it into TimescaleDB.

Usage
-----
    python scripts/download_history.py \\
        --instrument XAUUSD \\
        --start 2019-01-01 \\
        --end   2024-01-01

    # Dry-run: download and validate but do not insert into the database
    python scripts/download_history.py --instrument XAUUSD --start 2023-01-01 --end 2023-02-01 --dry-run

    # Verify continuous aggregates for a sample date after loading
    python scripts/download_history.py --verify-date 2023-06-15 --instrument XAUUSD

Environment variables
---------------------
DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
    Override the defaults from config/database.yaml.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the project root is on the import path so that ``src`` is importable
# regardless of how this script is invoked.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.downloader import DukascopyDownloader
from src.data.loader import TimescaleLoader
from src.data.normalizer import DataNormalizer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("download_history")


# ---------------------------------------------------------------------------
# Database configuration
# ---------------------------------------------------------------------------

def _load_db_config() -> dict:
    """
    Build database connection kwargs.

    Preference order (highest to lowest):
      1. DB_* environment variables
      2. config/database.yaml defaults
    """
    import yaml

    config_path = _PROJECT_ROOT / "config" / "database.yaml"
    if config_path.exists():
        with open(config_path) as fh:
            yaml_cfg: dict = yaml.safe_load(fh) or {}
    else:
        yaml_cfg = {}

    return {
        "host":     os.getenv("DB_HOST",     yaml_cfg.get("host",     "localhost")),
        "port":     int(os.getenv("DB_PORT", str(yaml_cfg.get("port", 5432)))),
        "dbname":   os.getenv("DB_NAME",     yaml_cfg.get("name",     "trading")),
        "user":     os.getenv("DB_USER",     yaml_cfg.get("user",     "trader")),
        "password": os.getenv("DB_PASSWORD", yaml_cfg.get("password", "")),
    }


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

def _make_progress_callback(total: int):
    """Return a callback that logs download progress every 5 % or 100 hours."""

    def callback(done: int, total_: int) -> None:
        pct = done * 100 // total_
        if done % 100 == 0 or done == total_:
            logger.info("  Downloaded %d / %d hours  (%d%%)", done, total_, pct)

    return callback


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    instrument: str,
    start: datetime,
    end: datetime,
    dry_run: bool,
    db_config: dict,
    point_divisor: float,
    batch_size: int,
    rate_limit_secs: float,
) -> int:
    """
    Execute the full download → normalise → validate → load pipeline.

    Returns
    -------
    int
        Exit code (0 = success, 1 = validation errors, 2 = fatal error).
    """
    downloader = DukascopyDownloader(
        instrument=instrument,
        point_divisor=point_divisor,
        rate_limit_secs=rate_limit_secs,
    )
    normalizer = DataNormalizer()

    # --- Download ticks ---
    logger.info("=== Phase 1: Download ticks %s → %s ===", start.date(), end.date())
    ticks = downloader.download_range(
        start_date=start,
        end_date=end,
        progress_callback=_make_progress_callback(0),
    )

    if ticks.empty:
        logger.error("No tick data downloaded — aborting.")
        return 2

    logger.info("Downloaded %d raw ticks.", len(ticks))

    # --- Aggregate to 1-minute bars ---
    logger.info("=== Phase 2: Aggregate to 1-minute OHLCV ===")
    bars = normalizer.ticks_to_1m_ohlcv(ticks)
    bars = normalizer.normalize_timestamps(bars)
    logger.info("Aggregated to %d 1-minute bars.", len(bars))

    # --- Detect and log gaps ---
    logger.info("=== Phase 3: Gap detection ===")
    gaps = normalizer.detect_gaps(bars)
    if gaps.empty:
        logger.info("No gaps found.")
    else:
        logger.info("Found %d gap(s):", len(gaps))
        for _, g in gaps.iterrows():
            logger.info(
                "  [%s] %s → %s  (%d missing bars)",
                g["reason"],
                g["gap_start"],
                g["gap_end"],
                g["missing_bars"],
            )

    # --- Validate ---
    logger.info("=== Phase 4: Validation ===")
    errors = normalizer.validate(bars)
    if errors:
        for err in errors:
            logger.warning("Validation: %s", err)
        logger.warning("%d validation issue(s) found.", len(errors))
    else:
        logger.info("Validation passed — no issues found.")

    if dry_run:
        logger.info("Dry-run mode: skipping database insert.")
        logger.info("Would insert %d rows for %s.", len(bars), instrument)
        return 0

    # --- Load into TimescaleDB ---
    logger.info("=== Phase 5: Bulk insert into TimescaleDB ===")
    loader = TimescaleLoader(db_config)
    try:
        inserted = loader.bulk_insert(bars, instrument=instrument, batch_size=batch_size)
        logger.info("Inserted %d rows.", inserted)
    except Exception as exc:
        logger.error("Database insert failed: %s", exc)
        return 2

    logger.info("=== Pipeline complete ===")
    logger.info("Rows in DB for %s: %d", instrument, loader.get_row_count(instrument))
    lo, hi = loader.get_data_range(instrument)
    logger.info("Data range: %s → %s", lo, hi)

    return 0 if not errors else 1


def run_verify(instrument: str, sample_date: datetime, db_config: dict) -> int:
    """Spot-check continuous aggregates for *sample_date*."""
    logger.info("Verifying continuous aggregates for %s on %s", instrument, sample_date.date())
    loader = TimescaleLoader(db_config)
    try:
        results = loader.verify_aggregates(instrument, sample_date)
    except Exception as exc:
        logger.error("Verification failed: %s", exc)
        return 2

    all_ok = True
    for view, info in results.items():
        status = info["status"]
        if status == "ok":
            logger.info("  %s: OK (%d bars)", view, info["db_bars"])
        elif status == "no_data":
            logger.warning("  %s: no data", view)
        else:
            all_ok = False
            logger.error("  %s: MISMATCH", view)
            for d in info["discrepancies"]:
                logger.error("    %s", d)

    return 0 if all_ok else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Dukascopy tick data and load into TimescaleDB."
    )
    parser.add_argument(
        "--instrument",
        default="XAUUSD",
        help="Dukascopy instrument symbol (default: XAUUSD)",
    )
    parser.add_argument(
        "--start",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc),
        help="Start date inclusive, format YYYY-MM-DD",
    )
    parser.add_argument(
        "--end",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc),
        help="End date exclusive, format YYYY-MM-DD",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download and validate but do not write to the database",
    )
    parser.add_argument(
        "--verify-date",
        metavar="YYYY-MM-DD",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc),
        help="Verify continuous aggregates for this date (skips download)",
    )
    parser.add_argument(
        "--point-divisor",
        type=float,
        default=1000.0,
        help="Price divisor: 1000.0 for gold, 100000.0 for forex (default: 1000.0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Rows per database insert batch (default: 10000)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.05,
        dest="rate_limit_secs",
        help="Minimum seconds between HTTP requests (default: 0.05)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    db_config = _load_db_config()

    if args.verify_date:
        sys.exit(run_verify(args.instrument, args.verify_date, db_config))

    if not args.start or not args.end:
        logger.error("--start and --end are required unless --verify-date is used.")
        sys.exit(2)

    if args.start >= args.end:
        logger.error("--start must be before --end.")
        sys.exit(2)

    sys.exit(
        run_pipeline(
            instrument=args.instrument,
            start=args.start,
            end=args.end,
            dry_run=args.dry_run,
            db_config=db_config,
            point_divisor=args.point_divisor,
            batch_size=args.batch_size,
            rate_limit_secs=args.rate_limit_secs,
        )
    )


if __name__ == "__main__":
    main()
