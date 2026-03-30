#!/usr/bin/env python3
"""
Historical data download with monthly checkpointing.

Downloads XAU/USD tick data from Dukascopy MONTH BY MONTH,
normalizes to 1-minute bars, saves a parquet checkpoint per month,
and bulk-inserts into TimescaleDB. Resumes from the last completed month.

Usage:
    python scripts/download_history.py --start 2019-01-01 --end 2026-03-30
    python scripts/download_history.py --merge-only  # just merge checkpoints
    python scripts/download_history.py --verify-date 2023-06-15
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.data.downloader import DukascopyDownloader
from src.data.normalizer import DataNormalizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("download_history")

CHECKPOINT_DIR = _PROJECT_ROOT / "data" / "checkpoints" / "dukascopy"
MERGED_OUTPUT = _PROJECT_ROOT / "data" / "xauusd_dukascopy_1m.parquet"


def _load_env():
    """Load .env file."""
    env_path = _PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())


def _load_db_config() -> dict:
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


def _generate_months(start: datetime, end: datetime) -> list[tuple[int, int]]:
    """Generate (year, month) tuples for the date range."""
    months = []
    current = start.replace(day=1)
    while current < end:
        months.append((current.year, current.month))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return months


def _month_checkpoint_path(year: int, month: int) -> Path:
    return CHECKPOINT_DIR / f"{year}-{month:02d}.parquet"


def _get_completed_months() -> set[tuple[int, int]]:
    """Return set of (year, month) that have checkpoint files."""
    if not CHECKPOINT_DIR.exists():
        return set()
    completed = set()
    for f in CHECKPOINT_DIR.glob("*.parquet"):
        try:
            parts = f.stem.split("-")
            completed.add((int(parts[0]), int(parts[1])))
        except (ValueError, IndexError):
            continue
    return completed


def download_month(
    downloader: DukascopyDownloader,
    normalizer: DataNormalizer,
    year: int,
    month: int,
    instrument: str,
    point_divisor: float,
) -> pd.DataFrame | None:
    """Download one month of data, normalize to 1M bars, save checkpoint."""

    # Month boundaries
    month_start = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        month_end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        month_end = datetime(year, month + 1, 1, tzinfo=timezone.utc)

    logger.info("Downloading %s %d-%02d ...", instrument, year, month)

    ticks = downloader.download_range(
        start_date=month_start,
        end_date=month_end,
    )

    if ticks.empty:
        logger.warning("No ticks for %d-%02d (weekend/holiday month?)", year, month)
        # Save empty checkpoint so we don't retry
        cp_path = _month_checkpoint_path(year, month)
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_parquet(cp_path)
        return None

    # Normalize to 1M bars
    bars = normalizer.ticks_to_1m_ohlcv(ticks)
    bars = normalizer.normalize_timestamps(bars)

    logger.info("  %d ticks → %d 1M bars for %d-%02d",
                len(ticks), len(bars), year, month)

    # Save checkpoint
    cp_path = _month_checkpoint_path(year, month)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(cp_path)
    logger.info("  Checkpoint saved: %s (%.1f MB)",
                cp_path.name, cp_path.stat().st_size / 1e6)

    return bars


def insert_to_db(bars: pd.DataFrame, instrument: str, db_config: dict, batch_size: int = 10_000):
    """Insert bars into TimescaleDB."""
    from src.data.loader import TimescaleLoader
    loader = TimescaleLoader(db_config)
    try:
        inserted = loader.bulk_insert(bars, instrument=instrument, batch_size=batch_size)
        logger.info("  Inserted %d rows into DB", inserted)
    except Exception as exc:
        logger.error("  DB insert failed: %s", exc)


def merge_checkpoints() -> pd.DataFrame:
    """Merge all month checkpoints into a single DataFrame."""
    checkpoints = sorted(CHECKPOINT_DIR.glob("*.parquet")) if CHECKPOINT_DIR.exists() else []
    if not checkpoints:
        return pd.DataFrame()

    dfs = []
    for cp in checkpoints:
        try:
            df = pd.read_parquet(cp)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            logger.warning("Skipping corrupt checkpoint %s: %s", cp.name, e)

    if not dfs:
        return pd.DataFrame()

    merged = pd.concat(dfs).sort_index()
    merged = merged[~merged.index.duplicated(keep="first")]
    return merged


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
    """Download month-by-month with checkpointing and DB insertion."""

    downloader = DukascopyDownloader(
        instrument=instrument,
        point_divisor=point_divisor,
        rate_limit_secs=rate_limit_secs,
    )
    normalizer = DataNormalizer()

    all_months = _generate_months(start, end)
    completed = _get_completed_months()
    remaining = [(y, m) for y, m in all_months if (y, m) not in completed]

    logger.info("Total months: %d, Already checkpointed: %d, Remaining: %d",
                len(all_months), len(completed), len(remaining))

    for idx, (year, month) in enumerate(remaining, 1):
        logger.info("=== Month %d/%d: %d-%02d ===", idx, len(remaining), year, month)

        bars = download_month(downloader, normalizer, year, month, instrument, point_divisor)

        if bars is not None and not bars.empty and not dry_run:
            insert_to_db(bars, instrument, db_config, batch_size)

    # Final merge
    logger.info("=== Merging all checkpoints ===")
    merged = merge_checkpoints()
    if not merged.empty:
        MERGED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(MERGED_OUTPUT)
        logger.info("Merged output: %s (%d bars, %.1f MB)",
                     MERGED_OUTPUT, len(merged), MERGED_OUTPUT.stat().st_size / 1e6)

    if not dry_run and not merged.empty:
        from src.data.loader import TimescaleLoader
        loader = TimescaleLoader(db_config)
        try:
            count = loader.get_row_count(instrument)
            lo, hi = loader.get_data_range(instrument)
            logger.info("DB total: %d rows, range: %s → %s", count, lo, hi)
        except Exception:
            pass

    logger.info("=== Pipeline complete ===")
    return 0


def run_verify(instrument: str, sample_date: datetime, db_config: dict) -> int:
    from src.data.loader import TimescaleLoader
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
    return 0 if all_ok else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Dukascopy data with monthly checkpointing."
    )
    parser.add_argument("--instrument", default="XAUUSD")
    parser.add_argument("--start",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc),
        help="Start date YYYY-MM-DD")
    parser.add_argument("--end",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc),
        help="End date YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true",
        help="Download and checkpoint but don't insert into DB")
    parser.add_argument("--merge-only", action="store_true",
        help="Just merge existing checkpoints into final parquet")
    parser.add_argument("--verify-date", metavar="YYYY-MM-DD",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc))
    parser.add_argument("--point-divisor", type=float, default=1000.0)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--rate-limit", type=float, default=0.05, dest="rate_limit_secs")
    parser.add_argument("--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    _load_env()

    db_config = _load_db_config()

    if args.verify_date:
        sys.exit(run_verify(args.instrument, args.verify_date, db_config))

    if args.merge_only:
        logger.info("Merge-only mode")
        merged = merge_checkpoints()
        if not merged.empty:
            MERGED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            merged.to_parquet(MERGED_OUTPUT)
            logger.info("Saved %d bars to %s", len(merged), MERGED_OUTPUT)
        else:
            logger.error("No checkpoints to merge")
        return

    if not args.start or not args.end:
        logger.error("--start and --end are required")
        sys.exit(2)

    sys.exit(run_pipeline(
        instrument=args.instrument,
        start=args.start,
        end=args.end,
        dry_run=args.dry_run,
        db_config=db_config,
        point_divisor=args.point_divisor,
        batch_size=args.batch_size,
        rate_limit_secs=args.rate_limit_secs,
    ))


if __name__ == "__main__":
    main()
