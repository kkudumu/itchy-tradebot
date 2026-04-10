"""
Download XAU/USD 1-minute OHLCV data from Twelve Data API with checkpointing.

Saves partial chunks to data/checkpoints/ as parquet files.
Merges all chunks into data/xauusd_1m.parquet on completion.
Resumes from last checkpoint on restart.

Usage:
    python scripts/download_twelvedata.py
    python scripts/download_twelvedata.py --start 2020-04-06 --end 2026-03-30
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_twelvedata")

CHECKPOINT_DIR = _PROJECT_ROOT / "data" / "checkpoints" / "twelvedata"
OUTPUT_PATH = _PROJECT_ROOT / "data" / "xauusd_1m.parquet"
CSV_PATH = _PROJECT_ROOT / "data" / "xauusd_1m.csv"

API_KEY = os.getenv("TWELVE_DATA_API_KEY", "")
SYMBOL = "XAU/USD"
INTERVAL = "1min"
OUTPUT_SIZE = 5000
BASE_URL = "https://api.twelvedata.com/time_series"


def _load_env():
    """Load .env file if present."""
    env_path = _PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())


def _get_existing_checkpoints() -> list[Path]:
    """Return sorted list of existing checkpoint files."""
    if not CHECKPOINT_DIR.exists():
        return []
    return sorted(CHECKPOINT_DIR.glob("chunk_*.parquet"))


def _get_earliest_checkpoint_date() -> str | None:
    """Find the earliest date in existing checkpoints to know where to resume."""
    checkpoints = _get_existing_checkpoints()
    if not checkpoints:
        return None

    earliest = None
    for cp in checkpoints:
        try:
            df = pd.read_parquet(cp)
            if not df.empty:
                ts = df.index.min()
                if earliest is None or ts < earliest:
                    earliest = ts
        except Exception:
            continue

    if earliest is not None:
        return (earliest - pd.Timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
    return None


def fetch_chunk(end_date: str | None = None) -> pd.DataFrame:
    """Fetch one chunk of 5000 bars ending at end_date."""
    global API_KEY
    if not API_KEY:
        API_KEY = os.getenv("TWELVE_DATA_API_KEY", "")

    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": OUTPUT_SIZE,
        "apikey": API_KEY,
        "format": "JSON",
        "dp": 5,
    }
    if end_date:
        params["end_date"] = end_date

    for attempt in range(5):
        try:
            r = requests.get(BASE_URL, params=params, timeout=30)
            data = r.json()

            if data.get("status") == "error":
                code = data.get("code", 0)
                if code == 429:
                    logger.warning("Rate limited, waiting 65s...")
                    time.sleep(65)
                    continue
                logger.error("API error: %s", data.get("message", ""))
                return pd.DataFrame()

            values = data.get("values", [])
            if not values:
                return pd.DataFrame()

            df = pd.DataFrame(values)
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col].astype(float)
            df = df.set_index("datetime").sort_index()
            return df

        except Exception as e:
            logger.warning("Request failed (attempt %d): %s", attempt + 1, e)
            time.sleep(10)

    return pd.DataFrame()


def save_checkpoint(chunk: pd.DataFrame, chunk_num: int) -> Path:
    """Save a chunk to a checkpoint file."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKPOINT_DIR / f"chunk_{chunk_num:04d}.parquet"
    chunk.to_parquet(path)
    return path


def merge_checkpoints() -> pd.DataFrame:
    """Merge all checkpoint files into a single DataFrame."""
    checkpoints = _get_existing_checkpoints()
    if not checkpoints:
        return pd.DataFrame()

    dfs = []
    for cp in checkpoints:
        try:
            dfs.append(pd.read_parquet(cp))
        except Exception as e:
            logger.warning("Skipping corrupt checkpoint %s: %s", cp.name, e)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def download_all(start_date: str = "2020-04-06", end_date: str = "2026-03-30") -> pd.DataFrame:
    """Download all available 1-min data with checkpointing."""

    # Check for existing checkpoints to resume
    resume_end = _get_earliest_checkpoint_date()
    existing_chunks = len(_get_existing_checkpoints())

    if resume_end:
        logger.info("Resuming from checkpoint — %d chunks exist, earliest data: %s",
                     existing_chunks, resume_end)
        current_end = resume_end
        chunk_num = existing_chunks
    else:
        logger.info("Starting fresh download: %s to %s", start_date, end_date)
        current_end = end_date
        chunk_num = 0

    request_count = 0

    while True:
        request_count += 1
        chunk_num += 1
        logger.info("Request #%d (chunk %d) — ending at %s", request_count, chunk_num, current_end)

        chunk = fetch_chunk(end_date=current_end)

        if chunk.empty:
            logger.info("No more data returned. Done downloading.")
            break

        # Filter out data before start_date
        start_ts = pd.Timestamp(start_date, tz="UTC")
        chunk = chunk[chunk.index >= start_ts]

        if chunk.empty:
            logger.info("All remaining data is before start_date. Done.")
            break

        # Save checkpoint immediately
        cp_path = save_checkpoint(chunk, chunk_num)
        logger.info("  Got %d bars [%s to %s] → saved %s",
                     len(chunk),
                     chunk.index.min().strftime("%Y-%m-%d %H:%M"),
                     chunk.index.max().strftime("%Y-%m-%d %H:%M"),
                     cp_path.name)

        if len(chunk) < OUTPUT_SIZE:
            logger.info("Received fewer bars than requested. Done.")
            break

        earliest = chunk.index.min()
        current_end = (earliest - pd.Timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")

        if earliest <= start_ts:
            logger.info("Reached start_date. Done.")
            break

        # Rate limiting
        if request_count % 7 == 0:
            logger.info("  Rate limit pause (15s)...")
            time.sleep(15)
        else:
            time.sleep(2)

    # Merge all checkpoints
    logger.info("Merging %d checkpoints...", len(_get_existing_checkpoints()))
    df = merge_checkpoints()

    if df.empty:
        logger.error("No data after merging checkpoints!")
        return df

    logger.info("Total: %d bars from %s to %s",
                len(df), df.index.min().date(), df.index.max().date())
    return df


def main():
    _load_env()

    import argparse
    parser = argparse.ArgumentParser(description="Download XAU/USD from Twelve Data with checkpointing")
    parser.add_argument("--start", default="2020-04-06", help="Start date (default: 2020-04-06)")
    parser.add_argument("--end", default="2026-03-30", help="End date (default: 2026-03-30)")
    parser.add_argument("--merge-only", action="store_true", help="Just merge existing checkpoints")
    args = parser.parse_args()

    if args.merge_only:
        logger.info("Merge-only mode — combining existing checkpoints")
        df = merge_checkpoints()
    else:
        df = download_all(start_date=args.start, end_date=args.end)

    if df.empty:
        logger.error("No data to save.")
        sys.exit(1)

    # Save final output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH)
    logger.info("Saved to %s (%d bars, %.1f MB)", OUTPUT_PATH, len(df),
                OUTPUT_PATH.stat().st_size / 1e6)

    df.to_csv(CSV_PATH)
    logger.info("CSV saved to %s", CSV_PATH)

    print(f"\n{'='*60}")
    print(f"  XAU/USD 1-Minute Data Download Complete")
    print(f"{'='*60}")
    print(f"  Bars:        {len(df):,}")
    print(f"  Date range:  {df.index.min().date()} to {df.index.max().date()}")
    print(f"  Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    print(f"  File:        {OUTPUT_PATH}")
    print(f"  Checkpoints: {CHECKPOINT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
