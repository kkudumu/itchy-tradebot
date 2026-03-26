"""Run the full pre-challenge validation pipeline from the command line.

Usage examples
--------------
# Validate against a date range using the database loader:
    python scripts/run_validation.py --from-db --instrument XAUUSD \\
        --data-start 2019-01-01 --data-end 2024-01-01

# Validate against a local CSV/Parquet file:
    python scripts/run_validation.py --data-file /path/to/xauusd_1m.parquet \\
        --data-start 2019-01-01 --data-end 2024-01-01

# Quick smoke-test with fewer trials and simulations:
    python scripts/run_validation.py --data-file data.parquet \\
        --wf-trials 50 --mc-sims 2000

Output
------
- Console: plain-text summary with PASS/FAIL per threshold and final verdict.
- HTML report: written to reports/validation_<timestamp>.html by default.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Make the project root importable when running from the scripts/ directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_validation")


# =============================================================================
# CLI argument definition
# =============================================================================


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pre-challenge go/no-go validation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data source (mutually exclusive).
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--data-file",
        metavar="PATH",
        help="Path to a local Parquet or CSV file with OHLCV 1-minute bars.",
    )
    src.add_argument(
        "--from-db",
        action="store_true",
        help="Load price data from the configured PostgreSQL database.",
    )

    # Instrument and date range.
    p.add_argument(
        "--instrument",
        default="XAUUSD",
        metavar="SYM",
        help="Instrument symbol used when loading from the database.",
    )
    p.add_argument(
        "--data-start",
        default="2019-01-01",
        metavar="YYYY-MM-DD",
        help="Start date for the data window.",
    )
    p.add_argument(
        "--data-end",
        default=None,
        metavar="YYYY-MM-DD",
        help="End date for the data window (defaults to today).",
    )

    # Walk-forward parameters.
    p.add_argument(
        "--wf-trials",
        type=int,
        default=200,
        metavar="N",
        help="Optuna trials per walk-forward IS window.",
    )
    p.add_argument(
        "--is-months",
        type=int,
        default=12,
        metavar="M",
        help="In-sample window length in months.",
    )
    p.add_argument(
        "--oos-months",
        type=int,
        default=3,
        metavar="M",
        help="Out-of-sample window length in months.",
    )
    p.add_argument(
        "--storage",
        default=None,
        metavar="URL",
        help="Optuna storage URL (e.g. postgresql://user:pass@host/db).",
    )

    # Monte Carlo parameters.
    p.add_argument(
        "--mc-sims",
        type=int,
        default=10_000,
        metavar="N",
        help="Number of Monte Carlo challenge simulations.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="INT",
        help="Random seed for reproducible Monte Carlo results.",
    )

    # Risk parameters.
    p.add_argument(
        "--initial-balance",
        type=float,
        default=10_000.0,
        metavar="FLOAT",
        help="Simulated account starting balance.",
    )
    p.add_argument(
        "--haircut",
        type=float,
        default=25.0,
        metavar="PCT",
        help="Haircut percentage applied to backtest metrics (0–99).",
    )

    # Output.
    p.add_argument(
        "--output-dir",
        default="reports",
        metavar="DIR",
        help="Directory for the HTML report output.",
    )
    p.add_argument(
        "--no-html",
        action="store_true",
        help="Skip saving the HTML report (print text summary only).",
    )

    return p


# =============================================================================
# Data loading
# =============================================================================


def _load_from_file(path: str, start: str, end: str | None) -> pd.DataFrame:
    """Load OHLCV data from a Parquet or CSV file."""
    path_obj = Path(path)
    if not path_obj.exists():
        logger.error("Data file not found: %s", path)
        sys.exit(1)

    if path_obj.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif path_obj.suffix.lower() == ".csv":
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        logger.error("Unsupported file format: %s", path_obj.suffix)
        sys.exit(1)

    # Ensure the index is a DatetimeIndex with UTC timezone.
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    df = _filter_date_range(df, start, end)
    logger.info("Loaded %d bars from file: %s", len(df), path)
    return df


def _load_from_db(instrument: str, start: str, end: str | None) -> pd.DataFrame:
    """Load OHLCV data from the PostgreSQL database via the project data loader."""
    try:
        from src.data.loader import DataLoader

        loader = DataLoader()
        df = loader.load(instrument=instrument, start=start, end=end or "")
    except ImportError:
        logger.error("DataLoader not available — check database configuration.")
        sys.exit(1)
    except Exception as exc:
        logger.error("Failed to load data from database: %s", exc)
        sys.exit(1)

    df = _filter_date_range(df, start, end)
    logger.info("Loaded %d bars from database for %s.", len(df), instrument)
    return df


def _filter_date_range(
    df: pd.DataFrame, start: str, end: str | None
) -> pd.DataFrame:
    """Slice a DataFrame to [start, end] using the DatetimeIndex."""
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.utcnow()

    mask = (df.index >= start_ts) & (df.index <= end_ts)
    return df.loc[mask].copy()


# =============================================================================
# Main entry point
# =============================================================================


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    end_date = args.data_end or datetime.utcnow().strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    if args.from_db:
        data = _load_from_db(args.instrument, args.data_start, end_date)
    else:
        data = _load_from_file(args.data_file, args.data_start, end_date)

    if data.empty:
        logger.error("No data available for the specified date range — aborting.")
        sys.exit(1)

    logger.info(
        "Data range: %s → %s  (%d bars)",
        data.index[0].date(),
        data.index[-1].date(),
        len(data),
    )

    # ------------------------------------------------------------------
    # Run validation pipeline
    # ------------------------------------------------------------------
    from src.validation.go_nogo import GoNoGoValidator
    from src.validation.report import ValidationReportGenerator

    config = {
        "is_months": args.is_months,
        "oos_months": args.oos_months,
    }

    logger.info(
        "Starting validation pipeline (WF trials=%d, MC sims=%d, haircut=%g%%).",
        args.wf_trials,
        args.mc_sims,
        args.haircut,
    )

    validator = GoNoGoValidator(
        data=data,
        config=config,
        initial_balance=args.initial_balance,
        haircut_pct=args.haircut,
    )

    result = validator.run_full_validation(
        n_wf_trials=args.wf_trials,
        n_mc_sims=args.mc_sims,
        storage=args.storage,
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    generator = ValidationReportGenerator()

    # Always print the text summary.
    print(generator.generate_text(result))

    # Save HTML report unless suppressed.
    if not args.no_html:
        os.makedirs(args.output_dir, exist_ok=True)
        ts_str = result.timestamp.strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(args.output_dir, f"validation_{ts_str}.html")
        html = generator.generate_html(result)
        saved_path = generator.save_report(html, report_path)
        logger.info("HTML report: %s", saved_path)

    # Exit with code 1 on NO-GO so CI pipelines can detect failure.
    if result.final_verdict == "NO-GO":
        logger.warning("Verdict is NO-GO — exiting with code 1.")
        sys.exit(1)

    logger.info("Verdict: %s — pipeline complete.", result.final_verdict)


if __name__ == "__main__":
    main()
