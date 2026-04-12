"""Manage downloading and caching 1-minute OHLCV data per instrument from ProjectX.

Reads the instrument list from ``config/optimizer_instruments.yaml``,
loads cached parquet files when available, and downloads fresh bars via
the ProjectX API when needed.  Downloaded data is calendar-stitched
across CME monthly contracts so the caller receives a single continuous
DataFrame per product.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import yaml

from src.config.loader import load_config
from src.providers import ProjectXApiError, build_projectx_stack

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_MONTH_CODES = ("F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z")
_MONTH_CODE_TO_MONTH = {code: idx + 1 for idx, code in enumerate(_MONTH_CODES)}

# ProjectX bar unit constants for 1-minute bars
_UNIT = 2  # minutes
_UNIT_NUMBER = 1
_BAR_STEP = timedelta(minutes=1)

# Earliest year for which ProjectX exposes futures contracts
_EARLIEST_VISIBLE_YEAR = 2024

# Throttle between consecutive API requests
_INTER_REQUEST_SLEEP = 0.5

# Default bars-per-request limit (ProjectX maximum)
_LIMIT_PER_REQUEST = 20_000


class DataManager:
    """Download, cache, and serve 1-minute OHLCV data for optimizer instruments."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = Path(data_dir) if data_dir else _PROJECT_ROOT / "data"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_instruments(self) -> list[dict]:
        """Load the instrument list from ``config/optimizer_instruments.yaml``."""
        cfg_path = _PROJECT_ROOT / "config" / "optimizer_instruments.yaml"
        with open(cfg_path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return raw.get("instruments", [])

    def get_data(self, instrument: dict) -> pd.DataFrame | None:
        """Return cached data for *instrument*, downloading if not on disk.

        Returns ``None`` when the data cannot be obtained (e.g. missing
        credentials or API failure) and logs the reason.
        """
        # Try the explicit data_file first
        data_file = instrument.get("data_file")
        if data_file:
            path = _PROJECT_ROOT / data_file
            if path.exists():
                logger.info("Loading cached data from %s", path)
                return self._read_parquet(path)

        # Try the canonical auto-generated name
        symbol = instrument["symbol"].lower()
        canonical = self._data_dir / f"projectx_{symbol}_1m.parquet"
        if canonical.exists():
            logger.info("Loading cached data from %s", canonical)
            return self._read_parquet(canonical)

        # Nothing cached — attempt download
        logger.info("No cached data for %s; attempting download.", instrument["symbol"])
        return self.download(instrument)

    def download(
        self,
        instrument: dict,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame | None:
        """Download 1-minute bars from ProjectX for *instrument*.

        Parameters
        ----------
        instrument:
            A dict from ``load_instruments()`` containing at minimum
            ``symbol`` and ``symbol_id``.
        start, end:
            UTC-aware datetimes bounding the requested range.
            Defaults to (now - 90 days, now).

        Returns the resulting DataFrame, or ``None`` on failure.
        """
        self._load_env()

        now = datetime.now(tz=timezone.utc)
        if end is None:
            end = now
        if start is None:
            start = end - timedelta(days=90)

        symbol = instrument["symbol"]
        symbol_id = instrument["symbol_id"]
        product = symbol_id.split(".")[-1]  # e.g. "F.US.MGC" → "MGC"

        try:
            cfg = load_config()
            client, _, _, _ = build_projectx_stack(cfg.provider.projectx, cfg.instruments)
        except Exception:
            logger.exception("Failed to build ProjectX client for %s", symbol)
            return None

        # --- Contract discovery ---
        contracts = self._discover_contracts(client, product, symbol_id, start, end)
        if not contracts:
            logger.warning("No contracts found for %s (%s)", symbol, product)
            return None

        # --- Fetch bars per contract window ---
        windows = self._build_windows(contracts, start, end)
        if not windows:
            logger.warning("No valid download windows for %s", symbol)
            return None

        frames: list[pd.DataFrame] = []
        for contract_id, win_start, win_end in windows:
            logger.info(
                "Fetching %s  %s -> %s",
                contract_id,
                win_start.isoformat(),
                win_end.isoformat(),
            )
            df = self._fetch_bars(client, contract_id, win_start, win_end)
            if df is not None and not df.empty:
                df["contract_id"] = contract_id
                frames.append(df)
            time.sleep(_INTER_REQUEST_SLEEP)

        if not frames:
            logger.warning("No bars returned for %s", symbol)
            return None

        result = pd.concat(frames).sort_index()
        result = result[~result.index.duplicated(keep="first")]

        # Persist to canonical path
        out_path = self._data_dir / f"projectx_{symbol.lower()}_1m.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(out_path)
        logger.info(
            "Saved %d rows for %s to %s  (%s -> %s)",
            len(result),
            symbol,
            out_path,
            result.index.min(),
            result.index.max(),
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_env() -> None:
        """Load variables from the project .env file (if present)."""
        env_path = _PROJECT_ROOT / ".env"
        if not env_path.exists():
            return
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    @staticmethod
    def _read_parquet(path: Path) -> pd.DataFrame:
        """Read a parquet file and normalise to the expected schema."""
        df = pd.read_parquet(path)
        # Ensure UTC DatetimeIndex named "time"
        if not isinstance(df.index, pd.DatetimeIndex):
            if "time" in df.columns:
                df = df.set_index("time")
            elif "timestamp" in df.columns:
                df = df.set_index("timestamp")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index.name = "time"
        # Lowercase column names
        df.columns = [c.lower() for c in df.columns]
        return df

    # --- Contract discovery ---

    @staticmethod
    def _contract_month_start(contract_id: str) -> datetime:
        """Parse a contract id like ``CON.F.US.MGC.M26`` into its month start."""
        code = contract_id.split(".")[-1]  # e.g. "M26"
        month_code = code[0]
        year_suffix = int(code[1:])
        year = 2000 + year_suffix
        month = _MONTH_CODE_TO_MONTH[month_code]
        return datetime(year, month, 1, tzinfo=timezone.utc)

    def _discover_contracts(
        self,
        client,
        product: str,
        symbol_id: str,
        start: datetime,
        end: datetime,
    ) -> list[dict]:
        """Iterate month codes and query ProjectX for valid contracts."""
        contracts: list[dict] = []
        start_year = max(_EARLIEST_VISIBLE_YEAR, start.year)
        end_year = end.year

        for year in range(start_year, end_year + 1):
            yy = year % 100
            for code in _MONTH_CODES:
                contract_id = f"CON.F.US.{product}.{code}{yy:02d}"
                try:
                    resp = client.search_contract_by_id(contract_id)
                except Exception:
                    continue
                contract = resp.get("contract") if isinstance(resp, dict) else None
                if contract and contract.get("symbolId") == symbol_id:
                    contracts.append(contract)
                time.sleep(_INTER_REQUEST_SLEEP)

        contracts.sort(key=lambda c: self._contract_month_start(c["id"]))
        return contracts

    def _build_windows(
        self,
        contracts: list[dict],
        requested_start: datetime,
        requested_end: datetime,
    ) -> list[tuple[str, datetime, datetime]]:
        """Divide the requested range into per-contract windows.

        Returns a list of ``(contract_id, window_start, window_end)``
        tuples covering the requested range with no overlaps.
        """
        if not contracts:
            return []

        windows: list[tuple[str, datetime, datetime]] = []
        for idx, contract in enumerate(contracts):
            # Window starts at the previous contract's month start (roll date)
            if idx == 0:
                win_start = requested_start
            else:
                win_start = self._contract_month_start(contracts[idx - 1]["id"])

            # Window ends at this contract's month start
            if idx == len(contracts) - 1:
                win_end = requested_end
            else:
                win_end = self._contract_month_start(contract["id"])

            # Clip to requested range
            clipped_start = max(win_start, requested_start)
            clipped_end = min(win_end, requested_end)
            if clipped_end > clipped_start:
                windows.append((contract["id"], clipped_start, clipped_end))

        return windows

    # --- Bar fetching ---

    @staticmethod
    def _fetch_bars(
        client,
        contract_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame | None:
        """Fetch 1-minute bars for a single contract, paginating as needed."""
        chunk_span = _BAR_STEP * _LIMIT_PER_REQUEST
        current_start = start_time
        chunks: list[pd.DataFrame] = []

        while current_start < end_time:
            current_end = min(end_time, current_start + chunk_span)
            resp = None

            for attempt in range(10):
                try:
                    resp = client.retrieve_bars(
                        contract_id=contract_id,
                        live=False,
                        start_time=current_start,
                        end_time=current_end,
                        unit=_UNIT,
                        unit_number=_UNIT_NUMBER,
                        limit=_LIMIT_PER_REQUEST,
                        include_partial_bar=False,
                    )
                    break
                except requests.exceptions.HTTPError as exc:
                    if "429" not in str(exc):
                        raise
                    wait = min(90, 10 * (attempt + 1))
                    logger.warning(
                        "Rate limited on %s (%s -> %s); sleeping %ds",
                        contract_id,
                        current_start.isoformat(),
                        current_end.isoformat(),
                        wait,
                    )
                    time.sleep(wait)
                except Exception:
                    logger.exception(
                        "Unexpected error fetching %s (%s -> %s)",
                        contract_id,
                        current_start.isoformat(),
                        current_end.isoformat(),
                    )
                    return None

            if resp is None:
                logger.error(
                    "Exceeded retries fetching %s (%s -> %s)",
                    contract_id,
                    current_start.isoformat(),
                    current_end.isoformat(),
                )
                return None

            bars = resp.get("bars", [])
            if bars:
                df = pd.DataFrame(
                    {
                        "time": pd.to_datetime([b["t"] for b in bars], utc=True),
                        "open": [float(b["o"]) for b in bars],
                        "high": [float(b["h"]) for b in bars],
                        "low": [float(b["l"]) for b in bars],
                        "close": [float(b["c"]) for b in bars],
                        "volume": [float(b["v"]) for b in bars],
                    }
                ).sort_values("time")
                chunks.append(df)
                current_start = pd.Timestamp(df["time"].max()).to_pydatetime() + _BAR_STEP
            else:
                current_start = current_end + _BAR_STEP

            time.sleep(_INTER_REQUEST_SLEEP)

        if not chunks:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        merged = (
            pd.concat(chunks, ignore_index=True)
            .drop_duplicates(subset=["time"])
            .sort_values("time")
        )
        idx = pd.DatetimeIndex(merged["time"], tz="UTC")
        idx.name = "time"
        return merged.set_index(idx).drop(columns=["time"])
