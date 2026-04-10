"""
Download continuous Gold futures bars from TopstepX / ProjectX.

This script builds a simple calendar-rolled continuous series by stitching
the standard metals contract cycle:
    G, J, M, Q, V, Z  -> Feb, Apr, Jun, Aug, Oct, Dec

It currently uses Micro Gold (MGC) by default because the repository is
configured around that product, but you can switch to full Gold (GCE).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config.loader import load_config
from src.providers import ProjectXApiError, build_projectx_stack


_MONTH_CODE_TO_MONTH = {
    "G": 2,
    "J": 4,
    "M": 6,
    "Q": 8,
    "V": 10,
    "Z": 12,
}
_PRODUCT_TO_SYMBOL_ID = {
    "MGC": "F.US.MGC",
    "GCE": "F.US.GCE",
}
_EARLIEST_VISIBLE_GOLD_YEAR = 2024
_TIMEFRAME_MAP = {
    "1M": (2, 1, timedelta(minutes=1)),
    "5M": (2, 5, timedelta(minutes=5)),
    "15M": (2, 15, timedelta(minutes=15)),
    "1H": (3, 1, timedelta(hours=1)),
}


@dataclass
class ContractWindow:
    contract_id: str
    symbol_id: str
    name: str
    window_start: datetime
    window_end: datetime


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
    parser = argparse.ArgumentParser(
        description="Download a calendar-stitched Gold futures series from TopstepX / ProjectX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start", required=True, help="UTC ISO start timestamp.")
    parser.add_argument("--end", required=True, help="UTC ISO end timestamp.")
    parser.add_argument("--product", choices=["MGC", "GCE"], default="MGC", help="Gold product to stitch.")
    parser.add_argument("--timeframe", choices=["1M", "5M", "15M", "1H"], default="1M")
    parser.add_argument("--limit-per-request", type=int, default=20000)
    parser.add_argument("--output", type=str, default=None, help="Optional output parquet path.")
    parser.add_argument("--metadata-json", type=str, default=None, help="Optional metadata JSON path.")
    return parser


def _parse_utc(value: str) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _discover_gold_contracts(client, product: str, start_year: int, end_year: int) -> list[dict]:
    contracts: list[dict] = []
    for year in range(start_year, end_year + 1):
        yy = year % 100
        for code in ("G", "J", "M", "Q", "V", "Z"):
            contract_id = f"CON.F.US.{product}.{code}{yy:02d}"
            try:
                resp = client.search_contract_by_id(contract_id)
            except Exception:
                continue
            contract = resp.get("contract")
            if contract:
                contracts.append(contract)
    contracts.sort(key=lambda c: (_contract_month_start(c["id"])))
    return contracts


def _contract_month_start(contract_id: str) -> datetime:
    code = contract_id.split(".")[-1]
    month_code = code[0]
    year_suffix = int(code[1:])
    year = 2000 + year_suffix
    month = _MONTH_CODE_TO_MONTH[month_code]
    return datetime(year, month, 1, tzinfo=timezone.utc)


def _build_windows(
    contracts: list[dict],
    requested_start: datetime,
    requested_end: datetime,
) -> tuple[list[ContractWindow], datetime | None]:
    if not contracts:
        return [], None

    earliest_supported_start = datetime(
        _contract_month_start(contracts[0]["id"]).year,
        1,
        1,
        tzinfo=timezone.utc,
    )

    effective_start = max(requested_start, earliest_supported_start)
    windows: list[ContractWindow] = []
    prev_month_start: datetime | None = None

    for contract in contracts:
        month_start = _contract_month_start(contract["id"])
        window_start = prev_month_start or effective_start
        window_end = month_start
        prev_month_start = month_start

        if window_end <= effective_start:
            continue
        if window_start >= requested_end:
            break

        clipped_start = max(window_start, effective_start)
        clipped_end = min(window_end, requested_end)
        if clipped_end > clipped_start:
            windows.append(
                ContractWindow(
                    contract_id=contract["id"],
                    symbol_id=contract["symbolId"],
                    name=contract["name"],
                    window_start=clipped_start,
                    window_end=clipped_end,
                )
            )

    if prev_month_start is not None and prev_month_start < requested_end:
        last_contract = contracts[-1]
        windows.append(
            ContractWindow(
                contract_id=last_contract["id"],
                symbol_id=last_contract["symbolId"],
                name=last_contract["name"],
                window_start=max(prev_month_start, effective_start),
                window_end=requested_end,
            )
        )

    deduped: list[ContractWindow] = []
    for win in windows:
        if win.window_end > win.window_start:
            deduped.append(win)
    return deduped, earliest_supported_start


def _fetch_contract_bars(
    client,
    contract_id: str,
    live: bool,
    timeframe: str,
    start_time: datetime,
    end_time: datetime,
    limit_per_request: int,
) -> pd.DataFrame:
    unit, unit_number, step = _TIMEFRAME_MAP[timeframe]
    chunk_span = step * limit_per_request
    current_start = start_time
    chunks: list[pd.DataFrame] = []

    while current_start < end_time:
        current_end = min(end_time, current_start + chunk_span)
        resp = None
        for attempt in range(10):
            try:
                resp = client.retrieve_bars(
                    contract_id=contract_id,
                    live=live,
                    start_time=current_start,
                    end_time=current_end,
                    unit=unit,
                    unit_number=unit_number,
                    limit=limit_per_request,
                    include_partial_bar=False,
                )
                break
            except requests.exceptions.HTTPError as exc:
                if "429" not in str(exc):
                    raise
                wait_seconds = min(90, 10 * (attempt + 1))
                print(
                    f"Rate limited on {contract_id} "
                    f"{current_start.isoformat()} -> {current_end.isoformat()}; "
                    f"sleeping {wait_seconds}s and retrying...",
                    flush=True,
                )
                time.sleep(wait_seconds)
        if resp is None:
            raise RuntimeError(
                f"Exceeded retries while fetching {contract_id} "
                f"{current_start.isoformat()} -> {current_end.isoformat()}"
            )
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
            current_start = pd.Timestamp(df["time"].max()).to_pydatetime() + step
        else:
            current_start = current_end + step

    if not chunks:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    merged = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time")
    return merged.set_index(pd.DatetimeIndex(merged["time"], tz="UTC")).drop(columns=["time"])


def download_gold_series(
    start: datetime,
    end: datetime,
    product: str,
    timeframe: str,
    limit_per_request: int,
    output_path: Path,
    metadata_path: Path | None = None,
) -> Path:
    _load_local_env()
    cfg = load_config()
    client, _, _, _ = build_projectx_stack(cfg.provider.projectx, cfg.instruments)

    contracts = _discover_gold_contracts(
        client,
        product=product,
        start_year=_EARLIEST_VISIBLE_GOLD_YEAR,
        end_year=end.year,
    )
    windows, earliest_supported_start = _build_windows(contracts, start, end)
    if not windows:
        raise RuntimeError(f"No {product} contracts available for the requested range.")

    if earliest_supported_start is not None and start < earliest_supported_start:
        print(
            f"Requested start {start.isoformat()} is earlier than the earliest API-visible "
            f"{product} contract window {earliest_supported_start.isoformat()}. "
            f"Series will be truncated."
        )

    frames: list[pd.DataFrame] = []
    metadata = {
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "product": product,
        "timeframe": timeframe,
        "earliest_supported_start": earliest_supported_start.isoformat() if earliest_supported_start else None,
        "windows": [],
    }

    for window in windows:
        print(
            f"Fetching {window.contract_id} ({window.name}) "
            f"{window.window_start.isoformat()} -> {window.window_end.isoformat()}",
            flush=True,
        )
        df = _fetch_contract_bars(
            client=client,
            contract_id=window.contract_id,
            live=False,
            timeframe=timeframe,
            start_time=window.window_start,
            end_time=window.window_end,
            limit_per_request=limit_per_request,
        )
        if not df.empty:
            df["contract_id"] = window.contract_id
            frames.append(df)
        metadata["windows"].append(
            {
                "contract_id": window.contract_id,
                "symbol_id": window.symbol_id,
                "name": window.name,
                "window_start": window.window_start.isoformat(),
                "window_end": window.window_end.isoformat(),
                "rows": int(len(df)),
            }
        )

    if not frames:
        raise RuntimeError("No bars were returned for the requested range.")

    result = pd.concat(frames).sort_index()
    result = result[~result.index.duplicated(keep="first")]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path)
    actual_start = result.index.min()
    actual_end = result.index.max()
    print(
        f"Saved {len(result)} rows to {output_path} "
        f"({actual_start} -> {actual_end})",
        flush=True,
    )
    if actual_start > start:
        print(
            f"Warning: requested start {start.isoformat()} but first returned bar is "
            f"{actual_start.isoformat()}. TopstepX did not provide the earlier portion "
            f"of the range.",
            flush=True,
        )
    if actual_end < end - _TIMEFRAME_MAP[timeframe][2]:
        print(
            f"Warning: requested end {end.isoformat()} but last returned bar is "
            f"{actual_end.isoformat()}. TopstepX did not provide the later portion "
            f"of the range.",
            flush=True,
        )

    if metadata_path is not None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Saved metadata to {metadata_path}", flush=True)

    return output_path


def main() -> int:
    args = _build_parser().parse_args()
    start = _parse_utc(args.start)
    end = _parse_utc(args.end)
    product = args.product.upper()
    if args.output:
        output_path = Path(args.output)
    else:
        stem = f"projectx_{product.lower()}_{args.timeframe.lower()}_{start:%Y%m%d}_{end:%Y%m%d}.parquet"
        output_path = _PROJECT_ROOT / "data" / stem

    metadata_path = Path(args.metadata_json) if args.metadata_json else output_path.with_suffix(".json")

    try:
        download_gold_series(
            start=start,
            end=end,
            product=product,
            timeframe=args.timeframe,
            limit_per_request=args.limit_per_request,
            output_path=output_path,
            metadata_path=metadata_path,
        )
        return 0
    except ProjectXApiError as exc:
        print(f"ProjectX API error: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
