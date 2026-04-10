"""Download TopstepX Micro Gold from 2026-01-01 through 2026-04-09."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from download_projectx_gold import _PROJECT_ROOT, download_gold_series


if __name__ == "__main__":
    output = _PROJECT_ROOT / "data" / "projectx_mgc_1m_20260101_20260409.parquet"
    metadata = output.with_suffix(".json")
    download_gold_series(
        start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end=datetime(2026, 4, 9, 23, 59, tzinfo=timezone.utc),
        product="MGC",
        timeframe="1M",
        limit_per_request=20_000,
        output_path=output,
        metadata_path=metadata,
    )
