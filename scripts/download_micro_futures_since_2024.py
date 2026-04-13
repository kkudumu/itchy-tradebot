"""Download ProjectX micro gold and micro oil from 2024-01-01 through now."""

from __future__ import annotations

from datetime import datetime, timezone

from download_projectx_gold import _PROJECT_ROOT, download_futures_series


START = datetime(2024, 1, 1, tzinfo=timezone.utc)
TIMEFRAME = "1M"
LIMIT_PER_REQUEST = 20_000
PRODUCTS = ("MGC", "MCLE")


if __name__ == "__main__":
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    for product in PRODUCTS:
        output = _PROJECT_ROOT / "data" / f"projectx_{product.lower()}_{TIMEFRAME.lower()}_{START:%Y%m%d}_{end:%Y%m%d}.parquet"
        metadata = output.with_suffix(".json")
        download_futures_series(
            start=START,
            end=end,
            product=product,
            timeframe=TIMEFRAME,
            limit_per_request=LIMIT_PER_REQUEST,
            output_path=output,
            metadata_path=metadata,
        )
