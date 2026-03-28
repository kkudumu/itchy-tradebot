# Data — Intent

## Functions

### DukascopyDownloader.__init__(instrument: str, point_divisor: float, rate_limit_secs: float, request_timeout: int, max_retries: int) -> None
- **Does**: Initialises the downloader with instrument settings and creates an HTTP session with retry/backoff.
- **Why**: Encapsulates all Dukascopy-specific connection setup so callers only provide the instrument name.
- **Relationships**: Calls _build_session(); instance used by download_hour() and download_range().
- **Decisions**: Default point_divisor=1000.0 targets gold (XAUUSD); forex pairs would use 100000.0.

### DukascopyDownloader.download_hour(year: int, month: int, day: int, hour: int) -> pd.DataFrame
- **Does**: Downloads and decodes one hour of tick data from Dukascopy's public feed.
- **Why**: Provides the atomic unit of data retrieval; download_range composes multiple calls to this method.
- **Relationships**: Calls _build_url(), _fetch(), _decode_bi5(); called by download_range().
- **Decisions**: Returns an empty DataFrame (not None) for missing hours so callers can concatenate without null checks.

### DukascopyDownloader.download_range(start_date: datetime, end_date: datetime, progress_callback: Callable | None) -> pd.DataFrame
- **Does**: Downloads tick data across a date range, skipping weekends, and concatenates all hours into one sorted DataFrame.
- **Why**: Main entry point for bulk historical data retrieval; handles the hour-by-hour iteration and progress reporting.
- **Relationships**: Calls download_hour() per market hour; calls _generate_market_hours() for hour list; called by the data pipeline.
- **Decisions**: Weekends are pre-filtered via _generate_market_hours rather than relying on 404s, reducing unnecessary HTTP requests.

### DataNormalizer.ticks_to_1m_ohlcv(ticks: pd.DataFrame) -> pd.DataFrame
- **Does**: Aggregates raw tick data into 1-minute OHLCV bars using mid-price for fair value.
- **Why**: Converts variable-frequency tick data into a fixed 1-minute series required by the strategy engine and TimescaleDB.
- **Relationships**: Called by the data pipeline after downloading; output consumed by DataNormalizer.validate() and TimescaleLoader.bulk_insert().
- **Decisions**: Uses mid-price ((bid+ask)/2) instead of bid or ask alone to represent fair value and reduce spread bias.

### DataNormalizer.normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame
- **Does**: Ensures the timestamp column is timezone-aware UTC, localising naive timestamps or converting other timezones.
- **Why**: Prevents silent timezone mismatches that would corrupt time-series joins and aggregation.
- **Relationships**: Called by the data pipeline before database insertion.
- **Decisions**: Naive timestamps are assumed UTC and localised (not converted) to avoid accidental double-conversion.

### DataNormalizer.detect_gaps(df: pd.DataFrame, expected_freq: str) -> pd.DataFrame
- **Does**: Identifies gaps in a 1-minute OHLCV time series and classifies each gap as weekend, daily_break, holiday, or unknown.
- **Why**: Surfaces data quality issues so the pipeline can distinguish expected market closures from unexpected data holes.
- **Relationships**: Calls _classify_gap(); called by the data pipeline for quality reporting.
- **Decisions**: Logs unknown gaps at WARNING level to flag genuinely missing data while keeping expected gaps at DEBUG.

### DataNormalizer.validate(df: pd.DataFrame) -> list[str]
- **Does**: Runs data quality checks on a 1-minute OHLCV DataFrame including timestamp, price range, OHLC consistency, and volume validation.
- **Why**: Acts as a gate before database ingestion to prevent corrupt or nonsensical data from entering the system.
- **Relationships**: Called by the data pipeline after aggregation; blocks TimescaleLoader.bulk_insert() if errors are found.
- **Decisions**: Gold price bounds (200-10000 USD/oz) are intentionally loose to accommodate multi-decade historical data.

### TimescaleLoader.__init__(connection_config: dict) -> None
- **Does**: Normalises connection config keys and stores them for psycopg2.connect().
- **Why**: Bridges the YAML config key names (e.g., "name") to psycopg2's expected keys (e.g., "dbname").
- **Relationships**: Config typically comes from DatabaseConfig.model_dump(); instance used by bulk_insert(), verify_aggregates(), get_data_range(), get_row_count().
- **Decisions**: Strips pool_min/pool_max/vector_dimensions keys that psycopg2 does not understand.

### TimescaleLoader.bulk_insert(df: pd.DataFrame, instrument: str, batch_size: int) -> int
- **Does**: Batch-inserts 1-minute OHLCV rows into the candles_1m hypertable using execute_values.
- **Why**: High-throughput path for loading historical data; ON CONFLICT DO NOTHING makes re-runs idempotent.
- **Relationships**: Calls _validate_columns(); called by the data pipeline after normalisation and validation.
- **Decisions**: Uses psycopg2 execute_values (not executemany) for significantly better insert throughput.

### TimescaleLoader.verify_aggregates(instrument: str, sample_date: datetime) -> dict
- **Does**: Spot-checks that TimescaleDB continuous aggregate views (5m, 15m, 1h, 4h) match a manual Python aggregation for a sample day.
- **Why**: Validates that the continuous aggregate definitions produce correct OHLCV values, catching schema or policy misconfigurations.
- **Relationships**: Calls _verify_one_aggregate() per view; called by pipeline verification step.
- **Decisions**: Uses relative error comparison with 1e-6 tolerance because floating-point aggregation can differ slightly between Python and PostgreSQL.

### TimescaleLoader.get_data_range(instrument: str) -> tuple[datetime | None, datetime | None]
- **Does**: Returns the earliest and latest timestamps stored in candles_1m for an instrument.
- **Why**: Lets the pipeline determine what date range has already been loaded to support incremental downloads.
- **Relationships**: Called by pipeline orchestration to compute resume points.
- **Decisions**: Returns (None, None) instead of raising when no data exists.

### TimescaleLoader.get_row_count(instrument: str) -> int
- **Does**: Returns the total number of 1-minute bars stored for an instrument.
- **Why**: Provides a quick sanity check for pipeline completion and data volume monitoring.
- **Relationships**: Called by pipeline diagnostics and tests.
- **Decisions**: None.
