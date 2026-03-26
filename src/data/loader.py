"""
Bulk insertion of 1-minute OHLCV data into TimescaleDB and aggregate verification.

Design decisions
----------------
- Uses psycopg2's execute_values for high-throughput batch inserts.
- ON CONFLICT DO NOTHING means re-running the pipeline is idempotent.
- verify_aggregates performs a spot-check on a sample day to confirm that the
  TimescaleDB continuous aggregates (candles_5m, candles_15m, candles_1h,
  candles_4h) produce values that match a manual aggregation from candles_1m.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)

# Columns expected in the input DataFrame, in order.
_REQUIRED_COLS = ("timestamp", "open", "high", "low", "close", "volume")

# Maximum allowed relative price deviation when comparing aggregates to manual
# computation.  Floating-point rounding means exact equality is unreliable.
_TOLERANCE = 1e-6

# Continuous aggregate views and their bucket width in minutes.
_AGGREGATE_VIEWS = {
    "candles_5m":  5,
    "candles_15m": 15,
    "candles_1h":  60,
    "candles_4h":  240,
}


class TimescaleLoader:
    """
    Load 1-minute OHLCV data into TimescaleDB and verify continuous aggregates.

    Parameters
    ----------
    connection_config : dict
        Keys accepted: host, port, dbname (or name), user, password.
        Forwarded directly to psycopg2.connect().
    """

    def __init__(self, connection_config: dict) -> None:
        # Normalise 'name' → 'dbname' so callers can pass the YAML key directly.
        cfg = dict(connection_config)
        if "name" in cfg and "dbname" not in cfg:
            cfg["dbname"] = cfg.pop("name")
        # Remove keys psycopg2 does not understand.
        for extra_key in ("pool_min", "pool_max", "vector_dimensions"):
            cfg.pop(extra_key, None)
        self._connect_kwargs = cfg

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _connect(self) -> psycopg2.extensions.connection:
        """Open and return a new psycopg2 connection."""
        return psycopg2.connect(**self._connect_kwargs)

    # ------------------------------------------------------------------
    # Bulk insert
    # ------------------------------------------------------------------

    def bulk_insert(
        self,
        df: pd.DataFrame,
        instrument: str = "XAUUSD",
        batch_size: int = 10_000,
    ) -> int:
        """
        Bulk insert 1-minute OHLCV rows into candles_1m.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: timestamp, open, high, low, close, volume.
        instrument : str
            Instrument label stored in the ``instrument`` column.
        batch_size : int
            Number of rows per execute_values call.

        Returns
        -------
        int
            Number of rows successfully inserted (excluding conflicts).
        """
        if df.empty:
            logger.info("bulk_insert called with empty DataFrame — nothing to do.")
            return 0

        _validate_columns(df)

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        sql = """
            INSERT INTO candles_1m (timestamp, instrument, open, high, low, close, volume)
            VALUES %s
            ON CONFLICT (timestamp, instrument) DO NOTHING
        """

        total_inserted = 0
        conn = self._connect()
        try:
            with conn:
                with conn.cursor() as cur:
                    for start in range(0, len(df), batch_size):
                        chunk = df.iloc[start : start + batch_size]
                        rows = [
                            (
                                row.timestamp.to_pydatetime(),
                                instrument,
                                float(row.open),
                                float(row.high),
                                float(row.low),
                                float(row.close),
                                float(row.volume),
                            )
                            for row in chunk.itertuples(index=False)
                        ]
                        execute_values(cur, sql, rows, page_size=batch_size)
                        batch_count = cur.rowcount if cur.rowcount >= 0 else len(rows)
                        total_inserted += batch_count
                        logger.debug(
                            "Inserted batch %d–%d: %d rows",
                            start,
                            start + len(chunk) - 1,
                            batch_count,
                        )
        finally:
            conn.close()

        logger.info(
            "bulk_insert complete: %d rows inserted for %s.", total_inserted, instrument
        )
        return total_inserted

    # ------------------------------------------------------------------
    # Aggregate verification
    # ------------------------------------------------------------------

    def verify_aggregates(self, instrument: str, sample_date: datetime) -> dict:
        """
        Verify that TimescaleDB continuous aggregates match manual aggregation.

        For each of candles_5m, candles_15m, candles_1h, candles_4h:
          1. Query the raw 1-minute data for *sample_date*.
          2. Manually aggregate it in Python.
          3. Query the continuous aggregate view for the same day.
          4. Compare and record any discrepancies.

        Parameters
        ----------
        instrument : str
            Instrument to check.
        sample_date : datetime
            The UTC calendar day to verify.

        Returns
        -------
        dict
            ``{"candles_5m": {"status": "ok", ...}, "candles_15m": ..., ...}``
            Each inner dict has keys: status ("ok" | "mismatch" | "no_data"),
            manual_bars, db_bars, and discrepancies (list of details).
        """
        sample_date = _floor_to_day(sample_date)
        day_start = sample_date
        day_end = sample_date + timedelta(days=1)

        conn = self._connect()
        results: dict = {}
        try:
            with conn:
                with conn.cursor() as cur:
                    # Fetch raw 1-minute data for the day
                    cur.execute(
                        """
                        SELECT timestamp, open, high, low, close, volume
                        FROM candles_1m
                        WHERE instrument = %s
                          AND timestamp >= %s
                          AND timestamp < %s
                        ORDER BY timestamp
                        """,
                        (instrument, day_start, day_end),
                    )
                    rows_1m = cur.fetchall()

                if not rows_1m:
                    for view in _AGGREGATE_VIEWS:
                        results[view] = {
                            "status":       "no_data",
                            "manual_bars":  0,
                            "db_bars":      0,
                            "discrepancies": ["No 1-minute data found for this date."],
                        }
                    return results

                df_1m = pd.DataFrame(
                    rows_1m, columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                df_1m["timestamp"] = pd.to_datetime(df_1m["timestamp"], utc=True)

                for view, bucket_minutes in _AGGREGATE_VIEWS.items():
                    results[view] = self._verify_one_aggregate(
                        cur, df_1m, instrument, view, bucket_minutes, day_start, day_end
                    )
        finally:
            conn.close()

        return results

    def _verify_one_aggregate(
        self,
        cur,
        df_1m: pd.DataFrame,
        instrument: str,
        view: str,
        bucket_minutes: int,
        day_start: datetime,
        day_end: datetime,
    ) -> dict:
        """Compare one continuous aggregate view against manual Python aggregation."""
        bucket = pd.Timedelta(minutes=bucket_minutes)
        df_1m = df_1m.copy()
        df_1m["bucket"] = df_1m["timestamp"].dt.floor(f"{bucket_minutes}min")

        manual = (
            df_1m.groupby("bucket")
            .agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
            )
            .reset_index()
        )

        cur.execute(
            f"""
            SELECT timestamp, open, high, low, close, volume
            FROM {view}
            WHERE instrument = %s
              AND timestamp >= %s
              AND timestamp < %s
            ORDER BY timestamp
            """,
            (instrument, day_start, day_end),
        )
        db_rows = cur.fetchall()
        db_df = pd.DataFrame(
            db_rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        if not db_df.empty:
            db_df["timestamp"] = pd.to_datetime(db_df["timestamp"], utc=True)

        discrepancies: list[str] = []

        if len(manual) != len(db_df):
            discrepancies.append(
                f"Bar count mismatch: manual={len(manual)}, db={len(db_df)}"
            )

        if not db_df.empty:
            db_df = db_df.set_index("timestamp")
            manual = manual.set_index("bucket")
            common_idx = manual.index.intersection(db_df.index)

            for ts in common_idx:
                for col in ("open", "high", "low", "close", "volume"):
                    m_val = float(manual.loc[ts, col])
                    d_val = float(db_df.loc[ts, col])
                    if m_val == 0:
                        rel_err = abs(d_val - m_val)
                    else:
                        rel_err = abs(d_val - m_val) / abs(m_val)
                    if rel_err > _TOLERANCE:
                        discrepancies.append(
                            f"{ts.isoformat()} {col}: manual={m_val:.6f}, db={d_val:.6f}"
                        )

        return {
            "status":        "ok" if not discrepancies else "mismatch",
            "manual_bars":   len(manual),
            "db_bars":       len(db_df),
            "discrepancies": discrepancies,
        }

    # ------------------------------------------------------------------
    # Metadata queries
    # ------------------------------------------------------------------

    def get_data_range(self, instrument: str) -> tuple[Optional[datetime], Optional[datetime]]:
        """
        Return the (min_timestamp, max_timestamp) of stored data for *instrument*.

        Returns (None, None) when no data is present.
        """
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT MIN(timestamp), MAX(timestamp)
                    FROM candles_1m
                    WHERE instrument = %s
                    """,
                    (instrument,),
                )
                row = cur.fetchone()
        finally:
            conn.close()

        if row is None or row[0] is None:
            return None, None
        return row[0], row[1]

    def get_row_count(self, instrument: str) -> int:
        """Return the total number of 1-minute bars stored for *instrument*."""
        conn = self._connect()
        try:
            with conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM candles_1m WHERE instrument = %s",
                    (instrument,),
                )
                row = cur.fetchone()
        finally:
            conn.close()

        return int(row[0]) if row else 0


# =============================================================================
# Private helpers
# =============================================================================

def _validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def _floor_to_day(dt: datetime) -> datetime:
    """Return midnight UTC of the given datetime."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)
