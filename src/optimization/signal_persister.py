"""Batch-insert signal log rows to PostgreSQL.

Receives a list of signal event dicts (from StrategyTelemetryCollector)
and persists them into the ``signal_log`` table in a single round-trip
using ``psycopg2.extras.execute_values``.

Usage
-----
from src.database.connection import DatabasePool
from src.optimization.signal_persister import SignalPersister

pool = DatabasePool(config)
pool.initialise()
persister = SignalPersister(pool)

count = persister.persist_signals(run_id, events)
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)

# Column list for the INSERT (excludes auto-generated ``id``).
_INSERT_COLUMNS = (
    "run_id",
    "timestamp",
    "strategy_name",
    "direction",
    "confluence_score",
    "entry_price",
    "stop_loss",
    "take_profit",
    "filtered_by",
    "entered",
    "trade_result_r",
    "exit_reason",
    "pnl_usd",
    "market_snapshot",
)

_INSERT_SQL = f"""
    INSERT INTO signal_log ({", ".join(_INSERT_COLUMNS)})
    VALUES %s
"""

# Value template — one %s per column.  The JSONB column needs an
# explicit cast so psycopg2 sends the JSON string correctly.
_VALUE_TEMPLATE = (
    "("
    + ", ".join(
        "%s::jsonb" if col == "market_snapshot" else "%s"
        for col in _INSERT_COLUMNS
    )
    + ")"
)

_PAGE_SIZE = 500


def _extract_row(run_id: uuid.UUID, event: dict[str, Any]) -> tuple:
    """Map a single telemetry event dict to a positional tuple matching
    ``_INSERT_COLUMNS``.

    Field-name mapping from the StrategyTelemetryCollector schema to the
    signal_log table:
        timestamp_utc  → timestamp
        price          → entry_price
        filtered_by OR rejection_reason → filtered_by
        realized_r     → trade_result_r
    """
    # Resolve the filtered_by field: prefer explicit "filtered_by", fall
    # back to "rejection_reason", then to "filter_stage".
    filtered_by = (
        event.get("filtered_by")
        or event.get("rejection_reason")
        or event.get("filter_stage")
    )

    # Build market_snapshot JSONB from context fields.
    market_snapshot = json.dumps(
        {
            "atr": event.get("atr"),
            "adx": event.get("adx"),
            "session": event.get("session"),
            "regime": event.get("regime"),
        },
        default=str,
    )

    return (
        str(run_id),                                   # run_id
        event.get("timestamp_utc") or event.get("timestamp"),  # timestamp
        event.get("strategy_name"),                    # strategy_name
        event.get("direction"),                        # direction
        _to_int(event.get("confluence_score")),        # confluence_score
        _to_float(event.get("price")                   # entry_price
                  or event.get("entry_price")),
        _to_float(event.get("stop_loss")),             # stop_loss
        _to_float(event.get("take_profit")),           # take_profit
        filtered_by,                                   # filtered_by
        event.get("entered"),                          # entered
        _to_float(event.get("realized_r")              # trade_result_r
                  or event.get("trade_result_r")),
        event.get("exit_reason"),                      # exit_reason
        _to_float(event.get("pnl_usd")),              # pnl_usd
        market_snapshot,                               # market_snapshot
    )


def _to_float(value: Any) -> float | None:
    """Cast to float, preserving None."""
    return float(value) if value is not None else None


def _to_int(value: Any) -> int | None:
    """Cast to int, preserving None."""
    return int(value) if value is not None else None


class SignalPersister:
    """Batch-insert signal events into the ``signal_log`` table.

    Parameters
    ----------
    db_pool:
        A ``DatabasePool`` (or compatible object) whose ``get_cursor``
        context manager yields a psycopg2 cursor and auto-commits on
        clean exit.
    """

    def __init__(self, db_pool) -> None:
        self._pool = db_pool

    def persist_signals(
        self,
        run_id: uuid.UUID,
        events: list[dict],
    ) -> int:
        """Batch-insert signal events for a trial.

        Parameters
        ----------
        run_id:
            Foreign key into ``optimization_runs``.
        events:
            List of dicts, each representing one signal event from the
            StrategyTelemetryCollector.

        Returns
        -------
        int — number of rows inserted.
        """
        if not events:
            return 0

        rows = [_extract_row(run_id, evt) for evt in events]

        with self._pool.get_cursor() as cur:
            execute_values(
                cur,
                _INSERT_SQL,
                rows,
                template=_VALUE_TEMPLATE,
                page_size=_PAGE_SIZE,
            )
            count = cur.rowcount

        logger.info(
            "SignalPersister: inserted %d signal_log rows for run_id=%s",
            count,
            run_id,
        )
        return count
