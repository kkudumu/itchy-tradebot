"""Persistent trade outcome database for the mega-vision agent.

Every closed trade is inserted into a SQLite table with full context
(strategy, direction, entry/exit, session, pattern, regime, confluence,
ATR, mega-vision pick, agreement flag). The agent calls ``query_recent``
and ``query`` at decision time to reason about historical behavior
in similar conditions.

The schema is intentionally flat — no joins, no foreign keys — so the
agent can read a single row and have everything it needs. Additional
fields go in ``extra_json`` rather than new columns, so the schema
stays forward-compatible.

Storage:
  * SQLite file (default ``reports/mega_vision_trade_memory.db``)
  * Optional parquet snapshot refreshed on backtest end + every N
    inserts in live mode (see ``snapshot_to_parquet``)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------


@dataclass
class TradeRecord:
    """One closed trade with full agent-relevant context."""

    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    opened_at: Optional[str] = None  # ISO datetime string
    closed_at: Optional[str] = None
    duration_minutes: Optional[float] = None

    strategy_name: str = ""
    instrument_class: str = "forex"
    symbol: str = ""
    direction: str = ""

    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    stop_price: Optional[float] = None
    tp_price: Optional[float] = None
    size: Optional[float] = None

    pnl_usd: Optional[float] = None
    r_multiple: Optional[float] = None

    session: Optional[str] = None
    hour_of_day_utc: Optional[int] = None
    day_of_week: Optional[int] = None

    pattern_type: Optional[str] = None
    regime: Optional[str] = None
    confluence_score: Optional[float] = None
    atr_at_entry: Optional[float] = None
    adx_at_entry: Optional[float] = None

    prop_firm_style: Optional[str] = None
    mega_vision_pick: Optional[str] = None
    mega_vision_agreed: Optional[bool] = None

    extra_json: str = "{}"


# ---------------------------------------------------------------------------
# TradeMemory
# ---------------------------------------------------------------------------


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trade_memory (
    trade_id TEXT PRIMARY KEY,
    opened_at TEXT,
    closed_at TEXT,
    duration_minutes REAL,
    strategy_name TEXT,
    instrument_class TEXT,
    symbol TEXT,
    direction TEXT,
    entry_price REAL,
    exit_price REAL,
    stop_price REAL,
    tp_price REAL,
    size REAL,
    pnl_usd REAL,
    r_multiple REAL,
    session TEXT,
    hour_of_day_utc INTEGER,
    day_of_week INTEGER,
    pattern_type TEXT,
    regime TEXT,
    confluence_score REAL,
    atr_at_entry REAL,
    adx_at_entry REAL,
    prop_firm_style TEXT,
    mega_vision_pick TEXT,
    mega_vision_agreed INTEGER,
    extra_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_trade_memory_strategy
    ON trade_memory (strategy_name);
CREATE INDEX IF NOT EXISTS idx_trade_memory_session
    ON trade_memory (session);
CREATE INDEX IF NOT EXISTS idx_trade_memory_closed
    ON trade_memory (closed_at);
"""


class TradeMemory:
    """SQLite-backed trade outcome store."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        with self._conn:
            self._conn.executescript(_SCHEMA_SQL)

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(self, trade: Dict[str, Any] | TradeRecord) -> str:
        """Insert a trade row. Returns the trade_id."""
        if isinstance(trade, TradeRecord):
            record = trade
        else:
            record = self._coerce_record(trade)

        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO trade_memory VALUES (
                    :trade_id, :opened_at, :closed_at, :duration_minutes,
                    :strategy_name, :instrument_class, :symbol, :direction,
                    :entry_price, :exit_price, :stop_price, :tp_price, :size,
                    :pnl_usd, :r_multiple,
                    :session, :hour_of_day_utc, :day_of_week,
                    :pattern_type, :regime, :confluence_score,
                    :atr_at_entry, :adx_at_entry,
                    :prop_firm_style, :mega_vision_pick, :mega_vision_agreed,
                    :extra_json
                )
                """,
                {
                    **asdict(record),
                    "mega_vision_agreed": (
                        1
                        if record.mega_vision_agreed is True
                        else 0
                        if record.mega_vision_agreed is False
                        else None
                    ),
                },
            )
        return record.trade_id

    def _coerce_record(self, trade: Dict[str, Any]) -> TradeRecord:
        payload = {}
        known = set(TradeRecord.__dataclass_fields__.keys())
        extra: Dict[str, Any] = {}
        for key, value in trade.items():
            if key in known:
                if key in ("opened_at", "closed_at") and isinstance(value, datetime):
                    value = value.isoformat()
                payload[key] = value
            else:
                extra[key] = value
        if extra and "extra_json" not in payload:
            payload["extra_json"] = json.dumps(extra, default=str)
        return TradeRecord(**payload)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, filters: Dict[str, Any] | None = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Return trades matching *filters* as a list of dicts."""
        sql = "SELECT * FROM trade_memory"
        params: List[Any] = []
        if filters:
            where_parts = []
            for key, value in filters.items():
                if value is None:
                    continue
                where_parts.append(f"{key} = ?")
                params.append(value)
            if where_parts:
                sql += " WHERE " + " AND ".join(where_parts)
        sql += " ORDER BY closed_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def query_recent(
        self,
        strategy: str | None = None,
        n: int = 20,
    ) -> List[Dict[str, Any]]:
        """Return the last *n* closed trades (optionally filtered by strategy)."""
        if strategy:
            return self.query({"strategy_name": strategy}, limit=n)
        return self.query({}, limit=n)

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM trade_memory").fetchone()[0])

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate_by(self, *keys: str) -> pd.DataFrame:
        """Group trades by the given columns and return summary stats.

        Produces per-group trade_count, win_count, win_rate, total_pnl,
        avg_r, expectancy.
        """
        rows = [dict(r) for r in self._conn.execute("SELECT * FROM trade_memory").fetchall()]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        if not all(k in df.columns for k in keys):
            missing = [k for k in keys if k not in df.columns]
            raise KeyError(f"unknown aggregation keys: {missing}")

        grouped = df.groupby(list(keys))
        out = grouped.agg(
            trade_count=("trade_id", "count"),
            total_pnl=("pnl_usd", "sum"),
            avg_pnl=("pnl_usd", "mean"),
            avg_r=("r_multiple", "mean"),
            win_count=("r_multiple", lambda s: int((s > 0).sum())),
        ).reset_index()
        out["win_rate"] = out["win_count"] / out["trade_count"]
        out["expectancy"] = out["avg_r"]
        return out

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot_to_parquet(self, path: str | Path) -> Path:
        """Write the full trade_memory table to a parquet file."""
        rows = [dict(r) for r in self._conn.execute("SELECT * FROM trade_memory").fetchall()]
        df = pd.DataFrame(rows)
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        return out

    def close(self) -> None:
        self._conn.close()
