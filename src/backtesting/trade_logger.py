"""
Trade logging to PostgreSQL with pgvector embeddings.

TradeLogger formats completed trade dicts into the shape expected by the
``trades`` and ``market_context`` database tables (as defined in
src/database/models.py) and inserts them in bulk via asyncpg or psycopg2
connection pools.

When no database pool is supplied (test mode) the logger operates in
dry-run mode: it formats and validates all records but returns synthetic
sequential IDs without touching any database.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema-level constants (mirror schema.sql / models.py)
# ---------------------------------------------------------------------------

_INSERT_TRADE_SQL = """
INSERT INTO trades (
    instrument, source, direction, entry_time, exit_time,
    entry_price, exit_price, stop_loss, take_profit,
    lot_size, risk_pct, r_multiple, pnl, pnl_pct,
    status, confluence_score, signal_tier, created_at
) VALUES (
    %(instrument)s, %(source)s, %(direction)s, %(entry_time)s, %(exit_time)s,
    %(entry_price)s, %(exit_price)s, %(stop_loss)s, %(take_profit)s,
    %(lot_size)s, %(risk_pct)s, %(r_multiple)s, %(pnl)s, %(pnl_pct)s,
    %(status)s, %(confluence_score)s, %(signal_tier)s, %(created_at)s
) RETURNING id;
"""

_INSERT_CONTEXT_SQL = """
INSERT INTO market_context (
    trade_id, timestamp, instrument,
    cloud_direction_4h, cloud_direction_1h, tk_cross_15m,
    chikou_confirmation, cloud_thickness_4h,
    adx_value, atr_value, rsi_value, bb_width_percentile,
    session, nearest_sr_distance, zone_confluence_score,
    context_embedding, created_at
) VALUES (
    %(trade_id)s, %(timestamp)s, %(instrument)s,
    %(cloud_direction_4h)s, %(cloud_direction_1h)s, %(tk_cross_15m)s,
    %(chikou_confirmation)s, %(cloud_thickness_4h)s,
    %(adx_value)s, %(atr_value)s, %(rsi_value)s, %(bb_width_percentile)s,
    %(session)s, %(nearest_sr_distance)s, %(zone_confluence_score)s,
    %(context_embedding)s::vector, %(created_at)s
) RETURNING id;
"""


class TradeLogger:
    """Log trades and market context to PostgreSQL with pgvector embeddings.

    Parameters
    ----------
    db_pool:
        A psycopg2 connection pool (or any object exposing a
        ``getconn()`` / ``putconn()`` interface).  When None, the
        logger operates in dry-run mode and returns synthetic IDs.
    embedding_engine:
        EmbeddingEngine instance used to produce 64-dimensional context
        vectors.  When None a default EmbeddingEngine is constructed.
    dry_run:
        Force dry-run mode even when db_pool is supplied.  Useful for
        testing format validation without database access.
    """

    def __init__(
        self,
        db_pool=None,
        embedding_engine=None,
        dry_run: bool = False,
    ) -> None:
        self._pool = db_pool
        self._dry_run = dry_run or (db_pool is None)

        if embedding_engine is None:
            from src.learning.embeddings import EmbeddingEngine
            self._embedding_engine = EmbeddingEngine()
        else:
            self._embedding_engine = embedding_engine

        self._next_dry_run_id: int = 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_trade(
        self,
        trade: dict,
        context: dict,
        source: str = "backtest",
    ) -> int:
        """Insert a single trade + market_context record into PostgreSQL.

        Parameters
        ----------
        trade:
            Trade dict from the backtesting simulation.  Expected keys
            mirror TradeManager's closed trade log format.
        context:
            Market context dict at trade entry.  Used to build the
            64-dim embedding and populate market_context columns.
        source:
            Trade source label: 'backtest', 'live', or 'paper'.

        Returns
        -------
        Database trade_id (integer).  In dry-run mode returns a
        sequential synthetic ID.
        """
        trade_row, context_row = self.format_for_db(trade, context)
        trade_row["source"] = source

        if self._dry_run:
            trade_id = self._next_dry_run_id
            self._next_dry_run_id += 1
            logger.debug("Dry-run trade log: trade_id=%d direction=%s r=%.2f",
                         trade_id, trade_row.get("direction"), trade_row.get("r_multiple") or 0)
            return trade_id

        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(_INSERT_TRADE_SQL, trade_row)
                trade_id = cur.fetchone()[0]

                context_row["trade_id"] = trade_id
                cur.execute(_INSERT_CONTEXT_SQL, context_row)

            conn.commit()
            return trade_id
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def log_batch(
        self,
        trades: List[dict],
        source: str = "backtest",
    ) -> List[int]:
        """Batch insert multiple trades efficiently.

        In dry-run mode all records are validated and synthetic IDs are
        returned without any database interaction.

        Parameters
        ----------
        trades:
            List of trade dicts.  Each dict must have a ``"context"``
            key containing the market context snapshot at entry.

        Returns
        -------
        List of trade_ids in the same order as the input list.
        """
        if not trades:
            return []

        if self._dry_run:
            ids = []
            for trade in trades:
                context = trade.get("context") or {}
                trade_id = self.log_trade(trade, context, source=source)
                ids.append(trade_id)
            return ids

        conn = self._pool.getconn()
        try:
            trade_ids: List[int] = []
            with conn.cursor() as cur:
                for trade in trades:
                    context = trade.get("context") or {}
                    trade_row, context_row = self.format_for_db(trade, context)
                    trade_row["source"] = source

                    cur.execute(_INSERT_TRADE_SQL, trade_row)
                    trade_id = cur.fetchone()[0]
                    trade_ids.append(trade_id)

                    context_row["trade_id"] = trade_id
                    cur.execute(_INSERT_CONTEXT_SQL, context_row)

            conn.commit()
            return trade_ids
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def format_for_db(
        self,
        trade: dict,
        context: dict,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Format a simulation trade dict and context dict for DB insertion.

        Produces two dicts whose keys map exactly to the ``trades`` and
        ``market_context`` table columns.

        Parameters
        ----------
        trade:
            Raw trade dict from the simulation engine.
        context:
            Market context at trade entry (used for embedding and context cols).

        Returns
        -------
        (trade_row, context_row) ready for psycopg2 parameterised queries.
        """
        now = datetime.now(timezone.utc)

        # Resolve direction value
        direction = str(trade.get("direction") or "long").lower()

        # R-multiple and P&L
        r_multiple = trade.get("r_multiple")
        pnl_points = trade.get("pnl_points") or trade.get("pnl") or 0.0
        lot_size = float(trade.get("lot_size") or 0.0)
        # Approximate monetary P&L: points × lot_size × 100 (standard XAU/USD point value)
        pnl = float(pnl_points) * lot_size * 100.0
        pnl_pct = trade.get("pnl_pct")

        # Status based on partial_exits
        partial_exits = trade.get("partial_exits") or []
        if trade.get("exit_price") is not None or trade.get("exit_time") is not None:
            status = "closed"
        elif partial_exits:
            status = "partial"
        else:
            status = "open"

        trade_row: Dict[str, Any] = {
            "instrument":       str(trade.get("instrument") or "XAUUSD"),
            "source":           "backtest",  # overridden by caller when needed
            "direction":        direction,
            "entry_time":       _to_dt(trade.get("entry_time")),
            "exit_time":        _to_dt(trade.get("exit_time")),
            "entry_price":      float(trade.get("entry_price") or 0.0),
            "exit_price":       _opt_float(trade.get("exit_price")),
            "stop_loss":        float(trade.get("original_stop") or trade.get("stop_loss") or 0.0),
            "take_profit":      _opt_float(trade.get("take_profit")),
            "lot_size":         lot_size,
            "risk_pct":         float(trade.get("risk_pct") or 0.0),
            "r_multiple":       _opt_float(r_multiple),
            "pnl":              round(pnl, 4),
            "pnl_pct":          _opt_float(pnl_pct),
            "status":           status,
            "confluence_score": trade.get("confluence_score"),
            "signal_tier":      trade.get("signal_tier"),
            "created_at":       now,
        }

        # Build embedding from context + trade outcome
        trade_result = None
        if r_multiple is not None:
            trade_result = {
                "r_multiple": float(r_multiple),
                "win": float(r_multiple) > 0.0,
            }
        embedding_data = self._embedding_engine.embed_trade(context, trade_result)
        embedding_list: List[float] = embedding_data["context_embedding"]

        context_row: Dict[str, Any] = {
            "trade_id":             None,  # filled in by the caller after INSERT
            "timestamp":            _to_dt(trade.get("entry_time")) or now,
            "instrument":           str(trade.get("instrument") or "XAUUSD"),
            "cloud_direction_4h":   _direction_label(context.get("cloud_direction_4h")),
            "cloud_direction_1h":   _direction_label(context.get("cloud_direction_1h")),
            "tk_cross_15m":         _direction_label(context.get("tk_cross_15m")),
            "chikou_confirmation":  _opt_bool(context.get("chikou_confirmation") or context.get("chikou_confirmed")),
            "cloud_thickness_4h":   _opt_float(context.get("cloud_thickness_4h")),
            "adx_value":            _opt_float(context.get("adx_value") or context.get("adx")),
            "atr_value":            _opt_float(context.get("atr_value") or context.get("atr")),
            "rsi_value":            _opt_float(context.get("rsi_value") or context.get("rsi")),
            "bb_width_percentile":  _opt_float(context.get("bb_width_percentile")),
            "session":              context.get("session"),
            "nearest_sr_distance":  _opt_float(context.get("nearest_sr_distance")),
            "zone_confluence_score": context.get("zone_confluence_score") or context.get("zone_confluence_count"),
            "context_embedding":    embedding_list,
            "created_at":           now,
        }

        return trade_row, context_row


# ---------------------------------------------------------------------------
# Module-level formatting helpers
# ---------------------------------------------------------------------------

def _to_dt(value: Any) -> Optional[datetime]:
    """Convert various timestamp representations to datetime, or None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return pd.Timestamp(value).to_pydatetime()
    except Exception:
        return None


def _opt_float(value: Any) -> Optional[float]:
    """Cast to float, returning None for missing/NaN values."""
    if value is None:
        return None
    try:
        f = float(value)
        return None if (f != f) else f  # NaN check without math import
    except (TypeError, ValueError):
        return None


def _opt_bool(value: Any) -> Optional[bool]:
    """Cast to bool, returning None for missing values."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return None


def _direction_label(value: Any) -> Optional[str]:
    """Convert integer directional values (1/-1/0) to 'bullish'/'bearish'/None."""
    if value is None:
        return None
    try:
        v = int(value)
        if v == 1:
            return "bullish"
        if v == -1:
            return "bearish"
        return "neutral"
    except (TypeError, ValueError):
        # Already a string label
        return str(value) if value else None


# Lazy import to avoid circular imports at module level
try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore
