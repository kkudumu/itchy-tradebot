"""Comprehensive trade and decision logging for the engine.

Logs EVERYTHING, including skipped signals — the learning loop needs this
data to improve signal quality over time.

All writes are best-effort: a logging failure must never interrupt the
trading pipeline.  Errors are captured and logged at WARNING level.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert an object to a JSON-serialisable form."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    # Fall back to string representation for complex objects
    return str(obj)


class EngineTradeLogger:
    """Log all engine decisions, trade entries/exits, screenshots, and zone updates.

    All database interactions are wrapped in try/except so that a DB outage
    never blocks the trading loop.  Failures are written to the Python logger.

    Parameters
    ----------
    db_pool:
        Database connection pool with a ``get_cursor()`` context manager.
        When None, all DB operations are no-ops (useful in tests without a DB).
    embedding_engine:
        :class:`~src.learning.embeddings.EmbeddingEngine` instance used to
        generate and store context embeddings alongside trade entries.
    """

    def __init__(self, db_pool=None, embedding_engine=None) -> None:
        self.db_pool = db_pool
        self.embedding_engine = embedding_engine

        # In-memory buffer used when db_pool is None or unavailable
        self._decision_buffer: List[dict] = []
        self._trade_buffer: List[dict] = []

    # ------------------------------------------------------------------
    # Decision logging
    # ------------------------------------------------------------------

    def log_decision(self, decision: Any) -> Optional[int]:
        """Insert a decision record into the ``decisions`` table.

        All fields from the Decision dataclass are stored, including the
        full ``edge_results``, ``similarity_data``, and ``reasoning`` trace.

        Parameters
        ----------
        decision:
            A :class:`~src.engine.decision_engine.Decision` instance.

        Returns
        -------
        int or None
            Inserted row ID, or None if unavailable.
        """
        record = {
            "timestamp": getattr(decision, "timestamp", datetime.now(timezone.utc)),
            "instrument": getattr(decision, "instrument", ""),
            "action": getattr(decision, "action", ""),
            "confluence_score": getattr(decision, "confluence_score", 0),
            "reasoning": getattr(decision, "reasoning", ""),
            "trade_id": getattr(decision, "trade_id", None),
            "executed": getattr(decision, "executed", False),
            "edge_results": _to_json_safe(getattr(decision, "edge_results", {})),
            "similarity_data": _to_json_safe(getattr(decision, "similarity_data", {})),
            "execution_detail": _to_json_safe(getattr(decision, "execution_detail", {})),
        }

        # Signal detail (optional)
        signal = getattr(decision, "signal", None)
        if signal is not None:
            record["direction"] = getattr(signal, "direction", "")
            record["entry_price"] = getattr(signal, "entry_price", None)
            record["stop_loss"] = getattr(signal, "stop_loss", None)
            record["take_profit"] = getattr(signal, "take_profit", None)
            record["signal_tier"] = getattr(signal, "quality_tier", "")
            record["atr"] = getattr(signal, "atr", None)
            record["zone_context"] = _to_json_safe(getattr(signal, "zone_context", {}))
            record["signal_reasoning"] = _to_json_safe(getattr(signal, "reasoning", {}))

        self._decision_buffer.append(record)

        if self.db_pool is None:
            return None

        try:
            with self.db_pool.get_cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO decisions (
                        timestamp, instrument, action, confluence_score, reasoning,
                        trade_id, executed, edge_results, similarity_data,
                        execution_detail, direction, entry_price, stop_loss,
                        take_profit, signal_tier, atr, zone_context, signal_reasoning
                    ) VALUES (
                        %(timestamp)s, %(instrument)s, %(action)s, %(confluence_score)s,
                        %(reasoning)s, %(trade_id)s, %(executed)s,
                        %(edge_results)s::jsonb, %(similarity_data)s::jsonb,
                        %(execution_detail)s::jsonb, %(direction)s, %(entry_price)s,
                        %(stop_loss)s, %(take_profit)s, %(signal_tier)s, %(atr)s,
                        %(zone_context)s::jsonb, %(signal_reasoning)s::jsonb
                    )
                    RETURNING id
                    """,
                    {
                        **record,
                        "edge_results": json.dumps(record.get("edge_results", {})),
                        "similarity_data": json.dumps(record.get("similarity_data", {})),
                        "execution_detail": json.dumps(record.get("execution_detail", {})),
                        "zone_context": json.dumps(record.get("zone_context", {})),
                        "signal_reasoning": json.dumps(record.get("signal_reasoning", {})),
                        "direction": record.get("direction", ""),
                        "entry_price": record.get("entry_price"),
                        "stop_loss": record.get("stop_loss"),
                        "take_profit": record.get("take_profit"),
                        "signal_tier": record.get("signal_tier", ""),
                        "atr": record.get("atr"),
                    },
                )
                row = cur.fetchone()
                if row:
                    return row.get("id") or row[0]
        except Exception as exc:  # noqa: BLE001
            logger.warning("log_decision DB write failed: %s", exc)

        return None

    # ------------------------------------------------------------------
    # Trade entry logging
    # ------------------------------------------------------------------

    def log_trade_entry(self, trade: dict, context: dict) -> Optional[int]:
        """Log a trade entry with full market context and generate an embedding.

        Parameters
        ----------
        trade:
            Dict containing at minimum: instrument, direction, entry_price,
            stop_loss, take_profit, lot_size, confluence_score, timestamp.
        context:
            Market context dict used to generate the context embedding.

        Returns
        -------
        int or None
            Inserted trade row ID.
        """
        # Generate embedding
        embedding_list: Optional[list] = None
        if self.embedding_engine is not None:
            try:
                emb = self.embedding_engine.create_embedding(context)
                embedding_list = emb.tolist()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Embedding generation failed for trade entry: %s", exc)

        record = {
            "timestamp": trade.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "instrument": trade.get("instrument", ""),
            "direction": trade.get("direction", ""),
            "entry_price": trade.get("entry_price"),
            "stop_loss": trade.get("stop_loss"),
            "take_profit": trade.get("take_profit"),
            "lot_size": trade.get("lot_size"),
            "confluence_score": trade.get("confluence_score", 0),
            "signal_tier": trade.get("signal_tier", ""),
            "atr": trade.get("atr"),
            "ticket": trade.get("ticket"),
            "internal_trade_id": trade.get("internal_trade_id"),
            "similarity_data": _to_json_safe(trade.get("similarity_data", {})),
            "edge_results": _to_json_safe(trade.get("edge_results", {})),
            "zone_context": _to_json_safe(trade.get("zone_context", {})),
            "signal_reasoning": _to_json_safe(trade.get("signal_reasoning", {})),
            "context_embedding": embedding_list,
        }
        self._trade_buffer.append(record)

        if self.db_pool is None:
            return None

        try:
            with self.db_pool.get_cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO trades (
                        entry_time, instrument, direction, entry_price,
                        stop_loss, take_profit, lot_size, confluence_score,
                        signal_tier, atr, broker_ticket, status
                    ) VALUES (
                        %(timestamp)s, %(instrument)s, %(direction)s, %(entry_price)s,
                        %(stop_loss)s, %(take_profit)s, %(lot_size)s, %(confluence_score)s,
                        %(signal_tier)s, %(atr)s, %(ticket)s, 'open'
                    )
                    RETURNING id
                    """,
                    record,
                )
                row = cur.fetchone()
                trade_db_id = row.get("id") or row[0] if row else None

                # Insert market context with embedding
                if trade_db_id and embedding_list:
                    vec_str = "[" + ",".join(f"{v:.8f}" for v in embedding_list) + "]"
                    cur.execute(
                        """
                        INSERT INTO market_context (
                            trade_id, cloud_direction_4h, cloud_direction_1h,
                            tk_cross_15m, session, adx_value, atr_value,
                            zone_confluence_score, context_embedding
                        ) VALUES (
                            %(trade_id)s, %(cloud_direction_4h)s, %(cloud_direction_1h)s,
                            %(tk_cross_15m)s, %(session)s, %(adx_value)s, %(atr_value)s,
                            %(zone_confluence_score)s, %(context_embedding)s::vector
                        )
                        """,
                        {
                            "trade_id": trade_db_id,
                            "cloud_direction_4h": context.get("cloud_direction_4h", 0),
                            "cloud_direction_1h": context.get("cloud_direction_1h", 0),
                            "tk_cross_15m": context.get("tk_cross_15m", 0),
                            "session": context.get("session", "unknown"),
                            "adx_value": context.get("adx_value", 0.0),
                            "atr_value": context.get("atr_value", 0.0),
                            "zone_confluence_score": context.get("zone_confluence_score", 0),
                            "context_embedding": vec_str,
                        },
                    )
                return trade_db_id
        except Exception as exc:  # noqa: BLE001
            logger.warning("log_trade_entry DB write failed: %s", exc)

        return None

    # ------------------------------------------------------------------
    # Trade exit logging
    # ------------------------------------------------------------------

    def log_trade_exit(self, trade_id: int, exit_data: dict) -> None:
        """Update the trade record in the DB with exit data and final P&L.

        Parameters
        ----------
        trade_id:
            Internal or broker trade identifier used as the lookup key.
        exit_data:
            Dict with keys: current_price, action, reason, r_multiple, etc.
        """
        if self.db_pool is None:
            return

        try:
            with self.db_pool.get_cursor() as cur:
                cur.execute(
                    """
                    UPDATE trades
                    SET
                        exit_price   = %(exit_price)s,
                        exit_time    = %(exit_time)s,
                        r_multiple   = %(r_multiple)s,
                        exit_reason  = %(exit_reason)s,
                        status       = 'closed'
                    WHERE id = %(trade_id)s
                       OR broker_ticket = %(trade_id)s
                    """,
                    {
                        "trade_id": trade_id,
                        "exit_price": exit_data.get("current_price") or exit_data.get("exit_price"),
                        "exit_time": exit_data.get("exit_time", datetime.now(timezone.utc)),
                        "r_multiple": exit_data.get("r_multiple"),
                        "exit_reason": exit_data.get("reason", ""),
                    },
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("log_trade_exit DB write failed for trade %s: %s", trade_id, exc)

    # ------------------------------------------------------------------
    # Screenshot logging
    # ------------------------------------------------------------------

    def log_screenshot(self, trade_id: int, phase: str, filepath: str) -> None:
        """Record a screenshot reference in the ``trade_screenshots`` table.

        Parameters
        ----------
        trade_id:
            Broker ticket or internal trade ID.
        phase:
            One of 'pre_entry', 'entry', 'periodic', 'exit'.
        filepath:
            Absolute path (or URL) to the saved screenshot file.
        """
        if self.db_pool is None:
            logger.debug("Screenshot logged (no DB): trade=%s phase=%s path=%s", trade_id, phase, filepath)
            return

        try:
            with self.db_pool.get_cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO trade_screenshots (trade_id, phase, filepath, captured_at)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (trade_id, phase, filepath, datetime.now(timezone.utc)),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("log_screenshot DB write failed: %s", exc)

    # ------------------------------------------------------------------
    # Zone update logging
    # ------------------------------------------------------------------

    def log_zone_update(self, zone_changes: List[dict]) -> None:
        """Log zone state changes to the ``zone_events`` table.

        Parameters
        ----------
        zone_changes:
            List of dicts describing each zone state change.
            Each dict should include: zone_id, old_status, new_status, reason, timestamp.
        """
        if not zone_changes:
            return

        if self.db_pool is None:
            logger.debug("Zone updates logged (no DB): %d changes", len(zone_changes))
            return

        try:
            with self.db_pool.get_cursor() as cur:
                for change in zone_changes:
                    cur.execute(
                        """
                        INSERT INTO zone_events (
                            zone_id, old_status, new_status, reason, event_time
                        ) VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            change.get("zone_id"),
                            change.get("old_status", ""),
                            change.get("new_status", ""),
                            change.get("reason", ""),
                            change.get("timestamp", datetime.now(timezone.utc)),
                        ),
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("log_zone_update DB write failed: %s", exc)

    # ------------------------------------------------------------------
    # Edge stats materialized view
    # ------------------------------------------------------------------

    def refresh_edge_stats(self) -> None:
        """Refresh the ``edge_stats`` materialized view.

        Called periodically by the engine (every 50 trades or 1 hour).
        """
        if self.db_pool is None:
            return

        try:
            with self.db_pool.get_cursor() as cur:
                cur.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY edge_stats")
            logger.debug("edge_stats materialized view refreshed")
        except Exception as exc:  # noqa: BLE001
            logger.warning("edge_stats refresh failed: %s", exc)

    # ------------------------------------------------------------------
    # Flush / shutdown
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Flush any pending in-memory records.

        In this implementation the buffer is advisory only (used for
        test assertions).  Production persistence goes directly to the DB.
        """
        logger.debug(
            "EngineTradeLogger flush — decisions=%d trades=%d",
            len(self._decision_buffer),
            len(self._trade_buffer),
        )

    # ------------------------------------------------------------------
    # Read-only accessors (for tests)
    # ------------------------------------------------------------------

    @property
    def decision_buffer(self) -> List[dict]:
        """Read-only snapshot of the in-memory decision buffer."""
        return list(self._decision_buffer)

    @property
    def trade_buffer(self) -> List[dict]:
        """Read-only snapshot of the in-memory trade entry buffer."""
        return list(self._trade_buffer)
