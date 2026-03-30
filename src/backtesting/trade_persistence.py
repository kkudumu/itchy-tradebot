"""
Trade persistence pipeline for the Agentic Self-Optimisation Loop.

Connects the EmbeddingEngine → SimilaritySearch → pgvector pipeline so that
every closed backtest trade is durably stored in the ``trades``,
``market_context``, and ``pattern_signatures`` tables with a ``run_id`` tag
that groups all trades from one backtest run.

Design notes
------------
- All three table writes happen inside **one transaction** so a crash can
  never leave partially-persisted trades.
- ``ON CONFLICT DO NOTHING`` makes re-runs idempotent.
- If the database pool is missing or the connection fails the class logs a
  warning and continues without persistence.
- ``run_id`` and ``strategy_tag`` columns are added by the migration at
  ``src/database/migrations/add_run_id_strategy_tag.sql``.

Usage
-----
    pool = DatabasePool()
    pool.initialise()
    engine = EmbeddingEngine()
    tp = TradePersistence(db_pool=pool, embedding_engine=engine)

    tp.persist_run(
        run_id="run_20260329_001",
        trades=[...],
        config_snapshot={"strategy": "ichimoku_v1", ...},
        metrics={"win_rate": 0.60, ...},
    )

    similar = tp.get_similar_past_configs(config_embedding, top_k=5)
    history = tp.get_run_history(last_n=10)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

class RunSummary:
    """Summary metrics for a single backtest run returned by get_run_history."""

    __slots__ = (
        "run_id",
        "strategy_tag",
        "trade_count",
        "win_rate",
        "avg_r",
        "total_pnl",
        "created_at",
    )

    def __init__(
        self,
        run_id: str,
        strategy_tag: Optional[str],
        trade_count: int,
        win_rate: float,
        avg_r: float,
        total_pnl: float,
        created_at: Optional[datetime],
    ) -> None:
        self.run_id = run_id
        self.strategy_tag = strategy_tag
        self.trade_count = trade_count
        self.win_rate = win_rate
        self.avg_r = avg_r
        self.total_pnl = total_pnl
        self.created_at = created_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "strategy_tag": self.strategy_tag,
            "trade_count": self.trade_count,
            "win_rate": self.win_rate,
            "avg_r": self.avg_r,
            "total_pnl": self.total_pnl,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class SimilarConfig:
    """A past run configuration that produced similar market conditions."""

    __slots__ = (
        "run_id",
        "strategy_tag",
        "similarity",
        "win_rate",
        "avg_r",
        "trade_count",
    )

    def __init__(
        self,
        run_id: str,
        strategy_tag: Optional[str],
        similarity: float,
        win_rate: float,
        avg_r: float,
        trade_count: int,
    ) -> None:
        self.run_id = run_id
        self.strategy_tag = strategy_tag
        self.similarity = similarity
        self.win_rate = win_rate
        self.avg_r = avg_r
        self.trade_count = trade_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "strategy_tag": self.strategy_tag,
            "similarity": self.similarity,
            "win_rate": self.win_rate,
            "avg_r": self.avg_r,
            "trade_count": self.trade_count,
        }


# ---------------------------------------------------------------------------
# TradePersistence
# ---------------------------------------------------------------------------

class TradePersistence:
    """Persist backtest trades to pgvector-backed PostgreSQL tables.

    Parameters
    ----------
    db_pool:
        A ``DatabasePool`` instance (or compatible object with
        ``get_connection()`` context manager).  Pass ``None`` to run without
        persistence (logs a warning on each attempted write).
    embedding_engine:
        An ``EmbeddingEngine`` instance used to create 64-dim context
        embeddings for each trade.  When ``None``, a default engine is
        created on first use.
    """

    # SQL for inserting a trade row (run_id and strategy_tag added by migration)
    _INSERT_TRADE_SQL = """
        INSERT INTO trades (
            instrument, source, direction,
            entry_time, exit_time,
            entry_price, exit_price,
            stop_loss, take_profit,
            lot_size, risk_pct,
            r_multiple, pnl, pnl_pct,
            status, confluence_score, signal_tier,
            atr, exit_reason,
            run_id, strategy_tag
        ) VALUES (
            %(instrument)s, %(source)s, %(direction)s,
            %(entry_time)s, %(exit_time)s,
            %(entry_price)s, %(exit_price)s,
            %(stop_loss)s, %(take_profit)s,
            %(lot_size)s, %(risk_pct)s,
            %(r_multiple)s, %(pnl)s, %(pnl_pct)s,
            %(status)s, %(confluence_score)s, %(signal_tier)s,
            %(atr)s, %(exit_reason)s,
            %(run_id)s, %(strategy_tag)s
        )
        RETURNING id
    """

    _INSERT_MARKET_CONTEXT_SQL = """
        INSERT INTO market_context (
            trade_id, timestamp, instrument,
            cloud_direction_4h, cloud_direction_1h,
            tk_cross_15m, chikou_confirmation, cloud_thickness_4h,
            adx_value, atr_value, rsi_value, bb_width_percentile,
            session,
            nearest_sr_distance, zone_confluence_score,
            context_embedding
        ) VALUES (
            %(trade_id)s, %(timestamp)s, %(instrument)s,
            %(cloud_direction_4h)s, %(cloud_direction_1h)s,
            %(tk_cross_15m)s, %(chikou_confirmation)s, %(cloud_thickness_4h)s,
            %(adx_value)s, %(atr_value)s, %(rsi_value)s, %(bb_width_percentile)s,
            %(session)s,
            %(nearest_sr_distance)s, %(zone_confluence_score)s,
            %(context_embedding)s::vector
        )
        RETURNING id
    """

    _INSERT_PATTERN_SIGNATURE_SQL = """
        INSERT INTO pattern_signatures (
            context_id, trade_id,
            embedding,
            outcome_r, win
        ) VALUES (
            %(context_id)s, %(trade_id)s,
            %(embedding)s::vector,
            %(outcome_r)s, %(win)s
        )
        ON CONFLICT DO NOTHING
    """

    # Query to find runs with similar average market-context embeddings
    _SIMILAR_CONFIGS_SQL = """
        SELECT
            t.run_id,
            t.strategy_tag,
            COUNT(t.id)                                         AS trade_count,
            AVG(CASE WHEN t.r_multiple > 0 THEN 1.0 ELSE 0.0 END) AS win_rate,
            AVG(t.r_multiple)                                   AS avg_r,
            AVG(1.0 - (mc.context_embedding <=> %(embedding)s::vector)) AS similarity
        FROM trades t
        JOIN market_context mc ON mc.trade_id = t.id
        WHERE t.run_id IS NOT NULL
          AND mc.context_embedding IS NOT NULL
        GROUP BY t.run_id, t.strategy_tag
        ORDER BY similarity DESC
        LIMIT %(top_k)s
    """

    _RUN_HISTORY_SQL = """
        SELECT
            run_id,
            strategy_tag,
            COUNT(id)                                         AS trade_count,
            AVG(CASE WHEN r_multiple > 0 THEN 1.0 ELSE 0.0 END) AS win_rate,
            AVG(r_multiple)                                   AS avg_r,
            SUM(pnl)                                          AS total_pnl,
            MIN(created_at)                                   AS created_at
        FROM trades
        WHERE run_id IS NOT NULL
          AND status = 'closed'
        GROUP BY run_id, strategy_tag
        ORDER BY MIN(created_at) DESC
        LIMIT %(last_n)s
    """

    def __init__(
        self,
        db_pool: Optional[Any] = None,
        embedding_engine: Optional[Any] = None,
    ) -> None:
        self._db_pool = db_pool
        self._embedding_engine = embedding_engine

        if db_pool is None:
            logger.warning(
                "TradePersistence: no db_pool supplied — "
                "persistence calls will be silently skipped"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def persist_run(
        self,
        run_id: str,
        trades: List[Dict[str, Any]],
        config_snapshot: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Persist all trades from a completed backtest run.

        Writes to ``trades``, ``market_context``, and ``pattern_signatures``
        inside a single transaction.  Returns the number of trades persisted.

        Parameters
        ----------
        run_id:
            Unique identifier for this backtest run (e.g. "run_20260329_001").
        trades:
            List of trade dicts.  Each dict should contain at minimum:
            ``instrument``, ``direction``, ``entry_price``, ``exit_price``,
            ``entry_time``, ``exit_time``, ``r_multiple``, ``pnl``.
        config_snapshot:
            Strategy configuration dict at the time of the run.  Stored as
            the ``strategy_tag`` field (serialised key).  Optional.
        metrics:
            Aggregate performance metrics dict.  Currently unused for storage
            but accepted for forward-compatibility.

        Returns
        -------
        int — number of trades successfully persisted (0 when DB unavailable).
        """
        if self._db_pool is None:
            logger.warning(
                "TradePersistence.persist_run: db_pool not available — skipping run %s",
                run_id,
            )
            return 0

        if not trades:
            logger.info("TradePersistence.persist_run: no trades to persist for run %s", run_id)
            return 0

        strategy_tag = self._extract_strategy_tag(config_snapshot)
        engine = self._get_embedding_engine()
        persisted = 0

        try:
            with self._db_pool.get_connection() as conn:
                with conn.cursor() as cur:
                    for trade in trades:
                        trade_id = self._insert_trade(
                            cur, trade, run_id, strategy_tag
                        )
                        if trade_id is None:
                            continue

                        context_embedding = self._build_embedding(engine, trade)
                        context_id = self._insert_market_context(
                            cur, trade, trade_id, context_embedding
                        )

                        if context_id is not None:
                            self._insert_pattern_signature(
                                cur, trade, trade_id, context_id, context_embedding
                            )

                        persisted += 1

        except Exception as exc:
            logger.error(
                "TradePersistence.persist_run: DB error while persisting run %s: %s",
                run_id,
                exc,
            )
            return persisted

        logger.info(
            "TradePersistence.persist_run: persisted %d/%d trades for run %s",
            persisted,
            len(trades),
            run_id,
        )
        return persisted

    def get_similar_past_configs(
        self,
        config_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[SimilarConfig]:
        """Find past runs whose market conditions are most similar to the query.

        Uses pgvector cosine distance on ``market_context.context_embedding``
        aggregated per run.

        Parameters
        ----------
        config_embedding:
            64-dim embedding representing the current market context.
        top_k:
            Maximum number of similar past configs to return.

        Returns
        -------
        List of ``SimilarConfig`` objects ordered by descending similarity.
        Empty list when the DB is unavailable.
        """
        if self._db_pool is None:
            logger.warning(
                "TradePersistence.get_similar_past_configs: no db_pool — returning empty"
            )
            return []

        vector_str = self._to_pgvector_literal(config_embedding)

        try:
            with self._db_pool.get_cursor() as cur:
                cur.execute(
                    self._SIMILAR_CONFIGS_SQL,
                    {"embedding": vector_str, "top_k": top_k},
                )
                rows = cur.fetchall()
        except Exception as exc:
            logger.error(
                "TradePersistence.get_similar_past_configs: query failed: %s", exc
            )
            return []

        results: List[SimilarConfig] = []
        for row in rows:
            run_id = row.get("run_id")
            if run_id is None:
                continue
            results.append(
                SimilarConfig(
                    run_id=str(run_id),
                    strategy_tag=row.get("strategy_tag"),
                    similarity=float(row.get("similarity") or 0.0),
                    win_rate=float(row.get("win_rate") or 0.0),
                    avg_r=float(row.get("avg_r") or 0.0),
                    trade_count=int(row.get("trade_count") or 0),
                )
            )
        return results

    def get_run_history(self, last_n: int = 10) -> List[RunSummary]:
        """Return summary metrics for the most recent backtest runs.

        Parameters
        ----------
        last_n:
            Number of recent runs to return.

        Returns
        -------
        List of ``RunSummary`` objects ordered by most-recent first.
        Empty list when the DB is unavailable.
        """
        if self._db_pool is None:
            logger.warning(
                "TradePersistence.get_run_history: no db_pool — returning empty"
            )
            return []

        try:
            with self._db_pool.get_cursor() as cur:
                cur.execute(self._RUN_HISTORY_SQL, {"last_n": last_n})
                rows = cur.fetchall()
        except Exception as exc:
            logger.error("TradePersistence.get_run_history: query failed: %s", exc)
            return []

        results: List[RunSummary] = []
        for row in rows:
            run_id = row.get("run_id")
            if run_id is None:
                continue
            results.append(
                RunSummary(
                    run_id=str(run_id),
                    strategy_tag=row.get("strategy_tag"),
                    trade_count=int(row.get("trade_count") or 0),
                    win_rate=float(row.get("win_rate") or 0.0),
                    avg_r=float(row.get("avg_r") or 0.0),
                    total_pnl=float(row.get("total_pnl") or 0.0),
                    created_at=row.get("created_at"),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_embedding_engine(self) -> Any:
        """Return the embedding engine, creating a default one if needed."""
        if self._embedding_engine is None:
            from src.learning.embeddings import EmbeddingEngine  # lazy import
            self._embedding_engine = EmbeddingEngine()
        return self._embedding_engine

    @staticmethod
    def _extract_strategy_tag(config_snapshot: Optional[Dict[str, Any]]) -> Optional[str]:
        """Derive a short strategy tag string from the config snapshot."""
        if config_snapshot is None:
            return None
        # Prefer an explicit 'strategy' key; fall back to a short JSON summary
        tag = config_snapshot.get("strategy") or config_snapshot.get("strategy_tag")
        if tag:
            return str(tag)[:100]
        # Build a compact key from all top-level string/numeric values
        short = json.dumps(
            {k: v for k, v in config_snapshot.items() if isinstance(v, (str, int, float, bool))},
            sort_keys=True,
        )
        return short[:100]

    @staticmethod
    def _build_embedding(engine: Any, trade: Dict[str, Any]) -> np.ndarray:
        """Create a 64-dim embedding from the trade's market context fields."""
        context = {
            "cloud_direction_4h": trade.get("cloud_direction_4h", 0),
            "cloud_direction_1h": trade.get("cloud_direction_1h", 0),
            "tk_cross_15m": trade.get("tk_cross_15m", 0),
            "chikou_confirmation": trade.get("chikou_confirmation", False),
            "cloud_thickness_4h": trade.get("cloud_thickness_4h", 0.0),
            "adx_value": trade.get("adx_value", 0.0),
            "atr_value": trade.get("atr", trade.get("atr_value", 0.0)),
            "rsi_value": trade.get("rsi_value", 50.0),
            "bb_width_percentile": trade.get("bb_width_percentile", 0.5),
            "session": trade.get("session", ""),
            "nearest_sr_distance": trade.get("nearest_sr_distance", 0.0),
            "zone_confluence_score": trade.get("zone_confluence_score", 0),
            "confluence_score": trade.get("confluence_score", 0),
        }
        return engine.create_embedding(context)

    @staticmethod
    def _to_pgvector_literal(embedding: np.ndarray) -> str:
        """Serialise a numpy array to the pgvector wire format '[v1,v2,...]'."""
        values = ",".join(f"{v:.8f}" for v in embedding.tolist())
        return f"[{values}]"

    def _insert_trade(
        self,
        cur: Any,
        trade: Dict[str, Any],
        run_id: str,
        strategy_tag: Optional[str],
    ) -> Optional[int]:
        """Insert one trade row and return its new id, or None on failure."""
        entry_time = trade.get("entry_time")
        exit_time = trade.get("exit_time")

        # Normalise bare date strings to datetime objects
        if isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time)
        if isinstance(exit_time, str):
            exit_time = datetime.fromisoformat(exit_time)

        params = {
            "instrument":       trade.get("instrument", "XAU/USD"),
            "source":           trade.get("source", "backtest"),
            "direction":        trade.get("direction", "long"),
            "entry_time":       entry_time or datetime.now(timezone.utc),
            "exit_time":        exit_time,
            "entry_price":      float(trade.get("entry_price") or 0.0),
            "exit_price":       _opt_float(trade.get("exit_price")),
            "stop_loss":        float(trade.get("stop_loss") or 0.0),
            "take_profit":      _opt_float(trade.get("take_profit")),
            "lot_size":         float(trade.get("lot_size") or 0.01),
            "risk_pct":         float(trade.get("risk_pct") or 1.0),
            "r_multiple":       _opt_float(trade.get("r_multiple")),
            "pnl":              _opt_float(trade.get("pnl")),
            "pnl_pct":          _opt_float(trade.get("pnl_pct")),
            "status":           trade.get("status", "closed"),
            "confluence_score": trade.get("confluence_score"),
            "signal_tier":      trade.get("signal_tier"),
            "atr":              _opt_float(trade.get("atr")),
            "exit_reason":      trade.get("exit_reason"),
            "run_id":           run_id,
            "strategy_tag":     strategy_tag,
        }

        try:
            cur.execute(self._INSERT_TRADE_SQL, params)
            row = cur.fetchone()
            if row is None:
                return None
            # RealDictCursor returns a dict; plain cursor returns a tuple
            if hasattr(row, "__getitem__"):
                try:
                    return int(row["id"])
                except (KeyError, TypeError):
                    return int(row[0])
            return int(row[0])
        except Exception as exc:
            logger.error("TradePersistence._insert_trade: failed: %s", exc)
            return None

    def _insert_market_context(
        self,
        cur: Any,
        trade: Dict[str, Any],
        trade_id: int,
        context_embedding: np.ndarray,
    ) -> Optional[int]:
        """Insert a market_context row for the trade and return its id."""
        entry_time = trade.get("entry_time")
        if isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time)

        vector_str = self._to_pgvector_literal(context_embedding)

        params = {
            "trade_id":              trade_id,
            "timestamp":             entry_time or datetime.now(timezone.utc),
            "instrument":            trade.get("instrument", "XAU/USD"),
            "cloud_direction_4h":    trade.get("cloud_direction_4h"),
            "cloud_direction_1h":    trade.get("cloud_direction_1h"),
            "tk_cross_15m":          trade.get("tk_cross_15m"),
            "chikou_confirmation":   trade.get("chikou_confirmation"),
            "cloud_thickness_4h":    _opt_float(trade.get("cloud_thickness_4h")),
            "adx_value":             _opt_float(trade.get("adx_value")),
            "atr_value":             _opt_float(trade.get("atr", trade.get("atr_value"))),
            "rsi_value":             _opt_float(trade.get("rsi_value")),
            "bb_width_percentile":   _opt_float(trade.get("bb_width_percentile")),
            "session":               trade.get("session"),
            "nearest_sr_distance":   _opt_float(trade.get("nearest_sr_distance")),
            "zone_confluence_score": trade.get("zone_confluence_score"),
            "context_embedding":     vector_str,
        }

        try:
            cur.execute(self._INSERT_MARKET_CONTEXT_SQL, params)
            row = cur.fetchone()
            if row is None:
                return None
            if hasattr(row, "__getitem__"):
                try:
                    return int(row["id"])
                except (KeyError, TypeError):
                    return int(row[0])
            return int(row[0])
        except Exception as exc:
            logger.error("TradePersistence._insert_market_context: failed: %s", exc)
            return None

    def _insert_pattern_signature(
        self,
        cur: Any,
        trade: Dict[str, Any],
        trade_id: int,
        context_id: int,
        context_embedding: np.ndarray,
    ) -> None:
        """Insert a pattern_signature row for the trade."""
        r_multiple = _opt_float(trade.get("r_multiple"))
        win = bool(r_multiple > 0) if r_multiple is not None else None
        vector_str = self._to_pgvector_literal(context_embedding)

        params = {
            "context_id": context_id,
            "trade_id":   trade_id,
            "embedding":  vector_str,
            "outcome_r":  r_multiple,
            "win":        win,
        }

        try:
            cur.execute(self._INSERT_PATTERN_SIGNATURE_SQL, params)
        except Exception as exc:
            logger.error("TradePersistence._insert_pattern_signature: failed: %s", exc)


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _opt_float(value: Any) -> Optional[float]:
    """Cast to float, preserving None."""
    return float(value) if value is not None else None
