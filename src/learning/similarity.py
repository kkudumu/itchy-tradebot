"""
Similarity search over stored trade embeddings using pgvector.

SimilaritySearch uses the cosine distance operator ``<=>`` via the HNSW index
on market_context.context_embedding to find the k most similar historical
trades for a given embedding, then computes aggregated performance statistics.

Confidence ramp-up
------------------
Data confidence grows with the number of matching trades:

    confidence = min(1.0, n_similar / 20)

- 0 trades  → confidence 0.0  (no evidence)
- 10 trades → confidence 0.5  (early signals)
- 20 trades → confidence 1.0  (baseline reliability)
- 100+      → confidence 1.0  (stable statistics)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Minimum number of trades for full confidence (ramp denominator)
_CONFIDENCE_RAMP_N: int = 20


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SimilarTrade:
    """A single historical trade that is contextually similar to the query."""

    trade_id: int
    similarity: float      # Cosine similarity in [0, 1]; 1 = identical
    r_multiple: float      # Outcome in R-multiples (positive = win)
    win: bool
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceStats:
    """Aggregated statistics over a set of similar trades."""

    win_rate: float        # Fraction of trades that were winners
    avg_r: float           # Mean R-multiple across all trades
    expectancy: float      # win_rate * avg_win_r - (1 - win_rate) * |avg_loss_r|
    n_trades: int          # Count of similar trades used
    confidence: float      # min(1.0, n_trades / 20)
    avg_win_r: float       # Mean R-multiple of winning trades (0 if none)
    avg_loss_r: float      # Mean R-multiple of losing trades (0 if none, negative)


# ---------------------------------------------------------------------------
# SimilaritySearch
# ---------------------------------------------------------------------------

class SimilaritySearch:
    """Search for similar trade contexts via pgvector cosine similarity.

    Parameters
    ----------
    db_pool:
        A ``DatabasePool`` instance (or any object with a ``get_cursor()``
        context manager method).  May be None for unit-test scenarios that
        mock the database layer.
    """

    # SQL template — uses pgvector's <=> (cosine distance) operator.
    # The HNSW index on context_embedding makes this O(log n).
    _SIMILARITY_SQL = """
        SELECT
            mc.trade_id,
            mc.cloud_direction_4h,
            mc.cloud_direction_1h,
            mc.tk_cross_15m,
            mc.session,
            mc.adx_value,
            mc.atr_value,
            mc.rsi_value,
            mc.nearest_sr_distance,
            mc.zone_confluence_score,
            t.r_multiple,
            t.pnl,
            t.direction,
            t.confluence_score,
            t.signal_tier,
            1.0 - (mc.context_embedding <=> %s::vector) AS similarity
        FROM market_context mc
        JOIN trades t ON mc.trade_id = t.id
        WHERE t.status = 'closed'
          AND mc.context_embedding IS NOT NULL
          AND 1.0 - (mc.context_embedding <=> %s::vector) >= %s
        ORDER BY mc.context_embedding <=> %s::vector
        LIMIT %s
    """

    _SIMILARITY_SQL_FILTERED = """
        SELECT
            mc.trade_id,
            mc.cloud_direction_4h,
            mc.cloud_direction_1h,
            mc.tk_cross_15m,
            mc.session,
            mc.adx_value,
            mc.atr_value,
            mc.rsi_value,
            mc.nearest_sr_distance,
            mc.zone_confluence_score,
            t.r_multiple,
            t.pnl,
            t.direction,
            t.confluence_score,
            t.signal_tier,
            1.0 - (mc.context_embedding <=> %s::vector) AS similarity
        FROM market_context mc
        JOIN trades t ON mc.trade_id = t.id
        WHERE t.status = 'closed'
          AND t.source = %s
          AND mc.context_embedding IS NOT NULL
          AND 1.0 - (mc.context_embedding <=> %s::vector) >= %s
        ORDER BY mc.context_embedding <=> %s::vector
        LIMIT %s
    """

    _SIMILARITY_SQL_STRATEGY = """
        SELECT
            mc.trade_id,
            mc.cloud_direction_4h,
            mc.cloud_direction_1h,
            mc.tk_cross_15m,
            mc.session,
            mc.adx_value,
            mc.atr_value,
            mc.rsi_value,
            mc.nearest_sr_distance,
            mc.zone_confluence_score,
            t.r_multiple,
            t.pnl,
            t.direction,
            t.confluence_score,
            t.signal_tier,
            1.0 - (mc.context_embedding <=> %s::vector) AS similarity
        FROM market_context mc
        JOIN trades t ON mc.trade_id = t.id
        WHERE t.status = 'closed'
          AND mc.strategy_tag = %s
          AND mc.context_embedding IS NOT NULL
          AND 1.0 - (mc.context_embedding <=> %s::vector) >= %s
        ORDER BY mc.context_embedding <=> %s::vector
        LIMIT %s
    """

    def __init__(self, db_pool=None) -> None:
        self.db_pool = db_pool

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_similar_trades(
        self,
        context_embedding: np.ndarray,
        k: int = 10,
        min_similarity: float = 0.7,
        source_filter: Optional[str] = None,
        strategy_filter: Optional[str] = None,
    ) -> List[SimilarTrade]:
        """Find the k most similar historical trades via cosine similarity.

        Parameters
        ----------
        context_embedding:
            64-dim query vector produced by EmbeddingEngine.
        k:
            Maximum number of similar trades to return.
        min_similarity:
            Minimum cosine similarity threshold [0, 1].
        source_filter:
            Optional trade source ("backtest", "live", "paper") to limit
            the search scope.  None searches all sources.
        strategy_filter:
            Optional strategy identifier (e.g., 'ichimoku') to limit
            results to trades tagged with that strategy.  None searches
            all strategies.  Requires the ``strategy_tag`` column to
            exist on ``market_context``; the parameter is additive and
            backward-compatible (legacy trades without a tag are excluded
            when a filter is specified).

        Returns
        -------
        List of SimilarTrade objects ordered by descending similarity.
        An empty list is returned when the database is unavailable.
        """
        if self.db_pool is None:
            logger.warning("SimilaritySearch: no db_pool configured — returning empty results")
            return []

        vector_str = self._to_pgvector_literal(context_embedding)

        try:
            with self.db_pool.get_cursor() as cur:
                if source_filter:
                    cur.execute(
                        self._SIMILARITY_SQL_FILTERED,
                        (vector_str, source_filter, vector_str, min_similarity, vector_str, k),
                    )
                elif strategy_filter is not None:
                    cur.execute(
                        self._SIMILARITY_SQL_STRATEGY,
                        (vector_str, strategy_filter, vector_str, min_similarity, vector_str, k),
                    )
                else:
                    cur.execute(
                        self._SIMILARITY_SQL,
                        (vector_str, vector_str, min_similarity, vector_str, k),
                    )
                rows = cur.fetchall()
        except Exception as exc:
            logger.error("SimilaritySearch query failed: %s", exc)
            return []

        results: List[SimilarTrade] = []
        for row in rows:
            trade_id = row.get("trade_id")
            if trade_id is None:
                continue

            r_mult = row.get("r_multiple")
            if r_mult is None:
                r_mult = 0.0
            else:
                r_mult = float(r_mult)

            similarity = float(row.get("similarity") or 0.0)

            # Derive win from r_multiple when no explicit column is present
            win = r_mult > 0.0

            context = {
                k: row.get(k)
                for k in (
                    "cloud_direction_4h",
                    "cloud_direction_1h",
                    "tk_cross_15m",
                    "session",
                    "adx_value",
                    "atr_value",
                    "rsi_value",
                    "nearest_sr_distance",
                    "zone_confluence_score",
                    "direction",
                    "confluence_score",
                    "signal_tier",
                )
            }

            results.append(
                SimilarTrade(
                    trade_id=int(trade_id),
                    similarity=similarity,
                    r_multiple=r_mult,
                    win=win,
                    context=context,
                )
            )

        return results

    def get_performance_stats(
        self,
        similar_trades: List[SimilarTrade],
    ) -> PerformanceStats:
        """Calculate performance statistics from a list of similar trades.

        Parameters
        ----------
        similar_trades:
            Output from :meth:`find_similar_trades`.

        Returns
        -------
        PerformanceStats with win_rate, avg_r, expectancy, n_trades,
        confidence, avg_win_r, and avg_loss_r.
        """
        n = len(similar_trades)
        confidence = self.get_confidence(n)

        if n == 0:
            return PerformanceStats(
                win_rate=0.0,
                avg_r=0.0,
                expectancy=0.0,
                n_trades=0,
                confidence=0.0,
                avg_win_r=0.0,
                avg_loss_r=0.0,
            )

        wins  = [t for t in similar_trades if t.win]
        losses = [t for t in similar_trades if not t.win]

        win_rate  = len(wins) / n
        avg_r     = sum(t.r_multiple for t in similar_trades) / n
        avg_win_r = sum(t.r_multiple for t in wins) / len(wins) if wins else 0.0
        avg_loss_r = sum(t.r_multiple for t in losses) / len(losses) if losses else 0.0

        # Expectancy: expected R per trade
        # = win_rate * avg_win - loss_rate * |avg_loss|
        loss_rate = 1.0 - win_rate
        expectancy = win_rate * avg_win_r + loss_rate * avg_loss_r

        return PerformanceStats(
            win_rate=win_rate,
            avg_r=avg_r,
            expectancy=expectancy,
            n_trades=n,
            confidence=confidence,
            avg_win_r=avg_win_r,
            avg_loss_r=avg_loss_r,
        )

    @staticmethod
    def get_confidence(n_similar: int) -> float:
        """Compute data confidence from the number of similar trades found.

        Ramp-up:  confidence = min(1.0, n_similar / 20)

        - 0 trades  → 0.0  (no signal)
        - 10 trades → 0.5
        - 20 trades → 1.0  (full confidence)
        - 100+      → 1.0

        Parameters
        ----------
        n_similar:
            Number of similar trades retrieved.

        Returns
        -------
        float in [0, 1].
        """
        if n_similar <= 0:
            return 0.0
        return min(1.0, n_similar / _CONFIDENCE_RAMP_N)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_pgvector_literal(embedding: np.ndarray) -> str:
        """Serialise a numpy array to the pgvector wire format '[v1,v2,...]'."""
        values = ",".join(f"{v:.8f}" for v in embedding.tolist())
        return f"[{values}]"

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Pure-Python cosine similarity for testing and offline use.

        Parameters
        ----------
        a, b:
            1-D arrays of the same length.

        Returns
        -------
        float in [-1, 1]; 1.0 for identical non-zero vectors.
        """
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
