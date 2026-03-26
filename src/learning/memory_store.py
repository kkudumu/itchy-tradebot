"""
In-memory similarity store for backtesting without a database.

Replaces pgvector-backed SimilaritySearch with a numpy array that
accumulates embeddings during a backtest run.  After enough trades
accumulate, cosine similarity queries return the k nearest neighbours
just like the DB-backed version — enabling the adaptive learning
engine to function identically in backtest and live modes.

Also provides InMemoryStatsAnalyzer, which replaces the DB-backed
StatsAnalyzer by computing win-rate breakdowns from an in-memory
trade list that grows as the backtest progresses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .similarity import SimilarTrade, PerformanceStats, SimilaritySearch

logger = logging.getLogger(__name__)

# Minimum number of stored trades before similarity queries are enabled.
# Below this count, find_similar_trades() returns an empty list.
_MIN_TRADES_FOR_SEARCH: int = 30

# ADX regime boundaries (mirrors stats_analyzer.py)
_ADX_LOW_MAX: float = 20.0
_ADX_MED_MAX: float = 35.0


def _adx_regime(adx: float) -> str:
    if adx < _ADX_LOW_MAX:
        return "low"
    if adx < _ADX_MED_MAX:
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# InMemorySimilarityStore
# ---------------------------------------------------------------------------

class InMemorySimilarityStore(SimilaritySearch):
    """Numpy-backed similarity search that accumulates embeddings in RAM.

    Drop-in replacement for SimilaritySearch — same public API, no database
    needed.  Embeddings are stored in a pre-allocated numpy matrix that grows
    as trades are recorded.

    Parameters
    ----------
    vector_dim:
        Dimensionality of embeddings.  Default: 64.
    initial_capacity:
        Pre-allocated rows in the embedding matrix.  Grows automatically.
    """

    def __init__(self, vector_dim: int = 64, initial_capacity: int = 2000) -> None:
        # Do NOT call super().__init__ with db_pool — we override everything
        self.db_pool = None  # satisfies any isinstance checks
        self._dim = vector_dim
        self._capacity = initial_capacity

        # Storage arrays
        self._embeddings = np.zeros((initial_capacity, vector_dim), dtype=np.float64)
        self._r_multiples = np.zeros(initial_capacity, dtype=np.float64)
        self._contexts: List[Dict[str, Any]] = []
        self._count = 0

    # ------------------------------------------------------------------
    # Recording trades
    # ------------------------------------------------------------------

    def record_trade(
        self,
        embedding: np.ndarray,
        r_multiple: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Store a completed trade's embedding and outcome.

        Parameters
        ----------
        embedding:
            64-dim numpy array (output of EmbeddingEngine.create_embedding).
        r_multiple:
            Trade outcome in R-multiples.
        context:
            Optional dict of trade metadata (session, adx_value, etc.).

        Returns
        -------
        Integer trade index (0-based).
        """
        if self._count >= self._capacity:
            self._grow()

        idx = self._count
        self._embeddings[idx] = embedding
        self._r_multiples[idx] = r_multiple
        self._contexts.append(context or {})
        self._count += 1
        return idx

    def _grow(self) -> None:
        """Double the capacity of internal storage arrays."""
        new_cap = self._capacity * 2
        new_emb = np.zeros((new_cap, self._dim), dtype=np.float64)
        new_emb[: self._capacity] = self._embeddings
        self._embeddings = new_emb

        new_r = np.zeros(new_cap, dtype=np.float64)
        new_r[: self._capacity] = self._r_multiples
        self._r_multiples = new_r

        self._capacity = new_cap
        logger.debug("InMemorySimilarityStore grew to capacity %d", new_cap)

    @property
    def trade_count(self) -> int:
        return self._count

    # ------------------------------------------------------------------
    # SimilaritySearch API override
    # ------------------------------------------------------------------

    def find_similar_trades(
        self,
        context_embedding: np.ndarray,
        k: int = 10,
        min_similarity: float = 0.7,
        source_filter: Optional[str] = None,
    ) -> List[SimilarTrade]:
        """Find the k most similar stored trades via cosine similarity.

        Returns an empty list when fewer than _MIN_TRADES_FOR_SEARCH
        trades have been recorded.
        """
        if self._count < _MIN_TRADES_FOR_SEARCH:
            return []

        # Compute cosine similarities against all stored embeddings
        active = self._embeddings[: self._count]
        norms = np.linalg.norm(active, axis=1)
        query_norm = np.linalg.norm(context_embedding)

        if query_norm == 0.0:
            return []

        # Avoid division by zero for any stored vector with zero norm
        safe_norms = np.where(norms > 0, norms, 1.0)
        similarities = active @ context_embedding / (safe_norms * query_norm)

        # Filter by min_similarity
        mask = similarities >= min_similarity
        valid_indices = np.where(mask)[0]

        if len(valid_indices) == 0:
            return []

        # Sort by similarity descending, take top k
        valid_sims = similarities[valid_indices]
        top_k_order = np.argsort(-valid_sims)[:k]
        top_indices = valid_indices[top_k_order]

        results: List[SimilarTrade] = []
        for idx in top_indices:
            r_mult = float(self._r_multiples[idx])
            results.append(
                SimilarTrade(
                    trade_id=int(idx),
                    similarity=float(similarities[idx]),
                    r_multiple=r_mult,
                    win=r_mult > 0.0,
                    context=self._contexts[idx] if idx < len(self._contexts) else {},
                )
            )

        return results

    # get_performance_stats and get_confidence are inherited from SimilaritySearch

    def clear(self) -> None:
        """Reset the store — useful between walk-forward windows."""
        self._embeddings[:] = 0.0
        self._r_multiples[:] = 0.0
        self._contexts.clear()
        self._count = 0


# ---------------------------------------------------------------------------
# InMemoryStatsAnalyzer
# ---------------------------------------------------------------------------

class InMemoryStatsAnalyzer:
    """In-memory trade statistics analyzer for backtesting.

    Replaces the DB-backed StatsAnalyzer by computing win-rate breakdowns
    from an in-memory trade list that grows as the backtest progresses.

    Uses the same public API as StatsAnalyzer so it can be injected into
    AdaptiveLearningEngine.
    """

    def __init__(self) -> None:
        self._trades: List[Dict[str, Any]] = []

    def record_trade(self, trade: Dict[str, Any]) -> None:
        """Add a completed trade to the stats pool.

        Expected keys: r_multiple, session, adx_value, confluence_score,
        day_of_week, direction, signal_tier.
        """
        self._trades.append(trade)

    @property
    def trade_count(self) -> int:
        return len(self._trades)

    def _to_df(self) -> pd.DataFrame:
        if not self._trades:
            return pd.DataFrame()
        return pd.DataFrame(self._trades)

    # ------------------------------------------------------------------
    # Win-rate breakdowns (same API as StatsAnalyzer)
    # ------------------------------------------------------------------

    def win_rate_by_session(self, min_trades: int = 20) -> Dict[str, dict]:
        df = self._to_df()
        if df.empty or "session" not in df.columns:
            return {}
        df = df.copy()
        df["win"] = df["r_multiple"].astype(float) > 0.0
        result: Dict[str, dict] = {}
        for session, group in df.groupby("session"):
            n = len(group)
            if n < min_trades:
                continue
            result[str(session)] = {
                "win_rate": round(float(group["win"].mean()), 4),
                "n_trades": n,
                "avg_r": round(float(group["r_multiple"].astype(float).mean()), 4),
            }
        return result

    def win_rate_by_regime(self, min_trades: int = 20) -> Dict[str, dict]:
        df = self._to_df()
        if df.empty or "adx_value" not in df.columns:
            return {}
        df = df.copy()
        df["adx_value"] = df["adx_value"].fillna(0.0).astype(float)
        df["regime"] = df["adx_value"].apply(_adx_regime)
        df["win"] = df["r_multiple"].astype(float) > 0.0
        result: Dict[str, dict] = {}
        for regime, group in df.groupby("regime"):
            n = len(group)
            if n < min_trades:
                continue
            result[str(regime)] = {
                "win_rate": round(float(group["win"].mean()), 4),
                "n_trades": n,
                "avg_r": round(float(group["r_multiple"].astype(float).mean()), 4),
            }
        return result

    # ------------------------------------------------------------------
    # Filter predicates (same API as StatsAnalyzer)
    # ------------------------------------------------------------------

    def should_filter_session(
        self, session: str, min_wr: float = 0.40, min_trades: int = 20
    ) -> bool:
        stats = self.win_rate_by_session(min_trades=min_trades)
        if session not in stats:
            return False
        return stats[session]["win_rate"] < min_wr

    def should_filter_regime(
        self, adx: float, min_wr: float = 0.40, min_trades: int = 20
    ) -> bool:
        regime = _adx_regime(adx)
        stats = self.win_rate_by_regime(min_trades=min_trades)
        if regime not in stats:
            return False
        return stats[regime]["win_rate"] < min_wr

    def clear(self) -> None:
        """Reset for a new walk-forward window."""
        self._trades.clear()
