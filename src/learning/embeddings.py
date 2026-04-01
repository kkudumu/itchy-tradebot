"""
Embedding creation and management for trade contexts.

EmbeddingEngine wraps FeatureVectorBuilder and provides:
- Single-context embedding creation
- Trade embedding dict ready for DB insertion
- Batch embedding for multiple contexts

The 64-dimensional vectors produced here are stored in
market_context.context_embedding (VECTOR(64)) and
pattern_signatures.embedding (VECTOR(64)) on the database side.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .feature_vector import FeatureVectorBuilder

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Create and manage 64-dimensional trade context embeddings.

    Parameters
    ----------
    feature_builder:
        Optional pre-configured FeatureVectorBuilder.  A default instance
        is created when None is supplied.
    """

    def __init__(self, feature_builder: Optional[FeatureVectorBuilder] = None) -> None:
        self.feature_builder: FeatureVectorBuilder = (
            feature_builder if feature_builder is not None else FeatureVectorBuilder()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_embedding(
        self,
        context: Dict[str, Any],
        strategy_tag: Optional[str] = None,
    ) -> np.ndarray:
        """Create a 64-dim embedding from a trade context dictionary.

        Parameters
        ----------
        context:
            Market / trade context dictionary.  Missing keys default to 0.
        strategy_tag:
            Optional strategy identifier (e.g., 'ichimoku'). Included in
            context metadata but NOT in the embedding vector itself.

        Returns
        -------
        np.ndarray of shape (64,), dtype float64, values in [0, 1].
        """
        return self.feature_builder.build(context)

    def embed_trade(
        self,
        trade_context: Dict[str, Any],
        trade_result: Optional[Dict[str, Any]] = None,
        strategy_tag: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create an embedding dict ready for database insertion.

        Parameters
        ----------
        trade_context:
            Market / signal context at the time of trade entry.
        trade_result:
            Optional dict containing trade outcome fields:
            - ``r_multiple`` (float): R-multiple result (positive = win)
            - ``pnl`` (float): Profit/loss in account currency
            - ``win`` (bool): Whether the trade closed profitably
        strategy_tag:
            Optional strategy identifier (e.g., 'ichimoku'). Stored as
            metadata alongside the embedding; not part of the vector.

        Returns
        -------
        dict with:
        - ``context_embedding`` (list[float], 64 elements)
        - ``outcome_r`` (float | None)
        - ``win`` (bool | None)
        - ``strategy_tag`` (str | None)
        """
        embedding = self.feature_builder.build(trade_context)

        result: Dict[str, Any] = {
            "context_embedding": embedding.tolist(),
            "outcome_r": None,
            "win": None,
            "strategy_tag": strategy_tag,
        }

        if trade_result:
            r_mult = trade_result.get("r_multiple")
            win_flag = trade_result.get("win")

            if r_mult is not None:
                result["outcome_r"] = float(r_mult)
                # Derive win flag from r_multiple when not explicitly provided
                if win_flag is None:
                    result["win"] = float(r_mult) > 0.0
                else:
                    result["win"] = bool(win_flag)
            elif win_flag is not None:
                result["win"] = bool(win_flag)

        return result

    def batch_embed(self, contexts: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Embed multiple contexts and return a list of arrays.

        Each context is processed independently, so the output order
        matches the input order.

        Parameters
        ----------
        contexts:
            List of context dicts.

        Returns
        -------
        List of np.ndarray, each of shape (64,).
        """
        return [self.feature_builder.build(ctx) for ctx in contexts]

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def embedding_to_list(self, embedding: np.ndarray) -> List[float]:
        """Convert a numpy embedding to a plain Python list for serialisation."""
        return embedding.tolist()

    def embedding_from_list(self, values: List[float]) -> np.ndarray:
        """Reconstruct a numpy array from a stored list of floats."""
        arr = np.array(values, dtype=np.float64)
        if arr.shape != (FeatureVectorBuilder.VECTOR_DIM,):
            raise ValueError(
                f"Expected {FeatureVectorBuilder.VECTOR_DIM} values, got {len(values)}"
            )
        return arr


# ---------------------------------------------------------------------------
# SSS convenience function
# ---------------------------------------------------------------------------

def create_sss_embedding(context: Dict[str, Any]) -> np.ndarray:
    """Create a 64-dim embedding for an SSS trade setup context.

    Uses :class:`SSSSequenceMLLayer` feature extraction so that SSS-specific
    keys (sequence_state, layer, f2_count_recent, etc.) are mapped correctly,
    keeping them separate from the Ichimoku feature layout.

    Parameters
    ----------
    context:
        SSS trade setup dictionary.

    Returns
    -------
    np.ndarray of shape (64,), dtype float64, values in [0, 1].
    """
    # Lazy import to avoid circular dependency at module load time.
    from src.strategy.strategies.sss.ml_sequence import SSSSequenceMLLayer  # noqa: PLC0415

    layer = SSSSequenceMLLayer(similarity_store=None)
    return layer.create_features(context)
