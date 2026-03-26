"""
Trade context embedding and similarity search.

Modules
-------
feature_vector  — Build 64-dimensional feature vectors from trade context dicts.
embeddings      — Create and manage embeddings; inline during backtest logging.
similarity      — pgvector-backed cosine similarity search and performance stats.
"""

from .feature_vector import FeatureVectorBuilder
from .embeddings import EmbeddingEngine
from .similarity import SimilaritySearch, SimilarTrade, PerformanceStats

__all__ = [
    "FeatureVectorBuilder",
    "EmbeddingEngine",
    "SimilaritySearch",
    "SimilarTrade",
    "PerformanceStats",
]
