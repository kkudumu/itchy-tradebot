"""
Trade context embedding, similarity search, and adaptive learning.

Modules
-------
feature_vector   — Build 64-dimensional feature vectors from trade context dicts.
embeddings       — Create and manage embeddings; inline during backtest logging.
similarity       — pgvector-backed cosine similarity search and performance stats.
stats_analyzer   — Statistical win-rate breakdowns by session, regime, confluence.
edge_reviewer    — Per-edge performance review and advisory suggestions.
adaptive_engine  — Self-learning engine (mechanical → statistical → similarity).
report_generator — Weekly performance report builder (text and HTML).
"""

from .feature_vector import FeatureVectorBuilder
from .embeddings import EmbeddingEngine
from .similarity import SimilaritySearch, SimilarTrade, PerformanceStats
from .stats_analyzer import StatsAnalyzer
from .edge_reviewer import EdgeReviewer, EdgeReviewResult, EdgeSuggestion
from .adaptive_engine import AdaptiveLearningEngine, PreTradeInsight
from .report_generator import ReportGenerator, WeeklyReport

__all__ = [
    "FeatureVectorBuilder",
    "EmbeddingEngine",
    "SimilaritySearch",
    "SimilarTrade",
    "PerformanceStats",
    "StatsAnalyzer",
    "EdgeReviewer",
    "EdgeReviewResult",
    "EdgeSuggestion",
    "AdaptiveLearningEngine",
    "PreTradeInsight",
    "ReportGenerator",
    "WeeklyReport",
]
