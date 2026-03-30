"""Tests for strategy_tag in embedding layer."""
from __future__ import annotations

import numpy as np
import pytest

from src.learning.embeddings import EmbeddingEngine


class TestEmbeddingStrategyTag:
    def test_create_embedding_with_strategy_tag(self):
        engine = EmbeddingEngine()
        context = {"session": "london", "adx": 35.0, "atr": 3.0}
        embedding = engine.create_embedding(context, strategy_tag="ichimoku")
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 64

    def test_create_embedding_without_strategy_tag(self):
        engine = EmbeddingEngine()
        context = {"session": "london", "adx": 35.0}
        embedding = engine.create_embedding(context)
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 64

    def test_embed_trade_includes_strategy_tag(self):
        engine = EmbeddingEngine()
        context = {"session": "london", "adx": 35.0, "atr": 3.0}
        result = engine.embed_trade(context, strategy_tag="ichimoku")
        assert result.get("strategy_tag") == "ichimoku"

    def test_embed_trade_no_strategy_tag_default(self):
        engine = EmbeddingEngine()
        context = {"session": "london", "adx": 35.0}
        result = engine.embed_trade(context)
        assert result.get("strategy_tag") is None
