"""Tests for config refactor (Task 5)."""
from __future__ import annotations

import pytest
from pathlib import Path
from src.config.loader import load_config, ConfigLoader
from src.config.models import StrategyConfig, IchimokuConfig, ADXConfig, ATRConfig


class TestStrategyYAML:
    def test_load_strategy_yaml(self):
        cfg = load_config()
        assert cfg.strategy.active_strategy == "ichimoku"
        assert "ichimoku" in cfg.strategy.strategies

    def test_backward_compat_ichimoku(self):
        cfg = load_config()
        assert cfg.strategy.ichimoku.tenkan_period == 9
        assert cfg.strategy.ichimoku.kijun_period == 26
        assert cfg.strategy.ichimoku.senkou_b_period == 52

    def test_backward_compat_adx(self):
        cfg = load_config()
        assert cfg.strategy.adx.period == 14
        assert cfg.strategy.adx.threshold == 28

    def test_backward_compat_atr(self):
        cfg = load_config()
        assert cfg.strategy.atr.period == 14
        assert cfg.strategy.atr.stop_multiplier == 1.5

    def test_risk_config_unchanged(self):
        cfg = load_config()
        assert cfg.strategy.risk.initial_risk_pct == 1.5

    def test_exit_config_unchanged(self):
        cfg = load_config()
        assert cfg.strategy.exit.tp_r_multiple == 2.0


class TestStrategyConfig:
    def test_default_strategy_config(self):
        cfg = StrategyConfig()
        assert cfg.active_strategy == "ichimoku"
        assert isinstance(cfg.strategies, dict)

    def test_ichimoku_property(self):
        cfg = StrategyConfig()
        assert isinstance(cfg.ichimoku, IchimokuConfig)
        assert cfg.ichimoku.tenkan_period == 9


class TestStrategyLoader:
    def test_loader_creation(self):
        from src.strategy.loader import StrategyLoader
        cfg = StrategyConfig()
        loader = StrategyLoader(cfg)
        assert loader.active_strategy_key == "ichimoku"

    def test_loader_unknown_strategy(self):
        from src.strategy.loader import StrategyLoader
        cfg = StrategyConfig(active_strategy="nonexistent")
        loader = StrategyLoader(cfg)
        with pytest.raises(ValueError, match="not found in registry"):
            loader.load()
