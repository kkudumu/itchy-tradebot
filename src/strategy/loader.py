"""Strategy loader — reads config and instantiates the active strategy."""
from __future__ import annotations

from typing import Optional

from src.strategy.base import STRATEGY_REGISTRY, Strategy


class StrategyLoader:
    """Load and instantiate the active strategy from config.

    Usage::

        loader = StrategyLoader(config)
        strategy = loader.load()
    """

    def __init__(self, strategy_config) -> None:
        """
        Parameters
        ----------
        strategy_config:
            A StrategyConfig instance with active_strategy and strategies dict.
        """
        self._config = strategy_config

    def load(self) -> Strategy:
        """Instantiate and return the active strategy.

        Raises
        ------
        ValueError
            If active_strategy key not found in STRATEGY_REGISTRY.
        """
        key = self._config.active_strategy

        if key not in STRATEGY_REGISTRY:
            available = list(STRATEGY_REGISTRY.keys())
            raise ValueError(
                f"Strategy '{key}' not found in registry. "
                f"Available strategies: {available}"
            )

        strategy_cls = STRATEGY_REGISTRY[key]

        # Get per-strategy config section
        strategy_params = self._config.strategies.get(key, {})

        # If the strategy class has a config_model, validate the params
        if hasattr(strategy_cls, "config_model") and strategy_cls.config_model is not None:
            config_instance = strategy_cls.config_model(
                **strategy_params.get(key, strategy_params)
            )
            return strategy_cls(config=config_instance)

        return strategy_cls()

    @property
    def active_strategy_key(self) -> str:
        return self._config.active_strategy
