"""Base abstractions for the strategy layer.

Defines the ``Strategy`` abstract base class and the global
``STRATEGY_REGISTRY`` that maps strategy keys to their implementation classes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional, Type


class Strategy(ABC):
    """Abstract base class for all trading strategies.

    Subclasses must implement :meth:`generate_signal` and may optionally
    set :attr:`config_model` to a Pydantic model class for typed configuration.

    Registration
    ------------
    Strategies are registered in :data:`STRATEGY_REGISTRY` under a unique key.
    Use :func:`register_strategy` to add a new strategy.

    Example::

        @register_strategy("my_strategy")
        class MyStrategy(Strategy):
            ...
    """

    #: Optional Pydantic model class for per-strategy configuration.
    config_model: ClassVar[Optional[Type[Any]]] = None

    @abstractmethod
    def generate_signal(self, *args: Any, **kwargs: Any) -> Any:
        """Generate a trading signal given market data inputs."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# Global strategy registry
# ---------------------------------------------------------------------------

#: Maps strategy key (str) → Strategy subclass.
STRATEGY_REGISTRY: dict[str, Type[Strategy]] = {}


def register_strategy(key: str):
    """Class decorator that registers a Strategy subclass under *key*.

    Parameters
    ----------
    key:
        Unique identifier for the strategy (must match ``active_strategy`` in
        the YAML config).

    Raises
    ------
    ValueError
        If *key* is already registered to a different class.
    """
    def decorator(cls: Type[Strategy]) -> Type[Strategy]:
        if key in STRATEGY_REGISTRY and STRATEGY_REGISTRY[key] is not cls:
            raise ValueError(
                f"Strategy key '{key}' is already registered to "
                f"{STRATEGY_REGISTRY[key].__name__}. "
                f"Cannot re-register to {cls.__name__}."
            )
        STRATEGY_REGISTRY[key] = cls
        return cls

    return decorator
