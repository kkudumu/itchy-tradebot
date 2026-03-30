"""Concrete strategy implementations."""
from .ichimoku import IchimokuStrategy
from .ema_pullback import EMAPullbackStrategy

__all__ = ['IchimokuStrategy', 'EMAPullbackStrategy']
