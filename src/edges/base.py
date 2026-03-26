"""
Base classes for the edge optimization pipeline.

EdgeContext carries all per-bar data needed by every filter.
EdgeResult is the standard return value: a (allowed, reason) pair.
EdgeFilter is the abstract base every edge must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class EdgeContext:
    """All market data and trade state available to each edge filter.

    Fields are populated by the trading engine before the edge pipeline
    is evaluated. Optional fields carry None when not yet relevant
    (e.g. active-trade fields are None for entry checks).
    """

    # Bar identity
    timestamp: datetime
    day_of_week: int        # 0 = Monday … 6 = Sunday

    # Price snapshot
    close_price: float
    high_price: float
    low_price: float

    # Market microstructure
    spread: float           # Points (broker-reported bid/ask spread)
    session: str            # 'london', 'new_york', 'overlap', 'asian', 'off_hours'

    # Volatility / trend strength
    adx: float
    atr: float

    # Cloud context
    cloud_thickness: float  # Absolute price distance between Senkou A and B
    kijun_value: float      # Current Kijun-sen level used for breakout confirmation

    # Derived indicators
    bb_squeeze: bool        # True when Bollinger Bands are in squeeze state
    confluence_score: int   # 0–8 raw score from ConfluenceScorer

    # Active trade state (None when evaluating entry edges)
    current_r: Optional[float] = None       # Unrealised profit in R multiples
    candles_since_entry: Optional[int] = None  # Bars elapsed since trade opened

    # Historical trade sequence (for equity curve filter)
    equity_curve: list = field(default_factory=list)  # List of trade P&L in R

    # Optional signal reference (full Signal object when available)
    signal: Optional[object] = None


@dataclass
class EdgeResult:
    """Standardised result returned by every edge filter.

    Attributes
    ----------
    allowed:
        True → this edge permits the action (entry, exit, or modifier applied).
    edge_name:
        The filter's registered name, used for logging and diagnostics.
    reason:
        Human-readable explanation of the decision.
    modifier:
        Optional float modifier value for MODIFIER-type edges (e.g. size multiplier).
        None for binary filters.
    """

    allowed: bool
    edge_name: str
    reason: str
    modifier: Optional[float] = None


class EdgeFilter(ABC):
    """Abstract base for all edge filters.

    Sub-classes must implement ``should_allow`` which receives the full
    ``EdgeContext`` and returns an ``EdgeResult``.

    Parameters
    ----------
    name:
        Unique identifier used in EdgeResult and log messages.
    config:
        Raw config dict for this edge (typically from models.py Edge* model).
    """

    def __init__(self, name: str, config: dict) -> None:
        self.name = name
        self.config = config
        self.enabled: bool = config.get("enabled", True)

    @abstractmethod
    def should_allow(self, context: EdgeContext) -> EdgeResult:
        """Evaluate the edge condition.

        Parameters
        ----------
        context:
            Current bar and trade state snapshot.

        Returns
        -------
        EdgeResult
            ``allowed=True`` if the condition is satisfied (do not block);
            ``allowed=False`` if the edge vetoes the action.
        """

    def _disabled_result(self) -> EdgeResult:
        """Return a pass-through result used when the edge is disabled."""
        return EdgeResult(
            allowed=True,
            edge_name=self.name,
            reason=f"{self.name} is disabled — skipping",
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, enabled={self.enabled})"
