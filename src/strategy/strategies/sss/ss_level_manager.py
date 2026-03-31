"""SSLevelManager — tracks SS levels (POIs) from inefficient sequences.

Bullish SS level swept when price < level.price; bearish when price > level.price.
Memory-bounded to max_active_levels (default 50) with LRU eviction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from .sequence_tracker import SequenceEvent

log = logging.getLogger(__name__)

# Trigger event types that can generate SS levels (inefficient sequences only)
_LEVEL_EVENT_TYPES = frozenset({"sequence_complete", "sequence_timeout"})


@dataclass
class SSLevel:
    """A price area created by an incomplete (inefficient) sequence."""

    price: float
    """The SS level price."""

    direction: str
    """'bullish' or 'bearish' — direction the level would attract price."""

    creation_time: datetime
    """When the inefficient sequence was detected."""

    candle_count: int
    """Candle count of the originating swing (for quality assessment)."""

    layer: str
    """'SS' or 'ISS' — from the originating sequence."""

    is_active: bool = True
    """False once price sweeps through this level."""

    swept_time: Optional[datetime] = None
    """When the level was invalidated."""


class SSLevelManager:
    """Tracks SS-level POIs created by inefficient sequences.

    Parameters
    ----------
    max_active_levels:
        Maximum number of active levels to keep. When exceeded, the oldest
        active level (by creation_time) is evicted (LRU).
    """

    def __init__(self, max_active_levels: int = 50) -> None:
        self._max_active = max_active_levels
        self._levels: list[SSLevel] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_sequence_event(self, event: SequenceEvent) -> Optional[SSLevel]:
        """Process a sequence event and optionally create an SS level.

        Only creates a level when:
          - event_type is 'sequence_complete' or 'sequence_timeout'
          - sequence_efficient is False

        Returns the new SSLevel if created, else None.
        """
        if event.event_type not in _LEVEL_EVENT_TYPES:
            return None
        if event.sequence_efficient:
            return None

        # CBC layer levels are not meaningful POIs — skip
        if event.layer == "CBC":
            return None

        new_level = SSLevel(
            price=event.level_price,
            direction=event.direction,
            creation_time=event.timestamp,
            candle_count=event.candle_count,
            layer=event.layer,
        )
        self._levels.append(new_level)
        log.debug(
            "SS level created: dir=%s price=%.5f layer=%s",
            new_level.direction,
            new_level.price,
            new_level.layer,
        )
        self._evict_if_needed()
        return new_level

    def update_price(self, timestamp: datetime, high: float, low: float) -> List[SSLevel]:
        """Check if price has swept any active SS levels.

        A bullish SS level is swept when price goes below level.price.
        A bearish SS level is swept when price goes above level.price.

        Returns list of levels that were just swept (invalidated).
        """
        swept: list[SSLevel] = []
        for lvl in self._levels:
            if not lvl.is_active:
                continue
            if lvl.direction == "bullish" and low < lvl.price:
                lvl.is_active = False
                lvl.swept_time = timestamp
                swept.append(lvl)
                log.debug("Bullish SS level swept: price=%.5f", lvl.price)
            elif lvl.direction == "bearish" and high > lvl.price:
                lvl.is_active = False
                lvl.swept_time = timestamp
                swept.append(lvl)
                log.debug("Bearish SS level swept: price=%.5f", lvl.price)
        return swept

    def get_nearest_targets(self, current_price: float, direction: str) -> List[SSLevel]:
        """Get active SS levels that are valid targets for the given direction.

        For bullish trades: bearish SS levels ABOVE current price.
        For bearish trades: bullish SS levels BELOW current price.

        Returns levels sorted by distance from current_price (nearest first).
        """
        if direction == "bullish":
            candidates = [
                lvl for lvl in self._levels
                if lvl.is_active and lvl.direction == "bearish" and lvl.price > current_price
            ]
            return sorted(candidates, key=lambda lvl: lvl.price - current_price)
        # bearish
        candidates = [
            lvl for lvl in self._levels
            if lvl.is_active and lvl.direction == "bullish" and lvl.price < current_price
        ]
        return sorted(candidates, key=lambda lvl: current_price - lvl.price)

    def get_nearest_target(self, current_price: float, direction: str) -> Optional[SSLevel]:
        """Return the single nearest valid target, or None."""
        targets = self.get_nearest_targets(current_price, direction)
        return targets[0] if targets else None

    def calculate_target_rr(
        self,
        entry_price: float,
        stop_loss: float,
        target_level: SSLevel,
    ) -> float:
        """Calculate the R:R ratio for a given target level.

        R:R = distance_to_target / risk_per_unit.
        Returns 0.0 if risk is zero or the target is on the wrong side.
        """
        risk = abs(entry_price - stop_loss)
        if risk == 0.0:
            return 0.0
        reward = abs(target_level.price - entry_price)
        return reward / risk

    @property
    def active_levels(self) -> List[SSLevel]:
        """All currently active (not swept) SS levels."""
        return [lvl for lvl in self._levels if lvl.is_active]

    @property
    def all_levels(self) -> List[SSLevel]:
        """All SS levels including swept ones (for analysis)."""
        return list(self._levels)

    def reset(self) -> None:
        """Clear all tracked levels."""
        self._levels.clear()
        log.debug("SSLevelManager reset")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        """Remove the oldest active level when max_active_levels is exceeded."""
        active = [lvl for lvl in self._levels if lvl.is_active]
        if len(active) <= self._max_active:
            return
        # Sort by creation_time, evict the oldest
        oldest = min(active, key=lambda lvl: lvl.creation_time)
        oldest.is_active = False
        log.debug(
            "SSLevelManager evicted oldest level: price=%.5f created=%s",
            oldest.price,
            oldest.creation_time,
        )
