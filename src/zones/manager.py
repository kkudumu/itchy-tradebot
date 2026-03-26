"""
Zone lifecycle manager — creation, updates, invalidation, and queries.

The :class:`ZoneManager` is the central registry for all zone objects
detected during a trading session or backtest run.  It handles:

* Adding new zones (with optional merge of overlapping zones)
* Updating zone status based on price action candles
* Invalidating zones when price body-closes through them
* Querying zones by proximity to a given price
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


@dataclass
class _ManagedZone:
    """Internal wrapper that attaches a unique ID to a zone object."""
    zone_id: int
    zone: Any           # SRZone, SDZone, or any object with price_high / price_low
    zone_family: str    # 'sr', 'sd', 'pivot', or 'generic'
    invalidation_reason: Optional[str] = None


class ZoneManager:
    """Manage zone lifecycle: creation, updates, invalidation, and queries.

    Zones can be of any type that exposes ``price_high`` and ``price_low``
    float attributes and a ``status`` string attribute (``'active'``,
    ``'tested'``, ``'invalidated'``).

    Parameters
    ----------
    merge_overlap:
        When ``True``, newly added zones that overlap with an existing active
        zone of the same type are merged (the existing zone's boundaries are
        expanded to encompass both).  Default: ``True``.
    max_invalidated_age_bars:
        Invalidated zones older than this many maintenance cycles are removed
        during :meth:`maintenance`.  Default: 500.
    """

    def __init__(
        self,
        merge_overlap: bool = True,
        max_invalidated_age_bars: int = 500,
    ) -> None:
        self.merge_overlap = merge_overlap
        self.max_invalidated_age_bars = max_invalidated_age_bars

        self._zones: list[_ManagedZone] = []
        self._next_id: int = 1
        self._maintenance_counter: int = 0

        # Maps zone_id → number of maintenance cycles since invalidation
        self._invalidated_ages: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Zone registry (read-only)
    # ------------------------------------------------------------------

    @property
    def zones(self) -> list[Any]:
        """All managed zone objects (active, tested, and invalidated)."""
        return [mz.zone for mz in self._zones]

    @property
    def active_zones(self) -> list[Any]:
        """Only zones with status == 'active' or 'tested'."""
        return [
            mz.zone for mz in self._zones
            if getattr(mz.zone, "status", "active") in ("active", "tested")
        ]

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def add_zone(self, zone: Any) -> int:
        """Add a zone to the manager.

        If ``merge_overlap`` is enabled and an active zone with the same
        ``zone_type`` overlaps the new zone, the existing zone's price
        boundaries are expanded rather than adding a duplicate.

        Parameters
        ----------
        zone:
            Any zone object with ``price_high``, ``price_low``, and
            ``status`` attributes.  Optionally ``zone_type``, ``touch_count``.

        Returns
        -------
        int
            The assigned zone ID (new or existing if merged).
        """
        if self.merge_overlap:
            merged_id = self._try_merge(zone)
            if merged_id is not None:
                return merged_id

        zone_id = self._next_id
        self._next_id += 1

        family = self._classify_family(zone)
        self._zones.append(_ManagedZone(zone_id=zone_id, zone=zone, zone_family=family))
        return zone_id

    def update_zone(self, zone_id: int, candle: Any) -> None:
        """Update a zone's status based on a new price action candle.

        Rules
        -----
        * Wick touches the zone (high/low enters zone but close is outside) →
          status becomes ``'tested'``, touch count incremented.
        * Body closes through the zone (close is on the far side of the zone) →
          status becomes ``'invalidated'``.

        Parameters
        ----------
        zone_id:
            ID of the zone to update.
        candle:
            Object (or dict) with ``open``, ``high``, ``low``, ``close``
            attributes / keys representing the closing candle.
        """
        mz = self._find(zone_id)
        if mz is None:
            return

        zone = mz.zone
        if getattr(zone, "status", "active") == "invalidated":
            return  # Already invalidated; skip processing

        o, h, l, c = self._extract_ohlc(candle)
        z_high = float(zone.price_high)
        z_low = float(zone.price_low)

        # --- Invalidation: body close through the zone ---
        zone_type = getattr(zone, "zone_type", "")
        if zone_type in ("support", "demand"):
            # Bearish body close below zone low
            if c < z_low and o > z_low:
                self.invalidate(zone_id, reason="body_close_below_support")
                return
        elif zone_type in ("resistance", "supply"):
            # Bullish body close above zone high
            if c > z_high and o < z_high:
                self.invalidate(zone_id, reason="body_close_above_resistance")
                return
        else:
            # Generic: invalidate if body closes through from either side
            if (c < z_low and o > z_low) or (c > z_high and o < z_high):
                self.invalidate(zone_id, reason="body_close_through")
                return

        # --- Tested: wick entered the zone ---
        wick_in_zone = (l <= z_high and l >= z_low) or (h >= z_low and h <= z_high)
        price_crossed_zone = l <= z_high and h >= z_low

        if price_crossed_zone or wick_in_zone:
            if hasattr(zone, "status"):
                zone.status = "tested"
            if hasattr(zone, "touch_count"):
                zone.touch_count += 1
            if hasattr(zone, "last_touched") and hasattr(candle, "timestamp"):
                zone.last_touched = candle.timestamp
            elif hasattr(zone, "last_tested") and hasattr(candle, "timestamp"):
                zone.last_tested = candle.timestamp

    def invalidate(self, zone_id: int, reason: str = "") -> None:
        """Mark a zone as invalidated.

        Parameters
        ----------
        zone_id:
            ID of the zone to invalidate.
        reason:
            Human-readable reason string stored on the internal wrapper.
        """
        mz = self._find(zone_id)
        if mz is None:
            return

        if hasattr(mz.zone, "status"):
            mz.zone.status = "invalidated"
        mz.invalidation_reason = reason
        self._invalidated_ages[zone_id] = 0

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    def get_nearby_zones(
        self,
        price: float,
        atr: float,
        max_distance_atr: float = 2.0,
    ) -> list[Any]:
        """Return all active zones within ``max_distance_atr × ATR`` of ``price``.

        Proximity is measured from ``price`` to the nearest boundary of each
        zone (or zero if price is inside the zone).

        Parameters
        ----------
        price:
            Reference price.
        atr:
            Current ATR for normalising distance.
        max_distance_atr:
            Maximum distance in ATR units to include a zone.

        Returns
        -------
        list of zone objects sorted by proximity (closest first).
        """
        proximity = max_distance_atr * atr
        nearby: list[tuple[float, Any]] = []

        for mz in self._zones:
            zone = mz.zone
            if getattr(zone, "status", "active") == "invalidated":
                continue

            z_high = float(zone.price_high)
            z_low = float(zone.price_low)

            if z_low <= price <= z_high:
                dist = 0.0
            else:
                dist = min(abs(price - z_high), abs(price - z_low))

            if dist <= proximity:
                nearby.append((dist, zone))

        nearby.sort(key=lambda t: t[0])
        return [zone for _, zone in nearby]

    def get_strongest_zone(
        self,
        price: float,
        atr: float,
        direction: str,
    ) -> Optional[Any]:
        """Return the strongest nearby zone for a given trade direction.

        For a long trade (``direction='long'``), looks for support zones
        (``'support'``, ``'demand'``) below or near the current price.
        For a short trade (``direction='short'``), looks for resistance zones
        (``'resistance'``, ``'supply'``) above or near the current price.

        The zone with the highest ``strength`` attribute is returned.

        Parameters
        ----------
        price:
            Reference price.
        atr:
            Current ATR.
        direction:
            ``'long'`` or ``'short'``.

        Returns
        -------
        The zone object with the highest ``strength``, or ``None`` if no
        matching zone is found within 2 ATR.
        """
        nearby = self.get_nearby_zones(price, atr, max_distance_atr=2.0)

        if direction == "long":
            target_types = {"support", "demand"}
        elif direction == "short":
            target_types = {"resistance", "supply"}
        else:
            return None

        candidates = [
            z for z in nearby
            if getattr(z, "zone_type", "") in target_types
        ]

        if not candidates:
            return None

        return max(candidates, key=lambda z: getattr(z, "strength", 0.0))

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def maintenance(self, current_candle: Any) -> None:
        """Run housekeeping on every candle close.

        Operations performed:

        1. Update all active/tested zones using the new candle.
        2. Age out very old invalidated zones to keep the list compact.

        Parameters
        ----------
        current_candle:
            The candle that just closed.  Must expose ``open``, ``high``,
            ``low``, ``close`` attributes or dict keys.
        """
        self._maintenance_counter += 1

        # Update all active zones with the latest candle
        for mz in list(self._zones):
            if getattr(mz.zone, "status", "active") in ("active", "tested"):
                self.update_zone(mz.zone_id, current_candle)

        # Age invalidated zones and prune old ones
        to_remove: list[int] = []
        for zone_id in list(self._invalidated_ages.keys()):
            self._invalidated_ages[zone_id] += 1
            if self._invalidated_ages[zone_id] >= self.max_invalidated_age_bars:
                to_remove.append(zone_id)

        if to_remove:
            remove_set = set(to_remove)
            self._zones = [mz for mz in self._zones if mz.zone_id not in remove_set]
            for zone_id in to_remove:
                del self._invalidated_ages[zone_id]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find(self, zone_id: int) -> Optional[_ManagedZone]:
        """Find a managed zone by ID, or return None."""
        for mz in self._zones:
            if mz.zone_id == zone_id:
                return mz
        return None

    def _try_merge(self, new_zone: Any) -> Optional[int]:
        """Attempt to merge ``new_zone`` into an existing overlapping zone.

        Two zones are considered overlapping if their price ranges intersect
        and they share the same ``zone_type``.

        Returns the ID of the zone that was merged into, or None if no merge.
        """
        new_high = float(new_zone.price_high)
        new_low = float(new_zone.price_low)
        new_type = getattr(new_zone, "zone_type", "")

        for mz in self._zones:
            zone = mz.zone
            if getattr(zone, "status", "active") == "invalidated":
                continue
            if getattr(zone, "zone_type", "") != new_type:
                continue

            z_high = float(zone.price_high)
            z_low = float(zone.price_low)

            # Overlap check: ranges intersect
            if new_low <= z_high and new_high >= z_low:
                # Expand the existing zone to encompass the new one
                if hasattr(zone, "price_high"):
                    zone.price_high = max(z_high, new_high)
                if hasattr(zone, "price_low"):
                    zone.price_low = min(z_low, new_low)
                if hasattr(zone, "touch_count") and hasattr(new_zone, "touch_count"):
                    zone.touch_count += new_zone.touch_count
                # Recompute center if present
                if hasattr(zone, "center"):
                    zone.center = (zone.price_high + zone.price_low) / 2.0
                return mz.zone_id

        return None

    @staticmethod
    def _classify_family(zone: Any) -> str:
        """Classify a zone into a broad family based on its zone_type."""
        zone_type = getattr(zone, "zone_type", "")
        if zone_type in ("support", "resistance"):
            return "sr"
        if zone_type in ("supply", "demand"):
            return "sd"
        if zone_type == "pivot":
            return "pivot"
        return "generic"

    @staticmethod
    def _extract_ohlc(candle: Any) -> tuple[float, float, float, float]:
        """Extract OHLC values from either an object or dict-like candle."""
        if isinstance(candle, dict):
            return (
                float(candle["open"]),
                float(candle["high"]),
                float(candle["low"]),
                float(candle["close"]),
            )
        return (
            float(candle.open),
            float(candle.high),
            float(candle.low),
            float(candle.close),
        )
