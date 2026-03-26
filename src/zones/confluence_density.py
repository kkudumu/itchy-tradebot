"""
Confluence density scoring — count and score zone types near a given price.

The scorer checks how many zone objects (S/R clusters, supply/demand zones) and
pivot levels fall within a configurable ATR-based proximity window around a
reference price.  A higher count indicates stronger confluence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ConfluenceScore:
    """Results of a confluence density evaluation at a price level.

    Attributes
    ----------
    total:
        Total number of distinct zone/pivot levels within proximity.
    sr_zones:
        Count of S/R cluster zones within proximity.
    sd_zones:
        Count of supply/demand zones within proximity.
    pivot_levels:
        Count of pivot point levels within proximity.
    multi_tf_count:
        Number of unique timeframes represented among the nearby zones.
    details:
        List of dicts describing each nearby item::

            {
                "type":          str,   # 'sr_zone', 'sd_zone', 'pivot'
                "price":         float,
                "distance_atr":  float,
                "timeframe":     str,   # '' when not applicable
            }
    """

    total: int = 0
    sr_zones: int = 0
    sd_zones: int = 0
    pivot_levels: int = 0
    multi_tf_count: int = 0
    details: list[dict] = field(default_factory=list)


class ConfluenceDensityScorer:
    """Count and score zone confluence near a reference price.

    Any zone or pivot level whose price boundary or centre falls within
    ``proximity_atr × ATR`` of the reference price is included in the score.

    Parameters
    ----------
    proximity_atr:
        Zones within this many ATR units of the reference price are counted.
        Default: 1.0.
    """

    def __init__(self, proximity_atr: float = 1.0) -> None:
        if proximity_atr <= 0:
            raise ValueError(f"proximity_atr must be > 0, got {proximity_atr}")
        self.proximity_atr = proximity_atr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        price: float,
        zones: list[Any],
        pivots: dict[str, float],
        atr: float,
    ) -> ConfluenceScore:
        """Evaluate confluence density at ``price``.

        Parameters
        ----------
        price:
            Reference price level to evaluate (e.g. current close or
            proposed entry price).
        zones:
            List of zone objects.  Accepts :class:`~src.zones.sr_clusters.SRZone`
            and :class:`~src.zones.supply_demand.SDZone` instances (and any
            object with ``price_high``, ``price_low`` attributes and optionally
            ``zone_type`` / ``timeframe``).
        pivots:
            Flat dict mapping level names to prices, e.g.::

                {"PP": 1920.5, "R1": 1935.0, "S1": 1906.0, ...}

            Multiple timeframe pivot dicts can be pre-merged by the caller
            before passing here.
        atr:
            Current ATR for the proximity window calculation.

        Returns
        -------
        ConfluenceScore
        """
        proximity = self.proximity_atr * atr
        details: list[dict] = []
        timeframes_seen: set[str] = set()

        sr_count = 0
        sd_count = 0
        pivot_count = 0

        # --- Evaluate zone objects ---
        for zone in zones:
            # Determine representative price for the zone (closest boundary to price)
            zone_high = getattr(zone, "price_high", None)
            zone_low = getattr(zone, "price_low", None)
            zone_center = getattr(zone, "center", None)
            timeframe = str(getattr(zone, "timeframe", ""))
            zone_type_attr = getattr(zone, "zone_type", "")

            if zone_high is None or zone_low is None:
                continue

            # Check if the price falls within the zone or is within proximity
            if zone_low <= price <= zone_high:
                dist = 0.0
            else:
                dist = min(abs(price - zone_high), abs(price - zone_low))
                if zone_center is not None:
                    dist = min(dist, abs(price - float(zone_center)))

            if dist <= proximity:
                dist_atr = dist / atr if atr > 0 else 0.0

                # Classify zone type
                is_sr = zone_type_attr in ("support", "resistance")
                is_sd = zone_type_attr in ("supply", "demand")

                if is_sr:
                    sr_count += 1
                    item_type = "sr_zone"
                elif is_sd:
                    sd_count += 1
                    item_type = "sd_zone"
                else:
                    # Unknown zone type still counts toward total
                    item_type = "zone"

                details.append({
                    "type": item_type,
                    "price": float(zone_center if zone_center is not None else (zone_high + zone_low) / 2),
                    "distance_atr": dist_atr,
                    "timeframe": timeframe,
                })

                if timeframe:
                    timeframes_seen.add(timeframe)

        # --- Evaluate pivot levels ---
        for level_name, level_price in pivots.items():
            if not isinstance(level_price, (int, float)):
                continue
            dist = abs(price - float(level_price))
            if dist <= proximity:
                dist_atr = dist / atr if atr > 0 else 0.0
                pivot_count += 1
                details.append({
                    "type": "pivot",
                    "price": float(level_price),
                    "distance_atr": dist_atr,
                    "timeframe": "",
                })

        total = sr_count + sd_count + pivot_count

        return ConfluenceScore(
            total=total,
            sr_zones=sr_count,
            sd_zones=sd_count,
            pivot_levels=pivot_count,
            multi_tf_count=len(timeframes_seen),
            details=details,
        )
