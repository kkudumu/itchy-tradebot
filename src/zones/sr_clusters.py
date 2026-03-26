"""
DBSCAN-based support and resistance zone clustering.

Swing point prices are grouped into S/R zones using DBSCAN with an ATR-based
epsilon, making the clustering adaptive to current market volatility.  Each
resulting cluster becomes a zone with a strength score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
from sklearn.cluster import DBSCAN


@dataclass
class SRZone:
    """A clustered support or resistance zone derived from swing points.

    Attributes
    ----------
    price_high:
        Upper boundary of the zone (max price of all clustered swing points).
    price_low:
        Lower boundary of the zone (min price of all clustered swing points).
    center:
        Central price of the zone (mean of cluster members).
    touch_count:
        Number of swing points that formed this cluster.
    timeframe:
        Timeframe label this zone was detected on (e.g. "H4", "H1").
    first_seen:
        Timestamp of the earliest swing point in the cluster.
    last_touched:
        Timestamp of the most recent swing point in the cluster.
    zone_type:
        ``'support'`` or ``'resistance'``.  Determined by whether the cluster
        was formed from swing lows or swing highs respectively.
    strength:
        Composite strength score (populated by :meth:`SRClusterDetector.score_zone`).
    """

    price_high: float
    price_low: float
    center: float
    touch_count: int
    timeframe: str
    first_seen: datetime
    last_touched: datetime
    zone_type: str  # 'support' or 'resistance'
    strength: float = 0.0


class SRClusterDetector:
    """Cluster swing point prices into S/R zones using DBSCAN.

    DBSCAN epsilon is set to ``atr * atr_multiplier``, making the clustering
    distance threshold proportional to current volatility.  ``min_samples`` is
    set to ``min_touches`` so that a zone requires at least that many swing
    points to be recognised.

    Parameters
    ----------
    atr_multiplier:
        Epsilon multiplier relative to ATR.  Default 1.0 means one ATR defines
        the neighbourhood radius for clustering.
    min_touches:
        Minimum number of swing points to form a valid cluster (DBSCAN
        ``min_samples``).  Default 2.
    """

    def __init__(
        self,
        atr_multiplier: float = 1.0,
        min_touches: int = 2,
    ) -> None:
        if atr_multiplier <= 0:
            raise ValueError(f"atr_multiplier must be > 0, got {atr_multiplier}")
        if min_touches < 1:
            raise ValueError(f"min_touches must be >= 1, got {min_touches}")
        self.atr_multiplier = atr_multiplier
        self.min_touches = min_touches

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cluster(
        self,
        swing_points: np.ndarray,
        atr: float,
        timeframe: str = "",
        timestamps: Optional[np.ndarray] = None,
        zone_type: str = "support",
    ) -> list[SRZone]:
        """Cluster swing point prices into S/R zones.

        Parameters
        ----------
        swing_points:
            1-D array of swing point prices (either all swing highs or all
            swing lows for a single call).
        atr:
            Current ATR value used to compute the DBSCAN epsilon.
        timeframe:
            Label to attach to each resulting zone.
        timestamps:
            Optional array of datetime objects aligned with ``swing_points``.
            Used to populate ``first_seen`` and ``last_touched`` on zones.
        zone_type:
            ``'support'`` for clusters of swing lows, ``'resistance'`` for
            clusters of swing highs.

        Returns
        -------
        list[SRZone]
            One zone per valid DBSCAN cluster (noise points are discarded).
        """
        if len(swing_points) == 0:
            return []

        prices = np.asarray(swing_points, dtype=float).reshape(-1, 1)

        epsilon = atr * self.atr_multiplier
        if epsilon <= 0:
            # Fallback: use 0.1% of price as minimum epsilon
            epsilon = float(np.mean(prices)) * 0.001

        db = DBSCAN(eps=epsilon, min_samples=self.min_touches, metric="euclidean")
        labels = db.fit_predict(prices)

        # Sentinel datetime used when timestamps are not provided
        _fallback_dt = datetime(2000, 1, 1)

        zones: list[SRZone] = []
        unique_labels = set(labels)
        unique_labels.discard(-1)  # -1 is noise in DBSCAN

        for label in unique_labels:
            mask = labels == label
            cluster_prices = prices[mask, 0]

            if timestamps is not None:
                cluster_ts = np.asarray(timestamps)[mask]
                first_seen = min(cluster_ts)
                last_touched = max(cluster_ts)
            else:
                first_seen = _fallback_dt
                last_touched = _fallback_dt

            zone = SRZone(
                price_high=float(np.max(cluster_prices)),
                price_low=float(np.min(cluster_prices)),
                center=float(np.mean(cluster_prices)),
                touch_count=int(np.sum(mask)),
                timeframe=timeframe,
                first_seen=first_seen,
                last_touched=last_touched,
                zone_type=zone_type,
                strength=0.0,
            )
            zones.append(zone)

        return zones

    def score_zone(
        self,
        zone: SRZone,
        current_price: float,
        current_time: datetime,
        atr: float,
        recency_half_life_bars: int = 100,
        multi_tf_bonus: float = 1.5,
    ) -> float:
        """Compute a composite strength score for a zone.

        Strength formula::

            strength = touch_count × recency_weight × proximity_weight

        where:

        * ``recency_weight`` — exponential decay based on days since the last
          touch.  At ``recency_half_life_bars`` bars the weight halves.
          Measured in seconds when ``current_time`` and ``last_touched`` are
          ``datetime`` objects.

        * ``proximity_weight`` — zones closer to ``current_price`` (in ATR
          units) score higher.  Zones beyond 10 ATR get a weight of 0.1;
          zones within 1 ATR get a weight of 1.0.

        * ``multi_tf_bonus`` — applied externally by the caller when the same
          zone is confirmed across multiple timeframes.  Default argument here
          is unused in this method; it is exposed for callers that want to
          multiply the return value.

        Parameters
        ----------
        zone:
            The zone to score.
        current_price:
            Most recent market price.
        current_time:
            Timestamp of the most recent bar.
        atr:
            Current ATR for normalising distance.
        recency_half_life_bars:
            Number of bars at which recency weight halves.  Used as a proxy
            for time decay (we divide time in seconds by an assumed bar
            duration of 3600 s = 1 H1 bar).
        multi_tf_bonus:
            Not applied inside this method.  Documented for callers.

        Returns
        -------
        float
            Composite strength score >= 0.
        """
        # --- Recency weight (exponential decay) ---
        if isinstance(current_time, datetime) and isinstance(zone.last_touched, datetime):
            elapsed_seconds = max(0.0, (current_time - zone.last_touched).total_seconds())
            # Assume 1 H1 bar = 3600 seconds; convert to "bar units"
            elapsed_bars = elapsed_seconds / 3600.0
        else:
            elapsed_bars = 0.0

        decay_constant = math.log(2.0) / max(1, recency_half_life_bars)
        recency_weight = math.exp(-decay_constant * elapsed_bars)

        # --- Proximity weight (Gaussian-like falloff, normalised to ATR) ---
        distance_atr = abs(current_price - zone.center) / atr if atr > 0 else 0.0
        # Close zones (< 1 ATR) → weight ~1.0; far zones (> 10 ATR) → weight → 0
        proximity_weight = math.exp(-0.2 * distance_atr)
        proximity_weight = max(0.05, proximity_weight)

        strength = zone.touch_count * recency_weight * proximity_weight
        return float(strength)
