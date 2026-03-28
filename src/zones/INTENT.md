# Zones — Intent

## Functions

### SwingPointDetector.__init__(lookback: int) -> None
- **Does**: Configures the symmetric lookback window size for fractal swing point detection.
- **Why**: Different lookback values capture different structural levels -- small values find short-term pivots, large values find major structure.
- **Relationships**: None.
- **Decisions**: None.

### SwingPointDetector.detect(high: np.ndarray, low: np.ndarray, timestamps: np.ndarray | None) -> SwingPoints
- **Does**: Identifies fractal swing highs and swing lows, returning them as lists of (index, price, optional_timestamp) tuples.
- **Why**: Swing points are the raw input for S/R clustering and supply/demand zone detection -- they mark where price reversed direction.
- **Relationships**: Calls detect_vectorized; called by zone detection pipelines that feed SRClusterDetector.cluster.
- **Decisions**: Uses inclusive inequality (>=, <=) so that flat-top/bottom formations are captured as valid swing points.

### SwingPointDetector.detect_vectorized(high: np.ndarray, low: np.ndarray) -> tuple[np.ndarray, np.ndarray]
- **Does**: Returns boolean masks marking swing high and swing low bars across the input arrays.
- **Why**: Provides a mask-based API for callers that need array-level operations rather than tuple lists.
- **Relationships**: Called by detect.
- **Decisions**: Only bars with full lookback windows on both sides can qualify, so the first and last `lookback` bars are always False.

---

### SRClusterDetector.__init__(atr_multiplier: float, min_touches: int) -> None
- **Does**: Configures the DBSCAN epsilon multiplier (relative to ATR) and minimum cluster membership.
- **Why**: ATR-scaled epsilon makes clustering adaptive to current volatility rather than using a fixed price distance.
- **Relationships**: None.
- **Decisions**: Default atr_multiplier of 1.0 means one ATR defines the neighbourhood radius for grouping nearby swing points.

### SRClusterDetector.cluster(swing_points: np.ndarray, atr: float, timeframe: str, timestamps: np.ndarray | None, zone_type: str) -> list[SRZone]
- **Does**: Groups swing point prices into support/resistance zones using DBSCAN with ATR-based epsilon.
- **Why**: Raw swing points are noisy; clustering aggregates them into meaningful price zones with touch counts and boundaries.
- **Relationships**: Called by zone detection pipelines; output consumed by ZoneManager.add_zone, ConfluenceDensityScorer.score.
- **Decisions**: DBSCAN noise points (label -1) are discarded rather than forming single-point zones.

### SRClusterDetector.score_zone(zone: SRZone, current_price: float, current_time: datetime, atr: float, recency_half_life_bars: int, multi_tf_bonus: float) -> float
- **Does**: Computes a composite strength score for a zone using touch count, exponential recency decay, and Gaussian proximity weighting.
- **Why**: Not all zones are equal -- recently tested, frequently touched zones near current price are more likely to hold.
- **Relationships**: Operates on SRZone objects; called by zone scoring pipelines.
- **Decisions**: Recency decay assumes 1-hour bars for time-to-bar conversion; proximity weight floors at 0.05 to prevent zero scores for distant zones.

---

### SupplyDemandDetector.__init__(impulse_threshold: float, consolidation_bars: int) -> None
- **Does**: Configures the minimum impulse move size (in ATR multiples) and maximum consolidation base length.
- **Why**: Thresholds filter out weak moves and overly wide bases that lack the institutional footprint of true supply/demand zones.
- **Relationships**: None.
- **Decisions**: Default impulse threshold of 2.5 ATR captures only significant institutional-grade moves.

### SupplyDemandDetector.detect(ohlc: pd.DataFrame, atr: np.ndarray) -> list[SDZone]
- **Does**: Scans OHLC data for consolidation bases immediately preceding large impulse moves, creating supply (bearish) or demand (bullish) zones.
- **Why**: Supply/demand zones mark price levels where institutional orders originated; price tends to revisit these bases.
- **Relationships**: Calls _extract_base; output consumed by ZoneManager.add_zone, ConfluenceDensityScorer.score.
- **Decisions**: Tracks used base-end indices to prevent duplicate zones from overlapping detection windows.

---

### PivotCalculator.standard_pivots(high: float, low: float, close: float) -> dict[str, float]
- **Does**: Calculates classic pivot point levels (PP, R1-R3, S1-S3) from a single period's HLC values.
- **Why**: Standard pivots provide objective, widely-watched price levels that act as intraday support/resistance.
- **Relationships**: Called by from_daily_ohlc, from_weekly_ohlc, from_monthly_ohlc via _apply_pivots.
- **Decisions**: None.

### PivotCalculator.fibonacci_pivots(high: float, low: float, close: float) -> dict[str, float]
- **Does**: Calculates Fibonacci-based pivot levels (PP, R1-R3, S1-S3) using 0.382/0.618/1.0 ratios applied to the HLC range.
- **Why**: Fibonacci pivots complement standard pivots by adding levels at key retracement ratios watched by institutional traders.
- **Relationships**: Called by _apply_pivots when kind="fibonacci".
- **Decisions**: None.

### PivotCalculator.from_daily_ohlc(daily_ohlc: pd.DataFrame) -> pd.DataFrame
- **Does**: Applies standard pivot calculations row-by-row to a daily OHLC DataFrame, producing pivot levels valid for the next trading day.
- **Why**: Daily pivots are the most commonly used timeframe for intraday gold trading reference levels.
- **Relationships**: Calls _apply_pivots; output consumed by ConfluenceDensityScorer.score.
- **Decisions**: None.

### PivotCalculator.from_weekly_ohlc(weekly_ohlc: pd.DataFrame) -> pd.DataFrame
- **Does**: Applies standard pivot calculations to a weekly OHLC DataFrame.
- **Why**: Weekly pivots provide higher-timeframe structural levels that carry more weight than daily pivots.
- **Relationships**: Calls _apply_pivots.
- **Decisions**: None.

### PivotCalculator.from_monthly_ohlc(monthly_ohlc: pd.DataFrame) -> pd.DataFrame
- **Does**: Applies standard pivot calculations to a monthly OHLC DataFrame.
- **Why**: Monthly pivots mark the strongest structural reference levels for swing-level position awareness.
- **Relationships**: Calls _apply_pivots.
- **Decisions**: None.

---

### ConfluenceDensityScorer.__init__(proximity_atr: float) -> None
- **Does**: Configures the ATR-based proximity window within which zones and pivots are counted as confluent.
- **Why**: Using ATR-relative distance ensures the confluence radius adapts to current market volatility.
- **Relationships**: None.
- **Decisions**: Default of 1.0 ATR balances sensitivity (catching nearby levels) with specificity (not over-counting distant levels).

### ConfluenceDensityScorer.score(price: float, zones: list[Any], pivots: dict[str, float], atr: float) -> ConfluenceScore
- **Does**: Counts S/R zones, supply/demand zones, and pivot levels within the ATR proximity window of a reference price, producing a detailed confluence breakdown.
- **Why**: Zone confluence density quantifies how many independent technical reasons support a price level, which correlates with the level's strength.
- **Relationships**: Consumes SRZone and SDZone objects from SRClusterDetector and SupplyDemandDetector; consumes pivot dicts from PivotCalculator; called by signal evaluation pipelines.
- **Decisions**: Price inside a zone's boundaries counts as distance 0; multi-timeframe count tracks unique timeframes to support multi-TF bonus scoring.

---

### ZoneManager.__init__(merge_overlap: bool, max_invalidated_age_bars: int) -> None
- **Does**: Initialises the zone lifecycle registry with merge and cleanup configuration.
- **Why**: Centralises zone state management so that detection, updates, invalidation, and queries all operate on a single consistent registry.
- **Relationships**: Called by SignalEngine.__init__.
- **Decisions**: Merge-overlap enabled by default to prevent duplicate zones from accumulating at the same price level.

### ZoneManager.zones -> list[Any]
- **Does**: Returns all managed zone objects regardless of status (active, tested, invalidated).
- **Why**: Provides read-only access to the full zone population for analysis and debugging.
- **Relationships**: Reads internal _zones list.
- **Decisions**: None.

### ZoneManager.active_zones -> list[Any]
- **Does**: Returns only zones with status "active" or "tested", excluding invalidated zones.
- **Why**: Most consumers only care about zones that are still in play for trade decisions.
- **Relationships**: Reads internal _zones list.
- **Decisions**: Treats "tested" as still active because a zone that has been tested but held is stronger, not weaker.

### ZoneManager.add_zone(zone: Any) -> int
- **Does**: Registers a new zone, optionally merging it into an existing overlapping zone of the same type if merge_overlap is enabled.
- **Why**: Provides the single entry point for all zone creation, ensuring deduplication and consistent ID assignment.
- **Relationships**: Calls _try_merge, _classify_family; called by detection pipelines feeding zones into the manager.
- **Decisions**: Merging expands existing zone boundaries and sums touch counts rather than creating a second overlapping zone.

### ZoneManager.update_zone(zone_id: int, candle: Any) -> None
- **Does**: Updates a zone's status based on a new price candle -- marks "tested" on wick touches and "invalidated" on body closes through the zone.
- **Why**: Zone lifecycle tracking is essential for knowing whether a zone is still valid for trade decisions.
- **Relationships**: Calls invalidate, _extract_ohlc; called by maintenance.
- **Decisions**: Invalidation requires a body close (open on one side, close on the other) rather than just a wick penetration, matching institutional S/R methodology.

### ZoneManager.invalidate(zone_id: int, reason: str) -> None
- **Does**: Sets a zone's status to "invalidated" and records the reason.
- **Why**: Explicit invalidation with reason tracking supports debugging and post-trade analysis.
- **Relationships**: Called by update_zone.
- **Decisions**: None.

### ZoneManager.get_nearby_zones(price: float, atr: float, max_distance_atr: float) -> list[Any]
- **Does**: Returns all active/tested zones within a configurable ATR-based distance of a reference price, sorted by proximity.
- **Why**: Entry and exit decisions need to know which zones are nearby to assess support/resistance context.
- **Relationships**: Called by SignalEngine.scan, get_strongest_zone.
- **Decisions**: Excludes invalidated zones; measures distance to nearest zone boundary (or 0 if price is inside the zone).

### ZoneManager.get_strongest_zone(price: float, atr: float, direction: str) -> Any | None
- **Does**: Returns the highest-strength zone matching the trade direction (support/demand for longs, resistance/supply for shorts) within 2 ATR.
- **Why**: Identifies the single most significant nearby zone for stop-loss placement and trade conviction assessment.
- **Relationships**: Calls get_nearby_zones.
- **Decisions**: Filters by zone_type alignment with trade direction before selecting by strength attribute.

### ZoneManager.maintenance(current_candle: Any) -> None
- **Does**: Runs per-candle housekeeping: updates all active zones with the new candle and prunes old invalidated zones.
- **Why**: Keeps the zone registry current and prevents unbounded growth of stale invalidated zones during long backtests.
- **Relationships**: Calls update_zone; called on every candle close by the backtest/live loop.
- **Decisions**: Invalidated zones are aged by maintenance cycle count and pruned after max_invalidated_age_bars (default 500) to bound memory usage.
