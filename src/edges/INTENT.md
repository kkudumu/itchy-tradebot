# Edges — Intent

## Functions

### EdgeContext (dataclass)
- **Does**: Carries all per-bar market data and trade state needed by every edge filter.
- **Why**: Decouples edge filters from the trading engine by providing a single, stable input contract.
- **Relationships**: Populated by the trading engine, consumed by all EdgeFilter subclasses via should_allow().
- **Decisions**: Optional fields (current_r, candles_since_entry, signal) are None for entry checks to distinguish entry vs active-trade evaluation contexts.

### EdgeResult (dataclass)
- **Does**: Standardised (allowed, reason) return value from every edge filter, with an optional modifier float.
- **Why**: Gives the EdgeManager a uniform type to collect, log, and act on decisions from heterogeneous filters.
- **Relationships**: Returned by EdgeFilter.should_allow(), consumed by EdgeManager pipeline methods.
- **Decisions**: The modifier field supports MODIFIER-type edges (size adjustment) without needing a separate return type.

### EdgeFilter.__init__(self, name: str, config: dict) -> None
- **Does**: Stores the edge name, raw config dict, and derives the enabled flag.
- **Why**: Provides a consistent construction contract so EdgeManager can instantiate any filter uniformly from config.
- **Relationships**: Called by all subclass constructors; config originates from EdgeConfig Pydantic model or plain dict.
- **Decisions**: Defaults enabled to True so edges are active unless explicitly disabled.

### EdgeFilter.should_allow(self, context: EdgeContext) -> EdgeResult
- **Does**: Abstract method that evaluates whether this edge permits the current action.
- **Why**: Defines the single evaluation contract that all 12 edge filters must implement.
- **Relationships**: Called by EdgeManager.check_entry(), check_exit(), get_modifiers().
- **Decisions**: Returns allowed=True to permit, allowed=False to veto; modifier edges always return allowed=True with a modifier float.

### EdgeFilter._disabled_result(self) -> EdgeResult
- **Does**: Returns a pass-through EdgeResult when the edge is disabled.
- **Why**: Avoids duplicating disabled-check boilerplate in every subclass.
- **Relationships**: Called by subclass should_allow() implementations when self.enabled is False.
- **Decisions**: None.

### EdgeManager.__init__(self, edge_configs: Any, news_calendar: NewsCalendar | None) -> None
- **Does**: Loads all 12 edge filters from config and sorts them into entry, exit, and modifier lists.
- **Why**: Centralises edge construction and categorisation so the trading engine has a single orchestration point.
- **Relationships**: Instantiates all EdgeFilter subclasses; injects NewsCalendar into NewsFilter if provided.
- **Decisions**: Uses a static _REGISTRY dict to map config keys to (constructor, category) tuples for extensibility.

### EdgeManager.check_entry(self, context: EdgeContext) -> tuple[bool, list[EdgeResult]]
- **Does**: Runs all entry edge filters in sequence, short-circuiting on the first failure.
- **Why**: Gates new position entries; short-circuiting makes the rejection reason unambiguous in logs.
- **Relationships**: Calls should_allow() on each entry edge; called by the trading engine before opening a trade.
- **Decisions**: Short-circuits on first failure rather than evaluating all edges, prioritising clarity of rejection reason.

### EdgeManager.check_exit(self, context: EdgeContext) -> tuple[bool, list[EdgeResult]]
- **Does**: Runs all exit edge filters without short-circuiting and returns whether any triggered.
- **Why**: Exit signals should all be evaluated so the full picture is captured for logging and analysis.
- **Relationships**: Calls should_allow() on each exit edge; called by the trading engine on every bar with an open position.
- **Decisions**: No short-circuit because multiple exit reasons can be relevant simultaneously.

### EdgeManager.get_modifiers(self, context: EdgeContext) -> dict[str, float]
- **Does**: Evaluates all modifier edges and returns a dict of edge name to modifier float.
- **Why**: Allows position size adjustments without blocking trades, supporting the modifier edge category.
- **Relationships**: Calls should_allow() on each modifier edge; called by get_combined_size_multiplier().
- **Decisions**: Disabled modifiers default to 1.0 (neutral) rather than being omitted.

### EdgeManager.get_combined_size_multiplier(self, context: EdgeContext) -> float
- **Does**: Combines confluence_scoring and equity_curve modifiers multiplicatively into a single [0.0, 1.0] multiplier.
- **Why**: Provides the position sizer with one number representing all modifier-edge adjustments.
- **Relationships**: Calls get_modifiers(); called by the trading engine's sizing logic.
- **Decisions**: bb_squeeze is excluded from multiplicative combination because it is an additive confluence boost, not a direct size scaler.

### EdgeManager.get_enabled_edges(self) -> list[str]
- **Does**: Returns the names of all currently enabled edges.
- **Why**: Supports diagnostics, logging, and runtime inspection of the active filter set.
- **Relationships**: Reads self._all_edges; used by __repr__ and external callers.
- **Decisions**: None.

### EdgeManager.toggle_edge(self, edge_name: str, enabled: bool) -> None
- **Does**: Enables or disables a named edge at runtime.
- **Why**: Allows the learning loop or operator to adjust the active filter set without restarting.
- **Relationships**: Mutates EdgeFilter.enabled on the target edge; raises KeyError for unknown names.
- **Decisions**: None.

### TimeOfDayFilter.__init__(self, config: dict) -> None
- **Does**: Parses start_utc and end_utc from config into minutes-since-midnight for fast comparison.
- **Why**: Gold has poor liquidity outside London/NY; restricting entries to 08:00-17:00 UTC reduces low-quality setups.
- **Relationships**: Inherits EdgeFilter; instantiated by EdgeManager via _REGISTRY.
- **Decisions**: Default window 08:00-17:00 UTC based on gold session analysis.

### TimeOfDayFilter.should_allow(self, context: EdgeContext) -> EdgeResult
- **Does**: Allows entries only when the bar timestamp falls within the configured UTC time window.
- **Why**: Prevents entries during low-liquidity hours where directional follow-through is poor.
- **Relationships**: Called by EdgeManager.check_entry().
- **Decisions**: None.

### NewsFilter.__init__(self, config: dict, calendar: NewsCalendar | None) -> None
- **Does**: Configures blackout windows around high-impact news events and sets the calendar source.
- **Why**: High-impact news causes unpredictable spikes that invalidate technical setups.
- **Relationships**: Inherits EdgeFilter; uses NewsCalendar protocol for event data; instantiated by EdgeManager.
- **Decisions**: Defaults to EmptyCalendar (pass-through) when no real calendar is provided, keeping the filter active but inert.

### NewsFilter.should_allow(self, context: EdgeContext) -> EdgeResult
- **Does**: Blocks entries within N minutes before and after any high-impact news event.
- **Why**: Prevents entering during news-driven volatility that widens spreads and invalidates setups.
- **Relationships**: Queries self._calendar.get_events(); called by EdgeManager.check_entry().
- **Decisions**: Only blocks for impact levels in the configured set (default: ["red"] only).

### NewsFilter.set_calendar(self, calendar: NewsCalendar) -> None
- **Does**: Replaces the calendar implementation at runtime.
- **Why**: Allows injecting a live calendar feed after construction, since the calendar may not be available at init time.
- **Relationships**: Called by EdgeManager.__init__() when a news_calendar is provided.
- **Decisions**: None.

### SpreadFilter.__init__(self, config: dict) -> None
- **Does**: Reads max_spread_points from config, defaulting to 30 points.
- **Why**: Wide spreads erode trade edge and indicate abnormal market conditions.
- **Relationships**: Inherits EdgeFilter; instantiated by EdgeManager via _REGISTRY.
- **Decisions**: Default 30 points based on normal XAU/USD spread (~0.30 USD/oz per lot).

### SpreadFilter.should_allow(self, context: EdgeContext) -> EdgeResult
- **Does**: Blocks entries when the current spread exceeds the configured maximum.
- **Why**: Ensures fill cost does not erode the setup's expected edge.
- **Relationships**: Called by EdgeManager.check_entry().
- **Decisions**: None.

### EquityCurveFilter.__init__(self, config: dict) -> None
- **Does**: Reads lookback_trades and reduced_size_multiplier from config.
- **Why**: Reduces position size during losing streaks to limit drawdown while keeping the strategy active.
- **Relationships**: Inherits EdgeFilter; instantiated by EdgeManager via _REGISTRY as a modifier edge.
- **Decisions**: Default lookback of 20 trades and 50% size reduction when equity curve is below its MA.

### EquityCurveFilter.should_allow(self, context: EdgeContext) -> EdgeResult
- **Does**: Returns modifier=1.0 when equity curve is above its MA, or reduced_size_multiplier when below.
- **Why**: Adapts position size to recent performance without blocking trades entirely.
- **Relationships**: Called by EdgeManager.get_modifiers(); reads context.equity_curve.
- **Decisions**: Always returns allowed=True (never blocks); requires at least 2 data points to compute a meaningful MA.

### NewsCalendar (Protocol)
- **Does**: Defines the minimal interface for economic calendar data sources with a single get_events(date) method.
- **Why**: Decouples the news filter from any specific calendar implementation so it can work with live feeds, YAML files, or mocks.
- **Relationships**: Implemented by EmptyCalendar; consumed by NewsFilter.
- **Decisions**: Deliberately minimal single-method protocol for maximum flexibility.

### EmptyCalendar.get_events(self, date: datetime) -> list[NewsEvent]
- **Does**: Returns an empty list, reporting no news events.
- **Why**: Placeholder calendar that keeps the filter enabled but effectively a no-op until a real calendar is injected.
- **Relationships**: Default calendar used by NewsFilter when none is configured.
- **Decisions**: None.

### NewsEvent (dataclass)
- **Does**: Represents a single economic news event with timestamp, title, and impact level.
- **Why**: Provides typed event data for the NewsFilter to check against blackout windows.
- **Relationships**: Returned by NewsCalendar.get_events(); consumed by NewsFilter.should_allow().
- **Decisions**: Impact defaults to "red" (high) so events are treated as high-impact unless explicitly marked otherwise.

### DayOfWeekFilter (day_of_week.py)
- **Does**: Allows entries only on configured weekdays (default Tuesday-Thursday, indices 1-3).
- **Why**: Monday and Friday exhibit reduced directional consistency in gold due to pre-weekend positioning and range-setting behaviour; restricting to mid-week improves signal quality.
- **Relationships**: Inherits EdgeFilter; instantiated by EdgeManager via _REGISTRY as an entry edge; reads context.day_of_week.
- **Decisions**: Default allowed days [1, 2, 3] (Tue-Thu) based on gold intraweek performance analysis; uses a set for O(1) membership checks.

### BBSqueezeAmplifier (bb_squeeze.py)
- **Does**: Boosts the confluence score by a configurable amount when the Bollinger Bands are expanding out of a squeeze state.
- **Why**: Compressed volatility (BB squeeze) often precedes directional breakouts; amplifying signals during this condition captures higher-probability setups.
- **Relationships**: Inherits EdgeFilter; instantiated by EdgeManager via _REGISTRY as a modifier edge; reads context.bb_squeeze from BollingerBandCalculator output; consumed by EdgeManager.get_modifiers().
- **Decisions**: Returns an additive integer boost via the modifier field (not multiplicative like other modifiers); always returns allowed=True since it never blocks entries. Excluded from EdgeManager.get_combined_size_multiplier() because it is an additive confluence boost, not a direct size scaler.

### CandleCloseConfirmationFilter (candle_close_confirmation.py)
- **Does**: Requires the H1 candle body to close beyond the Ichimoku cloud boundary before allowing a breakout entry.
- **Why**: Prevents entering on intrabar wicks that retrace back into the cloud, which is a common trap in breakout trading.
- **Relationships**: Inherits EdgeFilter; instantiated by EdgeManager via _REGISTRY as an entry edge; reads context.close_price, context.kijun_value, and context.signal.direction.
- **Decisions**: Defaults to allowing the trade (pass-through) when no signal direction is available, so the filter is inert for non-breakout contexts rather than blocking.

### ConfluenceScoringFilter (confluence_scoring.py)
- **Does**: Enforces a minimum confluence score before allowing entry and maps the score to a tiered position size multiplier (A+ = 1.0, B = 0.75, C = 0.50).
- **Why**: Higher-quality setups should receive full risk allocation while lower-quality setups are reduced to conserve capital for high-conviction opportunities.
- **Relationships**: Inherits EdgeFilter; instantiated by EdgeManager via _REGISTRY as both an entry edge (blocks below min_score) and a modifier edge (returns size multiplier); reads context.confluence_score; consumed by EdgeManager.get_combined_size_multiplier().
- **Decisions**: Dual role as both binary filter and modifier avoids needing two separate edges for the same concept; tier thresholds are configurable to allow strategy tuning.

### FridayCloseFilter (friday_close.py)
- **Does**: Signals that open positions should be closed on Friday at or after a configured UTC time (default 20:00).
- **Why**: Holding positions over the weekend exposes the account to gap risk from geopolitical events and weekend illiquidity.
- **Relationships**: Inherits EdgeFilter; instantiated by EdgeManager via _REGISTRY as an exit edge; reads context.day_of_week and context.timestamp; called by EdgeManager.check_exit().
- **Decisions**: Returns allowed=False to trigger exit (inverted semantics consistent with exit edge convention); configurable close day and time to accommodate different broker schedules.

### LondonOpenDelayFilter (london_open_delay.py)
- **Does**: Blocks entries within N minutes (default 30) of the London session open (default 08:00 UTC).
- **Why**: The first 30 minutes after London opens are characterised by erratic spread widening, stop-hunting, and false breakouts as liquidity is being established.
- **Relationships**: Inherits EdgeFilter; instantiated by EdgeManager via _REGISTRY as an entry edge; reads context.timestamp for bar time.
- **Decisions**: Default 30-minute blackout window based on observed gold market behaviour around the London open; uses minutes-since-midnight arithmetic for fast comparison.

### RegimeFilter (regime_filter.py)
- **Does**: Requires both ADX above a threshold (default 28) and cloud thickness above a minimum before allowing entry.
- **Why**: Ichimoku signals perform best in genuine trending regimes; ADX confirms directional momentum while cloud thickness confirms meaningful support/resistance structure.
- **Relationships**: Inherits EdgeFilter; instantiated by EdgeManager via _REGISTRY as an entry edge; reads context.adx and context.cloud_thickness.
- **Decisions**: Both conditions must be satisfied (AND logic) for a conservative regime gate; cloud_thickness_percentile parameter can be used as either a raw absolute threshold or a pre-normalised percentile rank depending on caller convention.

### TimeStopFilter (time_stop.py)
- **Does**: Triggers a breakeven exit when a trade has been open for at least N candles (default 12) and the unrealised profit is below a configurable R threshold (default 0.5R).
- **Why**: A trade that stalls without reaching minimum profit is unlikely to reach the full target; exiting at breakeven preserves capital and prevents small winners from becoming losers.
- **Relationships**: Inherits EdgeFilter; instantiated by EdgeManager via _REGISTRY as an exit edge; reads context.candles_since_entry and context.current_r; called by EdgeManager.check_exit().
- **Decisions**: Requires BOTH time and profit conditions to trigger (not just time alone), so trades that are progressing are not prematurely exited; passes through when trade state fields are absent so the filter is inert for non-trade contexts.
