# Database — Intent

## Functions

### DBConfig (class)
- **Does**: Dataclass holding all PostgreSQL connection parameters with defaults sourced from environment variables.
- **Why**: Centralises database configuration so connection details are defined once and injected into the pool.
- **Relationships**: Consumed by DatabasePool.__init__(); dsn() output passed to ThreadedConnectionPool.
- **Decisions**: Environment variable fallback lets production, CI, and local dev use the same code path without config files.

### DBConfig.dsn() -> str
- **Does**: Builds a libpq-compatible DSN string from the stored connection parameters.
- **Why**: psycopg2's ThreadedConnectionPool accepts a DSN string, so this bridges the typed config to the driver.
- **Relationships**: Called by DatabasePool.initialise().
- **Decisions**: None.

### DatabasePool.__init__(config: DBConfig | None) -> None
- **Does**: Stores the database configuration; does not open any connections yet.
- **Why**: Separates construction from initialisation so the pool can be configured before the database is available.
- **Relationships**: Receives DBConfig; pool opened later by initialise().
- **Decisions**: Defaults to a fresh DBConfig (all env-var defaults) when no config is provided.

### DatabasePool.initialise() -> None
- **Does**: Opens the psycopg2 ThreadedConnectionPool with the configured min/max connection count.
- **Why**: Explicit lifecycle method so the application controls exactly when database connections are created.
- **Relationships**: Calls DBConfig.dsn(); creates ThreadedConnectionPool; called once at application startup.
- **Decisions**: Idempotent -- logs a warning and returns if already initialised to prevent double-open bugs.

### DatabasePool.close() -> None
- **Does**: Closes all connections in the pool and releases resources.
- **Why**: Clean shutdown path that prevents connection leaks during application teardown.
- **Relationships**: Called at application shutdown; pairs with initialise().
- **Decisions**: Sets internal pool reference to None so is_open reflects the closed state.

### DatabasePool.get_connection() -> Generator
- **Does**: Context manager that yields a live psycopg2 connection, auto-committing on success or rolling back on exception.
- **Why**: Ensures connections are always returned to the pool and left in a clean transactional state.
- **Relationships**: Called by get_cursor(), health_check(), and application code; uses ThreadedConnectionPool.getconn/putconn.
- **Decisions**: Auto-commit on clean exit and auto-rollback on exception prevent stale transaction state from leaking to the next borrower.

### DatabasePool.get_cursor(cursor_factory) -> Generator
- **Does**: Shorthand context manager that yields a cursor directly, managing the parent connection lifecycle automatically.
- **Why**: Reduces boilerplate for the common pattern of "get connection, open cursor, execute, close both".
- **Relationships**: Calls get_connection(); defaults to RealDictCursor for dict-style row access.
- **Decisions**: RealDictCursor default chosen because most application queries benefit from column-name access over index-based.

### DatabasePool.health_check() -> bool
- **Does**: Executes a lightweight SELECT 1 query to verify the pool and database are operational.
- **Why**: Provides a non-throwing health probe for readiness checks and monitoring endpoints.
- **Relationships**: Calls get_cursor().
- **Decisions**: Returns bool instead of raising so callers can branch on connectivity without try/except.

### DatabasePool.is_open -> bool
- **Does**: Returns True when the pool has been initialised and not yet closed.
- **Why**: Lets application code guard against using a pool that has not been started or has been shut down.
- **Relationships**: Checked before operations in application code.
- **Decisions**: None.

### DatabasePool.config -> DBConfig
- **Does**: Returns the stored database configuration.
- **Why**: Exposes config for diagnostics and logging without allowing mutation.
- **Relationships**: Set during __init__.
- **Decisions**: None.

### init_default_pool(config: DBConfig | None) -> DatabasePool
- **Does**: Initialises a module-level singleton DatabasePool for convenience access.
- **Why**: Provides a global pool for modules that prefer import-and-use over dependency injection.
- **Relationships**: Creates and initialises a DatabasePool; enables get_connection() and get_cursor() module-level functions.
- **Decisions**: Idempotent to prevent accidental double-initialisation.

### get_connection() -> ContextManager
- **Does**: Returns a context manager for a connection from the module-level default pool.
- **Why**: Convenience wrapper so callers can use the global pool without importing DatabasePool.
- **Relationships**: Delegates to _default_pool.get_connection(); requires prior init_default_pool() call.
- **Decisions**: Raises RuntimeError if the default pool has not been initialised rather than silently creating one.

### get_cursor(cursor_factory) -> ContextManager
- **Does**: Returns a context manager for a cursor from the module-level default pool.
- **Why**: Convenience wrapper for the most common database access pattern using the global pool.
- **Relationships**: Delegates to _default_pool.get_cursor(); requires prior init_default_pool() call.
- **Decisions**: Same fail-fast RuntimeError approach as get_connection().

### close_default_pool() -> None
- **Does**: Closes the module-level default pool if it is open and resets the reference to None.
- **Why**: Clean shutdown path for the global pool, typically called during application teardown.
- **Relationships**: Calls DatabasePool.close() on _default_pool.
- **Decisions**: None.

## Schema (schema.sql)

### candles_1m (table)
- **Does**: Stores raw 1-minute OHLCV bars as a TimescaleDB hypertable partitioned by week.
- **Why**: Base time-series table from which all higher timeframes are derived via continuous aggregates.
- **Decisions**: Weekly chunk interval balances query performance against chunk management overhead; compression after 30 days reduces storage for historical data.

### trades (table)
- **Does**: Records every trade entry regardless of source (backtest, live, paper) with full entry/exit details and P&L.
- **Why**: Unified trade log enables cross-source performance comparison and audit trail.
- **Decisions**: Source column with CHECK constraint forces explicit labeling; r_multiple and pnl are nullable because they are only known after exit.

### market_context (table)
- **Does**: Captures a point-in-time snapshot of market regime indicators at the moment a trade decision is evaluated.
- **Why**: Enables pattern-based learning by storing the full context vector alongside each trade for similarity search.
- **Decisions**: 64-dimensional pgvector embedding with HNSW index supports approximate nearest-neighbour queries for pattern matching.

### pattern_signatures (table)
- **Does**: Stores one embedding per recognised pattern, linked to its trade outcome.
- **Why**: Feeds the pattern-matching system that weights new signals by historical performance of similar setups.
- **Decisions**: Separate from market_context to allow multiple patterns per context and independent clustering.

### zones (table)
- **Does**: Persists support/resistance/supply/demand zones across sessions with lifecycle tracking.
- **Why**: Zones are long-lived market structures that need to survive restarts and be queried for confluence scoring.
- **Decisions**: Status lifecycle (active -> tested -> invalidated) prevents stale zones from influencing decisions.

### decisions (table)
- **Does**: Logs every agent decision (enter, skip, exit, partial_exit, modify) with full signal data and edge results as JSONB.
- **Why**: Provides a complete audit trail and feeds the edge_stats materialized view for per-edge performance analysis.
- **Decisions**: JSONB for signal_data and edge_results allows schema-free evolution of edge definitions without migrations.

### edge_stats (materialized view)
- **Does**: Aggregates per-edge win rate, average R, and total P&L across all closed trades.
- **Why**: Gives the agent and operator a live view of which edges contribute most to profitability.
- **Decisions**: Materialized (not regular) view for query performance; CONCURRENTLY refresh via wrapper function avoids locking reads.

### Continuous aggregates (candles_5m, candles_15m, candles_1h, candles_4h)
- **Does**: TimescaleDB continuous aggregate views that automatically derive higher-timeframe OHLCV bars from candles_1m.
- **Why**: Eliminates manual resampling and keeps multi-timeframe analysis consistent with a single source of truth.
- **Decisions**: materialized_only=FALSE enables real-time aggregation so queries always include the latest partial bucket.

## ORM Models (models.py)

### TradeSource (enum)
- **Does**: Constrains trade source to backtest, live, or paper.
- **Why**: Mirrors the SQL CHECK constraint on trades.source so invalid values are caught at the Python layer before reaching the database.
- **Relationships**: Used by Trade dataclass; mapped to/from the source column in the trades table.
- **Decisions**: Inherits str+Enum for direct JSON serialisation without custom encoders.

### TradeDirection (enum)
- **Does**: Constrains trade direction to long or short.
- **Why**: Mirrors the SQL CHECK constraint on trades.direction.
- **Relationships**: Used by Trade dataclass; mapped to/from the direction column in the trades table.
- **Decisions**: str+Enum pattern consistent with all other enums in this module.

### TradeStatus (enum)
- **Does**: Constrains trade lifecycle status to open, partial, closed, or cancelled.
- **Why**: Mirrors the SQL CHECK constraint on trades.status; enforces valid lifecycle transitions at the application layer.
- **Relationships**: Used by Trade dataclass; mapped to/from the status column in the trades table.
- **Decisions**: Includes "partial" for partial exit support alongside the standard open/closed/cancelled states.

### ZoneType (enum)
- **Does**: Constrains zone classification to support, resistance, supply, demand, or pivot.
- **Why**: Mirrors the SQL CHECK constraint on zones.zone_type; prevents free-text zone labels.
- **Relationships**: Used by Zone dataclass; mapped to/from the zone_type column in the zones table.
- **Decisions**: Includes pivot as a distinct type alongside the S/R and supply/demand pairs.

### ZoneStatus (enum)
- **Does**: Constrains zone lifecycle to active, tested, or invalidated.
- **Why**: Mirrors the SQL CHECK constraint on zones.status; enforces the zone lifecycle (active -> tested -> invalidated).
- **Relationships**: Used by Zone dataclass; mapped to/from the status column in the zones table.
- **Decisions**: Three-state lifecycle prevents stale zones from influencing decisions while retaining tested zones for analysis.

### DecisionAction (enum)
- **Does**: Constrains decision action to enter, skip, exit, partial_exit, or modify.
- **Why**: Mirrors the SQL CHECK constraint on decisions.action; provides a fixed vocabulary for the audit trail.
- **Relationships**: Used by Decision dataclass; mapped to/from the action column in the decisions table.
- **Decisions**: Five actions cover all possible agent responses: take the trade, skip it, close it fully, close partially, or modify parameters.

### ScreenshotPhase (enum)
- **Does**: Constrains screenshot phase to pre_entry, entry, during, or exit.
- **Why**: Categorises chart screenshots by trade lifecycle phase for organised visual audit trails.
- **Relationships**: Used by Screenshot dataclass; mapped to/from the phase column in the screenshots table.
- **Decisions**: Four phases capture the complete visual narrative of a trade from analysis through closure.

### Candle (dataclass)
- **Does**: Represents a single OHLCV bar with instrument and timestamp, providing from_row() and to_dict() serialisation.
- **Why**: Typed Python representation of a candles_1m row for use throughout the application without raw dict access.
- **Relationships**: Populated from candles_1m/5m/15m/1h/4h table rows via from_row(); consumed by indicator calculators, signal engine, and backtester.
- **Decisions**: Plain dataclass with no ORM dependency; to_dict() converts timestamp to ISO format for JSON compatibility.

### Trade (dataclass)
- **Does**: Represents a complete trade record with entry/exit details, P&L, and metadata, providing from_row() and to_dict() serialisation.
- **Why**: Central trade data structure shared across backtesting, live execution, logging, and analysis.
- **Relationships**: Populated from trades table rows; referenced by MarketContext, PatternSignature, Screenshot, and Decision via trade_id foreign keys; uses TradeSource, TradeDirection, and TradeStatus enums.
- **Decisions**: Exit-related fields (exit_time, exit_price, r_multiple, pnl) are Optional because they are unknown until the trade closes; defaults status to OPEN for new trades.

### MarketContext (dataclass)
- **Does**: Captures a point-in-time snapshot of all market regime indicators (cloud direction, ADX, ATR, RSI, BB width, zones) at the moment of a trade decision.
- **Why**: Enables pattern-based learning by storing the full context vector alongside each trade for similarity search and performance analysis.
- **Relationships**: Populated from market_context table rows; linked to Trade via trade_id; context_embedding consumed by PatternSignature for nearest-neighbour queries.
- **Decisions**: 64-dimensional context_embedding stored as a float list; pgvector handles the VECTOR type on the database side. String-format embeddings are parsed for test environments without the pgvector adapter.

### PatternSignature (dataclass)
- **Does**: Stores one embedding per recognised pattern, linked to its trade outcome and optional cluster label.
- **Why**: Feeds the pattern-matching system that weights new signals by historical performance of similar setups.
- **Relationships**: Populated from pattern_signatures table rows; linked to MarketContext via context_id and Trade via trade_id; embedding used for approximate nearest-neighbour queries.
- **Decisions**: Separate from MarketContext to allow multiple patterns per context and independent clustering; string-format embeddings are parsed for adapter-free environments.

### Screenshot (dataclass)
- **Does**: Records chart screenshot metadata including file path, trade association, lifecycle phase, and timeframe.
- **Why**: Provides an organised visual audit trail linking chart images to specific trades and phases.
- **Relationships**: Populated from screenshots table rows; linked to Trade via trade_id; uses ScreenshotPhase enum.
- **Decisions**: Stores file_path rather than binary image data to keep the database lightweight; phase is Optional for screenshots not tied to a specific trade event.

### Zone (dataclass)
- **Does**: Represents a support/resistance/supply/demand zone with price bounds, strength score, touch count, and lifecycle status.
- **Why**: Zones are long-lived market structures that need to survive restarts and be queried for confluence scoring.
- **Relationships**: Populated from zones table rows; referenced by ZoneConfluence via zone_id; uses ZoneType and ZoneStatus enums; consumed by zone manager and confluence scorer.
- **Decisions**: Strength defaults to 0.0 and touch_count to 0 for newly detected zones; status defaults to ACTIVE with lifecycle managed externally.

### ZoneConfluence (dataclass)
- **Does**: Records a specific confluence factor (e.g. Fibonacci level, round number, volume cluster) associated with a zone.
- **Why**: Separating confluence factors from zones allows multiple factors per zone and flexible scoring without schema changes.
- **Relationships**: Populated from zone_confluences table rows; linked to Zone via zone_id; consumed by confluence density scorer.
- **Decisions**: confluence_type is a free-form string rather than an enum to allow new confluence types without migrations; value is Optional for binary confluence factors that are present/absent.

### Decision (dataclass)
- **Does**: Logs a complete agent decision record including action taken, signal data, edge filter results, pattern similarity data, and reasoning text.
- **Why**: Provides a complete audit trail for every trade decision and feeds the edge_stats materialised view for per-edge performance analysis.
- **Relationships**: Populated from decisions table rows; linked to Trade via trade_id; uses DecisionAction enum; signal_data and edge_results stored as JSONB dicts.
- **Decisions**: JSONB for signal_data and edge_results allows schema-free evolution of edge definitions without migrations; JSON string parsing handled in from_row() for environments without psycopg2 JSON adapter.
