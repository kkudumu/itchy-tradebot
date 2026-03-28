# Engine — Intent

## Functions

### Decision(timestamp: datetime, instrument: str, action: str, signal: Optional[object], edge_results: Dict, similarity_data: Dict, confluence_score: int, reasoning: str, trade_id: Optional[int], executed: bool, execution_detail: Dict) -> Decision
- **Does**: Dataclass that holds the full record of one scan cycle's reasoning and outcome.
- **Why**: Provides a structured, immutable trace of every decision the engine makes, including skips, so the learning loop can analyse all outcomes.
- **Relationships**: Created by DecisionEngine.scan and DecisionEngine._manage_open_trades; consumed by EngineTradeLogger.log_decision.
- **Decisions**: Uses a flat dataclass rather than nested objects so it serialises cleanly to JSON/DB.

### DecisionEngine.__init__(self, config: dict, signal_engine, edge_manager, trade_manager, similarity_search, embedding_engine, zone_manager, mt5_bridge, order_manager, account_monitor, screenshot_capture, db_pool) -> None
- **Does**: Initialises the central orchestrator by wiring together all injected dependencies and setting the operating mode (live or backtest).
- **Why**: Acts as the composition root that connects signal detection, edge filtering, similarity search, risk management, and execution into a single pipeline.
- **Relationships**: Creates EngineTradeLogger and ScanScheduler internally; receives all other components via dependency injection.
- **Decisions**: Determines live vs backtest mode based on whether mt5_bridge is provided.

### DecisionEngine.start(self) -> None
- **Does**: Starts the main trading loop, either entering the scheduler's blocking run_loop (live) or marking the engine ready for caller-driven iteration (backtest).
- **Why**: Provides a unified entry point for both live and backtest execution modes.
- **Relationships**: Calls ScanScheduler.run_loop in live mode; in backtest mode the caller drives via scan().
- **Decisions**: Live mode blocks the calling thread; backtest mode is non-blocking by design.

### DecisionEngine.stop(self) -> None
- **Does**: Signals the scheduler loop to exit and flushes pending log entries.
- **Why**: Ensures graceful shutdown so no trade data is lost during termination.
- **Relationships**: Sets _stop_event consumed by ScanScheduler.run_loop; calls EngineTradeLogger.flush.
- **Decisions**: None.

### DecisionEngine.scan(self, data: Optional[dict]) -> Decision
- **Does**: Runs one complete scan cycle through the full pipeline: data retrieval, zone maintenance, open trade management, signal detection, edge filtering, similarity search, confidence adjustment, sizing, execution, and logging.
- **Why**: This is the engine's core method that orchestrates all trading logic into a single deterministic pass per candle close.
- **Relationships**: Calls signal_engine.scan, edge_manager.check_entry, similarity_search.find_similar_trades, trade_manager.can_open_trade/open_trade, _execute_trade, and EngineTradeLogger.log_decision.
- **Decisions**: Logs every decision including skips, so the learning loop has complete data.

### ScanScheduler.__init__(self, interval_minutes: int) -> None
- **Does**: Creates a scheduler aligned to candle-close boundaries for a given minute interval.
- **Why**: Ensures the engine always evaluates fully formed candles rather than in-progress ones.
- **Relationships**: Used by DecisionEngine to time live scan cycles.
- **Decisions**: Validates that interval_minutes is positive.

### ScanScheduler.wait_for_next_close(self) -> datetime
- **Does**: Blocks the calling thread until the next clock-aligned candle-close boundary.
- **Why**: Aligns scan cycles to standard candle intervals (e.g. :00, :05, :10) so indicator calculations use complete candles.
- **Relationships**: Called by run_loop on each iteration.
- **Decisions**: Adds a 1-second buffer past the boundary to avoid firing on an incomplete candle.

### ScanScheduler.is_market_open(self, dt: Optional[datetime]) -> bool
- **Does**: Returns True when forex/spot-gold markets are open (Sunday 22:00 UTC through Friday 22:00 UTC).
- **Why**: Prevents the engine from scanning during market-closed periods when no price data updates.
- **Relationships**: Called by run_loop to skip closed-market periods.
- **Decisions**: Uses forex market hours since XAU/USD trades on the forex schedule.

### ScanScheduler.run_loop(self, callback: Callable, stop_event: Optional[threading.Event]) -> None
- **Does**: Continuously waits for candle closes and invokes the callback, skipping market-closed periods, until the stop event is set.
- **Why**: Provides the main execution loop for live trading with resilient error handling so one bad scan never halts the agent.
- **Relationships**: Calls wait_for_next_close and is_market_open; invokes the DecisionEngine._live_scan_callback.
- **Decisions**: Catches and logs callback exceptions without terminating the loop.

### EngineTradeLogger.__init__(self, db_pool, embedding_engine) -> None
- **Does**: Creates a trade logger with optional database persistence and embedding generation.
- **Why**: Centralises all logging for decisions, trades, screenshots, and zones so the learning system has complete data.
- **Relationships**: Used by DecisionEngine; optionally uses EmbeddingEngine for context embeddings.
- **Decisions**: Maintains in-memory buffers as fallback when db_pool is None.

### EngineTradeLogger.log_decision(self, decision: Any) -> Optional[int]
- **Does**: Inserts a full decision record into the decisions table, including edge results, similarity data, and signal details.
- **Why**: Logs every decision (including skips) so the adaptive learning engine can analyse signal quality and filter effectiveness.
- **Relationships**: Called by DecisionEngine._log_decision; writes to the decisions DB table.
- **Decisions**: DB failures are caught and logged at WARNING level to never interrupt the trading pipeline.

### EngineTradeLogger.log_trade_entry(self, trade: dict, context: dict) -> Optional[int]
- **Does**: Inserts a trade entry record and its market context embedding into the trades and market_context tables.
- **Why**: Persists full trade metadata and the 64-dim context embedding needed for future similarity searches.
- **Relationships**: Called by DecisionEngine._process_signal; uses EmbeddingEngine.create_embedding.
- **Decisions**: Embedding generation failures do not prevent trade logging.

### EngineTradeLogger.log_trade_exit(self, trade_id: int, exit_data: dict) -> None
- **Does**: Updates a trade record with exit price, time, R-multiple, and reason, marking it as closed.
- **Why**: Completes the trade lifecycle record so performance statistics can be computed.
- **Relationships**: Called by DecisionEngine._manage_open_trades; updates the trades DB table.
- **Decisions**: Matches on both internal ID and broker ticket for flexibility.

### EngineTradeLogger.log_screenshot(self, trade_id: int, phase: str, filepath: str) -> None
- **Does**: Records a screenshot file reference in the trade_screenshots table for a given trade phase.
- **Why**: Links visual chart captures to trade records for post-trade review and pattern analysis.
- **Relationships**: Called by DecisionEngine at pre_entry, entry, and exit phases.
- **Decisions**: None.

### EngineTradeLogger.log_zone_update(self, zone_changes: List[dict]) -> None
- **Does**: Batch-inserts zone state change events into the zone_events table.
- **Why**: Tracks supply/demand zone lifecycle changes for zone-based analysis and debugging.
- **Relationships**: Called by the engine during zone maintenance cycles.
- **Decisions**: None.

### EngineTradeLogger.refresh_edge_stats(self) -> None
- **Does**: Refreshes the edge_stats materialized view concurrently in the database.
- **Why**: Keeps pre-computed edge performance statistics up to date for fast query access.
- **Relationships**: Called periodically by DecisionEngine._maybe_refresh_edge_stats every 50 trades or 1 hour.
- **Decisions**: Uses CONCURRENTLY to avoid locking reads during refresh.

### EngineTradeLogger.flush(self) -> None
- **Does**: Flushes any pending in-memory log records (advisory only in production).
- **Why**: Ensures clean shutdown and provides a hook for test assertions on buffered data.
- **Relationships**: Called by DecisionEngine.stop.
- **Decisions**: Production writes go directly to DB; buffer is primarily for test introspection.

### EngineTradeLogger.decision_buffer -> List[dict]
- **Does**: Returns a read-only snapshot of the in-memory decision buffer.
- **Why**: Allows tests to inspect logged decisions without requiring a database connection.
- **Relationships**: Used by test assertions.
- **Decisions**: Returns a copy to prevent external mutation.

### EngineTradeLogger.trade_buffer -> List[dict]
- **Does**: Returns a read-only snapshot of the in-memory trade entry buffer.
- **Why**: Allows tests to inspect logged trade entries without requiring a database connection.
- **Relationships**: Used by test assertions.
- **Decisions**: Returns a copy to prevent external mutation.
