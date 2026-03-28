# Execution — Intent

## Functions

### MT5Bridge.__init__(self, login: int, password: str, server: str, path: str | None, timeout: int) -> None
- **Does**: Stores MT5 connection credentials and configuration without connecting.
- **Why**: Deferred connection allows the caller to control when the terminal is initialised and supports context manager usage.
- **Relationships**: Used by OrderManager, ScreenshotCapture, and AccountMonitor as their MT5 interface.
- **Decisions**: Lazy MT5 import via _import_mt5() isolates the Windows-only MetaTrader5 package so the rest of the system can be tested on Linux.

### MT5Bridge.connect(self) -> bool
- **Does**: Initialises the MT5 terminal, logs in, and verifies the connection by checking account info.
- **Why**: Separates connection from construction so callers can handle failures and retries explicitly.
- **Relationships**: Must succeed before get_rates(), get_tick(), or any data method is called.
- **Decisions**: Returns bool rather than raising so the caller can decide between retry and abort.

### MT5Bridge.disconnect(self) -> None
- **Does**: Shuts down the MT5 connection gracefully.
- **Why**: Ensures MT5 terminal resources are released cleanly on shutdown.
- **Relationships**: Called by __exit__ for context manager support; called explicitly on agent shutdown.
- **Decisions**: None.

### MT5Bridge.get_rates(self, instrument: str, timeframe: int, count: int) -> pd.DataFrame
- **Does**: Retrieves OHLCV bars from MT5 and returns a cleaned DataFrame with UTC timestamps.
- **Why**: Provides the primary market data feed for the signal engine and indicator calculations.
- **Relationships**: Called by get_multi_tf_rates() and ScreenshotCapture._fallback_chart().
- **Decisions**: Normalises MT5 numpy types to Python floats and drops unused columns (real_volume, spread) for downstream safety.

### MT5Bridge.get_multi_tf_rates(self, instrument: str, timeframes: dict[str, int] | None, count: int) -> dict[str, pd.DataFrame]
- **Does**: Retrieves OHLCV data for all required timeframes (5M, 15M, 1H, 4H) in one call.
- **Why**: The signal engine needs multiple timeframes simultaneously; batching the calls simplifies the caller.
- **Relationships**: Calls get_rates() for each timeframe; called by the trading engine's data pipeline.
- **Decisions**: Defaults to the four timeframes used by the Ichimoku signal engine when none are specified.

### MT5Bridge.get_tick(self, instrument: str) -> dict
- **Does**: Returns the current best bid, ask, spread, and time for an instrument.
- **Why**: Provides real-time price data needed for market order execution and spread filter checks.
- **Relationships**: Called by OrderManager.market_order() and close_position() for fill pricing.
- **Decisions**: None.

### MT5Bridge.get_symbol_info(self, instrument: str) -> dict
- **Does**: Returns static symbol properties including point, digits, tick_value, and volume constraints.
- **Why**: Needed for lot size normalisation, filling mode detection, and order construction.
- **Relationships**: Called by OrderManager._normalize_volume() and _detect_filling_mode().
- **Decisions**: Includes freeze_level and stops_level via getattr with defaults for brokers that do not report them.

### MT5Bridge.timeframe_constant(self, label: str) -> int | None
- **Does**: Converts a string label like "1H" to the corresponding MT5 TIMEFRAME integer constant.
- **Why**: Bridges the string-based timeframe labels used in config with MT5's integer constants.
- **Relationships**: Called by ScreenshotCapture for chart rendering.
- **Decisions**: None.

### MT5Bridge.is_connected (property) -> bool
- **Does**: Returns True while an active MT5 session is established.
- **Why**: Allows callers to check connection state before attempting operations.
- **Relationships**: Checked by get_rates(), get_tick(), get_symbol_info().
- **Decisions**: None.

### MT5Bridge.mt5 (property)
- **Does**: Returns a direct reference to the MetaTrader5 module.
- **Why**: OrderManager needs access to MT5 constants (ORDER_TYPE_BUY, TRADE_ACTION_DEAL, etc.) for order construction.
- **Relationships**: Used by OrderManager and AccountMonitor for MT5 API calls.
- **Decisions**: Exposed deliberately rather than wrapping every MT5 constant, avoiding an explosion of bridge methods.

### OrderResult (dataclass)
- **Does**: Holds the outcome of an order_send() request including success, ticket, price, volume, retcode, slippage, and error.
- **Why**: Provides a structured result type so callers do not need to parse raw MT5 responses.
- **Relationships**: Returned by all OrderManager order methods; consumed by the trading engine.
- **Decisions**: Tracks slippage unconditionally (even on success) for post-trade analysis.

### OrderManager.__init__(self, bridge: MT5Bridge, deviation: int) -> None
- **Does**: Stores the MT5Bridge reference and maximum slippage deviation, initialises the slippage log.
- **Why**: Centralises all order execution logic behind a single interface.
- **Relationships**: Depends on MT5Bridge for all MT5 interactions.
- **Decisions**: Default deviation of 20 points balances fill probability against slippage cost for gold CFDs.

### OrderManager.market_order(self, instrument: str, direction: str, lot_size: float, stop_loss: float, take_profit: float, comment: str) -> OrderResult
- **Does**: Executes an immediate-fill market order with SL/TP via MT5.
- **Why**: Primary entry method for the Ichimoku strategy when immediate execution is required.
- **Relationships**: Calls _normalize_volume(), _detect_filling_mode(), bridge.get_tick(), _parse_result().
- **Decisions**: Uses TRADE_ACTION_DEAL for instant execution; auto-detects filling mode per symbol.

### OrderManager.limit_order(self, instrument: str, direction: str, lot_size: float, price: float, stop_loss: float, take_profit: float, comment: str) -> OrderResult
- **Does**: Places a pending limit order at a specified price for Kijun pullback entries.
- **Why**: Supports the Kijun pullback entry type where the order rests until price retraces.
- **Relationships**: Calls _normalize_volume(), _detect_filling_mode(), _parse_result().
- **Decisions**: Uses TRADE_ACTION_PENDING with ORDER_TYPE_BUY_LIMIT or SELL_LIMIT.

### OrderManager.stop_limit_order(self, instrument: str, direction: str, lot_size: float, stop_price: float, limit_price: float, stop_loss: float, take_profit: float, comment: str) -> OrderResult
- **Does**: Places a stop-limit order for cloud breakout entries with a trigger price and maximum fill price.
- **Why**: Prevents chasing breakouts at bad prices by setting a worst-acceptable fill level.
- **Relationships**: Calls _normalize_volume(), _detect_filling_mode(), _parse_result().
- **Decisions**: Uses ORDER_TYPE_BUY_STOP_LIMIT or SELL_STOP_LIMIT with the stoplimit field for the limit price.

### OrderManager.modify_position(self, ticket: int, stop_loss: float | None, take_profit: float | None) -> bool
- **Does**: Modifies the SL and/or TP of an open position for trailing-stop updates.
- **Why**: Supports the Kijun-Sen trailing stop mechanism without closing and re-opening positions.
- **Relationships**: Called by the trading engine when TradeManager.update_trade() returns a trail_update.
- **Decisions**: Fetches current position to preserve unchanged SL/TP fields rather than requiring both.

### OrderManager.close_position(self, ticket: int, lot_size: float | None) -> OrderResult
- **Does**: Closes a position fully or partially by sending an opposite market order.
- **Why**: Supports both full exits and the 50% partial exit in the hybrid exit strategy.
- **Relationships**: Called by close_partial(); uses bridge.get_tick() for exit pricing.
- **Decisions**: MT5 closes positions via opposite-direction TRADE_ACTION_DEAL against the position ticket.

### OrderManager.close_partial(self, ticket: int, close_pct: float) -> OrderResult
- **Does**: Closes a percentage of an open position by calculating the partial volume and delegating to close_position().
- **Why**: Implements the 50% partial close at 2R in the hybrid exit strategy.
- **Relationships**: Calls close_position() with the calculated partial volume.
- **Decisions**: Default close_pct=0.5 matches the hybrid 50/50 exit design.

### OrderManager.slippage_log (property) -> list[dict]
- **Does**: Returns a read-only copy of all recorded slippage entries.
- **Why**: Enables post-trade slippage analysis for execution quality monitoring.
- **Relationships**: Populated by _log_slippage() on every order; consumed by analytics.
- **Decisions**: None.

### AccountInfo (dataclass)
- **Does**: Snapshot of the MT5 account state including balance, equity, margin, free_margin, unrealized_pnl, and leverage.
- **Why**: Provides a typed, immutable account snapshot for the circuit breaker and risk management layer.
- **Relationships**: Returned by AccountMonitor.get_account_info(); consumed by the trading engine.
- **Decisions**: None.

### PositionInfo (dataclass)
- **Does**: Represents the state of a single open MT5 position with ticket, instrument, direction, volume, prices, and P&L.
- **Why**: Provides typed position data for monitoring and reconciliation with TradeManager's internal state.
- **Relationships**: Returned by AccountMonitor.get_positions().
- **Decisions**: None.

### AccountMonitor.__init__(self, bridge: MT5Bridge) -> None
- **Does**: Stores the MT5Bridge reference for account and position queries.
- **Why**: Provides a monitoring layer that the circuit breaker and trading engine can poll without duplicating MT5 calls.
- **Relationships**: Depends on MT5Bridge; consumed by the trading engine and DailyCircuitBreaker.
- **Decisions**: None.

### AccountMonitor.get_account_info(self) -> AccountInfo | None
- **Does**: Fetches current account equity, balance, and margin from MT5.
- **Why**: Provides the real-time equity data that the circuit breaker needs to decide whether to halt trading.
- **Relationships**: Called on every bar by the trading engine; feeds DailyCircuitBreaker.can_trade().
- **Decisions**: Returns None rather than raising on error so the caller can handle gracefully.

### AccountMonitor.get_positions(self, instrument: str | None) -> list[PositionInfo]
- **Does**: Returns all open positions from MT5, optionally filtered by instrument.
- **Why**: Enables reconciliation between the trading engine's internal state and MT5's actual positions.
- **Relationships**: Queries MT5 positions_get(); consumed by the trading engine for state verification.
- **Decisions**: None.

### AccountMonitor.get_open_orders(self, instrument: str | None) -> list[dict]
- **Does**: Returns all pending (unfilled) orders from MT5, optionally filtered by instrument.
- **Why**: Allows the trading engine to track pending limit and stop-limit orders that have not yet filled.
- **Relationships**: Queries MT5 orders_get().
- **Decisions**: None.

### AccountMonitor.get_daily_pnl(self) -> float
- **Does**: Calculates today's realised P&L from the MT5 deal history.
- **Why**: Provides an independent P&L check from the broker's records rather than relying on internal tracking.
- **Relationships**: Queries MT5 history_deals_get() with today's date range.
- **Decisions**: Sums all deal profits including commissions; returns 0.0 on error.

### AccountMonitor.equity (property) -> float
- **Does**: Returns current account equity, or 0.0 on error.
- **Why**: Convenience shortcut for circuit-breaker polling that avoids unpacking AccountInfo.
- **Relationships**: Calls get_account_info() internally.
- **Decisions**: None.

### AccountMonitor.balance (property) -> float
- **Does**: Returns current account balance, or 0.0 on error.
- **Why**: Convenience shortcut for balance checks without full AccountInfo unpacking.
- **Relationships**: Calls get_account_info() internally.
- **Decisions**: None.

### ScreenshotCapture.__init__(self, bridge: MT5Bridge, save_dir: str, width: int, height: int) -> None
- **Does**: Configures the screenshot capture with bridge reference, output directory, and image dimensions.
- **Why**: Provides visual trade documentation at key lifecycle events for post-trade review.
- **Relationships**: Depends on MT5Bridge for both native screenshots and fallback chart data.
- **Decisions**: Default 1280x720 resolution balances detail with file size.

### ScreenshotCapture.capture(self, instrument: str, timeframe: str, phase: str, trade_id: int | None) -> str
- **Does**: Captures a chart screenshot, trying MT5 native API first and falling back to matplotlib.
- **Why**: Documents the chart state at entry, during, and exit phases for trade review and learning.
- **Relationships**: Calls _try_mt5_screenshot() then _fallback_chart(); called by the trading engine at trade lifecycle events.
- **Decisions**: Dual-path approach (MT5 native + matplotlib fallback) ensures screenshots work in Docker environments without a GUI terminal.
