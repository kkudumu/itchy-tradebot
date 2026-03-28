# Backtesting — Intent

## Functions

### BacktestResult (dataclass)
- **Does**: Holds the complete output of a backtest run including trade log, performance metrics, equity curve, prop firm status, and daily P&L.
- **Why**: Bundles all backtest artifacts into a single typed object so downstream consumers (dashboard, validation, reports) have a consistent interface.
- **Relationships**: Produced by IchimokuBacktester.run, consumed by BacktestDashboard, GoNoGoValidator, export utilities.
- **Decisions**: Includes skipped_signals and total_signals counters for edge filter diagnostics.

### IchimokuBacktester.__init__(self, config: Optional[dict], initial_balance: float, point_value: float, max_daily_loss_pct: float, prop_firm_profit_target_pct: float, prop_firm_max_daily_dd_pct: float, prop_firm_max_total_dd_pct: float, prop_firm_time_limit_days: int, db_pool) -> None
- **Does**: Assembles all strategy, risk, learning, and logging components from configuration, wiring SignalEngine, EdgeManager, TradeManager, AdaptiveLearningEngine, PropFirmTracker, and TradeLogger.
- **Why**: Centralizes the full backtester dependency graph in one constructor so run() operates on pre-configured components.
- **Relationships**: Creates SignalEngine, EdgeManager, TradeManager, AdaptivePositionSizer, DailyCircuitBreaker, HybridExitManager, BacktestDataPreparer, TradeLogger, AdaptiveLearningEngine, PropFirmTracker, PerformanceMetrics.
- **Decisions**: Uses Portfolio.from_orders() architecture (not from_signals()) because partial exits and Kijun trailing cannot be modeled with from_signals().

### IchimokuBacktester.run(self, candles_1m: pd.DataFrame, instrument: str, log_trades: bool, enable_learning: bool, enable_screenshots: bool, screenshot_dir: str, live_dashboard) -> BacktestResult
- **Does**: Executes a full event-driven backtest by iterating every 5M bar, scanning for signals, managing open trades with exits and partial closes, and tracking prop firm constraints.
- **Why**: Simulates realistic trading with all strategy rules, risk management, and prop firm constraints applied bar-by-bar as they would be in live trading.
- **Relationships**: Calls BacktestDataPreparer.prepare, SignalEngine, EdgeManager.check_entry/check_exit, TradeManager.open_trade/update_trade/close_trade, AdaptiveLearningEngine.pre_trade_analysis, PropFirmTracker.update, PerformanceMetrics.calculate; optionally pushes to LiveDashboardServer.
- **Decisions**: Resets in-memory learning state at the start of each run; pushes live dashboard updates every 5 bars to balance responsiveness with overhead; force-closes any open trade at end of data.

### BacktestDataPreparer.__init__(self, ichi_tenkan_period: int, ichi_kijun_period: int, ichi_senkou_b_period: int, adx_period: int, atr_period: int) -> None
- **Does**: Configures Ichimoku, ADX, and ATR indicator calculators with the specified periods.
- **Why**: Allows parameter optimization to pass custom indicator periods to the backtester without modifying the preparer internals.
- **Relationships**: Called by IchimokuBacktester.__init__; creates IchimokuCalculator, ADXCalculator, ATRCalculator.
- **Decisions**: Uses lazy imports to keep the module importable in test environments that mock indicator computations.

### BacktestDataPreparer.prepare(self, candles_1m: pd.DataFrame) -> Dict[str, pd.DataFrame]
- **Does**: Resamples 1-minute candles to 5M/15M/1H/4H timeframes, computes Ichimoku+ADX+ATR indicators on each, and applies shift(1) to all indicator columns to prevent lookahead bias.
- **Why**: Produces the multi-timeframe indicator dataset the backtester needs while guaranteeing no future data leaks into the simulation.
- **Relationships**: Called by IchimokuBacktester.run; calls _resample, _add_indicators_and_shift.
- **Decisions**: Shifts indicator columns (not raw OHLCV) because the bar's own close is legitimately available at bar close, but indicator values derived from it should only be visible on the next bar.

### BacktestDataPreparer.align_to_5m(self, tf_data: Dict[str, pd.DataFrame]) -> pd.DataFrame
- **Does**: Merges all timeframe DataFrames onto the 5M index via forward-fill, producing a single master DataFrame with prefixed columns.
- **Why**: Creates a flat row-per-bar structure where each 5M bar contains the latest available indicator readings from all timeframes.
- **Relationships**: Called by downstream analysis code; consumes output of prepare().
- **Decisions**: Forward-fills higher-TF data so a 4H bar closing at 08:00 only becomes visible from the next 4H open, which is conservative and correct.

### PropFirmStatus (dataclass)
- **Does**: Snapshots the prop firm challenge state at any point: status, profit percentage, worst daily/total drawdowns, days elapsed, and per-day breakdown.
- **Why**: Provides a structured checkpoint that can be queried at any bar or at backtest completion.
- **Relationships**: Produced by PropFirmTracker.check_pass, consumed by BacktestResult.
- **Decisions**: None.

### PropFirmTracker.__init__(self, profit_target_pct: float, max_daily_dd_pct: float, max_total_dd_pct: float, time_limit_days: int) -> None
- **Does**: Configures The5ers-style challenge constraints: profit target, daily drawdown limit, total drawdown limit, and time limit.
- **Why**: Encapsulates all prop firm rules so the backtester can track challenge pass/fail status alongside the simulation.
- **Relationships**: Called by IchimokuBacktester.__init__ and MonteCarloSimulator.
- **Decisions**: None.

### PropFirmTracker.initialise(self, initial_balance: float, start_date: datetime) -> None
- **Does**: Sets the challenge starting balance and date; must be called before any update() calls.
- **Why**: Separates construction from initialization so the tracker can be configured before the backtest start date is known.
- **Relationships**: Called by IchimokuBacktester.run before the main loop.
- **Decisions**: Raises ValueError on non-positive balance to fail fast.

### PropFirmTracker.update(self, timestamp: datetime, balance: float) -> None
- **Does**: Records the account balance at a given bar timestamp, updating daily and total drawdown tracking and challenge status.
- **Why**: Provides bar-by-bar constraint monitoring so the challenge status reflects the exact moment any limit is breached.
- **Relationships**: Called by IchimokuBacktester.run on every 5M bar.
- **Decisions**: Uses sticky terminal states so once a challenge is passed or failed, the status cannot revert.

### PropFirmTracker.check_pass(self) -> PropFirmStatus
- **Does**: Returns the current challenge status with full statistics and per-day drawdown breakdown.
- **Why**: Provides a final or intermediate checkpoint for the prop firm challenge outcome.
- **Relationships**: Called by IchimokuBacktester.run at backtest completion.
- **Decisions**: None.

### PropFirmTracker.daily_dd_series(self) -> pd.Series
- **Does**: Returns a pandas Series of maximum intraday drawdowns indexed by date, expressed as positive percentages.
- **Why**: Enables visualization and analysis of daily risk exposure over the backtest period.
- **Relationships**: Called by dashboard and reporting code.
- **Decisions**: Computes from recorded intrabar equity values rather than just open/close to capture true intraday extremes.

### PerformanceMetrics.calculate(self, trades: List[dict], equity_curve: pd.Series, initial_balance: float) -> dict
- **Does**: Calculates the full metrics suite (win rate, profit factor, Sharpe, Sortino, Calmar, drawdown, expectancy, streaks, duration) from trades and equity curve.
- **Why**: Provides all standard quant metrics needed for threshold checking, reporting, and strategy comparison.
- **Relationships**: Called by IchimokuBacktester.run, GoNoGoValidator._calculate_oos_metrics.
- **Decisions**: Returns None for profit_factor when gross_loss is zero (infinite) to avoid JSON serialization issues.

### PerformanceMetrics.sharpe_ratio(self, returns: pd.Series, risk_free: float) -> float
- **Does**: Computes the annualized Sharpe ratio from daily returns assuming 252 trading days per year.
- **Why**: Provides the industry-standard risk-adjusted return metric used by the go/no-go threshold checker.
- **Relationships**: Called by calculate.
- **Decisions**: Returns 0.0 rather than infinity when std is zero to avoid downstream arithmetic issues.

### PerformanceMetrics.sortino_ratio(self, returns: pd.Series, risk_free: float) -> float
- **Does**: Computes the annualized Sortino ratio using only downside deviation.
- **Why**: Provides a risk-adjusted return metric that penalizes only harmful volatility, complementing Sharpe.
- **Relationships**: Called by calculate.
- **Decisions**: Returns infinity when there are no losing days, which is mathematically correct.

### PerformanceMetrics.calmar_ratio(self, total_return_pct: float, max_drawdown_pct: float) -> float
- **Does**: Computes the Calmar ratio as total return divided by maximum drawdown.
- **Why**: Measures return efficiency relative to worst-case drawdown, important for prop firm risk assessment.
- **Relationships**: Called by calculate.
- **Decisions**: Returns 0.0 when max_drawdown_pct is zero rather than dividing by zero.

### PerformanceMetrics.max_drawdown(self, equity: pd.Series) -> Tuple[float, Optional[datetime], Optional[datetime]]
- **Does**: Calculates the maximum peak-to-trough drawdown percentage and identifies the peak and trough dates.
- **Why**: Provides the primary risk metric for prop firm compliance and the go/no-go threshold check.
- **Relationships**: Called by calculate.
- **Decisions**: None.

### TradeLogger.__init__(self, db_pool, embedding_engine, dry_run: bool) -> None
- **Does**: Configures the trade logger with an optional database pool and embedding engine, defaulting to dry-run mode when no pool is provided.
- **Why**: Allows the backtester to use identical logging code for both database-backed and test/dry-run scenarios.
- **Relationships**: Called by IchimokuBacktester.__init__; creates EmbeddingEngine if none provided.
- **Decisions**: Auto-detects dry-run mode when db_pool is None to simplify caller code.

### TradeLogger.log_trade(self, trade: dict, context: dict, source: str) -> int
- **Does**: Inserts a single trade and its market context record into PostgreSQL, or returns a synthetic ID in dry-run mode.
- **Why**: Persists trade data with pgvector embeddings for similarity-based learning and post-hoc analysis.
- **Relationships**: Called by log_batch and user code; calls format_for_db, EmbeddingEngine.embed_trade.
- **Decisions**: Commits trade and context in a single transaction to maintain referential integrity.

### TradeLogger.log_batch(self, trades: List[dict], source: str) -> List[int]
- **Does**: Batch inserts multiple trades with their contexts in a single database transaction.
- **Why**: Reduces connection overhead when persisting many trades at once (e.g., after a full backtest run).
- **Relationships**: Called by post-backtest persistence code; calls format_for_db internally.
- **Decisions**: Falls back to per-trade log_trade calls in dry-run mode for simplicity.

### TradeLogger.format_for_db(self, trade: dict, context: dict) -> Tuple[Dict, Dict]
- **Does**: Transforms a raw simulation trade dict and context dict into two dicts whose keys map exactly to the trades and market_context database table columns.
- **Why**: Separates formatting from insertion so trade data can be validated without database access.
- **Relationships**: Called by log_trade and log_batch; calls EmbeddingEngine.embed_trade.
- **Decisions**: Converts integer direction values (1/-1) to string labels (bullish/bearish) for human-readable database records.

### BacktestDashboard.__init__(self, title: str) -> None
- **Does**: Configures the dashboard title for the self-contained HTML output.
- **Why**: Allows customization of the dashboard header for different instruments or strategies.
- **Relationships**: Called by user code.
- **Decisions**: None.

### BacktestDashboard.generate(self, result: BacktestResult, initial_balance: float, learning_phase: str, learning_skipped: int, instrument: str) -> str
- **Does**: Generates a complete self-contained HTML dashboard with embedded base64 matplotlib charts for equity curve, trade distribution, learning phases, daily P&L, win rate heatmap, and prop firm tracking.
- **Why**: Produces a single-file visual report that opens in any browser with no dependencies, suitable for archiving and sharing.
- **Relationships**: Consumes BacktestResult; calls chart_equity_curve, chart_trades_on_price, chart_learning_phases, chart_daily_pnl, chart_win_rate_heatmap, chart_prop_firm_tracking.
- **Decisions**: Embeds all charts as base64 PNGs and all CSS inline to create a zero-dependency single HTML file.

### BacktestDashboard.save_and_open(self, result: BacktestResult, output_dir: str, initial_balance: float, learning_phase: str, learning_skipped: int, instrument: str, auto_open: bool) -> str
- **Does**: Generates the dashboard HTML, saves it to a timestamped file, and optionally opens it in the default browser.
- **Why**: Provides a one-call workflow for generating, persisting, and viewing backtest results.
- **Relationships**: Calls generate; uses webbrowser.open for auto-open.
- **Decisions**: Uses UTC timestamp in filename to ensure uniqueness across runs.

### chart_equity_curve(equity_curve: pd.Series, initial_balance: float, profit_target_pct: float, max_total_dd_pct: float) -> str
- **Does**: Renders the equity curve as a percentage change chart with prop firm target and drawdown limit reference lines.
- **Why**: Provides the primary visual for assessing overall backtest performance relative to challenge constraints.
- **Relationships**: Called by BacktestDashboard.generate.
- **Decisions**: Uses dark theme (#1a1a2e background) consistent with the dashboard's visual identity.

### chart_trades_on_price(trades: List[dict], equity_curve: pd.Series) -> str
- **Does**: Renders trade entry and exit markers on a scatter plot, color-coded by win/loss.
- **Why**: Visualizes where trades occurred and their outcomes to identify patterns in entry/exit timing.
- **Relationships**: Called by BacktestDashboard.generate.
- **Decisions**: None.

### chart_learning_phases(trades: List[dict], total_trades: int) -> str
- **Does**: Renders a horizontal bar showing the mechanical/statistical/similarity learning phase progression with a current-position marker.
- **Why**: Visualizes how far the adaptive learning engine has progressed through its three phases.
- **Relationships**: Called by BacktestDashboard.generate.
- **Decisions**: Uses fixed phase boundaries (0-99, 100-499, 500+) matching AdaptiveLearningEngine thresholds.

### chart_daily_pnl(daily_pnl: pd.Series) -> str
- **Does**: Renders a bar chart of daily returns colored green for positive and red for negative days.
- **Why**: Provides a day-level view of P&L consistency and identifies clusters of winning or losing days.
- **Relationships**: Called by BacktestDashboard.generate.
- **Decisions**: None.

### chart_win_rate_heatmap(trades: List[dict]) -> str
- **Does**: Renders a two-panel chart: win rate by trading session (left) and R-multiple distribution histogram (right).
- **Why**: Reveals which trading sessions perform best and shows the overall R-multiple distribution shape.
- **Relationships**: Called by BacktestDashboard.generate.
- **Decisions**: Names the chart "heatmap" but uses bar+histogram because the data is categorical rather than 2D continuous.

### chart_prop_firm_tracking(equity_curve: pd.Series, initial_balance: float, daily_pnl: pd.Series, max_daily_dd_pct: float, max_total_dd_pct: float, profit_target_pct: float) -> str
- **Does**: Renders three horizontal gauge bars showing profit progress, worst daily drawdown, and worst total drawdown relative to their limits.
- **Why**: Provides at-a-glance prop firm compliance status with color-coded danger levels.
- **Relationships**: Called by BacktestDashboard.generate.
- **Decisions**: Uses traffic-light coloring (green/yellow/red) at 50% and 80% of each limit threshold.

### LiveDashboardServer.__init__(self, port: int, auto_open: bool) -> None
- **Does**: Configures the local HTTP server port and auto-open behavior for real-time backtest monitoring.
- **Why**: Provides a zero-dependency live dashboard that polls for state updates without requiring any frontend framework.
- **Relationships**: Called by user code before IchimokuBacktester.run.
- **Decisions**: Uses stdlib http.server and threading to avoid any external dependency.

### LiveDashboardServer.start(self) -> None
- **Does**: Starts the HTTP server in a daemon background thread and optionally opens the dashboard URL in the default browser.
- **Why**: Runs the server non-blocking so the backtester can proceed with simulation while the dashboard serves requests.
- **Relationships**: Called by user code before backtest run.
- **Decisions**: Tries port+1 automatically if the configured port is already in use.

### LiveDashboardServer.update(self, updates: Dict[str, Any]) -> None
- **Does**: Pushes a state update dict to the thread-safe shared state container for the dashboard to poll.
- **Why**: Provides the mechanism for the backtester to stream metrics to the browser in real time.
- **Relationships**: Called by IchimokuBacktester.run every 5 bars.
- **Decisions**: Uses a simple dict merge under a threading lock rather than a message queue for simplicity.

### LiveDashboardServer.finish(self, final_state: Optional[Dict[str, Any]]) -> None
- **Does**: Marks the backtest as complete by setting done=True and updating final state values.
- **Why**: Signals the browser dashboard to stop polling and display the final results.
- **Relationships**: Called by IchimokuBacktester.run at backtest completion.
- **Decisions**: None.

### LiveDashboardServer.stop(self) -> None
- **Does**: Shuts down the HTTP server gracefully.
- **Why**: Releases the port and background thread resources after the dashboard is no longer needed.
- **Relationships**: Called by user code after backtest completes.
- **Decisions**: None.

### LiveDashboardServer.port (property) -> int
- **Does**: Returns the actual port the server is running on.
- **Why**: Needed when the server auto-incremented the port due to a conflict.
- **Relationships**: Called by user code.
- **Decisions**: None.

### LiveDashboardServer.url (property) -> str
- **Does**: Returns the full localhost URL of the running dashboard.
- **Why**: Convenience accessor for logging and display purposes.
- **Relationships**: Called by user code.
- **Decisions**: None.
