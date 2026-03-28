# Risk — Intent

## Functions

### PositionSize (dataclass)
- **Does**: Holds the result of a position size calculation including lot_size, risk_pct, risk_amount, stop_distance, and phase.
- **Why**: Bundles all sizing outputs into a single return value so callers get the full context of the sizing decision.
- **Relationships**: Returned by AdaptivePositionSizer.calculate_position_size(); consumed by TradeManager.open_trade().
- **Decisions**: None.

### AdaptivePositionSizer.__init__(self, initial_balance: float, initial_risk_pct: float, reduced_risk_pct: float, phase_threshold_pct: float, min_lot: float, max_lot: float) -> None
- **Does**: Initialises the two-phase position sizer with hard-coded safety rails clamping risk to [0.25%, 2.0%].
- **Why**: Implements the prop firm challenge strategy: aggressive 1.5% risk until +4% profit, then protective 0.75% risk.
- **Relationships**: Composed into TradeManager; safety rails are class constants that cannot be overridden.
- **Decisions**: Hard-coded _MAX_RISK_PCT=2.0 and _MIN_RISK_PCT=0.25 prevent the learning loop from configuring dangerous risk levels.

### AdaptivePositionSizer.get_risk_pct(self) -> float
- **Does**: Returns the current risk percentage based on whether the account has crossed the phase threshold.
- **Why**: Automates the phase switch so the trading engine does not need to track profit milestones manually.
- **Relationships**: Called by calculate_position_size(); depends on current_balance vs initial_balance.
- **Decisions**: None.

### AdaptivePositionSizer.get_phase(self) -> str
- **Does**: Returns the current phase label: "aggressive" or "protective".
- **Why**: Provides a human-readable phase indicator for logging, dashboards, and PositionSize output.
- **Relationships**: Called by calculate_position_size() to populate PositionSize.phase.
- **Decisions**: None.

### AdaptivePositionSizer.calculate_position_size(self, account_equity: float, atr: float, atr_multiplier: float, point_value: float, instrument: str) -> PositionSize
- **Does**: Calculates lot size from ATR-based stop distance and current risk percentage, clamped to broker lot limits.
- **Why**: Converts the abstract risk percentage into a concrete lot size that respects volatility and broker constraints.
- **Relationships**: Calls get_risk_pct() and get_phase(); called by TradeManager.open_trade().
- **Decisions**: Uses floor-clamping to [min_lot, max_lot] to avoid exceeding the risk budget.

### AdaptivePositionSizer.update_balance(self, new_balance: float) -> None
- **Does**: Updates the tracked current balance after a trade closes.
- **Why**: Keeps the phase calculation accurate as equity changes throughout the challenge.
- **Relationships**: Called by TradeManager or the trading engine after each trade closure.
- **Decisions**: None.

### AdaptivePositionSizer.initial_balance (property) -> float
- **Does**: Returns the starting balance recorded at construction time.
- **Why**: Read-only exposure of internal state for reporting and phase calculations.
- **Relationships**: Used by TradeManager for total drawdown checks.
- **Decisions**: None.

### AdaptivePositionSizer.current_balance (property) -> float
- **Does**: Returns the current tracked balance.
- **Why**: Read-only exposure for external monitoring and dashboards.
- **Relationships**: Used by TradeManager.get_equity_summary().
- **Decisions**: None.

### AdaptivePositionSizer.profit_pct (property) -> float
- **Does**: Returns the current profit as a percentage of the initial balance.
- **Why**: Convenience accessor for dashboards and equity summary reporting.
- **Relationships**: Used by TradeManager.get_equity_summary().
- **Decisions**: None.

### ActiveTrade (dataclass)
- **Does**: Tracks all state for an open trade including entry, stops, partial exits, and current R-multiple.
- **Why**: Centralises mutable trade state so the exit manager can evaluate and update it consistently.
- **Relationships**: Created by TradeManager.open_trade(); mutated by HybridExitManager and TradeManager.update_trade().
- **Decisions**: Preserves original_stop_loss separately from stop_loss to support breakeven-trap prevention checks.

### ActiveTrade.initial_risk (property) -> float
- **Does**: Returns the initial risk in price units (distance from entry to original stop).
- **Why**: Needed for R-multiple calculations throughout the trade lifecycle.
- **Relationships**: Used by HybridExitManager.calculate_r_multiple() indirectly.
- **Decisions**: None.

### ActiveTrade.is_partial (property) -> bool
- **Does**: Returns True if the first partial exit has been executed (remaining_pct < 1.0).
- **Why**: Controls whether trailing-stop logic activates, since trailing only begins after the first partial exit.
- **Relationships**: Checked by HybridExitManager.check_exit() to gate trail updates.
- **Decisions**: None.

### ExitDecision (dataclass)
- **Does**: Represents the exit manager's decision: action type, close percentage, new stop, reason, and R-multiple.
- **Why**: Provides a structured decision that TradeManager can act on without interpreting raw signals.
- **Relationships**: Returned by HybridExitManager.check_exit(); consumed by TradeManager.update_trade().
- **Decisions**: None.

### HybridExitManager.__init__(self, tp_r_multiple: float, breakeven_threshold_r: float, kijun_trail_start_r: float, higher_tf_kijun_start_r: float) -> None
- **Does**: Configures the 50/50 exit strategy with Kijun-Sen trailing and hard-coded R-multiple thresholds.
- **Why**: Implements gold-specific exit logic: close 50% at 2R, trail remainder via Kijun, never move to breakeven before 1R.
- **Relationships**: Composed into TradeManager; threshold constants are immutable.
- **Decisions**: No-breakeven-before-1R rule is critical for gold because gold retraces 0.3-0.7R within valid trends and premature BE gets stop-hunted.

### HybridExitManager.check_exit(self, trade: ActiveTrade, current_price: float, kijun_value: float, higher_tf_kijun: float | None) -> ExitDecision
- **Does**: Evaluates the current price against the trade and returns the highest-priority exit action.
- **Why**: Single entry point for all exit logic, evaluated in priority order: stop hit, partial exit, trail update, no action.
- **Relationships**: Calls calculate_r_multiple(), _stop_hit(), get_trailing_stop(); called by TradeManager.update_trade().
- **Decisions**: Priority ordering ensures stops are checked before partial exits, preventing missed exits.

### HybridExitManager.calculate_r_multiple(self, entry_price: float, current_price: float, stop_loss: float, direction: str) -> float
- **Does**: Calculates the current R-multiple relative to the original risk distance.
- **Why**: R-multiples normalise profit/loss across different volatility levels, enabling consistent exit thresholds.
- **Relationships**: Called by check_exit(); also called by TradeManager.close_trade() for final R calculation.
- **Decisions**: Returns 0.0 for degenerate cases where risk distance is zero or negative.

### HybridExitManager.get_trailing_stop(self, trade: ActiveTrade, kijun_value: float, higher_tf_kijun: float | None) -> float | None
- **Does**: Returns the appropriate trailing stop level based on the current R-multiple and Kijun values.
- **Why**: Implements the tiered trailing schedule: no trail below 1.5R, signal-TF Kijun at 1.5-3R, higher-TF Kijun at 3R+.
- **Relationships**: Called by check_exit() and TradeManager.update_trade().
- **Decisions**: Higher-TF Kijun at 3R+ gives room on extended moves by using a slower-reacting support level.

### DailyCircuitBreaker.__init__(self, max_daily_loss_pct: float) -> None
- **Does**: Initialises the daily loss limit enforcer, clamping the configured limit to the absolute ceiling of 5%.
- **Why**: Prevents catastrophic daily drawdowns that would breach prop firm rules.
- **Relationships**: Composed into TradeManager; checked before every trade.
- **Decisions**: Hard-coded _ABSOLUTE_MAX_DAILY_LOSS_PCT=5.0 cannot be overridden, even by misconfiguration.

### DailyCircuitBreaker.start_day(self, balance: float, date: date | None) -> None
- **Does**: Records the start-of-day balance and resets the triggered flag for a new session.
- **Why**: The circuit breaker operates on a daily cycle; each morning the slate is wiped clean.
- **Relationships**: Called by the trading engine at session start; must precede can_trade() calls.
- **Decisions**: None.

### DailyCircuitBreaker.can_trade(self, current_balance: float) -> bool
- **Does**: Returns True if daily loss is within the limit, False if the circuit has tripped.
- **Why**: Single gate function that the trading engine checks before every potential trade.
- **Relationships**: Called by TradeManager.can_open_trade(); calls daily_loss_pct() internally.
- **Decisions**: Once triggered, stays triggered for the rest of the day; only start_day() resets it.

### DailyCircuitBreaker.is_triggered(self) -> bool
- **Does**: Returns whether the circuit breaker has been tripped today.
- **Why**: Allows external systems to check the breaker state without passing a balance.
- **Relationships**: Called by TradeManager.get_equity_summary().
- **Decisions**: None.

### DailyCircuitBreaker.daily_loss_pct(self, current_balance: float) -> float
- **Does**: Calculates the current daily loss as a positive percentage, returning 0.0 for profitable days.
- **Why**: Provides the metric that can_trade() evaluates against the threshold.
- **Relationships**: Called by can_trade() and TradeManager.get_equity_summary().
- **Decisions**: Clamps negative values (profit) to 0.0 so callers always get an unsigned loss figure.

### DailyCircuitBreaker.remaining_risk_budget(self, current_balance: float) -> float
- **Does**: Returns the monetary amount that can still be lost today before the circuit trips.
- **Why**: Informs the position sizer how much risk budget remains for the day.
- **Relationships**: Can be used by the trading engine to cap individual trade risk.
- **Decisions**: Returns 0.0 if already triggered or at the limit.

### DailyCircuitBreaker.max_daily_loss_pct (property) -> float
- **Does**: Returns the configured (and clamped) maximum daily loss percentage.
- **Why**: Read-only access for logging and diagnostics.
- **Relationships**: Used by TradeManager.can_open_trade() for error messages.
- **Decisions**: None.

### DailyCircuitBreaker.daily_start_balance (property) -> float
- **Does**: Returns the balance recorded at start of the current trading day.
- **Why**: Read-only access for external reporting.
- **Relationships**: None.
- **Decisions**: None.

### DailyCircuitBreaker.current_date (property) -> date | None
- **Does**: Returns the calendar date of the current session, or None if start_day() has not been called.
- **Why**: Allows external systems to verify which day the breaker is tracking.
- **Relationships**: None.
- **Decisions**: None.

### TradeManager.__init__(self, position_sizer: AdaptivePositionSizer, circuit_breaker: DailyCircuitBreaker, exit_manager: HybridExitManager, max_concurrent: int) -> None
- **Does**: Orchestrates position sizing, circuit breaking, and exit management into a unified trade lifecycle manager.
- **Why**: Provides a single coordination point so the trading engine does not need to call three subsystems directly.
- **Relationships**: Composes AdaptivePositionSizer, DailyCircuitBreaker, and HybridExitManager.
- **Decisions**: Hard-coded absolute limits (_ABSOLUTE_MAX_RISK=2%, _ABSOLUTE_MAX_DAILY_LOSS=5%, _ABSOLUTE_MAX_TOTAL_DD=10%) are immutable safety rails matching prop firm rules.

### TradeManager.can_open_trade(self, current_balance: float, instrument: str) -> tuple[bool, str]
- **Does**: Checks circuit breaker, concurrent position limit, correlation, and total drawdown before allowing a new trade.
- **Why**: Centralises all pre-trade validation so no safety check can be accidentally skipped.
- **Relationships**: Calls DailyCircuitBreaker.can_trade(), check_correlation(); called by open_trade() and the trading engine.
- **Decisions**: Total drawdown guard at 10% matches The5ers prop firm maximum drawdown limit.

### TradeManager.open_trade(self, entry_price: float, stop_loss: float, take_profit: float, direction: str, atr: float, point_value: float, account_equity: float, atr_multiplier: float, instrument: str, entry_time: datetime | None) -> tuple[int, ActiveTrade, PositionSize]
- **Does**: Opens a new trade with adaptive position sizing after passing all pre-trade checks.
- **Why**: Combines validation, sizing, and trade creation into a single atomic operation.
- **Relationships**: Calls can_open_trade() and AdaptivePositionSizer.calculate_position_size(); returns trade_id for subsequent update_trade() calls.
- **Decisions**: Raises RuntimeError if pre-trade checks fail, forcing the caller to handle rejection explicitly.

### TradeManager.update_trade(self, trade_id: int, current_price: float, kijun_value: float, higher_tf_kijun: float | None) -> ExitDecision
- **Does**: Evaluates exit conditions and applies stop updates or archives the trade on full exit.
- **Why**: Drives the trade lifecycle forward on every bar, applying the hybrid exit strategy.
- **Relationships**: Calls HybridExitManager.check_exit() and get_trailing_stop(); mutates ActiveTrade in place.
- **Decisions**: On partial exit, immediately attempts to begin trailing the remaining position.

### TradeManager.close_trade(self, trade_id: int, exit_price: float, reason: str) -> dict
- **Does**: Force-closes a trade completely and returns a summary dict with P&L and R-multiple.
- **Why**: Supports non-standard exits like friday_close or news_event that bypass normal exit logic.
- **Relationships**: Calls HybridExitManager.calculate_r_multiple() and _archive_trade().
- **Decisions**: None.

### TradeManager.get_equity_summary(self, current_prices: dict[str, float] | None) -> dict
- **Does**: Returns a snapshot of current equity state including balance, open trades, P&L, phase, and circuit breaker status.
- **Why**: Provides a single structured view for dashboards and the learning loop.
- **Relationships**: Reads from AdaptivePositionSizer and DailyCircuitBreaker.
- **Decisions**: Unrealised P&L is a placeholder (0.0) pending full broker equity feed integration.

### TradeManager.active_trade_ids (property) -> list[int]
- **Does**: Returns IDs of all currently open trades.
- **Why**: Allows the trading engine to iterate over active trades for update_trade() calls.
- **Relationships**: Read by the trading engine's main loop.
- **Decisions**: None.

### TradeManager.closed_trades (property) -> list[dict]
- **Does**: Returns a read-only copy of the closed trade log.
- **Why**: Provides trade history for the learning loop, equity curve filter, and reporting.
- **Relationships**: Populated by _archive_trade(); consumed by external analysis.
- **Decisions**: None.

### TradeManager.check_correlation(self, instrument: str, direction: str) -> bool
- **Does**: Placeholder that always returns True for single-instrument operation.
- **Why**: Reserves the integration point for multi-instrument correlation limits when expanding beyond XAUUSD.
- **Relationships**: Called by can_open_trade().
- **Decisions**: Intentionally a no-op for the current single-instrument scope.
