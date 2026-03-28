# Strategy — Intent

## Functions

### MTFAnalyzer.__init__(ichimoku_calc: IchimokuCalculator | None, adx_calc: ADXCalculator | None, atr_calc: ATRCalculator | None) -> None
- **Does**: Stores pre-configured indicator calculators and initialises IchimokuSignals and SessionIdentifier instances.
- **Why**: Accepts injected calculators so that the SignalEngine can configure periods and thresholds in one place and pass them through.
- **Relationships**: Called by SignalEngine.__init__.
- **Decisions**: All calculators default to standard parameters if not provided, enabling standalone use.

### MTFAnalyzer.align_timeframes(data_1m: pd.DataFrame) -> dict[str, pd.DataFrame]
- **Does**: Resamples 1-minute OHLCV data into 5M/15M/1H/4H DataFrames, computes indicators on each, and shifts indicator columns forward by one bar.
- **Why**: The +1 bar shift is the critical lookahead prevention mechanism -- a 4H bar's indicators only become visible after that bar closes.
- **Relationships**: Calls _resample_ohlcv, compute_indicators; called by SignalEngine.scan.
- **Decisions**: Only indicator columns are shifted; raw OHLCV columns remain unshifted because they represent the bar that actually formed.

### MTFAnalyzer.compute_indicators(ohlcv: pd.DataFrame) -> dict
- **Does**: Computes Ichimoku, ADX, and ATR indicators for a single timeframe DataFrame and returns them in a dict.
- **Why**: Encapsulates the full indicator computation for one timeframe so align_timeframes can apply it uniformly across all four timeframes.
- **Relationships**: Calls IchimokuCalculator.calculate, ADXCalculator.calculate, ATRCalculator.calculate, IchimokuSignals.signal_state_at; called by align_timeframes.
- **Decisions**: None.

### MTFAnalyzer.get_current_state(tf_data: dict[str, pd.DataFrame], bar_index: int) -> MTFState
- **Does**: Extracts the aligned Ichimoku signal state, ADX, ATR, Kijun, close, session, and timestamp at a specific bar index across all four timeframes.
- **Why**: Produces the single MTFState snapshot that the SignalEngine and ConfluenceScorer evaluate for trade decisions.
- **Relationships**: Calls _state_from_row, _get_row, SessionIdentifier.identify; called by SignalEngine.scan.
- **Decisions**: ADX and ATR are sourced from the 15M timeframe because that is the signal generation timeframe in the hierarchy.

---

### ConfluenceScorer.__init__(adx_threshold: float, min_score: int, kijun_proximity_atr: float) -> None
- **Does**: Configures the ADX trending threshold, minimum tradeable score, and 5M Kijun proximity tolerance.
- **Why**: These thresholds control signal quality gating -- higher min_score means fewer but higher-quality signals.
- **Relationships**: Called by SignalEngine.__init__.
- **Decisions**: None.

### ConfluenceScorer.score(mtf_state: MTFState, direction: str, zone_confluence: int) -> ConfluenceResult
- **Does**: Scores a signal on a 0-8 scale by summing five Ichimoku alignment points (4H cloud, 1H TK, 15M TK cross, 15M Chikou, 5M Kijun proximity) and three confluence bonuses (ADX trending, active session, zone proximity), then assigns a quality tier.
- **Why**: The scoring system converts multiple binary conditions into a single comparable quality metric that gates trade execution.
- **Relationships**: Reads MTFState fields; called by SignalEngine.scan.
- **Decisions**: Quality tiers are A+ (7-8), B (5-6), C (4), no_trade (<4) -- the asymmetric ranges reflect that full alignment is rare and should be distinguished from partial alignment.

---

### SignalEngine.__init__(config: dict | None, instrument: str) -> None
- **Does**: Merges user config with defaults, instantiates all indicator calculators, MTFAnalyzer, ConfluenceScorer, and ZoneManager.
- **Why**: Single entry point that wires together the entire signal generation pipeline from configuration to execution.
- **Relationships**: Creates MTFAnalyzer, ConfluenceScorer, ZoneManager.
- **Decisions**: Uses a flat config dict with fallback defaults rather than nested config objects for simplicity.

### SignalEngine.scan(data_1m: pd.DataFrame, current_bar: int) -> Signal | None
- **Does**: Runs the full 4H->1H->15M->5M signal hierarchy against 1-minute data, returning a Signal with entry/stop/target levels and reasoning trace if all filters pass, or None otherwise.
- **Why**: This is the top-level API that the backtester and live runner call to get trade decisions -- it encapsulates the entire multi-timeframe Ichimoku strategy.
- **Relationships**: Calls MTFAnalyzer.align_timeframes, MTFAnalyzer.get_current_state, _check_4h_filter, _check_1h_confirmation, _check_15m_signal, _check_5m_entry, ConfluenceScorer.score, ZoneManager.get_nearby_zones, _calculate_levels.
- **Decisions**: Filters are applied in hierarchical order (4H first) with early return on failure, so expensive lower-timeframe checks are skipped when higher-timeframe direction is unclear.
