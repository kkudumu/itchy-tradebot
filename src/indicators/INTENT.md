# Indicators — Intent

## Functions

### wilders_smooth(values: np.ndarray, period: int) -> np.ndarray
- **Does**: Applies Wilder's exponential smoothing (RMA) to a 1-D array, seeding with a simple average of the first `period` values.
- **Why**: ADX, ATR, and RSI all require Wilder's specific smoothing recurrence; centralising it avoids duplication across indicator calculators.
- **Relationships**: Called by ADXCalculator.calculate, ATRCalculator.calculate, RSICalculator.calculate.
- **Decisions**: Uses an O(n) loop because Wilder's recurrence has a sequential dependency that cannot be vectorised.

---

### IchimokuCalculator.__init__(tenkan_period: int, kijun_period: int, senkou_b_period: int) -> None
- **Does**: Stores the three configurable Ichimoku period parameters with validation.
- **Why**: Allows non-standard period settings for alternative markets or timeframes while defaulting to the classic 9/26/52 Japanese settings.
- **Relationships**: Called by SignalEngine.__init__, MTFAnalyzer.__init__.
- **Decisions**: None.

### IchimokuCalculator.calculate(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> IchimokuResult
- **Does**: Computes all five Ichimoku components (Tenkan, Kijun, Senkou A, Senkou B, Chikou) using pandas rolling operations.
- **Why**: Provides the core indicator arrays that every downstream signal detection and confluence check depends on.
- **Relationships**: Calls _midpoint; called by MTFAnalyzer.compute_indicators.
- **Decisions**: Senkou spans are shifted forward and Chikou backward in array space to match standard charting displacement conventions.

### IchimokuCalculator.cloud_thickness(senkou_a: np.ndarray, senkou_b: np.ndarray) -> np.ndarray
- **Does**: Returns the absolute difference between Senkou A and Senkou B at each bar.
- **Why**: Cloud thickness measures conviction in the current trend; thin clouds indicate potential reversals.
- **Relationships**: Called by IchimokuSignals.signal_state_at.
- **Decisions**: None.

### IchimokuCalculator.cloud_direction(senkou_a: np.ndarray, senkou_b: np.ndarray) -> np.ndarray
- **Does**: Returns +1 (bullish), -1 (bearish), or 0 (neutral) based on the relative position of Senkou A vs Senkou B.
- **Why**: Cloud polarity is the primary trend direction filter used by the 4H hard filter in the signal engine.
- **Relationships**: Called by IchimokuSignals.signal_state_at, MTFAnalyzer._scalar_cloud_direction.
- **Decisions**: NaN inputs produce 0 (neutral) rather than propagating NaN, so downstream filters treat insufficient data as "no signal".

---

### IchimokuSignals.tk_cross(tenkan: np.ndarray, kijun: np.ndarray) -> np.ndarray
- **Does**: Detects bar-by-bar Tenkan/Kijun crossovers, returning +1 at bullish crosses and -1 at bearish crosses.
- **Why**: TK crosses are the primary entry trigger on the 15M timeframe in the multi-timeframe hierarchy.
- **Relationships**: Called by IchimokuSignals.signal_state_at; feeds into ConfluenceScorer.score.
- **Decisions**: Uses previous-bar vs current-bar diff comparison rather than sign-change detection to handle exact equality correctly.

### IchimokuSignals.cloud_position(close: np.ndarray, senkou_a: np.ndarray, senkou_b: np.ndarray) -> np.ndarray
- **Does**: Classifies each bar's close as above (+1), below (-1), or inside (0) the Ichimoku cloud.
- **Why**: Price position relative to the cloud confirms or denies the directional bias established by cloud direction.
- **Relationships**: Called by IchimokuSignals.signal_state_at; used in SignalEngine._check_15m_signal.
- **Decisions**: Inside-cloud bars return 0 to avoid false signals in congestion zones.

### IchimokuSignals.chikou_confirmation(chikou: np.ndarray, close: np.ndarray) -> np.ndarray
- **Does**: Compares Chikou Span to close price at each bar, returning +1 (bullish) or -1 (bearish) confirmation.
- **Why**: Chikou confirmation is one of the five Ichimoku conditions required for a full-strength signal.
- **Relationships**: Called by IchimokuSignals.signal_state_at; used in SignalEngine._check_15m_signal.
- **Decisions**: Compares chikou[i] vs close[i] element-wise to match standard charting platform behaviour.

### IchimokuSignals.cloud_twist(senkou_a: np.ndarray, senkou_b: np.ndarray) -> np.ndarray
- **Does**: Detects crossovers between Senkou A and Senkou B, signalling future cloud polarity changes.
- **Why**: Cloud twists in the displaced future warn of upcoming trend shifts before they reach current price action.
- **Relationships**: Called by IchimokuSignals.signal_state_at.
- **Decisions**: None.

### IchimokuSignals.cloud_breakout(close: np.ndarray, senkou_a: np.ndarray, senkou_b: np.ndarray) -> np.ndarray
- **Does**: Detects bars where price transitions from inside/below the cloud to above it (bullish) or vice versa (bearish).
- **Why**: Cloud breakouts are high-probability trend initiation events used as additional signal confirmation.
- **Relationships**: Called independently for breakout analysis; not used in signal_state_at.
- **Decisions**: Requires the previous bar to not already be on the breakout side, preventing re-triggering during sustained trends.

### IchimokuSignals.signal_state_at(idx: int, tenkan: np.ndarray, kijun: np.ndarray, close: np.ndarray, senkou_a: np.ndarray, senkou_b: np.ndarray, chikou: np.ndarray) -> IchimokuSignalState
- **Does**: Aggregates all signal dimensions (cloud direction, TK cross, cloud position, Chikou, twist, thickness) into a single snapshot at bar `idx`.
- **Why**: Provides a convenient single-bar state object that the MTFAnalyzer and ConfluenceScorer consume for decision-making.
- **Relationships**: Calls tk_cross, cloud_position, chikou_confirmation, cloud_twist, IchimokuCalculator.cloud_direction, IchimokuCalculator.cloud_thickness; called by MTFAnalyzer.compute_indicators.
- **Decisions**: Instantiates a fresh IchimokuCalculator internally for cloud_direction/thickness rather than requiring one as a parameter.

---

### ADXCalculator.__init__(period: int, threshold: float) -> None
- **Does**: Configures the ADX lookback period and the trending/non-trending threshold.
- **Why**: Gold's stronger trending behaviour justifies a higher default threshold (28) than the generic 25.
- **Relationships**: Called by SignalEngine.__init__.
- **Decisions**: Gold-specific default threshold of 28 instead of the standard 25.

### ADXCalculator.calculate(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> ADXResult
- **Does**: Computes ADX, +DI, and -DI using Wilder's 1978 algorithm with True Range and directional movement.
- **Why**: ADX quantifies trend strength regardless of direction, earning a confluence bonus point when above threshold.
- **Relationships**: Calls wilders_smooth; called by MTFAnalyzer.compute_indicators.
- **Decisions**: None.

---

### ATRCalculator.__init__(period: int) -> None
- **Does**: Stores the ATR smoothing period.
- **Why**: ATR is used throughout the system for volatility-adaptive stop placement, zone proximity, and clustering distance.
- **Relationships**: Called by SignalEngine.__init__.
- **Decisions**: None.

### ATRCalculator.calculate(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray
- **Does**: Computes Average True Range using Wilder's smoothing of the True Range series.
- **Why**: ATR normalises all distance-based calculations (stop-loss sizing, zone proximity, DBSCAN epsilon) to current volatility.
- **Relationships**: Calls wilders_smooth; called by MTFAnalyzer.compute_indicators.
- **Decisions**: None.

---

### RSICalculator.__init__(period: int, overbought: float, oversold: float) -> None
- **Does**: Configures RSI period and overbought/oversold threshold levels.
- **Why**: Allows customisation of RSI sensitivity for different market conditions.
- **Relationships**: Called by divergence analysis consumers.
- **Decisions**: None.

### RSICalculator.calculate(close: np.ndarray) -> RSIResult
- **Does**: Computes RSI using Wilder's smoothing of gains and losses, plus overbought/oversold boolean masks.
- **Why**: RSI feeds into divergence detection and provides momentum context for trade filtering.
- **Relationships**: Calls wilders_smooth; output consumed by DivergenceDetector.detect.
- **Decisions**: Uses Wilder's smoothing (not EMA) to match the original RSI specification.

---

### BollingerBandCalculator.__init__(period: int, std_dev: float, squeeze_lookback: int, squeeze_percentile: float) -> None
- **Does**: Configures Bollinger Band parameters and the volatility squeeze detection window.
- **Why**: Squeeze detection identifies compressed-volatility regimes that often precede directional breakouts in gold.
- **Relationships**: None.
- **Decisions**: Gold-specific default squeeze threshold at the 20th percentile.

### BollingerBandCalculator.calculate(close: np.ndarray) -> BBResult
- **Does**: Computes upper/lower bands, normalised width, rolling percentile rank of width, and a squeeze boolean flag.
- **Why**: Band squeeze detection provides a volatility-regime filter that complements trend-direction signals.
- **Relationships**: Standalone; output consumed by higher-level analysis.
- **Decisions**: Uses population standard deviation (ddof=0) to match Wilder convention; percentile rank uses a rolling-apply window.

---

### SessionIdentifier.identify(timestamps: np.ndarray) -> np.ndarray
- **Does**: Labels each UTC timestamp as "overlap", "london", "new_york", "asian", or "off_hours" based on session boundaries.
- **Why**: Session identification drives the active-session confluence bonus and filters out low-liquidity periods.
- **Relationships**: Calls session_mask; called by MTFAnalyzer.get_current_state.
- **Decisions**: Priority ordering (overlap > london/new_york > asian) ensures the highest-liquidity label wins when sessions overlap.

### SessionIdentifier.is_active_session(timestamps: np.ndarray) -> np.ndarray
- **Does**: Returns a boolean mask indicating whether each timestamp falls within London or New York trading hours.
- **Why**: Active session hours (08:00-21:00 UTC) represent peak gold liquidity when signals are most reliable.
- **Relationships**: Calls session_mask.
- **Decisions**: Extends beyond the common 08:00-17:00 window to capture the full NY afternoon session.

### SessionIdentifier.session_mask(timestamps: np.ndarray, session: str) -> np.ndarray
- **Does**: Returns a boolean mask for timestamps falling within a specific named session's UTC time boundaries.
- **Why**: Provides the low-level building block for both identify() and is_active_session() without duplicating boundary logic.
- **Relationships**: Called by identify, is_active_session.
- **Decisions**: None.

---

### DivergenceDetector.__init__(lookback: int, min_bars_between: int, max_bars_between: int) -> None
- **Does**: Configures the swing-point detection window and the minimum/maximum separation between divergence pivot pairs.
- **Why**: Constraining pivot pair separation prevents noise from very close pivots and irrelevant connections from very distant ones.
- **Relationships**: None.
- **Decisions**: None.

### DivergenceDetector.detect(close: np.ndarray, rsi: np.ndarray, high: np.ndarray, low: np.ndarray) -> DivergenceResult
- **Does**: Detects regular and hidden bullish/bearish divergences between price swing points and RSI swing points.
- **Why**: Divergences signal momentum exhaustion (regular) or trend continuation (hidden), adding a price-action confirmation layer.
- **Relationships**: Calls _find_swing_lows, _find_swing_highs, _compare_lows, _compare_highs; consumes RSICalculator output.
- **Decisions**: Divergence is confirmed at the second (most recent) pivot of each pair, ensuring the signal is only visible after the pattern completes.
