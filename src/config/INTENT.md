# Config — Intent

## Functions

### load_config(config_dir: str | Path | None) -> AppConfig
- **Does**: Convenience function that creates a ConfigLoader and returns a validated AppConfig in one call.
- **Why**: Provides a simple module-level entry point so callers do not need to instantiate ConfigLoader directly.
- **Relationships**: Creates ConfigLoader, calls ConfigLoader.load(); called by application bootstrap code.
- **Decisions**: Delegates entirely to ConfigLoader to keep the one-liner stateless.

### ConfigLoader.__init__(config_dir: str | Path | None) -> None
- **Does**: Resolves the configuration directory from an explicit argument, the CONFIG_DIR environment variable, or the project default.
- **Why**: Centralises directory resolution so all YAML loading uses a single, predictable source path.
- **Relationships**: Called by load_config(); stores path used by ConfigLoader.load().
- **Decisions**: Three-tier fallback (explicit arg > env var > default) lets tests and CI override without code changes.

### ConfigLoader.load() -> AppConfig
- **Does**: Reads all YAML config files from disk and returns a fully validated AppConfig assembled from their contents.
- **Why**: Single method that merges the four config domains (edges, strategy, database, instruments) into one typed object.
- **Relationships**: Calls _load_yaml for each file; constructs EdgeConfig, StrategyConfig, DatabaseConfig, InstrumentsConfig via Pydantic model_validate; called by load_config() and ConfigLoader.reload().
- **Decisions**: Each YAML file maps to exactly one Pydantic model to keep validation boundaries clear.

### ConfigLoader.reload() -> AppConfig
- **Does**: Re-reads all YAML files from disk and returns a fresh AppConfig.
- **Why**: Supports hot-reloading during live development without restarting the application.
- **Relationships**: Delegates to ConfigLoader.load().
- **Decisions**: None.

### ConfigLoader.config_dir -> Path
- **Does**: Returns the resolved configuration directory path.
- **Why**: Exposes the resolved path for testing and diagnostic tooling.
- **Relationships**: Set during __init__; read by load().
- **Decisions**: Exposed as a read-only property to prevent mutation after construction.

### AppConfig (class)
- **Does**: Root Pydantic model that holds all four configuration sections (edges, strategy, database, instruments).
- **Why**: Provides a single typed entry point for the entire configuration surface so consumers access one object.
- **Relationships**: Composed of EdgeConfig, StrategyConfig, DatabaseConfig, InstrumentsConfig; returned by ConfigLoader.load().
- **Decisions**: All fields default to their sub-model defaults, making a zero-config startup possible.

### EdgeConfig (class)
- **Does**: Container for all 12 edge toggle models, each controlling a specific trading filter or enhancement.
- **Why**: Groups edge toggles so they can be iterated, serialised, and validated as a unit.
- **Relationships**: Contains TimeOfDayEdge, DayOfWeekEdge, LondonOpenDelayEdge, CandleCloseConfirmationEdge, SpreadFilterEdge, NewsFilterEdge, FridayCloseEdge, RegimeFilterEdge, TimeStopEdge, BBSqueezeEdge, ConfluenceScoringEdge, EquityCurveEdge; owned by AppConfig.
- **Decisions**: Each edge is a separate model with enabled flag and params dict to allow runtime toggling without schema changes.

### StrategyConfig (class)
- **Does**: Groups all strategy-related sub-configs (Ichimoku, ADX, ATR, risk, exit, signal).
- **Why**: Keeps strategy parameters separate from edge toggles and infrastructure config.
- **Relationships**: Contains IchimokuConfig, ADXConfig, ATRConfig, RiskConfig, ExitConfig, SignalConfig; owned by AppConfig.
- **Decisions**: Defaults are tuned for XAU/USD on The5ers prop firm rules.

### DatabaseConfig (class)
- **Does**: Holds PostgreSQL connection parameters and pool sizing.
- **Why**: Centralises database config so connection.py and loader.py read from the same source.
- **Relationships**: Owned by AppConfig; consumed by DatabasePool and TimescaleLoader.
- **Decisions**: Password field defaults to empty string; real secrets come from environment variables enforced by a model validator.

### InstrumentsConfig (class)
- **Does**: Holds a list of per-instrument parameter overrides.
- **Why**: Allows instrument-specific tuning (e.g., different spread thresholds for gold vs forex) on top of the base strategy config.
- **Relationships**: Contains InstrumentOverride instances; owned by AppConfig.
- **Decisions**: None.

### InstrumentsConfig.get(symbol: str) -> InstrumentOverride | None
- **Does**: Returns the override block for a given symbol, or None if not configured.
- **Why**: Provides O(n) lookup by symbol without requiring callers to iterate the list manually.
- **Relationships**: Called by strategy engine when resolving instrument-specific parameters.
- **Decisions**: Linear scan is acceptable because the instrument list is small (typically 1-3 entries).

### InstrumentOverride (class)
- **Does**: Pydantic model holding optional per-instrument overrides for ADX threshold, spread max, ATR multiplier, and pip value.
- **Why**: Lets the system apply instrument-specific tuning without duplicating the entire strategy config.
- **Relationships**: Owned by InstrumentsConfig; consumed by strategy engine.
- **Decisions**: Override fields are Optional; None means "use the base strategy default".

### RiskConfig (class)
- **Does**: Defines risk management parameters including per-trade risk percentages, phase threshold, and circuit breaker.
- **Why**: Encodes The5ers prop firm risk rules (two-phase risk reduction, daily drawdown halt) as typed config.
- **Relationships**: Owned by StrategyConfig; consumed by position sizing and risk management modules.
- **Decisions**: Defaults (1.5% initial, 0.75% reduced, 2% daily circuit breaker) match the prop firm pass strategy.

### ExitConfig (class)
- **Does**: Defines trade exit strategy parameters including partial close, trailing stop type, and breakeven threshold.
- **Why**: Makes the hybrid 50/50 exit strategy (close half at TP, trail rest on Kijun) fully configurable.
- **Relationships**: Owned by StrategyConfig; consumed by exit management module.
- **Decisions**: Default is hybrid_50_50 with Kijun trailing, reflecting the Ichimoku-based exit approach.

### SignalConfig (class)
- **Does**: Defines signal scoring thresholds and the list of analysis timeframes.
- **Why**: Controls the minimum confluence score for trade entry and the tiered sizing system.
- **Relationships**: Owned by StrategyConfig; consumed by signal generation and confluence scoring.
- **Decisions**: Default timeframes [4H, 1H, 15M, 5M] follow the multi-timeframe Ichimoku analysis hierarchy.
