"""Pydantic v2 models for all configuration sections."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Edge toggle models
# ---------------------------------------------------------------------------


class TimeOfDayEdge(BaseModel):
    """Filter entries outside the configured UTC trading window."""

    enabled: bool = True
    params: dict[str, Any] = Field(
        default_factory=lambda: {
            "start_utc": "08:00",
            "end_utc": "17:00",
        }
    )


class DayOfWeekEdge(BaseModel):
    """Restrict trading to specific weekdays."""

    enabled: bool = True
    params: dict[str, Any] = Field(
        default_factory=lambda: {
            # 0=Mon … 6=Sun
            "allowed_days": [1, 2, 3],  # Tue, Wed, Thu
        }
    )


class LondonOpenDelayEdge(BaseModel):
    """Suppress entries for N minutes after the London session opens."""

    enabled: bool = True
    params: dict[str, Any] = Field(
        default_factory=lambda: {
            "london_open_utc": "08:00",
            "delay_minutes": 30,
        }
    )


class CandleCloseConfirmationEdge(BaseModel):
    """Require the H1 candle to close beyond the cloud before entry."""

    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


class SpreadFilterEdge(BaseModel):
    """Skip signals when the current spread exceeds the threshold."""

    enabled: bool = True
    params: dict[str, Any] = Field(
        default_factory=lambda: {
            # Points — 30 pts ≈ 0.30 USD/oz for gold
            "max_spread_points": 30,
        }
    )


class NewsFilterEdge(BaseModel):
    """Block entries around high-impact news events."""

    enabled: bool = True
    params: dict[str, Any] = Field(
        default_factory=lambda: {
            "block_minutes_before": 30,
            "block_minutes_after": 30,
            "impact_levels": ["red"],
        }
    )


class FridayCloseEdge(BaseModel):
    """Force-close all positions before the weekend."""

    enabled: bool = True
    params: dict[str, Any] = Field(
        default_factory=lambda: {
            "close_time_utc": "20:00",
            # Day index 4 = Friday
            "day": 4,
        }
    )


class RegimeFilterEdge(BaseModel):
    """Only trade when ADX is trending and the cloud has sufficient thickness."""

    enabled: bool = True
    params: dict[str, Any] = Field(
        default_factory=lambda: {
            # ADX must exceed this value
            "adx_min": 28,
            # Cloud thickness must be above the rolling median
            "cloud_thickness_percentile": 50,
        }
    )


class TimeStopEdge(BaseModel):
    """Exit at breakeven if the trade hasn't reached 0.5 R within N candles."""

    enabled: bool = True
    params: dict[str, Any] = Field(
        default_factory=lambda: {
            "candle_limit": 12,
            "breakeven_r_threshold": 0.5,
        }
    )


class BBSqueezeEdge(BaseModel):
    """Boost signal confidence when Bollinger Bands expand out of a squeeze."""

    enabled: bool = True
    params: dict[str, Any] = Field(
        default_factory=lambda: {
            "bb_period": 20,
            "bb_std": 2.0,
            # Bandwidth below this triggers squeeze state
            "squeeze_threshold": 0.05,
            "confidence_boost": 1,
        }
    )


class ConfluenceScoringEdge(BaseModel):
    """Enforce a minimum confluence score and enable tiered position sizing."""

    enabled: bool = True
    params: dict[str, Any] = Field(
        default_factory=lambda: {
            "min_score": 4,
            # Tier A+: full size; Tier B: 75%; Tier C: 50%
            "tier_a_plus_threshold": 7,
            "tier_b_threshold": 5,
            "tier_c_threshold": 4,
            "tier_b_size_pct": 0.75,
            "tier_c_size_pct": 0.50,
        }
    )


class EquityCurveEdge(BaseModel):
    """Reduce position size when the equity curve is below its moving average."""

    enabled: bool = True
    params: dict[str, Any] = Field(
        default_factory=lambda: {
            # Number of closed trades used for the equity MA
            "lookback_trades": 20,
            # Multiplier applied to base risk when below the MA
            "reduced_size_multiplier": 0.5,
        }
    )


class EdgeConfig(BaseModel):
    """Container for all 12 edge toggles."""

    time_of_day: TimeOfDayEdge = Field(default_factory=TimeOfDayEdge)
    day_of_week: DayOfWeekEdge = Field(default_factory=DayOfWeekEdge)
    london_open_delay: LondonOpenDelayEdge = Field(default_factory=LondonOpenDelayEdge)
    candle_close_confirmation: CandleCloseConfirmationEdge = Field(
        default_factory=CandleCloseConfirmationEdge
    )
    spread_filter: SpreadFilterEdge = Field(default_factory=SpreadFilterEdge)
    news_filter: NewsFilterEdge = Field(default_factory=NewsFilterEdge)
    friday_close: FridayCloseEdge = Field(default_factory=FridayCloseEdge)
    regime_filter: RegimeFilterEdge = Field(default_factory=RegimeFilterEdge)
    time_stop: TimeStopEdge = Field(default_factory=TimeStopEdge)
    bb_squeeze: BBSqueezeEdge = Field(default_factory=BBSqueezeEdge)
    confluence_scoring: ConfluenceScoringEdge = Field(default_factory=ConfluenceScoringEdge)
    equity_curve: EquityCurveEdge = Field(default_factory=EquityCurveEdge)


# ---------------------------------------------------------------------------
# Strategy config models
# ---------------------------------------------------------------------------


class IchimokuConfig(BaseModel):
    tenkan_period: int = 9
    kijun_period: int = 26
    senkou_b_period: int = 52


class ADXConfig(BaseModel):
    period: int = 14
    threshold: int = 28  # Higher threshold tuned for gold volatility


class ATRConfig(BaseModel):
    period: int = 14
    stop_multiplier: float = 1.5


class RiskConfig(BaseModel):
    initial_risk_pct: float = 1.5   # Risk per trade until +4% account growth
    reduced_risk_pct: float = 0.75  # Risk per trade after reaching +4%
    phase_threshold_pct: float = 4.0
    daily_circuit_breaker_pct: float = 2.0  # Max daily drawdown before halt
    max_concurrent_positions: int = 1


class ExitConfig(BaseModel):
    strategy: str = "hybrid_50_50"   # Close 50% at TP, trail rest on Kijun
    tp_r_multiple: float = 2.0
    trail_type: str = "kijun"
    breakeven_threshold_r: float = 1.0
    kijun_trail_start_r: float = 1.5
    higher_tf_kijun_start_r: float = 3.0


class SignalConfig(BaseModel):
    min_confluence_score: int = 4
    tier_a_plus: int = 7
    tier_b: int = 5
    tier_c: int = 4
    timeframes: list[str] = Field(default_factory=lambda: ["4H", "1H", "15M", "5M"])


class StrategyConfig(BaseModel):
    active_strategy: str = "ichimoku"
    strategies: dict[str, dict] = Field(default_factory=lambda: {
        "ichimoku": {
            "ichimoku": {"tenkan_period": 9, "kijun_period": 26, "senkou_b_period": 52},
            "adx": {"period": 14, "threshold": 28},
            "atr": {"period": 14, "stop_multiplier": 1.5},
            "signal": {
                "min_confluence_score": 4,
                "tier_a_plus": 7,
                "tier_b": 5,
                "tier_c": 4,
                "timeframes": ["4H", "1H", "15M", "5M"],
            },
        }
    })
    risk: RiskConfig = Field(default_factory=RiskConfig)
    exit: ExitConfig = Field(default_factory=ExitConfig)

    @model_validator(mode="before")
    @classmethod
    def _migrate_flat_format(cls, values: Any) -> Any:
        """Accept old flat format (ichimoku/adx/atr/signal at top level) for
        backward compatibility with tests and legacy YAML files.

        If the incoming data does not have an ``active_strategy`` or
        ``strategies`` key but does have flat strategy keys, migrate them into
        the nested structure automatically.
        """
        if not isinstance(values, dict):
            return values

        # Already new-format — nothing to do
        if "active_strategy" in values or "strategies" in values:
            return values

        flat_keys = {"ichimoku", "adx", "atr", "signal"}
        found_flat = {k: values.pop(k) for k in flat_keys if k in values}
        if found_flat:
            values.setdefault("active_strategy", "ichimoku")
            values.setdefault("strategies", {})
            values["strategies"].setdefault("ichimoku", {})
            values["strategies"]["ichimoku"].update(found_flat)

        return values

    # ------------------------------------------------------------------
    # Backward-compatible property accessors
    # ------------------------------------------------------------------

    @property
    def ichimoku(self) -> "IchimokuConfig":
        """Backward-compatible access to Ichimoku config."""
        data = self.strategies.get("ichimoku", {}).get("ichimoku", {})
        return IchimokuConfig(**data) if data else IchimokuConfig()

    @property
    def adx(self) -> "ADXConfig":
        data = self.strategies.get("ichimoku", {}).get("adx", {})
        return ADXConfig(**data) if data else ADXConfig()

    @property
    def atr(self) -> "ATRConfig":
        data = self.strategies.get("ichimoku", {}).get("atr", {})
        return ATRConfig(**data) if data else ATRConfig()

    @property
    def signal(self) -> "SignalConfig":
        data = self.strategies.get("ichimoku", {}).get("signal", {})
        return SignalConfig(**data) if data else SignalConfig()


# ---------------------------------------------------------------------------
# Database config models
# ---------------------------------------------------------------------------


class DatabaseConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    name: str = "trading"
    user: str = "trader"
    password: str = ""
    pool_min: int = 1
    pool_max: int = 5
    # pgvector dimension for embedding-based pattern storage
    vector_dimensions: int = 128

    @model_validator(mode="after")
    def password_not_in_connection_string(self) -> "DatabaseConfig":
        # Validation placeholder — real secrets come from environment variables
        return self


# ---------------------------------------------------------------------------
# Instrument config models
# ---------------------------------------------------------------------------


class InstrumentOverride(BaseModel):
    """Per-instrument parameter overrides applied on top of base strategy config."""

    symbol: str
    provider: str | None = None
    contract_id: str | None = None
    symbol_id: str | None = None
    tick_size: float | None = None
    tick_value: float | None = None
    price_scale: float | None = None
    commission_per_contract: float | None = None
    default_quantity: int | None = None
    # Optional field-level overrides; None means "use strategy default"
    adx_threshold: int | None = None
    spread_max_points: int | None = None
    atr_stop_multiplier: float | None = None
    pip_value_usd: float | None = None   # USD value of 1 pip/point per lot


class InstrumentsConfig(BaseModel):
    instruments: list[InstrumentOverride] = Field(default_factory=list)

    def get(self, symbol: str) -> InstrumentOverride | None:
        """Return the override block for *symbol*, or None if not configured."""
        for inst in self.instruments:
            if inst.symbol == symbol:
                return inst
        return None


# ---------------------------------------------------------------------------
# Provider config models
# ---------------------------------------------------------------------------


class ProjectXConfig(BaseModel):
    api_base_url: str = "https://api.topstepx.com"
    user_hub_url: str = "https://rtc.thefuturesdesk.projectx.com/hubs/user"
    market_hub_url: str = "https://rtc.thefuturesdesk.projectx.com/hubs/market"
    username_env: str = "PROJECTX_USERNAME"
    api_key_env: str = "PROJECTX_API_KEY"
    account_id: int | None = None
    live: bool = False
    request_timeout_seconds: float = 30.0
    default_contract_id: str | None = None
    default_symbol_id: str | None = None
    bar_limit: int = 500
    token_refresh_buffer_seconds: int = 300

    @model_validator(mode="after")
    def validate_limits(self) -> "ProjectXConfig":
        self.bar_limit = max(1, min(self.bar_limit, 20_000))
        self.token_refresh_buffer_seconds = max(0, self.token_refresh_buffer_seconds)
        return self


class ProviderConfig(BaseModel):
    provider: str = "projectx"
    projectx: ProjectXConfig = Field(default_factory=ProjectXConfig)


# ---------------------------------------------------------------------------
# Root application config
# ---------------------------------------------------------------------------


class AppConfig(BaseModel):
    """Unified configuration object assembled from all YAML sources."""

    edges: EdgeConfig = Field(default_factory=EdgeConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    instruments: InstrumentsConfig = Field(default_factory=InstrumentsConfig)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
