"""ContextBuilder — assembles the input bundle the agent sees at each decision.

Every time the native blender produces candidate signals, the engine
calls ``ContextBuilder.build()`` to assemble:

  * the last 60 bars of OHLCV
  * current indicators (ATR, ADX, EMAs, Ichimoku cloud values, regime tag)
  * a rolling summary of recent telemetry events per strategy
  * per-strategy performance buckets filtered to the current session/
    pattern/regime
  * the last 10 closed trades from trade_memory
  * a freshly-rendered screenshot path (or None when capture is disabled)
  * current prop firm tracker state
  * current risk state (position, equity, daily P&L)

Returns a ``ContextBundle`` dataclass that the agent's tools can
reach into. The builder itself doesn't call the agent — that's the
``MegaStrategyAgent``'s job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ContextBundle:
    """Input bundle for one agent decision."""

    timestamp_utc: datetime
    candidate_signals: List[Any] = field(default_factory=list)
    recent_bars: Any = None  # pd.DataFrame or None
    current_indicators: Dict[str, Any] = field(default_factory=dict)
    recent_telemetry_summary: Dict[str, Any] = field(default_factory=dict)
    performance_buckets: Dict[str, Any] = field(default_factory=dict)
    recent_trades: List[Dict[str, Any]] = field(default_factory=list)
    screenshot_path: Optional[str] = None
    prop_firm_state: Dict[str, Any] = field(default_factory=dict)
    risk_state: Dict[str, Any] = field(default_factory=dict)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Compact dict form for logging / shadow recording."""
        return {
            "timestamp_utc": (
                self.timestamp_utc.isoformat()
                if hasattr(self.timestamp_utc, "isoformat")
                else str(self.timestamp_utc)
            ),
            "candidate_signal_count": len(self.candidate_signals),
            "candidate_signals": [
                {
                    "strategy_name": getattr(s, "strategy_name", None),
                    "direction": getattr(s, "direction", None),
                    "entry_price": getattr(s, "entry_price", None),
                    "stop_loss": getattr(s, "stop_loss", None),
                    "take_profit": getattr(s, "take_profit", None),
                    "confluence_score": getattr(s, "confluence_score", None),
                }
                for s in self.candidate_signals
            ],
            "indicator_count": len(self.current_indicators),
            "recent_trade_count": len(self.recent_trades),
            "screenshot_path": self.screenshot_path,
            "prop_firm_state": self.prop_firm_state,
            "risk_state": self.risk_state,
        }


class ContextBuilder:
    """Builds :class:`ContextBundle` instances for the agent.

    All inputs are optional — passing ``None`` for a collaborator
    just leaves that section of the bundle empty. This makes the
    builder easy to unit-test with a small subset of fakes.
    """

    def __init__(
        self,
        telemetry_collector: Any | None = None,
        trade_memory: Any | None = None,
        performance_buckets: Any | None = None,
        regime_detector: Any | None = None,
        screenshot_provider: Any | None = None,
    ) -> None:
        self._telemetry = telemetry_collector
        self._trade_memory = trade_memory
        self._perf = performance_buckets
        self._regime = regime_detector
        self._screenshots = screenshot_provider
        # Mutable state used by the SDK MCP tools so they can look up
        # "the current bar" without plumbing it through every call.
        self.current_ts: Optional[datetime] = None
        self.current_candles: Any = None
        self.last_pick: Optional[Dict[str, Any]] = None

    def build(
        self,
        ts: datetime,
        candidate_signals: List[Any],
        current_state: Dict[str, Any],
    ) -> ContextBundle:
        """Assemble the bundle.

        *current_state* carries the engine's "what do I know about
        this bar" dict: recent_bars, indicators, prop_firm_state,
        risk_state. Unknown keys are stored in the bundle's catch-all
        fields.
        """
        self.current_ts = ts
        recent_bars = current_state.get("recent_bars")
        self.current_candles = recent_bars

        current_indicators = current_state.get("current_indicators") or {}
        prop_firm_state = current_state.get("prop_firm_state") or {}
        risk_state = current_state.get("risk_state") or {}

        recent_telemetry_summary: Dict[str, Any] = {}
        if self._telemetry is not None:
            try:
                recent_telemetry_summary = self._telemetry.summary()
            except Exception:
                recent_telemetry_summary = {}

        performance_buckets: Dict[str, Any] = {}
        if self._perf is not None:
            try:
                performance_buckets = self._perf.get_buckets()
            except Exception:
                performance_buckets = {}

        recent_trades: List[Dict[str, Any]] = []
        if self._trade_memory is not None:
            try:
                recent_trades = self._trade_memory.query_recent(n=10)
            except Exception:
                recent_trades = []

        screenshot_path: Optional[str] = None
        if self._screenshots is not None:
            try:
                path = self._screenshots.render_for_decision(
                    ts=ts, candles_window=recent_bars
                )
                if path is not None:
                    screenshot_path = str(path)
            except Exception:
                screenshot_path = None

        # Regime detector is optional — inject into indicators if present
        if self._regime is not None:
            try:
                regime = self._regime.current_regime()
                if regime:
                    current_indicators.setdefault("regime", regime)
            except Exception:
                pass

        return ContextBundle(
            timestamp_utc=ts,
            candidate_signals=list(candidate_signals),
            recent_bars=recent_bars,
            current_indicators=dict(current_indicators),
            recent_telemetry_summary=recent_telemetry_summary,
            performance_buckets=performance_buckets,
            recent_trades=recent_trades,
            screenshot_path=screenshot_path,
            prop_firm_state=dict(prop_firm_state),
            risk_state=dict(risk_state),
        )

    # Convenience access used by the SDK MCP tools ---------------------

    def current_market_state(self) -> Dict[str, Any]:
        return {
            "timestamp_utc": (
                self.current_ts.isoformat()
                if self.current_ts and hasattr(self.current_ts, "isoformat")
                else None
            ),
            "bar_count": (
                len(self.current_candles)
                if self.current_candles is not None and hasattr(self.current_candles, "__len__")
                else 0
            ),
        }
