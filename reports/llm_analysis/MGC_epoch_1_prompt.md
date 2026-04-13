You are a quantitative trading strategy analyst. You have access to the
complete results of an optimization run on MGC.

Your job: analyze all the data below, find patterns, identify what's
working and what's broken, and produce specific actionable changes.

## Trial Summary
Total trials: 196
Combine passes: 107 (54.6%)
Best return: 15.16%
Worst return: -3.42%
Avg trades per trial: 19
Avg win rate: 0.0%


## Signal Pattern Analysis
Strategy | TradeType | Dir | HTF Trend | Entered | Wins | WR | Avg R
--------------------------------------------------------------------------------
fx_at_one_glance |                  | short | ?        |  507 | 156 |  30.8% | +4.16
ema_pullback |                  | short | ?        |  390 | 234 |  60.0% | +6.09
ema_pullback |                  | long  | ?        |  273 | 117 |  42.9% | -0.27
fx_at_one_glance |                  | long  | ?        |   78 |  39 |  50.0% | -0.40

## Charts
No chart screenshots available this epoch.

## Your Prior Analysis
No prior analysis history.

## Current Strategy Code
### src/strategy/strategies/sss/strategy.py
```python
"""SSSStrategy — orchestrates all SSS sub-components per 1M bar."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from ...signal_engine import Signal
from .breathing_room import BreathingRoomDetector, SwingPoint
from .cbc_detector import CBCDetector, CBCSignal
from .fifty_tap import FiftyTapCalculator, FiftyTapLevel
from .sequence_exit import SequenceExitMode
from .sequence_tracker import SequenceTracker, TWO_CONFIRMED, THREE_ACTIVE
from .ss_level_manager import SSLevelManager

log = logging.getLogger(__name__)

_DEFAULT_CONFIG: dict = {
    "swing_lookback_n": 3,
    "min_swing_pips": 1.0,
    "pip_value": 0.1,
    "ss_candle_min": 10,
    "iss_candle_min": 4,
    "iss_candle_max": 8,
    "max_bars_in_state": 50,
    "max_active_levels": 50,
    "require_cbc_context": True,
    "fifty_tap_level": 0.5,
    "fifty_tap_tolerance_pips": 0.5,
    "entry_mode": "combined",   # "cbc_only", "fifty_tap", "combined"
    "min_confluence_score": 4,
    "tier_a_plus": 7,
    "tier_b": 5,
    "warmup_bars": 100,
    "spread_multiplier": 2.0,
    "min_stop_pips": 10.0,
    "instrument": "XAUUSD",
}

_ENTRY_STATES = frozenset({TWO_CONFIRMED, THREE_ACTIVE})


class SSSStrategy:
    """Main SSS strategy — orchestrates all components per bar.

    Parameters
    ----------
    config:
        Optional overrides for any key in _DEFAULT_CONFIG.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self._cfg: dict = {**_DEFAULT_CONFIG, **(config or {})}

        self._swing_detector = BreathingRoomDetector(
            lookback_n=int(self._cfg["swing_lookback_n"]),
            min_swing_pips=float(self._cfg["min_swing_pips"]),
            pip_value=float(self._cfg["pip_value"]),
        )
        self._seq_tracker = SequenceTracker(config={
            "ss_candle_min": self._cfg["ss_candle_min"],
            "iss_candle_min": self._cfg["iss_candle_min"],
            "iss_candle_max": self._cfg["iss_candle_max"],
            "max_bars_in_state": self._cfg["max_bars_in_state"],
        })
        self._level_mgr = SSLevelManager(max_active_levels=int(self._cfg["max_active_levels"]))
        self._cbc_detector = CBCDetector(require_context=bool(self._cfg["require_cbc_context"]))
        self._fifty_tap = FiftyTapCalculator(
            tap_level=float(self._cfg["fifty_tap_level"]),
            tolerance_pips=float(self._cfg["fifty_tap_tolerance_pips"]),
            pip_value=float(self._cfg["pip_value"]),
        )
        self._exit_mode = SequenceExitMode(
            spread_multiplier=float(self._cfg["spread_multiplier"]),
            min_stop_pips=float(self._cfg["min_stop_pips"]),
            pip_value=float(self._cfg["pip_value"]),
        )

        self._bar_count: int = 0
        self._swing_history: List[SwingPoint] = []
        self._last_cbc: Optional[CBCSignal] = None
        self._active_fifty_tap: Optional[FiftyTapLevel] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_bar(
        self,
        timestamp: datetime,
        *,
        open: float,
        high: float,
        low: float,
        close: float,
        atr: float,
        spread: float = 0.0,
    ) -> Optional[Signal]:
        """Process one 1M bar. Returns Signal if entry conditions met."""
        self._bar_count += 1

        # 1. Swing detection
        new_swing = self._swing_detector.detect_incremental(
            self._bar_count, timestamp, high, low, self._swing_history
        )

        # 2–3. Sequence state machine + SS level creation
        if new_swing is not None:
            for evt in self._seq_tracker.on_swing(new_swing):
                self._level_mgr.on_sequence_event(evt)

        # 4. Sweep check
        self._level_mgr.update_price(timestamp, high, low)

        # 5. CBC detection
        seq_dir = self._seq_tracker.direction
        cbc_signal = self._cbc_detector.on_bar(
            self._bar_count, timestamp, open, high, low, close,
            context_direction=seq_dir,
        )
        if cbc_signal is not None:
            self._last_cbc = cbc_signal

        # 6. 50% tap update
        self._update_fifty_tap(high, low)

        # 7. Warmup guard
        if self._bar_count < self._cfg["warmup_bars"]:
            return None

        return self._check_entry(timestamp, close, atr, spread)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_fifty_tap(self, high: float, low: float) -> None:
        """Recalculate or update the active 50% tap level."""
        direction = self._seq_tracker.direction
        if direction is None or len(self._swing_history) < 2:
            self._active_fifty_tap = None
            return

        last = self._swing_history[-1]
        prev = self._swing_history[-2]
        expected_type = "low" if direction == "bullish" else "high"

        if last.swing_type == expected_type and prev.swing_type == expected_type:
            self._active_fifty_tap = self._fifty_tap.calculate_level(
```

### src/strategy/strategies/ema_pullback.py
```python
"""EMA Pullback State Machine strategy for gold (XAUUSD).

Based on the ilahuerta-IA/backtrader-pullback-window-xauusd approach that
achieved 55% WR, PF 1.64, 5.8% max DD over 5 years of gold trading.

4-phase state machine
~~~~~~~~~~~~~~~~~~~~~
1. SCANNING    — Wait for EMA alignment (fast>mid>slow or fast<mid<slow) at a
                 sufficient angle, then detect the first counter-trend
                 (pullback) candle.
2. ARMED       — Count consecutive pullback candles.  Track the pre-pullback
                 swing high (long) or low (short) as the breakout reference.
                 Advance to WINDOW_OPEN when pullback_candles_min are met.
                 Reset to SCANNING if pullback_candles_max exceeded.
3. WINDOW_OPEN — Wait for close to break above breakout_level (long) or below
                 (short).  Emit a Signal on breakout.  Reset to SCANNING if
                 breakout_window_bars expire without a breakout.
4. Back to SCANNING after any signal or expiry.
"""

from __future__ import annotations

import math
import datetime as dt
from typing import Optional

from ..signal_engine import Signal


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: dict = {
    "fast_ema": 14,
    "mid_ema": 18,
    "slow_ema": 24,
    "min_ema_angle_deg": 30,
    "max_ema_angle_deg": 95,
    "pullback_candles_min": 1,
    "pullback_candles_max": 3,
    "breakout_window_bars": 20,
    "rr_ratio": 2.0,
    "weight": 1.0,
    "instrument": "XAUUSD",
}


# ---------------------------------------------------------------------------
# EMAPullbackStrategy
# ---------------------------------------------------------------------------

class EMAPullbackStrategy:
    """Bar-by-bar EMA pullback strategy using a 4-phase state machine.

    Parameters
    ----------
    config:
        Optional overrides for any key in ``_DEFAULT_CONFIG``.
    """

    # Valid states
    _STATES = ("SCANNING", "ARMED", "WINDOW_OPEN")

    def __init__(self, config: Optional[dict] = None) -> None:
        self._cfg: dict = {**_DEFAULT_CONFIG, **(config or {})}

        # Current state
        self._state: str = "SCANNING"

        # Previous bar's ema_fast — used for angle calculation
        self._prev_ema_fast: Optional[float] = None

        # Trend direction detected in SCANNING: 'long' | 'short' | None
        self._trend: Optional[str] = None

        # Pre-pullback high (long) or low (short) used as breakout level
        self._breakout_level: Optional[float] = None

        # Running count of consecutive pullback candles in ARMED state
        self._pullback_count: int = 0

        # Bar counter inside WINDOW_OPEN state
        self._window_bars: int = 0

        # SL anchor: slow EMA at the bar we armed (updated each bar in ARMED)
        self._armed_slow_ema: Optional[float] = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        """Current state machine state."""
        return self._state

    # ------------------------------------------------------------------
    # Core bar handler
    # ------------------------------------------------------------------

    def on_bar(
        self,
        timestamp: dt.datetime,
        *,
        open: float,
        high: float,
        low: float,
        close: float,
        ema_fast: float,
        ema_mid: float,
        ema_slow: float,
        atr: float = 5.0,
    ) -> Optional[Signal]:
        """Process one bar and optionally return a Signal.

        Parameters
        ----------
        timestamp:
            Bar open timestamp (timezone-aware UTC recommended).
        open, high, low, close:
            OHLCV prices for the bar.
        ema_fast, ema_mid, ema_slow:
            Pre-computed EMA values.
        atr:
            Average True Range at this bar (used for angle normalisation).

        Returns
        -------
        Signal or None
        """
        signal: Optional[Signal] = None

        if self._state == "SCANNING":
            signal = self._handle_scanning(
                timestamp, open, high, low, close,
                ema_fast, ema_mid, ema_slow, atr,
            )
        elif self._state == "ARMED":
            signal = self._handle_armed(
                timestamp, open, high, low, close,
                ema_fast, ema_mid, ema_slow, atr,
            )
        elif self._state == "WINDOW_OPEN":
            signal = self._handle_window_open(
                timestamp, open, high, low, close,
                ema_fast, ema_mid, ema_slow, atr,
            )

        # Always update previous ema_fast for angle calc on next bar
```

### src/strategy/strategies/asian_breakout.py
```python
"""Asian Range Breakout strategy for London session gold trading.

Marks the Asian session (21:00–06:00 UTC) high/low on gold, then trades the
breakout during the London session (06:00–10:00 UTC).

Session windows (UTC):
    Asian  : 21:00 – 06:00  (next day)
    London : 06:00 – 10:00

Signal rules:
    - Lock Asian range at 06:00 UTC.
    - Validate range: min_range_pips <= range_pips <= max_range_pips.
      (Gold: 1 pip = $0.10, so range_pips = (high - low) * 10)
    - Long  : close > asian_high during London window.
    - Short : close < asian_low  during London window.
    - SL    : opposite side of the range.
    - TP    : entry + risk * rr_ratio.
    - Max 1 signal per calendar day.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

from ..signal_engine import Signal


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: dict = {
    "min_range_pips": 10,    # minimum Asian range in pips
    "max_range_pips": 200,   # maximum Asian range in pips
    "rr_ratio": 2.0,         # reward-to-risk ratio for TP
    "pip_value": 0.1,        # price delta per "pip" (gold: $0.10 = 1 pip; MGC: 1 tick = $0.10)
    "instrument": "XAUUSD",
}


# ---------------------------------------------------------------------------
# AsianBreakoutStrategy
# ---------------------------------------------------------------------------

class AsianBreakoutStrategy:
    """Standalone Asian Range Breakout strategy.

    Unlike the abstract Strategy base, this class uses a simple bar-by-bar
    ``on_bar`` interface (no evaluator matrix required) so it can be used
    both in backtests and in the multi-strategy signal blender.

    Parameters
    ----------
    config:
        Optional config dict with keys: min_range_pips, max_range_pips,
        rr_ratio, instrument.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self._cfg: dict = {**_DEFAULT_CONFIG, **(config or {})}

        # Asian range accumulators (reset each session)
        self._asian_high: Optional[float] = None
        self._asian_low: Optional[float] = None
        self._range_locked: bool = False   # True once we pass 06:00 UTC

        # Day-tracking to trigger a daily reset
        self._session_date: Optional[dt.date] = None  # date of 21:00 candle
        self._signal_fired: bool = False               # one signal per day

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def asian_high(self) -> Optional[float]:
        """Highest high seen during the current Asian session."""
        return self._asian_high

    @property
    def asian_low(self) -> Optional[float]:
        """Lowest low seen during the current Asian session."""
        return self._asian_low

    @property
    def asian_range_pips(self) -> Optional[float]:
        """Range expressed in pips.

        A "pip" here is configurable via ``pip_value`` in the strategy
        config (default 0.1 price units, matching the historical gold
        convention of $0.10 per pip). For MGC futures where tick_size
        is also 0.10, setting pip_value=0.1 yields the same pip count
        as the forex path — no re-tuning needed at the boundary.
        """
        if self._asian_high is None or self._asian_low is None:
            return None
        pip_value = float(self._cfg.get("pip_value", 0.1))
        if pip_value <= 0:
            pip_value = 0.1
        return (self._asian_high - self._asian_low) / pip_value

    @property
    def range_valid(self) -> bool:
        """True if the locked Asian range is within configured pip limits."""
        rp = self.asian_range_pips
        if rp is None:
            return False
        return (
            rp > self._cfg["min_range_pips"]
            and rp < self._cfg["max_range_pips"]
        )

    # ------------------------------------------------------------------
    # Bar processing
    # ------------------------------------------------------------------

    def on_bar(
        self,
        timestamp: dt.datetime,
        *,
        high: float,
        low: float,
        close: float,
    ) -> Optional[Signal]:
        """Process one OHLCV bar and optionally return a Signal.

        Parameters
        ----------
        timestamp:
            Bar open time (timezone-aware UTC datetime).
        high, low, close:
            OHLCV values for this bar.

        Returns
        -------
        Signal or None
        """
        hour = timestamp.hour
        minute = timestamp.minute
        bar_date = timestamp.date()

        # ------------------------------------------------------------------
        # 1. Session reset: when we enter a new Asian session (21:00 UTC)
        #    on a different calendar day than the previous session.
        # ------------------------------------------------------------------
        if hour == 21 and (self._session_date is None or bar_date != self._session_date):
            self._asian_high = None
            self._asian_low = None
```

### src/strategy/strategies/ichimoku.py
```python
"""IchimokuStrategy — wraps the existing Ichimoku MTF cascade as a Strategy subclass."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from ..base import (
    Strategy,
    EvalMatrix,
    EvalRequirement,
    ConfluenceResult as BaseConfluenceResult,
    EvaluatorResult,
)
from ..signal_engine import Signal
from ..trading_modes.kijun_exit import KijunExitMode
from ...zones.manager import ZoneManager


def _is_nan(v: float) -> bool:
    try:
        return math.isnan(v)
    except (TypeError, ValueError):
        return True


class IchimokuStrategy(Strategy, key='ichimoku'):
    """MTF Ichimoku strategy: 4H->1H->15M->5M cascade (default) or 15M->5M mode.

    Replicates the exact signal generation logic from SignalEngine.scan()
    and confluence scoring from ConfluenceScorer.score().

    When ``config["timeframes"]`` is ``["15M", "5M"]`` the 4H and 1H gates
    are dropped and the 15M cloud direction is used as the primary trend
    filter, producing significantly more signals for prop-firm trading.
    """

    required_evaluators = [
        EvalRequirement('ichimoku', ['4H', '1H', '15M', '5M']),
        EvalRequirement('adx', ['15M']),
        EvalRequirement('atr', ['15M']),
        EvalRequirement('session', ['5M']),
    ]

    config_model = None  # Will be set to IchimokuConfig after import
    trading_mode = None  # Set in __init__
    warmup_bars = 0  # Will be calculated

    # Confluence constants
    _ACTIVE_SESSIONS = {"london", "new_york", "overlap"}

    def __init__(self, config=None, instrument: str = "XAUUSD") -> None:
        # Default config values matching _DEFAULT_CONFIG in signal_engine.py
        self._cfg = {
            "tenkan_period": 9,
            "kijun_period": 26,
            "senkou_b_period": 52,
            "adx_threshold": 28,
            "adx_period": 14,
            "atr_period": 14,
            "atr_stop_multiplier": 1.5,
            "min_confluence_score": 4,
            "rr_ratio": 2.0,
            "kijun_proximity_atr": 0.5,
        }

        if config is not None:
            # Config can be a Pydantic model or a dict
            if hasattr(config, 'model_dump'):
                config_dict = config.model_dump()
            elif isinstance(config, dict):
                config_dict = config
            else:
                config_dict = {}
            self._cfg.update(config_dict)

        self.instrument = instrument
        self.trading_mode = KijunExitMode()
        self.zone_manager = ZoneManager()

        # Determine mode from configured timeframes
        timeframes = self._cfg.get("timeframes", ["4H", "1H", "15M", "5M"])
        self._15m_only = ("4H" not in timeframes and "1H" not in timeframes)

        # Dynamic required_evaluators based on mode
        if self._15m_only:
            self.required_evaluators = [
                EvalRequirement('ichimoku', ['15M', '5M']),
                EvalRequirement('adx', ['15M']),
                EvalRequirement('atr', ['15M']),
                EvalRequirement('session', ['5M']),
            ]
            # 15M = 60 1M bars, kijun needs 26 bars: 26 * 60 = 1560
            self.warmup_bars = self._cfg["kijun_period"] * 60
        else:
            self.required_evaluators = [
                EvalRequirement('ichimoku', ['4H', '1H', '15M', '5M']),
                EvalRequirement('adx', ['15M']),
                EvalRequirement('atr', ['15M']),
                EvalRequirement('session', ['5M']),
            ]
            # 4H = 240 1M bars, kijun needs 26 bars: 26 * 240 = 6240
            self.warmup_bars = self._cfg["kijun_period"] * 240

    # ------------------------------------------------------------------
    # Strategy ABC: decide()
    # ------------------------------------------------------------------

    def decide(self, eval_matrix: EvalMatrix) -> Optional[Signal]:
        """4H->1H->15M->5M cascade (default) or 15M->5M mode. Returns Signal or None.

        This replicates the exact logic from SignalEngine.scan().
        In 15M-only mode, the 4H and 1H gates are skipped and the 15M cloud
        direction is used as the primary trend filter.
        """
        reasoning: dict = {}

        if self._15m_only:
            direction, reasoning = self._decide_15m_trend(eval_matrix, reasoning)
        else:
            direction, reasoning = self._decide_4h_cascade(eval_matrix, reasoning)

        if direction is None:
            return None

        # --- Step 4: 5M entry timing (pullback to Kijun) ---
        ichi_5m = eval_matrix.get('ichimoku_5M')
        if ichi_5m is None:
            return None

        close_5m = ichi_5m.metadata.get('close', 0.0)
        kijun_5m = ichi_5m.metadata.get('kijun', float('nan'))

        atr_result = eval_matrix.get('atr_15M')
        atr = atr_result.metadata.get('atr', 0.0) if atr_result else 0.0

        if _is_nan(close_5m) or _is_nan(kijun_5m) or _is_nan(atr) or atr <= 0:
            return None

        proximity = self._cfg["kijun_proximity_atr"] * atr
        if abs(close_5m - kijun_5m) > proximity:
            return None  # Too far from Kijun

        reasoning["5m_entry"] = {"pass": True, "close": close_5m, "kijun": kijun_5m}

        # --- Step 5: Zone context ---
        nearby_zones = self.zone_manager.get_nearby_zones(
```

### src/edges/trend_direction.py
```python
"""
Trend direction alignment filter.

Blocks counter-trend entries by comparing the signal direction against the
higher-timeframe trend determined by EMA slope. When the 1H (or 4H) EMA
is falling, long entries are blocked; when rising, short entries are
blocked.

The trend is measured by comparing the current HTF close to an N-bar
simple moving average of HTF closes. If current close > SMA → bullish;
if below → bearish. This is robust across price scales (works equally
for gold at $5000 and oil at $70) because it's relative.

Config keys (via ``params``):
    htf_lookback:    int   — Number of 1H bars to compute SMA over.
                             Default 50 (≈2 trading days).
    require_both_tf: bool  — When True, both 1H and 4H must agree.
                             When False, 1H is sufficient. Default False.
    tolerance_pct:   float — Deadband: if price is within this % of the
                             SMA, trend is "neutral" and all signals pass.
                             Default 0.5 (0.5%).
"""

from __future__ import annotations

from .base import EdgeContext, EdgeFilter, EdgeResult


class TrendDirectionFilter(EdgeFilter):
    """Block entries that fight the higher-timeframe trend."""

    def __init__(self, config: dict) -> None:
        super().__init__("trend_direction", config)
        params = config.get("params", {})
        self._htf_lookback = int(params.get("htf_lookback", 50))
        self._require_both_tf = bool(params.get("require_both_tf", False))
        self._tolerance_pct = float(params.get("tolerance_pct", 0.5))
        self._block_without_data = bool(params.get("block_without_data", False))

    def should_allow(self, context: EdgeContext) -> EdgeResult:
        if not self.enabled:
            return self._disabled_result()

        # Get signal direction
        direction = None
        if context.signal is not None:
            direction = getattr(context.signal, "direction", None)
        if direction is None:
            return EdgeResult(
                allowed=True,
                edge_name=self.name,
                reason="no signal direction — filter skipped",
            )

        # Determine HTF trend from indicator_values populated by the engine
        trend_1h = context.indicator_values.get("htf_trend_1h")  # "bullish", "bearish", "neutral"
        trend_4h = context.indicator_values.get("htf_trend_4h")

        if trend_1h is None and trend_4h is None:
            return EdgeResult(
                allowed=True,
                edge_name=self.name,
                reason="no HTF trend data — filter skipped",
            )

        # Use 1H as primary, 4H as secondary
        primary_trend = trend_1h or trend_4h or "neutral"

        if self._require_both_tf and trend_1h and trend_4h:
            # Both must agree for a strong trend call
            if trend_1h != trend_4h:
                primary_trend = "neutral"

        # Block counter-trend entries
        if direction == "long" and primary_trend == "bearish":
            return EdgeResult(
                allowed=False,
                edge_name=self.name,
                reason=f"long blocked: HTF trend is bearish (1H={trend_1h}, 4H={trend_4h})",
            )

        if direction == "short" and primary_trend == "bullish":
            return EdgeResult(
                allowed=False,
                edge_name=self.name,
                reason=f"short blocked: HTF trend is bullish (1H={trend_1h}, 4H={trend_4h})",
            )

        return EdgeResult(
            allowed=True,
            edge_name=self.name,
            reason=f"trend aligned: {direction} with {primary_trend} (1H={trend_1h}, 4H={trend_4h})",
        )

```


---

Respond with EXACTLY three sections:

### REASONING
Your analysis of what's working, what's failing, and why. Reference
specific signal patterns, trade types, and win rates. Be precise.

### CONFIG_CHANGES
```json
{"config_changes": {"dotted.path.to.param": value}}
```

Dotted paths target strategy.yaml keys. Examples:
- "strategies.sss.entry_mode": "fifty_tap"
- "strategies.ema_pullback.enabled": false
- "risk.initial_risk_pct": 0.3
- "strategies.ichimoku.adx.threshold": 12
- "edges.trend_direction.enabled": true

### CODE_PATCHES
```json
{"code_patches": [{"file": "src/path/to/file.py", "description": "why this change helps", "search": "exact old code to find", "replace": "new code to put there"}]}
```

Rules:
- Only suggest code patches you are confident will improve results
- Config changes are safe — be aggressive
- Code patches will be tested on a branch — suggest bold structural fixes
- Reference specific signal patterns and trade types in your reasoning
- If a strategy is a net loser on this instrument, disable it via config
- If stops are too tight or wide, fix the code that computes them
- If a pattern has >60% WR, suggest increasing its weight or priority
- If a pattern has <20% WR, suggest filtering it out
