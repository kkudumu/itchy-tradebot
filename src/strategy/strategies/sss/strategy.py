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
                direction=direction,
                anchor_swing=last,
                target_swing=prev,
            )

        if self._active_fifty_tap is not None:
            self._active_fifty_tap = self._fifty_tap.check_invalidation(
                self._active_fifty_tap, high, low
            )
            if not self._active_fifty_tap.is_invalidated:
                self._active_fifty_tap = self._fifty_tap.check_tap(
                    self._active_fifty_tap, self._bar_count, high, low
                )

    def _check_entry(
        self, timestamp: datetime, close: float, atr: float, spread: float
    ) -> Optional[Signal]:
        """Evaluate all entry conditions and return Signal or None."""
        state = self._seq_tracker.state
        direction = self._seq_tracker.direction

        if state not in _ENTRY_STATES or direction is None:
            return None

        trade_dir = "long" if direction == "bullish" else "short"

        cbc_ok = self._last_cbc is not None and self._last_cbc.direction == direction
        tap_ok = (
            self._active_fifty_tap is not None
            and self._active_fifty_tap.is_tapped
            and not self._active_fifty_tap.is_invalidated
            and self._active_fifty_tap.direction == direction
        )

        mode = self._cfg["entry_mode"]
        if mode == "cbc_only" and not cbc_ok:
            return None
        if mode == "fifty_tap" and not tap_ok:
            return None
        if mode == "combined" and not (cbc_ok and tap_ok):
            return None

        score, breakdown = self._score_confluence(trade_dir, direction, cbc_ok, tap_ok, close, atr)
        if score < self._cfg["min_confluence_score"]:
            return None

        invalidation = self._get_invalidation_price(trade_dir)
        if invalidation is None:
            return None

        stop_loss = self._exit_mode.calculate_initial_stop(
            direction=trade_dir,
            invalidation_price=invalidation,
            spread=spread,
            entry_price=close,
        )
        if stop_loss is None:
            return None

        take_profit = self._get_take_profit(close, stop_loss, trade_dir)
        nearest_target = self._nearest_target_price(close, trade_dir)

        return Signal(
            timestamp=timestamp,
            instrument=self._cfg["instrument"],
            direction=trade_dir,
            entry_price=close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confluence_score=score,
            quality_tier=self._quality_tier(score),
            atr=atr,
            strategy_name="sss",
            reasoning={
                "sequence_state": state,
                "sequence_direction": direction,
                "ss_level_target": nearest_target,
                "cbc_type": self._last_cbc.cbc_type if self._last_cbc else None,
                "fifty_tap_status": (
                    "tapped" if tap_ok
                    else ("active" if self._active_fifty_tap else "none")
                ),
                "layers_aligned": breakdown.get("ss_aligned", False),
                "confluence_breakdown": breakdown,
            },
        )

    def _score_confluence(
        self,
        trade_dir: str,
        seq_dir: str,
        cbc_ok: bool,
        tap_ok: bool,
        current_price: float,
        atr: float,
    ) -> tuple[int, dict]:
        """Score confluence 0–8. Returns (score, breakdown)."""
        score = 0
        bd: dict = {}

        hist = self._seq_tracker.history
        entry_evts = ("two_confirmed", "three_active")

        ss_events = [e for e in hist if e.layer == "SS" and e.event_type in entry_evts]
        iss_events = [e for e in hist if e.layer == "ISS" and e.event_type in entry_evts]

        ss_aligned = bool(ss_events) and ss_events[-1].direction == seq_dir
        iss_aligned = bool(iss_events) and iss_events[-1].direction == seq_dir

        score += 2 if ss_aligned else 0
        score += 2 if iss_aligned else 0
        score += 1 if cbc_ok else 0
        score += 1 if tap_ok else 0

        nearest = self._level_mgr.get_nearest_target(current_price, trade_dir)
        target_near = nearest is not None and abs(nearest.price - current_price) <= 5.0 * atr
        score += 1 if target_near else 0

        recent_f2 = any(
            e.event_type == "f2_detected" and e.direction == seq_dir
            for e in hist[-10:]
        )
        no_f2 = not recent_f2
        score += 1 if no_f2 else 0

        bd["ss_aligned"] = ss_aligned
        bd["iss_aligned"] = iss_aligned
        bd["cbc_confirmed"] = cbc_ok
        bd["fifty_tap_confirmed"] = tap_ok
        bd["target_proximity"] = target_near
        bd["no_f2_conflict"] = no_f2

        return score, bd

    def _get_invalidation_price(self, trade_dir: str) -> Optional[float]:
        if not self._swing_history:
            return None
        if trade_dir == "long":
            lows = [s for s in self._swing_history if s.swing_type == "low"]
            return lows[-1].price if lows else None
        highs = [s for s in self._swing_history if s.swing_type == "high"]
        return highs[-1].price if highs else None

    def _get_take_profit(self, entry: float, stop_loss: float, trade_dir: str) -> float:
        nearest = self._level_mgr.get_nearest_target(entry, trade_dir)
        if nearest is not None:
            return nearest.price
        risk = abs(entry - stop_loss)
        return entry + risk * 2.0 if trade_dir == "long" else entry - risk * 2.0

    def _nearest_target_price(self, current_price: float, trade_dir: str) -> Optional[float]:
        lvl = self._level_mgr.get_nearest_target(current_price, trade_dir)
        return lvl.price if lvl is not None else None

    def _quality_tier(self, score: int) -> str:
        if score >= self._cfg["tier_a_plus"]:
            return "A+"
        if score >= self._cfg["tier_b"]:
            return "B"
        return "C"
