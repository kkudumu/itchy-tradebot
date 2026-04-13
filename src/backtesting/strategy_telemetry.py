"""Strategy telemetry collection for offline analysis and mega-vision training.

Buffers every signal generation event, every filter rejection, every
entry, and every exit with full context, then flushes to Parquet at
run end. Downstream consumers include:

* Strategy retuning (plan Task 17) — distribution analysis of
  rejection reasons, session buckets, pattern types.
* Mega-vision training data pipeline (plan Task 27) — labeled
  (state → outcome) examples for agent fine-tuning.
* Dashboard visualization (plan Task 13 / 21) — per-strategy
  generated vs entered counts, top rejection stages.

The collector is thread-safe at the emit boundary (a single lock
guards the internal list) so the engine and trade_manager can both
append events from different call sites without coordinating.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event schema
# ---------------------------------------------------------------------------


EventType = Literal[
    "signal_generated",
    "signal_filtered",
    "signal_entered",
    "signal_rejected_in_trade",
    "signal_rejected_no_open",
    "signal_rejected_edge",
    "signal_rejected_learning",
    "signal_rejected_risk",
    "trade_exited",
]


Session = Literal["asian", "london", "overlap", "ny", "off"]


@dataclass
class TelemetryEvent:
    """A single telemetry data point.

    All fields except timestamp/strategy/event_type are optional. The
    ``extra`` dict catches any additional context a caller wants to
    persist without adding a new column.
    """

    timestamp_utc: datetime
    strategy_name: str
    event_type: EventType
    session: Session = "off"
    hour_of_day_utc: int = 0
    day_of_week: int = 0  # 0=Mon
    pattern_type: Optional[str] = None
    direction: Optional[str] = None
    price: Optional[float] = None
    atr: Optional[float] = None
    adx: Optional[float] = None
    confluence_score: Optional[float] = None
    regime: Optional[str] = None
    filter_stage: Optional[str] = None
    rejection_reason: Optional[str] = None
    planned_stop_pips: Optional[float] = None
    planned_tp_pips: Optional[float] = None
    planned_size: Optional[float] = None
    realized_r: Optional[float] = None
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Session + time-of-day helpers
# ---------------------------------------------------------------------------


def classify_session(hour_utc: int) -> Session:
    """Return the forex session label for a UTC hour.

    Bucket ranges:
      * 00:00–07:00 UTC → Asian
      * 07:00–12:00 UTC → London
      * 12:00–16:00 UTC → Overlap (London + NY)
      * 16:00–21:00 UTC → NY
      * 21:00–24:00 UTC → Off-hours
    """
    if 0 <= hour_utc < 7:
        return "asian"
    if 7 <= hour_utc < 12:
        return "london"
    if 12 <= hour_utc < 16:
        return "overlap"
    if 16 <= hour_utc < 21:
        return "ny"
    return "off"


def ts_to_event_fields(ts: datetime) -> dict[str, Any]:
    """Extract session/hour/day-of-week fields from a UTC datetime."""
    hour = ts.hour
    return {
        "hour_of_day_utc": hour,
        "day_of_week": ts.weekday(),
        "session": classify_session(hour),
    }


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


class StrategyTelemetryCollector:
    """Buffered in-memory telemetry collector.

    Constructed once per backtest run by the engine; threaded through
    to trade_manager and any other call site that wants to emit.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._events: list[TelemetryEvent] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Core emit
    # ------------------------------------------------------------------

    def emit(self, event: TelemetryEvent) -> None:
        with self._lock:
            self._events.append(event)

    # ------------------------------------------------------------------
    # Convenience emitters
    # ------------------------------------------------------------------

    def emit_signal_generated(
        self,
        ts: datetime,
        strategy_name: str,
        *,
        direction: str | None = None,
        price: float | None = None,
        atr: float | None = None,
        adx: float | None = None,
        confluence_score: float | None = None,
        pattern_type: str | None = None,
        regime: str | None = None,
        planned_stop_pips: float | None = None,
        planned_tp_pips: float | None = None,
        planned_size: float | None = None,
        **extra: Any,
    ) -> None:
        self.emit(
            TelemetryEvent(
                timestamp_utc=ts,
                strategy_name=strategy_name,
                event_type="signal_generated",
                direction=direction,
                price=price,
                atr=atr,
                adx=adx,
                confluence_score=confluence_score,
                pattern_type=pattern_type,
                regime=regime,
                planned_stop_pips=planned_stop_pips,
                planned_tp_pips=planned_tp_pips,
                planned_size=planned_size,
                extra=extra,
                **ts_to_event_fields(ts),
            )
        )

    def emit_filter_rejection(
        self,
        ts: datetime,
        strategy_name: str,
        filter_stage: str,
        rejection_reason: str,
        *,
        event_type: EventType = "signal_rejected_edge",
        **extra: Any,
    ) -> None:
        self.emit(
            TelemetryEvent(
                timestamp_utc=ts,
                strategy_name=strategy_name,
                event_type=event_type,
                filter_stage=filter_stage,
                rejection_reason=rejection_reason,
                extra=extra,
                **ts_to_event_fields(ts),
            )
        )

    def emit_entry(
        self,
        ts: datetime,
        strategy_name: str,
        *,
        direction: str | None = None,
        price: float | None = None,
        planned_size: float | None = None,
        planned_stop_pips: float | None = None,
        planned_tp_pips: float | None = None,
        pattern_type: str | None = None,
        confluence_score: float | None = None,
        regime: str | None = None,
        **extra: Any,
    ) -> None:
        self.emit(
            TelemetryEvent(
                timestamp_utc=ts,
                strategy_name=strategy_name,
                event_type="signal_entered",
                direction=direction,
                price=price,
                confluence_score=confluence_score,
                pattern_type=pattern_type,
                regime=regime,
                planned_size=planned_size,
                planned_stop_pips=planned_stop_pips,
                planned_tp_pips=planned_tp_pips,
                extra=extra,
                **ts_to_event_fields(ts),
            )
        )

    def emit_trade_exited(
        self,
        ts: datetime,
        strategy_name: str,
        *,
        direction: str | None = None,
        price: float | None = None,
        realized_r: float | None = None,
        pattern_type: str | None = None,
        confluence_score: float | None = None,
        regime: str | None = None,
        **extra: Any,
    ) -> None:
        self.emit(
            TelemetryEvent(
                timestamp_utc=ts,
                strategy_name=strategy_name,
                event_type="trade_exited",
                direction=direction,
                price=price,
                realized_r=realized_r,
                confluence_score=confluence_score,
                pattern_type=pattern_type,
                regime=regime,
                extra=extra,
                **ts_to_event_fields(ts),
            )
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def event_count(self) -> int:
        return len(self._events)

    def events(self) -> list[TelemetryEvent]:
        """Return a snapshot of the event buffer (copy)."""
        with self._lock:
            return list(self._events)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_parquet(self, path: Path | str) -> Path:
        """Write the buffered events to *path* and return the resolved path.

        An empty collector still writes a valid (empty) parquet so
        downstream tools don't need to handle "file missing" as a
        separate case.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = [f.name for f in fields(TelemetryEvent)]

        if not self._events:
            pd.DataFrame(columns=columns).to_parquet(path)
            return path

        rows = [asdict(e) for e in self._events]
        df = pd.DataFrame(rows, columns=columns)
        # Convert the extra dict column to JSON strings so parquet can
        # serialize it without the pyarrow struct headache.
        df["extra"] = df["extra"].apply(lambda d: json.dumps(d, default=str))
        df.to_parquet(path, index=False)
        return path

    def summary(self) -> dict[str, Any]:
        """Return an aggregated summary suitable for JSON dump."""
        per_strategy: dict[str, dict[str, int]] = {}
        per_session: dict[str, int] = {}
        per_pattern: dict[str, int] = {}
        rejection_stages: dict[str, int] = {}

        for evt in self._events:
            s = evt.strategy_name
            per_strategy.setdefault(
                s,
                {"generated": 0, "entered": 0, "rejected": 0, "exited": 0},
            )
            if evt.event_type == "signal_generated":
                per_strategy[s]["generated"] += 1
                if evt.pattern_type:
                    per_pattern[evt.pattern_type] = per_pattern.get(evt.pattern_type, 0) + 1
            elif evt.event_type == "signal_entered":
                per_strategy[s]["entered"] += 1
            elif evt.event_type == "trade_exited":
                per_strategy[s]["exited"] += 1
            elif evt.event_type.startswith("signal_rejected") or evt.event_type == "signal_filtered":
                per_strategy[s]["rejected"] += 1
                if evt.filter_stage:
                    rejection_stages[evt.filter_stage] = (
                        rejection_stages.get(evt.filter_stage, 0) + 1
                    )
            per_session[evt.session] = per_session.get(evt.session, 0) + 1

        # Derived: entry rate
        for s, counts in per_strategy.items():
            gen = counts["generated"]
            counts["entry_rate_pct"] = round(
                (counts["entered"] / gen * 100.0) if gen else 0.0, 2
            )

        return {
            "run_id": self.run_id,
            "total_events": len(self._events),
            "per_strategy": per_strategy,
            "per_session": per_session,
            "per_pattern": per_pattern,
            "top_rejection_stages": dict(
                sorted(rejection_stages.items(), key=lambda kv: kv[1], reverse=True)[:10]
            ),
        }

    def to_summary_json(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.summary(), indent=2, default=str),
            encoding="utf-8",
        )
        return path

    def log_console_summary(self) -> None:
        """Emit a brief summary to the logger at INFO level."""
        summary = self.summary()
        logger.info("Strategy telemetry summary (run_id=%s):", self.run_id)
        for s, counts in summary["per_strategy"].items():
            logger.info(
                "  %-15s: %d generated → %d entered (%.2f%%), %d rejected",
                s,
                counts["generated"],
                counts["entered"],
                counts["entry_rate_pct"],
                counts["rejected"],
            )
        top_rej = summary["top_rejection_stages"]
        if top_rej:
            logger.info("  Top rejection stages:")
            for stage, count in list(top_rej.items())[:5]:
                logger.info("    %-25s: %d", stage, count)
