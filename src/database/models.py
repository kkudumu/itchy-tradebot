"""
Python dataclasses that mirror the database schema.

All models are plain dataclasses — no ORM dependency.  Each model provides:
  - from_row(row)   — construct from a psycopg2 tuple/dict row
  - to_dict()       — serialise to a plain dict (e.g. for JSON logging)

Enum classes enforce the same CHECK constraints defined in schema.sql.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


# =============================================================================
# Enums (mirror SQL CHECK constraints)
# =============================================================================

class TradeSource(str, Enum):
    BACKTEST = "backtest"
    LIVE     = "live"
    PAPER    = "paper"


class TradeDirection(str, Enum):
    LONG  = "long"
    SHORT = "short"


class TradeStatus(str, Enum):
    OPEN      = "open"
    PARTIAL   = "partial"
    CLOSED    = "closed"
    CANCELLED = "cancelled"


class ZoneType(str, Enum):
    SUPPORT    = "support"
    RESISTANCE = "resistance"
    SUPPLY     = "supply"
    DEMAND     = "demand"
    PIVOT      = "pivot"


class ZoneStatus(str, Enum):
    ACTIVE      = "active"
    TESTED      = "tested"
    INVALIDATED = "invalidated"


class DecisionAction(str, Enum):
    ENTER        = "enter"
    SKIP         = "skip"
    EXIT         = "exit"
    PARTIAL_EXIT = "partial_exit"
    MODIFY       = "modify"


class ScreenshotPhase(str, Enum):
    PRE_ENTRY = "pre_entry"
    ENTRY     = "entry"
    DURING    = "during"
    EXIT      = "exit"


# =============================================================================
# Helper
# =============================================================================

def _as_enum(cls, value):
    """Return enum member for value, or None if value is None."""
    if value is None:
        return None
    return cls(value)


# =============================================================================
# 1. Candle
# =============================================================================

@dataclass
class Candle:
    timestamp:  datetime
    instrument: str
    open:       float
    high:       float
    low:        float
    close:      float
    volume:     float

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "Candle":
        return cls(
            timestamp  = row["timestamp"],
            instrument = row["instrument"],
            open       = float(row["open"]),
            high       = float(row["high"]),
            low        = float(row["low"]),
            close      = float(row["close"]),
            volume     = float(row["volume"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


# =============================================================================
# 2. Trade
# =============================================================================

@dataclass
class Trade:
    id:               Optional[int]
    instrument:       str
    source:           TradeSource
    direction:        TradeDirection
    entry_time:       datetime
    entry_price:      float
    stop_loss:        float
    lot_size:         float
    risk_pct:         float
    exit_time:        Optional[datetime]          = None
    exit_price:       Optional[float]             = None
    take_profit:      Optional[float]             = None
    r_multiple:       Optional[float]             = None
    pnl:              Optional[float]             = None
    pnl_pct:          Optional[float]             = None
    status:           TradeStatus                 = TradeStatus.OPEN
    confluence_score: Optional[int]               = None
    signal_tier:      Optional[str]               = None
    created_at:       Optional[datetime]          = None

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "Trade":
        return cls(
            id               = row.get("id"),
            instrument       = row["instrument"],
            source           = TradeSource(row["source"]),
            direction        = TradeDirection(row["direction"]),
            entry_time       = row["entry_time"],
            exit_time        = row.get("exit_time"),
            entry_price      = float(row["entry_price"]),
            exit_price       = float(row["exit_price"])       if row.get("exit_price")   is not None else None,
            stop_loss        = float(row["stop_loss"]),
            take_profit      = float(row["take_profit"])      if row.get("take_profit")  is not None else None,
            lot_size         = float(row["lot_size"]),
            risk_pct         = float(row["risk_pct"]),
            r_multiple       = float(row["r_multiple"])       if row.get("r_multiple")   is not None else None,
            pnl              = float(row["pnl"])              if row.get("pnl")          is not None else None,
            pnl_pct          = float(row["pnl_pct"])          if row.get("pnl_pct")      is not None else None,
            status           = TradeStatus(row.get("status", "open")),
            confluence_score = row.get("confluence_score"),
            signal_tier      = row.get("signal_tier"),
            created_at       = row.get("created_at"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":               self.id,
            "instrument":       self.instrument,
            "source":           self.source.value,
            "direction":        self.direction.value,
            "entry_time":       self.entry_time.isoformat(),
            "exit_time":        self.exit_time.isoformat()  if self.exit_time  else None,
            "entry_price":      self.entry_price,
            "exit_price":       self.exit_price,
            "stop_loss":        self.stop_loss,
            "take_profit":      self.take_profit,
            "lot_size":         self.lot_size,
            "risk_pct":         self.risk_pct,
            "r_multiple":       self.r_multiple,
            "pnl":              self.pnl,
            "pnl_pct":          self.pnl_pct,
            "status":           self.status.value,
            "confluence_score": self.confluence_score,
            "signal_tier":      self.signal_tier,
            "created_at":       self.created_at.isoformat() if self.created_at else None,
        }


# =============================================================================
# 3. MarketContext
# =============================================================================

@dataclass
class MarketContext:
    id:                     Optional[int]
    timestamp:              datetime
    instrument:             str
    trade_id:               Optional[int]            = None
    cloud_direction_4h:     Optional[str]            = None
    cloud_direction_1h:     Optional[str]            = None
    tk_cross_15m:           Optional[str]            = None
    chikou_confirmation:    Optional[bool]           = None
    cloud_thickness_4h:     Optional[float]          = None
    adx_value:              Optional[float]          = None
    atr_value:              Optional[float]          = None
    rsi_value:              Optional[float]          = None
    bb_width_percentile:    Optional[float]          = None
    session:                Optional[str]            = None
    nearest_sr_distance:    Optional[float]          = None
    zone_confluence_score:  Optional[int]            = None
    # Stored as a list of floats; pgvector handles the VECTOR type on the DB side
    context_embedding:      Optional[List[float]]    = None
    created_at:             Optional[datetime]       = None

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "MarketContext":
        embedding = row.get("context_embedding")
        # psycopg2 with pgvector adapter returns a list; plain strings are
        # also handled for test environments that do not have the adapter.
        if isinstance(embedding, str):
            embedding = [float(x) for x in embedding.strip("[]").split(",")]

        return cls(
            id                    = row.get("id"),
            trade_id              = row.get("trade_id"),
            timestamp             = row["timestamp"],
            instrument            = row["instrument"],
            cloud_direction_4h    = row.get("cloud_direction_4h"),
            cloud_direction_1h    = row.get("cloud_direction_1h"),
            tk_cross_15m          = row.get("tk_cross_15m"),
            chikou_confirmation   = row.get("chikou_confirmation"),
            cloud_thickness_4h    = _opt_float(row.get("cloud_thickness_4h")),
            adx_value             = _opt_float(row.get("adx_value")),
            atr_value             = _opt_float(row.get("atr_value")),
            rsi_value             = _opt_float(row.get("rsi_value")),
            bb_width_percentile   = _opt_float(row.get("bb_width_percentile")),
            session               = row.get("session"),
            nearest_sr_distance   = _opt_float(row.get("nearest_sr_distance")),
            zone_confluence_score = row.get("zone_confluence_score"),
            context_embedding     = embedding,
            created_at            = row.get("created_at"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":                    self.id,
            "trade_id":              self.trade_id,
            "timestamp":             self.timestamp.isoformat(),
            "instrument":            self.instrument,
            "cloud_direction_4h":    self.cloud_direction_4h,
            "cloud_direction_1h":    self.cloud_direction_1h,
            "tk_cross_15m":          self.tk_cross_15m,
            "chikou_confirmation":   self.chikou_confirmation,
            "cloud_thickness_4h":    self.cloud_thickness_4h,
            "adx_value":             self.adx_value,
            "atr_value":             self.atr_value,
            "rsi_value":             self.rsi_value,
            "bb_width_percentile":   self.bb_width_percentile,
            "session":               self.session,
            "nearest_sr_distance":   self.nearest_sr_distance,
            "zone_confluence_score": self.zone_confluence_score,
            "context_embedding":     self.context_embedding,
            "created_at":            self.created_at.isoformat() if self.created_at else None,
        }


# =============================================================================
# 4. PatternSignature
# =============================================================================

@dataclass
class PatternSignature:
    id:            Optional[int]
    embedding:     List[float]
    context_id:    Optional[int]   = None
    trade_id:      Optional[int]   = None
    outcome_r:     Optional[float] = None
    win:           Optional[bool]  = None
    cluster_label: Optional[int]   = None
    created_at:    Optional[datetime] = None

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "PatternSignature":
        embedding = row["embedding"]
        if isinstance(embedding, str):
            embedding = [float(x) for x in embedding.strip("[]").split(",")]

        return cls(
            id            = row.get("id"),
            context_id    = row.get("context_id"),
            trade_id      = row.get("trade_id"),
            embedding     = embedding,
            outcome_r     = _opt_float(row.get("outcome_r")),
            win           = row.get("win"),
            cluster_label = row.get("cluster_label"),
            created_at    = row.get("created_at"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":            self.id,
            "context_id":    self.context_id,
            "trade_id":      self.trade_id,
            "embedding":     self.embedding,
            "outcome_r":     self.outcome_r,
            "win":           self.win,
            "cluster_label": self.cluster_label,
            "created_at":    self.created_at.isoformat() if self.created_at else None,
        }


# =============================================================================
# 5. Screenshot
# =============================================================================

@dataclass
class Screenshot:
    id:         Optional[int]
    file_path:  str
    trade_id:   Optional[int]          = None
    phase:      Optional[ScreenshotPhase] = None
    timeframe:  Optional[str]          = None
    created_at: Optional[datetime]     = None

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "Screenshot":
        return cls(
            id         = row.get("id"),
            trade_id   = row.get("trade_id"),
            phase      = _as_enum(ScreenshotPhase, row.get("phase")),
            file_path  = row["file_path"],
            timeframe  = row.get("timeframe"),
            created_at = row.get("created_at"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":         self.id,
            "trade_id":   self.trade_id,
            "phase":      self.phase.value if self.phase else None,
            "file_path":  self.file_path,
            "timeframe":  self.timeframe,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# =============================================================================
# 6. Zone
# =============================================================================

@dataclass
class Zone:
    id:          Optional[int]
    instrument:  str
    zone_type:   ZoneType
    price_high:  float
    price_low:   float
    timeframe:   str
    first_seen:  datetime
    strength:    float                = 0.0
    touch_count: int                  = 0
    status:      ZoneStatus           = ZoneStatus.ACTIVE
    last_tested: Optional[datetime]   = None
    created_at:  Optional[datetime]   = None

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "Zone":
        return cls(
            id          = row.get("id"),
            instrument  = row["instrument"],
            zone_type   = ZoneType(row["zone_type"]),
            price_high  = float(row["price_high"]),
            price_low   = float(row["price_low"]),
            timeframe   = row["timeframe"],
            strength    = float(row.get("strength") or 0.0),
            touch_count = int(row.get("touch_count") or 0),
            status      = ZoneStatus(row.get("status") or "active"),
            first_seen  = row["first_seen"],
            last_tested = row.get("last_tested"),
            created_at  = row.get("created_at"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":          self.id,
            "instrument":  self.instrument,
            "zone_type":   self.zone_type.value,
            "price_high":  self.price_high,
            "price_low":   self.price_low,
            "timeframe":   self.timeframe,
            "strength":    self.strength,
            "touch_count": self.touch_count,
            "status":      self.status.value,
            "first_seen":  self.first_seen.isoformat(),
            "last_tested": self.last_tested.isoformat() if self.last_tested else None,
            "created_at":  self.created_at.isoformat()  if self.created_at  else None,
        }


# =============================================================================
# 7. ZoneConfluence
# =============================================================================

@dataclass
class ZoneConfluence:
    id:               Optional[int]
    confluence_type:  str
    zone_id:          Optional[int]   = None
    value:            Optional[float] = None
    timeframe:        Optional[str]   = None
    created_at:       Optional[datetime] = None

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "ZoneConfluence":
        return cls(
            id               = row.get("id"),
            zone_id          = row.get("zone_id"),
            confluence_type  = row["confluence_type"],
            value            = _opt_float(row.get("value")),
            timeframe        = row.get("timeframe"),
            created_at       = row.get("created_at"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":              self.id,
            "zone_id":         self.zone_id,
            "confluence_type": self.confluence_type,
            "value":           self.value,
            "timeframe":       self.timeframe,
            "created_at":      self.created_at.isoformat() if self.created_at else None,
        }


# =============================================================================
# 8. Decision
# =============================================================================

@dataclass
class Decision:
    id:               Optional[int]
    timestamp:        datetime
    instrument:       str
    action:           DecisionAction
    trade_id:         Optional[int]         = None
    signal_data:      Optional[Dict]        = None
    edge_results:     Optional[Dict]        = None
    similarity_data:  Optional[Dict]        = None
    confluence_score: Optional[int]         = None
    reasoning:        Optional[str]         = None
    created_at:       Optional[datetime]    = None

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "Decision":
        def _json(val):
            if val is None:
                return None
            if isinstance(val, str):
                return json.loads(val)
            return val  # already a dict (psycopg2 with json adapter)

        return cls(
            id               = row.get("id"),
            timestamp        = row["timestamp"],
            instrument       = row["instrument"],
            action           = DecisionAction(row["action"]),
            trade_id         = row.get("trade_id"),
            signal_data      = _json(row.get("signal_data")),
            edge_results     = _json(row.get("edge_results")),
            similarity_data  = _json(row.get("similarity_data")),
            confluence_score = row.get("confluence_score"),
            reasoning        = row.get("reasoning"),
            created_at       = row.get("created_at"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":               self.id,
            "timestamp":        self.timestamp.isoformat(),
            "instrument":       self.instrument,
            "action":           self.action.value,
            "trade_id":         self.trade_id,
            "signal_data":      self.signal_data,
            "edge_results":     self.edge_results,
            "similarity_data":  self.similarity_data,
            "confluence_score": self.confluence_score,
            "reasoning":        self.reasoning,
            "created_at":       self.created_at.isoformat() if self.created_at else None,
        }


# =============================================================================
# Private helpers
# =============================================================================

def _opt_float(value) -> Optional[float]:
    """Cast to float, preserving None."""
    return float(value) if value is not None else None
