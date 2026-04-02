"""Chart pattern detection from swing points.

Detects classical chart patterns (double top/bottom, head & shoulders,
triangles, wedges) from a sequence of SwingPoints produced by the
existing BreathingRoomDetector. Strategy-agnostic -- works with any
OHLCV data that produces swing points.

Pattern detection uses geometric relationships between consecutive swing
highs and lows, with tolerance thresholds scaled to ATR for volatility
adaptation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ChartPattern:
    """A detected chart pattern with metadata.

    Attributes
    ----------
    pattern_type:
        One of: double_top, double_bottom, head_and_shoulders,
        inverse_head_and_shoulders, ascending_triangle,
        descending_triangle, symmetrical_triangle,
        rising_wedge, falling_wedge.
    direction:
        'bullish' or 'bearish' -- the implied directional bias.
    confidence:
        0.0 to 1.0 -- how well the swing points match the ideal pattern.
    key_prices:
        List of significant price levels (e.g. peaks, neckline).
    start_index:
        Bar index where the pattern begins.
    end_index:
        Bar index where the pattern completes.
    description:
        Human-readable explanation.
    swing_indices:
        Bar indices of the swing points forming this pattern.
    """

    pattern_type: str
    direction: str
    confidence: float
    key_prices: List[float]
    start_index: int
    end_index: int
    description: str
    swing_indices: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "pattern_type": self.pattern_type,
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "key_prices": [round(p, 2) for p in self.key_prices],
            "start_index": self.start_index,
            "end_index": self.end_index,
            "description": self.description,
            "swing_indices": self.swing_indices,
        }
