"""Pattern-based confluence scoring for the edge filter pipeline.

Detected chart patterns add or subtract confluence points based on:
1. Pattern direction vs trade direction (aligned = bonus, conflicting = penalty)
2. Pattern confidence (high confidence = stronger adjustment)
3. Pattern type importance (H&S > double top > triangle > wedge)

Integrates with the existing EdgeFilter ABC via PatternConfluenceEdge.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.edges.base import EdgeContext, EdgeFilter, EdgeResult

logger = logging.getLogger(__name__)

# Pattern type weights: how strongly each pattern type affects confluence
_PATTERN_WEIGHTS: Dict[str, float] = {
    "head_and_shoulders": 2.0,
    "inverse_head_and_shoulders": 2.0,
    "double_top": 1.5,
    "double_bottom": 1.5,
    "ascending_triangle": 1.2,
    "descending_triangle": 1.2,
    "symmetrical_triangle": 0.8,
    "rising_wedge": 1.3,
    "falling_wedge": 1.3,
}

# Direction mapping: what direction each pattern implies
_PATTERN_DIRECTION: Dict[str, str] = {
    "double_top": "bearish",
    "double_bottom": "bullish",
    "head_and_shoulders": "bearish",
    "inverse_head_and_shoulders": "bullish",
    "ascending_triangle": "bullish",
    "descending_triangle": "bearish",
    "symmetrical_triangle": "neutral",
    "rising_wedge": "bearish",
    "falling_wedge": "bullish",
}


class PatternConfluenceScorer:
    """Scores confluence adjustment from detected chart patterns.

    Parameters
    ----------
    max_adjustment:
        Maximum absolute confluence adjustment. Default 3 (on a 0-8 scale).
    confidence_threshold:
        Minimum confidence for a pattern to contribute. Default 0.4.
    """

    def __init__(
        self,
        max_adjustment: int = 3,
        confidence_threshold: float = 0.4,
    ) -> None:
        self._max_adj = max_adjustment
        self._conf_threshold = confidence_threshold

    def score(
        self,
        patterns: List,
        trade_direction: str,
    ) -> int:
        """Calculate confluence adjustment from detected patterns.

        Parameters
        ----------
        patterns: List of ChartPattern objects or dicts with to_dict() output.
        trade_direction: 'long' or 'short'.

        Returns
        -------
        Integer confluence adjustment (can be negative for conflicts).
        """
        if not patterns:
            return 0

        total_score = 0.0
        trade_bias = "bullish" if trade_direction == "long" else "bearish"

        for p in patterns:
            # Support both ChartPattern objects and dicts
            if hasattr(p, "to_dict"):
                d = p.to_dict()
            elif isinstance(p, dict):
                d = p
            else:
                continue

            conf = float(d.get("confidence", 0))
            if conf < self._conf_threshold:
                continue

            pattern_type = d.get("pattern_type", "")
            pattern_dir = d.get("direction", _PATTERN_DIRECTION.get(pattern_type, "neutral"))
            weight = _PATTERN_WEIGHTS.get(pattern_type, 1.0)

            # Aligned: pattern direction matches trade direction
            if pattern_dir == trade_bias:
                total_score += weight * conf
            elif pattern_dir == "neutral":
                total_score += 0.2 * conf  # neutral patterns give a tiny boost
            else:
                # Conflicting: pattern opposes trade direction
                total_score -= weight * conf

        # Round and clamp
        adjustment = int(round(total_score))
        return max(-self._max_adj, min(self._max_adj, adjustment))


class PatternConfluenceEdge(EdgeFilter):
    """EdgeFilter that adjusts or blocks trades based on chart patterns.

    Reads active patterns from EdgeContext.indicator_values['_active_patterns']
    (list of pattern dicts) and the intended trade direction from
    EdgeContext.indicator_values['_trade_direction'].

    Config keys:
    - enabled: bool (default True)
    - block_on_conflict: bool -- if True, block entry when patterns strongly
      oppose the trade direction (default False)
    - min_pattern_score: int -- minimum net pattern score to allow entry (default -2)
    """

    def __init__(self, name: str, config: dict) -> None:
        super().__init__(name, config)
        self._scorer = PatternConfluenceScorer(
            max_adjustment=config.get("max_adjustment", 3),
            confidence_threshold=config.get("confidence_threshold", 0.4),
        )
        self._block_on_conflict = config.get("block_on_conflict", False)
        self._min_score = config.get("min_pattern_score", -2)

    def should_allow(self, context: EdgeContext) -> EdgeResult:
        """Evaluate chart pattern confluence for the proposed trade."""
        if not self.enabled:
            return self._disabled_result()

        iv = context.indicator_values or {}
        patterns_raw = iv.get("_active_patterns", [])
        trade_dir = iv.get("_trade_direction", "long")

        if not patterns_raw:
            return EdgeResult(
                allowed=True,
                edge_name=self.name,
                reason="No active chart patterns -- allowing",
            )

        # Import here to avoid circular dependency
        from src.discovery.chart_patterns import ChartPattern

        # Convert dicts back to ChartPattern if needed
        patterns = []
        for p in patterns_raw:
            if isinstance(p, dict):
                patterns.append(p)
            elif hasattr(p, "to_dict"):
                patterns.append(p.to_dict())
            else:
                patterns.append(p)

        score = self._scorer.score(patterns, trade_direction=trade_dir)

        if self._block_on_conflict and score < self._min_score:
            # Find the strongest conflicting pattern for the reason
            trade_bias = "bullish" if trade_dir == "long" else "bearish"
            conflicting = [
                p for p in patterns
                if p.get("direction", "") != trade_bias and p.get("direction", "") != "neutral"
            ]
            conflict_desc = conflicting[0].get("pattern_type", "unknown") if conflicting else "unknown"
            return EdgeResult(
                allowed=False,
                edge_name=self.name,
                reason=(
                    f"Conflicting chart pattern: {conflict_desc} "
                    f"({score:+d} score, trade={trade_dir})"
                ),
            )

        return EdgeResult(
            allowed=True,
            edge_name=self.name,
            reason=f"Pattern confluence: {score:+d} (patterns: {len(patterns_raw)})",
            modifier=float(score),
        )
