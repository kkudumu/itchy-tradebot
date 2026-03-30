"""Multi-strategy signal coordinator.

Receives signals from multiple strategies, applies multi-agree bonuses,
and selects the highest-confluence signal for execution.
"""
from __future__ import annotations

from collections import Counter
from typing import List, Optional

from src.strategy.signal_engine import Signal


class SignalBlender:
    """Select the best signal from multiple strategy candidates.

    Parameters
    ----------
    multi_agree_bonus:
        Score bonus added when 2+ strategies agree on direction. Default: 2.
    """

    def __init__(self, multi_agree_bonus: int = 2):
        self._bonus = multi_agree_bonus

    def select(self, signals: List[Signal]) -> Optional[Signal]:
        if not signals:
            return None

        direction_counts = Counter(s.direction for s in signals)

        scored = []
        for sig in signals:
            effective_score = sig.confluence_score
            if direction_counts[sig.direction] >= 2:
                effective_score += self._bonus
            scored.append((effective_score, sig))

        # Sort descending by score; stable sort preserves input order on ties
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]
