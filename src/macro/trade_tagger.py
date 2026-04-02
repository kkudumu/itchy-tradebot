"""Tag backtest trades with macro regime and event proximity metadata.

Enriches trade dicts with:
    - regime: daily macro regime label (risk_on, risk_off, etc.)
    - near_event: bool -- whether entry was within N hours of NFP/FOMC/CPI
    - nearest_event: str -- name of the nearest event (if near_event=True)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from src.macro.econ_calendar import EconCalendar
from src.macro.regime_classifier import RegimeClassifier, RegimeLabel

logger = logging.getLogger(__name__)


class TradeTagger:
    """Enrich trade dicts with macro regime and event proximity tags.

    Parameters
    ----------
    regime_series:
        Pre-computed regime series (from RegimeClassifier.classify()).
        If None, all trades get regime="mixed".
    """

    def __init__(
        self,
        regime_series: Optional[pd.Series] = None,
    ) -> None:
        self._regimes = regime_series
        self._calendar = EconCalendar()
        self._classifier = RegimeClassifier()

    def tag_trades(
        self,
        trades: List[Dict[str, Any]],
        hours_before: int = 4,
        hours_after: int = 2,
    ) -> List[Dict[str, Any]]:
        """Tag each trade with regime and event proximity.

        Parameters
        ----------
        trades:
            List of trade dicts. Each should have 'entry_time' (datetime).
        hours_before:
            Event blackout window before event (hours).
        hours_after:
            Event blackout window after event (hours).

        Returns
        -------
        List of trade dicts with added 'regime', 'near_event',
        'nearest_event' keys. Original dicts are not modified.
        """
        tagged: List[Dict[str, Any]] = []

        for trade in trades:
            enriched = dict(trade)  # shallow copy

            entry_time = trade.get("entry_time")
            if entry_time is None:
                enriched["regime"] = RegimeLabel.MIXED.value
                enriched["near_event"] = False
                enriched["nearest_event"] = ""
                tagged.append(enriched)
                continue

            # Ensure timezone-aware
            if isinstance(entry_time, datetime) and entry_time.tzinfo is None:
                entry_time = entry_time.replace(tzinfo=timezone.utc)

            # Regime tag
            if self._regimes is not None:
                regime = self._classifier.get_regime_for_date(
                    self._regimes, pd.Timestamp(entry_time)
                )
            else:
                regime = RegimeLabel.MIXED.value
            enriched["regime"] = regime

            # Event proximity tag
            is_near, event = self._calendar.is_near_event(
                entry_time,
                hours_before=hours_before,
                hours_after=hours_after,
            )
            enriched["near_event"] = is_near
            enriched["nearest_event"] = event.title if event else ""

            tagged.append(enriched)

        logger.info(
            "Tagged %d trades: %d near events, regimes: %s",
            len(tagged),
            sum(1 for t in tagged if t.get("near_event")),
            pd.Series([t.get("regime", "unknown") for t in tagged]).value_counts().to_dict(),
        )
        return tagged

    @staticmethod
    def compute_regime_stats(
        tagged_trades: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Compute per-regime trade statistics.

        Returns
        -------
        Dict mapping regime label to:
            count: number of trades
            win_rate: fraction of winning trades (r_multiple > 0)
            avg_r: average R-multiple
            near_event_count: trades near high-impact events
        """
        from collections import defaultdict

        buckets: Dict[str, List[float]] = defaultdict(list)
        event_counts: Dict[str, int] = defaultdict(int)

        for trade in tagged_trades:
            regime = trade.get("regime", "mixed")
            r = trade.get("r_multiple")
            if r is not None:
                buckets[regime].append(float(r))
            if trade.get("near_event"):
                event_counts[regime] += 1

        stats: Dict[str, Dict[str, Any]] = {}
        for regime, r_values in buckets.items():
            wins = sum(1 for r in r_values if r > 0)
            stats[regime] = {
                "count": len(r_values),
                "win_rate": round(wins / len(r_values), 4) if r_values else 0.0,
                "avg_r": round(sum(r_values) / len(r_values), 4) if r_values else 0.0,
                "near_event_count": event_counts.get(regime, 0),
            }

        return stats
