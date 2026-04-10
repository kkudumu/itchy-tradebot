"""Per-strategy performance buckets derived from trade memory.

The mega-vision agent calls ``get_buckets(strategy_name=...)`` at
decision time to see how each candidate strategy has been performing
in similar conditions. The output is a nested dict keyed by
(strategy → session → pattern → regime) with trade_count, win_rate,
avg_r, expectancy, and a simple max drawdown placeholder.

Results are cached with a 60-second TTL during live runs so the
agent's rapid-fire queries don't re-scan the whole trade memory on
every call. In backtest mode the cache is bypassed (no TTL clock).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .trade_memory import TradeMemory


@dataclass
class PerformanceBuckets:
    """Wraps a :class:`TradeMemory` with a small cache layer."""

    trade_memory: TradeMemory
    cache_ttl_seconds: float = 60.0
    _cache: Optional[Dict[str, Any]] = None
    _cache_ts: float = 0.0

    def get_buckets(
        self,
        strategy_name: str | None = None,
        lookback_days: int = 90,
    ) -> Dict[str, Any]:
        """Return per-strategy performance buckets.

        When *strategy_name* is set, only returns that one strategy's
        buckets.
        """
        now = time.monotonic()
        if (
            self._cache is not None
            and self._cache_ts
            and now - self._cache_ts < self.cache_ttl_seconds
        ):
            buckets = self._cache
        else:
            buckets = self._compute_buckets(lookback_days)
            self._cache = buckets
            self._cache_ts = now

        if strategy_name:
            return {strategy_name: buckets.get(strategy_name, {})}
        return buckets

    def invalidate(self) -> None:
        self._cache = None
        self._cache_ts = 0.0

    # ------------------------------------------------------------------

    def _compute_buckets(self, lookback_days: int) -> Dict[str, Any]:
        # Pull all trades once; downstream aggregation is cheap
        all_trades = self.trade_memory.query({}, limit=10_000)
        out: Dict[str, Any] = {}
        if not all_trades:
            return out

        for trade in all_trades:
            strat = trade.get("strategy_name") or "unknown"
            session = trade.get("session") or "off"
            pattern = trade.get("pattern_type") or "none"
            regime = trade.get("regime") or "none"
            r = trade.get("r_multiple")
            pnl = trade.get("pnl_usd")

            bucket = (
                out.setdefault(strat, {})
                .setdefault(session, {})
                .setdefault(pattern, {})
                .setdefault(
                    regime,
                    {
                        "trade_count": 0,
                        "win_count": 0,
                        "total_pnl": 0.0,
                        "sum_r": 0.0,
                        "min_pnl": 0.0,
                    },
                )
            )
            bucket["trade_count"] += 1
            if r is not None and r > 0:
                bucket["win_count"] += 1
            if pnl is not None:
                bucket["total_pnl"] += float(pnl)
                if float(pnl) < bucket["min_pnl"]:
                    bucket["min_pnl"] = float(pnl)
            if r is not None:
                bucket["sum_r"] += float(r)

        # Compute derived fields
        for strat in out:
            for session in out[strat]:
                for pattern in out[strat][session]:
                    for regime in out[strat][session][pattern]:
                        b = out[strat][session][pattern][regime]
                        n = b["trade_count"]
                        b["win_rate"] = b["win_count"] / n if n else 0.0
                        b["avg_r"] = b["sum_r"] / n if n else 0.0
                        b["expectancy"] = b["avg_r"]
                        b["max_drawdown_usd"] = -b["min_pnl"] if b["min_pnl"] < 0 else 0.0
                        del b["sum_r"]
                        del b["min_pnl"]

        return out
