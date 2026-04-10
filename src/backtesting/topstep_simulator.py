"""TopstepX Combine simulator — single-account, dollar-based replay.

Unlike the5ers which uses rolling windows + Monte Carlo across a 2-step
pct-based challenge, a TopstepX Combine is a single continuous
account that runs until it passes or fails. This module replays a list
of trades through a :class:`TopstepCombineTracker` and returns a
verdict dataclass.

It's wired into the engine's post-backtest pipeline via
:class:`ChallengeSimulator` — when ``prop_firm.style ==
"topstep_combine_dollar"``, the simulator delegates here instead of
running the rolling-window + MC logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from src.config.models import TopstepCombineConfig
from src.risk.topstep_tracker import TopstepCombineTracker


@dataclass
class TopstepCombineResult:
    """Verdict from a single TopstepX Combine replay."""

    passed: bool
    failure_reason: Optional[str]
    final_balance: float
    peak_balance: float
    mll_at_failure: float
    days_traded: int
    total_trades: int
    consistency_check_passed: bool
    total_profit: float
    best_day_profit: float
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "style": "topstep_combine_dollar",
            "passed": self.passed,
            "failure_reason": self.failure_reason,
            "final_balance": self.final_balance,
            "peak_balance": self.peak_balance,
            "mll_at_failure": self.mll_at_failure,
            "days_traded": self.days_traded,
            "total_trades": self.total_trades,
            "consistency_check_passed": self.consistency_check_passed,
            "total_profit": self.total_profit,
            "best_day_profit": self.best_day_profit,
            **self.extra,
        }


class TopstepCombineSimulator:
    """Replays a trade list through a :class:`TopstepCombineTracker`.

    The simulator is stateless — ``run()`` builds a fresh tracker
    instance per call so callers can invoke it repeatedly without
    worrying about cross-run pollution.
    """

    def __init__(self, config: TopstepCombineConfig) -> None:
        self._config = config

    def run(
        self,
        trades: List[Dict[str, Any]],
        start_balance: Optional[float] = None,
        start_ts: Optional[datetime] = None,
    ) -> TopstepCombineResult:
        """Replay *trades* through a fresh TopstepCombineTracker.

        Each trade dict must have at minimum:
          * ``pnl_usd`` (or ``pnl``): realised dollar P&L
          * ``entry_time`` or ``timestamp``: UTC datetime for day binning

        Unknown fields are ignored. Trades without a timestamp are
        assigned to *start_ts* (or wall clock when not given).
        """
        tracker = TopstepCombineTracker(config=self._config)
        initial = float(start_balance or self._config.account_size)
        first_ts = start_ts or (
            _infer_first_ts(trades) or datetime.now(tz=timezone.utc)
        )
        tracker.initialise(initial, first_ts)

        balance = initial
        trades_applied = 0
        for trade in trades:
            pnl = float(
                trade.get("pnl_usd")
                or trade.get("pnl")
                or 0.0
            )
            ts = _coerce_ts(trade.get("entry_time") or trade.get("timestamp") or first_ts)
            # Fire the tracker BEFORE applying P&L so the day-rollover
            # handler closes the prior day with the pre-trade balance.
            # Then apply the P&L and fire update again so the tracker
            # sees the intraday change as a daily P&L, not a next-day
            # opening balance.
            tracker.update(ts, balance)
            if tracker.status != "pending":
                break
            balance += pnl
            tracker.update(ts, balance)
            trades_applied += 1
            if tracker.status != "pending":
                break

        # Roll over any remaining session so the consistency check has
        # full visibility of the final day.
        tracker._on_eod(tracker.current_balance)
        verdict = tracker.check_pass()

        days = len(tracker.eod_balances)
        mll_at_fail = tracker.mll if str(verdict["status"]).startswith("failed") else 0.0

        return TopstepCombineResult(
            passed=verdict["status"] == "passed",
            failure_reason=verdict["failure_reason"],
            final_balance=tracker.current_balance,
            peak_balance=tracker.peak_balance,
            mll_at_failure=mll_at_fail,
            days_traded=days,
            total_trades=trades_applied,
            consistency_check_passed=verdict["status"] != "failed_consistency",
            total_profit=tracker.total_profit,
            best_day_profit=tracker.best_day_profit,
            extra={"verdict": verdict},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _infer_first_ts(trades: List[Dict[str, Any]]) -> Optional[datetime]:
    for t in trades:
        raw = t.get("entry_time") or t.get("timestamp")
        if raw is not None:
            try:
                return _coerce_ts(raw)
            except (TypeError, ValueError):
                continue
    return None


def _coerce_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    # String / pandas Timestamp → datetime
    import pandas as pd

    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.to_pydatetime()
