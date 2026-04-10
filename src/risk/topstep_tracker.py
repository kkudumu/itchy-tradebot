"""TopstepX $50K Combine prop firm tracker.

Implements the dollar-based Combine rules:

* **Trailing maximum loss limit (MLL)**: starts at ``initial_balance -
  max_loss_limit_usd_trailing``. Trails each end-of-day balance upward.
  Once the account first reaches ``initial_balance + max_loss_limit_usd_trailing``
  the MLL locks at ``initial_balance`` and never moves again.
* **Daily loss limit**: absolute dollar loss against the day's opening
  balance that triggers failure.
* **Consistency rule**: once the profit target is reached, the single
  best trading day cannot represent more than ``consistency_pct`` of
  total profit.
* **Day boundaries**: rollover at 5pm ``America/Chicago`` local time.
  DST is handled via ``zoneinfo`` / ``pytz``.

See ``tests/test_topstep_tracker.py`` for the full spec in test form.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Literal

try:
    # Prefer stdlib zoneinfo (Python 3.9+); falls back to pytz when missing
    # (older environments or Windows without tzdata).
    from zoneinfo import ZoneInfo

    def _localize(ts: datetime, tz_name: str) -> datetime:
        return ts.astimezone(ZoneInfo(tz_name))

except ImportError:  # pragma: no cover - exercised on legacy py3.8
    import pytz  # type: ignore[import-not-found]

    def _localize(ts: datetime, tz_name: str) -> datetime:
        tz = pytz.timezone(tz_name)
        if ts.tzinfo is None:
            return tz.localize(ts)
        return ts.astimezone(tz)


from src.config.models import TopstepCombineConfig


TrackerStatus = Literal[
    "pending",
    "passed",
    "failed_mll",
    "failed_daily_loss",
    "failed_consistency",
]


@dataclass
class TopstepCombineTracker:
    """Dollar-based trailing MLL + daily loss + consistency tracker.

    Mirrors the ``PropFirmTrackerProtocol`` (initialise / update /
    check_pass / to_dict) so the engine can swap it in for the legacy
    percentage-based trackers without code changes elsewhere.
    """

    config: TopstepCombineConfig

    # Mutable state
    initial_balance: float = 0.0
    current_balance: float = 0.0
    mll: float = 0.0
    mll_locked: bool = False
    daily_open_balance: float = 0.0
    daily_pnl: float = 0.0
    current_trading_day: date | None = None
    total_profit: float = 0.0
    best_day_profit: float = 0.0
    eod_balances: list[tuple[date, float]] = field(default_factory=list)
    status: TrackerStatus = "pending"
    failure_reason: str | None = None
    # Derived diagnostic state
    peak_balance: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialise(self, initial_balance: float, first_ts: datetime) -> None:
        """Reset and begin tracking from *first_ts* at *initial_balance*.

        The tracker starts in ``pending`` status with MLL at
        ``initial_balance - max_loss_limit_usd_trailing`` and the first
        trading day set from *first_ts* via the 5pm CT rollover rule.
        """
        self.initial_balance = float(initial_balance)
        self.current_balance = float(initial_balance)
        self.peak_balance = float(initial_balance)
        self.mll = float(initial_balance) - float(self.config.max_loss_limit_usd_trailing)
        self.mll_locked = False
        self.current_trading_day = self._trading_day_for(first_ts)
        self.daily_open_balance = float(initial_balance)
        self.daily_pnl = 0.0
        self.total_profit = 0.0
        self.best_day_profit = 0.0
        self.eod_balances = []
        self.status = "pending"
        self.failure_reason = None

    # ------------------------------------------------------------------
    # Day boundary math
    # ------------------------------------------------------------------

    def _trading_day_for(self, ts: datetime) -> date:
        """Return the TopstepX trading day for a UTC timestamp.

        TopstepX rolls the trading day over at ``daily_reset_hour`` local
        time in ``daily_reset_tz`` (default 5pm America/Chicago). A
        timestamp at 4:59pm CT belongs to that calendar day; a timestamp
        at 5:00pm CT belongs to the NEXT trading day. The implementation
        subtracts ``daily_reset_hour`` hours from the localized time so a
        ``date()`` call returns the right value on either side of the
        boundary, even across DST transitions.
        """
        if ts.tzinfo is None:
            # Treat naive timestamps as UTC — matches the rest of the codebase
            from datetime import timezone as _tz

            ts = ts.replace(tzinfo=_tz.utc)
        local = _localize(ts, self.config.daily_reset_tz)
        shifted = local - timedelta(hours=self.config.daily_reset_hour)
        return shifted.date()

    # ------------------------------------------------------------------
    # Update loop
    # ------------------------------------------------------------------

    def update(self, ts: datetime, balance: float) -> None:
        """Feed a new timestamp + balance to the tracker.

        Called on every bar (or at least frequently enough that day
        rollovers and intraday MLL/daily-loss breaches are caught
        within a single bar of the actual event).
        """
        if self.status != "pending":
            # Already passed/failed — further updates are no-ops so the
            # engine can call update unconditionally without crashing.
            return

        trading_day = self._trading_day_for(ts)
        if self.current_trading_day is None:
            self.current_trading_day = trading_day
            self.daily_open_balance = float(balance)
        elif trading_day != self.current_trading_day:
            # Close out the previous day — this trails MLL and resets
            # the daily accumulator for the new day.
            self._on_eod(self.current_balance)
            self.current_trading_day = trading_day
            self.daily_open_balance = float(balance)
            self.daily_pnl = 0.0

        self.current_balance = float(balance)
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        self.daily_pnl = self.current_balance - self.daily_open_balance

        # Check daily loss limit (absolute dollars)
        if self.daily_pnl <= -float(self.config.daily_loss_limit_usd):
            self.status = "failed_daily_loss"
            self.failure_reason = (
                f"Daily loss limit breached: {self.daily_pnl:+.2f} USD "
                f"(limit {-self.config.daily_loss_limit_usd:.2f} USD)"
            )
            return

        # Check trailing MLL (intraday touch fails the account)
        if self.current_balance <= self.mll:
            self.status = "failed_mll"
            self.failure_reason = (
                f"Maximum loss limit breached: balance {self.current_balance:.2f} USD "
                f"at or below MLL {self.mll:.2f} USD"
            )
            return

        # Check profit target (passing still needs consistency check)
        if self.current_balance >= self.initial_balance + float(self.config.profit_target_usd):
            self.status = "passed"
            self.failure_reason = None
            # Don't early-return — let check_pass() enforce consistency

    # ------------------------------------------------------------------
    # End-of-day handler
    # ------------------------------------------------------------------

    def _on_eod(self, eod_balance: float) -> None:
        """Close out a trading day.

        Appends the EOD balance to ``eod_balances``, updates total/best
        profit accumulators, trails the MLL upward, and locks the MLL at
        the initial balance once the profit target buffer is first
        reached.
        """
        if self.current_trading_day is not None:
            self.eod_balances.append((self.current_trading_day, float(eod_balance)))

        # Track best day by day's P&L (not EOD balance vs start)
        day_pnl = float(eod_balance) - float(self.daily_open_balance)
        if day_pnl > self.best_day_profit:
            self.best_day_profit = day_pnl

        # Track cumulative profit (clamped at 0 — losing days don't
        # reduce "total profit" for consistency math)
        cumulative = float(eod_balance) - float(self.initial_balance)
        if cumulative > self.total_profit:
            self.total_profit = cumulative

        # Trail the MLL upward on winning days. If the new candidate
        # would be lower than the current MLL (down day), leave it alone.
        if not self.mll_locked:
            new_mll_candidate = float(eod_balance) - float(
                self.config.max_loss_limit_usd_trailing
            )
            if new_mll_candidate > self.mll:
                self.mll = new_mll_candidate

            # Lock check: once the account has ever touched
            # initial_balance + max_loss_limit_usd_trailing at EOD, the
            # MLL locks at initial_balance permanently.
            if float(eod_balance) >= float(self.initial_balance) + float(
                self.config.max_loss_limit_usd_trailing
            ):
                self.mll = float(self.initial_balance)
                self.mll_locked = True

    def on_eod(self, eod_balance: float) -> None:
        """Public wrapper for manual EOD rollover (used by simulator)."""
        self._on_eod(eod_balance)

    # ------------------------------------------------------------------
    # Terminal checks
    # ------------------------------------------------------------------

    def check_pass(self) -> dict[str, Any]:
        """Return the current tracker state with the final pass/fail verdict.

        When the tracker is already in a ``failed_*`` state, returns the
        failure verbatim. When ``pending``, returns ``pending``. When
        ``passed``, applies the consistency check: the best day's profit
        must not exceed ``consistency_pct`` of total profit. Failing the
        consistency check flips the status to ``failed_consistency``.
        """
        if self.status == "passed":
            # Consistency rule is evaluated at pass time — a run that
            # crosses the profit target but violates consistency is a
            # failure, not a pass.
            if self.total_profit > 0 and self.best_day_profit / self.total_profit > (
                self.config.consistency_pct / 100.0
            ):
                self.status = "failed_consistency"
                self.failure_reason = (
                    f"Consistency rule breached: best day "
                    f"{self.best_day_profit:.2f} USD is "
                    f"{(self.best_day_profit / self.total_profit) * 100:.1f}% of "
                    f"total profit {self.total_profit:.2f} USD "
                    f"(limit {self.config.consistency_pct:.1f}%)"
                )

        return self._snapshot()

    def _snapshot(self) -> dict[str, Any]:
        distance_to_mll = self.current_balance - self.mll
        distance_to_target = (
            self.initial_balance + float(self.config.profit_target_usd)
        ) - self.current_balance
        return {
            "style": "topstep_combine_dollar",
            "status": self.status,
            "failure_reason": self.failure_reason,
            "initial_balance": self.initial_balance,
            "current_balance": self.current_balance,
            "peak_balance": self.peak_balance,
            "mll": self.mll,
            "mll_locked": self.mll_locked,
            "daily_pnl": self.daily_pnl,
            "daily_open_balance": self.daily_open_balance,
            "total_profit": self.total_profit,
            "best_day_profit": self.best_day_profit,
            "distance_to_mll": distance_to_mll,
            "distance_to_target": distance_to_target,
            "current_trading_day": (
                self.current_trading_day.isoformat()
                if self.current_trading_day
                else None
            ),
            "eod_balance_count": len(self.eod_balances),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a dict snapshot suitable for serialization to JSON/Parquet."""
        return self._snapshot()
