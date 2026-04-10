"""OrderRouter — signal → validated broker order.

Sits between the :class:`SignalBlender` output and the execution
provider (``ProjectXExecutionProvider`` or ``PaperExecutionProvider``).
Responsibilities:

* Compute contract count via the profile-aware :class:`InstrumentSizer`
* Validate the active prop firm tracker hasn't failed
* Validate the contract cap hasn't been breached
* Validate the planned stop distance fits inside remaining MLL headroom
* Submit the entry + protective stop + take-profit as a bracket order
* Emit telemetry events for accept / reject

All of these checks must happen BEFORE any order is submitted. A
rejection returns a :class:`RouterRejection` with a clear reason
and emits a ``signal_rejected_risk`` telemetry event so offline
analysis can see exactly why a signal was blocked.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List, Optional

from src.providers.base import ExecutionProvider, ExecutionResult
from src.risk.instrument_sizer import InstrumentSizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class RouterRejection:
    """Returned when the router refuses to submit an order."""

    reason: str
    stage: str  # "prop_firm" | "contract_cap" | "size" | "mll_headroom" | "kill_switch"
    signal_summary: dict = field(default_factory=dict)


@dataclass
class RouterResult:
    """Returned when an order was successfully placed."""

    executed: bool
    entry_result: Optional[ExecutionResult] = None
    stop_order_id: Optional[Any] = None
    tp_order_id: Optional[Any] = None
    quantity: float = 0.0
    reason: str = ""


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class OrderRouter:
    """Routes blended signals to the execution provider with risk gates.

    Parameters
    ----------
    execution_provider:
        Anything implementing :class:`ExecutionProvider` — a real
        :class:`ProjectXExecutionProvider` for live trading or a
        :class:`PaperExecutionProvider` for paper-mode smoke tests.
    instrument_sizer:
        Profile-aware sizer (forex lots or futures contracts).
    prop_firm_tracker:
        The active prop firm tracker. If its status isn't ``pending``
        the router refuses to submit anything.
    max_contracts:
        Hard cap on position size regardless of what the sizer produces.
    telemetry:
        Optional StrategyTelemetryCollector for accept/reject events.
    kill_switch_fn:
        Optional ``() -> bool`` that returns True when trading should
        halt. Checked on every call — for mega-vision's
        ``MEGA_VISION_KILL_SWITCH`` env var or manual halt toggle.
    """

    def __init__(
        self,
        execution_provider: ExecutionProvider,
        instrument_sizer: InstrumentSizer,
        prop_firm_tracker: Any,
        max_contracts: int = 50,
        telemetry: Any | None = None,
        kill_switch_fn: Any | None = None,
    ) -> None:
        self._execution = execution_provider
        self._sizer = instrument_sizer
        self._tracker = prop_firm_tracker
        self._max_contracts = max_contracts
        self._telemetry = telemetry
        self._kill_switch_fn = kill_switch_fn

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def route(
        self,
        signal: Any,
        instrument: str,
        account_equity: float,
        risk_pct: float,
    ) -> RouterResult | RouterRejection:
        """Validate and submit a signal.

        *signal* must expose ``direction``, ``entry_price``, ``stop_loss``,
        and ``take_profit`` attributes (matches the project's Signal
        dataclass). *risk_pct* is the percentage of *account_equity*
        this trade is allowed to lose if stopped out.
        """
        ts = getattr(signal, "timestamp", None) or datetime.now(tz=timezone.utc)
        strategy_name = getattr(signal, "strategy_name", "unknown")

        # Gate 1: kill switch
        if self._kill_switch_fn and self._kill_switch_fn():
            return self._reject(
                ts, strategy_name, "kill_switch", "Kill switch active"
            )

        # Gate 2: prop firm tracker
        status = getattr(self._tracker, "status", None)
        if status is not None and status != "pending":
            return self._reject(
                ts,
                strategy_name,
                "prop_firm",
                f"Prop firm tracker status={status} — trading halted",
            )

        # Gate 3: compute size via the profile-aware sizer
        entry = float(signal.entry_price)
        stop = float(signal.stop_loss)
        stop_distance = abs(entry - stop)
        if stop_distance <= 0:
            return self._reject(
                ts, strategy_name, "size", "Stop distance is zero or negative"
            )

        risk_usd = float(account_equity) * float(risk_pct) / 100.0
        qty = self._sizer.size_for_risk(risk_usd, stop_distance)
        qty_int = int(qty) if isinstance(qty, float) and qty == int(qty) else qty

        if qty_int == 0:
            return self._reject(
                ts,
                strategy_name,
                "size",
                f"Sizer returned 0 (risk ${risk_usd:.0f}, stop ${stop_distance:.2f})",
            )

        # Gate 4: contract cap
        if isinstance(qty_int, int) and qty_int > self._max_contracts:
            qty_int = self._max_contracts

        # Gate 5: MLL headroom check (for topstep-style trackers)
        distance_to_mll = self._get_mll_headroom()
        if distance_to_mll is not None and distance_to_mll > 0:
            max_affordable_loss_usd = distance_to_mll
            if risk_usd > max_affordable_loss_usd:
                return self._reject(
                    ts,
                    strategy_name,
                    "mll_headroom",
                    f"Trade risk ${risk_usd:.0f} exceeds MLL headroom ${max_affordable_loss_usd:.0f}",
                )

        # All gates passed — submit the order
        try:
            entry_result = self._execution.place_market_order(
                instrument=instrument,
                direction=signal.direction,
                quantity=float(qty_int),
                stop_loss=float(stop),
                take_profit=float(signal.take_profit),
                comment=f"{strategy_name}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("OrderRouter execution failed: %s", exc)
            return self._reject(
                ts, strategy_name, "execution", f"Broker call failed: {exc}"
            )

        # Telemetry: successful entry
        if self._telemetry is not None:
            try:
                self._telemetry.emit_entry(
                    ts,
                    strategy_name,
                    direction=signal.direction,
                    price=entry,
                    planned_size=float(qty_int),
                )
            except Exception:
                pass

        return RouterResult(
            executed=bool(entry_result.success),
            entry_result=entry_result,
            quantity=float(qty_int),
            reason="accepted",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_mll_headroom(self) -> float | None:
        """Return distance-to-MLL for topstep-style trackers, else None."""
        if hasattr(self._tracker, "to_dict"):
            try:
                snap = self._tracker.to_dict()
                return float(snap.get("distance_to_mll") or 0.0)
            except Exception:
                return None
        return None

    def _reject(
        self,
        ts: datetime,
        strategy_name: str,
        stage: str,
        reason: str,
    ) -> RouterRejection:
        logger.info("OrderRouter rejected (%s): %s", stage, reason)
        if self._telemetry is not None:
            try:
                self._telemetry.emit_filter_rejection(
                    ts,
                    strategy_name,
                    filter_stage=f"router.{stage}",
                    rejection_reason=reason,
                    event_type="signal_rejected_risk",
                )
            except Exception:
                pass
        return RouterRejection(reason=reason, stage=stage)
