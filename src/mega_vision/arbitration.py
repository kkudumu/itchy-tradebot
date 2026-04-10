"""Arbitrator — combines agent picks with native blender output.

Shadow mode: always returns the native signal set unchanged. The
agent's pick is recorded separately but NEVER affects execution.

Authority mode: filters native signals to those whose strategy_name
is in the agent's strategy_picks list. If the agent's pick is a
fallback (agent errored / no valid pick in 3 retries), returns the
full native set.

Either mode always emits telemetry events so offline analysis can
see what was executed, what the agent picked, and where they
diverged.
"""

from __future__ import annotations

import logging
from typing import Any, List, Literal

logger = logging.getLogger(__name__)


Mode = Literal["disabled", "shadow", "authority"]


class Arbitrator:
    """Decides what the trade manager actually sees at each decision."""

    def __init__(
        self,
        mode: Mode = "disabled",
        telemetry: Any | None = None,
    ) -> None:
        self._mode = mode
        self._telemetry = telemetry

    @property
    def mode(self) -> Mode:
        return self._mode

    def arbitrate(
        self,
        agent_pick: dict | None,
        native_signals: List[Any],
    ) -> List[Any]:
        """Return the signal set to forward to the trade manager.

        *agent_pick* is the dict returned by the agent's
        ``record_strategy_pick`` call (or a fallback when the agent
        failed). When it's ``None`` or has ``fallback=True``, the
        full native signal list is returned regardless of mode.

        In shadow mode, the native list is returned verbatim. In
        authority mode, the native list is filtered to signals whose
        ``strategy_name`` is in the agent's picks.
        """
        if self._mode == "disabled":
            return list(native_signals)

        if agent_pick is None or agent_pick.get("fallback"):
            self._emit(
                "fallback",
                {"reason": agent_pick.get("reasoning") if agent_pick else "agent_none"},
            )
            return list(native_signals)

        if self._mode == "shadow":
            # Shadow NEVER changes execution — native set returned as-is.
            self._emit("shadow", {"agent_picks": agent_pick.get("strategy_picks")})
            return list(native_signals)

        # Authority: filter by the agent's picks
        picks = set(agent_pick.get("strategy_picks") or [])
        filtered = [
            s
            for s in native_signals
            if getattr(s, "strategy_name", None) in picks
        ]
        self._emit(
            "authority",
            {
                "agent_picks": list(picks),
                "native_count": len(native_signals),
                "filtered_count": len(filtered),
            },
        )
        return filtered

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _emit(self, kind: str, extra: dict) -> None:
        if self._telemetry is None:
            return
        try:
            # Re-use the telemetry collector's emit interface via
            # extra-field passthrough so we don't need a new event type.
            from datetime import datetime, timezone

            ts = datetime.now(timezone.utc)
            self._telemetry.emit_filter_rejection(
                ts,
                "mega_vision",
                filter_stage=f"arbitration.{kind}",
                rejection_reason=str(extra),
                event_type="signal_filtered",
            )
        except Exception:
            pass
