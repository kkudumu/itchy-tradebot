"""Pre-tool-use safety gates for the mega-vision agent.

When the agent calls ``record_strategy_pick``, the Agent SDK's
``PreToolUse`` hook fires ``validate_pick()`` BEFORE the tool
executes. If any gate fails, the hook returns a deny payload and
the agent retries within bounds. After 3 retries the system falls
back to the native blender silently.

The gates are a safety floor the agent cannot bypass even in
``bypassPermissions`` mode:

  1. kill switch env var (MEGA_VISION_KILL_SWITCH)
  2. picks ⊆ active_strategies
  3. prop firm tracker status != "pending" → reject
  4. cost tracker can_afford() is False → reject
  5. position/contract cap would be breached → reject
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SafetyGates:
    """Holds references to everything the gates need to check."""

    active_strategies: List[str] = field(default_factory=list)
    prop_firm_tracker: Any = None
    instrument_sizer: Any = None
    cost_tracker: Any = None
    kill_switch_env: str = "MEGA_VISION_KILL_SWITCH"
    max_positions: int = 3

    def validate_pick(self, pick: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Return ``(valid, reason)`` for a proposed pick."""
        # Gate 1: kill switch
        if self._kill_switch_active():
            return False, "kill switch active"

        # Gate 2: picks ⊆ active_strategies
        picks = pick.get("strategy_picks") or []
        if not isinstance(picks, list):
            return False, "strategy_picks must be a list"
        for p in picks:
            if p not in self.active_strategies:
                return False, f"strategy {p!r} not in active_strategies={self.active_strategies}"

        # Gate 3: prop firm status
        if self.prop_firm_tracker is not None:
            status = getattr(self.prop_firm_tracker, "status", None)
            if status is not None and status != "pending":
                return False, f"prop firm tracker status={status}"

        # Gate 4: cost budget
        if self.cost_tracker is not None:
            if not self.cost_tracker.can_afford():
                return False, "cost budget exhausted"

        # Gate 5: position cap (picks would exceed max_positions)
        if len(picks) > self.max_positions:
            return False, f"picks count {len(picks)} exceeds max_positions={self.max_positions}"

        return True, None

    def make_pre_tool_use_hook(self) -> Callable:
        """Return a hook compatible with ``ClaudeAgentOptions.hooks``.

        The hook signature matches what the Agent SDK's ``PreToolUse``
        matcher expects — it receives the tool name and arguments and
        returns a dict with ``deny`` / ``reason`` keys when the call
        should be blocked.
        """

        async def hook(input_data: Any, tool_use_id: Any, context: Any) -> Dict[str, Any]:
            try:
                tool_name = input_data.get("tool_name") if isinstance(input_data, dict) else None
                tool_input = input_data.get("tool_input") if isinstance(input_data, dict) else None
            except Exception:
                return {}

            if tool_name != "record_strategy_pick":
                return {}

            ok, reason = self.validate_pick(tool_input or {})
            if not ok:
                logger.info("SafetyGates blocked pick: %s", reason)
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": reason or "unknown",
                    }
                }
            return {}

        return hook

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _kill_switch_active(self) -> bool:
        value = os.environ.get(self.kill_switch_env or "", "")
        return bool(value) and value not in ("0", "false", "False", "")
