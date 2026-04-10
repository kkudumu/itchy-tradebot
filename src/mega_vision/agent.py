"""MegaStrategyAgent — claude-agent-sdk wrapper for the strategy selector.

Runs one agent decision per signal event. The flow:

  1. Check kill switch env var — if set, return fallback immediately
  2. Check cost tracker — if budget exhausted, return fallback
  3. Assemble user message from the prompt template + context bundle
  4. Run ``query()`` with the trading MCP server + safety hook
  5. Extract ``ctx.last_pick`` (set by the record_strategy_pick tool)
  6. Return the pick, or a fallback on any error

The agent inherits authentication from the installed Claude CLI
subscription — it does NOT require an ``ANTHROPIC_API_KEY`` env var
per the user's subscription-mode preference.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cost_tracker import CostTracker
from .safety_gates import SafetyGates

logger = logging.getLogger(__name__)


try:
    from claude_agent_sdk import (  # type: ignore[import-not-found]
        ClaudeAgentOptions,
        HookMatcher,
        query,
    )

    _HAS_SDK = True
except ImportError:
    _HAS_SDK = False
    ClaudeAgentOptions = None  # type: ignore[assignment]
    HookMatcher = None  # type: ignore[assignment]

    async def query(*args: Any, **kwargs: Any):  # type: ignore[no-redef]
        # Fake async generator — returns nothing
        if False:
            yield None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MegaVisionConfig:
    mode: str = "disabled"  # disabled | shadow | authority
    shadow_model: str = "claude-opus-4-6"
    live_model: str = "claude-haiku-4-5-20251001"
    decision_cadence: str = "per_signal"
    per_n_bars: int = 5
    cost_budget_usd: Optional[float] = None  # None = subscription mode
    kill_switch_env_var: str = "MEGA_VISION_KILL_SWITCH"
    max_retries_per_decision: int = 3
    fallback_on_timeout_seconds: int = 30
    subscription_mode: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "MegaVisionConfig":
        if not data:
            return cls()
        return cls(
            mode=str(data.get("mode", "disabled")),
            shadow_model=str(data.get("shadow_model", "claude-opus-4-6")),
            live_model=str(data.get("live_model", "claude-haiku-4-5-20251001")),
            decision_cadence=str(data.get("decision_cadence", "per_signal")),
            per_n_bars=int(data.get("per_n_bars", 5)),
            cost_budget_usd=(
                float(data["cost_budget_usd"])
                if "cost_budget_usd" in data and data["cost_budget_usd"] is not None
                else None
            ),
            kill_switch_env_var=str(data.get("kill_switch_env_var", "MEGA_VISION_KILL_SWITCH")),
            max_retries_per_decision=int(data.get("max_retries_per_decision", 3)),
            fallback_on_timeout_seconds=int(data.get("fallback_on_timeout_seconds", 30)),
            subscription_mode=bool(data.get("subscription_mode", True)),
        )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class MegaStrategyAgent:
    """Claude-powered strategy selector.

    The agent is construction-stateful — ``ctx`` is passed in and
    used by the MCP tools during inference. Each call to
    :meth:`decide` resets ``ctx.last_pick`` before invoking the
    query loop.
    """

    def __init__(
        self,
        config: MegaVisionConfig,
        ctx: Any,
        prop_firm_tracker: Any,
        instrument_sizer: Any = None,
        telemetry_collector: Any = None,
        active_strategies: Optional[List[str]] = None,
        prompts_dir: Path | str | None = None,
    ) -> None:
        self.config = config
        self.ctx = ctx
        self.cost_tracker = CostTracker(
            budget_usd=config.cost_budget_usd,
            subscription_mode=config.subscription_mode,
        )
        self.gates = SafetyGates(
            active_strategies=active_strategies or [],
            prop_firm_tracker=prop_firm_tracker,
            instrument_sizer=instrument_sizer,
            cost_tracker=self.cost_tracker,
            kill_switch_env=config.kill_switch_env_var,
        )
        self._telemetry = telemetry_collector
        self._prompts_dir = Path(prompts_dir or _default_prompts_dir())
        self._system_prompt: Optional[str] = None
        self._user_template: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def decide(
        self,
        ts: datetime,
        candidate_signals: List[Any],
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ask the agent for a pick. Returns the pick dict or a fallback."""
        active_mode = mode or self.config.mode
        if active_mode == "disabled":
            return self._fallback("disabled")

        # Kill switch — checked first so the agent is never even called
        if os.environ.get(self.config.kill_switch_env_var or "", ""):
            return self._fallback("kill_switch")

        model = (
            self.config.shadow_model
            if active_mode == "shadow"
            else self.config.live_model
        )

        if not self.cost_tracker.can_afford(model):
            return self._fallback("cost_budget_exceeded")

        if not _HAS_SDK:
            logger.info("claude_agent_sdk not installed — returning fallback")
            return self._fallback("sdk_not_installed")

        # Prepare the context for the tools
        self.ctx.current_ts = ts
        self.ctx.last_pick = None

        system_prompt = self._load_system_prompt()
        user_message = self._render_user_message(ts, candidate_signals)

        # Build options
        from .mcp_server import make_trading_mcp_server

        mcp_server = make_trading_mcp_server(self.ctx)
        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            mcp_servers={"trading": mcp_server},
            allowed_tools=[
                "mcp__trading__get_market_state",
                "mcp__trading__get_recent_telemetry",
                "mcp__trading__get_strategy_performance_buckets",
                "mcp__trading__get_recent_trades",
                "mcp__trading__get_regime_tag",
                "mcp__trading__view_chart_screenshot",
                "mcp__trading__record_strategy_pick",
            ],
            permission_mode="bypassPermissions",
            model=model,
            max_turns=10,
            hooks={
                "PreToolUse": [
                    HookMatcher(
                        matcher="record_strategy_pick",
                        hooks=[self.gates.make_pre_tool_use_hook()],
                    )
                ]
            },
        )

        try:
            async for message in query(prompt=user_message, options=options):
                usage = getattr(message, "usage", None)
                if usage is not None:
                    self.cost_tracker.record(model, dict(usage) if not isinstance(usage, dict) else usage)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MegaStrategyAgent query failed: %s", exc)
            return self._fallback(f"agent_error:{type(exc).__name__}")

        if self.ctx.last_pick is None:
            return self._fallback("no_pick_recorded")

        return self.ctx.last_pick

    # ------------------------------------------------------------------
    # Prompt loading
    # ------------------------------------------------------------------

    def _load_system_prompt(self) -> str:
        if self._system_prompt is not None:
            return self._system_prompt
        path = self._prompts_dir / "mega_vision_system.md"
        if path.exists():
            self._system_prompt = path.read_text(encoding="utf-8")
        else:
            self._system_prompt = _DEFAULT_SYSTEM_PROMPT
        return self._system_prompt

    def _load_user_template(self) -> str:
        if self._user_template is not None:
            return self._user_template
        path = self._prompts_dir / "mega_vision_user_template.md"
        if path.exists():
            self._user_template = path.read_text(encoding="utf-8")
        else:
            self._user_template = _DEFAULT_USER_TEMPLATE
        return self._user_template

    def _render_user_message(
        self,
        ts: datetime,
        candidate_signals: List[Any],
    ) -> str:
        template = self._load_user_template()
        summaries = [
            f"  - {getattr(s, 'strategy_name', '?')} {getattr(s, 'direction', '?')}"
            f" entry={getattr(s, 'entry_price', None)}"
            f" stop={getattr(s, 'stop_loss', None)}"
            f" tp={getattr(s, 'take_profit', None)}"
            for s in candidate_signals
        ]
        return template.format(
            timestamp_utc=ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            candidate_signals_summary="\n".join(summaries) or "(none)",
            prop_firm_style="topstep_combine_dollar",
            distance_to_mll_usd=0.0,
            daily_loss_used_usd=0.0,
            daily_loss_limit_usd=1000.0,
            distance_to_target_usd=0.0,
            open_positions_count=0,
            equity_usd=50000.0,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fallback(self, reason: str) -> Dict[str, Any]:
        return {
            "strategy_picks": None,
            "confidence": 0.0,
            "reasoning": f"FALLBACK: {reason}",
            "fallback": True,
            "ts": datetime.now(timezone.utc).isoformat(),
        }


# ---------------------------------------------------------------------------
# Defaults (used when prompts/ files are missing)
# ---------------------------------------------------------------------------


_DEFAULT_SYSTEM_PROMPT = """You are the Mega-Strategy Trading Agent for itchy-tradebot.
Your job: at each decision point, decide which candidate strategies should fire.
Use your tools to investigate state, then call record_strategy_pick with your picks,
confidence, and reasoning.
"""


_DEFAULT_USER_TEMPLATE = """Decision request at {timestamp_utc}.

Candidate signals from native blender:
{candidate_signals_summary}

Prop firm state: {prop_firm_style} | MLL headroom ${distance_to_mll_usd} | daily loss {daily_loss_used_usd}/{daily_loss_limit_usd}
Risk state: positions={open_positions_count} equity=${equity_usd}

Investigate and call record_strategy_pick.
"""


def _default_prompts_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "prompts"
