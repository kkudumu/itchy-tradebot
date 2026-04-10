"""Cost tracker for mega-vision agent calls.

Even in subscription mode (no per-call Anthropic billing), we track
token counts + decision counts so the dashboard can show agent
activity and so the offline eval harness can compute cost-per-
decision metrics. When a dollar budget is configured (e.g. for runs
against the raw API), the tracker refuses further calls when the
budget is exhausted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# Approximate per-million token prices used when computing run cost.
# Subscription runs report "subscription" as the cost category
# instead of a dollar figure.
_PRICES_USD: Dict[str, Dict[str, float]] = {
    "claude-opus-4-6": {"input_per_m": 5.00, "output_per_m": 25.00},
    "claude-sonnet-4-6": {"input_per_m": 3.00, "output_per_m": 15.00},
    "claude-haiku-4-5": {"input_per_m": 1.00, "output_per_m": 5.00},
    "claude-haiku-4-5-20251001": {"input_per_m": 1.00, "output_per_m": 5.00},
}


@dataclass
class CostTracker:
    """Accumulates token counts + derived cost for a run.

    Parameters
    ----------
    budget_usd:
        Optional dollar budget. When set, ``can_afford()`` returns
        False once ``total_cost_usd`` meets or exceeds the budget.
        Pass ``None`` for subscription mode (no hard cap).
    subscription_mode:
        When True, cost tracking logs "subscription" as the category
        and ``total_cost_usd`` stays at 0.0 (no dollar billing).
    """

    budget_usd: Optional[float] = None
    subscription_mode: bool = True
    decision_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost_usd: float = 0.0
    per_model: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def can_afford(self, model: str | None = None) -> bool:
        """Return True when another call fits inside the remaining budget."""
        if self.subscription_mode or self.budget_usd is None:
            return True
        return self.total_cost_usd < self.budget_usd

    def record(self, model: str, usage: Dict[str, Any] | None) -> None:
        """Accumulate usage from one agent call.

        *usage* is the dict returned by the Agent SDK's usage hook —
        typically ``{"input_tokens": N, "output_tokens": M}``.
        """
        self.decision_count += 1
        if not usage:
            return
        in_t = int(usage.get("input_tokens") or 0)
        out_t = int(usage.get("output_tokens") or 0)
        self.input_tokens += in_t
        self.output_tokens += out_t

        entry = self.per_model.setdefault(
            model, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        )
        entry["calls"] += 1
        entry["input_tokens"] += in_t
        entry["output_tokens"] += out_t

        if not self.subscription_mode:
            prices = _PRICES_USD.get(model, {})
            in_price = prices.get("input_per_m", 0.0) / 1_000_000
            out_price = prices.get("output_per_m", 0.0) / 1_000_000
            call_cost = in_t * in_price + out_t * out_price
            entry["cost_usd"] += call_cost
            self.total_cost_usd += call_cost

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_count": self.decision_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_cost_usd": self.total_cost_usd if not self.subscription_mode else None,
            "cost_category": "subscription" if self.subscription_mode else "api",
            "budget_usd": self.budget_usd,
            "per_model": self.per_model,
        }
