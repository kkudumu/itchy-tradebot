"""Custom trading tools for the mega-vision agent (plan Task 24).

The seven tools the agent uses to investigate state + record its pick:

  1. get_market_state — current bars + indicators + regime
  2. get_recent_telemetry — rolling signal/rejection summary
  3. get_strategy_performance_buckets — per-strategy perf in similar
     conditions
  4. get_recent_trades — last N closed trades
  5. get_regime_tag — explicit regime classification
  6. view_chart_screenshot — PNG chart rendered on-demand
  7. record_strategy_pick — terminal tool; records the agent's
     final decision and ends its turn

These functions are factory-constructed via :func:`make_tools(ctx)`
so each agent invocation gets a scoped tool set pointing at the same
:class:`ContextBuilder` / :class:`TradeMemory` / etc. references.

When ``claude_agent_sdk`` is installed, the tools are decorated with
``@tool`` and exposed as MCP tools. When it's NOT installed (dev
boxes without the SDK), the module falls back to plain callables so
unit tests can still import and exercise the tool logic without
pulling in the SDK as a hard dependency.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional SDK import — tools work without it for unit tests
# ---------------------------------------------------------------------------


try:
    from claude_agent_sdk import tool as _sdk_tool  # type: ignore[import-not-found]

    _HAS_SDK = True
except ImportError:
    _HAS_SDK = False

    def _sdk_tool(name: str, description: str, schema: Dict[str, Any]):
        """Fallback @tool decorator that just attaches metadata."""

        def decorator(fn):
            fn.tool_name = name
            fn.tool_description = description
            fn.tool_schema = schema
            return fn

        return decorator


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------


def make_tools(ctx: Any) -> List[Callable]:
    """Build the tool list for a fresh agent invocation.

    *ctx* must expose at minimum:
      * context_builder  — a :class:`ContextBuilder`
      * telemetry_collector  — optional
      * trade_memory  — optional
      * performance_buckets  — optional
      * regime_detector  — optional
      * screenshot_provider  — optional
      * current_ts / current_candles  — set by the agent before
        running the query
      * last_pick  — set by ``record_strategy_pick``, read by the
        agent after the query loop ends

    The tools capture *ctx* in closures so each invocation has its
    own scoped view of the trading state.
    """

    @_sdk_tool(
        "get_market_state",
        "Returns current OHLCV bars (last 60), indicators (ATR, ADX, EMAs, "
        "ichimoku cloud), and current regime tag.",
        {},
    )
    async def get_market_state(args: Dict[str, Any] | None = None) -> Dict[str, Any]:
        state = ctx.context_builder.current_market_state()
        if hasattr(ctx, "current_candles") and ctx.current_candles is not None:
            candles = ctx.current_candles
            try:
                tail = candles.tail(5) if hasattr(candles, "tail") else candles
                state["recent_closes"] = (
                    tail["close"].tolist() if "close" in getattr(tail, "columns", []) else []
                )
            except Exception:
                state["recent_closes"] = []
        return {
            "content": [
                {"type": "text", "text": json.dumps(state, default=str)}
            ]
        }

    @_sdk_tool(
        "get_recent_telemetry",
        "Returns rolling-window summary of recent signal events per strategy: "
        "counts generated, counts entered, top rejection reasons.",
        {"window_n": int},
    )
    async def get_recent_telemetry(args: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if ctx.telemetry_collector is None:
            return _text_content({"error": "no telemetry"})
        try:
            summary = ctx.telemetry_collector.summary()
        except Exception as exc:  # noqa: BLE001
            return _text_content({"error": str(exc)})
        return _text_content(summary)

    @_sdk_tool(
        "get_strategy_performance_buckets",
        "Returns per-strategy performance buckets (win rate, avg R, expectancy) "
        "for the current session/pattern/regime, optionally filtered by strategy name.",
        {"strategy_name": str},
    )
    async def get_strategy_performance_buckets(args: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if ctx.performance_buckets is None:
            return _text_content({})
        name = (args or {}).get("strategy_name")
        try:
            buckets = ctx.performance_buckets.get_buckets(strategy_name=name)
        except Exception as exc:  # noqa: BLE001
            return _text_content({"error": str(exc)})
        return _text_content(buckets)

    @_sdk_tool(
        "get_recent_trades",
        "Returns the last N closed trades with full context.",
        {"n": int, "strategy_name": str},
    )
    async def get_recent_trades(args: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if ctx.trade_memory is None:
            return _text_content([])
        n = int((args or {}).get("n", 10))
        name = (args or {}).get("strategy_name")
        try:
            if name:
                rows = ctx.trade_memory.query_recent(strategy=name, n=n)
            else:
                rows = ctx.trade_memory.query({}, limit=n)
        except Exception as exc:  # noqa: BLE001
            return _text_content({"error": str(exc)})
        return _text_content(rows)

    @_sdk_tool(
        "get_regime_tag",
        "Returns the current market regime tag (trend_up, trend_down, range, "
        "high_vol, low_vol) plus the indicators that drove the classification.",
        {},
    )
    async def get_regime_tag(args: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if ctx.regime_detector is None:
            return _text_content({"regime": "unknown"})
        try:
            regime = ctx.regime_detector.current_regime()
        except Exception as exc:  # noqa: BLE001
            return _text_content({"error": str(exc)})
        return _text_content(regime)

    @_sdk_tool(
        "view_chart_screenshot",
        "Returns a chart screenshot of the current market state with active "
        "strategies' indicators overlaid. Use this to visually analyze the "
        "price action.",
        {},
    )
    async def view_chart_screenshot(args: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if ctx.screenshot_provider is None:
            return _text_content({"error": "screenshot capture disabled"})
        try:
            path = ctx.screenshot_provider.render_for_decision(
                ts=ctx.current_ts, candles_window=ctx.current_candles
            )
        except Exception as exc:  # noqa: BLE001
            return _text_content({"error": str(exc)})
        if path is None:
            return _text_content({"error": "screenshot capture disabled"})
        img_bytes = Path(path).read_bytes()
        img_b64 = base64.standard_b64encode(img_bytes).decode("utf-8")
        return {
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_b64,
                    },
                }
            ]
        }

    @_sdk_tool(
        "record_strategy_pick",
        "Records the agent's final strategy pick for this decision. Pass "
        "strategy_picks (list of strategy names to enable), confidence (0.0-1.0), "
        "and reasoning (string). Calling this tool ends your decision turn.",
        {"strategy_picks": list, "confidence": float, "reasoning": str},
    )
    async def record_strategy_pick(args: Dict[str, Any] | None = None) -> Dict[str, Any]:
        args = args or {}
        ctx.last_pick = {
            "strategy_picks": list(args.get("strategy_picks") or []),
            "confidence": float(args.get("confidence") or 0.0),
            "reasoning": str(args.get("reasoning") or ""),
            "ts": ctx.current_ts,
        }
        return _text_content("Pick recorded. End your turn.")

    return [
        get_market_state,
        get_recent_telemetry,
        get_strategy_performance_buckets,
        get_recent_trades,
        get_regime_tag,
        view_chart_screenshot,
        record_strategy_pick,
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text_content(payload: Any) -> Dict[str, Any]:
    return {
        "content": [
            {"type": "text", "text": json.dumps(payload, default=str)}
        ]
    }
