"""SDK MCP server factory for the mega-vision trading tools.

Wraps :func:`make_tools` in an in-process MCP server that the
:class:`MegaStrategyAgent` passes to ``ClaudeAgentOptions`` via
``mcp_servers={"trading": server}``. The tools run as Python
callables in the same process — no network hop, no subprocess.

When ``claude_agent_sdk`` is not installed (dev / test), returns a
lightweight fake object with the same interface so unit tests can
exercise the wiring without pulling in the SDK as a hard dep.
"""

from __future__ import annotations

from typing import Any

from .tools import make_tools


try:
    from claude_agent_sdk import create_sdk_mcp_server  # type: ignore[import-not-found]

    _HAS_SDK = True
except ImportError:
    _HAS_SDK = False

    class _FakeMcpServer:
        def __init__(self, name: str, tools: list) -> None:
            self.name = name
            self.tools = tools

    def create_sdk_mcp_server(name: str, tools: list, version: str = "1.0.0") -> Any:
        return _FakeMcpServer(name=name, tools=tools)


def make_trading_mcp_server(ctx: Any) -> Any:
    """Return an SDK MCP server wrapping the mega-vision trading tools."""
    tools = make_tools(ctx)
    return create_sdk_mcp_server(name="mega-vision-trading-tools", tools=tools)
