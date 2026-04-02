"""Claude visual analysis of trade screenshots.

Sends annotated trade charts to Claude via the CLI for pattern
identification at entry/exit points. Returns structured feedback
including detected patterns, entry/exit quality assessment, and
confluence scoring adjustments.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def build_visual_prompt(
    trade: Dict[str, Any],
    strategy_name: str,
    patterns: Optional[List] = None,
    extra_context: str = "",
) -> str:
    """Build a structured prompt for Claude visual analysis of a trade chart.

    Parameters
    ----------
    trade: Trade metadata dict.
    strategy_name: Name of the strategy that generated the trade.
    patterns: Optional list of ChartPattern objects detected near the trade.
    extra_context: Additional context to include in the prompt.
    """
    r = trade.get("r_multiple", 0)
    direction = trade.get("direction", "unknown")
    entry_price = trade.get("entry_price", 0)
    exit_price = trade.get("exit_price", 0)
    stop_loss = trade.get("stop_loss", 0)
    reason = trade.get("_selection_reason", "")
    exit_reason = trade.get("exit_reason", "")

    pattern_section = ""
    if patterns:
        pattern_section = "\n## Pre-Detected Chart Patterns\n"
        for p in patterns:
            d = p.to_dict() if hasattr(p, "to_dict") else p
            pattern_section += (
                f"- {d.get('pattern_type', 'unknown')}: {d.get('description', '')}"
                f" (confidence: {d.get('confidence', 0):.0%})\n"
            )

    prompt = f"""You are a professional gold (XAU/USD) chart analyst reviewing a trade screenshot
from the {strategy_name} strategy. Analyze the attached candlestick chart.

## Trade Metadata
- **Direction:** {direction}
- **Entry Price:** {entry_price}
- **Exit Price:** {exit_price}
- **Stop Loss:** {stop_loss}
- **R-Multiple:** {r:+.2f}R
- **Selection Reason:** {reason}
- **Exit Reason:** {exit_reason}
{pattern_section}
{extra_context}

## Your Task

1. **Identify chart patterns** visible at the entry point (e.g., double bottom, head & shoulders,
   triangle breakout, wedge, flag, channel, support/resistance bounce)
2. **Assess entry quality** -- was this a good entry location given the visible pattern?
3. **Assess exit quality** -- could the exit have been better?
4. **Suggest confluence adjustment** -- should this pattern type add or subtract confluence points?

Respond with a JSON block:

```json
{{
  "patterns_at_entry": ["list of pattern names visible at entry"],
  "patterns_at_exit": ["list of pattern names visible at exit"],
  "entry_quality": "excellent|good|fair|poor",
  "exit_quality": "excellent|good|fair|poor",
  "confluence_adjustment": 0,
  "reasoning": "One paragraph explaining your analysis",
  "improvement_suggestion": "One actionable suggestion for the strategy"
}}
```

Be specific about what you see in the chart. The green triangle marks entry, red triangle marks exit."""

    return prompt


def parse_visual_response(response: str) -> Optional[Dict[str, Any]]:
    """Parse Claude's visual analysis response into a structured dict.

    Extracts the JSON block from the response. Returns None if parsing fails.
    """
    if not response or not response.strip():
        return None

    # Try ```json block first
    match = re.search(r"```json\s*\n(.*?)\n\s*```", response, re.DOTALL)
    if not match:
        # Try raw JSON object
        match = re.search(r"\{[^{}]*\"patterns_at_entry\"[^{}]*\}", response, re.DOTALL)
        if not match:
            logger.warning("Could not parse visual analysis response")
            return None

    try:
        raw = match.group(1) if match.lastindex else match.group(0)
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("JSON parse error in visual response: %s", e)
        return None

    # Validate required fields
    required = {"patterns_at_entry", "entry_quality", "confluence_adjustment"}
    if not required.issubset(result.keys()):
        logger.warning("Missing required fields: %s", required - result.keys())
        return None

    return result


def analyze_screenshot(
    screenshot_path: str,
    trade: Dict[str, Any],
    strategy_name: str,
    patterns: Optional[List] = None,
    cli_command: Optional[List[str]] = None,
    timeout: int = 120,
) -> Optional[Dict[str, Any]]:
    """Send a trade screenshot to Claude for visual analysis.

    Parameters
    ----------
    screenshot_path: Path to the PNG screenshot file.
    trade: Trade metadata dict.
    strategy_name: Strategy that generated the trade.
    patterns: Optional detected ChartPattern objects near this trade.
    cli_command: CLI command list. Defaults to ["claude", "-p"].
    timeout: Subprocess timeout in seconds.

    Returns
    -------
    Parsed analysis dict, or None on failure.
    """
    path = Path(screenshot_path)
    if not path.exists():
        logger.warning("Screenshot not found: %s", screenshot_path)
        return None

    prompt = build_visual_prompt(trade, strategy_name, patterns)

    # Claude Code CLI supports image input via file attachment
    cmd = cli_command or ["claude", "-p"]
    full_prompt = f"[Image: {screenshot_path}]\n\n{prompt}"

    logger.info("Analyzing screenshot: %s", path.name)

    try:
        result = subprocess.run(
            cmd,
            input=full_prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        response = result.stdout
        if not response.strip():
            logger.warning("Empty response from visual analysis CLI")
            return None
    except subprocess.TimeoutExpired:
        logger.error("Visual analysis CLI timed out after %ds", timeout)
        return None
    except FileNotFoundError:
        logger.error("CLI command not found: %s", cmd)
        return None

    analysis = parse_visual_response(response)
    if analysis:
        analysis["screenshot_path"] = str(path)
        analysis["trade_r"] = float(trade.get("r_multiple", 0))
        logger.info(
            "Visual analysis complete: entry=%s, exit=%s, adjustment=%+d",
            analysis.get("entry_quality"),
            analysis.get("exit_quality"),
            analysis.get("confluence_adjustment", 0),
        )
    return analysis
