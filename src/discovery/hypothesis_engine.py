"""Claude-powered hypothesis generation from SHAP insights.

Builds a structured prompt from SHAPInsight, sends it to Claude Code CLI
(or Codex CLI as fallback), and parses the response into hypothesis dicts.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import uuid
from typing import Any, Dict, List, Optional

from src.discovery.xgb_analyzer import SHAPInsight

logger = logging.getLogger(__name__)


def build_hypothesis_prompt(
    insight: SHAPInsight,
    strategy_name: str,
    window_id: str,
    extra_context: str = "",
) -> str:
    """Build a structured prompt for Claude to generate hypotheses.

    The prompt includes SHAP feature importance, top interactions,
    actionable rules with win rates, and asks Claude to propose
    concrete config changes as testable hypotheses.
    """
    # Format feature importance
    feat_lines = ""
    for name, imp in insight.feature_importance.items():
        feat_lines += f"  {name}: {imp:.4f}\n"

    # Format interactions
    inter_lines = ""
    for (f_a, f_b), strength in insight.top_interactions:
        inter_lines += f"  {f_a} x {f_b}: {strength:.4f}\n"

    # Format rules
    rule_lines = ""
    for r in insight.actionable_rules:
        wr_pct = r['quadrant_win_rate'] * 100
        base_pct = r['baseline_win_rate'] * 100
        rule_lines += (
            f"  [{r['recommendation'].upper()}] {r['condition']}\n"
            f"    Win rate: {wr_pct:.1f}% vs baseline {base_pct:.1f}% "
            f"(lift {r['lift']:.2f}x, n={r['n_trades']})\n"
        )

    prompt = f"""You are a quantitative trading analyst reviewing SHAP analysis results
for the {strategy_name} strategy on XAU/USD (gold). Window: {window_id}.

## Top Features by Importance (mean |SHAP|)
{feat_lines or '  (no significant features)'}

## Top Feature Interactions (pairwise SHAP)
{inter_lines or '  (no significant interactions)'}

## Actionable Rules Discovered
{rule_lines or '  (no actionable rules found)'}

{extra_context}

## Your Task

Based on the SHAP analysis above, propose 1-3 testable hypotheses as concrete
config changes for the {strategy_name} strategy. Each hypothesis should:

1. Explain WHY the feature interaction matters for gold trading
2. Propose a specific config change (edge filter threshold, strategy parameter, etc.)
3. Estimate the expected improvement based on the SHAP evidence

Respond with a JSON array of hypothesis objects:

```json
[
  {{
    "description": "Human-readable explanation of the hypothesis",
    "config_change": {{"strategies": {{"{strategy_name}": {{"param": "value"}}}}}},
    "expected_improvement": "What we expect to see and why",
    "confidence": "high|medium|low"
  }}
]
```

Be creative but grounded in the data. Only propose changes supported by the SHAP evidence."""

    return prompt


def parse_hypotheses_response(
    response: str,
    strategy_name: str,
) -> List[Dict[str, Any]]:
    """Parse Claude's response into structured hypothesis dicts.

    Extracts the JSON array from a ```json code block in the response.
    Adds id, strategy, and status fields to each hypothesis.
    """
    # Extract JSON block
    match = re.search(r"```json\s*\n(.*?)\n\s*```", response, re.DOTALL)
    if not match:
        # Try raw JSON array
        match = re.search(r"\[\s*\{.*?\}\s*\]", response, re.DOTALL)
        if not match:
            logger.warning("Could not parse hypotheses from response")
            return []

    try:
        raw = match.group(1) if match.lastindex else match.group(0)
        hypotheses = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("JSON parse error: %s", e)
        return []

    if not isinstance(hypotheses, list):
        hypotheses = [hypotheses]

    for hyp in hypotheses:
        hyp["id"] = f"hyp_{uuid.uuid4().hex[:8]}"
        hyp["strategy"] = strategy_name
        hyp["status"] = "proposed"

    return hypotheses


def generate_hypotheses(
    insight: SHAPInsight,
    strategy_name: str,
    window_id: str,
    extra_context: str = "",
    cli_command: Optional[List[str]] = None,
    timeout: int = 300,
) -> List[Dict[str, Any]]:
    """Generate hypotheses by invoking Claude Code CLI.

    Parameters
    ----------
    insight: SHAP analysis results.
    strategy_name: Which strategy to target.
    window_id: Current optimization window identifier.
    extra_context: Additional context (e.g., prior learnings).
    cli_command: CLI command list. Defaults to ["claude", "-p"].
    timeout: Subprocess timeout in seconds.

    Returns
    -------
    List of hypothesis dicts with id, description, config_change, etc.
    """
    prompt = build_hypothesis_prompt(insight, strategy_name, window_id, extra_context)

    cmd = cli_command or ["claude", "-p"]
    logger.info("Generating hypotheses via %s for %s window %s", cmd[0], strategy_name, window_id)

    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        response = result.stdout
        if not response.strip():
            logger.warning("Empty response from CLI")
            return []
    except subprocess.TimeoutExpired:
        logger.error("CLI timed out after %ds", timeout)
        return []
    except FileNotFoundError:
        logger.error("CLI command not found: %s", cmd)
        return []

    hypotheses = parse_hypotheses_response(response, strategy_name)
    logger.info("Generated %d hypotheses", len(hypotheses))
    return hypotheses
