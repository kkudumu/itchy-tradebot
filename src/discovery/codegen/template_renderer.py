"""Jinja2 template rendering for EdgeFilter code generation.

Renders two types of output:
1. A structured prompt for the LLM, including the hypothesis evidence,
   existing EdgeFilter patterns, and the expected class structure.
2. Skeleton code from Jinja2 templates (used as reference, the LLM
   generates the final version with actual logic).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from jinja2 import Environment, FileSystemLoader, StrictUndefined

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _get_env() -> Environment:
    """Create a Jinja2 environment pointing at the templates directory."""
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_filter_template(filter_spec: Dict[str, Any]) -> str:
    """Render the EdgeFilter skeleton from the Jinja2 template.

    Parameters
    ----------
    filter_spec:
        Dict with keys: name, class_name, category, description, params,
        logic_description.

    Returns
    -------
    Python source code string for the EdgeFilter subclass skeleton.
    """
    env = _get_env()
    template = env.get_template("edge_filter.py.j2")
    return template.render(filter_spec=filter_spec)


def render_test_template(filter_spec: Dict[str, Any]) -> str:
    """Render the test class skeleton from the Jinja2 template.

    Parameters
    ----------
    filter_spec:
        Same dict as render_filter_template.

    Returns
    -------
    Python source code string for the test class skeleton.
    """
    env = _get_env()
    template = env.get_template("edge_test.py.j2")
    return template.render(filter_spec=filter_spec)


# -- Reference code snippets for the LLM prompt --

_SPREAD_FILTER_EXAMPLE = '''class SpreadFilter(EdgeFilter):
    """Block entries when the current spread exceeds max_spread_points."""

    def __init__(self, config: dict) -> None:
        super().__init__("spread_filter", config)
        params = config.get("params", {})
        self._max_spread: float = float(params.get("max_spread_points", 30))

    def should_allow(self, context: EdgeContext) -> EdgeResult:
        if not self.enabled:
            return self._disabled_result()
        spread = context.spread
        if spread <= self._max_spread:
            return EdgeResult(
                allowed=True, edge_name=self.name,
                reason=f"Spread {spread:.1f} pts within limit {self._max_spread:.0f} pts",
            )
        return EdgeResult(
            allowed=False, edge_name=self.name,
            reason=f"Spread {spread:.1f} pts exceeds max {self._max_spread:.0f} pts",
        )'''

_REGIME_FILTER_EXAMPLE = '''class RegimeFilter(EdgeFilter):
    """Require ADX above threshold AND cloud thickness above minimum."""

    def __init__(self, config: dict) -> None:
        super().__init__("regime_filter", config)
        params = config.get("params", {})
        self._adx_min: float = float(params.get("adx_min", 28))
        self._cloud_min: float = float(params.get("cloud_thickness_percentile", 50))

    def should_allow(self, context: EdgeContext) -> EdgeResult:
        if not self.enabled:
            return self._disabled_result()
        adx_ok = context.adx >= self._adx_min
        cloud_thickness = context.indicator_values.get("cloud_thickness")
        if cloud_thickness is None:
            return EdgeResult(allowed=True, edge_name=self.name, reason="no cloud_thickness")
        cloud_ok = cloud_thickness >= self._cloud_min
        return EdgeResult(
            allowed=adx_ok and cloud_ok, edge_name=self.name,
            reason=f"ADX {'OK' if adx_ok else 'low'}; cloud {'OK' if cloud_ok else 'thin'}",
        )'''


def render_codegen_prompt(hypothesis: Dict[str, Any]) -> str:
    """Build a structured prompt for Claude to generate a complete EdgeFilter.

    The prompt includes:
    1. The hypothesis description and SHAP evidence
    2. Two real EdgeFilter examples from the codebase
    3. The EdgeContext/EdgeResult API reference
    4. The filter_spec with parameter definitions and logic description
    5. Instructions to generate both the filter class AND a test class

    Parameters
    ----------
    hypothesis:
        Dict with keys: id, description, evidence, strategy, filter_spec.

    Returns
    -------
    Prompt string ready to send to Claude Code CLI.
    """
    spec = hypothesis.get("filter_spec", {})
    evidence = hypothesis.get("evidence", {})

    # Format params
    params_desc = ""
    for pname, pdef in spec.get("params", {}).items():
        params_desc += f"    - {pname}: {pdef['type']} (default: {pdef['default']})\n"

    # Format evidence
    evidence_desc = ""
    if evidence:
        evidence_desc = (
            f"  - Feature A: {evidence.get('feature_a', 'N/A')}\n"
            f"  - Feature B: {evidence.get('feature_b', 'N/A')}\n"
            f"  - Condition: {evidence.get('condition', 'N/A')}\n"
            f"  - Win rate: {evidence.get('quadrant_win_rate', 0) * 100:.1f}% "
            f"vs baseline {evidence.get('baseline_win_rate', 0) * 100:.1f}%\n"
            f"  - Lift: {evidence.get('lift', 0):.2f}x over {evidence.get('n_trades', 0)} trades\n"
        )

    prompt = f"""You are generating a Python EdgeFilter subclass for an XAU/USD trading bot.

## Hypothesis
{hypothesis.get('description', 'No description')}

## SHAP Evidence
{evidence_desc or '  (no quantitative evidence)'}

## Filter Specification
- Name: {spec.get('name', 'unnamed')}
- Class: {spec.get('class_name', 'UnnamedFilter')}
- Category: {spec.get('category', 'entry')}
- Description: {spec.get('description', '')}
- Logic: {spec.get('logic_description', '')}
- Parameters:
{params_desc or '    (none)'}

## API Reference

EdgeContext fields available:
  timestamp (datetime), day_of_week (int 0-6), close_price (float),
  high_price (float), low_price (float), spread (float),
  session (str: 'london'|'new_york'|'overlap'|'asian'|'off_hours'),
  adx (float), atr (float), indicator_values (dict[str, float]),
  bb_squeeze (bool), confluence_score (int 0-8), signal (Optional[object])

EdgeResult(allowed: bool, edge_name: str, reason: str, modifier: Optional[float] = None)

EdgeFilter base class provides:
  self.name (str), self.config (dict), self.enabled (bool)
  self._disabled_result() -> EdgeResult (returns allowed=True passthrough)

## Existing filter examples (follow this exact pattern)

Example 1 -- SpreadFilter:
```python
{_SPREAD_FILTER_EXAMPLE}
```

Example 2 -- RegimeFilter:
```python
{_REGIME_FILTER_EXAMPLE}
```

## Requirements

1. Generate a COMPLETE Python file for `src/edges/generated/{spec.get('name', 'unnamed')}.py`
2. The class must inherit from EdgeFilter and implement should_allow()
3. Use `super().__init__("{spec.get('name', 'unnamed')}", config)` in __init__
4. Read all params from `config.get("params", {{}})`
5. Return `self._disabled_result()` when `not self.enabled`
6. Return EdgeResult with descriptive reason strings
7. Only import from: numpy, pandas, src.edges.base, src.indicators.*
8. No os, subprocess, eval, exec, __import__, open, network, or file writes

Then generate a COMPLETE test file with at least 3 test methods:
  - test_allows_when_condition_met
  - test_blocks_when_condition_violated
  - test_disabled_passes_through

Respond with two code blocks:

```python
# FILE: src/edges/generated/{spec.get('name', 'unnamed')}.py
<complete filter code>
```

```python
# FILE: tests/test_generated_{spec.get('name', 'unnamed')}.py
<complete test code>
```
"""
    return prompt
