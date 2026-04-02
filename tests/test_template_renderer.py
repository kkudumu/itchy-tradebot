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
