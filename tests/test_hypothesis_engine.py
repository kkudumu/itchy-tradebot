"""Tests for hypothesis generation from SHAP insights."""

import pytest


class TestBuildPrompt:
    def test_prompt_contains_shap_rules(self):
        from src.discovery.hypothesis_engine import build_hypothesis_prompt
        from src.discovery.xgb_analyzer import SHAPInsight

        insight = SHAPInsight(
            feature_importance={"adx_value": 0.15, "sess_london": 0.12},
            top_interactions=[(("adx_value", "sess_london"), 0.08)],
            actionable_rules=[{
                "feature_a": "adx_value",
                "feature_b": "sess_london",
                "condition": "adx_value>=0.50 AND sess_london>=0.50",
                "quadrant_win_rate": 0.55,
                "baseline_win_rate": 0.37,
                "n_trades": 25,
                "lift": 1.486,
                "recommendation": "strong_filter",
            }],
        )
        prompt = build_hypothesis_prompt(insight, strategy_name="sss", window_id="w_003")

        assert "adx_value" in prompt
        assert "sess_london" in prompt
        assert "strong_filter" in prompt.lower() or "STRONG_FILTER" in prompt
        assert "55.0%" in prompt or "0.55" in prompt
        assert "sss" in prompt

    def test_prompt_includes_strategy_context(self):
        from src.discovery.hypothesis_engine import build_hypothesis_prompt
        from src.discovery.xgb_analyzer import SHAPInsight

        insight = SHAPInsight()
        prompt = build_hypothesis_prompt(insight, strategy_name="ichimoku", window_id="w_001")
        assert "ichimoku" in prompt


class TestParseHypotheses:
    def test_parses_json_block(self):
        from src.discovery.hypothesis_engine import parse_hypotheses_response

        response = '''Here's my analysis:

```json
[
  {
    "description": "ADX trending + London session improves SSS win rate",
    "config_change": {"strategies": {"sss": {"min_confluence_score": 3}}},
    "expected_improvement": "Win rate +8% based on SHAP lift 1.49",
    "confidence": "high"
  }
]
```

This is based on the interaction between ADX and session.'''

        hypotheses = parse_hypotheses_response(response, strategy_name="sss")
        assert len(hypotheses) == 1
        assert hypotheses[0]["description"] == "ADX trending + London session improves SSS win rate"
        assert hypotheses[0]["config_change"]["strategies"]["sss"]["min_confluence_score"] == 3
        assert hypotheses[0]["strategy"] == "sss"
        assert "id" in hypotheses[0]
