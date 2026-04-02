"""Tests for Claude visual analysis of trade screenshots."""

import pytest


class TestBuildVisualPrompt:
    def test_prompt_contains_trade_metadata(self):
        from src.discovery.visual_analyzer import build_visual_prompt

        trade = {
            "r_multiple": -1.5,
            "direction": "long",
            "entry_price": 2050.0,
            "exit_price": 2042.5,
            "stop_loss": 2045.0,
            "_selection_reason": "worst_loser",
            "exit_reason": "stop_loss",
        }
        prompt = build_visual_prompt(trade, strategy_name="sss")

        assert "long" in prompt.lower()
        assert "2050" in prompt
        assert "-1.5" in prompt or "-1.50" in prompt
        assert "worst_loser" in prompt
        assert "sss" in prompt

    def test_prompt_asks_for_pattern_identification(self):
        from src.discovery.visual_analyzer import build_visual_prompt

        trade = {"r_multiple": 2.0, "direction": "short", "_selection_reason": "best_winner"}
        prompt = build_visual_prompt(trade, strategy_name="sss")

        assert "pattern" in prompt.lower()
        assert "entry" in prompt.lower()
        assert "json" in prompt.lower()

    def test_prompt_includes_detected_patterns(self):
        from src.discovery.visual_analyzer import build_visual_prompt
        from src.discovery.chart_patterns import ChartPattern

        trade = {"r_multiple": 1.0, "direction": "long", "_selection_reason": "best_winner"}
        patterns = [
            ChartPattern(
                pattern_type="double_bottom",
                direction="bullish",
                confidence=0.8,
                key_prices=[2020.0, 2040.0, 2021.0],
                start_index=10,
                end_index=40,
                description="Double bottom near 2020",
            )
        ]
        prompt = build_visual_prompt(trade, strategy_name="sss", patterns=patterns)
        assert "double_bottom" in prompt
        assert "2020" in prompt


class TestParseVisualAnalysis:
    def test_parses_json_response(self):
        from src.discovery.visual_analyzer import parse_visual_response

        response = '''Based on my analysis:

```json
{
  "patterns_at_entry": ["double_bottom", "support_bounce"],
  "patterns_at_exit": ["resistance_rejection"],
  "entry_quality": "good",
  "exit_quality": "poor",
  "confluence_adjustment": 1,
  "reasoning": "Entry was well-timed at double bottom support. Exit was premature.",
  "improvement_suggestion": "Hold through minor resistance when trend is strong."
}
```
'''
        result = parse_visual_response(response)

        assert result is not None
        assert "double_bottom" in result["patterns_at_entry"]
        assert result["entry_quality"] == "good"
        assert result["confluence_adjustment"] == 1

    def test_handles_empty_response(self):
        from src.discovery.visual_analyzer import parse_visual_response

        result = parse_visual_response("")
        assert result is None

    def test_handles_malformed_json(self):
        from src.discovery.visual_analyzer import parse_visual_response

        result = parse_visual_response("This response has no JSON block at all.")
        assert result is None
