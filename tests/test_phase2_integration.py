"""Integration test: pattern detection -> screenshot selection -> visual analysis -> confluence."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest


def _make_swing(price, swing_type, index=0):
    from src.strategy.strategies.sss.breathing_room import SwingPoint
    return SwingPoint(
        index=index,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        price=price,
        swing_type=swing_type,
        bar_count_since_prev=10,
    )


class TestPhase2Pipeline:
    def test_pattern_detection_to_confluence(self):
        """Patterns detected -> confluence scored -> EdgeContext injectable."""
        from src.discovery.pattern_hook import PatternHook

        hook = PatternHook(atr=5.0)

        # Build a swing history with a clear double bottom (highs far apart to avoid double top)
        swings = [
            _make_swing(2020.0, "low", index=10),
            _make_swing(2040.0, "high", index=20),
            _make_swing(2019.5, "low", index=30),
            _make_swing(2055.0, "high", index=40),  # far from first high -- no double top
        ]

        patterns = hook.detect(swings)
        assert len(patterns) >= 1
        assert any(p.pattern_type == "double_bottom" for p in patterns)

        # Confluence should boost longs (double bottom is bullish)
        adj_long = hook.get_confluence_adjustment(patterns, "long")
        assert adj_long > 0

        # Confluence should penalize shorts
        adj_short = hook.get_confluence_adjustment(patterns, "short")
        assert adj_short < 0

        # Serializable for EdgeContext
        dicts = hook.patterns_as_dicts()
        assert isinstance(dicts, list)
        assert all(isinstance(d, dict) for d in dicts)

    def test_screenshot_selection_with_trades(self):
        """Trade list -> selector picks worst/best -> no crashes."""
        from src.discovery.screenshot_selector import ScreenshotSelector

        selector = ScreenshotSelector(n_worst=3, n_best=3)
        trades = []
        for i in range(20):
            r = float(np.random.default_rng(i).uniform(-2.0, 3.0))
            trades.append({
                "r_multiple": r,
                "entry_bar_idx": i * 50,
                "exit_bar_idx": i * 50 + 30,
                "entry_price": 2050.0,
                "exit_price": 2050.0 + r * 5.0,
                "stop_loss": 2045.0,
                "direction": "long",
                "entry_time": datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
                "exit_time": datetime(2026, 1, 15, 14, 0, tzinfo=timezone.utc),
            })

        selected = selector.select_trades(trades)
        assert 3 <= len(selected) <= 10  # at most n_worst + n_best + near_SL
        reasons = {t["_selection_reason"] for t in selected}
        assert "worst_loser" in reasons
        assert "best_winner" in reasons

    def test_visual_prompt_and_response_round_trip(self):
        """Build prompt -> mock response -> parse -> valid result."""
        from src.discovery.visual_analyzer import build_visual_prompt, parse_visual_response

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
        assert len(prompt) > 100

        # Simulate Claude response
        mock_response = '''Analysis:

```json
{
  "patterns_at_entry": ["rising_wedge"],
  "patterns_at_exit": ["support_break"],
  "entry_quality": "poor",
  "exit_quality": "fair",
  "confluence_adjustment": -1,
  "reasoning": "Entry was against the rising wedge pattern. Should have waited for breakout.",
  "improvement_suggestion": "Add wedge detection to filter entries against wedge direction."
}
```
'''
        result = parse_visual_response(mock_response)
        assert result is not None
        assert result["entry_quality"] == "poor"
        assert result["confluence_adjustment"] == -1

    def test_knowledge_base_stores_pattern_findings(self):
        """Detected patterns persist to knowledge base for future reference."""
        from src.discovery.knowledge_base import KnowledgeBase

        kb_dir = str(Path(tempfile.mkdtemp()) / "kb")
        kb = KnowledgeBase(base_dir=kb_dir)

        # Save pattern-derived hypothesis
        hyp = {
            "id": "hyp_pattern_001",
            "description": "Double bottom at 2020 support consistently leads to 2R+ winners",
            "source": "chart_pattern_visual",
            "strategy": "sss",
            "status": "proposed",
            "evidence": {
                "pattern_type": "double_bottom",
                "occurrences": 5,
                "avg_r_when_aligned": 1.8,
                "avg_r_when_opposed": -0.7,
            },
        }
        kb.save_hypothesis(hyp)
        loaded = kb.load_hypothesis("hyp_pattern_001")
        assert loaded["source"] == "chart_pattern_visual"
        assert loaded["evidence"]["pattern_type"] == "double_bottom"

    def test_full_pipeline_no_crashes(self):
        """End-to-end: swings -> patterns -> selection -> prompt -> parse.

        Verifies the entire pipeline runs without exceptions,
        even without actual Claude CLI calls.
        """
        from src.discovery.pattern_hook import PatternHook
        from src.discovery.screenshot_selector import ScreenshotSelector
        from src.discovery.visual_analyzer import build_visual_prompt, parse_visual_response

        # 1. Detect patterns
        hook = PatternHook(atr=5.0)
        swings = [
            _make_swing(2050.0, "high", index=10),
            _make_swing(2030.0, "low", index=20),
            _make_swing(2055.0, "high", index=30),
            _make_swing(2025.0, "low", index=35),
            _make_swing(2041.0, "high", index=45),
            _make_swing(2020.0, "low", index=55),
        ]
        patterns = hook.detect(swings)

        # 2. Select trades
        selector = ScreenshotSelector(n_worst=2, n_best=2)
        trades = [
            {
                "r_multiple": -1.5, "entry_bar_idx": 20, "exit_bar_idx": 50,
                "entry_price": 2050.0, "exit_price": 2042.5, "stop_loss": 2045.0,
                "direction": "long", "_selection_reason": "worst_loser",
                "entry_time": datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
                "exit_time": datetime(2026, 1, 15, 14, 0, tzinfo=timezone.utc),
            },
            {
                "r_multiple": 2.0, "entry_bar_idx": 30, "exit_bar_idx": 55,
                "entry_price": 2030.0, "exit_price": 2040.0, "stop_loss": 2025.0,
                "direction": "long", "_selection_reason": "best_winner",
                "entry_time": datetime(2026, 1, 16, 10, 0, tzinfo=timezone.utc),
                "exit_time": datetime(2026, 1, 16, 14, 0, tzinfo=timezone.utc),
            },
        ]
        selected = selector.select_trades(trades)
        assert len(selected) >= 1

        # 3. Build prompts
        for trade in selected:
            prompt = build_visual_prompt(trade, strategy_name="sss", patterns=patterns)
            assert len(prompt) > 50

        # 4. Get confluence
        adj = hook.get_confluence_adjustment(patterns, "long")
        assert isinstance(adj, int)
