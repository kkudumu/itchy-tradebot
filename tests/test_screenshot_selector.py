"""Tests for selective trade screenshot capture."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest


def _make_trade_dict(r_multiple: float, entry_bar: int = 50, exit_bar: int = 80, **kwargs) -> dict:
    """Build a minimal trade dict for testing."""
    base = {
        "r_multiple": r_multiple,
        "entry_bar_idx": entry_bar,
        "exit_bar_idx": exit_bar,
        "entry_price": 2050.0,
        "exit_price": 2050.0 + r_multiple * 5.0,
        "stop_loss": 2045.0,
        "direction": "long",
        "entry_time": datetime(2026, 1, 15, 10, 0, tzinfo=timezone.utc),
        "exit_time": datetime(2026, 1, 15, 14, 0, tzinfo=timezone.utc),
    }
    base.update(kwargs)
    return base


class TestTradeSelection:
    def test_selects_worst_losers(self):
        from src.discovery.screenshot_selector import ScreenshotSelector

        selector = ScreenshotSelector(n_worst=3, n_best=2, near_sl_threshold=0.2)
        trades = [
            _make_trade_dict(-2.0),
            _make_trade_dict(-1.5),
            _make_trade_dict(-1.0),
            _make_trade_dict(-0.5),
            _make_trade_dict(1.0),
            _make_trade_dict(1.5),
            _make_trade_dict(2.0),
        ]
        selected = selector.select_trades(trades)

        worst = [t for t in selected if t["_selection_reason"] == "worst_loser"]
        assert len(worst) == 3
        # Should be the 3 worst by R
        worst_rs = [t["r_multiple"] for t in worst]
        assert sorted(worst_rs) == [-2.0, -1.5, -1.0]

    def test_selects_best_winners(self):
        from src.discovery.screenshot_selector import ScreenshotSelector

        selector = ScreenshotSelector(n_worst=2, n_best=3)
        trades = [
            _make_trade_dict(-1.0),
            _make_trade_dict(0.5),
            _make_trade_dict(1.0),
            _make_trade_dict(1.5),
            _make_trade_dict(2.5),
        ]
        selected = selector.select_trades(trades)

        best = [t for t in selected if t["_selection_reason"] == "best_winner"]
        assert len(best) == 3
        best_rs = [t["r_multiple"] for t in best]
        assert sorted(best_rs) == [1.0, 1.5, 2.5]

    def test_selects_near_sl_exits(self):
        from src.discovery.screenshot_selector import ScreenshotSelector

        selector = ScreenshotSelector(n_worst=1, n_best=1, near_sl_threshold=0.3)
        # Near-SL: trade that hit within 0.3R of the stop before winning
        trades = [
            _make_trade_dict(-1.0),
            _make_trade_dict(1.0, min_r_during_trade=-0.85),  # near SL
            _make_trade_dict(2.0),
        ]
        selected = selector.select_trades(trades)

        near_sl = [t for t in selected if t.get("_selection_reason") == "near_sl_exit"]
        # min_r_during_trade is -0.85, threshold is -1.0 + 0.3 = -0.7. -0.85 < -0.7 so this qualifies
        assert len(near_sl) >= 1

    def test_no_duplicates_in_selection(self):
        from src.discovery.screenshot_selector import ScreenshotSelector

        selector = ScreenshotSelector(n_worst=5, n_best=5)
        # Only 3 trades -- no duplication
        trades = [
            _make_trade_dict(-1.0),
            _make_trade_dict(0.5),
            _make_trade_dict(2.0),
        ]
        selected = selector.select_trades(trades)
        assert len(selected) == 3  # all selected, no duplicates

    def test_empty_trades_returns_empty(self):
        from src.discovery.screenshot_selector import ScreenshotSelector

        selector = ScreenshotSelector()
        assert selector.select_trades([]) == []
