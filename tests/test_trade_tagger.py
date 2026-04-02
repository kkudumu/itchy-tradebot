# tests/test_trade_tagger.py
"""Tests for trade regime/event tagging."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone
from unittest.mock import patch


def _make_trades(n=10, seed=42):
    """Generate trades with timestamps."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-06", periods=n, freq="B", tz="UTC")
    trades = []
    for i in range(n):
        trades.append({
            "r_multiple": float(rng.choice([-1.0, 1.5])),
            "entry_time": dates[i].to_pydatetime(),
            "context": {
                "adx_value": float(rng.uniform(10, 40)),
                "session": "london",
            },
        })
    return trades


def _make_regime_series(dates, seed=42):
    """Generate a mock regime series."""
    rng = np.random.default_rng(seed)
    labels = ["risk_on", "risk_off", "dollar_driven", "inflation_fear", "mixed"]
    return pd.Series(
        rng.choice(labels, len(dates)),
        index=dates,
        name="regime",
    )


class TestTradeTagger:
    def test_tags_trades_with_regime(self):
        from src.macro.trade_tagger import TradeTagger

        trades = _make_trades()
        dates = pd.date_range("2025-01-01", periods=30, freq="B", tz="UTC")
        regimes = _make_regime_series(dates)

        tagger = TradeTagger(regime_series=regimes)
        tagged = tagger.tag_trades(trades)

        assert len(tagged) == len(trades)
        for trade in tagged:
            assert "regime" in trade
            assert trade["regime"] in [
                "risk_on", "risk_off", "dollar_driven",
                "inflation_fear", "mixed",
            ]

    def test_tags_trades_with_event_proximity(self):
        from src.macro.trade_tagger import TradeTagger

        # Trade on NFP day (Jan 3, 2025 is first Friday)
        trades = [{
            "r_multiple": 1.5,
            "entry_time": datetime(2025, 1, 3, 12, 0, tzinfo=timezone.utc),
            "context": {"session": "london"},
        }]

        dates = pd.date_range("2025-01-01", periods=10, freq="B", tz="UTC")
        regimes = _make_regime_series(dates)

        tagger = TradeTagger(regime_series=regimes)
        tagged = tagger.tag_trades(trades, hours_before=4, hours_after=2)

        assert tagged[0]["near_event"] is True
        assert "NFP" in tagged[0].get("nearest_event", "") or \
               "Non-Farm" in tagged[0].get("nearest_event", "")

    def test_quiet_day_not_near_event(self):
        from src.macro.trade_tagger import TradeTagger

        trades = [{
            "r_multiple": -1.0,
            "entry_time": datetime(2025, 2, 18, 10, 0, tzinfo=timezone.utc),
            "context": {"session": "london"},
        }]

        dates = pd.date_range("2025-01-01", periods=60, freq="B", tz="UTC")
        regimes = _make_regime_series(dates)

        tagger = TradeTagger(regime_series=regimes)
        tagged = tagger.tag_trades(trades)

        assert tagged[0]["near_event"] is False

    def test_preserves_existing_trade_fields(self):
        from src.macro.trade_tagger import TradeTagger

        trades = [{
            "r_multiple": 2.0,
            "entry_time": datetime(2025, 1, 6, 10, 0, tzinfo=timezone.utc),
            "context": {"adx_value": 35.0},
            "custom_field": "preserved",
        }]

        dates = pd.date_range("2025-01-01", periods=30, freq="B", tz="UTC")
        regimes = _make_regime_series(dates)

        tagger = TradeTagger(regime_series=regimes)
        tagged = tagger.tag_trades(trades)

        assert tagged[0]["custom_field"] == "preserved"
        assert tagged[0]["r_multiple"] == 2.0

    def test_regime_stats(self):
        from src.macro.trade_tagger import TradeTagger

        trades = _make_trades(n=20)
        dates = pd.date_range("2025-01-01", periods=30, freq="B", tz="UTC")
        regimes = _make_regime_series(dates)

        tagger = TradeTagger(regime_series=regimes)
        tagged = tagger.tag_trades(trades)
        stats = tagger.compute_regime_stats(tagged)

        assert isinstance(stats, dict)
        # At least one regime should appear
        assert len(stats) > 0
        for regime, info in stats.items():
            assert "count" in info
            assert "win_rate" in info
            assert "avg_r" in info
