# tests/test_macro_integration.py
"""Integration test: full macro pipeline from FX data to tagged trades."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock


class TestMacroPipelineIntegration:
    def _make_fx_data(self, n=20, seed=42):
        """Generate synthetic daily FX data for all 6 pairs."""
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2025-01-06", periods=n, freq="B", tz="UTC")
        return {
            "EURUSD": pd.DataFrame({"close": 1.08 + np.cumsum(rng.normal(0, 0.002, n))}, index=dates),
            "USDJPY": pd.DataFrame({"close": 150.0 + np.cumsum(rng.normal(0, 0.5, n))}, index=dates),
            "GBPUSD": pd.DataFrame({"close": 1.265 + np.cumsum(rng.normal(0, 0.002, n))}, index=dates),
            "USDCAD": pd.DataFrame({"close": 1.36 + np.cumsum(rng.normal(0, 0.003, n))}, index=dates),
            "USDSEK": pd.DataFrame({"close": 10.5 + np.cumsum(rng.normal(0, 0.05, n))}, index=dates),
            "USDCHF": pd.DataFrame({"close": 0.88 + np.cumsum(rng.normal(0, 0.002, n))}, index=dates),
        }

    def _make_trades(self, dates, n=10, seed=42):
        """Generate trades with entry times matching given dates."""
        rng = np.random.default_rng(seed)
        trades = []
        for i in range(min(n, len(dates))):
            trades.append({
                "r_multiple": float(rng.choice([-1.0, 1.5, 2.0, -0.5])),
                "entry_time": dates[i].to_pydatetime().replace(hour=10, minute=0),
                "context": {
                    "adx_value": float(rng.uniform(15, 40)),
                    "atr_value": float(rng.uniform(3, 8)),
                    "session": "london",
                },
            })
        return trades

    def test_dxy_to_regime_to_tagged_trades(self):
        """Full pipeline: FX data -> DXY -> regime -> tagged trades."""
        from src.macro.dxy_synthesizer import compute_dxy_series
        from src.macro.regime_classifier import RegimeClassifier
        from src.macro.trade_tagger import TradeTagger

        fx_data = self._make_fx_data(n=20)
        dates = list(fx_data["EURUSD"].index)

        # Step 1: Compute DXY
        dxy = compute_dxy_series(fx_data)
        assert len(dxy) == 20
        assert 90 < dxy.mean() < 120  # Reasonable DXY range

        # Step 2: Build mock macro panel (skip yfinance in test)
        panel = pd.DataFrame({
            "dxy_close": dxy,
            "dxy_pct_change": dxy.pct_change(),
            "spx_close": 4800.0 + np.cumsum(np.random.default_rng(42).normal(0, 20, 20)),
            "spx_pct_change": np.random.default_rng(42).normal(0, 0.8, 20),
            "us10y_close": 4.25 + np.cumsum(np.random.default_rng(42).normal(0, 0.05, 20)),
            "us10y_pct_change": np.random.default_rng(42).normal(0, 1.0, 20),
        }, index=dxy.index)

        # Step 3: Classify regimes
        classifier = RegimeClassifier()
        regimes = classifier.classify(panel)
        assert len(regimes) == 20
        assert all(r in ["risk_on", "risk_off", "dollar_driven", "inflation_fear", "mixed"]
                   for r in regimes)

        # Step 4: Tag trades
        trades = self._make_trades(pd.DatetimeIndex(dates), n=10)
        tagger = TradeTagger(regime_series=regimes)
        tagged = tagger.tag_trades(trades)

        assert len(tagged) == 10
        for trade in tagged:
            assert "regime" in trade
            assert "near_event" in trade
            assert isinstance(trade["near_event"], bool)

        # Step 5: Compute regime stats
        stats = tagger.compute_regime_stats(tagged)
        assert isinstance(stats, dict)
        total = sum(s["count"] for s in stats.values())
        assert total == 10

    def test_feature_vector_receives_macro_data(self):
        """FeatureVectorBuilder correctly encodes macro context."""
        from src.learning.feature_vector import FeatureVectorBuilder

        builder = FeatureVectorBuilder()
        context = {
            "adx_value": 30.0,
            "atr_value": 5.0,
            "session": "london",
            "dxy_pct_change": 0.8,
            "spx_pct_change": -1.2,
            "us10y_pct_change": 2.5,
            "macro_regime": "risk_off",
            "hours_to_event": 3.0,
        }
        vec = builder.build(context)

        assert len(vec) == 64
        assert vec[59] > 0.5   # Positive DXY change
        assert vec[60] < 0.5   # Negative SPX change
        assert vec[61] > 0.5   # Positive yield change
        assert 0.2 <= vec[62] <= 0.3  # risk_off ordinal
        assert vec[63] > 0.0   # Near-ish event

    def test_econ_calendar_consistency(self):
        """Calendar events are consistent and properly dated."""
        from src.macro.econ_calendar import EconCalendar

        cal = EconCalendar()
        events_2025 = cal.get_all_events(2025)

        # 12 NFP + 8 FOMC + 12 CPI = 32 events
        assert len(events_2025) == 32

        # All events should be in 2025
        for event in events_2025:
            assert event.timestamp.year == 2025
            assert event.impact == "red"

        # Events should be chronologically sorted
        timestamps = [e.timestamp for e in events_2025]
        assert timestamps == sorted(timestamps)

    def test_event_proximity_filter_integration(self):
        """EventProximityFilter works when registered in EdgeManager."""
        from src.edges.base import EdgeContext
        from src.macro.event_proximity import EventProximityFilter

        config = {
            "enabled": True,
            "params": {"hours_before": 4, "hours_after": 2},
        }
        edge = EventProximityFilter(config)

        # NFP day: Jan 3, 2025, 13:30 UTC
        ctx_nfp = EdgeContext(
            timestamp=datetime(2025, 1, 3, 12, 0, tzinfo=timezone.utc),
            day_of_week=4,
            close_price=2650.0,
            high_price=2655.0,
            low_price=2645.0,
            spread=0.30,
            session="london",
            adx=30.0,
            atr=5.0,
        )
        result = edge.should_allow(ctx_nfp)
        assert result.allowed is False

        # Quiet day
        ctx_quiet = EdgeContext(
            timestamp=datetime(2025, 2, 18, 10, 0, tzinfo=timezone.utc),
            day_of_week=1,
            close_price=2650.0,
            high_price=2655.0,
            low_price=2645.0,
            spread=0.30,
            session="london",
            adx=30.0,
            atr=5.0,
        )
        result = edge.should_allow(ctx_quiet)
        assert result.allowed is True
