"""End-to-end integration test for The5ers 2-Step pipeline."""
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta


def _make_trending_gold_data(days: int = 60) -> pd.DataFrame:
    """Generate synthetic 1M gold data with trend + mean reversion."""
    np.random.seed(42)
    n_bars = days * 24 * 60
    prices = [2000.0]
    for i in range(1, n_bars):
        hour = (i // 60) % 24
        volatility = 0.3 if 6 <= hour < 14 else 0.1
        trend = 0.001 if (i // (24 * 60)) % 10 < 7 else -0.001
        prices.append(prices[-1] + np.random.normal(trend, volatility))
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="1min", tz="UTC")
    df = pd.DataFrame({
        "open": prices,
        "high": [p + abs(np.random.normal(0, 0.5)) for p in prices],
        "low": [p - abs(np.random.normal(0, 0.5)) for p in prices],
        "close": prices,
        "volume": np.random.randint(100, 1000, n_bars),
    }, index=dates)
    df.index.name = "timestamp"
    return df


class TestChallengeSimulatorIntegration:
    def test_challenge_simulation_returns_valid_result(self):
        from src.backtesting.challenge_simulator import ChallengeSimulator
        np.random.seed(42)
        trades = []
        for i in range(200):
            r = 2.0 if np.random.random() < 0.55 else -1.0
            trades.append({"r_multiple": r, "risk_pct": 1.5, "day_index": i // 3})
        sim = ChallengeSimulator()
        result = sim.run(trades, total_trading_days=200)
        assert 0.0 <= result.pass_rate <= 1.0
        assert result.total_windows > 0

    def test_multi_phase_tracker_full_progression(self):
        from src.backtesting.metrics import MultiPhasePropFirmTracker
        tracker = MultiPhasePropFirmTracker()
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        tracker.initialise(10_000.0, ts)
        balance = 10_000.0
        for day in range(20):
            balance += 50.0
            tracker.update(ts + timedelta(days=day), balance)
        status = tracker.get_status()
        assert status["phase_1_passed"] is True

    def test_signal_blender_selects_highest(self):
        from src.strategy.signal_blender import SignalBlender
        from src.strategy.signal_engine import Signal
        blender = SignalBlender()
        sig_a = Signal(
            timestamp=datetime(2024, 1, 1, 8, tzinfo=timezone.utc),
            instrument="XAUUSD", direction="long", entry_price=2050.0,
            stop_loss=2040.0, take_profit=2070.0, confluence_score=3,
            quality_tier="C", atr=5.0, reasoning={"strategy": "ichi"},
        )
        sig_b = Signal(
            timestamp=datetime(2024, 1, 1, 8, tzinfo=timezone.utc),
            instrument="XAUUSD", direction="long", entry_price=2050.0,
            stop_loss=2040.0, take_profit=2070.0, confluence_score=7,
            quality_tier="A+", atr=5.0, reasoning={"strategy": "asian"},
        )
        result = blender.select([sig_a, sig_b])
        assert result.reasoning["strategy"] == "asian"
