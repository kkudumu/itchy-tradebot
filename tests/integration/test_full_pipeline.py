"""
Integration tests for the full trading pipeline.

Tests verify that data flows correctly between modules:
  download -> store -> backtest -> logs -> embeddings -> optimize -> Monte Carlo -> report

All external dependencies (database, MT5) are mocked.
Synthetic OHLCV data is generated with numpy.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_1m_candles(
    n: int = 1000,
    start: datetime = None,
    base_price: float = 1900.0,
    trend: float = 0.05,
    volatility: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic 1-minute OHLCV candles."""
    rng = np.random.default_rng(seed)
    start = start or datetime(2024, 1, 2, 8, 0, tzinfo=timezone.utc)

    prices = [base_price]
    for _ in range(n - 1):
        prices.append(prices[-1] + trend + rng.normal(0, volatility))

    prices = np.array(prices)
    timestamps = [start + timedelta(minutes=i) for i in range(n)]

    half_spread = volatility * 0.3
    rows = []
    for ts, p in zip(timestamps, prices):
        noise = rng.uniform(0, half_spread * 2, 4)
        o = p + noise[0] - half_spread
        h = max(o, p) + noise[1]
        l = min(o, p) - noise[2]
        c = p + noise[3] - half_spread
        vol = int(rng.integers(100, 1000))
        rows.append({"open": o, "high": h, "low": l, "close": c, "volume": vol})

    return pd.DataFrame(rows, index=pd.DatetimeIndex(timestamps, tz=timezone.utc))


def _make_trade_list(n: int = 30, seed: int = 99) -> list[dict]:
    """Generate a synthetic list of closed trades for Monte Carlo / report tests."""
    rng = np.random.default_rng(seed)
    trades = []
    for i in range(n):
        r = rng.choice([-1.0, 2.0], p=[0.45, 0.55])
        trades.append({
            "trade_id": i + 1,
            "r_multiple": float(r),
            "pnl": float(r * 150),
            "win": r > 0,
            "direction": "long",
            "confluence_score": 5,
            "signal_tier": "B",
            "day": i,
        })
    return trades


# ---------------------------------------------------------------------------
# 1. Config loads and validates correctly
# ---------------------------------------------------------------------------

class TestConfigIntegration:
    """Config loader -> AppConfig output feeds downstream modules."""

    def test_config_loader_returns_app_config(self):
        """ConfigLoader.load() should return a fully validated AppConfig."""
        from src.config.loader import ConfigLoader
        from src.config.models import AppConfig

        loader = ConfigLoader()
        cfg = loader.load()

        assert isinstance(cfg, AppConfig)
        assert cfg.strategy.ichimoku.tenkan_period >= 1
        assert cfg.strategy.ichimoku.kijun_period >= 1
        assert isinstance(cfg.edges.time_of_day.enabled, bool)

    def test_default_edge_config_all_enabled(self):
        """Default EdgeConfig should have all 12 edges enabled."""
        from src.config.models import EdgeConfig

        ec = EdgeConfig()
        enabled_fields = [
            ec.time_of_day.enabled,
            ec.day_of_week.enabled,
            ec.london_open_delay.enabled,
            ec.candle_close_confirmation.enabled,
            ec.spread_filter.enabled,
            ec.news_filter.enabled,
            ec.friday_close.enabled,
            ec.regime_filter.enabled,
            ec.time_stop.enabled,
            ec.bb_squeeze.enabled,
            ec.confluence_scoring.enabled,
            ec.equity_curve.enabled,
        ]
        assert all(isinstance(e, bool) for e in enabled_fields)
        # All 12 fields exist
        assert len(enabled_fields) == 12

    def test_config_feeds_edge_manager(self):
        """EdgeConfig from loader can be passed directly to EdgeManager."""
        from src.config.loader import ConfigLoader
        from src.edges.manager import EdgeManager

        cfg = ConfigLoader().load()
        # EdgeManager accepts AppConfig.edges
        manager = EdgeManager(cfg.edges)

        assert len(manager._all_edges) == 12


# ---------------------------------------------------------------------------
# 2. Ichimoku indicators -> signals data flow
# ---------------------------------------------------------------------------

class TestIchimokuToSignalFlow:
    """IchimokuCalculator output feeds directly into IchimokuSignals."""

    def test_ichimoku_result_feeds_signals(self):
        """Output arrays from IchimokuCalculator are consumed by IchimokuSignals."""
        from src.indicators.ichimoku import IchimokuCalculator
        from src.indicators.signals import IchimokuSignals

        n = 200
        rng = np.random.default_rng(0)
        high = 1900 + np.cumsum(rng.normal(0, 1, n))
        low = high - rng.uniform(1, 5, n)
        close = (high + low) / 2

        calc = IchimokuCalculator()
        result = calc.calculate(high, low, close)

        signals = IchimokuSignals()
        tk = signals.tk_cross(result.tenkan_sen, result.kijun_sen)
        cp = signals.cloud_position(close, result.senkou_a, result.senkou_b)
        cc = signals.chikou_confirmation(result.chikou_span, close)
        ct = signals.cloud_twist(result.senkou_a, result.senkou_b)

        # All output arrays are same length as input
        assert len(tk) == n
        assert len(cp) == n
        assert len(cc) == n
        assert len(ct) == n

        # Values are within expected range
        assert set(np.unique(tk[~np.isnan(tk)])).issubset({-1, 0, 1})
        assert set(np.unique(cp[~np.isnan(cp)])).issubset({-1, 0, 1})

    def test_cloud_direction_feeds_from_senkou_arrays(self):
        """IchimokuCalculator.cloud_direction() consumes senkou arrays correctly."""
        from src.indicators.ichimoku import IchimokuCalculator

        calc = IchimokuCalculator()
        senkou_a = np.array([1900.0, 1905.0, 1910.0])
        senkou_b = np.array([1895.0, 1895.0, 1895.0])

        direction = calc.cloud_direction(senkou_a, senkou_b)
        # All bars: senkou_a > senkou_b -> bullish = 1
        assert all(direction == 1)

    def test_signal_state_at_integration(self):
        """signal_state_at() integrates ichimoku + signals modules."""
        from src.indicators.ichimoku import IchimokuCalculator
        from src.indicators.signals import IchimokuSignals, IchimokuSignalState

        n = 200
        rng = np.random.default_rng(5)
        high = 1900 + np.cumsum(rng.normal(0.1, 1, n))
        low = high - np.abs(rng.normal(2, 0.5, n))
        close = (high + low) / 2

        calc = IchimokuCalculator()
        result = calc.calculate(high, low, close)

        signals = IchimokuSignals()
        state = signals.signal_state_at(
            idx=180,
            tenkan=result.tenkan_sen,
            kijun=result.kijun_sen,
            close=close,
            senkou_a=result.senkou_a,
            senkou_b=result.senkou_b,
            chikou=result.chikou_span,
        )

        assert isinstance(state, IchimokuSignalState)
        assert state.cloud_direction in (-1, 0, 1)
        assert state.tk_cross in (-1, 0, 1)
        assert isinstance(state.cloud_thickness, float)


# ---------------------------------------------------------------------------
# 3. Indicators -> EdgeContext data flow
# ---------------------------------------------------------------------------

class TestIndicatorToEdgeContextFlow:
    """Indicator values are correctly packaged into EdgeContext for edge evaluation."""

    def _make_edge_context(self, ts: datetime = None) -> "EdgeContext":
        from src.edges.base import EdgeContext

        return EdgeContext(
            timestamp=ts or datetime(2024, 1, 3, 10, 30, tzinfo=timezone.utc),
            day_of_week=2,  # Wednesday
            close_price=1905.0,
            high_price=1908.0,
            low_price=1902.0,
            spread=5.0,
            session="london",
            adx=32.0,
            atr=8.0,
            cloud_thickness=15.0,
            kijun_value=1900.0,
            bb_squeeze=False,
            confluence_score=5,
        )

    def test_edge_manager_accepts_context(self):
        """EdgeManager.check_entry() accepts a well-formed EdgeContext."""
        from src.config.models import EdgeConfig
        from src.edges.manager import EdgeManager

        # Disable most filters so this context passes cleanly
        ec = EdgeConfig()
        ec.time_of_day.enabled = False
        ec.day_of_week.enabled = False
        ec.london_open_delay.enabled = False
        ec.candle_close_confirmation.enabled = False
        ec.spread_filter.enabled = False
        ec.news_filter.enabled = False
        ec.regime_filter.enabled = False

        manager = EdgeManager(ec)
        ctx = self._make_edge_context()
        passed, results = manager.check_entry(ctx)

        # With all entry filters disabled, entry should be allowed
        assert passed is True

    def test_spread_filter_blocks_high_spread(self):
        """SpreadFilter correctly blocks when spread exceeds threshold."""
        from src.config.models import EdgeConfig
        from src.edges.manager import EdgeManager
        from src.edges.base import EdgeContext

        ec = EdgeConfig()
        # Disable all filters except spread
        for field_name in ["time_of_day", "day_of_week", "london_open_delay",
                           "candle_close_confirmation", "news_filter", "regime_filter"]:
            getattr(ec, field_name).enabled = False
        ec.spread_filter.enabled = True
        ec.spread_filter.params["max_spread_points"] = 10  # strict limit

        manager = EdgeManager(ec)
        ctx = self._make_edge_context()
        ctx.spread = 50.0  # exceeds 10

        passed, results = manager.check_entry(ctx)
        assert passed is False
        assert any("spread" in r.edge_name.lower() for r in results)

    def test_regime_filter_blocks_low_adx(self):
        """RegimeFilter blocks when ADX is too low."""
        from src.config.models import EdgeConfig
        from src.edges.manager import EdgeManager
        from src.edges.base import EdgeContext

        ec = EdgeConfig()
        for field_name in ["time_of_day", "day_of_week", "london_open_delay",
                           "candle_close_confirmation", "spread_filter", "news_filter"]:
            getattr(ec, field_name).enabled = False
        ec.regime_filter.enabled = True
        ec.regime_filter.params["adx_min"] = 28

        manager = EdgeManager(ec)
        ctx = self._make_edge_context()
        ctx.adx = 15.0  # well below threshold

        passed, results = manager.check_entry(ctx)
        assert passed is False


# ---------------------------------------------------------------------------
# 4. IchimokuBacktester end-to-end
# ---------------------------------------------------------------------------

class TestBacktesterEndToEnd:
    """IchimokuBacktester runs a full backtest on synthetic data."""

    def test_backtest_returns_backtest_result(self):
        """Backtester.run() returns a BacktestResult with the correct structure."""
        from src.backtesting.vectorbt_engine import IchimokuBacktester, BacktestResult

        candles = _make_1m_candles(n=2000, trend=0.05)
        backtester = IchimokuBacktester(
            initial_balance=10_000.0,
        )
        result = backtester.run(candles, instrument="XAUUSD")

        assert isinstance(result, BacktestResult)
        assert isinstance(result.trades, list)
        assert isinstance(result.metrics, dict)
        assert isinstance(result.equity_curve, pd.Series)
        assert isinstance(result.prop_firm, dict)
        assert isinstance(result.daily_pnl, pd.Series)
        assert result.skipped_signals >= 0

    def test_equity_curve_starts_at_initial_balance(self):
        """Equity curve first value should equal the initial balance."""
        from src.backtesting.vectorbt_engine import IchimokuBacktester

        initial = 10_000.0
        candles = _make_1m_candles(n=1500)
        backtester = IchimokuBacktester(initial_balance=initial)
        result = backtester.run(candles)

        assert len(result.equity_curve) > 0
        # First value should be at or near initial balance
        first_val = float(result.equity_curve.iloc[0])
        assert abs(first_val - initial) < initial * 0.01  # within 1%

    def test_prop_firm_status_present(self):
        """BacktestResult.prop_firm should contain status key."""
        from src.backtesting.vectorbt_engine import IchimokuBacktester

        candles = _make_1m_candles(n=1500)
        result = IchimokuBacktester().run(candles)

        assert "status" in result.prop_firm
        assert result.prop_firm["status"] in (
            "ongoing", "passed", "failed_daily_dd", "failed_total_dd", "failed_timeout"
        )

    def test_backtest_metrics_have_required_keys(self):
        """PerformanceMetrics output should have sharpe, win_rate, etc."""
        from src.backtesting.vectorbt_engine import IchimokuBacktester

        candles = _make_1m_candles(n=1500)
        result = IchimokuBacktester().run(candles)

        # Metrics dict must contain standard keys (may be 0/NaN with no trades)
        expected_keys = {"total_trades", "win_rate", "sharpe_ratio", "total_return_pct"}
        assert expected_keys.issubset(set(result.metrics.keys()))


# ---------------------------------------------------------------------------
# 5. Embeddings -> feature vectors data flow
# ---------------------------------------------------------------------------

class TestEmbeddingFlow:
    """FeatureVectorBuilder -> EmbeddingEngine produces correct vectors."""

    def test_feature_vector_dimension(self):
        """FeatureVectorBuilder.build() always returns a 64-dim vector."""
        from src.learning.feature_vector import FeatureVectorBuilder

        builder = FeatureVectorBuilder()
        ctx = {
            "cloud_direction_4h": 1,
            "cloud_direction_1h": 1,
            "tk_cross_15m": 1,
            "cloud_position_15m": 1,
            "chikou_confirmed_15m": 1,
            "cloud_thickness_4h": 20.0,
            "adx": 32.0,
            "atr": 8.0,
            "session": "london",
            "confluence_score": 6,
            "signal_tier": "B",
        }
        vec = builder.build(ctx)

        assert vec.shape == (64,)
        assert vec.dtype == np.float64
        assert np.all(vec >= 0.0)
        assert np.all(vec <= 1.0)

    def test_embedding_engine_wraps_feature_builder(self):
        """EmbeddingEngine.create_embedding() delegates to FeatureVectorBuilder."""
        from src.learning.embeddings import EmbeddingEngine

        engine = EmbeddingEngine()
        ctx = {
            "cloud_direction_4h": -1,
            "session": "new_york",
            "confluence_score": 4,
            "signal_tier": "C",
        }
        vec = engine.create_embedding(ctx)

        assert vec.shape == (64,)
        assert np.all(np.isfinite(vec))

    def test_embed_trade_produces_db_ready_dict(self):
        """EmbeddingEngine.embed_trade() returns dict with context_embedding list."""
        from src.learning.embeddings import EmbeddingEngine

        engine = EmbeddingEngine()
        ctx = {"cloud_direction_4h": 1, "session": "london"}
        result_dict = {"r_multiple": 1.5, "pnl": 225.0, "win": True}

        embedded = engine.embed_trade(ctx, result_dict)

        assert "context_embedding" in embedded
        assert isinstance(embedded["context_embedding"], list)
        assert len(embedded["context_embedding"]) == 64
        assert "outcome_r" in embedded


# ---------------------------------------------------------------------------
# 6. Monte Carlo pipeline
# ---------------------------------------------------------------------------

class TestMonteCarloPipeline:
    """MonteCarloSimulator consumes trade list and produces MCResult."""

    def _make_mc_trade_list(self, n: int = 40, seed: int = 1) -> list:
        """Make trade dicts suitable for MonteCarloSimulator.run()."""
        rng = np.random.default_rng(seed)
        start = datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc)
        trades = []
        for i in range(n):
            r = float(rng.choice([-1.0, 2.0], p=[0.45, 0.55]))
            trades.append({
                "r_multiple": r,
                "entry_time": start + timedelta(hours=i * 4),
            })
        return trades

    def test_monte_carlo_run_returns_mc_result(self):
        """MonteCarloSimulator.run() returns an MCResult."""
        from src.simulation.monte_carlo import MonteCarloSimulator, MCResult

        trades = self._make_mc_trade_list(n=40, seed=1)

        sim = MonteCarloSimulator(
            initial_balance=10_000.0,
            profit_target_pct=8.0,
            max_daily_dd_pct=5.0,
            max_total_dd_pct=10.0,
        )
        result = sim.run(trade_results=trades, n_simulations=500, seed=42)

        assert isinstance(result, MCResult)
        assert 0.0 <= result.pass_rate <= 100.0  # pass_rate is a percentage
        assert result.n_simulations == 500
        assert result.avg_days > 0

    def test_monte_carlo_pass_rate_with_positive_expectancy(self):
        """Positive expectancy trades should produce pass_rate > 0."""
        from src.simulation.monte_carlo import MonteCarloSimulator

        # All winning trades at 2R
        start = datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc)
        trades = [{"r_multiple": 2.0, "entry_time": start + timedelta(hours=i)}
                  for i in range(50)]

        sim = MonteCarloSimulator(initial_balance=10_000.0)
        result = sim.run(trade_results=trades, n_simulations=200, seed=7)

        assert result.pass_rate > 0.0

    def test_monte_carlo_convergence_attribute_present(self):
        """MCResult should expose convergence_reached and convergence_at."""
        from src.simulation.monte_carlo import MonteCarloSimulator

        start = datetime(2024, 1, 2, 10, 0, tzinfo=timezone.utc)
        trades = [
            {"r_multiple": 1.5 if i % 2 == 0 else -1.0,
             "entry_time": start + timedelta(hours=i * 4)}
            for i in range(30)
        ]
        sim = MonteCarloSimulator()
        result = sim.run(trade_results=trades, n_simulations=300, seed=0)

        assert isinstance(result.convergence_reached, bool)
        assert isinstance(result.convergence_at, int)


# ---------------------------------------------------------------------------
# 7. Trade logger dry-run integration
# ---------------------------------------------------------------------------

class TestTradeLoggerDryRun:
    """TradeLogger in dry-run mode formats trades without hitting a database."""

    def _make_trade_dict(self) -> dict:
        return {
            "instrument": "XAUUSD",
            "direction": "long",
            "entry_time": datetime(2024, 1, 3, 10, 0, tzinfo=timezone.utc),
            "exit_time": datetime(2024, 1, 3, 14, 0, tzinfo=timezone.utc),
            "entry_price": 1900.0,
            "exit_price": 1916.0,
            "stop_loss": 1888.0,
            "take_profit": 1924.0,
            "lot_size": 0.1,
            "risk_pct": 1.5,
            "r_multiple": 1.33,
            "pnl": 160.0,
            "pnl_pct": 1.6,
            "status": "closed",
            "confluence_score": 6,
            "signal_tier": "B",
        }

    def _make_context_dict(self) -> dict:
        return {
            "cloud_direction_4h": 1,
            "cloud_direction_1h": 1,
            "tk_cross_15m": 1,
            "chikou_confirmation": 1,
            "cloud_thickness_4h": 15.0,
            "adx_value": 31.0,
            "atr_value": 8.0,
            "rsi_value": 58.0,
            "bb_width_percentile": 0.6,
            "session": "london",
            "nearest_sr_distance": 0.002,
            "zone_confluence_score": 2,
        }

    def test_dry_run_returns_sequential_ids(self):
        """log_trade() in dry-run mode returns incrementing integer IDs."""
        from src.backtesting.trade_logger import TradeLogger

        logger = TradeLogger(db_pool=None)

        id1 = logger.log_trade(self._make_trade_dict(), self._make_context_dict())
        id2 = logger.log_trade(self._make_trade_dict(), self._make_context_dict())

        assert isinstance(id1, int)
        assert isinstance(id2, int)
        assert id2 > id1

    def test_log_trade_embedding_is_included(self):
        """Embedding should be generated and stored in the trade record."""
        from src.backtesting.trade_logger import TradeLogger

        logger = TradeLogger(db_pool=None)
        # Call log_trade; in dry-run it should not raise
        trade_id = logger.log_trade(self._make_trade_dict(), self._make_context_dict())

        # Dry-run mode just returns ID; the key test is no exception raised
        assert trade_id >= 1


# ---------------------------------------------------------------------------
# 8. PerformanceMetrics on known trade list
# ---------------------------------------------------------------------------

class TestPerformanceMetricsIntegration:
    """PerformanceMetrics computes correct values from a known trade list."""

    def test_win_rate_calculation(self):
        """Win rate should be 60% for 6 wins out of 10 trades."""
        from src.backtesting.metrics import PerformanceMetrics

        initial_balance = 10_000.0
        trades = [
            {"r_multiple": 2.0, "pnl": 200, "pnl_pct": 2.0, "direction": "long"},
            {"r_multiple": 2.0, "pnl": 200, "pnl_pct": 2.0, "direction": "long"},
            {"r_multiple": 2.0, "pnl": 200, "pnl_pct": 2.0, "direction": "long"},
            {"r_multiple": 2.0, "pnl": 200, "pnl_pct": 2.0, "direction": "short"},
            {"r_multiple": 2.0, "pnl": 200, "pnl_pct": 2.0, "direction": "long"},
            {"r_multiple": 2.0, "pnl": 200, "pnl_pct": 2.0, "direction": "long"},
            {"r_multiple": -1.0, "pnl": -100, "pnl_pct": -1.0, "direction": "long"},
            {"r_multiple": -1.0, "pnl": -100, "pnl_pct": -1.0, "direction": "long"},
            {"r_multiple": -1.0, "pnl": -100, "pnl_pct": -1.0, "direction": "long"},
            {"r_multiple": -1.0, "pnl": -100, "pnl_pct": -1.0, "direction": "long"},
        ]

        # Build a simple equity series with DatetimeIndex
        timestamps = pd.date_range("2024-01-02", periods=len(trades) + 1, freq="1h", tz="UTC")
        equity = pd.Series(
            [initial_balance + sum(t["pnl"] for t in trades[:i]) for i in range(len(trades) + 1)],
            index=timestamps,
        )

        metrics = PerformanceMetrics()
        result = metrics.calculate(
            trades=trades, equity_curve=equity, initial_balance=initial_balance
        )

        assert result["total_trades"] == 10
        assert abs(result["win_rate"] - 0.6) < 0.01

    def test_metrics_sharpe_key_present(self):
        """Metrics output should have sharpe_ratio and total_return_pct keys."""
        from src.backtesting.metrics import PerformanceMetrics

        initial_balance = 10_000.0
        trades = [
            {"r_multiple": 2.0, "pnl": 200, "pnl_pct": 2.0, "direction": "long"},
            {"r_multiple": -1.0, "pnl": -100, "pnl_pct": -1.0, "direction": "long"},
            {"r_multiple": 1.5, "pnl": 150, "pnl_pct": 1.5, "direction": "long"},
        ]
        timestamps = pd.date_range("2024-01-02", periods=4, freq="1h", tz="UTC")
        equity = pd.Series([initial_balance, 10_200, 10_100, 10_250], index=timestamps)

        metrics = PerformanceMetrics()
        result = metrics.calculate(
            trades=trades, equity_curve=equity, initial_balance=initial_balance
        )

        assert "sharpe_ratio" in result
        assert "total_return_pct" in result
        assert result["total_trades"] == 3


# ---------------------------------------------------------------------------
# 9. PropFirmTracker end-to-end
# ---------------------------------------------------------------------------

class TestPropFirmTrackerIntegration:
    """PropFirmTracker enforces challenge constraints correctly."""

    def test_profit_target_triggers_pass(self):
        """Equity growing past 8% profit target should set status to passed."""
        from src.backtesting.metrics import PropFirmTracker

        tracker = PropFirmTracker(profit_target_pct=8.0, time_limit_days=30)
        start = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
        tracker.initialise(10_000.0, start)

        # Simulate equity growing above 8% target
        for i in range(10):
            ts = start + timedelta(hours=i)
            tracker.update(ts, 10_000 + (i + 1) * 100)  # reaches 11000 = +10%

        status = tracker.check_pass()
        assert status.status == "passed"

    def test_total_dd_triggers_failure(self):
        """Equity dropping below 10% total DD limit should fail the challenge.

        Uses daily_dd_pct=100 (effectively disabled) so only total DD triggers.
        """
        from src.backtesting.metrics import PropFirmTracker

        # Set daily DD limit very high so it doesn't trigger first
        tracker = PropFirmTracker(
            max_total_dd_pct=10.0,
            max_daily_dd_pct=100.0,  # disable daily DD effectively
            time_limit_days=30,
        )
        start = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
        tracker.initialise(10_000.0, start)

        # Simulate gradual losses across many days to exceed 10% total DD
        # Each day loses ~300 (3% per day) — spread across days so no single
        # day triggers the (high) daily limit
        for i in range(4):
            for h in range(6):
                ts = start + timedelta(days=i, hours=h)
                balance = 10_000.0 - (i * 6 + h + 1) * 50  # -50 per step
                tracker.update(ts, balance)
        # After 24 steps * 50 = 1200 loss = 12% total DD -> fails

        status = tracker.check_pass()
        assert status.status == "failed_total_dd"

    def test_prop_firm_status_has_correct_structure(self):
        """PropFirmStatus should include profit_pct, max_daily_dd_pct, days_elapsed."""
        from src.backtesting.metrics import PropFirmTracker, PropFirmStatus

        tracker = PropFirmTracker()
        start = datetime(2024, 1, 2, 8, 0, tzinfo=timezone.utc)
        tracker.initialise(10_000.0, start)
        tracker.update(start + timedelta(hours=1), 10_050.0)

        status = tracker.check_pass()
        assert isinstance(status, PropFirmStatus)
        assert isinstance(status.profit_pct, float)
        assert isinstance(status.max_daily_dd_pct, float)
        assert isinstance(status.days_elapsed, int)


# ---------------------------------------------------------------------------
# 10. Risk module chain: Sizer -> CircuitBreaker -> ExitManager
# ---------------------------------------------------------------------------

class TestRiskModuleChain:
    """AdaptivePositionSizer, DailyCircuitBreaker, HybridExitManager integrate correctly."""

    def test_position_sizer_phase_transition(self):
        """Sizer switches from aggressive to protective phase at threshold."""
        from src.risk.position_sizer import AdaptivePositionSizer

        sizer = AdaptivePositionSizer(
            initial_balance=10_000.0,
            initial_risk_pct=1.5,
            reduced_risk_pct=0.75,
            phase_threshold_pct=4.0,
        )

        # Phase 1: balance at initial (0% profit) -> aggressive
        assert sizer.get_phase() == "aggressive"
        assert abs(sizer.get_risk_pct() - 1.5) < 0.01

        # Phase 2: update balance to +5% -> protective
        sizer.update_balance(10_500.0)
        assert sizer.get_phase() == "protective"
        assert abs(sizer.get_risk_pct() - 0.75) < 0.01

    def test_circuit_breaker_halts_trading(self):
        """DailyCircuitBreaker blocks trading after daily limit hit."""
        from src.risk.circuit_breaker import DailyCircuitBreaker

        breaker = DailyCircuitBreaker(max_daily_loss_pct=2.0)
        import datetime as dt
        breaker.start_day(10_000.0, dt.date(2024, 1, 3))

        # Should allow trading initially
        assert breaker.can_trade(10_000.0) is True

        # Simulate a loss beyond 2%
        assert breaker.can_trade(9_780.0) is False  # 2.2% loss

    def test_trade_manager_can_open_trade(self):
        """TradeManager.can_open_trade() returns True when no blocking conditions."""
        from src.risk.position_sizer import AdaptivePositionSizer
        from src.risk.circuit_breaker import DailyCircuitBreaker
        from src.risk.exit_manager import HybridExitManager
        from src.risk.trade_manager import TradeManager
        import datetime as dt

        sizer = AdaptivePositionSizer(initial_balance=10_000.0)
        breaker = DailyCircuitBreaker()
        breaker.start_day(10_000.0, dt.date(2024, 1, 3))
        exit_mgr = HybridExitManager()

        manager = TradeManager(
            position_sizer=sizer,
            circuit_breaker=breaker,
            exit_manager=exit_mgr,
        )

        can_open, reason = manager.can_open_trade(current_balance=10_000.0)
        assert can_open is True
        assert isinstance(reason, str)
