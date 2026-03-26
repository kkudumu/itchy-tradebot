"""
Tests for the backtest dashboard HTML generator.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from src.backtesting.dashboard import (
    BacktestDashboard,
    chart_equity_curve,
    chart_trades_on_price,
    chart_learning_phases,
    chart_daily_pnl,
    chart_win_rate_heatmap,
    chart_prop_firm_tracking,
)


# ---------------------------------------------------------------------------
# Helper: minimal BacktestResult-like object
# ---------------------------------------------------------------------------

class FakeResult:
    def __init__(self, n_trades=10, n_bars=100):
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="5min", tz="UTC")
        equity_vals = 10000.0 + np.cumsum(np.random.randn(n_bars) * 10)

        self.equity_curve = pd.Series(equity_vals, index=dates, name="equity")
        self.daily_pnl = pd.Series(
            np.random.randn(5) * 0.01,
            index=pd.date_range("2024-01-01", periods=5, freq="1D"),
            name="daily_pnl",
        )
        self.trades = []
        for i in range(n_trades):
            r = np.random.choice([1.5, -0.8, 2.0, -0.5, 0.3])
            self.trades.append({
                "entry_price": 2000.0 + i,
                "exit_price": 2000.0 + i + r * 5,
                "r_multiple": r,
                "confluence_score": 5,
                "signal_tier": "B",
                "context": {"session": np.random.choice(["london", "new_york", "asian"]),
                            "adx_value": 30.0},
            })
        self.metrics = {
            "total_trades": n_trades,
            "win_rate": 0.55,
            "total_return_pct": 6.5,
            "sharpe_ratio": 1.2,
            "max_drawdown_pct": -3.5,
            "profit_factor": 1.4,
            "expectancy": 0.2,
        }
        self.prop_firm = {
            "status": "active",
            "profit_pct": 6.5,
            "max_daily_dd_pct": 1.8,
            "max_total_dd_pct": 3.5,
            "days_elapsed": 15,
        }
        self.skipped_signals = 5
        self.total_signals = 25


# ===========================================================================
# Individual chart tests
# ===========================================================================

class TestChartEquityCurve:
    def test_returns_base64_string(self):
        dates = pd.date_range("2024-01-01", periods=50, freq="5min", tz="UTC")
        equity = pd.Series(10000 + np.arange(50) * 10.0, index=dates)
        b64 = chart_equity_curve(equity, 10000.0)
        assert isinstance(b64, str)
        assert len(b64) > 100

    def test_empty_equity(self):
        b64 = chart_equity_curve(pd.Series(dtype=float), 10000.0)
        assert isinstance(b64, str)
        assert len(b64) > 50


class TestChartTradesOnPrice:
    def test_returns_base64_with_trades(self):
        trades = [
            {"entry_price": 2000, "exit_price": 2010, "r_multiple": 1.5},
            {"entry_price": 2005, "exit_price": 1995, "r_multiple": -0.8},
        ]
        b64 = chart_trades_on_price(trades, pd.Series(dtype=float))
        assert isinstance(b64, str)
        assert len(b64) > 100

    def test_empty_trades(self):
        b64 = chart_trades_on_price([], pd.Series(dtype=float))
        assert isinstance(b64, str)


class TestChartLearningPhases:
    def test_mechanical_only(self):
        b64 = chart_learning_phases([], total_trades=50)
        assert isinstance(b64, str)
        assert len(b64) > 100

    def test_all_phases(self):
        b64 = chart_learning_phases([], total_trades=700)
        assert isinstance(b64, str)

    def test_zero_trades(self):
        b64 = chart_learning_phases([], total_trades=0)
        assert isinstance(b64, str)


class TestChartDailyPnl:
    def test_returns_base64(self):
        pnl = pd.Series([0.01, -0.005, 0.02, -0.01, 0.005],
                         index=pd.date_range("2024-01-01", periods=5, freq="1D"))
        b64 = chart_daily_pnl(pnl)
        assert isinstance(b64, str)
        assert len(b64) > 100

    def test_empty_pnl(self):
        b64 = chart_daily_pnl(pd.Series(dtype=float))
        assert isinstance(b64, str)


class TestChartWinRateHeatmap:
    def test_with_trades(self):
        trades = [
            {"r_multiple": 1.5, "context": {"session": "london"}},
            {"r_multiple": -0.8, "context": {"session": "london"}},
            {"r_multiple": 2.0, "context": {"session": "new_york"}},
        ]
        b64 = chart_win_rate_heatmap(trades)
        assert isinstance(b64, str)
        assert len(b64) > 100

    def test_empty_trades(self):
        b64 = chart_win_rate_heatmap([])
        assert isinstance(b64, str)


class TestChartPropFirmTracking:
    def test_returns_base64(self):
        dates = pd.date_range("2024-01-01", periods=50, freq="5min", tz="UTC")
        equity = pd.Series(10000 + np.arange(50) * 10.0, index=dates)
        daily_pnl = pd.Series([0.01, -0.005], index=pd.date_range("2024-01-01", periods=2, freq="1D"))
        b64 = chart_prop_firm_tracking(equity, 10000.0, daily_pnl)
        assert isinstance(b64, str)
        assert len(b64) > 100

    def test_empty_data(self):
        b64 = chart_prop_firm_tracking(
            pd.Series(dtype=float), 10000.0, pd.Series(dtype=float)
        )
        assert isinstance(b64, str)


# ===========================================================================
# Full dashboard generation
# ===========================================================================

class TestBacktestDashboard:
    def test_generate_returns_html(self):
        result = FakeResult(n_trades=10)
        dashboard = BacktestDashboard()
        html = dashboard.generate(result, initial_balance=10000.0, learning_phase="mechanical")
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert "Equity Curve" in html
        assert "data:image/png;base64," in html

    def test_html_contains_all_sections(self):
        result = FakeResult(n_trades=20)
        dashboard = BacktestDashboard()
        html = dashboard.generate(result, learning_phase="statistical", learning_skipped=3)
        assert "Win Rate" in html
        assert "Sharpe" in html
        assert "Prop" in html
        assert "Learning" in html
        assert "Statistical" in html

    def test_html_contains_metrics(self):
        result = FakeResult(n_trades=5)
        dashboard = BacktestDashboard()
        html = dashboard.generate(result, learning_phase="mechanical")
        assert "55.0%" in html  # win rate
        assert "1.20" in html   # sharpe

    def test_save_and_open_creates_file(self, tmp_path):
        result = FakeResult(n_trades=5)
        dashboard = BacktestDashboard()
        path = dashboard.save_and_open(
            result=result,
            output_dir=str(tmp_path),
            initial_balance=10000.0,
            learning_phase="mechanical",
            auto_open=False,
        )
        assert Path(path).exists()
        content = Path(path).read_text()
        assert "<!DOCTYPE html>" in content

    def test_save_and_open_auto_open_flag(self, tmp_path):
        result = FakeResult(n_trades=3)
        dashboard = BacktestDashboard()
        with patch("src.backtesting.dashboard.webbrowser.open") as mock_open:
            dashboard.save_and_open(
                result=result,
                output_dir=str(tmp_path),
                auto_open=True,
            )
            mock_open.assert_called_once()

    def test_zero_trades_dashboard(self):
        result = FakeResult(n_trades=0)
        result.trades = []
        result.metrics["total_trades"] = 0
        dashboard = BacktestDashboard()
        html = dashboard.generate(result, learning_phase="mechanical")
        assert "<!DOCTYPE html>" in html

    def test_custom_title(self):
        result = FakeResult(n_trades=3)
        dashboard = BacktestDashboard(title="Custom Test Dashboard")
        html = dashboard.generate(result)
        assert "Custom Test Dashboard" in html


from pathlib import Path
