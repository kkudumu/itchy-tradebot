"""
Tests for the live backtest dashboard server.
"""

import json
import time
import pytest
from unittest.mock import patch
from urllib.request import urlopen
from urllib.error import URLError

from src.backtesting.live_dashboard import LiveDashboardServer, _DashboardState


class TestDashboardState:
    """Tests for the thread-safe state container."""

    def test_initial_state(self):
        state = _DashboardState()
        s = state.get()
        assert s["running"] is False
        assert s["done"] is False
        assert s["bar_index"] == 0
        assert s["equity"] == 10000.0

    def test_update(self):
        state = _DashboardState()
        state.update({"bar_index": 42, "equity": 10500.0})
        s = state.get()
        assert s["bar_index"] == 42
        assert s["equity"] == 10500.0

    def test_get_returns_copy(self):
        state = _DashboardState()
        s1 = state.get()
        s1["bar_index"] = 999
        s2 = state.get()
        assert s2["bar_index"] == 0  # original unchanged

    def test_concurrent_updates(self):
        """Multiple updates don't corrupt state."""
        state = _DashboardState()
        for i in range(100):
            state.update({"bar_index": i})
        s = state.get()
        assert s["bar_index"] == 99


class TestLiveDashboardServer:
    """Tests for the HTTP server lifecycle."""

    def test_start_and_stop(self):
        server = LiveDashboardServer(port=18501, auto_open=False)
        server.start()
        time.sleep(0.2)
        assert server.port == 18501
        assert server.url == "http://127.0.0.1:18501"
        server.stop()

    def test_serves_html(self):
        server = LiveDashboardServer(port=18502, auto_open=False)
        server.start()
        time.sleep(0.2)
        try:
            response = urlopen("http://127.0.0.1:18502/")
            html = response.read().decode("utf-8")
            assert "<!DOCTYPE html>" in html
            assert "Itchy Tradebot" in html
            assert "equity-chart" in html
        finally:
            server.stop()

    def test_serves_json_state(self):
        server = LiveDashboardServer(port=18503, auto_open=False)
        server.start()
        time.sleep(0.2)
        try:
            server.update({"bar_index": 50, "n_trades": 5})
            response = urlopen("http://127.0.0.1:18503/api/state")
            data = json.loads(response.read())
            assert data["bar_index"] == 50
            assert data["n_trades"] == 5
            assert data["running"] is True
        finally:
            server.stop()

    def test_finish_marks_done(self):
        server = LiveDashboardServer(port=18504, auto_open=False)
        server.start()
        time.sleep(0.2)
        try:
            server.finish({"n_trades": 100})
            response = urlopen("http://127.0.0.1:18504/api/state")
            data = json.loads(response.read())
            assert data["done"] is True
            assert data["running"] is False
            assert data["n_trades"] == 100
        finally:
            server.stop()

    def test_port_fallback(self):
        """If port is in use, server tries next port."""
        server1 = LiveDashboardServer(port=18505, auto_open=False)
        server1.start()
        time.sleep(0.2)
        try:
            server2 = LiveDashboardServer(port=18505, auto_open=False)
            server2.start()
            time.sleep(0.2)
            assert server2.port == 18506
            server2.stop()
        finally:
            server1.stop()

    def test_update_multiple_fields(self):
        server = LiveDashboardServer(port=18507, auto_open=False)
        server.start()
        time.sleep(0.2)
        try:
            server.update({
                "bar_index": 100,
                "equity": 10800.0,
                "n_trades": 15,
                "win_rate": 0.6,
                "learning_phase": "statistical",
                "equity_history": [10000, 10100, 10200, 10500, 10800],
                "recent_trades": [
                    {"id": 1, "dir": "long", "entry": "2001.5", "r": "1.50"},
                ],
            })
            response = urlopen("http://127.0.0.1:18507/api/state")
            data = json.loads(response.read())
            assert data["equity"] == 10800.0
            assert data["learning_phase"] == "statistical"
            assert len(data["equity_history"]) == 5
            assert len(data["recent_trades"]) == 1
        finally:
            server.stop()

    def test_html_contains_all_ui_elements(self):
        server = LiveDashboardServer(port=18508, auto_open=False)
        server.start()
        time.sleep(0.2)
        try:
            response = urlopen("http://127.0.0.1:18508/")
            html = response.read().decode("utf-8")
            # Check key UI elements exist
            assert "equity-chart" in html
            assert "winloss-chart" in html
            assert "progress-bar" in html
            assert "trade-list" in html
            assert "phase-bar" in html
            assert "gauge" in html
            assert "/api/state" in html  # polling endpoint
        finally:
            server.stop()

    def test_auto_open_calls_webbrowser(self):
        with patch("src.backtesting.live_dashboard.webbrowser.open") as mock_open:
            server = LiveDashboardServer(port=18509, auto_open=True)
            server.start()
            time.sleep(0.2)
            mock_open.assert_called_once_with("http://127.0.0.1:18509")
            server.stop()


class TestBacktesterLiveDashboardIntegration:
    """Test that the backtester correctly pushes updates to the live dashboard."""

    def test_run_with_live_dashboard(self):
        import numpy as np
        import pandas as pd
        from src.backtesting.vectorbt_engine import IchimokuBacktester

        server = LiveDashboardServer(port=18510, auto_open=False)
        server.start()
        time.sleep(0.2)

        try:
            bt = IchimokuBacktester(initial_balance=10_000)
            n_bars = 500
            dates = pd.date_range("2024-01-01", periods=n_bars, freq="1min", tz="UTC")
            np.random.seed(42)
            prices = 2000.0 + np.cumsum(np.random.randn(n_bars) * 0.5)
            candles = pd.DataFrame({
                "open": prices,
                "high": prices + np.abs(np.random.randn(n_bars)) * 0.3,
                "low": prices - np.abs(np.random.randn(n_bars)) * 0.3,
                "close": prices + np.random.randn(n_bars) * 0.2,
                "tick_volume": np.random.randint(100, 1000, n_bars),
            }, index=dates)

            result = bt.run(candles, live_dashboard=server)
            assert result is not None

            # Check server received final state
            response = urlopen("http://127.0.0.1:18510/api/state")
            data = json.loads(response.read())
            assert data["done"] is True
            assert data["pct_complete"] == 100.0
        finally:
            server.stop()

    def test_run_without_live_dashboard(self):
        """live_dashboard=None should work exactly as before."""
        import numpy as np
        import pandas as pd
        from src.backtesting.vectorbt_engine import IchimokuBacktester

        bt = IchimokuBacktester(initial_balance=10_000)
        n_bars = 300
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="1min", tz="UTC")
        np.random.seed(99)
        prices = 2000.0 + np.cumsum(np.random.randn(n_bars) * 0.5)
        candles = pd.DataFrame({
            "open": prices,
            "high": prices + 0.3,
            "low": prices - 0.3,
            "close": prices + 0.1,
            "tick_volume": np.random.randint(100, 500, n_bars),
        }, index=dates)

        result = bt.run(candles, live_dashboard=None)
        assert result is not None
