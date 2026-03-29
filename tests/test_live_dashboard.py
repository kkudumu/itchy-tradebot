"""
Tests for the live backtest dashboard server.
"""

import json
import time
import threading
import pytest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

from src.backtesting.live_dashboard import (
    LiveDashboardServer, _DashboardState, _CandleBuffer,
)


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
            assert "chart-container" in html
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
        time.sleep(0.5)
        try:
            server2 = LiveDashboardServer(port=18505, auto_open=False)
            server2.start()
            time.sleep(0.5)
            # server2 should fallback to next port since server1 holds 18505
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
            assert "chart-container" in html
            assert "equity-canvas" in html
            assert "trade-panel" in html
            assert "trade-list-container" in html
            assert "settings-panel" in html
            assert "gauge" in html
            assert "/api/state" in html  # polling endpoint
            assert "/api/candles" in html  # candle polling
            assert "/api/trades" in html  # trade polling
            assert "/api/config" in html  # config API
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

            # Verify candles were actually pushed
            candle_resp = urlopen("http://127.0.0.1:18510/api/candles?tf=5m")
            candle_data = json.loads(candle_resp.read())
            assert len(candle_data) > 0, "No candles pushed to dashboard"
            assert len(candle_data[0]) == 11, "Candle should have 11 fields"

            # Verify timestamps were populated
            assert data["data_start_date"] != "", "data_start_date not set"
            assert data["data_end_date"] != "", "data_end_date not set"
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


class TestCandleBuffer:
    """Tests for the thread-safe candle buffer."""

    def test_append_and_get_candles(self):
        buf = _CandleBuffer()
        candle = [1709215500, 2040.0, 2042.0, 2038.0, 2041.0, 100, 0, 0, 0, 0, 0]
        buf.append_candle("5m", candle)
        result = buf.get_candles("5m")
        assert len(result) == 1
        assert result[0] == candle

    def test_get_candles_with_since(self):
        buf = _CandleBuffer()
        buf.append_candle("5m", [100, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0])
        buf.append_candle("5m", [200, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0])
        buf.append_candle("5m", [300, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0])
        result = buf.get_candles("5m", since=150)
        assert len(result) == 2
        assert result[0][0] == 200
        assert result[1][0] == 300

    def test_get_candles_without_since(self):
        buf = _CandleBuffer()
        for i in range(5):
            buf.append_candle("15m", [i * 100, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0])
        result = buf.get_candles("15m")
        assert len(result) == 5

    def test_buffer_cap(self):
        buf = _CandleBuffer()
        for i in range(10_050):
            buf.append_candle("5m", [i, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0])
        result = buf.get_candles("5m")
        assert len(result) == 10_000

    def test_append_trade_and_get(self):
        buf = _CandleBuffer()
        trade = {"id": 1, "direction": "long", "r_multiple": 1.5}
        buf.append_trade(trade)
        result = buf.get_trades(0)
        assert len(result["trades"]) == 1
        assert result["trades"][0] == trade
        assert result["next_index"] == 1

    def test_get_trades_pagination(self):
        buf = _CandleBuffer()
        for i in range(5):
            buf.append_trade({"id": i + 1})
        result = buf.get_trades(3)
        assert len(result["trades"]) == 2
        assert result["next_index"] == 5

    def test_unknown_timeframe_ignored(self):
        buf = _CandleBuffer()
        buf.append_candle("10m", [100, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0])
        assert buf.get_candles("10m") == []

    def test_multiple_timeframes(self):
        buf = _CandleBuffer()
        buf.append_candle("5m", [100, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0])
        buf.append_candle("1h", [200, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0])
        assert len(buf.get_candles("5m")) == 1
        assert len(buf.get_candles("1h")) == 1

    def test_thread_safety(self):
        """Hammer append_candle from multiple threads."""
        buf = _CandleBuffer()
        errors = []

        def writer(start):
            try:
                for i in range(500):
                    buf.append_candle("5m", [start + i, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0])
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(writer, i * 1000) for i in range(4)]
            for f in futures:
                f.result()

        assert len(errors) == 0
        assert len(buf.get_candles("5m")) == 2000


class TestCandleAndTradeEndpoints:
    """Tests for /api/candles and /api/trades HTTP endpoints."""

    def test_candles_endpoint(self):
        server = LiveDashboardServer(port=18520, auto_open=False)
        server.start()
        time.sleep(0.2)
        try:
            server.append_candle("5m", [1000, 2040, 2042, 2038, 2041, 100, 0, 0, 0, 0, 0])
            server.append_candle("5m", [1300, 2041, 2043, 2039, 2042, 110, 0, 0, 0, 0, 0])
            response = urlopen("http://127.0.0.1:18520/api/candles?tf=5m")
            data = json.loads(response.read())
            assert len(data) == 2
            assert data[0][0] == 1000
            assert data[1][0] == 1300
        finally:
            server.stop()

    def test_candles_since_filter(self):
        server = LiveDashboardServer(port=18521, auto_open=False)
        server.start()
        time.sleep(0.2)
        try:
            server.append_candle("5m", [1000, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0])
            server.append_candle("5m", [2000, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0])
            server.append_candle("5m", [3000, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0])
            response = urlopen("http://127.0.0.1:18521/api/candles?tf=5m&since=1500")
            data = json.loads(response.read())
            assert len(data) == 2
        finally:
            server.stop()

    def test_candles_timeframe_param(self):
        server = LiveDashboardServer(port=18522, auto_open=False)
        server.start()
        time.sleep(0.2)
        try:
            server.append_candle("1h", [3600, 1, 2, 3, 4, 5, 0, 0, 0, 0, 0])
            response = urlopen("http://127.0.0.1:18522/api/candles?tf=1h")
            data = json.loads(response.read())
            assert len(data) == 1
            assert data[0][0] == 3600
        finally:
            server.stop()

    def test_trades_endpoint(self):
        server = LiveDashboardServer(port=18523, auto_open=False)
        server.start()
        time.sleep(0.2)
        try:
            server.append_trade({"id": 1, "direction": "long", "r_multiple": 1.5})
            server.append_trade({"id": 2, "direction": "short", "r_multiple": -0.8})
            response = urlopen("http://127.0.0.1:18523/api/trades?since=0")
            data = json.loads(response.read())
            assert len(data["trades"]) == 2
            assert data["next_index"] == 2
        finally:
            server.stop()

    def test_trades_pagination(self):
        server = LiveDashboardServer(port=18524, auto_open=False)
        server.start()
        time.sleep(0.2)
        try:
            for i in range(5):
                server.append_trade({"id": i + 1})
            response = urlopen("http://127.0.0.1:18524/api/trades?since=3")
            data = json.loads(response.read())
            assert len(data["trades"]) == 2
            assert data["next_index"] == 5
        finally:
            server.stop()


class TestDashboardStateTimestamps:
    """Tests for timestamp fields in dashboard state."""

    def test_state_has_timestamp_fields(self):
        state = _DashboardState()
        s = state.get()
        assert "current_timestamp" in s
        assert "data_start_date" in s
        assert "data_end_date" in s

    def test_timestamp_fields_update(self):
        state = _DashboardState()
        state.update({
            "current_timestamp": "2025-03-15T14:30:00+00:00",
            "data_start_date": "2025-03-01",
            "data_end_date": "2025-03-20",
        })
        s = state.get()
        assert s["current_timestamp"] == "2025-03-15T14:30:00+00:00"
        assert s["data_start_date"] == "2025-03-01"


class TestConfigAPI:
    """Tests for the config read/write API endpoints."""

    def test_get_config_without_edge_manager(self):
        server = LiveDashboardServer(port=18530, auto_open=False)
        server.start()
        time.sleep(0.2)
        try:
            response = urlopen("http://127.0.0.1:18530/api/config")
            data = json.loads(response.read())
            assert "edges" in data
            assert "strategy" in data
            assert data["edges"] == {}
        finally:
            server.stop()

    def test_get_config_with_edge_manager(self):
        from src.edges.manager import EdgeManager
        em = EdgeManager({
            "time_of_day": {"enabled": True, "params": {"start_utc": "08:00", "end_utc": "17:00"}},
        })
        server = LiveDashboardServer(
            port=18531, auto_open=False,
            edge_manager=em,
            app_config={"strategy": {"risk": {"initial_risk_pct": 1.5}}},
        )
        server.start()
        time.sleep(0.2)
        try:
            response = urlopen("http://127.0.0.1:18531/api/config")
            data = json.loads(response.read())
            assert "time_of_day" in data["edges"]
            assert data["edges"]["time_of_day"]["enabled"] is True
            assert "risk" in data["strategy"]
        finally:
            server.stop()

    def test_post_edge_toggle(self):
        from src.edges.manager import EdgeManager
        em = EdgeManager({
            "time_of_day": {"enabled": True, "params": {"start_utc": "08:00", "end_utc": "17:00"}},
        })
        server = LiveDashboardServer(port=18532, auto_open=False, edge_manager=em)
        server.start()
        time.sleep(0.2)
        try:
            body = json.dumps({"edge_name": "time_of_day", "enabled": False}).encode()
            req = Request(
                "http://127.0.0.1:18532/api/config/edges",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            response = urlopen(req)
            result = json.loads(response.read())
            assert result["status"] == "ok"
            assert em._all_edges["time_of_day"].enabled is False
        finally:
            server.stop()

    def test_post_edge_params(self):
        from src.edges.manager import EdgeManager
        em = EdgeManager({
            "regime_filter": {"enabled": True, "params": {"adx_min": 20, "cloud_thickness_percentile": 30}},
        })
        server = LiveDashboardServer(port=18533, auto_open=False, edge_manager=em)
        server.start()
        time.sleep(0.2)
        try:
            body = json.dumps({
                "edge_name": "regime_filter",
                "params": {"adx_min": 25},
            }).encode()
            req = Request(
                "http://127.0.0.1:18533/api/config/edges",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            response = urlopen(req)
            result = json.loads(response.read())
            assert result["status"] == "ok"
            cfg = em.get_all_config()
            assert cfg["regime_filter"]["params"]["adx_min"] == 25
        finally:
            server.stop()

    def test_post_strategy_update(self):
        server = LiveDashboardServer(
            port=18534, auto_open=False,
            app_config={"strategy": {"risk": {"initial_risk_pct": 1.5}}},
        )
        server.start()
        time.sleep(0.2)
        try:
            body = json.dumps({
                "section": "risk",
                "params": {"initial_risk_pct": 1.0},
            }).encode()
            req = Request(
                "http://127.0.0.1:18534/api/config/strategy",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            response = urlopen(req)
            result = json.loads(response.read())
            assert result["status"] == "ok"
            assert server.strategy_config["risk"]["initial_risk_pct"] == 1.0
        finally:
            server.stop()

    def test_post_config_save_no_dir(self):
        """Save without config_dir returns 400 error."""
        server = LiveDashboardServer(port=18535, auto_open=False)
        server.start()
        time.sleep(0.2)
        try:
            body = json.dumps({}).encode()
            req = Request(
                "http://127.0.0.1:18535/api/config/save",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with pytest.raises(HTTPError) as exc_info:
                urlopen(req)
            assert exc_info.value.code == 400
        finally:
            server.stop()

    def test_html_served_from_file(self):
        """Verify HTML is served from the separate file, not inline."""
        server = LiveDashboardServer(port=18536, auto_open=False)
        server.start()
        time.sleep(0.2)
        try:
            response = urlopen("http://127.0.0.1:18536/")
            html = response.read().decode("utf-8")
            assert "lightweight-charts" in html
            assert "chart-container" in html
            assert len(html) > 1000
        finally:
            server.stop()
