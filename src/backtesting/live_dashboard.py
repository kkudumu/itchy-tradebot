"""
Live-updating backtest dashboard served via a local HTTP server.

Starts a lightweight server on localhost that serves a single-page dashboard.
The dashboard polls /api/state every 500ms and updates charts and metrics
in-place without page reloads.

Usage from the backtester::

    from src.backtesting.live_dashboard import LiveDashboardServer

    server = LiveDashboardServer(port=8501)
    server.start()                       # opens browser automatically
    server.update(state_dict)            # called every N bars
    server.finish(final_state_dict)      # marks run as complete
    server.stop()                        # shuts down server

No external dependencies — uses only Python stdlib (http.server, threading, json).
"""

from __future__ import annotations

import json
import logging
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared state container
# ---------------------------------------------------------------------------

class _DashboardState:
    """Thread-safe state container shared between backtest and HTTP server."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: Dict[str, Any] = {
            "running": False,
            "done": False,
            "bar_index": 0,
            "total_bars": 0,
            "pct_complete": 0.0,
            "elapsed_seconds": 0.0,
            "equity": 10000.0,
            "initial_balance": 10000.0,
            "equity_history": [],
            "balance_pct": 0.0,
            "n_trades": 0,
            "n_wins": 0,
            "n_losses": 0,
            "win_rate": 0.0,
            "total_return_pct": 0.0,
            "max_dd_pct": 0.0,
            "worst_daily_dd_pct": 0.0,
            "sharpe": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "learning_phase": "mechanical",
            "learning_trades": 0,
            "learning_skipped": 0,
            "prop_status": "active",
            "prop_profit_pct": 0.0,
            "total_signals": 0,
            "skipped_signals": 0,
            "recent_trades": [],
            "instrument": "XAUUSD",
            "start_time": "",
            "current_timestamp": "",
            "data_start_date": "",
            "data_end_date": "",
            # Health monitor fields
            "health_state": "normal",
            "health_drought": False,
            "health_relaxation_tier": 0,
            "health_regime": None,
            "health_bottleneck": None,
            "health_message": "Not started",
        }

    def update(self, updates: Dict[str, Any]) -> None:
        with self._lock:
            self._state.update(updates)

    def get(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._state)


# ---------------------------------------------------------------------------
# Candle buffer
# ---------------------------------------------------------------------------

class _CandleBuffer:
    """Thread-safe buffer for OHLCV + Ichimoku candle data and trades."""

    MAX_BARS = 10_000

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._candles: Dict[str, List[list]] = {
            "1m": [], "5m": [], "15m": [], "30m": [], "1h": [], "4h": [], "1d": [],
        }
        self._trades: List[dict] = []

    def append_candle(self, tf: str, data: list) -> None:
        """Append a candle [time, o, h, l, c, vol, tenkan, kijun, senkou_a, senkou_b, chikou]."""
        tf = tf.lower()
        with self._lock:
            buf = self._candles.get(tf)
            if buf is not None:
                buf.append(data)
                if len(buf) > self.MAX_BARS:
                    del buf[:len(buf) - self.MAX_BARS]

    def append_trade(self, trade: dict) -> None:
        """Append a completed trade."""
        with self._lock:
            self._trades.append(trade)

    def get_candles(self, tf: str, since: int = 0) -> List[list]:
        """Return candles for timeframe, optionally filtered by time > since."""
        tf = tf.lower()
        with self._lock:
            buf = self._candles.get(tf, [])
            if since:
                return [c for c in buf if c[0] > since]
            return list(buf)

    def get_trades(self, since_idx: int = 0) -> dict:
        """Return trades starting from index, with next_index for pagination."""
        with self._lock:
            trades = self._trades[since_idx:]
            return {
                "trades": trades,
                "next_index": len(self._trades),
            }


# ---------------------------------------------------------------------------
# HTTP Request Handler
# ---------------------------------------------------------------------------

class _DashboardHandler(BaseHTTPRequestHandler):
    """Serves the dashboard HTML, JSON state API, candle/trade data, and config API."""

    # Assigned by the server before handling requests
    state: _DashboardState
    candle_buffer: _CandleBuffer
    server_ref: "LiveDashboardServer"

    # mtime-based cache for the HTML file
    _html_cache: dict = {"content": "", "mtime": 0.0}

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/state":
            self._serve_json()
        elif path == "/api/candles":
            self._serve_candles(parse_qs(parsed.query))
        elif path == "/api/trades":
            self._serve_trades(parse_qs(parsed.query))
        elif path == "/api/config":
            self._serve_config()
        elif path == "/" or path == "/index.html":
            self._serve_html()
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length)) if content_length > 0 else {}

        if path == "/api/config/edges":
            self._handle_edge_update(body)
        elif path == "/api/config/strategy":
            self._handle_strategy_update(body)
        elif path == "/api/config/save":
            self._handle_config_save()
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _serve_json(self):
        data = self.state.get()
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _serve_html(self):
        html_path = Path(__file__).parent / "dashboard_chart.html"
        try:
            mtime = html_path.stat().st_mtime
            if mtime != _DashboardHandler._html_cache["mtime"]:
                _DashboardHandler._html_cache["content"] = html_path.read_text(encoding="utf-8")
                _DashboardHandler._html_cache["mtime"] = mtime
            html = _DashboardHandler._html_cache["content"].encode("utf-8")
        except OSError:
            self.send_error(500, "Dashboard HTML file not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(html)

    def _serve_candles(self, params):
        tf = params.get("tf", ["5m"])[0]
        try:
            since = int(params.get("since", ["0"])[0])
        except (ValueError, TypeError):
            since = 0
        data = self.candle_buffer.get_candles(tf, since)
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _serve_trades(self, params):
        try:
            since_idx = max(0, int(params.get("since", ["0"])[0]))
        except (ValueError, TypeError):
            since_idx = 0
        data = self.candle_buffer.get_trades(since_idx)
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _serve_config(self):
        """Return current edges + strategy config."""
        server = self.server_ref
        config: Dict[str, Any] = {"edges": {}, "strategy": {}}
        with server._config_lock:
            if server._edge_manager is not None:
                config["edges"] = server._edge_manager.get_all_config()
            config["strategy"] = dict(server._strategy_config)
        self._send_json_response(200, config)

    def _handle_edge_update(self, body):
        """Toggle or update edge params."""
        server = self.server_ref
        try:
            edge_name = body.get("edge_name")
            if not edge_name or server._edge_manager is None:
                self._send_json_response(400, {"status": "error", "message": "Missing edge_name or no edge manager"})
                return
            with server._config_lock:
                if "enabled" in body:
                    server._edge_manager.toggle_edge(edge_name, bool(body["enabled"]))
                if "params" in body and isinstance(body["params"], dict):
                    server._edge_manager.set_edge_params(edge_name, body["params"])
            self._send_json_response(200, {"status": "ok"})
        except KeyError as e:
            self._send_json_response(400, {"status": "error", "message": str(e)})
        except Exception as e:
            self._send_json_response(500, {"status": "error", "message": str(e)})

    def _handle_strategy_update(self, body):
        """Update strategy params."""
        server = self.server_ref
        section = body.get("section")
        params = body.get("params", {})
        if not section or not isinstance(params, dict):
            self._send_json_response(400, {"status": "error", "message": "Missing section or params"})
            return
        with server._config_lock:
            if section not in server._strategy_config:
                server._strategy_config[section] = {}
            server._strategy_config[section].update(params)
        self._send_json_response(200, {"status": "ok"})

    def _handle_config_save(self):
        """Persist current runtime config back to YAML files."""
        server = self.server_ref
        if not server._config_dir:
            self._send_json_response(400, {"status": "error", "message": "No config directory configured"})
            return
        try:
            import yaml
            config_path = Path(server._config_dir)

            with server._config_lock:
                if server._edge_manager is not None:
                    edges_data = server._edge_manager.get_all_config()
                    edges_path = config_path / "edges.yaml"
                    edges_path.write_text(
                        yaml.dump(edges_data, default_flow_style=False, sort_keys=False),
                        encoding="utf-8",
                    )
                if server._strategy_config:
                    strategy_path = config_path / "strategy.yaml"
                    strategy_path.write_text(
                        yaml.dump(dict(server._strategy_config), default_flow_style=False, sort_keys=False),
                        encoding="utf-8",
                    )
            self._send_json_response(200, {"status": "ok"})
        except Exception as e:
            self._send_json_response(500, {"status": "error", "message": str(e)})

    def _send_json_response(self, code: int, data: Any) -> None:
        """Helper to send a JSON response with CORS headers."""
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Suppress request logging to keep terminal clean
        pass


# ---------------------------------------------------------------------------
# LiveDashboardServer
# ---------------------------------------------------------------------------

class LiveDashboardServer:
    """Local HTTP server for live backtest monitoring.

    Parameters
    ----------
    port:
        Port to serve on.  Default: 8501.
    auto_open:
        Automatically open the dashboard in the default browser.
    edge_manager:
        Optional EdgeManager instance for runtime config changes.
    app_config:
        Optional AppConfig dict for reading strategy parameters.
    config_dir:
        Optional path to config directory for save-to-YAML.
    """

    def __init__(
        self,
        port: int = 8501,
        auto_open: bool = True,
        edge_manager=None,
        app_config: Optional[dict] = None,
        config_dir: Optional[str] = None,
    ) -> None:
        self._port = port
        self._auto_open = auto_open
        self._state = _DashboardState()
        self._candle_buffer = _CandleBuffer()
        self._edge_manager = edge_manager
        self._app_config = app_config or {}
        self._config_dir = config_dir
        self._strategy_config: dict = dict(self._app_config.get("strategy", {})) if self._app_config else {}
        self._config_lock = threading.Lock()
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the HTTP server in a background thread."""
        handler_class = type(
            "_Handler",
            (_DashboardHandler,),
            {
                "state": self._state,
                "candle_buffer": self._candle_buffer,
                "server_ref": self,
            },
        )

        # Probe whether the port is genuinely available before binding.
        import socket

        def _port_in_use(port: int) -> bool:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.connect(("127.0.0.1", port))
                    return True
                except (ConnectionRefusedError, OSError):
                    return False

        if _port_in_use(self._port):
            self._port += 1
            logger.warning("Port in use, trying %d", self._port)

        try:
            self._server = HTTPServer(("127.0.0.1", self._port), handler_class)
        except OSError:
            self._port += 1
            logger.warning("Port in use, trying %d", self._port)
            self._server = HTTPServer(("127.0.0.1", self._port), handler_class)

        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

        url = f"http://127.0.0.1:{self._port}"
        logger.info("Live dashboard running at %s", url)

        self._state.update({"running": True, "start_time": datetime.now(timezone.utc).isoformat()})

        if self._auto_open:
            try:
                webbrowser.open(url)
            except Exception:
                pass

    def update(self, updates: Dict[str, Any]) -> None:
        """Push a state update to the dashboard."""
        self._state.update(updates)

    def finish(self, final_state: Optional[Dict[str, Any]] = None) -> None:
        """Mark the backtest as complete."""
        updates = {"done": True, "running": False, "pct_complete": 100.0}
        if final_state:
            updates.update(final_state)
        self._state.update(updates)

    def append_candle(self, tf: str, data: list) -> None:
        """Push a candle bar to the buffer."""
        self._candle_buffer.append_candle(tf, data)

    def append_trade(self, trade: dict) -> None:
        """Push a completed trade to the buffer."""
        self._candle_buffer.append_trade(trade)

    def stop(self) -> None:
        """Shut down the HTTP server."""
        if self._server:
            self._server.shutdown()
            logger.info("Live dashboard server stopped")

    @property
    def port(self) -> int:
        return self._port

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    @property
    def strategy_config(self) -> dict:
        """Return current strategy config dict."""
        with self._config_lock:
            return dict(self._strategy_config)
