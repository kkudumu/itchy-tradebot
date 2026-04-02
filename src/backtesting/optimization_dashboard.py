"""Unified tabbed dashboard for the optimization loop.

Serves a single-page dashboard with two tabs:
  - **Live Trading** — real-time equity curve, open trades, bar progress
  - **Optimization** — iteration table, pass rates, Claude's reasoning

Combines the functionality of the former live_dashboard (port 8501) and
optimization_dashboard (port 8502) into one server on a single port.

No external dependencies — uses only Python stdlib.
"""
from __future__ import annotations

import glob
import json
import logging
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class OptimizationDashboardServer:
    """Unified dashboard server for optimization + live trading.

    Parameters
    ----------
    port: Port to serve on. Default: 8501.
    reports_dir: Directory to scan for opt_iter_*.json reports.
    auto_open: Open browser automatically.
    live_dashboard: Optional LiveDashboardServer instance whose state is proxied.
    """

    def __init__(
        self,
        port: int = 8501,
        reports_dir: str = "reports",
        auto_open: bool = True,
        live_dashboard: Optional[Any] = None,
    ):
        self._port = port
        self._reports_dir = _REPO_ROOT / reports_dir
        self._auto_open = auto_open
        self._live_dashboard = live_dashboard
        self._server = None
        self._thread = None

    def set_live_dashboard(self, live_dashboard: Any) -> None:
        """Attach a LiveDashboardServer for proxying its state."""
        self._live_dashboard = live_dashboard
        if hasattr(self, "_handler_class"):
            self._handler_class._live_ref = live_dashboard

    def start(self):
        handler_class = _make_handler(self._reports_dir, self._live_dashboard)
        self._handler_class = handler_class
        self._server = HTTPServer(("127.0.0.1", self._port), handler_class)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        url = f"http://localhost:{self._port}"
        logger.info("Dashboard at %s", url)
        if self._auto_open:
            webbrowser.open(url)

    def stop(self):
        if self._server:
            self._server.shutdown()

    @property
    def port(self) -> int:
        return self._port

    @property
    def url(self) -> str:
        return f"http://localhost:{self._port}"


def _make_handler(reports_dir: Path, live_dashboard: Optional[Any] = None):
    """Factory to create a handler class with reports_dir and live state baked in."""

    class _Handler(BaseHTTPRequestHandler):
        _reports = reports_dir
        _html_cache: Dict[str, Any] = {"content": "", "mtime": 0.0}
        _live_ref = live_dashboard

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/api/opt-state":
                self._serve_opt_state()
            elif path == "/api/live-state":
                self._serve_live_state()
            elif path == "/api/candles":
                self._serve_candles(parse_qs(parsed.query))
            elif path == "/api/trades":
                self._serve_trades(parse_qs(parsed.query))
            elif path == "/api/discovery-state":
                self._serve_discovery_state()
            # Legacy endpoint alias
            elif path == "/api/state":
                self._serve_opt_state()
            elif path == "/" or path == "/index.html":
                self._serve_html()
            else:
                self.send_error(404)

        def _send_json(self, data: Any) -> None:
            body = json.dumps(data, default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        # ---- Optimization state ----

        def _serve_opt_state(self):
            pattern = str(self._reports / "opt_iter_*.json")
            files = sorted(glob.glob(pattern))

            iterations = []
            for f in files:
                try:
                    with open(f, encoding="utf-8") as fp:
                        data = json.load(fp)
                    iterations.append({
                        "run_id": data.get("run_id", ""),
                        "timestamp": data.get("timestamp", ""),
                        "metrics": data.get("metrics", {}),
                        "challenge_simulation": data.get("challenge_simulation", {}),
                        "config_snapshot": data.get("config_snapshot", {}),
                        "claude_reasoning": data.get("claude_reasoning", ""),
                        "claude_prompt": data.get("claude_prompt", ""),
                    })
                except (json.JSONDecodeError, OSError):
                    continue

            status_path = self._reports / "loop_status.json"
            loop_status: Dict[str, Any] = {}
            if status_path.exists():
                try:
                    loop_status = json.loads(status_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    pass

            claude_md = _REPO_ROOT / "CLAUDE.md"
            learnings = ""
            if claude_md.exists():
                try:
                    learnings = claude_md.read_text(encoding="utf-8")
                except OSError:
                    pass

            self._send_json({
                "iterations": iterations,
                "loop_status": loop_status,
                "learnings": learnings,
                "total_iterations": len(iterations),
            })

        # ---- Live trade state ----

        def _serve_live_state(self):
            if _Handler._live_ref is not None and hasattr(_Handler._live_ref, "_state"):
                data = _Handler._live_ref._state.get()
            else:
                data = {
                    "running": False, "done": False, "bar_index": 0,
                    "total_bars": 0, "pct_complete": 0.0, "equity": 10000.0,
                    "initial_balance": 10000.0, "equity_history": [],
                    "n_trades": 0, "n_wins": 0, "n_losses": 0,
                    "win_rate": 0.0, "total_return_pct": 0.0,
                    "max_dd_pct": 0.0, "sharpe": 0.0,
                    "learning_phase": "mechanical",
                    "recent_trades": [], "instrument": "XAUUSD",
                }
            self._send_json(data)

        def _serve_candles(self, params):
            if _Handler._live_ref is not None and hasattr(_Handler._live_ref, "_candle_buffer"):
                tf = params.get("tf", ["5m"])[0]
                try:
                    since = int(params.get("since", ["0"])[0])
                except (ValueError, TypeError):
                    since = 0
                data = _Handler._live_ref._candle_buffer.get_candles(tf, since)
            else:
                data = []
            self._send_json(data)

        def _serve_trades(self, params):
            if _Handler._live_ref is not None and hasattr(_Handler._live_ref, "_candle_buffer"):
                try:
                    since_idx = max(0, int(params.get("since", ["0"])[0]))
                except (ValueError, TypeError):
                    since_idx = 0
                data = _Handler._live_ref._candle_buffer.get_trades(since_idx)
            else:
                data = {"trades": [], "next_index": 0}
            self._send_json(data)

        # ---- Discovery state ----

        def _serve_discovery_state(self):
            """Serve discovery agent findings for the dashboard."""
            from src.discovery.window_report import WindowReportGenerator

            reports_dir = str(self._reports / "discovery")
            try:
                gen = WindowReportGenerator(reports_dir=reports_dir)
                payload = gen.get_dashboard_payload()
                self._send_json(payload)
            except Exception as exc:
                self._send_json({"discovery": {}, "error": str(exc)})

        # ---- HTML ----

        def _serve_html(self):
            html_path = Path(__file__).parent / "optimization_dashboard.html"
            try:
                mtime = html_path.stat().st_mtime
                if mtime != _Handler._html_cache["mtime"]:
                    _Handler._html_cache["content"] = html_path.read_text(encoding="utf-8")
                    _Handler._html_cache["mtime"] = mtime
                html = _Handler._html_cache["content"].encode("utf-8")
            except OSError:
                self.send_error(500, "Dashboard HTML not found")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(html)

        def log_message(self, format, *args):
            pass
    return _Handler
