"""Live optimization loop dashboard.

Polls reports directory for iteration JSON files and serves a
real-time dashboard showing pass rate progression, Claude's changes,
and current loop status.

No external dependencies -- uses only Python stdlib.
"""
from __future__ import annotations

import glob
import json
import logging
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class OptimizationDashboardServer:
    """Local HTTP server for live optimization monitoring.

    Parameters
    ----------
    port: Port to serve on. Default: 8502.
    reports_dir: Directory to scan for opt_iter_*.json reports.
    auto_open: Open browser automatically.
    """

    def __init__(self, port: int = 8502, reports_dir: str = "reports", auto_open: bool = True):
        self._port = port
        self._reports_dir = _REPO_ROOT / reports_dir
        self._auto_open = auto_open
        self._server = None
        self._thread = None

    def start(self):
        handler_class = _make_handler(self._reports_dir)
        self._server = HTTPServer(("127.0.0.1", self._port), handler_class)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        url = f"http://localhost:{self._port}"
        logger.info("Optimization dashboard at %s", url)
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


def _make_handler(reports_dir: Path):
    """Factory to create a handler class with the reports_dir baked in."""

    class _Handler(BaseHTTPRequestHandler):
        _reports = reports_dir
        _html_cache: Dict[str, Any] = {"content": "", "mtime": 0.0}

        def do_GET(self):
            if self.path == "/api/state":
                self._serve_state()
            elif self.path == "/" or self.path == "/index.html":
                self._serve_html()
            else:
                self.send_error(404)

        def _serve_state(self):
            # Scan for opt_iter_*.json files
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
                    })
                except (json.JSONDecodeError, OSError):
                    continue

            # Read loop_status.json if exists
            status_path = self._reports / "loop_status.json"
            loop_status: Dict[str, Any] = {}
            if status_path.exists():
                try:
                    loop_status = json.loads(status_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    pass

            # Read CLAUDE.md learnings
            claude_md = _REPO_ROOT / "CLAUDE.md"
            learnings = ""
            if claude_md.exists():
                try:
                    learnings = claude_md.read_text(encoding="utf-8")
                except OSError:
                    pass

            state = {
                "iterations": iterations,
                "loop_status": loop_status,
                "learnings": learnings,
                "total_iterations": len(iterations),
            }

            body = json.dumps(state).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

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
            pass  # Suppress request logging

    return _Handler
