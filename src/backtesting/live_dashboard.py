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
import time
import webbrowser
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional

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
        }

    def update(self, updates: Dict[str, Any]) -> None:
        with self._lock:
            self._state.update(updates)

    def get(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._state)


# ---------------------------------------------------------------------------
# HTTP Request Handler
# ---------------------------------------------------------------------------

class _DashboardHandler(BaseHTTPRequestHandler):
    """Serves the dashboard HTML and JSON state API."""

    # Assigned by the server before handling requests
    state: _DashboardState

    def do_GET(self):
        if self.path == "/api/state":
            self._serve_json()
        elif self.path == "/" or self.path == "/index.html":
            self._serve_html()
        else:
            self.send_error(404)

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
        html = _DASHBOARD_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(html)

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
    """

    def __init__(self, port: int = 8501, auto_open: bool = True) -> None:
        self._port = port
        self._auto_open = auto_open
        self._state = _DashboardState()
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the HTTP server in a background thread."""
        handler_class = type(
            "_Handler",
            (_DashboardHandler,),
            {"state": self._state},
        )

        try:
            self._server = HTTPServer(("127.0.0.1", self._port), handler_class)
        except OSError:
            # Port in use — try next
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


# ---------------------------------------------------------------------------
# Dashboard HTML (single-page app with inline JS)
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Itchy Tradebot - Live Backtest</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    background: #0f0f23;
    color: #e0e0e0;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Consolas', monospace;
    padding: 16px;
    overflow-x: hidden;
}
.container { max-width: 1400px; margin: 0 auto; }

/* Header */
.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 20px;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 12px;
    border: 1px solid #333;
    margin-bottom: 16px;
}
.header h1 { font-size: 20px; color: #00d2ff; letter-spacing: -0.5px; }
.header .status {
    display: flex; align-items: center; gap: 8px;
    font-size: 13px;
}
.pulse {
    width: 10px; height: 10px; border-radius: 50%;
    background: #00ff88;
    animation: pulse 1.5s ease-in-out infinite;
}
.pulse.done { background: #00d2ff; animation: none; }
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.3); }
}

/* Progress bar */
.progress-container {
    background: #1a1a2e;
    border-radius: 8px;
    border: 1px solid #333;
    padding: 12px 16px;
    margin-bottom: 16px;
}
.progress-bar-outer {
    width: 100%; height: 6px;
    background: #333; border-radius: 3px;
    margin-top: 6px;
}
.progress-bar-inner {
    height: 100%; border-radius: 3px;
    background: linear-gradient(90deg, #00d2ff, #00ff88);
    transition: width 0.3s ease;
}
.progress-text {
    display: flex; justify-content: space-between;
    font-size: 12px; color: #888;
}
.progress-text span { color: #00d2ff; font-weight: 600; }

/* Metrics grid */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 10px;
    margin-bottom: 16px;
}
.metric {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #333;
    border-radius: 10px;
    padding: 14px 10px;
    text-align: center;
}
.metric .val {
    font-size: 22px; font-weight: 700;
    letter-spacing: -0.5px;
    transition: color 0.3s;
}
.metric .lbl {
    font-size: 10px; color: #666;
    text-transform: uppercase; letter-spacing: 1px;
    margin-top: 2px;
}

/* Charts row */
.charts-row {
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 12px;
    margin-bottom: 16px;
}
.chart-box {
    background: #1a1a2e;
    border: 1px solid #333;
    border-radius: 10px;
    padding: 14px;
}
.chart-box h3 {
    font-size: 12px; color: #888;
    text-transform: uppercase; letter-spacing: 1px;
    margin-bottom: 10px;
}
canvas { width: 100%; height: 200px; }

/* Learning phase bar */
.phase-bar {
    display: flex; height: 28px;
    border-radius: 6px; overflow: hidden;
    margin-top: 8px;
}
.phase-segment {
    display: flex; align-items: center; justify-content: center;
    font-size: 10px; font-weight: 600; color: white;
    transition: flex-grow 0.5s ease;
}
.phase-mechanical { background: #ff6b6b; }
.phase-statistical { background: #ffd93d; color: #333; }
.phase-similarity { background: #6bcb77; }

/* Gauges */
.gauges-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 16px;
}
.gauge {
    background: #1a1a2e;
    border: 1px solid #333;
    border-radius: 10px;
    padding: 12px;
    text-align: center;
}
.gauge .gauge-label { font-size: 11px; color: #888; margin-bottom: 6px; }
.gauge .gauge-value { font-size: 18px; font-weight: 700; margin-bottom: 6px; }
.gauge-bar-outer {
    width: 100%; height: 6px;
    background: #333; border-radius: 3px;
}
.gauge-bar-inner {
    height: 100%; border-radius: 3px;
    transition: width 0.3s ease, background 0.3s ease;
}

/* Trade log */
.trade-log {
    background: #1a1a2e;
    border: 1px solid #333;
    border-radius: 10px;
    padding: 14px;
    max-height: 240px;
    overflow-y: auto;
}
.trade-log h3 {
    font-size: 12px; color: #888;
    text-transform: uppercase; letter-spacing: 1px;
    margin-bottom: 10px;
}
.trade-entry {
    display: flex; justify-content: space-between;
    padding: 6px 8px;
    border-radius: 4px;
    font-size: 12px;
    margin-bottom: 2px;
}
.trade-entry.win { background: rgba(0, 255, 136, 0.08); }
.trade-entry.loss { background: rgba(255, 68, 68, 0.08); }
.trade-entry .r-val { font-weight: 700; }
.trade-entry .r-val.pos { color: #00ff88; }
.trade-entry .r-val.neg { color: #ff4444; }

@media (max-width: 800px) {
    .charts-row { grid-template-columns: 1fr; }
    .gauges-row { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<div class="container">

<div class="header">
    <h1>Itchy Tradebot</h1>
    <div class="status">
        <div class="pulse" id="pulse"></div>
        <span id="status-text">Connecting...</span>
    </div>
</div>

<div class="progress-container">
    <div class="progress-text">
        <span>Bar <span id="bar-idx">0</span> / <span id="bar-total">0</span></span>
        <span><span id="pct">0</span>% complete</span>
        <span>Elapsed: <span id="elapsed">0s</span></span>
    </div>
    <div class="progress-bar-outer">
        <div class="progress-bar-inner" id="progress-bar" style="width: 0%"></div>
    </div>
</div>

<div class="metrics-grid">
    <div class="metric"><div class="val" id="m-trades" style="color:#00d2ff">0</div><div class="lbl">Trades</div></div>
    <div class="metric"><div class="val" id="m-winrate" style="color:#888">0.0%</div><div class="lbl">Win Rate</div></div>
    <div class="metric"><div class="val" id="m-return" style="color:#888">0.00%</div><div class="lbl">Return</div></div>
    <div class="metric"><div class="val" id="m-sharpe" style="color:#888">0.00</div><div class="lbl">Sharpe</div></div>
    <div class="metric"><div class="val" id="m-maxdd" style="color:#888">0.00%</div><div class="lbl">Max DD</div></div>
    <div class="metric"><div class="val" id="m-expectancy" style="color:#888">0.000R</div><div class="lbl">Expectancy</div></div>
    <div class="metric"><div class="val" id="m-signals" style="color:#00d2ff">0</div><div class="lbl">Signals</div></div>
    <div class="metric"><div class="val" id="m-phase" style="color:#a78bfa">Mechanical</div><div class="lbl">Learning</div></div>
</div>

<div class="charts-row">
    <div class="chart-box">
        <h3>Equity Curve</h3>
        <canvas id="equity-chart" height="200"></canvas>
    </div>
    <div class="chart-box">
        <h3>Learning Phase</h3>
        <div style="margin-bottom:12px;">
            <div style="font-size:11px;color:#888;margin-bottom:4px;">
                <span id="phase-count">0</span> trades completed
            </div>
            <div class="phase-bar">
                <div class="phase-segment phase-mechanical" id="seg-mech" style="flex-grow:1">Mechanical</div>
                <div class="phase-segment phase-statistical" id="seg-stat" style="flex-grow:0"></div>
                <div class="phase-segment phase-similarity" id="seg-sim" style="flex-grow:0"></div>
            </div>
        </div>
        <h3 style="margin-top:16px;">Win/Loss</h3>
        <canvas id="winloss-chart" height="80"></canvas>
    </div>
</div>

<div class="gauges-row">
    <div class="gauge">
        <div class="gauge-label">Profit Target (8%)</div>
        <div class="gauge-value" id="g-profit" style="color:#00d2ff">0.00%</div>
        <div class="gauge-bar-outer"><div class="gauge-bar-inner" id="gb-profit" style="width:0%;background:#00d2ff"></div></div>
    </div>
    <div class="gauge">
        <div class="gauge-label">Daily DD Limit (5%)</div>
        <div class="gauge-value" id="g-daily-dd" style="color:#00ff88">0.00%</div>
        <div class="gauge-bar-outer"><div class="gauge-bar-inner" id="gb-daily" style="width:0%;background:#00ff88"></div></div>
    </div>
    <div class="gauge">
        <div class="gauge-label">Total DD Limit (10%)</div>
        <div class="gauge-value" id="g-total-dd" style="color:#00ff88">0.00%</div>
        <div class="gauge-bar-outer"><div class="gauge-bar-inner" id="gb-total" style="width:0%;background:#00ff88"></div></div>
    </div>
</div>

<div class="trade-log">
    <h3>Recent Trades</h3>
    <div id="trade-list"></div>
</div>

</div>

<script>
const POLL_MS = 500;
let polling = true;
let equityHistory = [];

function formatTime(s) {
    if (s < 60) return s.toFixed(0) + 's';
    if (s < 3600) return (s / 60).toFixed(1) + 'm';
    return (s / 3600).toFixed(1) + 'h';
}

function colorForWR(wr) {
    if (wr >= 0.55) return '#00ff88';
    if (wr >= 0.45) return '#ffd93d';
    return '#ff6b6b';
}

function colorForReturn(r) {
    if (r >= 8) return '#00ff88';
    if (r >= 0) return '#00d2ff';
    return '#ff6b6b';
}

function colorForDD(dd) {
    dd = Math.abs(dd);
    if (dd < 3) return '#00ff88';
    if (dd < 6) return '#ffd93d';
    return '#ff6b6b';
}

function drawEquityChart(canvas, history, initial) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.offsetWidth * 2;
    const h = canvas.height = 400;
    ctx.clearRect(0, 0, w, h);

    if (history.length < 2) return;

    const pcts = history.map(v => (v / initial - 1) * 100);
    const minP = Math.min(...pcts, -10);
    const maxP = Math.max(...pcts, 8);
    const range = maxP - minP || 1;
    const pad = 20;

    function x(i) { return pad + (i / (pcts.length - 1)) * (w - 2 * pad); }
    function y(v) { return pad + (1 - (v - minP) / range) * (h - 2 * pad); }

    // Fill area
    ctx.beginPath();
    ctx.moveTo(x(0), y(0));
    for (let i = 0; i < pcts.length; i++) ctx.lineTo(x(i), y(pcts[i]));
    ctx.lineTo(x(pcts.length - 1), y(0));
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, 'rgba(0, 210, 255, 0.3)');
    grad.addColorStop(1, 'rgba(0, 210, 255, 0.02)');
    ctx.fillStyle = grad;
    ctx.fill();

    // Line
    ctx.beginPath();
    for (let i = 0; i < pcts.length; i++) {
        if (i === 0) ctx.moveTo(x(i), y(pcts[i]));
        else ctx.lineTo(x(i), y(pcts[i]));
    }
    ctx.strokeStyle = '#00d2ff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Target line (8%)
    ctx.beginPath();
    ctx.moveTo(pad, y(8)); ctx.lineTo(w - pad, y(8));
    ctx.strokeStyle = 'rgba(0, 255, 136, 0.5)';
    ctx.lineWidth = 1; ctx.setLineDash([6, 4]);
    ctx.stroke(); ctx.setLineDash([]);

    // DD limit (-10%)
    ctx.beginPath();
    ctx.moveTo(pad, y(-10)); ctx.lineTo(w - pad, y(-10));
    ctx.strokeStyle = 'rgba(255, 68, 68, 0.5)';
    ctx.lineWidth = 1; ctx.setLineDash([6, 4]);
    ctx.stroke(); ctx.setLineDash([]);

    // Zero line
    ctx.beginPath();
    ctx.moveTo(pad, y(0)); ctx.lineTo(w - pad, y(0));
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.lineWidth = 1; ctx.setLineDash([]);
    ctx.stroke();

    // Current value label
    const last = pcts[pcts.length - 1];
    ctx.font = 'bold 24px Segoe UI';
    ctx.fillStyle = last >= 0 ? '#00ff88' : '#ff6b6b';
    ctx.textAlign = 'right';
    ctx.fillText((last >= 0 ? '+' : '') + last.toFixed(2) + '%', w - pad, pad + 20);
}

function drawWinLossChart(canvas, wins, losses) {
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.offsetWidth * 2;
    const h = canvas.height = 160;
    ctx.clearRect(0, 0, w, h);

    const total = wins + losses;
    if (total === 0) return;

    const winPct = wins / total;
    const barH = 30;
    const cy = h / 2;

    // Win bar
    ctx.fillStyle = '#00ff88';
    ctx.fillRect(0, cy - barH / 2, w * winPct, barH);

    // Loss bar
    ctx.fillStyle = '#ff4444';
    ctx.fillRect(w * winPct, cy - barH / 2, w * (1 - winPct), barH);

    // Labels
    ctx.font = 'bold 20px Segoe UI';
    ctx.textAlign = 'left';
    ctx.fillStyle = '#fff';
    if (winPct > 0.15) ctx.fillText(wins + 'W', 8, cy + 7);
    ctx.textAlign = 'right';
    if (winPct < 0.85) ctx.fillText(losses + 'L', w - 8, cy + 7);
}

async function poll() {
    try {
        const res = await fetch('/api/state');
        const s = await res.json();

        // Status
        document.getElementById('status-text').textContent =
            s.done ? 'Complete' : (s.running ? 'Running...' : 'Waiting...');
        document.getElementById('pulse').className = s.done ? 'pulse done' : 'pulse';

        // Progress
        document.getElementById('bar-idx').textContent = s.bar_index;
        document.getElementById('bar-total').textContent = s.total_bars;
        document.getElementById('pct').textContent = s.pct_complete.toFixed(1);
        document.getElementById('elapsed').textContent = formatTime(s.elapsed_seconds);
        document.getElementById('progress-bar').style.width = s.pct_complete + '%';

        // Metrics
        document.getElementById('m-trades').textContent = s.n_trades;
        const wr = s.win_rate;
        document.getElementById('m-winrate').textContent = (wr * 100).toFixed(1) + '%';
        document.getElementById('m-winrate').style.color = colorForWR(wr);
        document.getElementById('m-return').textContent = (s.total_return_pct >= 0 ? '+' : '') + s.total_return_pct.toFixed(2) + '%';
        document.getElementById('m-return').style.color = colorForReturn(s.total_return_pct);
        document.getElementById('m-sharpe').textContent = s.sharpe.toFixed(2);
        document.getElementById('m-sharpe').style.color = s.sharpe >= 1 ? '#00ff88' : s.sharpe >= 0.5 ? '#ffd93d' : '#ff6b6b';
        document.getElementById('m-maxdd').textContent = s.max_dd_pct.toFixed(2) + '%';
        document.getElementById('m-maxdd').style.color = colorForDD(s.max_dd_pct);
        document.getElementById('m-expectancy').textContent = s.expectancy.toFixed(3) + 'R';
        document.getElementById('m-signals').textContent = s.total_signals;
        document.getElementById('m-phase').textContent = s.learning_phase.charAt(0).toUpperCase() + s.learning_phase.slice(1);

        // Equity chart
        if (s.equity_history && s.equity_history.length > 0) {
            equityHistory = s.equity_history;
        }
        drawEquityChart(document.getElementById('equity-chart'), equityHistory, s.initial_balance);

        // Win/Loss chart
        drawWinLossChart(document.getElementById('winloss-chart'), s.n_wins, s.n_losses);

        // Learning phases
        document.getElementById('phase-count').textContent = s.learning_trades;
        const lt = s.learning_trades;
        const mechW = Math.min(lt, 100);
        const statW = Math.max(0, Math.min(lt - 100, 400));
        const simW = Math.max(0, lt - 500);
        document.getElementById('seg-mech').style.flexGrow = Math.max(mechW, 0.1);
        document.getElementById('seg-stat').style.flexGrow = statW;
        document.getElementById('seg-sim').style.flexGrow = simW;
        if (statW > 0) document.getElementById('seg-stat').textContent = 'Statistical';
        if (simW > 0) document.getElementById('seg-sim').textContent = 'Similarity';

        // Prop firm gauges
        document.getElementById('g-profit').textContent = s.prop_profit_pct.toFixed(2) + '%';
        const profitPct = Math.min(Math.max(s.prop_profit_pct / 8 * 100, 0), 100);
        document.getElementById('gb-profit').style.width = profitPct + '%';
        document.getElementById('gb-profit').style.background = s.prop_profit_pct >= 8 ? '#00ff88' : '#00d2ff';

        const dailyDD = Math.abs(s.worst_daily_dd_pct);
        document.getElementById('g-daily-dd').textContent = '-' + dailyDD.toFixed(2) + '%';
        const dailyPct = Math.min(dailyDD / 5 * 100, 100);
        document.getElementById('gb-daily').style.width = dailyPct + '%';
        const ddCol = dailyDD >= 4 ? '#ff4444' : dailyDD >= 2.5 ? '#ffd93d' : '#00ff88';
        document.getElementById('gb-daily').style.background = ddCol;
        document.getElementById('g-daily-dd').style.color = ddCol;

        const totalDD = Math.abs(s.max_dd_pct);
        document.getElementById('g-total-dd').textContent = '-' + totalDD.toFixed(2) + '%';
        const totalPct = Math.min(totalDD / 10 * 100, 100);
        document.getElementById('gb-total').style.width = totalPct + '%';
        const tdCol = totalDD >= 8 ? '#ff4444' : totalDD >= 5 ? '#ffd93d' : '#00ff88';
        document.getElementById('gb-total').style.background = tdCol;
        document.getElementById('g-total-dd').style.color = tdCol;

        // Trade log
        const logEl = document.getElementById('trade-list');
        if (s.recent_trades && s.recent_trades.length > 0) {
            logEl.innerHTML = s.recent_trades.map(t => {
                const r = parseFloat(t.r || 0);
                const cls = r > 0 ? 'win' : 'loss';
                const rCls = r > 0 ? 'pos' : 'neg';
                return '<div class="trade-entry ' + cls + '">' +
                    '<span>#' + (t.id || '?') + ' ' + (t.dir || '') + ' @ ' + (t.entry || '?') + '</span>' +
                    '<span class="r-val ' + rCls + '">' + (r > 0 ? '+' : '') + r.toFixed(2) + 'R</span>' +
                    '</div>';
            }).reverse().join('');
        }

        if (s.done) polling = false;

    } catch (e) {
        document.getElementById('status-text').textContent = 'Reconnecting...';
    }

    if (polling) setTimeout(poll, POLL_MS);
}

poll();
</script>
</body>
</html>"""
