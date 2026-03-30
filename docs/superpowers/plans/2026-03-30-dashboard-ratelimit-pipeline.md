# Trade Detail View + Rate Limit Fix + Pipeline Debug

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add clickable trade snapshots with candlestick charts to the dashboard, fix rate-limit detection in the optimization loop with Codex fallback, and add pipeline logging to diagnose the 154-signal → 30-trade funnel leak.

**Architecture:** Three independent tracks that can be parallelized. Track 1 is pure frontend (HTML/JS in the unified dashboard). Track 2 modifies `_spawn_claude()` with retry/fallback logic. Track 3 adds structured logging to the backtest engine's signal pipeline.

**Tech Stack:** lightweight-charts v5.1.0 (CDN), Python subprocess, logging module

---

## File Map

| Track | File | Action | Responsibility |
|-------|------|--------|---------------|
| 1 | `src/backtesting/optimization_dashboard.html` | Modify | Add trade detail UI (list + chart + TF tabs) |
| 2 | `scripts/run_optimization_loop.py` | Modify | Rate-limit detection, retry, Codex fallback |
| 2 | `tests/test_optimization_loop.py` | Modify | Add tests for rate-limit + fallback |
| 3 | `src/backtesting/vectorbt_engine.py` | Modify | Add pipeline stage counters |

---

## Track 1: Trade Detail View

### Task 1: Add lightweight-charts and trade detail UI structure

**Files:**
- Modify: `src/backtesting/optimization_dashboard.html`

- [ ] **Step 1: Add lightweight-charts CDN script tag**

Add before the closing `</body>` tag, before the existing `<script>` block:

```html
<script src="https://unpkg.com/lightweight-charts@5.1.0/dist/lightweight-charts.standalone.production.js"></script>
```

- [ ] **Step 2: Add CSS for trade detail panel**

Add these styles inside the existing `<style>` block, after the `.trade-chip.short` rule (line ~219):

```css
/* ── Trade Detail View ── */
#trade-overlay {
  display: none;
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0,0,0,0.85);
  z-index: 1000;
  overflow-y: auto;
}
#trade-overlay.visible { display: block; }

#trade-panel {
  max-width: 1200px;
  margin: 24px auto;
  padding: 20px;
}

#trade-panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}
#trade-panel-header h2 { color: var(--accent); font-size: 16px; }

.close-btn {
  background: transparent;
  border: 1px solid var(--border);
  color: var(--muted);
  padding: 4px 12px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}
.close-btn:hover { color: var(--text); border-color: var(--text); }

/* Trade list table */
#trade-list-summary {
  display: flex;
  gap: 16px;
  margin-bottom: 12px;
  flex-wrap: wrap;
}
#trade-list-summary .stat {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 8px 14px;
}
#trade-list-summary .stat .label { font-size: 10px; color: var(--muted); text-transform: uppercase; }
#trade-list-summary .stat .val { font-size: 16px; font-weight: 700; margin-top: 2px; }

#trade-list-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
  margin-bottom: 16px;
}
#trade-list-table th {
  text-align: left;
  padding: 6px 8px;
  color: var(--muted);
  border-bottom: 1px solid var(--border);
  font-weight: 600;
  cursor: pointer;
}
#trade-list-table th:hover { color: var(--accent); }
#trade-list-table td { padding: 6px 8px; border-bottom: 1px solid rgba(255,255,255,0.04); }
#trade-list-table tr { cursor: pointer; transition: background 0.15s; }
#trade-list-table tr:hover { background: rgba(0,210,255,0.08); }
#trade-list-table tr.selected { background: rgba(0,210,255,0.15); }

/* Chart section */
#trade-chart-section {
  display: none;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 12px;
}
#trade-chart-section.visible { display: block; }

#trade-chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}
#trade-chart-title { font-size: 13px; color: var(--accent); }

.tf-tabs {
  display: flex;
  gap: 4px;
}
.tf-tab {
  background: transparent;
  border: 1px solid var(--border);
  color: var(--muted);
  padding: 4px 12px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  transition: all 0.15s;
}
.tf-tab:hover { color: var(--text); border-color: var(--text); }
.tf-tab.active { color: var(--accent); border-color: var(--accent); background: rgba(0,210,255,0.1); }

#trade-chart-container { width: 100%; height: 450px; }

#trade-chart-info {
  display: flex;
  gap: 16px;
  margin-top: 8px;
  font-size: 11px;
  color: var(--muted);
}
```

- [ ] **Step 3: Add HTML structure for trade overlay**

Add before the closing `</body>` tag (before the scripts):

```html
<!-- ═══════ TRADE DETAIL OVERLAY ═══════ -->
<div id="trade-overlay">
  <div id="trade-panel">
    <div id="trade-panel-header">
      <h2>Trade History</h2>
      <button class="close-btn" onclick="closeTradeOverlay()">&times; Close</button>
    </div>
    <div id="trade-list-summary"></div>
    <table id="trade-list-table">
      <thead><tr>
        <th data-sort="entry_time">Time</th>
        <th data-sort="direction">Dir</th>
        <th data-sort="entry_price">Entry</th>
        <th data-sort="exit_price">Exit</th>
        <th data-sort="original_stop">SL</th>
        <th data-sort="take_profit">TP</th>
        <th data-sort="r_multiple">R</th>
        <th data-sort="pnl_points">P&L</th>
        <th>Reason</th>
      </tr></thead>
      <tbody id="trade-list-body"></tbody>
    </table>
    <div id="trade-chart-section">
      <div id="trade-chart-header">
        <span id="trade-chart-title">Trade Snapshot</span>
        <div class="tf-tabs">
          <button class="tf-tab active" data-tf="5m">5M</button>
          <button class="tf-tab" data-tf="15m">15M</button>
          <button class="tf-tab" data-tf="1h">1H</button>
          <button class="tf-tab" data-tf="4h">4H</button>
        </div>
      </div>
      <div id="trade-chart-container"></div>
      <div id="trade-chart-info"></div>
    </div>
  </div>
</div>
```

- [ ] **Step 4: Make the Trades metric box clickable**

Change the Trades metric box div (line ~248) to add a click handler and cursor style:

```html
<div class="metric-box" style="cursor:pointer" onclick="openTradeOverlay()" title="Click to view trade details">
  <div class="label">Trades</div>
  <div class="value" id="lv-trades">0</div>
</div>
```

- [ ] **Step 5: Verify HTML renders** — Open http://localhost:8501, confirm no JS errors in console, click Trades box shows empty overlay.

### Task 2: Implement trade list fetching and rendering

**Files:**
- Modify: `src/backtesting/optimization_dashboard.html` (script section)

- [ ] **Step 1: Add trade overlay open/close + fetch logic**

Add these functions inside the `<script>` block (after the existing `pollOpt` function):

```javascript
// ── Trade Detail View ──
let allTrades = [];
let selectedTradeIdx = -1;
let tradeChart = null;
let tradeSeries = null;

function openTradeOverlay() {
  document.getElementById('trade-overlay').classList.add('visible');
  fetchAllTrades();
}

function closeTradeOverlay() {
  document.getElementById('trade-overlay').classList.remove('visible');
  if (tradeChart) { tradeChart.remove(); tradeChart = null; tradeSeries = null; }
  document.getElementById('trade-chart-section').classList.remove('visible');
  selectedTradeIdx = -1;
}

// Close overlay on Escape key
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeTradeOverlay();
});

async function fetchAllTrades() {
  let trades = [];
  let idx = 0;
  // Paginate through all trades
  for (let i = 0; i < 100; i++) {
    try {
      const res = await fetch(`/api/trades?since=${idx}`);
      const d = await res.json();
      if (!d.trades || d.trades.length === 0) break;
      trades = trades.concat(d.trades);
      idx = d.next_index;
      if (d.trades.length < 50) break; // last page
    } catch(e) { break; }
  }
  allTrades = trades;
  renderTradeList();
}

function renderTradeList() {
  const trades = allTrades;
  // Summary stats
  const total = trades.length;
  const wins = trades.filter(t => (t.r_multiple || 0) > 0).length;
  const wr = total > 0 ? (wins / total * 100).toFixed(1) : '0.0';
  const avgR = total > 0 ? (trades.reduce((s, t) => s + (t.r_multiple || 0), 0) / total).toFixed(2) : '0.00';
  const totalPnl = trades.reduce((s, t) => s + (t.pnl_points || 0), 0).toFixed(2);
  const pf = (() => {
    const gross = trades.filter(t => (t.pnl_points||0) > 0).reduce((s,t) => s + t.pnl_points, 0);
    const loss = Math.abs(trades.filter(t => (t.pnl_points||0) < 0).reduce((s,t) => s + t.pnl_points, 0));
    return loss > 0 ? (gross / loss).toFixed(2) : gross > 0 ? '∞' : '0.00';
  })();

  document.getElementById('trade-list-summary').innerHTML = [
    {l: 'Total Trades', v: total},
    {l: 'Win Rate', v: wr + '%'},
    {l: 'Avg R', v: avgR + 'R'},
    {l: 'Total P&L', v: totalPnl + ' pts'},
    {l: 'Profit Factor', v: pf},
  ].map(s => `<div class="stat"><div class="label">${s.l}</div><div class="val">${s.v}</div></div>`).join('');

  // Trade rows
  const tbody = document.getElementById('trade-list-body');
  tbody.innerHTML = trades.map((t, i) => {
    const r = (t.r_multiple || 0).toFixed(2);
    const pnl = (t.pnl_points || 0).toFixed(2);
    const dir = t.direction || '?';
    const rColor = parseFloat(r) >= 0 ? 'var(--win)' : 'var(--loss)';
    return `<tr onclick="selectTrade(${i})" class="${i === selectedTradeIdx ? 'selected' : ''}">
      <td>${(t.entry_time || '').replace('T', ' ').slice(0, 19)}</td>
      <td><span class="trade-chip ${dir}">${dir.toUpperCase()}</span></td>
      <td>${(t.entry_price || 0).toFixed(2)}</td>
      <td>${(t.exit_price || 0).toFixed(2)}</td>
      <td>${(t.original_stop || t.stop_loss || 0).toFixed(2)}</td>
      <td>${(t.take_profit || 0).toFixed(2)}</td>
      <td style="color:${rColor};font-weight:700">${r}R</td>
      <td style="color:${rColor}">${pnl}</td>
      <td style="font-size:10px;color:var(--muted)">${t.reason || ''}</td>
    </tr>`;
  }).join('');
}
```

- [ ] **Step 2: Add column sorting**

Add after `renderTradeList`:

```javascript
// Column sorting
document.querySelectorAll('#trade-list-table th[data-sort]').forEach(th => {
  th.addEventListener('click', () => {
    const key = th.dataset.sort;
    const dir = th.dataset.dir === 'asc' ? 'desc' : 'asc';
    th.dataset.dir = dir;
    allTrades.sort((a, b) => {
      let va = a[key] ?? 0, vb = b[key] ?? 0;
      if (typeof va === 'string') return dir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
      return dir === 'asc' ? va - vb : vb - va;
    });
    renderTradeList();
  });
});
```

- [ ] **Step 3: Verify** — Click Trades box, confirm trade list renders with stats, column sorting works.

### Task 3: Implement candlestick chart with trade markers

**Files:**
- Modify: `src/backtesting/optimization_dashboard.html` (script section)

- [ ] **Step 1: Add trade selection and chart rendering**

Add after the sorting code:

```javascript
function selectTrade(idx) {
  selectedTradeIdx = idx;
  renderTradeList(); // re-highlight row
  const section = document.getElementById('trade-chart-section');
  section.classList.add('visible');
  const activeTf = document.querySelector('.tf-tab.active').dataset.tf;
  loadTradeChart(idx, activeTf);
}

async function loadTradeChart(tradeIdx, tf) {
  const trade = allTrades[tradeIdx];
  if (!trade) return;

  // Calculate time window: 100 bars before entry, 50 bars after exit
  const entryTs = new Date(trade.entry_time).getTime() / 1000;
  const exitTs = trade.exit_time ? new Date(trade.exit_time).getTime() / 1000 : entryTs + 3600;
  const tfSeconds = {
    '5m': 300, '15m': 900, '1h': 3600, '4h': 14400
  }[tf] || 300;
  const since = entryTs - (100 * tfSeconds);

  // Update chart title
  const dir = (trade.direction || '?').toUpperCase();
  const r = (trade.r_multiple || 0).toFixed(2);
  document.getElementById('trade-chart-title').textContent =
    `${dir} @ ${(trade.entry_price||0).toFixed(2)} → ${(trade.exit_price||0).toFixed(2)} (${r}R) — ${tf.toUpperCase()}`;

  // Fetch candles
  let candles = [];
  try {
    const res = await fetch(`/api/candles?tf=${tf}&since=${since}`);
    candles = await res.json();
  } catch(e) { return; }

  if (!Array.isArray(candles) || candles.length === 0) {
    document.getElementById('trade-chart-info').textContent = 'No candle data available for this timeframe.';
    return;
  }

  // Destroy old chart
  const container = document.getElementById('trade-chart-container');
  if (tradeChart) { tradeChart.remove(); tradeChart = null; }
  container.innerHTML = '';

  // Create chart
  tradeChart = LightweightCharts.createChart(container, {
    width: container.clientWidth,
    height: 450,
    layout: { background: { color: '#1a1a2e' }, textColor: '#888' },
    grid: {
      vertLines: { color: 'rgba(255,255,255,0.04)' },
      horzLines: { color: 'rgba(255,255,255,0.04)' },
    },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    timeScale: { timeVisible: true, secondsVisible: false },
  });

  // Add candlestick series
  tradeSeries = tradeChart.addCandlestickSeries({
    upColor: '#26a69a', downColor: '#ef5350',
    borderUpColor: '#26a69a', borderDownColor: '#ef5350',
    wickUpColor: '#26a69a', wickDownColor: '#ef5350',
  });

  // Parse candle data: [time, o, h, l, c, vol, tenkan, kijun, senkou_a, senkou_b, chikou]
  const candleData = candles
    .filter(c => c[0] && c[1] != null)
    .map(c => ({
      time: Math.floor(c[0]),
      open: c[1], high: c[2], low: c[3], close: c[4],
    }));

  if (candleData.length === 0) return;
  tradeSeries.setData(candleData);

  // Add Ichimoku cloud (senkou_a / senkou_b fill)
  const cloudA = candles.filter(c => c[8] != null).map(c => ({ time: Math.floor(c[0]), value: c[8] }));
  const cloudB = candles.filter(c => c[9] != null).map(c => ({ time: Math.floor(c[0]), value: c[9] }));
  if (cloudA.length > 0) {
    const lineA = tradeChart.addLineSeries({ color: 'rgba(0,210,255,0.3)', lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
    lineA.setData(cloudA);
  }
  if (cloudB.length > 0) {
    const lineB = tradeChart.addLineSeries({ color: 'rgba(255,167,38,0.3)', lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
    lineB.setData(cloudB);
  }

  // ── Trade marker lines ──
  const entryPrice = trade.entry_price;
  const exitPrice = trade.exit_price;
  const slPrice = trade.original_stop || trade.stop_loss;
  const tpPrice = trade.take_profit;
  const isWin = (trade.r_multiple || 0) > 0;

  // Entry line (green)
  if (entryPrice) {
    tradeSeries.createPriceLine({
      price: entryPrice,
      color: '#00ff88',
      lineWidth: 2,
      lineStyle: LightweightCharts.LineStyle.Solid,
      axisLabelVisible: true,
      title: 'Entry ' + entryPrice.toFixed(2),
    });
  }

  // Exit line
  if (exitPrice) {
    tradeSeries.createPriceLine({
      price: exitPrice,
      color: isWin ? '#00ff88' : '#ff4444',
      lineWidth: 2,
      lineStyle: LightweightCharts.LineStyle.Solid,
      axisLabelVisible: true,
      title: 'Exit ' + exitPrice.toFixed(2),
    });
  }

  // Stop loss line (dashed red)
  if (slPrice) {
    tradeSeries.createPriceLine({
      price: slPrice,
      color: '#ff4444',
      lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Dashed,
      axisLabelVisible: true,
      title: 'SL ' + slPrice.toFixed(2),
    });
  }

  // Take profit line (dashed green)
  if (tpPrice) {
    tradeSeries.createPriceLine({
      price: tpPrice,
      color: '#00d2ff',
      lineWidth: 1,
      lineStyle: LightweightCharts.LineStyle.Dashed,
      axisLabelVisible: true,
      title: 'TP ' + tpPrice.toFixed(2),
    });
  }

  // Fit chart to trade window
  const minTime = entryTs - (30 * tfSeconds);
  const maxTime = exitTs + (20 * tfSeconds);
  tradeChart.timeScale().setVisibleRange({
    from: minTime,
    to: maxTime,
  });

  // Trade info below chart
  const duration = trade.exit_time && trade.entry_time
    ? Math.round((new Date(trade.exit_time) - new Date(trade.entry_time)) / 60000)
    : 0;
  document.getElementById('trade-chart-info').innerHTML =
    `<span>Duration: ${duration}min</span>
     <span>Risk: ${slPrice ? Math.abs(entryPrice - slPrice).toFixed(2) : '?'} pts</span>
     <span>Reward: ${tpPrice ? Math.abs(tpPrice - entryPrice).toFixed(2) : '?'} pts</span>
     <span>Reason: ${trade.reason || 'N/A'}</span>`;

  // Handle resize
  const ro = new ResizeObserver(() => {
    if (tradeChart) tradeChart.applyOptions({ width: container.clientWidth });
  });
  ro.observe(container);
}

// TF tab switching
document.querySelectorAll('.tf-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tf-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    if (selectedTradeIdx >= 0) loadTradeChart(selectedTradeIdx, tab.dataset.tf);
  });
});
```

- [ ] **Step 2: Verify end-to-end** — Run a backtest, click Trades, click a trade row, verify chart renders with entry/exit/SL/TP lines. Switch timeframe tabs.

- [ ] **Step 3: Commit**

```bash
git add src/backtesting/optimization_dashboard.html
git commit -m "feat: add trade detail view with candlestick charts and TF tabs"
```

---

## Track 2: Rate Limit Detection + Codex Fallback

### Task 4: Add rate-limit detection and Codex fallback to _spawn_claude

**Files:**
- Modify: `scripts/run_optimization_loop.py:600-640`

- [ ] **Step 1: Write failing test for rate-limit detection**

Add to `tests/test_optimization_loop.py`:

```python
class TestRateLimitDetection:
    """Verify that _spawn_claude detects rate limits and retries."""

    def test_detects_rate_limit_in_output(self, basic_loop, monkeypatch):
        """When Claude returns rate-limit text, _spawn_claude retries once then returns output."""
        call_count = 0
        def fake_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = type('R', (), {'returncode': 0, 'stdout': '', 'stderr': ''})()
            if call_count == 1:
                result.stdout = "You've hit your limit - resets 2pm"
            else:
                result.stdout = "I'll change stop_multiplier to 1.8"
            return result
        monkeypatch.setattr("subprocess.run", fake_run)
        output = basic_loop._spawn_claude("test prompt")
        assert call_count == 2
        assert "stop_multiplier" in output

    def test_rate_limit_all_retries_exhausted(self, basic_loop, monkeypatch):
        """When all retries are rate-limited, returns empty string and sets rate_limited flag."""
        def fake_run(*args, **kwargs):
            result = type('R', (), {'returncode': 0, 'stdout': "You've hit your limit - resets 5pm", 'stderr': ''})()
            return result
        monkeypatch.setattr("subprocess.run", fake_run)
        monkeypatch.setattr("time.sleep", lambda s: None)  # skip actual sleep
        output = basic_loop._spawn_claude("test prompt")
        assert output == ""

    def test_codex_fallback_on_rate_limit(self, basic_loop, monkeypatch):
        """When Claude is rate-limited and codex is configured, falls back to codex."""
        call_args = []
        def fake_run(*args, **kwargs):
            call_args.append(args[0] if args else kwargs.get('args'))
            result = type('R', (), {'returncode': 0, 'stdout': '', 'stderr': ''})()
            cmd = args[0] if args else kwargs.get('args', [])
            if 'claude' in str(cmd):
                result.stdout = "You've hit your limit - resets 5pm"
            else:
                result.stdout = "Changed config parameter"
            return result
        monkeypatch.setattr("subprocess.run", fake_run)
        monkeypatch.setattr("time.sleep", lambda s: None)
        # Enable codex fallback
        basic_loop._config.setdefault("claude", {})["codex_fallback"] = True
        output = basic_loop._spawn_claude("test prompt")
        assert "Changed config parameter" in output
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python -m pytest tests/test_optimization_loop.py::TestRateLimitDetection -v
```

Expected: FAIL (no rate-limit logic exists yet)

- [ ] **Step 3: Implement rate-limit detection and Codex fallback**

Replace the `_spawn_claude` method (line 600) in `scripts/run_optimization_loop.py`:

```python
def _spawn_claude(self, prompt: str) -> str:
    """Run Claude CLI with *prompt* on stdin and return stdout.

    Detects rate-limit responses (\"You've hit your limit\") and retries
    after a delay.  When all retries fail and ``config.claude.codex_fallback``
    is enabled, falls back to ``codex exec --yolo``.

    Returns an empty string if Claude is not installed or times out.
    """
    import re
    import time

    claude_cfg = self._config.get("claude", {})
    command: List[str] = claude_cfg.get(
        "command", ["claude", "-p", "--dangerously-skip-permissions"]
    )
    timeout: int = int(claude_cfg.get("timeout_seconds", 300))
    max_retries: int = int(claude_cfg.get("rate_limit_retries", 1))
    codex_fallback: bool = bool(claude_cfg.get("codex_fallback", False))

    for attempt in range(1 + max_retries):
        try:
            completed = subprocess.run(
                command,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(_REPO_ROOT),
                encoding="utf-8",
            )
            output = completed.stdout or ""

            # Detect rate limit
            if re.search(r"(?i)you.ve hit your limit|rate.limit|resets?\s+\d", output):
                # Try to parse reset time for smarter sleep
                wait = 60  # default 60s
                m = re.search(r"resets?\s+(\d+)\s*m", output, re.IGNORECASE)
                if m:
                    wait = int(m.group(1)) * 60
                    wait = min(wait, 300)  # cap at 5 minutes

                if attempt < max_retries:
                    logger.warning(
                        "_spawn_claude: Rate limited (attempt %d/%d). Waiting %ds before retry.",
                        attempt + 1, 1 + max_retries, wait,
                    )
                    time.sleep(wait)
                    continue
                else:
                    logger.warning(
                        "_spawn_claude: Rate limited after %d attempts.", 1 + max_retries,
                    )
                    # Try codex fallback
                    if codex_fallback:
                        return self._spawn_codex_fallback(prompt, timeout)
                    return ""

            if completed.returncode != 0:
                logger.warning(
                    "_spawn_claude: Claude exited with code %d. stderr: %s",
                    completed.returncode,
                    completed.stderr[:500] if completed.stderr else "",
                )
            return output

        except FileNotFoundError:
            logger.error(
                "_spawn_claude: Claude CLI not found. "
                "Install with: npm install -g @anthropic-ai/claude-code"
            )
            if codex_fallback:
                return self._spawn_codex_fallback(prompt, timeout)
            return ""
        except subprocess.TimeoutExpired:
            logger.error("_spawn_claude: Claude CLI timed out after %ds.", timeout)
            return ""
        except OSError as exc:
            logger.error("_spawn_claude: OS error spawning Claude: %s", exc)
            return ""

    return ""

def _spawn_codex_fallback(self, prompt: str, timeout: int) -> str:
    """Fall back to Codex CLI when Claude is rate-limited.

    Uses ``codex exec --yolo`` with the same prompt.
    Returns empty string on failure.
    """
    logger.info("_spawn_codex_fallback: Attempting Codex CLI fallback.")
    try:
        completed = subprocess.run(
            ["codex", "exec", "--yolo", prompt[:8000]],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(_REPO_ROOT),
            encoding="utf-8",
        )
        output = completed.stdout or ""
        if completed.returncode != 0:
            logger.warning(
                "_spawn_codex_fallback: Codex exited with code %d. stderr: %s",
                completed.returncode,
                completed.stderr[:500] if completed.stderr else "",
            )
        else:
            logger.info("_spawn_codex_fallback: Codex returned %d chars.", len(output))
        return output
    except FileNotFoundError:
        logger.error("_spawn_codex_fallback: Codex CLI not found.")
        return ""
    except subprocess.TimeoutExpired:
        logger.error("_spawn_codex_fallback: Codex timed out after %ds.", timeout)
        return ""
    except OSError as exc:
        logger.error("_spawn_codex_fallback: OS error: %s", exc)
        return ""
```

- [ ] **Step 4: Add early-stop on sustained rate limiting in the main loop**

In the `run()` method, after `claude_output = self._spawn_claude(prompt)` (line ~297), add:

```python
# Detect sustained rate limiting — stop loop early
if not claude_output.strip():
    self._consecutive_empty += 1
    if self._consecutive_empty >= 2:
        stop_reason = "rate_limited"
        logger.warning("Two consecutive empty Claude outputs — likely rate limited. Stopping.")
        break
else:
    self._consecutive_empty = 0
```

And in `__init__`, add: `self._consecutive_empty: int = 0`

- [ ] **Step 5: Run tests**

```bash
.venv/Scripts/python -m pytest tests/test_optimization_loop.py -v
```

Expected: All tests pass including the new rate-limit tests.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_optimization_loop.py tests/test_optimization_loop.py
git commit -m "fix: detect rate limits in optimization loop, add Codex fallback"
```

---

## Track 3: Pipeline Signal Logging

### Task 5: Add pipeline stage counters to vectorbt_engine

**Files:**
- Modify: `src/backtesting/vectorbt_engine.py`

- [ ] **Step 1: Add pipeline counter dict to the engine's run() method**

Near the top of the `run()` method, after initial variable declarations, add:

```python
# Pipeline diagnostics — tracks signal attrition at each stage
_pipeline_counts = {
    "signals_generated": 0,
    "signals_filtered_edge": 0,
    "signals_filtered_confluence": 0,
    "signals_filtered_in_trade": 0,
    "signals_filtered_circuit_breaker": 0,
    "signals_entered": 0,
}
```

- [ ] **Step 2: Increment counters at each pipeline stage**

Where strategies generate signals (line ~412-435 area), increment `signals_generated` for each non-None signal.

Where signals are filtered by edges, increment `signals_filtered_edge`.

Where signals are filtered by min_confluence_score, increment `signals_filtered_confluence`.

Where signals are rejected because a trade is already open, increment `signals_filtered_in_trade`.

Where circuit breaker blocks entry, increment `signals_filtered_circuit_breaker`.

Where a trade is opened, increment `signals_entered`.

- [ ] **Step 3: Log pipeline summary at end of backtest**

After the main loop, before returning results:

```python
logger.info(
    "Pipeline: generated=%d | filtered_in_trade=%d | filtered_edge=%d | "
    "filtered_confluence=%d | filtered_breaker=%d | entered=%d",
    _pipeline_counts["signals_generated"],
    _pipeline_counts["signals_filtered_in_trade"],
    _pipeline_counts["signals_filtered_edge"],
    _pipeline_counts["signals_filtered_confluence"],
    _pipeline_counts["signals_filtered_circuit_breaker"],
    _pipeline_counts["signals_entered"],
)
```

- [ ] **Step 4: Include pipeline counts in the result metrics**

Add to the metrics dict before returning:

```python
metrics["pipeline_counts"] = _pipeline_counts
```

This way the optimization loop can include pipeline data in Claude's prompt for better analysis.

- [ ] **Step 5: Verify** — Run a backtest and check logs for pipeline breakdown.

- [ ] **Step 6: Commit**

```bash
git add src/backtesting/vectorbt_engine.py
git commit -m "feat: add pipeline signal logging to diagnose trade funnel"
```

---

## Execution Order

All three tracks are independent:
- **Track 1** (Tasks 1-3): Trade Detail View — pure frontend
- **Track 2** (Tasks 4): Rate limit + Codex fallback — backend
- **Track 3** (Task 5): Pipeline logging — engine instrumentation

Can be executed in parallel via subagent-driven development.
