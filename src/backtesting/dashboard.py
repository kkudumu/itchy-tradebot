"""
Self-contained HTML dashboard for backtest results.

Generates a single HTML file with embedded base64 matplotlib charts that
auto-opens in the default browser.  No external assets, no server needed.

Charts included:
1. Equity curve with prop firm limits
2. Trade entry/exit markers on price
3. Learning phase timeline
4. Win rate heatmap (session x regime)
5. Daily P&L bars
6. Prop firm tracking (profit vs target, DD vs limits)
"""

from __future__ import annotations

import base64
import io
import logging
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chart generation helpers
# ---------------------------------------------------------------------------

def _fig_to_base64(fig) -> str:
    """Convert a matplotlib Figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="#1a1a2e", edgecolor="none")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    import matplotlib.pyplot as plt
    plt.close(fig)
    return b64


def _import_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------------------
# Individual chart builders
# ---------------------------------------------------------------------------

def chart_equity_curve(
    equity_curve: pd.Series,
    initial_balance: float,
    profit_target_pct: float = 8.0,
    max_total_dd_pct: float = 10.0,
) -> str:
    """Equity curve with prop firm limit lines. Returns base64 PNG."""
    plt = _import_plt()
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    if equity_curve.empty:
        ax.text(0.5, 0.5, "No equity data", ha="center", va="center",
                color="white", fontsize=16, transform=ax.transAxes)
        return _fig_to_base64(fig)

    # Plot equity
    pct = (equity_curve / initial_balance - 1.0) * 100.0
    ax.fill_between(pct.index, 0, pct.values, alpha=0.3,
                    color="#00d2ff", linewidth=0)
    ax.plot(pct.index, pct.values, color="#00d2ff", linewidth=1.5,
            label="Equity")

    # Reference lines
    ax.axhline(profit_target_pct, color="#00ff88", linewidth=1.5,
               linestyle="--", alpha=0.8, label=f"Target (+{profit_target_pct}%)")
    ax.axhline(-max_total_dd_pct, color="#ff4444", linewidth=1.5,
               linestyle="--", alpha=0.8, label=f"DD Limit (-{max_total_dd_pct}%)")
    ax.axhline(0, color="#ffffff", linewidth=0.5, alpha=0.3)

    # Styling
    ax.set_title("Equity Curve", color="white", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Account Change (%)", color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=8)
    ax.legend(loc="upper left", fontsize=8, facecolor="#16213e",
              edgecolor="#333", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.grid(True, alpha=0.15, color="white")

    return _fig_to_base64(fig)


def chart_trades_on_price(
    trades: List[dict],
    equity_curve: pd.Series,
) -> str:
    """Price series with trade entry/exit markers. Returns base64 PNG."""
    plt = _import_plt()
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    if not trades:
        ax.text(0.5, 0.5, "No trades", ha="center", va="center",
                color="white", fontsize=16, transform=ax.transAxes)
        return _fig_to_base64(fig)

    # Extract entry/exit prices and times
    entries_x, entries_y = [], []
    exits_x, exits_y = [], []
    wins, losses = [], []

    for t in trades:
        entry_price = t.get("entry_price")
        exit_price = t.get("exit_price")
        entry_time = t.get("entry_time")
        exit_time = t.get("exit_time")
        r_mult = float(t.get("r_multiple") or 0.0)

        if entry_price is not None:
            entries_x.append(entry_time or len(entries_x))
            entries_y.append(float(entry_price))
        if exit_price is not None:
            exits_x.append(exit_time or len(exits_x))
            exits_y.append(float(exit_price))
            if r_mult > 0:
                wins.append((exit_time or len(wins), float(exit_price)))
            else:
                losses.append((exit_time or len(losses), float(exit_price)))

    # Plot trade entries
    if entries_y:
        ax.scatter(range(len(entries_y)), entries_y, marker="^", color="#00d2ff",
                   s=30, zorder=5, alpha=0.7, label=f"Entries ({len(entries_y)})")

    # Plot wins and losses
    if wins:
        ax.scatter(range(len(wins)), [w[1] for w in wins], marker="v",
                   color="#00ff88", s=30, zorder=5, alpha=0.7,
                   label=f"Wins ({len(wins)})")
    if losses:
        ax.scatter([len(wins) + i for i in range(len(losses))],
                   [l[1] for l in losses], marker="v",
                   color="#ff4444", s=30, zorder=5, alpha=0.7,
                   label=f"Losses ({len(losses)})")

    ax.set_title("Trade Distribution", color="white", fontsize=14,
                 fontweight="bold", pad=12)
    ax.set_ylabel("Price", color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=8)
    ax.legend(loc="upper left", fontsize=8, facecolor="#16213e",
              edgecolor="#333", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.grid(True, alpha=0.15, color="white")

    return _fig_to_base64(fig)


def chart_learning_phases(
    trades: List[dict],
    total_trades: int,
) -> str:
    """Learning phase progression timeline. Returns base64 PNG."""
    plt = _import_plt()
    fig, ax = plt.subplots(figsize=(14, 2.5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    phases = [
        ("Mechanical\n(0-99)", 0, min(100, total_trades), "#ff6b6b"),
        ("Statistical\n(100-499)", 100, min(500, max(100, total_trades)), "#ffd93d"),
        ("Similarity\n(500+)", 500, max(500, total_trades), "#6bcb77"),
    ]

    for label, start, end, color in phases:
        if end > start:
            width = end - start
            ax.barh(0, width, left=start, height=0.6, color=color,
                    alpha=0.8, edgecolor="white", linewidth=0.5)
            mid = start + width / 2
            if width > 30:
                ax.text(mid, 0, label, ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold")

    # Current position marker
    ax.axvline(total_trades, color="white", linewidth=2, linestyle="-",
               label=f"Current: {total_trades} trades")

    ax.set_xlim(0, max(total_trades * 1.1, 550))
    ax.set_yticks([])
    ax.set_xlabel("Trades Completed", color="white", fontsize=10)
    ax.set_title("Learning Phase Progression", color="white", fontsize=14,
                 fontweight="bold", pad=12)
    ax.tick_params(colors="white", labelsize=8)
    ax.legend(loc="upper right", fontsize=8, facecolor="#16213e",
              edgecolor="#333", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_color("#333")

    return _fig_to_base64(fig)


def chart_daily_pnl(daily_pnl: pd.Series) -> str:
    """Daily P&L bar chart. Returns base64 PNG."""
    plt = _import_plt()
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    if daily_pnl.empty:
        ax.text(0.5, 0.5, "No daily P&L data", ha="center", va="center",
                color="white", fontsize=16, transform=ax.transAxes)
        return _fig_to_base64(fig)

    pnl_pct = daily_pnl * 100.0
    colors = ["#00ff88" if v >= 0 else "#ff4444" for v in pnl_pct.values]

    ax.bar(range(len(pnl_pct)), pnl_pct.values, color=colors, alpha=0.8,
           edgecolor="none", width=0.8)
    ax.axhline(0, color="white", linewidth=0.5, alpha=0.3)

    ax.set_title("Daily P&L", color="white", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Daily Return (%)", color="white", fontsize=10)
    ax.set_xlabel("Trading Day", color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.grid(True, axis="y", alpha=0.15, color="white")

    return _fig_to_base64(fig)


def chart_win_rate_heatmap(trades: List[dict]) -> str:
    """Win rate heatmap: session x R-multiple distribution. Returns base64 PNG."""
    plt = _import_plt()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), gridspec_kw={"width_ratios": [1, 1]})
    fig.patch.set_facecolor("#1a1a2e")

    if not trades:
        for ax_item in axes:
            ax_item.set_facecolor("#16213e")
            ax_item.text(0.5, 0.5, "No data", ha="center", va="center",
                         color="white", fontsize=14, transform=ax_item.transAxes)
        return _fig_to_base64(fig)

    # Left: Win rate by session
    ax1 = axes[0]
    ax1.set_facecolor("#16213e")
    sessions = {}
    for t in trades:
        ctx = t.get("context") or {}
        session = ctx.get("session", "unknown")
        r = float(t.get("r_multiple") or 0.0)
        if session not in sessions:
            sessions[session] = {"wins": 0, "total": 0}
        sessions[session]["total"] += 1
        if r > 0:
            sessions[session]["wins"] += 1

    if sessions:
        names = sorted(sessions.keys())
        win_rates = [sessions[s]["wins"] / max(sessions[s]["total"], 1) * 100 for s in names]
        counts = [sessions[s]["total"] for s in names]
        colors = ["#00ff88" if wr >= 50 else "#ff6b6b" if wr < 40 else "#ffd93d"
                  for wr in win_rates]

        bars = ax1.barh(range(len(names)), win_rates, color=colors, alpha=0.8,
                        edgecolor="none", height=0.6)
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels([f"{n} ({c})" for n, c in zip(names, counts)],
                            color="white", fontsize=9)
        for bar, wr in zip(bars, win_rates):
            ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                     f"{wr:.0f}%", va="center", color="white", fontsize=9)
        ax1.axvline(50, color="white", linewidth=0.5, linestyle="--", alpha=0.4)

    ax1.set_title("Win Rate by Session", color="white", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Win Rate (%)", color="white", fontsize=9)
    ax1.tick_params(colors="white", labelsize=8)
    ax1.set_xlim(0, 100)
    for spine in ax1.spines.values():
        spine.set_color("#333")

    # Right: R-multiple distribution
    ax2 = axes[1]
    ax2.set_facecolor("#16213e")
    r_multiples = [float(t.get("r_multiple") or 0.0) for t in trades]
    if r_multiples:
        colors_hist = ["#00ff88" if r > 0 else "#ff4444" for r in r_multiples]
        bins = np.linspace(min(r_multiples) - 0.5, max(r_multiples) + 0.5, 30)
        ax2.hist(r_multiples, bins=bins, color="#00d2ff", alpha=0.7,
                 edgecolor="none")
        ax2.axvline(0, color="white", linewidth=1, linestyle="-", alpha=0.5)
        mean_r = np.mean(r_multiples)
        ax2.axvline(mean_r, color="#ffd93d", linewidth=1.5, linestyle="--",
                    alpha=0.8, label=f"Mean: {mean_r:.2f}R")
        ax2.legend(fontsize=8, facecolor="#16213e", edgecolor="#333", labelcolor="white")

    ax2.set_title("R-Multiple Distribution", color="white", fontsize=12, fontweight="bold")
    ax2.set_xlabel("R-Multiple", color="white", fontsize=9)
    ax2.tick_params(colors="white", labelsize=8)
    for spine in ax2.spines.values():
        spine.set_color("#333")

    fig.tight_layout(pad=2.0)
    return _fig_to_base64(fig)


def chart_prop_firm_tracking(
    equity_curve: pd.Series,
    initial_balance: float,
    daily_pnl: pd.Series,
    max_daily_dd_pct: float = 5.0,
    max_total_dd_pct: float = 10.0,
    profit_target_pct: float = 8.0,
) -> str:
    """Prop firm constraint gauges. Returns base64 PNG."""
    plt = _import_plt()
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    fig.patch.set_facecolor("#1a1a2e")

    # --- Gauge 1: Profit progress ---
    ax1 = axes[0]
    ax1.set_facecolor("#16213e")
    if not equity_curve.empty:
        final_pct = (equity_curve.iloc[-1] / initial_balance - 1.0) * 100.0
    else:
        final_pct = 0.0
    progress = min(final_pct / profit_target_pct * 100, 100) if profit_target_pct > 0 else 0
    color = "#00ff88" if final_pct >= profit_target_pct else "#00d2ff"
    ax1.barh(0, progress, height=0.5, color=color, alpha=0.8)
    ax1.barh(0, 100, height=0.5, color="#333", alpha=0.3)
    ax1.set_xlim(0, 100)
    ax1.set_yticks([])
    ax1.set_title(f"Profit: {final_pct:.1f}% / {profit_target_pct}%",
                  color="white", fontsize=11, fontweight="bold")
    ax1.tick_params(colors="white", labelsize=8)
    for spine in ax1.spines.values():
        spine.set_color("#333")

    # --- Gauge 2: Max Daily DD ---
    ax2 = axes[1]
    ax2.set_facecolor("#16213e")
    if not daily_pnl.empty:
        worst_daily = float(daily_pnl.min() * 100.0)
    else:
        worst_daily = 0.0
    dd_used = min(abs(worst_daily) / max_daily_dd_pct * 100, 100) if max_daily_dd_pct > 0 else 0
    dd_color = "#ff4444" if abs(worst_daily) >= max_daily_dd_pct * 0.8 else "#ffd93d" if abs(worst_daily) >= max_daily_dd_pct * 0.5 else "#00ff88"
    ax2.barh(0, dd_used, height=0.5, color=dd_color, alpha=0.8)
    ax2.barh(0, 100, height=0.5, color="#333", alpha=0.3)
    ax2.set_xlim(0, 100)
    ax2.set_yticks([])
    ax2.set_title(f"Daily DD: {worst_daily:.2f}% / -{max_daily_dd_pct}%",
                  color="white", fontsize=11, fontweight="bold")
    ax2.tick_params(colors="white", labelsize=8)
    for spine in ax2.spines.values():
        spine.set_color("#333")

    # --- Gauge 3: Max Total DD ---
    ax3 = axes[2]
    ax3.set_facecolor("#16213e")
    if not equity_curve.empty:
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max * 100.0
        worst_total = float(drawdown.min())
    else:
        worst_total = 0.0
    total_dd_used = min(abs(worst_total) / max_total_dd_pct * 100, 100) if max_total_dd_pct > 0 else 0
    td_color = "#ff4444" if abs(worst_total) >= max_total_dd_pct * 0.8 else "#ffd93d" if abs(worst_total) >= max_total_dd_pct * 0.5 else "#00ff88"
    ax3.barh(0, total_dd_used, height=0.5, color=td_color, alpha=0.8)
    ax3.barh(0, 100, height=0.5, color="#333", alpha=0.3)
    ax3.set_xlim(0, 100)
    ax3.set_yticks([])
    ax3.set_title(f"Total DD: {worst_total:.2f}% / -{max_total_dd_pct}%",
                  color="white", fontsize=11, fontweight="bold")
    ax3.tick_params(colors="white", labelsize=8)
    for spine in ax3.spines.values():
        spine.set_color("#333")

    fig.tight_layout(pad=2.0)
    return _fig_to_base64(fig)


# ---------------------------------------------------------------------------
# Metric card HTML helper
# ---------------------------------------------------------------------------

def _metric_card(label: str, value: str, color: str = "#00d2ff") -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-value" style="color: {color}">{value}</div>
        <div class="metric-label">{label}</div>
    </div>"""


# ---------------------------------------------------------------------------
# BacktestDashboard
# ---------------------------------------------------------------------------

class BacktestDashboard:
    """Generate a self-contained HTML dashboard from backtest results.

    Parameters
    ----------
    title:
        Dashboard title.
    """

    def __init__(self, title: str = "XAU/USD Ichimoku Backtest Dashboard") -> None:
        self._title = title

    def generate(
        self,
        result,
        initial_balance: float = 10_000.0,
        learning_phase: str = "disabled",
        learning_skipped: int = 0,
        instrument: str = "XAUUSD",
    ) -> str:
        """Generate the full HTML dashboard.

        Parameters
        ----------
        result:
            BacktestResult from IchimokuBacktester.run().
        initial_balance:
            Starting balance for percentage calculations.
        learning_phase:
            Current adaptive learning phase name.
        learning_skipped:
            Number of signals blocked by the learning engine.
        instrument:
            Instrument symbol for display.

        Returns
        -------
        str: Complete HTML document.
        """
        m = result.metrics
        pf = result.prop_firm
        trades = result.trades
        n_trades = len(trades)

        # Generate all charts
        equity_b64 = chart_equity_curve(result.equity_curve, initial_balance)
        trades_b64 = chart_trades_on_price(trades, result.equity_curve)
        learning_b64 = chart_learning_phases(trades, n_trades)
        daily_b64 = chart_daily_pnl(result.daily_pnl)
        heatmap_b64 = chart_win_rate_heatmap(trades)
        prop_b64 = chart_prop_firm_tracking(
            result.equity_curve, initial_balance, result.daily_pnl,
        )

        # Build metric cards
        win_rate = m.get("win_rate", 0)
        sharpe = m.get("sharpe_ratio", 0)
        total_return = m.get("total_return_pct", 0)
        max_dd = m.get("max_drawdown_pct", 0)
        expectancy = m.get("expectancy", 0)
        prop_status = pf.get("status", "N/A")

        wr_color = "#00ff88" if win_rate >= 0.5 else "#ff6b6b"
        sharpe_color = "#00ff88" if sharpe >= 1.0 else "#ffd93d" if sharpe >= 0.5 else "#ff6b6b"
        ret_color = "#00ff88" if total_return >= 8 else "#ffd93d" if total_return >= 0 else "#ff6b6b"
        dd_color = "#00ff88" if abs(max_dd) < 5 else "#ffd93d" if abs(max_dd) < 8 else "#ff6b6b"
        verdict_color = "#00ff88" if prop_status == "passed" else "#ffd93d" if prop_status == "active" else "#ff6b6b"

        metrics_html = "".join([
            _metric_card("Trades", str(n_trades)),
            _metric_card("Win Rate", f"{win_rate:.1%}", wr_color),
            _metric_card("Sharpe", f"{sharpe:.2f}", sharpe_color),
            _metric_card("Return", f"{total_return:+.2f}%", ret_color),
            _metric_card("Max DD", f"{max_dd:.2f}%", dd_color),
            _metric_card("Expectancy", f"{expectancy:.3f}R"),
            _metric_card("Prop Status", prop_status.upper(), verdict_color),
            _metric_card("Learning", learning_phase.title(), "#a78bfa"),
        ])

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{self._title}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        background: #0f0f23;
        color: #e0e0e0;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
        padding: 24px;
        line-height: 1.5;
    }}
    .container {{ max-width: 1400px; margin: 0 auto; }}
    .header {{
        text-align: center;
        margin-bottom: 32px;
        padding: 24px;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        border: 1px solid #333;
    }}
    .header h1 {{
        font-size: 28px;
        color: #00d2ff;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }}
    .header .subtitle {{
        color: #888;
        font-size: 14px;
    }}
    .metrics-row {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 12px;
        margin-bottom: 24px;
    }}
    .metric-card {{
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #333;
        border-radius: 12px;
        padding: 16px 12px;
        text-align: center;
    }}
    .metric-value {{
        font-size: 24px;
        font-weight: 700;
        letter-spacing: -0.5px;
    }}
    .metric-label {{
        font-size: 11px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }}
    .chart-section {{
        margin-bottom: 20px;
    }}
    .chart-section img {{
        width: 100%;
        border-radius: 12px;
        border: 1px solid #333;
    }}
    .chart-row {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
        margin-bottom: 20px;
    }}
    .chart-row img {{
        width: 100%;
        border-radius: 12px;
        border: 1px solid #333;
    }}
    .info-bar {{
        display: flex;
        justify-content: center;
        gap: 24px;
        margin-bottom: 20px;
        flex-wrap: wrap;
    }}
    .info-item {{
        background: #1a1a2e;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 8px 16px;
        font-size: 12px;
    }}
    .info-item span {{ color: #00d2ff; font-weight: 600; }}
    .footer {{
        text-align: center;
        padding: 20px;
        color: #555;
        font-size: 11px;
        border-top: 1px solid #222;
        margin-top: 32px;
    }}
    @media (max-width: 768px) {{
        .chart-row {{ grid-template-columns: 1fr; }}
        .metrics-row {{ grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); }}
    }}
</style>
</head>
<body>
<div class="container">

<div class="header">
    <h1>{self._title}</h1>
    <div class="subtitle">{instrument} &bull; Generated {ts}</div>
</div>

<div class="info-bar">
    <div class="info-item">Signals Generated: <span>{result.total_signals}</span></div>
    <div class="info-item">Edge Filtered: <span>{result.skipped_signals - learning_skipped}</span></div>
    <div class="info-item">Learning Filtered: <span>{learning_skipped}</span></div>
    <div class="info-item">Prop Firm Profit: <span>{pf.get('profit_pct', 0):.2f}%</span></div>
    <div class="info-item">Days Elapsed: <span>{pf.get('days_elapsed', 'N/A')}</span></div>
</div>

<div class="metrics-row">
    {metrics_html}
</div>

<div class="chart-section">
    <img src="data:image/png;base64,{equity_b64}" alt="Equity Curve">
</div>

<div class="chart-section">
    <img src="data:image/png;base64,{prop_b64}" alt="Prop Firm Tracking">
</div>

<div class="chart-section">
    <img src="data:image/png;base64,{learning_b64}" alt="Learning Phases">
</div>

<div class="chart-row">
    <img src="data:image/png;base64,{heatmap_b64}" alt="Win Rate Heatmap">
    <img src="data:image/png;base64,{daily_b64}" alt="Daily P&L">
</div>

<div class="chart-section">
    <img src="data:image/png;base64,{trades_b64}" alt="Trade Distribution">
</div>

<div class="footer">
    Itchy Tradebot &bull; XAU/USD Ichimoku Kinko Hyo &bull; The5ers High Stakes Classic
</div>

</div>
</body>
</html>"""
        return html

    def save_and_open(
        self,
        result,
        output_dir: str = "reports",
        initial_balance: float = 10_000.0,
        learning_phase: str = "disabled",
        learning_skipped: int = 0,
        instrument: str = "XAUUSD",
        auto_open: bool = True,
    ) -> str:
        """Generate dashboard, save to file, and optionally open in browser.

        Returns
        -------
        str: Absolute path of the saved HTML file.
        """
        html = self.generate(
            result=result,
            initial_balance=initial_balance,
            learning_phase=learning_phase,
            learning_skipped=learning_skipped,
            instrument=instrument,
        )

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        file_path = out / f"dashboard_{ts}.html"
        file_path.write_text(html, encoding="utf-8")

        logger.info("Dashboard saved to %s", file_path)

        if auto_open:
            try:
                webbrowser.open(str(file_path.resolve()))
                logger.info("Dashboard opened in default browser")
            except Exception as exc:
                logger.warning("Could not auto-open dashboard: %s", exc)

        return str(file_path.resolve())
