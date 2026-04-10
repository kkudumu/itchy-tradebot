"""Dashboard visualization helpers (plan Task 21).

Small, self-contained render functions that produce HTML/SVG strings
for embedding in the post-run dashboard and the live HTTP dashboard
at :8501. All helpers take plain dicts / lists as input (not pandas
DataFrames) so they can run in either context without adapting.

Output is always a ``str`` (HTML) so callers can concatenate into a
larger dashboard document without a separate rendering pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Per-strategy panel
# ---------------------------------------------------------------------------


def render_per_strategy_panel(telemetry_summary: Dict[str, Any]) -> str:
    """Render a per-strategy breakdown with generated/entered/rejected counts.

    *telemetry_summary* is the dict produced by
    :meth:`StrategyTelemetryCollector.summary`.
    """
    per_strategy = telemetry_summary.get("per_strategy", {}) or {}
    if not per_strategy:
        return '<div class="panel-empty">No strategy telemetry available.</div>'

    rows: List[str] = []
    rows.append(
        '<table class="strategy-panel" style="width:100%;border-collapse:collapse;">'
        '<thead><tr>'
        '<th style="text-align:left;padding:6px;border-bottom:1px solid #333;">Strategy</th>'
        '<th style="text-align:right;padding:6px;border-bottom:1px solid #333;">Generated</th>'
        '<th style="text-align:right;padding:6px;border-bottom:1px solid #333;">Entered</th>'
        '<th style="text-align:right;padding:6px;border-bottom:1px solid #333;">Entry %</th>'
        '<th style="text-align:right;padding:6px;border-bottom:1px solid #333;">Rejected</th>'
        '</tr></thead><tbody>'
    )
    for name, counts in per_strategy.items():
        rows.append(
            f'<tr>'
            f'<td style="padding:6px;">{name}</td>'
            f'<td style="padding:6px;text-align:right;">{int(counts.get("generated", 0))}</td>'
            f'<td style="padding:6px;text-align:right;">{int(counts.get("entered", 0))}</td>'
            f'<td style="padding:6px;text-align:right;">{float(counts.get("entry_rate_pct", 0.0)):.2f}%</td>'
            f'<td style="padding:6px;text-align:right;">{int(counts.get("rejected", 0))}</td>'
            f'</tr>'
        )
    rows.append("</tbody></table>")
    return "".join(rows)


# ---------------------------------------------------------------------------
# Top rejection stages bar chart
# ---------------------------------------------------------------------------


def render_top_rejection_stages(telemetry_summary: Dict[str, Any], top_n: int = 10) -> str:
    """Inline-SVG bar chart of the most-rejecting filter stages."""
    stages = telemetry_summary.get("top_rejection_stages", {}) or {}
    if not stages:
        return '<div class="panel-empty">No rejections recorded.</div>'

    items = list(stages.items())[:top_n]
    max_count = max((v for _, v in items), default=1)
    bar_height = 20
    gap = 6
    label_width = 180
    bar_max_width = 320
    svg_width = label_width + bar_max_width + 80
    svg_height = len(items) * (bar_height + gap)

    bars: List[str] = [
        f'<svg viewBox="0 0 {svg_width} {svg_height}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;max-width:{svg_width}px;height:auto;">'
    ]
    for i, (stage, count) in enumerate(items):
        y = i * (bar_height + gap)
        bar_w = (count / max_count) * bar_max_width if max_count else 0
        bars.append(
            f'<text x="0" y="{y + 14}" fill="#aaa" font-family="monospace" font-size="11">{stage}</text>'
            f'<rect x="{label_width}" y="{y}" width="{bar_w:.1f}" height="{bar_height}" fill="#ff6b6b"/>'
            f'<text x="{label_width + bar_w + 4}" y="{y + 14}" fill="#ccc" font-size="11">{count}</text>'
        )
    bars.append("</svg>")
    return "".join(bars)


# ---------------------------------------------------------------------------
# Per-session distribution
# ---------------------------------------------------------------------------


def render_session_distribution(telemetry_summary: Dict[str, Any]) -> str:
    """Horizontal stacked bar showing event distribution by session."""
    sessions = telemetry_summary.get("per_session", {}) or {}
    if not sessions:
        return '<div class="panel-empty">No session data.</div>'
    total = sum(sessions.values()) or 1
    colors = {
        "asian": "#3498db",
        "london": "#e74c3c",
        "overlap": "#f39c12",
        "ny": "#2ecc71",
        "off": "#888",
    }
    parts: List[str] = ['<div style="display:flex;width:100%;height:28px;border-radius:4px;overflow:hidden;">']
    for session, count in sessions.items():
        pct = count / total * 100.0
        color = colors.get(session, "#666")
        parts.append(
            f'<div style="width:{pct:.1f}%;background:{color};'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-size:11px;color:#fff;" '
            f'title="{session}: {count} events ({pct:.1f}%)">'
            f'{session} {count}</div>'
        )
    parts.append("</div>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Pattern histogram
# ---------------------------------------------------------------------------


def render_pattern_histogram(telemetry_summary: Dict[str, Any]) -> str:
    """Horizontal bar chart of pattern_type frequencies."""
    patterns = telemetry_summary.get("per_pattern", {}) or {}
    if not patterns:
        return '<div class="panel-empty">No patterns recorded.</div>'
    max_v = max(patterns.values()) or 1
    rows: List[str] = []
    for name, count in sorted(patterns.items(), key=lambda kv: kv[1], reverse=True):
        w = count / max_v * 100
        rows.append(
            f'<div style="display:flex;align-items:center;margin:4px 0;">'
            f'<div style="width:140px;font-size:11px;color:#aaa;">{name}</div>'
            f'<div style="flex:1;background:#1a1a2e;height:16px;border-radius:3px;overflow:hidden;">'
            f'<div style="width:{w:.1f}%;height:100%;background:#00ff88;"></div>'
            f'</div>'
            f'<div style="width:50px;text-align:right;font-size:11px;color:#ccc;">{count}</div>'
            f'</div>'
        )
    return "".join(rows)


# ---------------------------------------------------------------------------
# MLL gauge (TopstepX)
# ---------------------------------------------------------------------------


def render_mll_gauge(prop_firm_state: Dict[str, Any]) -> str:
    """Simple SVG gauge showing MLL distance as a fraction of initial buffer.

    When the prop firm state isn't topstep-style, renders an empty
    placeholder so callers can always concatenate.
    """
    active = prop_firm_state.get("active_tracker") if prop_firm_state else None
    if not active or active.get("style") != "topstep_combine_dollar":
        return ""

    current = float(active.get("current_balance") or 0.0)
    mll = float(active.get("mll") or 0.0)
    initial = float(active.get("initial_balance") or 0.0)
    distance = max(0.0, current - mll)
    max_buffer = max(initial - (initial - 2_000.0), 2_000.0)  # $2K initial buffer
    pct = min(1.0, distance / max_buffer) if max_buffer else 0.0
    color = "#00ff88" if pct > 0.75 else "#ffd93d" if pct > 0.33 else "#ff6b6b"
    return (
        f'<div style="padding:12px;background:#1a1a2e;border-radius:6px;">'
        f'<div style="font-size:11px;color:#aaa;margin-bottom:4px;">DIST → MLL</div>'
        f'<div style="font-size:22px;color:{color};font-weight:700;">${distance:,.0f}</div>'
        f'<div style="height:8px;background:#111;border-radius:4px;margin-top:8px;overflow:hidden;">'
        f'<div style="width:{pct * 100:.1f}%;height:100%;background:{color};"></div>'
        f'</div>'
        f'</div>'
    )


def render_daily_loss_gauge(prop_firm_state: Dict[str, Any]) -> str:
    """Gauge showing daily loss used vs limit."""
    active = prop_firm_state.get("active_tracker") if prop_firm_state else None
    if not active or active.get("style") != "topstep_combine_dollar":
        return ""
    daily_pnl = float(active.get("daily_pnl") or 0.0)
    limit = 1_000.0  # TopstepX $50K default; read from config in prod
    used = max(0.0, -daily_pnl)
    pct = min(1.0, used / limit)
    color = "#00ff88" if pct < 0.5 else "#ffd93d" if pct < 0.8 else "#ff6b6b"
    return (
        f'<div style="padding:12px;background:#1a1a2e;border-radius:6px;">'
        f'<div style="font-size:11px;color:#aaa;margin-bottom:4px;">DAILY LOSS USED</div>'
        f'<div style="font-size:22px;color:{color};font-weight:700;">${used:,.0f} / ${limit:,.0f}</div>'
        f'<div style="height:8px;background:#111;border-radius:4px;margin-top:8px;overflow:hidden;">'
        f'<div style="width:{pct * 100:.1f}%;height:100%;background:{color};"></div>'
        f'</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Top-level composed panel
# ---------------------------------------------------------------------------


def render_telemetry_summary_panel(
    telemetry_summary: Dict[str, Any],
    prop_firm_state: Optional[Dict[str, Any]] = None,
) -> str:
    """Compose the per-strategy + rejection + session + pattern panels."""
    parts = [
        '<div class="telemetry-panel">',
        '<h3 style="margin:12px 0 6px 0;color:#ccc;">Per-strategy funnel</h3>',
        render_per_strategy_panel(telemetry_summary),
        '<h3 style="margin:18px 0 6px 0;color:#ccc;">Top rejection stages</h3>',
        render_top_rejection_stages(telemetry_summary),
        '<h3 style="margin:18px 0 6px 0;color:#ccc;">Session distribution</h3>',
        render_session_distribution(telemetry_summary),
        '<h3 style="margin:18px 0 6px 0;color:#ccc;">Pattern histogram</h3>',
        render_pattern_histogram(telemetry_summary),
    ]
    if prop_firm_state:
        parts.append('<h3 style="margin:18px 0 6px 0;color:#ccc;">TopstepX status</h3>')
        parts.append(
            '<div style="display:flex;gap:12px;">'
            + render_mll_gauge(prop_firm_state)
            + render_daily_loss_gauge(prop_firm_state)
            + '</div>'
        )
    parts.append("</div>")
    return "".join(parts)
