"""
Weekly performance report generator for the adaptive learning system.

ReportGenerator consolidates output from StatsAnalyzer, EdgeReviewer, and
the AdaptiveLearningEngine into a structured WeeklyReport.  Reports are
rendered as plain text summaries or HTML for distribution.

Reports are read-only — they observe the system state and propose no changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .edge_reviewer import EdgeReviewResult, EdgeSuggestion, EdgeReviewer
from .stats_analyzer import StatsAnalyzer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WeeklyReport dataclass
# ---------------------------------------------------------------------------

@dataclass
class WeeklyReport:
    """A consolidated weekly performance summary.

    Attributes
    ----------
    period_start:
        UTC datetime of the first trade in the reporting period.
    period_end:
        UTC datetime of the last trade (or report generation time).
    total_trades:
        Total closed trades in the period.
    win_rate:
        Overall win rate for the period.
    pnl:
        Total realised PnL (in R-multiples) for the period.
    session_breakdown:
        Win rate and trade counts by trading session.
    setup_breakdown:
        Win rate by signal tier / setup type (A+, B, C).
    edge_review:
        List of EdgeReviewResult for all tracked edges.
    similarity_quality:
        Dict with keys: avg_confidence, pct_above_threshold,
        total_similarity_queries.
    suggestions:
        Advisory parameter / edge-state suggestions (human approval needed).
    learning_phase:
        Current phase: "mechanical", "statistical", or "similarity".
    total_trades_lifetime:
        Total closed trades all-time (drives phase progression).
    by_regime:
        Win rate and trade counts by ADX regime.
    by_day:
        Win rate and trade counts by day of week.
    """

    period_start: datetime
    period_end: datetime
    total_trades: int
    win_rate: float
    pnl: float
    session_breakdown: Dict[str, Any] = field(default_factory=dict)
    setup_breakdown: Dict[str, Any] = field(default_factory=dict)
    edge_review: List[EdgeReviewResult] = field(default_factory=list)
    similarity_quality: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[EdgeSuggestion] = field(default_factory=list)
    learning_phase: str = "mechanical"
    total_trades_lifetime: int = 0
    by_regime: Dict[str, Any] = field(default_factory=dict)
    by_day: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------

class ReportGenerator:
    """Generate periodic performance reports.

    Parameters
    ----------
    stats_analyzer:
        StatsAnalyzer instance for win-rate breakdowns.
    edge_reviewer:
        EdgeReviewer instance for edge-level analysis.
    db_pool:
        Database pool used to fetch period-specific trade data.
        May be None; inject a trade list via ``_inject_period_trades``.
    """

    # Period trade query — fetches trades closed within the reporting window
    _PERIOD_TRADES_QUERY = """
        SELECT
            t.id,
            t.r_multiple,
            t.pnl,
            t.signal_tier,
            t.entry_time,
            t.exit_time,
            mc.session
        FROM trades t
        LEFT JOIN market_context mc ON mc.trade_id = t.id
        WHERE t.status = 'closed'
          AND t.exit_time BETWEEN %s AND %s
          AND t.r_multiple IS NOT NULL
        ORDER BY t.exit_time ASC
    """

    # Similarity quality query — aggregates stored insight confidence values
    _SIMILARITY_QUALITY_QUERY = """
        SELECT
            COUNT(*) AS total_queries,
            AVG(confidence) AS avg_confidence,
            SUM(CASE WHEN confidence >= 0.25 THEN 1 ELSE 0 END) AS above_threshold
        FROM pre_trade_insights
        WHERE created_at BETWEEN %s AND %s
    """

    def __init__(
        self,
        stats_analyzer: Optional[StatsAnalyzer] = None,
        edge_reviewer: Optional[EdgeReviewer] = None,
        db_pool=None,
    ) -> None:
        self._stats = stats_analyzer or StatsAnalyzer(db_pool=db_pool)
        self._edge_reviewer = edge_reviewer or EdgeReviewer(db_pool=db_pool)
        self._db_pool = db_pool

        # Optional injection for tests
        self._period_trades: Optional[List[dict]] = None
        self._similarity_quality_override: Optional[dict] = None
        self._learning_phase: str = "mechanical"
        self._total_trades_lifetime: int = 0

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------

    def _inject_period_trades(self, trades: List[dict]) -> None:
        """Inject a list of trade dicts for testing without a database."""
        self._period_trades = list(trades)

    def _inject_similarity_quality(self, quality: dict) -> None:
        """Inject similarity quality statistics for testing."""
        self._similarity_quality_override = dict(quality)

    def _set_learning_context(self, phase: str, total_trades: int) -> None:
        """Set learning phase and trade count for report context."""
        self._learning_phase = phase
        self._total_trades_lifetime = total_trades

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def weekly_report(
        self,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> WeeklyReport:
        """Generate a weekly performance summary.

        Parameters
        ----------
        period_start:
            Start of the reporting period.  Defaults to 7 days before
            period_end.
        period_end:
            End of the reporting period.  Defaults to UTC now.

        Returns
        -------
        WeeklyReport with all sections populated.
        """
        now = datetime.now(tz=timezone.utc)
        if period_end is None:
            period_end = now
        if period_start is None:
            from datetime import timedelta
            period_start = period_end - timedelta(days=7)

        trades = self._load_period_trades(period_start, period_end)

        total_trades = len(trades)
        win_rate = 0.0
        pnl = 0.0

        if total_trades > 0:
            wins = sum(1 for t in trades if float(t.get("r_multiple", 0) or 0) > 0)
            win_rate = wins / total_trades
            pnl = sum(float(t.get("r_multiple", 0) or 0) for t in trades)

        session_breakdown = self._stats.win_rate_by_session(min_trades=5)
        setup_breakdown = self._stats.win_rate_by_confluence(min_trades=5)
        by_regime = self._stats.win_rate_by_regime(min_trades=5)
        by_day = self._stats.win_rate_by_day(min_trades=5)

        edge_review = self._edge_reviewer.review_all_edges()
        suggestions = self._edge_reviewer.suggest_edge_changes()

        similarity_quality = self._load_similarity_quality(period_start, period_end)
        period_setup = self._build_setup_breakdown(trades)

        # Prefer the setup breakdown from period trades if available
        if period_setup:
            setup_breakdown = period_setup

        return WeeklyReport(
            period_start=period_start,
            period_end=period_end,
            total_trades=total_trades,
            win_rate=round(win_rate, 4),
            pnl=round(pnl, 4),
            session_breakdown=session_breakdown,
            setup_breakdown=setup_breakdown,
            edge_review=edge_review,
            similarity_quality=similarity_quality,
            suggestions=suggestions,
            learning_phase=self._learning_phase,
            total_trades_lifetime=self._total_trades_lifetime,
            by_regime=by_regime,
            by_day=by_day,
        )

    def to_text(self, report: WeeklyReport) -> str:
        """Render a WeeklyReport as a plain-text summary.

        Returns
        -------
        Multi-line string suitable for logging or email.
        """
        lines: List[str] = []
        sep = "-" * 60

        lines.append("=" * 60)
        lines.append("  WEEKLY PERFORMANCE REPORT")
        lines.append("=" * 60)
        lines.append(
            f"Period  : {report.period_start.strftime('%Y-%m-%d')} – "
            f"{report.period_end.strftime('%Y-%m-%d')}"
        )
        lines.append(f"Phase   : {report.learning_phase.upper()}")
        lines.append(
            f"Lifetime trades: {report.total_trades_lifetime}"
        )
        lines.append(sep)

        lines.append("PERIOD SUMMARY")
        lines.append(f"  Total trades  : {report.total_trades}")
        lines.append(f"  Win rate      : {report.win_rate:.1%}")
        lines.append(f"  Total PnL (R) : {report.pnl:+.2f}")
        lines.append(sep)

        if report.session_breakdown:
            lines.append("SESSION PERFORMANCE")
            for session, stats in sorted(report.session_breakdown.items()):
                wr = stats.get("win_rate", 0)
                n = stats.get("n_trades", 0)
                avg_r = stats.get("avg_r", 0)
                lines.append(
                    f"  {session:<16} WR={wr:.1%}  n={n:4d}  avg_R={avg_r:+.3f}"
                )
            lines.append(sep)

        if report.setup_breakdown:
            lines.append("SETUP BREAKDOWN (by confluence tier)")
            for tier, stats in sorted(report.setup_breakdown.items()):
                wr = stats.get("win_rate", 0)
                n = stats.get("n_trades", 0)
                avg_r = stats.get("avg_r", 0)
                lines.append(
                    f"  Tier {tier:<6} WR={wr:.1%}  n={n:4d}  avg_R={avg_r:+.3f}"
                )
            lines.append(sep)

        if report.by_regime:
            lines.append("REGIME PERFORMANCE (ADX)")
            for regime, stats in sorted(report.by_regime.items()):
                wr = stats.get("win_rate", 0)
                n = stats.get("n_trades", 0)
                lines.append(
                    f"  {regime:<10} WR={wr:.1%}  n={n:4d}"
                )
            lines.append(sep)

        if report.edge_review:
            lines.append("EDGE PERFORMANCE")
            for er in report.edge_review:
                if er.total_trades_affected == 0:
                    continue
                lines.append(
                    f"  {er.edge_name:<28} "
                    f"n={er.total_trades_affected:4d}  "
                    f"WR={er.win_rate_when_active:.1%}  "
                    f"marginal={er.marginal_impact:+.3f}R  "
                    f"filter_rate={er.filter_rate:.1%}"
                )
            lines.append(sep)

        sim = report.similarity_quality
        if sim:
            lines.append("SIMILARITY QUALITY")
            lines.append(f"  Queries        : {sim.get('total_queries', 0)}")
            lines.append(
                f"  Avg confidence : {sim.get('avg_confidence', 0.0):.3f}"
            )
            lines.append(
                f"  Above threshold: {sim.get('pct_above_threshold', 0.0):.1%}"
            )
            lines.append(sep)

        if report.suggestions:
            lines.append("ADVISORY SUGGESTIONS (require human approval)")
            for s in report.suggestions:
                action = "ENABLE" if s.suggested_state else "DISABLE"
                lines.append(f"  [{action}] {s.edge_name}")
                lines.append(f"    Reason    : {s.reason}")
                lines.append(f"    Confidence: {s.confidence:.2f}")
            lines.append(sep)
        else:
            lines.append("SUGGESTIONS: None — current configuration looks good.")
            lines.append(sep)

        return "\n".join(lines)

    def to_html(self, report: WeeklyReport) -> str:
        """Render a WeeklyReport as an HTML string.

        Returns
        -------
        A complete HTML document as a string.
        """
        def _pct(v: float) -> str:
            return f"{v:.1%}"

        def _r(v: float) -> str:
            return f"{v:+.3f}R"

        def _wr_class(wr: float) -> str:
            if wr >= 0.55:
                return 'color:green'
            if wr < 0.40:
                return 'color:red'
            return 'color:orange'

        parts: List[str] = []
        parts.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
        parts.append("<title>Weekly Report</title>")
        parts.append("<style>body{font-family:monospace;max-width:900px;margin:0 auto;padding:16px}")
        parts.append("table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:6px 10px}")
        parts.append("th{background:#f0f0f0}h2{border-bottom:2px solid #333}</style>")
        parts.append("</head><body>")

        parts.append("<h1>Weekly Performance Report</h1>")
        parts.append(
            f"<p><b>Period</b>: "
            f"{report.period_start.strftime('%Y-%m-%d')} &ndash; "
            f"{report.period_end.strftime('%Y-%m-%d')} &nbsp;|&nbsp; "
            f"<b>Phase</b>: {report.learning_phase.upper()} &nbsp;|&nbsp; "
            f"<b>Lifetime trades</b>: {report.total_trades_lifetime}</p>"
        )

        # Period summary
        parts.append("<h2>Period Summary</h2>")
        parts.append("<table><tr><th>Metric</th><th>Value</th></tr>")
        parts.append(f"<tr><td>Total trades</td><td>{report.total_trades}</td></tr>")
        wr_style = _wr_class(report.win_rate)
        parts.append(
            f"<tr><td>Win rate</td>"
            f"<td style='{wr_style}'>{_pct(report.win_rate)}</td></tr>"
        )
        parts.append(f"<tr><td>Total PnL (R)</td><td>{_r(report.pnl)}</td></tr>")
        parts.append("</table>")

        # Session performance
        if report.session_breakdown:
            parts.append("<h2>Session Performance</h2>")
            parts.append("<table><tr><th>Session</th><th>Win Rate</th><th>Trades</th><th>Avg R</th></tr>")
            for session, stats in sorted(report.session_breakdown.items()):
                wr = stats.get("win_rate", 0)
                n = stats.get("n_trades", 0)
                avg_r = stats.get("avg_r", 0)
                style = _wr_class(wr)
                parts.append(
                    f"<tr><td>{session}</td>"
                    f"<td style='{style}'>{_pct(wr)}</td>"
                    f"<td>{n}</td><td>{_r(avg_r)}</td></tr>"
                )
            parts.append("</table>")

        # Setup breakdown
        if report.setup_breakdown:
            parts.append("<h2>Setup Breakdown (Confluence Tier)</h2>")
            parts.append("<table><tr><th>Tier</th><th>Win Rate</th><th>Trades</th><th>Avg R</th></tr>")
            for tier, stats in sorted(report.setup_breakdown.items()):
                wr = stats.get("win_rate", 0)
                n = stats.get("n_trades", 0)
                avg_r = stats.get("avg_r", 0)
                style = _wr_class(wr)
                parts.append(
                    f"<tr><td>{tier}</td>"
                    f"<td style='{style}'>{_pct(wr)}</td>"
                    f"<td>{n}</td><td>{_r(avg_r)}</td></tr>"
                )
            parts.append("</table>")

        # Edge performance
        if report.edge_review:
            parts.append("<h2>Edge Performance</h2>")
            parts.append(
                "<table><tr>"
                "<th>Edge</th><th>Trades</th><th>Win Rate</th>"
                "<th>Avg R</th><th>Marginal</th><th>Filter Rate</th>"
                "</tr>"
            )
            for er in report.edge_review:
                if er.total_trades_affected == 0:
                    continue
                style = _wr_class(er.win_rate_when_active)
                parts.append(
                    f"<tr>"
                    f"<td>{er.edge_name}</td>"
                    f"<td>{er.total_trades_affected}</td>"
                    f"<td style='{style}'>{_pct(er.win_rate_when_active)}</td>"
                    f"<td>{_r(er.avg_r_when_active)}</td>"
                    f"<td>{_r(er.marginal_impact)}</td>"
                    f"<td>{_pct(er.filter_rate)}</td>"
                    f"</tr>"
                )
            parts.append("</table>")

        # Similarity quality
        sim = report.similarity_quality
        if sim:
            parts.append("<h2>Similarity Quality</h2>")
            parts.append("<table><tr><th>Metric</th><th>Value</th></tr>")
            parts.append(f"<tr><td>Queries</td><td>{sim.get('total_queries', 0)}</td></tr>")
            parts.append(
                f"<tr><td>Avg confidence</td>"
                f"<td>{sim.get('avg_confidence', 0.0):.3f}</td></tr>"
            )
            parts.append(
                f"<tr><td>Above threshold</td>"
                f"<td>{_pct(sim.get('pct_above_threshold', 0.0))}</td></tr>"
            )
            parts.append("</table>")

        # Suggestions
        if report.suggestions:
            parts.append("<h2>Advisory Suggestions (Human Approval Required)</h2>")
            parts.append("<table><tr><th>Edge</th><th>Action</th><th>Reason</th><th>Confidence</th></tr>")
            for s in report.suggestions:
                action = "ENABLE" if s.suggested_state else "DISABLE"
                parts.append(
                    f"<tr>"
                    f"<td>{s.edge_name}</td>"
                    f"<td><b>{action}</b></td>"
                    f"<td>{s.reason}</td>"
                    f"<td>{s.confidence:.2f}</td>"
                    f"</tr>"
                )
            parts.append("</table>")
        else:
            parts.append("<h2>Suggestions</h2><p>None — current configuration looks good.</p>")

        parts.append("</body></html>")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_period_trades(
        self, period_start: datetime, period_end: datetime
    ) -> List[dict]:
        """Load trades closed within the reporting period."""
        if self._period_trades is not None:
            return self._period_trades

        if self._db_pool is None:
            return []

        try:
            with self._db_pool.get_cursor() as cur:
                cur.execute(self._PERIOD_TRADES_QUERY, (period_start, period_end))
                rows = cur.fetchall()
            return [dict(row) for row in rows]
        except Exception as exc:
            logger.error("ReportGenerator: failed to load period trades: %s", exc)
            return []

    def _load_similarity_quality(
        self, period_start: datetime, period_end: datetime
    ) -> Dict[str, Any]:
        """Load similarity query quality statistics for the period."""
        if self._similarity_quality_override is not None:
            return self._similarity_quality_override

        if self._db_pool is None:
            return {}

        try:
            with self._db_pool.get_cursor() as cur:
                cur.execute(self._SIMILARITY_QUALITY_QUERY, (period_start, period_end))
                row = cur.fetchone()
            if row is None:
                return {}

            total = int(row.get("total_queries") or 0)
            avg_conf = float(row.get("avg_confidence") or 0.0)
            above = int(row.get("above_threshold") or 0)
            pct_above = above / total if total > 0 else 0.0

            return {
                "total_queries": total,
                "avg_confidence": round(avg_conf, 4),
                "pct_above_threshold": round(pct_above, 4),
            }
        except Exception as exc:
            logger.debug("ReportGenerator: similarity quality query failed: %s", exc)
            return {}

    @staticmethod
    def _build_setup_breakdown(trades: List[dict]) -> Dict[str, Any]:
        """Compute win rate by signal_tier from a list of period trades."""
        from collections import defaultdict

        tier_wins: dict = defaultdict(int)
        tier_total: dict = defaultdict(int)
        tier_r: dict = defaultdict(float)

        for t in trades:
            tier = str(t.get("signal_tier") or "unknown")
            r = float(t.get("r_multiple") or 0)
            tier_total[tier] += 1
            tier_r[tier] += r
            if r > 0:
                tier_wins[tier] += 1

        result: Dict[str, Any] = {}
        for tier, n in tier_total.items():
            wr = tier_wins[tier] / n if n > 0 else 0.0
            avg_r = tier_r[tier] / n if n > 0 else 0.0
            result[tier] = {
                "win_rate": round(wr, 4),
                "n_trades": n,
                "avg_r": round(avg_r, 4),
            }

        return result
