"""Validation report generator — HTML and plain-text output.

The HTML report includes:
- Executive summary with a large GO / NO-GO / BORDERLINE verdict banner.
- Threshold checklist table (green/red per row, haircut details).
- OOS metrics summary table.
- Walk-forward performance chart (IS vs OOS Sharpe per window).
- Monte Carlo pass rate and failure mode breakdown chart.
- Overfitting analysis section.
- Action-item recommendations.

Charts are generated with matplotlib and embedded as base-64 PNG data URIs so
the report is a single self-contained HTML file requiring no external assets.

Plain-text output is a concise console-friendly summary suitable for logging.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import List

from src.simulation.visualizer import MCVisualizer

logger = logging.getLogger(__name__)

# Colour constants for HTML output.
_GREEN = "#28a745"
_RED = "#dc3545"
_AMBER = "#ffc107"
_DARK = "#212529"
_LIGHT_BG = "#f8f9fa"
_BORDER = "#dee2e6"

# Verdict colours.
_VERDICT_COLOURS = {
    "GO": _GREEN,
    "NO-GO": _RED,
    "BORDERLINE": _AMBER,
}


class ValidationReportGenerator:
    """Generate HTML and plain-text validation reports.

    Parameters
    ----------
    title:
        Report title string.  Default: 'XAU/USD Ichimoku Pre-Challenge Validation'.
    """

    def __init__(self, title: str = "XAU/USD Ichimoku Pre-Challenge Validation") -> None:
        self._title = title

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_html(self, result) -> str:
        """Generate a self-contained HTML report.

        Parameters
        ----------
        result:
            FullValidationResult from GoNoGoValidator.run_full_validation().

        Returns
        -------
        str
            Complete HTML document as a string.
        """
        sections = [
            self._html_head(),
            "<body>",
            '<div class="container">',
            f'<h1 class="report-title">{self._title}</h1>',
            f'<p class="timestamp">Generated: {result.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")}</p>',
            self._verdict_banner(result.final_verdict),
            self._executive_summary_html(result),
            self._threshold_table_html(result.validation_result),
            self._metrics_summary_html(result.oos_metrics, result.win_rate_ci),
            self._walk_forward_chart_html(result.wf_result),
            self._monte_carlo_html(result.monte_carlo),
            self._overfit_html(result.overfit_report),
            self._recommendations_html(result.recommendations),
            "</div>",
            "</body>",
            "</html>",
        ]
        return "\n".join(sections)

    def generate_text(self, result) -> str:
        """Generate a plain-text summary suitable for console / log output.

        Parameters
        ----------
        result:
            FullValidationResult from GoNoGoValidator.run_full_validation().

        Returns
        -------
        str
            Multi-line plain-text report.
        """
        lines: List[str] = []
        sep = "=" * 70

        lines.append(sep)
        lines.append(f"  {self._title}")
        lines.append(f"  Generated: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(sep)
        lines.append(f"  FINAL VERDICT: {result.final_verdict}")
        lines.append(sep)
        lines.append("")

        lines.append("THRESHOLD CHECKLIST")
        lines.append("-" * 70)
        for r in result.validation_result.results:
            status = "PASS" if r.passed else "FAIL"
            hc_note = " (haircutted)" if r.haircut_applied else ""
            lines.append(
                f"  [{status}] {r.name:<40} "
                f"value={r.haircutted_value:.4f}{hc_note}  "
                f"threshold={r.threshold}  margin={r.margin:+.4f}"
            )

        lines.append("")
        lines.append(f"  Passed: {result.validation_result.n_passed}  "
                     f"Failed: {result.validation_result.n_failed}")

        lines.append("")
        lines.append("OOS SUMMARY")
        lines.append("-" * 70)
        m = result.oos_metrics
        lines.append(f"  OOS Trades:        {result.n_oos_trades}")
        ci_lo, ci_hi = result.win_rate_ci
        lines.append(f"  Win Rate:          {m.get('win_rate', 0.0):.2%}  "
                     f"(Wilson 95% CI: [{ci_lo:.4f}, {ci_hi:.4f}])")
        lines.append(f"  Profit Factor:     {m.get('profit_factor', 0.0):.4f}")
        lines.append(f"  Sharpe Ratio:      {m.get('sharpe_ratio', 0.0):.4f}")
        lines.append(f"  Sortino Ratio:     {m.get('sortino_ratio', 0.0):.4f}")
        lines.append(f"  Max Drawdown:      {m.get('max_drawdown_pct', 0.0):.2f}%")
        lines.append(f"  WFE:               {m.get('wfe', 0.0):.4f}")
        lines.append(f"  DSR:               {m.get('dsr', 0.0):.4f}")
        lines.append(f"  MC Pass Rate:      {result.monte_carlo.pass_rate:.2f}%")

        lines.append("")
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 70)
        for rec in result.recommendations:
            lines.append(f"  {rec}")

        lines.append("")
        lines.append(sep)

        return "\n".join(lines)

    def save_report(self, html: str, path: str) -> str:
        """Write the HTML report to a file.

        Parameters
        ----------
        html:
            HTML string from generate_html().
        path:
            Destination file path.  The file is overwritten if it exists.

        Returns
        -------
        str
            Absolute path to the saved file.
        """
        import os

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
        logger.info("Report saved to %s", path)
        return os.path.abspath(path)

    # ------------------------------------------------------------------
    # HTML component builders
    # ------------------------------------------------------------------

    def _html_head(self) -> str:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{self._title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f0f2f5; color: {_DARK}; margin: 0; padding: 20px; }}
  .container {{ max-width: 1100px; margin: 0 auto; background: #fff;
               border-radius: 8px; padding: 30px; box-shadow: 0 2px 12px rgba(0,0,0,.1); }}
  h1.report-title {{ font-size: 1.6rem; margin-bottom: 4px; }}
  p.timestamp {{ color: #6c757d; font-size: 0.85rem; margin-top: 0; }}
  h2 {{ font-size: 1.2rem; border-bottom: 2px solid {_BORDER}; padding-bottom: 6px; margin-top: 30px; }}
  .verdict-banner {{ text-align: center; padding: 30px 20px; border-radius: 8px;
                     margin: 20px 0; color: #fff; }}
  .verdict-banner h2 {{ border: none; font-size: 3rem; margin: 0; color: #fff; }}
  .verdict-banner p {{ font-size: 1.1rem; margin: 8px 0 0; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; margin-top: 10px; }}
  th {{ background: {_LIGHT_BG}; border: 1px solid {_BORDER}; padding: 8px 10px; text-align: left; }}
  td {{ border: 1px solid {_BORDER}; padding: 7px 10px; }}
  .pass {{ color: {_GREEN}; font-weight: 600; }}
  .fail {{ color: {_RED}; font-weight: 600; }}
  .chart-container {{ margin: 15px 0; text-align: center; }}
  .chart-container img {{ max-width: 100%; border: 1px solid {_BORDER}; border-radius: 4px; }}
  .rec-list {{ list-style: disc; padding-left: 20px; }}
  .rec-list li {{ margin: 6px 0; line-height: 1.5; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 15px 0; }}
  .summary-card {{ background: {_LIGHT_BG}; border: 1px solid {_BORDER}; border-radius: 6px;
                   padding: 12px; text-align: center; }}
  .summary-card .label {{ font-size: 0.78rem; color: #6c757d; }}
  .summary-card .value {{ font-size: 1.3rem; font-weight: 700; margin-top: 4px; }}
</style>
</head>"""

    def _verdict_banner(self, verdict: str) -> str:
        colour = _VERDICT_COLOURS.get(verdict, _DARK)
        subtexts = {
            "GO": "All thresholds passed — challenge submission authorised.",
            "NO-GO": "One or more thresholds failed — do not submit a funded challenge.",
            "BORDERLINE": "All thresholds pass but margins are thin — manual review required.",
        }
        sub = subtexts.get(verdict, "")
        return (
            f'<div class="verdict-banner" style="background:{colour};">'
            f"<h2>{verdict}</h2>"
            f"<p>{sub}</p>"
            f"</div>"
        )

    def _executive_summary_html(self, result) -> str:
        m = result.oos_metrics
        cards = [
            ("OOS Trades", str(result.n_oos_trades), None),
            ("Win Rate", f"{m.get('win_rate', 0.0):.1%}", None),
            ("Profit Factor", f"{m.get('profit_factor', 0.0):.3f}", None),
            ("Sharpe", f"{m.get('sharpe_ratio', 0.0):.3f}", None),
            ("Max DD", f"{m.get('max_drawdown_pct', 0.0):.2f}%", None),
            ("MC Pass Rate", f"{result.monte_carlo.pass_rate:.1f}%", None),
        ]
        html_cards = ""
        for label, value, _ in cards:
            html_cards += (
                f'<div class="summary-card">'
                f'<div class="label">{label}</div>'
                f'<div class="value">{value}</div>'
                f"</div>"
            )
        ci_lo, ci_hi = result.win_rate_ci
        return (
            f"<h2>Executive Summary</h2>"
            f'<div class="summary-grid">{html_cards}</div>'
            f'<p style="font-size:0.85rem;color:#6c757d;">Win Rate Wilson 95% CI: '
            f"[{ci_lo:.4f}, {ci_hi:.4f}] — lower bound check against 45% threshold.</p>"
        )

    def _threshold_table_html(self, validation) -> str:
        rows = ""
        for r in validation.results:
            status_cls = "pass" if r.passed else "fail"
            status_txt = "PASS" if r.passed else "FAIL"
            hc_txt = f"{r.haircutted_value:.4f} (−25%)" if r.haircut_applied else f"{r.haircutted_value:.4f}"
            margin_txt = f"{r.margin:+.4f}"
            rows += (
                f"<tr>"
                f"<td>{r.name}</td>"
                f'<td class="{status_cls}">{status_txt}</td>'
                f"<td>{r.raw_value:.4f}</td>"
                f"<td>{hc_txt}</td>"
                f"<td>{r.threshold}</td>"
                f"<td>{margin_txt}</td>"
                f"</tr>"
            )

        summary_colour = _GREEN if validation.overall_pass else _RED
        summary_txt = f"{validation.n_passed} passed / {validation.n_failed} failed"
        return (
            f"<h2>Go/No-Go Threshold Checklist</h2>"
            f"<table>"
            f"<thead><tr>"
            f"<th>Metric</th><th>Status</th><th>Raw Value</th>"
            f"<th>After Haircut</th><th>Threshold</th><th>Margin</th>"
            f"</tr></thead>"
            f"<tbody>{rows}</tbody>"
            f"</table>"
            f'<p style="font-weight:600;color:{summary_colour};">{summary_txt}</p>'
        )

    def _metrics_summary_html(self, metrics: dict, win_rate_ci: tuple) -> str:
        ci_lo, ci_hi = win_rate_ci
        rows_data = [
            ("Total OOS Trades", metrics.get("oos_trade_count", 0)),
            ("Win Rate", f"{metrics.get('win_rate', 0.0):.4f}"),
            ("Win Rate CI Lower (95%)", f"{ci_lo:.4f}"),
            ("Win Rate CI Upper (95%)", f"{ci_hi:.4f}"),
            ("Profit Factor", f"{metrics.get('profit_factor', 0.0):.4f}"),
            ("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0.0):.4f}"),
            ("Sortino Ratio", f"{metrics.get('sortino_ratio', 0.0):.4f}"),
            ("Calmar Ratio", f"{metrics.get('calmar_ratio', 0.0):.4f}"),
            ("Max Drawdown %", f"{metrics.get('max_drawdown_pct', 0.0):.4f}"),
            ("Max Daily DD %", f"{metrics.get('max_daily_dd_pct', 0.0):.4f}"),
            ("Total Return %", f"{metrics.get('total_return_pct', 0.0):.4f}"),
            ("Walk-Forward Efficiency", f"{metrics.get('wfe', 0.0):.4f}"),
            ("Deflated Sharpe Ratio", f"{metrics.get('dsr', 0.0):.4f}"),
            ("Avg R Multiple", f"{metrics.get('avg_r_multiple', 0.0):.4f}"),
            ("Expectancy (R)", f"{metrics.get('expectancy', 0.0):.4f}"),
        ]
        rows = "".join(
            f"<tr><td>{label}</td><td>{value}</td></tr>" for label, value in rows_data
        )
        return (
            f"<h2>OOS Metrics Summary</h2>"
            f"<table>"
            f"<thead><tr><th>Metric</th><th>Value</th></tr></thead>"
            f"<tbody>{rows}</tbody>"
            f"</table>"
        )

    def _walk_forward_chart_html(self, wf_result) -> str:
        """Render IS vs OOS Sharpe per window as an embedded PNG chart."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            is_sharpes = wf_result.is_sharpes or []
            oos_sharpes = wf_result.oos_sharpes or []
            if not is_sharpes and not oos_sharpes:
                return "<h2>Walk-Forward Performance</h2><p>No walk-forward data available.</p>"

            n = max(len(is_sharpes), len(oos_sharpes))
            x = list(range(1, n + 1))

            fig, ax = plt.subplots(figsize=(9, 4))
            if is_sharpes:
                ax.bar([i - 0.2 for i in x[:len(is_sharpes)]], is_sharpes, 0.35,
                       label="IS Sharpe", color="#4e79a7", alpha=0.85)
            if oos_sharpes:
                ax.bar([i + 0.2 for i in x[:len(oos_sharpes)]], oos_sharpes, 0.35,
                       label="OOS Sharpe", color="#f28e2b", alpha=0.85)
            ax.axhline(0, color=_DARK, linewidth=0.8, linestyle="--")
            ax.set_xlabel("Window")
            ax.set_ylabel("Sharpe Ratio")
            ax.set_title(f"Walk-Forward IS vs OOS Sharpe  (WFE={wf_result.wfe:.3f})")
            ax.set_xticks(x)
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()

            png_b64 = _fig_to_base64(fig)
            plt.close(fig)
            return (
                "<h2>Walk-Forward Performance</h2>"
                f'<div class="chart-container"><img src="data:image/png;base64,{png_b64}" '
                f'alt="Walk-forward chart"></div>'
            )
        except Exception as exc:
            logger.warning("Could not render walk-forward chart: %s", exc)
            return "<h2>Walk-Forward Performance</h2><p>Chart unavailable.</p>"

    def _monte_carlo_html(self, mc_result) -> str:
        """Render Monte Carlo failure mode breakdown and running pass-rate chart."""
        section = "<h2>Monte Carlo Simulation</h2>"

        # Stats table.
        rows = [
            ("Simulations", mc_result.n_simulations),
            ("Pass Rate", f"{mc_result.pass_rate:.2f}%"),
            ("Avg Days to Result", f"{mc_result.avg_days:.1f}"),
            ("Median Days to Result", f"{mc_result.median_days:.1f}"),
            ("Daily DD Failure Rate", f"{mc_result.daily_dd_failure_rate:.2f}%"),
            ("Total DD Failure Rate", f"{mc_result.total_dd_failure_rate:.2f}%"),
            ("Timeout Rate", f"{mc_result.timeout_rate:.2f}%"),
            ("Circuit Breaker Rate", f"{mc_result.circuit_breaker_rate:.2f}%"),
            ("Convergence Reached", str(mc_result.convergence_reached)),
        ]
        tbl_rows = "".join(
            f"<tr><td>{label}</td><td>{value}</td></tr>" for label, value in rows
        )
        section += (
            "<table>"
            "<thead><tr><th>Stat</th><th>Value</th></tr></thead>"
            f"<tbody>{tbl_rows}</tbody>"
            "</table>"
        )

        # Use MCVisualizer for rich charts (equity fan + pass-rate convergence).
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            viz = MCVisualizer()

            # Equity fan chart — shows spread of equity curve outcomes.
            if hasattr(mc_result, "outcomes") and mc_result.outcomes:
                fig_fan = viz.equity_fan(mc_result.outcomes, n_sample=100)
                png_b64 = _fig_to_base64(fig_fan)
                plt.close(fig_fan)
                section += (
                    f'<div class="chart-container">'
                    f'<img src="data:image/png;base64,{png_b64}" alt="Equity fan chart">'
                    f"</div>"
                )

            # Pass-rate convergence chart — confirms simulation count is sufficient.
            fig_conv = viz.pass_rate_convergence(mc_result)
            png_b64 = _fig_to_base64(fig_conv)
            plt.close(fig_conv)
            section += (
                f'<div class="chart-container">'
                f'<img src="data:image/png;base64,{png_b64}" alt="MC convergence chart">'
                f"</div>"
            )
        except Exception as exc:
            logger.warning("Could not render MC charts: %s", exc)

        return section

    def _overfit_html(self, overfit_report) -> str:
        """Render the overfitting analysis section."""
        rows = [
            ("Deflated Sharpe Ratio (DSR)", f"{overfit_report.dsr:.4f}",
             "≥ 0.90", overfit_report.dsr_pass),
            ("Walk-Forward Efficiency (WFE)", f"{overfit_report.wfe:.4f}",
             "≥ 0.50", overfit_report.wfe_pass),
            ("Plateau Test (CV)", f"{overfit_report.plateau_cv:.4f}",
             "CV < 0.30", overfit_report.plateau_pass),
        ]
        tbl_rows = ""
        for name, value, threshold, passed in rows:
            cls = "pass" if passed else "fail"
            txt = "PASS" if passed else "FAIL"
            tbl_rows += (
                f"<tr><td>{name}</td>"
                f'<td class="{cls}">{txt}</td>'
                f"<td>{value}</td><td>{threshold}</td></tr>"
            )
        notes_html = "".join(f"<li>{n}</li>" for n in overfit_report.notes)
        return (
            "<h2>Overfitting Analysis</h2>"
            "<table>"
            "<thead><tr><th>Check</th><th>Status</th><th>Value</th><th>Required</th></tr></thead>"
            f"<tbody>{tbl_rows}</tbody>"
            "</table>"
            f"<ul class='rec-list'>{notes_html}</ul>"
        )

    def _recommendations_html(self, recommendations: List[str]) -> str:
        items = "".join(f"<li>{r}</li>" for r in recommendations)
        return (
            "<h2>Recommendations</h2>"
            f'<ul class="rec-list">{items}</ul>'
        )


# =============================================================================
# Utility
# =============================================================================


def _fig_to_base64(fig) -> str:
    """Encode a matplotlib figure as a base-64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
