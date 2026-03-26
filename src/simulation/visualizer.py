"""
Visualisation tools for Monte Carlo simulation results.

MCVisualizer produces four chart types:

1. equity_fan — fan chart of sampled equity curves showing the spread
   of outcomes, with percentile bands and colour-coded pass/fail paths.

2. pass_rate_convergence — running pass-rate versus simulation number,
   useful for confirming that 10,000 simulations is sufficient and for
   diagnosing any instability in the estimate.

3. failure_breakdown — horizontal bar (or pie) chart decomposing all
   failed attempts into their terminal failure reason: daily drawdown,
   total drawdown, or timeout.

4. daily_dd_distribution — histogram of the worst daily drawdown seen
   in each simulation, showing how close the strategy comes to the limit
   in typical attempts.

All methods return matplotlib Figure objects so callers can save, show,
or embed them as they please.  Matplotlib is an optional dependency; if
it is not installed, an ImportError is raised only at call time.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from src.simulation.monte_carlo import ChallengeOutcome, MCResult

logger = logging.getLogger(__name__)


# =============================================================================
# MCVisualizer
# =============================================================================

class MCVisualizer:
    """Visualise Monte Carlo simulation results.

    Parameters
    ----------
    style:
        Matplotlib style name.  Default: 'seaborn-v0_8-darkgrid' with
        automatic fallback to 'ggplot' for older matplotlib versions.
    figsize_default:
        Default figure size (width, height) in inches.
    """

    _PASS_COLOUR = "#2ecc71"
    _FAIL_COLOUR = "#e74c3c"
    _NEUTRAL_COLOUR = "#3498db"

    def __init__(
        self,
        style: str = "seaborn-v0_8-darkgrid",
        figsize_default: tuple = (12, 6),
    ) -> None:
        self._style = style
        self._figsize = figsize_default

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def equity_fan(
        self,
        outcomes: List[ChallengeOutcome],
        n_sample: int = 100,
        show_percentile_bands: bool = True,
    ):
        """Fan chart of equity curves from a sample of simulation outcomes.

        Parameters
        ----------
        outcomes:
            List of ChallengeOutcome objects from MCResult.outcomes.
        n_sample:
            How many individual equity curves to draw.  Curves are chosen
            to include representative pass and fail paths.
        show_percentile_bands:
            Overlay 10th / 50th / 90th percentile bands computed from all
            outcomes rather than just the sampled subset.

        Returns
        -------
        matplotlib.figure.Figure
        """
        plt = self._import_matplotlib()
        fig, ax = plt.subplots(figsize=self._figsize)
        self._apply_style(plt)

        if not outcomes:
            ax.set_title("Equity Fan — no data")
            return fig

        # Separate passing and failing outcomes that have equity curves
        with_curves = [o for o in outcomes if o.equity_curve]
        passed = [o for o in with_curves if o.passed]
        failed = [o for o in with_curves if not o.passed]

        rng = np.random.default_rng(42)

        # Sample curves: prefer a balanced mix of pass/fail
        def _sample(lst, k):
            if len(lst) <= k:
                return lst
            idx = rng.choice(len(lst), size=k, replace=False)
            return [lst[i] for i in idx]

        n_pass_sample = min(n_sample // 2, len(passed))
        n_fail_sample = min(n_sample - n_pass_sample, len(failed))

        sampled_pass = _sample(passed, n_pass_sample)
        sampled_fail = _sample(failed, n_fail_sample)

        # Normalise equity curves to percentages from initial balance
        initial = outcomes[0].equity_curve[0] if outcomes[0].equity_curve else 10000.0

        def _pct_curve(o: ChallengeOutcome):
            curve = np.array(o.equity_curve, dtype=float)
            return (curve / initial - 1.0) * 100.0

        for o in sampled_fail:
            ax.plot(_pct_curve(o), color=self._FAIL_COLOUR, alpha=0.15, linewidth=0.8)

        for o in sampled_pass:
            ax.plot(_pct_curve(o), color=self._PASS_COLOUR, alpha=0.25, linewidth=0.8)

        # Percentile bands from all outcomes
        if show_percentile_bands and len(with_curves) >= 10:
            max_len = max(len(o.equity_curve) for o in with_curves)
            padded = np.full((len(with_curves), max_len), np.nan)
            for i, o in enumerate(with_curves):
                c = np.array(o.equity_curve, dtype=float)
                padded[i, : len(c)] = (c / initial - 1.0) * 100.0

            x = np.arange(max_len)
            p10 = np.nanpercentile(padded, 10, axis=0)
            p50 = np.nanpercentile(padded, 50, axis=0)
            p90 = np.nanpercentile(padded, 90, axis=0)

            ax.fill_between(x, p10, p90, color=self._NEUTRAL_COLOUR, alpha=0.12,
                            label="10th–90th percentile")
            ax.plot(x, p50, color=self._NEUTRAL_COLOUR, linewidth=2.0,
                    linestyle="--", label="Median (50th pct)")

        # Reference lines
        initial_bal = outcomes[0].equity_curve[0] if outcomes[0].equity_curve else 10000.0
        ax.axhline(8.0, color=self._PASS_COLOUR, linewidth=1.5, linestyle=":",
                   label="Profit target (8%)")
        ax.axhline(-10.0, color=self._FAIL_COLOUR, linewidth=1.5, linestyle=":",
                   label="Total DD limit (10%)")
        ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="-")

        ax.set_title(
            f"Equity Curve Fan — {len(with_curves):,} simulations "
            f"({len(passed):,} passed, {len(failed):,} failed)"
        )
        ax.set_xlabel("Day")
        ax.set_ylabel("Account Change (%)")
        ax.legend(loc="upper left", fontsize=8)

        return fig

    def pass_rate_convergence(
        self,
        result: MCResult,
    ):
        """Running pass-rate vs simulation number.

        Shows whether the estimate has stabilised by 10,000 simulations.
        A vertical line marks the convergence point if detected.

        Returns
        -------
        matplotlib.figure.Figure
        """
        plt = self._import_matplotlib()
        fig, ax = plt.subplots(figsize=self._figsize)
        self._apply_style(plt)

        rates = result.running_pass_rates
        if not rates:
            ax.set_title("Pass-Rate Convergence — no data")
            return fig

        x = np.arange(1, len(rates) + 1)
        ax.plot(x, rates, color=self._NEUTRAL_COLOUR, linewidth=1.2,
                label="Running pass rate (%)")

        # Final pass rate reference line
        final = rates[-1]
        ax.axhline(final, color=self._PASS_COLOUR, linewidth=1.5, linestyle="--",
                   label=f"Final: {final:.2f}%")

        # Convergence marker
        if result.convergence_reached and result.convergence_at < len(rates):
            ax.axvline(result.convergence_at, color="orange", linewidth=1.5,
                       linestyle=":", label=f"Converged @ sim {result.convergence_at:,}")

        # Tolerance band
        tol = 1.0
        ax.fill_between(x, final - tol, final + tol,
                        color=self._PASS_COLOUR, alpha=0.08,
                        label=f"±{tol:.0f}% tolerance")

        ax.set_title("Pass-Rate Convergence")
        ax.set_xlabel("Simulations completed")
        ax.set_ylabel("Pass rate (%)")
        ax.legend(loc="upper right", fontsize=8)

        return fig

    def failure_breakdown(
        self,
        result: MCResult,
        chart_type: str = "bar",
    ):
        """Failure mode breakdown chart.

        Parameters
        ----------
        result:
            MCResult from MonteCarloSimulator.run().
        chart_type:
            'bar' (default) for a horizontal bar chart, 'pie' for a
            pie chart.

        Returns
        -------
        matplotlib.figure.Figure
        """
        plt = self._import_matplotlib()
        fig, ax = plt.subplots(figsize=(9, 5))
        self._apply_style(plt)

        n = result.n_simulations or 1

        labels = ["Passed", "Daily DD Fail", "Total DD Fail", "Timeout"]
        values = [
            result.pass_rate,
            result.daily_dd_failure_rate,
            result.total_dd_failure_rate,
            result.timeout_rate,
        ]
        colours = [
            self._PASS_COLOUR,
            "#e67e22",
            self._FAIL_COLOUR,
            "#9b59b6",
        ]

        if chart_type == "pie":
            wedges, texts, autotexts = ax.pie(
                values,
                labels=labels,
                colors=colours,
                autopct="%1.1f%%",
                startangle=90,
                pctdistance=0.75,
            )
            for at in autotexts:
                at.set_fontsize(9)
            ax.set_title(f"Challenge Outcome Breakdown (n={n:,})")
        else:
            # Horizontal bar chart
            y = np.arange(len(labels))
            bars = ax.barh(y, values, color=colours, edgecolor="white", height=0.5)

            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            ax.set_xlabel("Rate (%)")
            ax.set_title(f"Challenge Outcome Breakdown (n={n:,})")
            ax.set_xlim(0, 100)

            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_width() + 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%",
                    va="center", ha="left", fontsize=9,
                )

        return fig

    def daily_dd_distribution(
        self,
        outcomes: List[ChallengeOutcome],
        bins: int = 40,
    ):
        """Histogram of the worst daily drawdown seen per simulation.

        Parameters
        ----------
        outcomes:
            List from MCResult.outcomes.
        bins:
            Number of histogram bins.

        Returns
        -------
        matplotlib.figure.Figure
        """
        plt = self._import_matplotlib()
        fig, ax = plt.subplots(figsize=self._figsize)
        self._apply_style(plt)

        if not outcomes:
            ax.set_title("Daily DD Distribution — no data")
            return fig

        max_dds = np.array([o.max_daily_dd for o in outcomes], dtype=float)

        ax.hist(
            max_dds,
            bins=bins,
            color=self._NEUTRAL_COLOUR,
            edgecolor="white",
            alpha=0.85,
            label="Max daily DD per sim",
        )

        # Mark the hard limit and circuit breaker on the x-axis
        circuit_pct = 2.0  # default; cannot access simulator params here easily
        limit_pct = 5.0

        ax.axvline(limit_pct, color=self._FAIL_COLOUR, linewidth=1.5,
                   linestyle=":", label=f"Daily DD limit ({limit_pct:.0f}%)")
        ax.axvline(circuit_pct, color="orange", linewidth=1.5,
                   linestyle=":", label=f"Circuit breaker ({circuit_pct:.0f}%)")

        mean_dd = float(max_dds.mean())
        ax.axvline(mean_dd, color=self._PASS_COLOUR, linewidth=1.5,
                   linestyle="--", label=f"Mean: {mean_dd:.2f}%")

        ax.set_title("Distribution of Max Daily Drawdown Across Simulations")
        ax.set_xlabel("Max Daily Drawdown (%)")
        ax.set_ylabel("Count")
        ax.legend(loc="upper right", fontsize=8)

        return fig

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_style(self, plt) -> None:
        """Apply the configured style, falling back gracefully."""
        try:
            plt.style.use(self._style)
        except OSError:
            try:
                plt.style.use("ggplot")
            except OSError:
                pass  # use matplotlib defaults

    @staticmethod
    def _import_matplotlib():
        """Import matplotlib.pyplot, raising a clear error if missing."""
        try:
            import matplotlib.pyplot as plt  # type: ignore
            return plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for visualisation. "
                "Install it with: pip install matplotlib"
            ) from exc
