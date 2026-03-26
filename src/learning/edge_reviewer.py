"""
Edge performance review for the adaptive learning system.

EdgeReviewer queries the trade history to assess each strategy edge's
real-world contribution: win rate when the edge is active, filter rate
(fraction of signals it removes), and marginal R impact.

Suggestions produced here are advisory only — they require explicit
human approval before any parameter changes are applied.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Minimum number of trades that must be attributed to an edge before the
# reviewer emits suggestions.  Below this threshold we have too little
# data to distinguish signal from noise.
_MIN_TRADES_FOR_SUGGESTION: int = 30

# Win rate below this level triggers a "consider disabling" suggestion.
_LOW_WIN_RATE_THRESHOLD: float = 0.38

# Win rate above this level when edge is currently disabled triggers a
# "consider enabling" suggestion.
_HIGH_WIN_RATE_THRESHOLD: float = 0.52

# Edge names that the reviewer tracks — mirrors the list in edge_tester.py
_TRACKED_EDGES: List[str] = [
    "time_of_day",
    "day_of_week",
    "london_open_delay",
    "candle_close_confirmation",
    "spread_filter",
    "friday_close",
    "regime_filter",
    "time_stop",
    "bb_squeeze",
    "confluence_scoring",
    "equity_curve",
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EdgeReviewResult:
    """Performance statistics for a single strategy edge.

    Attributes
    ----------
    edge_name:
        Identifier of the edge (e.g. ``"regime_filter"``).
    total_trades_affected:
        Number of closed trades where this edge was recorded as active
        (i.e. edge_signals column contains this edge name).
    win_rate_when_active:
        Win rate fraction for trades where the edge was active.
    avg_r_when_active:
        Mean R-multiple for trades where the edge was active.
    filter_rate:
        Fraction of raw signals filtered / removed by this edge, based on
        the edge_signals log.  Value in [0, 1].
    marginal_impact:
        Difference in avg_r between trades WITH vs WITHOUT this edge.
        Positive means the edge improves outcome R.
    """

    edge_name: str
    total_trades_affected: int
    win_rate_when_active: float
    avg_r_when_active: float
    filter_rate: float
    marginal_impact: float


@dataclass
class EdgeSuggestion:
    """A suggested change to an edge's enabled/disabled state.

    These are advisory only — the engine or operator must approve
    before applying any configuration change.

    Attributes
    ----------
    edge_name:
        The edge to change.
    current_state:
        Current enabled flag (True = enabled, False = disabled).
    suggested_state:
        Proposed new state.
    reason:
        Human-readable justification for the suggestion.
    confidence:
        Confidence level of the suggestion in [0, 1], based on the
        number of supporting trades (ramps from 0 to 1.0 over 100 trades).
    """

    edge_name: str
    current_state: bool
    suggested_state: bool
    reason: str
    confidence: float


# ---------------------------------------------------------------------------
# EdgeReviewer
# ---------------------------------------------------------------------------

class EdgeReviewer:
    """Review each strategy edge's real-world performance.

    Parameters
    ----------
    db_pool:
        A DatabasePool with ``get_cursor()``.  May be None for testing;
        inject a DataFrame via ``_inject_trade_cache``.
    """

    # Columns needed from the trades table.  edge_signals is expected to be
    # a TEXT/JSONB field listing which edges were active at entry; filter_log
    # records edges that blocked a potential signal.
    _TRADE_QUERY = """
        SELECT
            t.id,
            t.r_multiple,
            t.edge_signals,
            t.filter_log
        FROM trades t
        WHERE t.status = 'closed'
          AND t.r_multiple IS NOT NULL
        ORDER BY t.entry_time ASC
    """

    # Query to get the current enabled state of each edge from config
    _CONFIG_QUERY = """
        SELECT key, value
        FROM system_config
        WHERE key LIKE 'edge.%.enabled'
    """

    def __init__(self, db_pool=None) -> None:
        self._db_pool = db_pool
        self._trade_cache: Optional[pd.DataFrame] = None
        # Map of edge_name → enabled: used to compute suggestions.
        # Pre-populated with all edges enabled (conservative default).
        self._edge_states: dict = {e: True for e in _TRACKED_EDGES}

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------

    def _inject_trade_cache(self, df: pd.DataFrame) -> None:
        """Inject a pre-built DataFrame for testing without a database."""
        self._trade_cache = df

    def _inject_edge_states(self, states: dict) -> None:
        """Override the edge enabled/disabled states for testing."""
        self._edge_states = dict(states)

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def _load_trades(self) -> pd.DataFrame:
        """Load closed trades with edge signal metadata."""
        if self._trade_cache is not None:
            return self._trade_cache.copy()

        if self._db_pool is None:
            logger.warning("EdgeReviewer: no db_pool configured — returning empty trade set")
            return pd.DataFrame()

        try:
            with self._db_pool.get_cursor() as cur:
                cur.execute(self._TRADE_QUERY)
                rows = cur.fetchall()
        except Exception as exc:
            logger.error("EdgeReviewer: failed to load trades: %s", exc)
            return pd.DataFrame()

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)

    def _load_edge_states(self) -> None:
        """Attempt to load current edge enabled states from the database."""
        if self._db_pool is None:
            return

        try:
            with self._db_pool.get_cursor() as cur:
                cur.execute(self._CONFIG_QUERY)
                rows = cur.fetchall()
            for row in rows:
                key = str(row["key"])           # e.g. "edge.regime_filter.enabled"
                parts = key.split(".")
                if len(parts) == 3:
                    edge_name = parts[1]
                    value = str(row["value"]).lower()
                    self._edge_states[edge_name] = value in ("true", "1", "yes")
        except Exception as exc:
            logger.debug("EdgeReviewer: could not load edge states: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def review_all_edges(self) -> List[EdgeReviewResult]:
        """Calculate performance metrics for every tracked edge.

        Returns
        -------
        List of EdgeReviewResult, one per tracked edge.  Edges with no
        trade data return zero-filled results.
        """
        df = self._load_trades()
        results: List[EdgeReviewResult] = []

        for edge_name in _TRACKED_EDGES:
            result = self._review_single_edge(edge_name, df)
            results.append(result)
            logger.debug(
                "Edge '%s': n=%d, wr=%.3f, avg_r=%.3f, filter_rate=%.3f, marginal=%.3f",
                result.edge_name,
                result.total_trades_affected,
                result.win_rate_when_active,
                result.avg_r_when_active,
                result.filter_rate,
                result.marginal_impact,
            )

        return results

    def suggest_edge_changes(self) -> List[EdgeSuggestion]:
        """Propose enabling/disabling edges based on performance.

        Suggestions require human approval before being applied.

        Returns
        -------
        List of EdgeSuggestion with proposed state changes.  An empty list
        means no changes are warranted with the current data.
        """
        self._load_edge_states()
        review_results = self.review_all_edges()
        suggestions: List[EdgeSuggestion] = []

        for review in review_results:
            n = review.total_trades_affected
            if n < _MIN_TRADES_FOR_SUGGESTION:
                continue

            # Confidence ramps from 0 to 1 over the first 100 trades
            confidence = min(1.0, n / 100.0)

            current_state = self._edge_states.get(review.edge_name, True)
            suggestion = self._evaluate_suggestion(review, current_state, confidence)
            if suggestion is not None:
                suggestions.append(suggestion)

        return suggestions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _review_single_edge(
        self, edge_name: str, df: pd.DataFrame
    ) -> EdgeReviewResult:
        """Compute EdgeReviewResult for one edge from the trade DataFrame."""
        if df.empty:
            return self._empty_result(edge_name)

        # Trades where this edge was active at signal time
        with_edge = self._filter_trades_with_edge(df, edge_name, "edge_signals")
        # Trades where this edge filtered out a potential signal
        filtered_signals = self._filter_trades_with_edge(df, edge_name, "filter_log")

        n_with = len(with_edge)
        n_total = len(df)
        n_filtered = len(filtered_signals)

        if n_with == 0:
            return self._empty_result(edge_name)

        r_values = with_edge["r_multiple"].astype(float)
        win_rate = float((r_values > 0).mean())
        avg_r = float(r_values.mean())

        # Marginal impact: avg_r WITH the edge vs avg_r WITHOUT it
        without_edge = df[~df.index.isin(with_edge.index)]
        if len(without_edge) > 0:
            avg_r_without = float(without_edge["r_multiple"].astype(float).mean())
            marginal_impact = avg_r - avg_r_without
        else:
            marginal_impact = 0.0

        # Filter rate: fraction of total candidate signals blocked by this edge
        total_candidates = n_total + n_filtered
        filter_rate = n_filtered / total_candidates if total_candidates > 0 else 0.0

        return EdgeReviewResult(
            edge_name=edge_name,
            total_trades_affected=n_with,
            win_rate_when_active=round(win_rate, 4),
            avg_r_when_active=round(avg_r, 4),
            filter_rate=round(filter_rate, 4),
            marginal_impact=round(marginal_impact, 4),
        )

    @staticmethod
    def _filter_trades_with_edge(
        df: pd.DataFrame, edge_name: str, column: str
    ) -> pd.DataFrame:
        """Return rows where *column* contains *edge_name*.

        Handles both string and list/dict serialisations stored in the
        edge_signals / filter_log columns.
        """
        if column not in df.columns:
            return pd.DataFrame()

        def _contains_edge(cell) -> bool:
            if cell is None:
                return False
            if isinstance(cell, list):
                return edge_name in cell
            if isinstance(cell, dict):
                return edge_name in cell
            # Treat as string (CSV or JSON)
            return edge_name in str(cell)

        mask = df[column].apply(_contains_edge)
        return df[mask]

    def _evaluate_suggestion(
        self,
        review: EdgeReviewResult,
        current_state: bool,
        confidence: float,
    ) -> Optional[EdgeSuggestion]:
        """Determine whether a state-change suggestion is warranted."""
        if current_state:
            # Edge is currently enabled — suggest disabling if it underperforms
            if review.win_rate_when_active < _LOW_WIN_RATE_THRESHOLD:
                return EdgeSuggestion(
                    edge_name=review.edge_name,
                    current_state=True,
                    suggested_state=False,
                    reason=(
                        f"Win rate when active is {review.win_rate_when_active:.1%}, "
                        f"below the {_LOW_WIN_RATE_THRESHOLD:.1%} threshold "
                        f"(n={review.total_trades_affected} trades, "
                        f"marginal_impact={review.marginal_impact:+.3f}R)."
                    ),
                    confidence=round(confidence, 3),
                )
        else:
            # Edge is currently disabled — suggest enabling if it would help
            if review.win_rate_when_active > _HIGH_WIN_RATE_THRESHOLD:
                return EdgeSuggestion(
                    edge_name=review.edge_name,
                    current_state=False,
                    suggested_state=True,
                    reason=(
                        f"Historical win rate when active is {review.win_rate_when_active:.1%}, "
                        f"above the re-enable threshold of {_HIGH_WIN_RATE_THRESHOLD:.1%} "
                        f"(n={review.total_trades_affected} trades, "
                        f"marginal_impact={review.marginal_impact:+.3f}R)."
                    ),
                    confidence=round(confidence, 3),
                )

        return None

    @staticmethod
    def _empty_result(edge_name: str) -> EdgeReviewResult:
        """Return a zero-filled EdgeReviewResult for an edge with no data."""
        return EdgeReviewResult(
            edge_name=edge_name,
            total_trades_affected=0,
            win_rate_when_active=0.0,
            avg_r_when_active=0.0,
            filter_rate=0.0,
            marginal_impact=0.0,
        )
