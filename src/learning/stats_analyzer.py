"""
Statistical analysis of trade performance by category.

StatsAnalyzer queries closed trade history from the database and computes
win rate breakdowns by session, ADX regime, confluence tier, and day of week.
Results drive the statistical filtering phase (100-500 trades) of the
adaptive learning engine.

Filter thresholds are intentionally conservative — a category must have a
minimum number of trades before filtering is applied, to avoid reacting to
noise in small samples.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ADX regime boundaries
_ADX_LOW_MAX: float = 20.0      # ADX < 20 → low trend / ranging
_ADX_MED_MAX: float = 35.0      # 20 <= ADX < 35 → medium trend
# ADX >= 35 → high trend

# Confluence tier labels mapped from integer score ranges
# These align with the existing confluence_scoring edge tiers
_CONFLUENCE_TIERS: List[Tuple[str, int, int]] = [
    ("C",   0,  4),    # score 0-4 → tier C
    ("B",   5,  6),    # score 5-6 → tier B
    ("A+",  7, 99),    # score 7+  → tier A+
]

# Session canonical name set
_KNOWN_SESSIONS = ("london", "new_york", "asian", "overlap")


def _adx_regime(adx: float) -> str:
    """Categorise an ADX value into a named regime."""
    if adx < _ADX_LOW_MAX:
        return "low"
    if adx < _ADX_MED_MAX:
        return "medium"
    return "high"


def _confluence_tier(score: int) -> str:
    """Map an integer confluence score to a tier label."""
    for label, low, high in _CONFLUENCE_TIERS:
        if low <= score <= high:
            return label
    return "C"


class StatsAnalyzer:
    """Statistical analysis of trade performance by category.

    Parameters
    ----------
    db_pool:
        A DatabasePool instance with a ``get_cursor()`` context manager.
        May be None; in that case all methods return empty results and log
        a warning — used in tests with mock data injected via
        ``_inject_trade_cache``.
    """

    # Columns fetched from the DB for analysis
    _TRADE_QUERY = """
        SELECT
            t.id,
            t.r_multiple,
            t.confluence_score,
            t.signal_tier,
            t.direction,
            EXTRACT(DOW FROM t.entry_time) AS day_of_week,
            mc.session,
            mc.adx_value
        FROM trades t
        LEFT JOIN market_context mc ON mc.trade_id = t.id
        WHERE t.status = 'closed'
          AND t.r_multiple IS NOT NULL
        ORDER BY t.entry_time ASC
    """

    def __init__(self, db_pool=None) -> None:
        self._db_pool = db_pool
        # Optional cache: tests inject a DataFrame here to avoid DB access
        self._trade_cache: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Test helper (not for production use)
    # ------------------------------------------------------------------

    def _inject_trade_cache(self, df: pd.DataFrame) -> None:
        """Inject a pre-built DataFrame for testing without a database."""
        self._trade_cache = df

    # ------------------------------------------------------------------
    # Core data access
    # ------------------------------------------------------------------

    def _load_trades(self) -> pd.DataFrame:
        """Return a DataFrame of all closed trades with context data."""
        if self._trade_cache is not None:
            return self._trade_cache.copy()

        if self._db_pool is None:
            logger.warning("StatsAnalyzer: no db_pool configured — returning empty trade set")
            return pd.DataFrame()

        try:
            with self._db_pool.get_cursor() as cur:
                cur.execute(self._TRADE_QUERY)
                rows = cur.fetchall()
        except Exception as exc:
            logger.error("StatsAnalyzer: failed to load trades: %s", exc)
            return pd.DataFrame()

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Win-rate breakdowns
    # ------------------------------------------------------------------

    def win_rate_by_session(self, min_trades: int = 20) -> Dict[str, dict]:
        """Win rate breakdown by trading session.

        Parameters
        ----------
        min_trades:
            Minimum number of trades in a session bucket before reporting
            that session's statistics.

        Returns
        -------
        dict keyed by session name, each containing:
        ``{"win_rate": float, "n_trades": int, "avg_r": float}``
        or an empty dict if the session has fewer than min_trades.
        """
        df = self._load_trades()
        if df.empty or "session" not in df.columns:
            return {}

        df = df.copy()
        df["win"] = df["r_multiple"].astype(float) > 0.0

        result: Dict[str, dict] = {}
        for session, group in df.groupby("session"):
            n = len(group)
            if n < min_trades:
                continue
            win_rate = float(group["win"].mean())
            avg_r = float(group["r_multiple"].astype(float).mean())
            result[str(session)] = {
                "win_rate": round(win_rate, 4),
                "n_trades": n,
                "avg_r": round(avg_r, 4),
            }

        return result

    def win_rate_by_regime(self, min_trades: int = 20) -> Dict[str, dict]:
        """Win rate breakdown by ADX trend regime.

        Regimes: ``low`` (ADX < 20), ``medium`` (20 <= ADX < 35),
        ``high`` (ADX >= 35).

        Parameters
        ----------
        min_trades:
            Minimum trades required per regime bucket.

        Returns
        -------
        dict keyed by ``"low"``, ``"medium"``, ``"high"``.
        """
        df = self._load_trades()
        if df.empty or "adx_value" not in df.columns:
            return {}

        df = df.copy()
        df["adx_value"] = df["adx_value"].fillna(0.0).astype(float)
        df["regime"] = df["adx_value"].apply(_adx_regime)
        df["win"] = df["r_multiple"].astype(float) > 0.0

        result: Dict[str, dict] = {}
        for regime, group in df.groupby("regime"):
            n = len(group)
            if n < min_trades:
                continue
            win_rate = float(group["win"].mean())
            avg_r = float(group["r_multiple"].astype(float).mean())
            result[str(regime)] = {
                "win_rate": round(win_rate, 4),
                "n_trades": n,
                "avg_r": round(avg_r, 4),
            }

        return result

    def win_rate_by_confluence(self, min_trades: int = 10) -> Dict[str, dict]:
        """Win rate by confluence score tier (A+, B, C).

        Parameters
        ----------
        min_trades:
            Minimum trades per tier bucket.

        Returns
        -------
        dict keyed by ``"A+"``, ``"B"``, ``"C"``.
        """
        df = self._load_trades()
        if df.empty or "confluence_score" not in df.columns:
            return {}

        df = df.copy()
        df["confluence_score"] = df["confluence_score"].fillna(0).astype(int)
        df["tier"] = df["confluence_score"].apply(_confluence_tier)
        df["win"] = df["r_multiple"].astype(float) > 0.0

        result: Dict[str, dict] = {}
        for tier, group in df.groupby("tier"):
            n = len(group)
            if n < min_trades:
                continue
            win_rate = float(group["win"].mean())
            avg_r = float(group["r_multiple"].astype(float).mean())
            result[str(tier)] = {
                "win_rate": round(win_rate, 4),
                "n_trades": n,
                "avg_r": round(avg_r, 4),
            }

        return result

    def win_rate_by_day(self, min_trades: int = 15) -> Dict[str, dict]:
        """Win rate by day of week (Monday–Friday).

        Parameters
        ----------
        min_trades:
            Minimum trades per day bucket.

        Returns
        -------
        dict keyed by ``"Monday"``, ``"Tuesday"`` … ``"Friday"``.
        """
        _DOW_NAMES = {
            0: "Sunday", 1: "Monday", 2: "Tuesday",
            3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday",
        }

        df = self._load_trades()
        if df.empty or "day_of_week" not in df.columns:
            return {}

        df = df.copy()
        df["day_of_week"] = df["day_of_week"].fillna(0).astype(int)
        df["day_name"] = df["day_of_week"].map(_DOW_NAMES)
        df["win"] = df["r_multiple"].astype(float) > 0.0

        result: Dict[str, dict] = {}
        for day_name, group in df.groupby("day_name"):
            n = len(group)
            if n < min_trades:
                continue
            win_rate = float(group["win"].mean())
            avg_r = float(group["r_multiple"].astype(float).mean())
            result[str(day_name)] = {
                "win_rate": round(win_rate, 4),
                "n_trades": n,
                "avg_r": round(avg_r, 4),
            }

        return result

    def performance_heatmap(self) -> pd.DataFrame:
        """Session × regime performance matrix.

        Returns
        -------
        pd.DataFrame with sessions as rows, regimes as columns, and
        ``(win_rate, n_trades)`` tuples as values.  Returns an empty
        DataFrame when insufficient data is available.
        """
        df = self._load_trades()
        if df.empty or "session" not in df.columns or "adx_value" not in df.columns:
            return pd.DataFrame()

        df = df.copy()
        df["adx_value"] = df["adx_value"].fillna(0.0).astype(float)
        df["regime"] = df["adx_value"].apply(_adx_regime)
        df["win"] = df["r_multiple"].astype(float) > 0.0

        sessions = sorted(df["session"].dropna().unique())
        regimes = ["low", "medium", "high"]

        matrix: Dict[str, Dict[str, str]] = {}
        for session in sessions:
            matrix[session] = {}
            for regime in regimes:
                mask = (df["session"] == session) & (df["regime"] == regime)
                subset = df[mask]
                n = len(subset)
                if n == 0:
                    matrix[session][regime] = "0 trades"
                else:
                    wr = float(subset["win"].mean())
                    matrix[session][regime] = f"{wr:.1%} ({n})"

        return pd.DataFrame(matrix).T

    # ------------------------------------------------------------------
    # Filter predicates
    # ------------------------------------------------------------------

    def should_filter_session(
        self, session: str, min_wr: float = 0.40, min_trades: int = 20
    ) -> bool:
        """Decide whether to skip trading in a session based on win rate.

        Returns True (filter out / skip) when the session's historical
        win rate is below *min_wr* and at least *min_trades* have been
        taken in that session.

        Parameters
        ----------
        session:
            Session name (e.g. ``"london"``, ``"new_york"``).
        min_wr:
            Win rate threshold below which the session is filtered.
        min_trades:
            Minimum trade count required before the filter activates.

        Returns
        -------
        bool — True means "skip this session".
        """
        stats = self.win_rate_by_session(min_trades=min_trades)
        if session not in stats:
            return False  # insufficient data → do not filter
        return stats[session]["win_rate"] < min_wr

    def should_filter_regime(
        self, adx: float, min_wr: float = 0.40, min_trades: int = 20
    ) -> bool:
        """Decide whether to skip a trade based on the current ADX regime.

        Parameters
        ----------
        adx:
            Current ADX value.
        min_wr:
            Win rate threshold.
        min_trades:
            Minimum trades required for the regime bucket before filtering.

        Returns
        -------
        bool — True means "skip this regime".
        """
        regime = _adx_regime(adx)
        stats = self.win_rate_by_regime(min_trades=min_trades)
        if regime not in stats:
            return False
        return stats[regime]["win_rate"] < min_wr

    def get_all_stats(self, min_trades_session: int = 20,
                      min_trades_regime: int = 20,
                      min_trades_confluence: int = 10,
                      min_trades_day: int = 15) -> dict:
        """Return all statistical breakdowns in a single dict.

        Convenience method used by the adaptive engine and report generator.
        """
        return {
            "by_session": self.win_rate_by_session(min_trades=min_trades_session),
            "by_regime": self.win_rate_by_regime(min_trades=min_trades_regime),
            "by_confluence": self.win_rate_by_confluence(min_trades=min_trades_confluence),
            "by_day": self.win_rate_by_day(min_trades=min_trades_day),
        }
