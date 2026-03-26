"""
Adaptive learning engine for the XAU/USD Ichimoku trading system.

The engine operates in three learning phases that activate automatically
based on the number of completed trades:

Phase 1 — Mechanical (0–99 trades)
    Pure signal execution with full logging.  No learning adjustments
    are applied; the system builds a clean baseline dataset.

Phase 2 — Statistical (100–499 trades)
    Session and regime win-rate filters are applied.  Setups in
    historically poor-performing conditions are skipped.

Phase 3 — Similarity-based (500+ trades)
    pgvector nearest-neighbour lookups provide pre-trade expected
    performance.  Confidence scores influence position sizing within
    the hard risk bounds set by the risk management module.

CRITICAL CONSTRAINTS
--------------------
- This engine CANNOT override risk management rules.
- Confidence adjustments are capped at +/-0.5.
- All parameter suggestions require human approval before use.
- During the mechanical phase (< 100 trades) no adjustments are made.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .embeddings import EmbeddingEngine
from .similarity import SimilaritySearch, SimilarTrade, PerformanceStats
from .stats_analyzer import StatsAnalyzer

logger = logging.getLogger(__name__)

# Learning phase thresholds
_MECHANICAL_MAX: int = 99       # 0–99 inclusive  → mechanical
_STATISTICAL_MAX: int = 499     # 100–499 inclusive → statistical
# 500+                          → similarity-based

# Maximum absolute confidence adjustment that the engine may propose.
# The risk management module enforces its own hard limits; this is an
# additional guard so the learning layer cannot push sizing out of bounds.
_MAX_CONFIDENCE_DELTA: float = 0.5

# Minimum similarity confidence before adjustment is applied.
# Below this the engine has too few matching trades to be reliable.
_MIN_SIMILARITY_CONFIDENCE: float = 0.25

# Statistical filter threshold — sessions / regimes with win rates below
# this value are skipped in the statistical phase.
_FILTER_WIN_RATE_THRESHOLD: float = 0.40
_FILTER_MIN_TRADES: int = 20


# ---------------------------------------------------------------------------
# PreTradeInsight
# ---------------------------------------------------------------------------

@dataclass
class PreTradeInsight:
    """Output of a pre-trade analysis.

    All fields are advisory — the trading engine decides whether to act
    on each piece of information.

    Attributes
    ----------
    similar_trades:
        Up to 10 nearest-neighbour historical trades from pgvector search.
    expected_win_rate:
        Win rate of the similar-trade cohort (0.0 when no data).
    expected_avg_r:
        Average R-multiple of the similar-trade cohort.
    expected_expectancy:
        Expectancy (win_rate * avg_win_r + loss_rate * avg_loss_r).
    confidence:
        Similarity confidence in [0, 1].  Ramps up with number of matching
        trades (see SimilaritySearch.get_confidence).
    confidence_adjustment:
        Proposed sizing multiplier delta in [-0.5, +0.5].  Positive means
        "consider increasing confidence / size", negative means "reduce".
    statistical_filters:
        dict of ``{filter_name: bool}`` — True means "filter triggered
        (would skip this setup)".
    recommendation:
        One of ``"proceed"``, ``"caution"``, or ``"skip"``.
    reasoning:
        Human-readable explanation of the recommendation.
    """

    similar_trades: List[SimilarTrade] = field(default_factory=list)
    expected_win_rate: float = 0.0
    expected_avg_r: float = 0.0
    expected_expectancy: float = 0.0
    confidence: float = 0.0
    confidence_adjustment: float = 0.0
    statistical_filters: Dict[str, bool] = field(default_factory=dict)
    recommendation: str = "proceed"
    reasoning: str = ""


# ---------------------------------------------------------------------------
# AdaptiveLearningEngine
# ---------------------------------------------------------------------------

class AdaptiveLearningEngine:
    """Self-learning engine that improves strategy execution over time.

    Parameters
    ----------
    similarity_search:
        Optional SimilaritySearch instance.  When None, similarity-based
        features are disabled (mechanical and statistical phases still work).
    embedding_engine:
        Optional EmbeddingEngine for creating context embeddings.
    db_pool:
        Optional database pool for StatsAnalyzer queries.
    stats_analyzer:
        Optional pre-built StatsAnalyzer.  Created automatically from
        db_pool when not supplied.
    """

    def __init__(
        self,
        similarity_search: Optional[SimilaritySearch] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        db_pool=None,
        stats_analyzer: Optional[StatsAnalyzer] = None,
    ) -> None:
        self._similarity_search = similarity_search
        self._embedding_engine = embedding_engine or EmbeddingEngine()
        self._db_pool = db_pool
        self._stats = stats_analyzer or StatsAnalyzer(db_pool=db_pool)

        # Running trade count — updated by post_trade_analysis
        self._total_trades: int = 0

    # ------------------------------------------------------------------
    # Phase management
    # ------------------------------------------------------------------

    def get_phase(self, total_trades: Optional[int] = None) -> str:
        """Return the current learning phase name.

        Parameters
        ----------
        total_trades:
            Override the internal counter.  Useful for testing.

        Returns
        -------
        One of ``"mechanical"``, ``"statistical"``, ``"similarity"``.
        """
        n = total_trades if total_trades is not None else self._total_trades

        if n <= _MECHANICAL_MAX:
            return "mechanical"
        if n <= _STATISTICAL_MAX:
            return "statistical"
        return "similarity"

    def set_total_trades(self, n: int) -> None:
        """Manually set the total trade counter (used for bootstrapping)."""
        self._total_trades = max(0, int(n))

    # ------------------------------------------------------------------
    # Pre-trade analysis
    # ------------------------------------------------------------------

    def pre_trade_analysis(self, context: dict) -> PreTradeInsight:
        """Analyse a candidate trade setup before entry.

        The returned PreTradeInsight is advisory.  The calling engine
        is responsible for respecting risk rules regardless of the insight.

        Parameters
        ----------
        context:
            Trade context dictionary (same schema as FeatureVectorBuilder).

        Returns
        -------
        PreTradeInsight with similarity data, filter flags, and recommendation.
        """
        phase = self.get_phase()

        # ------------------------------------------------------------------
        # Mechanical phase: log only, no adjustments
        # ------------------------------------------------------------------
        if phase == "mechanical":
            return PreTradeInsight(
                recommendation="proceed",
                reasoning=(
                    f"Mechanical phase ({self._total_trades}/{_MECHANICAL_MAX} trades). "
                    "No learning adjustments applied — pure signal execution."
                ),
            )

        # ------------------------------------------------------------------
        # Statistical phase and above: apply session/regime filters
        # ------------------------------------------------------------------
        stat_filters = self._compute_statistical_filters(context)
        any_filter_triggered = any(stat_filters.values())

        # ------------------------------------------------------------------
        # Similarity phase: add nearest-neighbour lookups
        # ------------------------------------------------------------------
        similar_trades: List[SimilarTrade] = []
        perf: Optional[PerformanceStats] = None
        conf_adjustment: float = 0.0

        if phase == "similarity" and self._similarity_search is not None:
            similar_trades, perf = self._run_similarity_search(context)
            if perf is not None and perf.confidence >= _MIN_SIMILARITY_CONFIDENCE:
                conf_adjustment = self._compute_confidence_adjustment(perf)

        # Build the insight
        insight = self._build_insight(
            phase=phase,
            similar_trades=similar_trades,
            perf=perf,
            stat_filters=stat_filters,
            conf_adjustment=conf_adjustment,
            any_filter_triggered=any_filter_triggered,
        )

        return insight

    # ------------------------------------------------------------------
    # Post-trade analysis
    # ------------------------------------------------------------------

    def post_trade_analysis(self, trade_result: dict) -> None:
        """Update internal state after a trade closes.

        Increments the trade counter.  Future versions may persist
        outcome statistics here.

        Parameters
        ----------
        trade_result:
            Dict containing at minimum ``r_multiple`` and optional
            ``session``, ``adx_value``, ``confluence_score`` keys.
        """
        self._total_trades += 1
        r = trade_result.get("r_multiple", 0.0)
        win = float(r) > 0.0 if r is not None else False

        logger.debug(
            "post_trade_analysis: trade #%d | r=%.3f | win=%s | phase=%s",
            self._total_trades,
            float(r) if r is not None else 0.0,
            win,
            self.get_phase(),
        )

    # ------------------------------------------------------------------
    # Confidence adjustment
    # ------------------------------------------------------------------

    def get_confidence_adjustment(
        self, context: dict, base_confluence: int
    ) -> float:
        """Suggest a confidence / sizing adjustment for a setup.

        The adjustment is capped to [-0.5, +0.5] and is zero in the
        mechanical phase or when similarity data is unavailable.

        Parameters
        ----------
        context:
            Trade context dictionary.
        base_confluence:
            Integer confluence score from the confluence_scoring edge.

        Returns
        -------
        float in [-0.5, +0.5].  Positive = increase confidence slightly;
        negative = reduce confidence slightly.  Zero = no adjustment.
        """
        phase = self.get_phase()

        # No adjustments in the mechanical phase
        if phase == "mechanical":
            return 0.0

        # No similarity search configured
        if self._similarity_search is None:
            return 0.0

        similar_trades, perf = self._run_similarity_search(context)
        if perf is None or perf.confidence < _MIN_SIMILARITY_CONFIDENCE:
            return 0.0

        return self._compute_confidence_adjustment(perf)

    # ------------------------------------------------------------------
    # Statistical filtering
    # ------------------------------------------------------------------

    def should_filter(self, context: dict) -> Tuple[bool, str]:
        """Decide whether to skip a setup based on statistical history.

        Only active in the statistical and similarity phases.

        Parameters
        ----------
        context:
            Trade context dictionary with ``session`` and ``adx_value``.

        Returns
        -------
        (should_skip: bool, reason: str)
        """
        phase = self.get_phase()

        if phase == "mechanical":
            return False, "Mechanical phase — statistical filters inactive."

        filters = self._compute_statistical_filters(context)
        triggered = {k: v for k, v in filters.items() if v}

        if not triggered:
            return False, "No statistical filters triggered."

        reasons = []
        if triggered.get("session_filter"):
            session = context.get("session", "unknown")
            reasons.append(f"session '{session}' has below-threshold win rate")
        if triggered.get("regime_filter"):
            adx = context.get("adx_value", 0.0)
            reasons.append(f"ADX regime ({adx:.1f}) has below-threshold win rate")

        reason_str = "; ".join(reasons)
        return True, f"Statistical filter triggered: {reason_str}."

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_statistical_filters(self, context: dict) -> Dict[str, bool]:
        """Run all statistical filter checks and return a flags dict."""
        filters: Dict[str, bool] = {}

        session = str(context.get("session") or "").lower()
        if session:
            filters["session_filter"] = self._stats.should_filter_session(
                session,
                min_wr=_FILTER_WIN_RATE_THRESHOLD,
                min_trades=_FILTER_MIN_TRADES,
            )

        adx = context.get("adx_value")
        if adx is not None:
            filters["regime_filter"] = self._stats.should_filter_regime(
                float(adx),
                min_wr=_FILTER_WIN_RATE_THRESHOLD,
                min_trades=_FILTER_MIN_TRADES,
            )

        return filters

    def _run_similarity_search(
        self, context: dict
    ) -> Tuple[List[SimilarTrade], Optional[PerformanceStats]]:
        """Run a pgvector nearest-neighbour search and return results."""
        if self._similarity_search is None:
            return [], None

        try:
            embedding = self._embedding_engine.create_embedding(context)
            similar = self._similarity_search.find_similar_trades(
                context_embedding=embedding,
                k=10,
                min_similarity=0.70,
            )
            perf = self._similarity_search.get_performance_stats(similar)
            return similar, perf
        except Exception as exc:
            logger.warning("Similarity search failed: %s", exc)
            return [], None

    @staticmethod
    def _compute_confidence_adjustment(perf: PerformanceStats) -> float:
        """Compute a confidence delta from PerformanceStats.

        Adjustment logic:
        - Expectancy > 0.3 R and win rate > 0.55  → positive boost
        - Expectancy < -0.1 R or win rate < 0.40  → negative reduction
        - Otherwise                                → no adjustment

        The raw adjustment is scaled by the similarity confidence (how
        many matching trades we have) and clamped to [-0.5, +0.5].
        """
        raw_adj: float = 0.0

        if perf.expectancy > 0.3 and perf.win_rate > 0.55:
            # Scale positive boost by how much expectancy exceeds threshold
            raw_adj = min(0.5, (perf.expectancy - 0.3) * 1.0)
        elif perf.expectancy < -0.1 or perf.win_rate < 0.40:
            raw_adj = max(-0.5, perf.expectancy * 0.5)

        # Scale by confidence so we only apply strong adjustments when
        # we have enough similar trades
        scaled = raw_adj * perf.confidence

        # Hard clamp: never exceed the engine's maximum delta
        return max(-_MAX_CONFIDENCE_DELTA, min(_MAX_CONFIDENCE_DELTA, scaled))

    def _build_insight(
        self,
        phase: str,
        similar_trades: List[SimilarTrade],
        perf: Optional[PerformanceStats],
        stat_filters: Dict[str, bool],
        conf_adjustment: float,
        any_filter_triggered: bool,
    ) -> PreTradeInsight:
        """Assemble the PreTradeInsight from computed components."""
        expected_win_rate = perf.win_rate if perf else 0.0
        expected_avg_r = perf.avg_r if perf else 0.0
        expected_expectancy = perf.expectancy if perf else 0.0
        confidence = perf.confidence if perf else 0.0

        # Recommendation logic
        if any_filter_triggered:
            recommendation = "skip"
            reasoning = (
                f"Statistical filter triggered in {phase} phase. "
                "Current session/regime has historically poor performance."
            )
        elif phase == "similarity" and confidence >= _MIN_SIMILARITY_CONFIDENCE:
            if expected_expectancy < -0.15:
                recommendation = "skip"
                reasoning = (
                    f"Similar trades show negative expectancy ({expected_expectancy:.3f}R) "
                    f"with {len(similar_trades)} matching setups "
                    f"(confidence={confidence:.2f})."
                )
            elif expected_expectancy < 0.0:
                recommendation = "caution"
                reasoning = (
                    f"Similar trades show marginal expectancy ({expected_expectancy:.3f}R). "
                    f"Proceed with reduced position or skip at discretion."
                )
            else:
                recommendation = "proceed"
                reasoning = (
                    f"Similar trades: wr={expected_win_rate:.1%}, "
                    f"expectancy={expected_expectancy:.3f}R "
                    f"({len(similar_trades)} matches, confidence={confidence:.2f})."
                )
        else:
            recommendation = "proceed"
            phase_note = f"{phase} phase"
            if phase == "similarity" and confidence < _MIN_SIMILARITY_CONFIDENCE:
                phase_note = f"similarity phase (low confidence: {confidence:.2f})"
            reasoning = (
                f"No filters triggered ({phase_note}). "
                "Proceed with standard signal rules."
            )

        return PreTradeInsight(
            similar_trades=similar_trades,
            expected_win_rate=round(expected_win_rate, 4),
            expected_avg_r=round(expected_avg_r, 4),
            expected_expectancy=round(expected_expectancy, 4),
            confidence=round(confidence, 4),
            confidence_adjustment=round(conf_adjustment, 4),
            statistical_filters=stat_filters,
            recommendation=recommendation,
            reasoning=reasoning,
        )
