"""Discovery engine orchestrator.

Coordinates the full loop: accumulate trades across windows, run
XGBoost/SHAP analysis at intervals, generate hypotheses via Claude,
and persist everything to the knowledge base.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.discovery.knowledge_base import KnowledgeBase
from src.discovery.xgb_analyzer import (
    SHAPInsight,
    build_training_data,
    run_shap_analysis,
    train_classifier,
)

logger = logging.getLogger(__name__)


class DiscoveryRunner:
    """Orchestrates the discovery engine across optimization windows.

    Parameters
    ----------
    shap_every_n_windows:
        Run full SHAP analysis every N windows (default 3 = ~90 days).
    min_trades_for_shap:
        Minimum accumulated trades before SHAP runs (default 30).
    knowledge_dir:
        Path for the JSON knowledge base.
    """

    def __init__(
        self,
        shap_every_n_windows: int = 3,
        min_trades_for_shap: int = 30,
        knowledge_dir: str = "reports/agent_knowledge",
    ) -> None:
        self._shap_interval = shap_every_n_windows
        self._min_trades = min_trades_for_shap
        self._kb = KnowledgeBase(base_dir=knowledge_dir)

    def should_run_shap(self, window_index: int) -> bool:
        """Check if SHAP analysis should run for this window."""
        return window_index > 0 and (window_index + 1) % self._shap_interval == 0

    def accumulate_window(self, window_id: str, trades: List[Dict[str, Any]]) -> None:
        """Save a window's trades to the knowledge base for accumulation."""
        self._kb.save_window_trades(trades, window_id=window_id)
        logger.info("Accumulated %d trades for window %s", len(trades), window_id)

    def get_accumulated_trades(self) -> List[Dict[str, Any]]:
        """Get all accumulated trades across windows."""
        return self._kb.get_accumulated_trades()

    def analyze(self, strategy_name: str = "sss") -> Optional[SHAPInsight]:
        """Run XGBoost/SHAP analysis on accumulated trades.

        Returns SHAPInsight if enough trades, None otherwise.
        """
        all_trades = self.get_accumulated_trades()
        if len(all_trades) < self._min_trades:
            logger.info(
                "Only %d accumulated trades (need %d) -- skipping SHAP",
                len(all_trades), self._min_trades,
            )
            return None

        logger.info("Running SHAP analysis on %d accumulated trades", len(all_trades))
        X, y_binary, y_r = build_training_data(all_trades)

        if len(X) < self._min_trades:
            return None

        model = train_classifier(X, y_binary, y_r)
        insight = run_shap_analysis(model, X, y_binary, y_r)

        logger.info(
            "SHAP complete: %d features, %d interactions, %d rules",
            len(insight.feature_importance),
            len(insight.top_interactions),
            len(insight.actionable_rules),
        )
        return insight

    def run_full_cycle(
        self,
        window_id: str,
        window_index: int,
        trades: List[Dict[str, Any]],
        strategy_name: str,
        base_config: Dict[str, Any],
        enable_claude: bool = True,
    ) -> Dict[str, Any]:
        """Run one full discovery cycle for a window.

        1. Accumulate this window's trades
        2. If SHAP interval reached, run analysis
        3. If analysis produced rules, optionally generate hypotheses
        4. Return results dict

        Returns
        -------
        Dict with keys: shap_ran, insight, hypotheses, rules_applied
        """
        self.accumulate_window(window_id, trades)

        result: Dict[str, Any] = {
            "window_id": window_id,
            "window_index": window_index,
            "shap_ran": False,
            "insight": None,
            "hypotheses": [],
            "changes": [],
        }

        if not self.should_run_shap(window_index):
            return result

        insight = self.analyze(strategy_name=strategy_name)
        if insight is None:
            return result

        result["shap_ran"] = True
        result["insight"] = insight

        # Save SHAP rules to knowledge base
        self._kb.save_shap_rules(insight.actionable_rules, window_id=window_id)

        # Generate hypotheses via Claude (if enabled and rules found)
        if enable_claude and insight.actionable_rules:
            try:
                from src.discovery.hypothesis_engine import generate_hypotheses
                hypotheses = generate_hypotheses(
                    insight, strategy_name, window_id,
                )
                for hyp in hypotheses:
                    self._kb.save_hypothesis(hyp)
                result["hypotheses"] = hypotheses
            except Exception as exc:
                logger.warning("Hypothesis generation failed: %s", exc)

        # Apply strong rules as config changes
        from src.discovery.rule_applier import apply_rules_to_config
        _, changes = apply_rules_to_config(
            insight.actionable_rules, base_config, strategy=strategy_name,
        )
        result["changes"] = changes

        return result
