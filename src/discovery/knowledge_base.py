"""JSON-based knowledge base for the discovery agent.

Persists hypotheses, SHAP rules, and accumulated trade data to
reports/agent_knowledge/ as inspectable, git-trackable JSON files.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """File-based knowledge store for discovery agent learnings.

    Directory layout:
        base_dir/
            hypotheses/
                hyp_001.json
                hyp_002.json
            shap_rules/
                w_001.json
            window_trades/
                w_001.json
            index.json
    """

    def __init__(self, base_dir: str = "reports/agent_knowledge") -> None:
        self._base = Path(base_dir)
        self._hyp_dir = self._base / "hypotheses"
        self._rules_dir = self._base / "shap_rules"
        self._trades_dir = self._base / "window_trades"

        for d in (self._hyp_dir, self._rules_dir, self._trades_dir):
            d.mkdir(parents=True, exist_ok=True)

    # -- Hypotheses --

    def save_hypothesis(self, hypothesis: Dict[str, Any]) -> Path:
        hyp_id = hypothesis["id"]
        hypothesis.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        path = self._hyp_dir / f"{hyp_id}.json"
        path.write_text(json.dumps(hypothesis, indent=2, default=str), encoding="utf-8")
        return path

    def load_hypothesis(self, hyp_id: str) -> Dict[str, Any]:
        path = self._hyp_dir / f"{hyp_id}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def update_hypothesis_status(
        self, hyp_id: str, status: str, metrics: Optional[Dict] = None
    ) -> None:
        hyp = self.load_hypothesis(hyp_id)
        hyp["status"] = status
        hyp["updated_at"] = datetime.now(timezone.utc).isoformat()
        if metrics:
            hyp["validation_metrics"] = metrics
        self.save_hypothesis(hyp)

    def list_hypotheses(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        results = []
        for p in sorted(self._hyp_dir.glob("*.json")):
            hyp = json.loads(p.read_text(encoding="utf-8"))
            if status is None or hyp.get("status") == status:
                results.append(hyp)
        return results

    # -- SHAP Rules --

    def save_shap_rules(self, rules: List[Dict[str, Any]], window_id: str) -> Path:
        path = self._rules_dir / f"{window_id}.json"
        path.write_text(json.dumps(rules, indent=2, default=str), encoding="utf-8")
        return path

    def load_shap_rules(self, window_id: str) -> List[Dict[str, Any]]:
        path = self._rules_dir / f"{window_id}.json"
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    # -- Window Trades (for accumulation) --

    def save_window_trades(self, trades: List[Dict[str, Any]], window_id: str) -> Path:
        path = self._trades_dir / f"{window_id}.json"
        path.write_text(json.dumps(trades, indent=2, default=str), encoding="utf-8")
        return path

    def get_accumulated_trades(self) -> List[Dict[str, Any]]:
        all_trades: List[Dict[str, Any]] = []
        for p in sorted(self._trades_dir.glob("*.json")):
            trades = json.loads(p.read_text(encoding="utf-8"))
            all_trades.extend(trades)
        return all_trades
