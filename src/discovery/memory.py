"""Three-tier layered memory system for the discovery agent.

Short-term: current window trades + context (in-memory dict, capped).
Working memory: cross-window patterns that persist (JSON in reports/agent_knowledge/patterns/).
Long-term: validated edges absorbed into config/edges.yaml.
"""

from __future__ import annotations

import copy
import json
import logging
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class LayeredMemory:
    """Three-tier memory for the discovery orchestrator.

    Parameters
    ----------
    knowledge_dir:
        Base directory for working memory JSON files.
    edges_yaml_path:
        Path to config/edges.yaml for long-term absorption.
    short_term_max_windows:
        Maximum number of windows to retain in short-term memory.
    """

    def __init__(
        self,
        knowledge_dir: str = "reports/agent_knowledge",
        edges_yaml_path: str = "config/edges.yaml",
        short_term_max_windows: int = 3,
    ) -> None:
        self._knowledge_dir = Path(knowledge_dir)
        self._patterns_dir = self._knowledge_dir / "patterns"
        self._absorption_dir = self._knowledge_dir / "absorptions"
        self._edges_yaml_path = Path(edges_yaml_path)

        for d in (self._patterns_dir, self._absorption_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._short_term_max = short_term_max_windows
        self._short_term: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    # ------------------------------------------------------------------
    # Short-term memory (in-memory, current + recent windows)
    # ------------------------------------------------------------------

    def store_short_term(self, window_id: str, context: Dict[str, Any]) -> None:
        """Store a window's context in short-term memory.

        Evicts oldest entries when capacity is exceeded.
        """
        self._short_term[window_id] = context
        self._short_term.move_to_end(window_id)

        while len(self._short_term) > self._short_term_max:
            evicted_id, _ = self._short_term.popitem(last=False)
            logger.debug("Evicted short-term memory for window %s", evicted_id)

    def get_short_term(self, window_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a window's context from short-term memory."""
        return self._short_term.get(window_id)

    def get_recent_contexts(self) -> List[Dict[str, Any]]:
        """Return all short-term contexts, oldest first."""
        return list(self._short_term.values())

    # ------------------------------------------------------------------
    # Working memory (cross-window patterns, persisted as JSON)
    # ------------------------------------------------------------------

    def save_working_pattern(self, pattern: Dict[str, Any]) -> Path:
        """Save a cross-window pattern to working memory."""
        pat_id = pattern["id"]
        pattern.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        path = self._patterns_dir / f"{pat_id}.json"
        path.write_text(json.dumps(pattern, indent=2, default=str), encoding="utf-8")
        logger.info("Saved working pattern %s", pat_id)
        return path

    def load_working_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """Load a pattern from working memory by ID."""
        path = self._patterns_dir / f"{pattern_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list_working_patterns(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all patterns in working memory, optionally filtered by status."""
        results = []
        for p in sorted(self._patterns_dir.glob("*.json")):
            pat = json.loads(p.read_text(encoding="utf-8"))
            if status is None or pat.get("status") == status:
                results.append(pat)
        return results

    def promote_pattern(
        self,
        pattern_id: str,
        new_status: str,
        oos_results: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update a pattern's status and attach OOS validation results."""
        pat = self.load_working_pattern(pattern_id)
        if pat is None:
            raise ValueError(f"Pattern {pattern_id} not found in working memory")
        pat["status"] = new_status
        pat["updated_at"] = datetime.now(timezone.utc).isoformat()
        if oos_results:
            pat["oos_results"] = oos_results
        self.save_working_pattern(pat)
        logger.info("Promoted pattern %s to status=%s", pattern_id, new_status)

    # ------------------------------------------------------------------
    # Long-term memory (edges.yaml absorption)
    # ------------------------------------------------------------------

    def absorb_edge(
        self,
        edge_name: str,
        params: Dict[str, Any],
        source_pattern_id: str,
    ) -> None:
        """Absorb a validated edge into config/edges.yaml.

        Saves the previous state for potential revert, then enables the
        edge with the new params.
        """
        # Load current edges.yaml
        if self._edges_yaml_path.exists():
            edges = yaml.safe_load(
                self._edges_yaml_path.read_text(encoding="utf-8")
            ) or {}
        else:
            edges = {}

        # Save pre-absorption snapshot for revert
        previous_state = copy.deepcopy(edges.get(edge_name, {}))
        absorption_record = {
            "edge_name": edge_name,
            "source_pattern_id": source_pattern_id,
            "previous_state": previous_state,
            "new_params": params,
            "absorbed_at": datetime.now(timezone.utc).isoformat(),
        }
        absorption_path = self._absorption_dir / f"{edge_name}.json"
        absorption_path.write_text(
            json.dumps(absorption_record, indent=2, default=str), encoding="utf-8"
        )

        # Apply: enable the edge and merge params
        if edge_name not in edges:
            edges[edge_name] = {"enabled": True, "params": params}
        else:
            edges[edge_name]["enabled"] = True
            existing_params = edges[edge_name].get("params", {})
            existing_params.update(params)
            edges[edge_name]["params"] = existing_params

        self._edges_yaml_path.write_text(
            yaml.dump(edges, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
        logger.info(
            "Absorbed edge %s into %s (source: %s)",
            edge_name, self._edges_yaml_path, source_pattern_id,
        )

    def revert_absorption(self, edge_name: str) -> None:
        """Revert an absorbed edge to its pre-absorption state."""
        absorption_path = self._absorption_dir / f"{edge_name}.json"
        if not absorption_path.exists():
            raise ValueError(f"No absorption record for edge {edge_name}")

        record = json.loads(absorption_path.read_text(encoding="utf-8"))
        previous_state = record["previous_state"]

        # Reload edges.yaml and restore
        edges = yaml.safe_load(
            self._edges_yaml_path.read_text(encoding="utf-8")
        ) or {}
        edges[edge_name] = previous_state
        self._edges_yaml_path.write_text(
            yaml.dump(edges, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )

        absorption_path.unlink()
        logger.info("Reverted edge %s to pre-absorption state", edge_name)

    def get_absorption_log(self) -> List[Dict[str, Any]]:
        """Return all absorption records, sorted by time."""
        records = []
        for p in sorted(self._absorption_dir.glob("*.json")):
            records.append(json.loads(p.read_text(encoding="utf-8")))
        records.sort(key=lambda r: r.get("absorbed_at", ""))
        return records
