"""Apply dotted-path config changes to YAML files."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class ConfigApplicator:
    """Apply LLM-suggested config changes to strategy.yaml and edges.yaml."""

    def apply(self, changes: dict[str, object]) -> list[str]:
        """Apply dotted-path changes. Returns list of changes applied."""
        if not changes:
            return []

        strategy_path = _PROJECT_ROOT / "config" / "strategy.yaml"
        edges_path = _PROJECT_ROOT / "config" / "edges.yaml"

        applied = []

        # Load both files
        with strategy_path.open() as f:
            strategy = yaml.safe_load(f) or {}
        with edges_path.open() as f:
            edges = yaml.safe_load(f) or {}

        for dotted_key, value in changes.items():
            parts = dotted_key.split(".")

            # Route to the right file
            if parts[0] == "edges":
                target = edges
                parts = parts[1:]  # strip "edges." prefix
            else:
                target = strategy

            # Navigate to the parent, create intermediate dicts if needed
            node = target
            for part in parts[:-1]:
                if part not in node or not isinstance(node[part], dict):
                    node[part] = {}
                node = node[part]

            old_value = node.get(parts[-1])
            node[parts[-1]] = value
            applied.append(f"{dotted_key}: {old_value} -> {value}")
            logger.info("Config: %s = %s (was %s)", dotted_key, value, old_value)

        # Write back
        with strategy_path.open("w") as f:
            yaml.dump(strategy, f, default_flow_style=False, sort_keys=False)
        with edges_path.open("w") as f:
            yaml.dump(edges, f, default_flow_style=False, sort_keys=False)

        # Git commit
        if applied:
            try:
                subprocess.run(
                    ["git", "add", str(strategy_path), str(edges_path)],
                    cwd=str(_PROJECT_ROOT), capture_output=True,
                )
                msg = "feat(llm-meta): config changes\n\n" + "\n".join(applied)
                subprocess.run(
                    ["git", "commit", "-m", msg],
                    cwd=str(_PROJECT_ROOT), capture_output=True,
                )
            except Exception as exc:
                logger.warning("Git commit failed: %s", exc)

        return applied
