"""Converts SHAP rules and hypotheses into config overrides.

Takes actionable rules from the SHAP analysis or hypotheses from Claude
and produces modified config dicts for the next backtest window.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def apply_rules_to_config(
    rules: List[Dict[str, Any]],
    base_config: Dict[str, Any],
    strategy: str,
    max_changes: int = 2,
) -> Tuple[Dict[str, Any], List[str]]:
    """Apply SHAP-derived rules as config modifications.

    Only strong_filter rules are applied automatically. Each rule
    generates a config change based on the feature interaction pattern.

    Parameters
    ----------
    rules: Actionable rules from SHAPInsight.
    base_config: Current strategy config dict.
    strategy: Strategy name (e.g., "sss").
    max_changes: Maximum config changes to apply per iteration.

    Returns
    -------
    (new_config, changes): Modified config and list of change descriptions.
    """
    new_config = copy.deepcopy(base_config)
    changes: List[str] = []

    strong_rules = [r for r in rules if r.get("recommendation") == "strong_filter"]
    if not strong_rules:
        return new_config, changes

    strat_cfg = new_config.get("strategies", {}).get(strategy, {})

    for rule in strong_rules[:max_changes]:
        change = _rule_to_config_change(rule, strat_cfg, strategy)
        if change:
            param, old_val, new_val, desc = change
            strat_cfg[param] = new_val
            changes.append(desc)
            logger.info("Applied rule: %s", desc)

    new_config.setdefault("strategies", {})[strategy] = strat_cfg
    return new_config, changes


def _rule_to_config_change(
    rule: Dict[str, Any],
    strat_cfg: Dict[str, Any],
    strategy: str,
) -> Tuple[str, Any, Any, str] | None:
    """Map a SHAP rule to a concrete config parameter change.

    Returns (param_name, old_value, new_value, description) or None.
    """
    feat_a = rule.get("feature_a", "")
    feat_b = rule.get("feature_b", "")
    lift = rule.get("lift", 1.0)

    # High-lift rules where confluence is involved -> adjust min_confluence_score
    if "confluence" in feat_a or "confluence" in feat_b:
        old = strat_cfg.get("min_confluence_score", 4)
        if lift > 1.2:
            new = max(1, old - 1)  # lower threshold since these conditions boost WR
        else:
            new = min(6, old + 1)  # raise threshold to filter low-quality
        if new != old:
            return (
                "min_confluence_score", old, new,
                f"Adjusted min_confluence_score {old}->{new} "
                f"(SHAP: {feat_a} x {feat_b}, lift={lift:.2f})"
            )

    # ADX interaction -> adjust swing parameters
    if "adx" in feat_a or "adx" in feat_b:
        old = strat_cfg.get("min_swing_pips", 1.0)
        if lift > 1.3:
            new = max(0.5, old - 0.5)
        else:
            new = min(5.0, old + 0.5)
        if new != old:
            return (
                "min_swing_pips", old, new,
                f"Adjusted min_swing_pips {old}->{new} "
                f"(SHAP: {feat_a} x {feat_b}, lift={lift:.2f})"
            )

    return None


def apply_hypothesis_to_config(
    hypothesis: Dict[str, Any],
    base_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply a hypothesis's config_change on top of base_config.

    Deep-merges at the strategy level so unmodified params are preserved.
    """
    new_config = copy.deepcopy(base_config)
    config_change = hypothesis.get("config_change", {})

    for key, val in config_change.items():
        if key == "strategies" and isinstance(val, dict):
            for strat_name, strat_params in val.items():
                if strat_name in new_config.get("strategies", {}) and isinstance(strat_params, dict):
                    new_config["strategies"][strat_name] = {
                        **new_config["strategies"][strat_name],
                        **strat_params,
                    }
                else:
                    new_config.setdefault("strategies", {})[strat_name] = strat_params
        else:
            new_config[key] = val

    return new_config
