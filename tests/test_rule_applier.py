"""Tests for converting SHAP rules into config overrides."""

import pytest


class TestRuleApplier:
    def test_strong_boost_rule_lowers_confluence(self):
        from src.discovery.rule_applier import apply_rules_to_config

        rules = [{
            "feature_a": "adx_value",
            "feature_b": "sess_london",
            "condition": "adx_value>=0.50 AND sess_london>=0.50",
            "quadrant_win_rate": 0.55,
            "baseline_win_rate": 0.37,
            "lift": 1.486,
            "recommendation": "strong_filter",
            "n_trades": 25,
        }]
        base_config = {"strategies": {"sss": {"min_confluence_score": 4}}}
        new_config, changes = apply_rules_to_config(rules, base_config, strategy="sss")

        assert len(changes) > 0

    def test_no_rules_returns_unchanged_config(self):
        from src.discovery.rule_applier import apply_rules_to_config

        base_config = {"strategies": {"sss": {"min_confluence_score": 4}}}
        new_config, changes = apply_rules_to_config([], base_config, strategy="sss")

        assert new_config == base_config
        assert len(changes) == 0

    def test_hypothesis_config_change_applied(self):
        from src.discovery.rule_applier import apply_hypothesis_to_config

        hypothesis = {
            "config_change": {"strategies": {"sss": {"min_confluence_score": 3}}},
        }
        base_config = {"strategies": {"sss": {"min_confluence_score": 4, "entry_mode": "cbc_only"}}}
        new_config = apply_hypothesis_to_config(hypothesis, base_config)

        assert new_config["strategies"]["sss"]["min_confluence_score"] == 3
        assert new_config["strategies"]["sss"]["entry_mode"] == "cbc_only"  # preserved
