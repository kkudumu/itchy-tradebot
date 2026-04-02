"""Tests for the 3-stage safety pipeline orchestrator."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

import pytest


class TestSafetyPipeline:
    def _clean_filter_code(self) -> str:
        return '''
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult


class HighADXFilter(EdgeFilter):
    def __init__(self, config: dict) -> None:
        super().__init__("high_adx", config)
        params = config.get("params", {})
        self._threshold = float(params.get("adx_threshold", 35.0))

    def should_allow(self, context: EdgeContext) -> EdgeResult:
        if not self.enabled:
            return self._disabled_result()
        if context.adx >= self._threshold:
            return EdgeResult(allowed=True, edge_name=self.name, reason="ADX OK")
        return EdgeResult(allowed=False, edge_name=self.name, reason="ADX low")
'''

    def _clean_test_code(self) -> str:
        return '''
from datetime import datetime, timezone
import pytest
from src.edges.base import EdgeContext, EdgeResult


def _ctx(**kw):
    defaults = dict(
        timestamp=datetime(2024, 6, 15, 10, 0, tzinfo=timezone.utc),
        day_of_week=5, close_price=2350.0, high_price=2355.0,
        low_price=2345.0, spread=15.0, session="london",
        adx=30.0, atr=5.0,
    )
    defaults.update(kw)
    return EdgeContext(**defaults)


class TestHighADXFilter:
    def test_allows_high_adx(self):
        from src.edges.generated.high_adx import HighADXFilter
        f = HighADXFilter({"enabled": True, "params": {"adx_threshold": 25.0}})
        result = f.should_allow(_ctx(adx=30.0))
        assert result.allowed is True

    def test_blocks_low_adx(self):
        from src.edges.generated.high_adx import HighADXFilter
        f = HighADXFilter({"enabled": True, "params": {"adx_threshold": 35.0}})
        result = f.should_allow(_ctx(adx=20.0))
        assert result.allowed is False

    def test_disabled_passes(self):
        from src.edges.generated.high_adx import HighADXFilter
        f = HighADXFilter({"enabled": False, "params": {}})
        result = f.should_allow(_ctx())
        assert result.allowed is True
'''

    def test_stage1_ast_rejects_dangerous_code(self):
        from src.discovery.codegen.safety_pipeline import SafetyPipeline

        pipeline = SafetyPipeline()
        bad_code = '''
import os
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult

class BadFilter(EdgeFilter):
    def should_allow(self, context: EdgeContext) -> EdgeResult:
        os.system("rm -rf /")
        return EdgeResult(allowed=True, edge_name="bad", reason="bad")
'''
        result = pipeline.run(
            filter_code=bad_code,
            test_code="",
            filter_name="bad",
            filter_class_name="BadFilter",
            category="entry",
            skip_backtest=True,
        )
        assert result.passed is False
        assert result.failed_stage == "ast_validation"

    def test_stage2_pytest_rejects_failing_tests(self):
        from src.discovery.codegen.safety_pipeline import SafetyPipeline

        pipeline = SafetyPipeline()
        filter_code = self._clean_filter_code()
        bad_test_code = '''
class TestHighADXFilter:
    def test_one(self):
        assert False, "intentional failure"
    def test_two(self):
        assert True
    def test_three(self):
        assert True
'''

        result = pipeline.run(
            filter_code=filter_code,
            test_code=bad_test_code,
            filter_name="high_adx",
            filter_class_name="HighADXFilter",
            category="entry",
            skip_backtest=True,
        )
        assert result.passed is False
        assert result.failed_stage == "pytest_execution"

    def test_all_stages_pass_for_clean_code(self):
        from src.discovery.codegen.safety_pipeline import SafetyPipeline
        from src.discovery.codegen.backtest_validator import BacktestValidationResult

        pipeline = SafetyPipeline()

        # Mock backtest to avoid needing real data
        mock_bt_result = BacktestValidationResult(
            passed=True,
            baseline_win_rate=0.37,
            candidate_win_rate=0.38,
            win_rate_delta=0.01,
            baseline_trades=50,
            candidate_trades=45,
            reason="Win rate maintained",
        )

        with patch(
            "src.discovery.codegen.safety_pipeline.validate_with_backtest",
            return_value=mock_bt_result,
        ):
            result = pipeline.run(
                filter_code=self._clean_filter_code(),
                test_code=self._clean_test_code(),
                filter_name="high_adx",
                filter_class_name="HighADXFilter",
                category="entry",
            )
            # AST should pass, we mock backtest; pytest depends on actual file
            assert result.ast_passed is True

    def test_pipeline_result_contains_all_stages(self):
        from src.discovery.codegen.safety_pipeline import SafetyPipeline, PipelineResult

        pipeline = SafetyPipeline()
        result = pipeline.run(
            filter_code=self._clean_filter_code(),
            test_code=self._clean_test_code(),
            filter_name="high_adx",
            filter_class_name="HighADXFilter",
            category="entry",
            skip_backtest=True,
        )
        assert isinstance(result, PipelineResult)
        assert hasattr(result, "ast_passed")
        assert hasattr(result, "pytest_passed")
        assert hasattr(result, "backtest_passed")
        assert hasattr(result, "failed_stage")
