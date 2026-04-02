"""3-stage safety pipeline for generated EdgeFilter code.

Stage 1: AST whitelist validation -- reject dangerous imports/calls.
Stage 2: pytest execution -- generated tests must pass.
Stage 3: Mini-backtest validation -- filter must not degrade win rate.

Each stage short-circuits: if a stage fails, subsequent stages are skipped.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.discovery.codegen.ast_validator import validate_ast, ASTValidationResult
from src.discovery.codegen.test_executor import run_generated_tests, TestResult
from src.discovery.codegen.backtest_validator import (
    validate_with_backtest,
    BacktestValidationResult,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete result of the 3-stage safety pipeline."""
    passed: bool = False
    failed_stage: Optional[str] = None

    # Stage 1: AST validation
    ast_passed: bool = False
    ast_result: Optional[ASTValidationResult] = None

    # Stage 2: pytest execution
    pytest_passed: bool = False
    pytest_result: Optional[TestResult] = None

    # Stage 3: backtest validation
    backtest_passed: bool = False
    backtest_result: Optional[BacktestValidationResult] = None

    # Metadata
    filter_name: str = ""
    filter_class_name: str = ""
    filter_file: str = ""
    test_file: str = ""


class SafetyPipeline:
    """Orchestrate 3-stage validation for generated EdgeFilter code.

    Usage:
        pipeline = SafetyPipeline()
        result = pipeline.run(
            filter_code=generated_filter_source,
            test_code=generated_test_source,
            filter_name="high_adx_london",
            filter_class_name="HighADXLondonFilter",
            category="entry",
        )
        if result.passed:
            # Safe to register and use the filter
            ...
    """

    def __init__(
        self,
        output_dir: str = "src/edges/generated",
        test_timeout: int = 60,
        min_tests: int = 3,
        max_win_rate_degradation: float = 0.05,
        mini_backtest_bars: int = 1000,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._test_timeout = test_timeout
        self._min_tests = min_tests
        self._max_wr_degradation = max_win_rate_degradation
        self._mini_bt_bars = mini_backtest_bars

    def run(
        self,
        filter_code: str,
        test_code: str,
        filter_name: str,
        filter_class_name: str,
        category: str = "entry",
        skip_backtest: bool = False,
    ) -> PipelineResult:
        """Execute the full 3-stage safety pipeline.

        Parameters
        ----------
        filter_code:
            Complete Python source for the EdgeFilter subclass.
        test_code:
            Complete Python source for the test class.
        filter_name:
            Registered name (e.g., 'high_adx_london').
        filter_class_name:
            Class name (e.g., 'HighADXLondonFilter').
        category:
            Edge category: 'entry', 'exit', or 'modifier'.
        skip_backtest:
            If True, skip stage 3 (useful for unit testing the pipeline).

        Returns
        -------
        PipelineResult with per-stage pass/fail and details.
        """
        result = PipelineResult(
            filter_name=filter_name,
            filter_class_name=filter_class_name,
        )

        # ---- Stage 1: AST Whitelist Validation ----
        logger.info("[Stage 1/3] AST validation for %s", filter_name)
        ast_result = validate_ast(filter_code)
        result.ast_result = ast_result
        result.ast_passed = ast_result.is_valid

        if not ast_result.is_valid:
            result.failed_stage = "ast_validation"
            logger.warning(
                "AST validation FAILED for %s: %s",
                filter_name, "; ".join(ast_result.violations),
            )
            return result

        logger.info("[Stage 1/3] AST validation PASSED for %s", filter_name)

        # ---- Write files to disk for pytest and backtest ----
        # Ensure output directory exists
        self._output_dir.mkdir(parents=True, exist_ok=True)

        filter_file = self._output_dir / f"{filter_name}.py"
        filter_file.write_text(filter_code, encoding="utf-8")
        result.filter_file = str(filter_file)

        # Ensure __init__.py exists in generated dir
        init_file = self._output_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text(
                '"""Auto-generated EdgeFilter modules."""\n',
                encoding="utf-8",
            )

        # Write test file to a temp directory
        test_dir = Path(tempfile.mkdtemp(prefix="edge_test_"))
        test_file = test_dir / f"test_generated_{filter_name}.py"
        test_file.write_text(test_code, encoding="utf-8")
        result.test_file = str(test_file)

        # ---- Stage 2: Pytest Execution ----
        logger.info("[Stage 2/3] Pytest execution for %s", filter_name)
        pytest_result = run_generated_tests(
            str(test_file),
            timeout=self._test_timeout,
            min_tests=self._min_tests,
        )
        result.pytest_result = pytest_result
        result.pytest_passed = pytest_result.passed

        if not pytest_result.passed:
            result.failed_stage = "pytest_execution"
            logger.warning(
                "Pytest FAILED for %s: %s (passed=%d, failed=%d, errors=%d)",
                filter_name, pytest_result.error,
                pytest_result.num_passed, pytest_result.num_failed, pytest_result.num_errors,
            )
            # Clean up the filter file since it failed validation
            filter_file.unlink(missing_ok=True)
            return result

        logger.info(
            "[Stage 2/3] Pytest PASSED for %s (%d/%d tests)",
            filter_name, pytest_result.num_passed, pytest_result.num_collected,
        )

        # ---- Stage 3: Mini-Backtest Validation ----
        if skip_backtest:
            logger.info("[Stage 3/3] Backtest SKIPPED for %s", filter_name)
            result.backtest_passed = True
            result.passed = True
            return result

        logger.info("[Stage 3/3] Mini-backtest validation for %s", filter_name)
        bt_result = validate_with_backtest(
            filter_code=filter_code,
            filter_name=filter_name,
            filter_class_name=filter_class_name,
            category=category,
            max_win_rate_degradation=self._max_wr_degradation,
            n_bars=self._mini_bt_bars,
        )
        result.backtest_result = bt_result
        result.backtest_passed = bt_result.passed

        if not bt_result.passed:
            result.failed_stage = "backtest_validation"
            logger.warning(
                "Backtest validation FAILED for %s: %s", filter_name, bt_result.reason,
            )
            # Clean up the filter file since it failed validation
            filter_file.unlink(missing_ok=True)
            return result

        logger.info("[Stage 3/3] Backtest validation PASSED for %s: %s", filter_name, bt_result.reason)
        result.passed = True
        return result
