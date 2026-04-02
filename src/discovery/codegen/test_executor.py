"""Subprocess pytest executor for generated EdgeFilter test code.

Runs pytest in a subprocess with a timeout to validate that generated
test classes pass. Captures stdout/stderr and parses results.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of running pytest on generated tests."""
    passed: bool
    num_passed: int = 0
    num_failed: int = 0
    num_errors: int = 0
    num_collected: int = 0
    stdout: str = ""
    stderr: str = ""
    error: str = ""


def run_generated_tests(
    test_file: str,
    timeout: int = 60,
    min_tests: int = 3,
    extra_args: list[str] | None = None,
) -> TestResult:
    """Run pytest on a generated test file in a subprocess.

    Parameters
    ----------
    test_file:
        Absolute path to the test file.
    timeout:
        Subprocess timeout in seconds.
    min_tests:
        Minimum number of tests that must be collected.
    extra_args:
        Additional pytest arguments.

    Returns
    -------
    TestResult with pass/fail status and details.
    """
    test_path = Path(test_file)
    if not test_path.exists():
        return TestResult(
            passed=False,
            error=f"Test file does not exist: {test_file}",
        )

    cmd = [
        sys.executable, "-m", "pytest",
        str(test_path),
        "-v",
        "--tb=short",
        "--no-header",
        "-q",
    ]
    if extra_args:
        cmd.extend(extra_args)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path.cwd()),
        )
    except subprocess.TimeoutExpired:
        return TestResult(
            passed=False,
            error=f"Timeout: tests did not complete within {timeout}s",
        )
    except Exception as exc:
        return TestResult(
            passed=False,
            error=f"Failed to run pytest: {exc}",
        )

    stdout = proc.stdout
    stderr = proc.stderr

    # Parse pytest output for counts
    num_passed = 0
    num_failed = 0
    num_errors = 0
    num_collected = 0

    # Match "X passed", "X failed", "X error" patterns
    passed_match = re.search(r"(\d+)\s+passed", stdout)
    failed_match = re.search(r"(\d+)\s+failed", stdout)
    error_match = re.search(r"(\d+)\s+error", stdout)

    if passed_match:
        num_passed = int(passed_match.group(1))
    if failed_match:
        num_failed = int(failed_match.group(1))
    if error_match:
        num_errors = int(error_match.group(1))

    num_collected = num_passed + num_failed + num_errors

    # Check minimum test count
    if num_collected < min_tests:
        return TestResult(
            passed=False,
            num_passed=num_passed,
            num_failed=num_failed,
            num_errors=num_errors,
            num_collected=num_collected,
            stdout=stdout,
            stderr=stderr,
            error=(
                f"Only {num_collected} tests collected, minimum {min_tests} required. "
                f"Generated tests must include at least {min_tests} test methods."
            ),
        )

    all_passed = num_failed == 0 and num_errors == 0 and proc.returncode == 0

    return TestResult(
        passed=all_passed,
        num_passed=num_passed,
        num_failed=num_failed,
        num_errors=num_errors,
        num_collected=num_collected,
        stdout=stdout,
        stderr=stderr,
        error="" if all_passed else f"{num_failed} failed, {num_errors} errors",
    )
