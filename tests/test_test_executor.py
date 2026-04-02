"""Tests for subprocess pytest executor for generated code."""

import tempfile
from pathlib import Path

import pytest


class TestTestExecutor:
    def test_passing_test_returns_success(self, tmp_path):
        from src.discovery.codegen.test_executor import run_generated_tests

        # Write a minimal passing test file
        test_file = tmp_path / "test_passing.py"
        test_file.write_text('''
def test_one():
    assert 1 + 1 == 2

def test_two():
    assert True

def test_three():
    assert "hello".startswith("h")
''')

        result = run_generated_tests(str(test_file))
        assert result.passed is True
        assert result.num_passed >= 3
        assert result.num_failed == 0

    def test_failing_test_returns_failure(self, tmp_path):
        from src.discovery.codegen.test_executor import run_generated_tests

        test_file = tmp_path / "test_failing.py"
        test_file.write_text('''
def test_one():
    assert 1 + 1 == 2

def test_two():
    assert False, "intentional failure"

def test_three():
    assert True
''')

        result = run_generated_tests(str(test_file))
        assert result.passed is False
        assert result.num_failed >= 1

    def test_import_error_returns_failure(self, tmp_path):
        from src.discovery.codegen.test_executor import run_generated_tests

        test_file = tmp_path / "test_import_error.py"
        test_file.write_text('''
from nonexistent_module import something

def test_one():
    assert True
''')

        result = run_generated_tests(str(test_file))
        assert result.passed is False

    def test_timeout_returns_failure(self, tmp_path):
        from src.discovery.codegen.test_executor import run_generated_tests

        test_file = tmp_path / "test_timeout.py"
        test_file.write_text('''
import time

def test_slow():
    time.sleep(60)
''')

        result = run_generated_tests(str(test_file), timeout=2)
        assert result.passed is False
        assert "timeout" in result.error.lower()

    def test_requires_minimum_test_count(self, tmp_path):
        from src.discovery.codegen.test_executor import run_generated_tests

        test_file = tmp_path / "test_few.py"
        test_file.write_text('''
def test_one():
    assert True
''')

        result = run_generated_tests(str(test_file), min_tests=3)
        assert result.passed is False
        assert "minimum" in result.error.lower() or "3" in result.error
