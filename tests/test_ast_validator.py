"""Tests for AST whitelist validation of generated EdgeFilter code."""

import pytest


class TestASTValidator:
    def test_clean_edge_filter_passes(self):
        from src.discovery.codegen.ast_validator import validate_ast

        code = '''
import numpy as np
import pandas as pd
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
        return EdgeResult(allowed=False, edge_name=self.name, reason="ADX too low")
'''
        result = validate_ast(code)
        assert result.is_valid is True
        assert len(result.violations) == 0

    def test_rejects_os_import(self):
        from src.discovery.codegen.ast_validator import validate_ast

        code = '''
import os
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult

class BadFilter(EdgeFilter):
    def should_allow(self, context):
        os.system("rm -rf /")
        return EdgeResult(allowed=True, edge_name="bad", reason="bad")
'''
        result = validate_ast(code)
        assert result.is_valid is False
        assert any("os" in v for v in result.violations)

    def test_rejects_subprocess_import(self):
        from src.discovery.codegen.ast_validator import validate_ast

        code = '''
import subprocess
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult

class BadFilter(EdgeFilter):
    def should_allow(self, context):
        subprocess.run(["ls"])
        return EdgeResult(allowed=True, edge_name="bad", reason="bad")
'''
        result = validate_ast(code)
        assert result.is_valid is False
        assert any("subprocess" in v for v in result.violations)

    def test_rejects_eval_call(self):
        from src.discovery.codegen.ast_validator import validate_ast

        code = '''
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult

class BadFilter(EdgeFilter):
    def should_allow(self, context):
        result = eval("1+1")
        return EdgeResult(allowed=True, edge_name="bad", reason="bad")
'''
        result = validate_ast(code)
        assert result.is_valid is False
        assert any("eval" in v for v in result.violations)

    def test_rejects_exec_call(self):
        from src.discovery.codegen.ast_validator import validate_ast

        code = '''
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult

class BadFilter(EdgeFilter):
    def should_allow(self, context):
        exec("import os")
        return EdgeResult(allowed=True, edge_name="bad", reason="bad")
'''
        result = validate_ast(code)
        assert result.is_valid is False
        assert any("exec" in v for v in result.violations)

    def test_rejects_dunder_import(self):
        from src.discovery.codegen.ast_validator import validate_ast

        code = '''
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult

class BadFilter(EdgeFilter):
    def should_allow(self, context):
        mod = __import__("os")
        return EdgeResult(allowed=True, edge_name="bad", reason="bad")
'''
        result = validate_ast(code)
        assert result.is_valid is False
        assert any("__import__" in v for v in result.violations)

    def test_rejects_open_call(self):
        from src.discovery.codegen.ast_validator import validate_ast

        code = '''
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult

class BadFilter(EdgeFilter):
    def should_allow(self, context):
        f = open("/etc/passwd", "r")
        return EdgeResult(allowed=True, edge_name="bad", reason="bad")
'''
        result = validate_ast(code)
        assert result.is_valid is False
        assert any("open" in v for v in result.violations)

    def test_rejects_network_imports(self):
        from src.discovery.codegen.ast_validator import validate_ast

        code = '''
import socket
import urllib.request
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult

class BadFilter(EdgeFilter):
    def should_allow(self, context):
        return EdgeResult(allowed=True, edge_name="bad", reason="bad")
'''
        result = validate_ast(code)
        assert result.is_valid is False
        assert any("socket" in v for v in result.violations)

    def test_allows_indicator_imports(self):
        from src.discovery.codegen.ast_validator import validate_ast

        code = '''
import numpy as np
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult
from src.indicators.ichimoku import IchimokuCalculator

class IchiFilter(EdgeFilter):
    def __init__(self, config: dict) -> None:
        super().__init__("ichi_filter", config)

    def should_allow(self, context: EdgeContext) -> EdgeResult:
        if not self.enabled:
            return self._disabled_result()
        return EdgeResult(allowed=True, edge_name=self.name, reason="ok")
'''
        result = validate_ast(code)
        assert result.is_valid is True

    def test_requires_edge_filter_subclass(self):
        from src.discovery.codegen.ast_validator import validate_ast

        code = '''
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult

def my_function():
    return 42
'''
        result = validate_ast(code)
        assert result.is_valid is False
        assert any("EdgeFilter subclass" in v for v in result.violations)

    def test_requires_should_allow_method(self):
        from src.discovery.codegen.ast_validator import validate_ast

        code = '''
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult

class IncompleteFilter(EdgeFilter):
    def __init__(self, config: dict) -> None:
        super().__init__("incomplete", config)
'''
        result = validate_ast(code)
        assert result.is_valid is False
        assert any("should_allow" in v for v in result.violations)

    def test_syntax_error_fails(self):
        from src.discovery.codegen.ast_validator import validate_ast

        code = '''
def broken(
    return True
'''
        result = validate_ast(code)
        assert result.is_valid is False
        assert any("syntax" in v.lower() or "parse" in v.lower() for v in result.violations)
