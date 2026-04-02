"""Tests for LLM code generation with Claude/Codex fallback."""

from unittest.mock import patch, MagicMock
import subprocess

import pytest


class TestParseGeneratedCode:
    def test_extracts_filter_and_test_blocks(self):
        from src.discovery.codegen.llm_generator import parse_generated_code

        response = '''Here's the implementation:

```python
# FILE: src/edges/generated/high_adx_london.py
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult

class HighADXLondonFilter(EdgeFilter):
    def __init__(self, config: dict) -> None:
        super().__init__("high_adx_london", config)
        params = config.get("params", {})
        self._threshold = float(params.get("adx_threshold", 35.0))

    def should_allow(self, context: EdgeContext) -> EdgeResult:
        if not self.enabled:
            return self._disabled_result()
        if context.adx >= self._threshold:
            return EdgeResult(allowed=True, edge_name=self.name, reason="ADX OK")
        return EdgeResult(allowed=False, edge_name=self.name, reason="ADX too low")
```

```python
# FILE: tests/test_generated_high_adx_london.py
import pytest
from src.edges.base import EdgeContext, EdgeResult

class TestHighADXLondonFilter:
    def test_allows_when_condition_met(self):
        pass
    def test_blocks_when_condition_violated(self):
        pass
    def test_disabled_passes_through(self):
        pass
```
'''
        result = parse_generated_code(response)

        assert result.filter_code is not None
        assert "HighADXLondonFilter" in result.filter_code
        assert result.test_code is not None
        assert "TestHighADXLondonFilter" in result.test_code

    def test_handles_missing_test_block(self):
        from src.discovery.codegen.llm_generator import parse_generated_code

        response = '''
```python
# FILE: src/edges/generated/simple.py
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult

class SimpleFilter(EdgeFilter):
    def __init__(self, config: dict) -> None:
        super().__init__("simple", config)
    def should_allow(self, context: EdgeContext) -> EdgeResult:
        return EdgeResult(allowed=True, edge_name=self.name, reason="ok")
```
'''
        result = parse_generated_code(response)
        assert result.filter_code is not None
        assert result.test_code is None

    def test_handles_empty_response(self):
        from src.discovery.codegen.llm_generator import parse_generated_code

        result = parse_generated_code("")
        assert result.filter_code is None
        assert result.test_code is None


class TestGenerateEdgeFilter:
    def test_calls_claude_cli_first(self):
        from src.discovery.codegen.llm_generator import generate_edge_filter

        mock_result = MagicMock()
        mock_result.stdout = '''```python
# FILE: src/edges/generated/test_filter.py
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult

class TestFilter(EdgeFilter):
    def __init__(self, config: dict) -> None:
        super().__init__("test_filter", config)
    def should_allow(self, context: EdgeContext) -> EdgeResult:
        if not self.enabled:
            return self._disabled_result()
        return EdgeResult(allowed=True, edge_name=self.name, reason="ok")
```

```python
# FILE: tests/test_generated_test_filter.py
class TestTestFilter:
    def test_allows_when_condition_met(self):
        pass
    def test_blocks_when_condition_violated(self):
        pass
    def test_disabled_passes_through(self):
        pass
```'''
        mock_result.returncode = 0

        hypothesis = {
            "description": "Test hypothesis",
            "evidence": {},
            "filter_spec": {
                "name": "test_filter",
                "class_name": "TestFilter",
                "category": "entry",
                "description": "Test filter",
                "params": {},
                "logic_description": "Always allow",
            },
        }

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = generate_edge_filter(hypothesis)
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args[0][0][0] == "claude"
            assert result.filter_code is not None

    def test_falls_back_to_codex_on_claude_failure(self):
        from src.discovery.codegen.llm_generator import generate_edge_filter

        # Claude fails
        claude_exc = FileNotFoundError("claude not found")
        # Codex succeeds
        codex_result = MagicMock()
        codex_result.stdout = '''```python
# FILE: src/edges/generated/test_filter.py
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult

class TestFilter(EdgeFilter):
    def __init__(self, config: dict) -> None:
        super().__init__("test_filter", config)
    def should_allow(self, context: EdgeContext) -> EdgeResult:
        if not self.enabled:
            return self._disabled_result()
        return EdgeResult(allowed=True, edge_name=self.name, reason="ok")
```

```python
# FILE: tests/test_generated_test_filter.py
class TestTestFilter:
    def test_allows(self): pass
    def test_blocks(self): pass
    def test_disabled(self): pass
```'''
        codex_result.returncode = 0

        hypothesis = {
            "description": "Test",
            "evidence": {},
            "filter_spec": {
                "name": "test_filter",
                "class_name": "TestFilter",
                "category": "entry",
                "description": "Test",
                "params": {},
                "logic_description": "Test",
            },
        }

        with patch("subprocess.run", side_effect=[claude_exc, codex_result]) as mock_run:
            result = generate_edge_filter(hypothesis)
            assert mock_run.call_count == 2
            assert result.filter_code is not None

    def test_returns_empty_on_total_failure(self):
        from src.discovery.codegen.llm_generator import generate_edge_filter

        hypothesis = {
            "description": "Test",
            "evidence": {},
            "filter_spec": {
                "name": "fail_filter",
                "class_name": "FailFilter",
                "category": "entry",
                "description": "Test",
                "params": {},
                "logic_description": "Test",
            },
        }

        with patch("subprocess.run", side_effect=FileNotFoundError("not found")):
            result = generate_edge_filter(hypothesis)
            assert result.filter_code is None
