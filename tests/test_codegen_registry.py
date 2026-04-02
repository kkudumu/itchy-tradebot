"""Tests for dynamic EdgeFilter registration into EdgeManager._REGISTRY."""

from unittest.mock import patch
from pathlib import Path

import pytest


class TestDynamicRegistry:
    def _make_filter_code(self, name: str, class_name: str) -> str:
        return f'''
from src.edges.base import EdgeFilter, EdgeContext, EdgeResult


class {class_name}(EdgeFilter):
    def __init__(self, config: dict) -> None:
        super().__init__("{name}", config)
        params = config.get("params", {{}})
        self._threshold = float(params.get("threshold", 30.0))

    def should_allow(self, context: EdgeContext) -> EdgeResult:
        if not self.enabled:
            return self._disabled_result()
        if context.adx >= self._threshold:
            return EdgeResult(allowed=True, edge_name=self.name, reason="OK")
        return EdgeResult(allowed=False, edge_name=self.name, reason="blocked")
'''

    def test_register_filter_adds_to_registry(self):
        from src.discovery.codegen.registry import register_filter
        from src.edges.manager import _REGISTRY

        name = "test_dynamic_adx_001"
        code = self._make_filter_code(name, "TestDynamicADXFilter001")

        original_keys = set(_REGISTRY.keys())
        try:
            register_filter(
                filter_name=name,
                filter_class_name="TestDynamicADXFilter001",
                filter_code=code,
                category="entry",
            )
            assert name in _REGISTRY
            cls, cat = _REGISTRY[name]
            assert cls.__name__ == "TestDynamicADXFilter001"
            assert cat == "entry"
        finally:
            _REGISTRY.pop(name, None)

    def test_registered_filter_can_be_instantiated(self):
        from src.discovery.codegen.registry import register_filter
        from src.edges.manager import _REGISTRY
        from src.edges.base import EdgeContext, EdgeResult

        name = "test_dynamic_adx_002"
        code = self._make_filter_code(name, "TestDynamicADXFilter002")

        try:
            register_filter(
                filter_name=name,
                filter_class_name="TestDynamicADXFilter002",
                filter_code=code,
                category="entry",
            )
            cls, _ = _REGISTRY[name]
            instance = cls({"enabled": True, "params": {"threshold": 25.0}})
            assert instance.name == name
            assert instance.enabled is True
        finally:
            _REGISTRY.pop(name, None)

    def test_unregister_filter_removes_from_registry(self):
        from src.discovery.codegen.registry import register_filter, unregister_filter
        from src.edges.manager import _REGISTRY

        name = "test_dynamic_adx_003"
        code = self._make_filter_code(name, "TestDynamicADXFilter003")

        try:
            register_filter(
                filter_name=name,
                filter_class_name="TestDynamicADXFilter003",
                filter_code=code,
                category="entry",
            )
            assert name in _REGISTRY
            unregister_filter(name)
            assert name not in _REGISTRY
        finally:
            _REGISTRY.pop(name, None)

    def test_list_generated_filters(self):
        from src.discovery.codegen.registry import register_filter, list_generated_filters
        from src.edges.manager import _REGISTRY

        name = "test_dynamic_adx_004"
        code = self._make_filter_code(name, "TestDynamicADXFilter004")

        try:
            register_filter(
                filter_name=name,
                filter_class_name="TestDynamicADXFilter004",
                filter_code=code,
                category="entry",
            )
            generated = list_generated_filters()
            assert name in generated
        finally:
            _REGISTRY.pop(name, None)

    def test_persist_and_load_from_disk(self, tmp_path):
        from src.discovery.codegen.registry import (
            register_filter,
            save_registry_manifest,
            load_generated_filters,
        )
        from src.edges.manager import _REGISTRY

        name = "test_dynamic_adx_005"
        code = self._make_filter_code(name, "TestDynamicADXFilter005")

        gen_dir = tmp_path / "generated"
        gen_dir.mkdir()

        try:
            # Write filter file
            filter_file = gen_dir / f"{name}.py"
            filter_file.write_text(code, encoding="utf-8")

            # Register
            register_filter(
                filter_name=name,
                filter_class_name="TestDynamicADXFilter005",
                filter_code=code,
                category="entry",
            )

            # Save manifest
            manifest_path = save_registry_manifest(str(gen_dir))
            assert Path(manifest_path).exists()

            # Unregister and reload
            _REGISTRY.pop(name, None)
            assert name not in _REGISTRY

            load_generated_filters(str(gen_dir))
            assert name in _REGISTRY
        finally:
            _REGISTRY.pop(name, None)
