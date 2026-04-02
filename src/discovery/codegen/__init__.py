"""LLM Code Generation + Safety Pipeline for EdgeFilter discovery.

Generates new EdgeFilter subclasses from discovery findings using Claude
Code CLI, validates through AST whitelist + pytest + mini-backtest, and
auto-registers passing filters into EdgeManager._REGISTRY.
"""

from src.discovery.codegen.ast_validator import validate_ast, ASTValidationResult
from src.discovery.codegen.llm_generator import generate_edge_filter, GeneratedCode
from src.discovery.codegen.template_renderer import (
    render_codegen_prompt,
    render_filter_template,
    render_test_template,
)
from src.discovery.codegen.safety_pipeline import SafetyPipeline, PipelineResult
from src.discovery.codegen.registry import (
    register_filter,
    unregister_filter,
    list_generated_filters,
    save_registry_manifest,
    load_generated_filters,
)

__all__ = [
    "validate_ast",
    "ASTValidationResult",
    "generate_edge_filter",
    "GeneratedCode",
    "render_codegen_prompt",
    "render_filter_template",
    "render_test_template",
    "SafetyPipeline",
    "PipelineResult",
    "register_filter",
    "unregister_filter",
    "list_generated_filters",
    "save_registry_manifest",
    "load_generated_filters",
]
