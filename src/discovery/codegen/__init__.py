"""LLM Code Generation + Safety Pipeline for EdgeFilter discovery.

Generates new EdgeFilter subclasses from discovery findings using Claude
Code CLI, validates through AST whitelist + pytest + mini-backtest, and
auto-registers passing filters into EdgeManager._REGISTRY.
"""

from src.discovery.codegen.ast_validator import validate_ast, ASTValidationResult

__all__ = ["validate_ast", "ASTValidationResult"]
