"""AST whitelist validator for generated EdgeFilter code.

Parses generated Python source with the ast module and enforces:
1. Only allowed imports (numpy, pandas, src.edges.base, src.indicators.*)
2. No dangerous builtins (eval, exec, __import__, open, compile, getattr)
3. No dangerous modules (os, subprocess, sys, socket, http, urllib, shutil, pathlib, io)
4. Must contain exactly one class that inherits from EdgeFilter
5. That class must define a should_allow method
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import List, Set

logger = logging.getLogger(__name__)

# Modules that generated code is allowed to import
ALLOWED_MODULES: Set[str] = {
    "numpy",
    "np",
    "pandas",
    "pd",
    "math",
    "statistics",
    "dataclasses",
    "typing",
    "datetime",
    "src.edges.base",
    "src.edges",
    "src.indicators",
    "src.indicators.ichimoku",
    "src.indicators.confluence",
    "src.indicators.divergence",
    "src.indicators.sessions",
    "src.indicators.signals",
    "__future__",
}

# Module prefixes that are also allowed (for src.indicators.*)
ALLOWED_MODULE_PREFIXES: List[str] = [
    "src.edges.",
    "src.indicators.",
]

# Modules explicitly banned (even if they look harmless)
BANNED_MODULES: Set[str] = {
    "os",
    "subprocess",
    "sys",
    "socket",
    "http",
    "urllib",
    "requests",
    "shutil",
    "pathlib",
    "io",
    "importlib",
    "ctypes",
    "multiprocessing",
    "threading",
    "signal",
    "tempfile",
    "glob",
    "fnmatch",
    "pickle",
    "shelve",
    "marshal",
    "code",
    "codeop",
    "compileall",
    "webbrowser",
    "smtplib",
    "ftplib",
    "xmlrpc",
}

# Builtin function calls that are banned
BANNED_BUILTINS: Set[str] = {
    "eval",
    "exec",
    "__import__",
    "open",
    "compile",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "dir",
    "breakpoint",
    "input",
    "print",  # no side effects in filters
}


@dataclass
class ASTValidationResult:
    """Result of AST whitelist validation."""
    is_valid: bool
    violations: List[str] = field(default_factory=list)
    class_name: str = ""
    has_should_allow: bool = False


def _is_module_allowed(module_name: str) -> bool:
    """Check if a module import is on the whitelist."""
    if module_name in ALLOWED_MODULES:
        return True
    for prefix in ALLOWED_MODULE_PREFIXES:
        if module_name.startswith(prefix):
            return True
    return False


def validate_ast(source_code: str) -> ASTValidationResult:
    """Validate generated Python source code against the AST whitelist.

    Parameters
    ----------
    source_code:
        Complete Python source code string to validate.

    Returns
    -------
    ASTValidationResult with is_valid=True if all checks pass,
    or is_valid=False with a list of violation descriptions.
    """
    violations: List[str] = []

    # Step 1: Parse the AST
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        return ASTValidationResult(
            is_valid=False,
            violations=[f"Syntax/parse error: {e}"],
        )

    # Step 2: Check imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name
                if module_name in BANNED_MODULES or module_name.split(".")[0] in BANNED_MODULES:
                    violations.append(
                        f"Banned import: '{module_name}' (line {node.lineno})"
                    )
                elif not _is_module_allowed(module_name) and not _is_module_allowed(module_name.split(".")[0]):
                    violations.append(
                        f"Disallowed import: '{module_name}' (line {node.lineno}). "
                        f"Only numpy, pandas, src.edges.base, src.indicators.* are allowed."
                    )

        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if module_name in BANNED_MODULES or module_name.split(".")[0] in BANNED_MODULES:
                violations.append(
                    f"Banned import: 'from {module_name}' (line {node.lineno})"
                )
            elif not _is_module_allowed(module_name) and not _is_module_allowed(module_name.split(".")[0]):
                violations.append(
                    f"Disallowed import: 'from {module_name}' (line {node.lineno}). "
                    f"Only numpy, pandas, src.edges.base, src.indicators.* are allowed."
                )

    # Step 3: Check for banned builtin calls
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            func_name = None
            if isinstance(func, ast.Name):
                func_name = func.id
            elif isinstance(func, ast.Attribute):
                func_name = func.attr

            if func_name and func_name in BANNED_BUILTINS:
                violations.append(
                    f"Banned builtin call: '{func_name}()' (line {node.lineno})"
                )

    # Step 4: Check for EdgeFilter subclass
    edge_filter_classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                base_name = None
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                if base_name == "EdgeFilter":
                    edge_filter_classes.append(node)

    if not edge_filter_classes:
        violations.append(
            "No EdgeFilter subclass found. Generated code must define "
            "exactly one class that inherits from EdgeFilter."
        )
    else:
        cls = edge_filter_classes[0]
        class_name = cls.name

        # Step 5: Check for should_allow method
        has_should_allow = False
        for item in cls.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if item.name == "should_allow":
                    has_should_allow = True
                    break

        if not has_should_allow:
            violations.append(
                f"Class '{class_name}' must define a 'should_allow' method."
            )

        result = ASTValidationResult(
            is_valid=len(violations) == 0,
            violations=violations,
            class_name=class_name,
            has_should_allow=has_should_allow,
        )
        return result

    return ASTValidationResult(
        is_valid=len(violations) == 0,
        violations=violations,
    )
