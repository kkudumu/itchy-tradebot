"""Dynamic EdgeFilter registration into EdgeManager._REGISTRY.

Provides runtime registration of generated EdgeFilter subclasses so
they participate in the edge pipeline without manual code changes.
Also handles persistence via a JSON manifest for reload across restarts.
"""

from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.edges.base import EdgeFilter

logger = logging.getLogger(__name__)

# Track which filters were dynamically registered (vs built-in)
_GENERATED_FILTER_NAMES: Set[str] = set()


def register_filter(
    filter_name: str,
    filter_class_name: str,
    filter_code: str,
    category: str = "entry",
) -> None:
    """Dynamically register a generated EdgeFilter into _REGISTRY.

    Compiles the source code, extracts the class, and adds it to
    EdgeManager's _REGISTRY dict.

    Parameters
    ----------
    filter_name:
        The registered name (e.g., 'high_adx_london').
    filter_class_name:
        The class name (e.g., 'HighADXLondonFilter').
    filter_code:
        Complete Python source for the EdgeFilter module.
    category:
        Edge category: 'entry', 'exit', or 'modifier'.

    Raises
    ------
    ValueError:
        If the class cannot be found in the compiled code.
    """
    from src.edges.manager import _REGISTRY

    # Compile and execute the code in an isolated module
    module_name = f"src.edges.generated.{filter_name}"
    module = types.ModuleType(module_name)
    module.__file__ = f"<generated:{filter_name}>"

    try:
        compiled = compile(filter_code, module.__file__, "exec")
        exec(compiled, module.__dict__)
    except Exception as exc:
        raise ValueError(f"Failed to compile generated code for {filter_name}: {exc}") from exc

    # Extract the filter class
    filter_cls = getattr(module, filter_class_name, None)
    if filter_cls is None:
        raise ValueError(
            f"Class '{filter_class_name}' not found in generated code for {filter_name}"
        )

    # Verify it's an EdgeFilter subclass
    if not (isinstance(filter_cls, type) and issubclass(filter_cls, EdgeFilter)):
        raise ValueError(
            f"'{filter_class_name}' is not an EdgeFilter subclass"
        )

    # Register in _REGISTRY
    _REGISTRY[filter_name] = (filter_cls, category)
    _GENERATED_FILTER_NAMES.add(filter_name)

    # Also register the module in sys.modules so imports work
    sys.modules[module_name] = module

    logger.info(
        "Registered generated EdgeFilter: %s (%s) as '%s'",
        filter_class_name, category, filter_name,
    )


def unregister_filter(filter_name: str) -> None:
    """Remove a generated filter from _REGISTRY.

    Parameters
    ----------
    filter_name:
        The registered name to remove.
    """
    from src.edges.manager import _REGISTRY

    _REGISTRY.pop(filter_name, None)
    _GENERATED_FILTER_NAMES.discard(filter_name)

    module_name = f"src.edges.generated.{filter_name}"
    sys.modules.pop(module_name, None)

    logger.info("Unregistered generated EdgeFilter: %s", filter_name)


def list_generated_filters() -> List[str]:
    """Return names of all dynamically registered generated filters."""
    return sorted(_GENERATED_FILTER_NAMES)


def save_registry_manifest(generated_dir: str) -> str:
    """Save a JSON manifest of all generated filters for reload.

    The manifest records filter_name, class_name, category, and the
    source file path so filters can be reloaded on next startup.

    Parameters
    ----------
    generated_dir:
        Directory containing generated filter .py files.

    Returns
    -------
    Path to the saved manifest.json file.
    """
    from src.edges.manager import _REGISTRY

    manifest: List[Dict[str, str]] = []
    gen_dir = Path(generated_dir)

    for name in sorted(_GENERATED_FILTER_NAMES):
        if name in _REGISTRY:
            cls, category = _REGISTRY[name]
            source_file = gen_dir / f"{name}.py"
            manifest.append({
                "filter_name": name,
                "class_name": cls.__name__,
                "category": category,
                "source_file": str(source_file),
            })

    manifest_path = gen_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved registry manifest with %d filters to %s", len(manifest), manifest_path)
    return str(manifest_path)


def load_generated_filters(generated_dir: str) -> int:
    """Load and register all generated filters from the manifest.

    Called at startup to restore previously validated filters.

    Parameters
    ----------
    generated_dir:
        Directory containing generated filter .py files and manifest.json.

    Returns
    -------
    Number of filters successfully loaded.
    """
    gen_dir = Path(generated_dir)
    manifest_path = gen_dir / "manifest.json"

    if not manifest_path.exists():
        logger.debug("No manifest.json found in %s -- no generated filters to load", gen_dir)
        return 0

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    loaded = 0

    for entry in manifest:
        filter_name = entry["filter_name"]
        class_name = entry["class_name"]
        category = entry["category"]
        source_file = Path(entry["source_file"])

        if not source_file.exists():
            logger.warning("Source file missing for %s: %s", filter_name, source_file)
            continue

        try:
            filter_code = source_file.read_text(encoding="utf-8")
            register_filter(
                filter_name=filter_name,
                filter_class_name=class_name,
                filter_code=filter_code,
                category=category,
            )
            loaded += 1
        except Exception as exc:
            logger.warning("Failed to load generated filter %s: %s", filter_name, exc)

    logger.info("Loaded %d/%d generated filters from manifest", loaded, len(manifest))
    return loaded
