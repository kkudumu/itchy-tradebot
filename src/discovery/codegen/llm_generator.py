"""LLM code generation for EdgeFilter subclasses.

Uses Claude Code CLI (claude -p) as the primary generator, with Codex CLI
(codex exec) as fallback when Claude hits rate limits or is unavailable.
Parses the LLM response to extract filter code and test code blocks.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.discovery.codegen.template_renderer import render_codegen_prompt

logger = logging.getLogger(__name__)


@dataclass
class GeneratedCode:
    """Result of LLM code generation."""
    filter_code: Optional[str] = None
    test_code: Optional[str] = None
    filter_path: Optional[str] = None
    test_path: Optional[str] = None
    provider: str = ""  # "claude" or "codex"
    raw_response: str = ""


def parse_generated_code(response: str) -> GeneratedCode:
    """Parse LLM response to extract filter and test code blocks.

    Expects two ```python code blocks, the first containing the filter
    implementation and the second containing the test class. Identifies
    which is which via the FILE comment or content heuristics.

    Parameters
    ----------
    response:
        Raw LLM response string.

    Returns
    -------
    GeneratedCode with filter_code and test_code extracted.
    """
    if not response or not response.strip():
        return GeneratedCode(raw_response=response)

    # Extract all ```python blocks
    blocks = re.findall(
        r"```python\s*\n(.*?)\n\s*```",
        response,
        re.DOTALL,
    )

    if not blocks:
        return GeneratedCode(raw_response=response)

    result = GeneratedCode(raw_response=response)

    for block in blocks:
        block = block.strip()
        # Identify by FILE comment or content
        is_test = (
            "# FILE: tests/" in block
            or "class Test" in block
            and "def test_" in block
            and "EdgeFilter" not in block.split("class Test")[0]
        )
        is_filter = (
            "# FILE: src/edges/" in block
            or "(EdgeFilter)" in block
        )

        if is_filter and result.filter_code is None:
            result.filter_code = block
            # Extract path from FILE comment
            path_match = re.search(r"# FILE:\s*(\S+)", block)
            if path_match:
                result.filter_path = path_match.group(1)
        elif is_test and result.test_code is None:
            result.test_code = block
            path_match = re.search(r"# FILE:\s*(\S+)", block)
            if path_match:
                result.test_path = path_match.group(1)

    # If we only found one block and it has EdgeFilter, it's the filter
    if len(blocks) == 1 and result.filter_code is None and "(EdgeFilter)" in blocks[0]:
        result.filter_code = blocks[0].strip()

    return result


def _run_claude(prompt: str, timeout: int = 300) -> str:
    """Invoke Claude Code CLI and return stdout."""
    result = subprocess.run(
        ["claude", "-p"],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout


def _run_codex(prompt: str, timeout: int = 300) -> str:
    """Invoke Codex CLI as fallback and return stdout."""
    result = subprocess.run(
        ["codex", "exec", "--yolo", "--ephemeral"],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout


def generate_edge_filter(
    hypothesis: Dict[str, Any],
    timeout: int = 300,
    claude_only: bool = False,
) -> GeneratedCode:
    """Generate an EdgeFilter subclass using LLM code generation.

    Tries Claude Code CLI first. On failure (rate limit, timeout, not
    installed), falls back to Codex CLI unless claude_only=True.

    Parameters
    ----------
    hypothesis:
        Dict with keys: description, evidence, filter_spec.
    timeout:
        CLI subprocess timeout in seconds.
    claude_only:
        If True, skip Codex fallback.

    Returns
    -------
    GeneratedCode with filter_code and test_code (or None on failure).
    """
    prompt = render_codegen_prompt(hypothesis)

    # Try Claude first
    try:
        logger.info("Generating EdgeFilter via Claude Code CLI")
        response = _run_claude(prompt, timeout=timeout)
        if response.strip():
            result = parse_generated_code(response)
            result.provider = "claude"
            if result.filter_code:
                logger.info("Claude generated filter code (%d chars)", len(result.filter_code))
                return result
            logger.warning("Claude response did not contain valid code blocks")
    except FileNotFoundError:
        logger.warning("Claude CLI not found, trying Codex fallback")
    except subprocess.TimeoutExpired:
        logger.warning("Claude CLI timed out after %ds", timeout)
    except Exception as exc:
        logger.warning("Claude CLI failed: %s", exc)

    # Codex fallback
    if not claude_only:
        try:
            logger.info("Falling back to Codex CLI")
            response = _run_codex(prompt, timeout=timeout)
            if response.strip():
                result = parse_generated_code(response)
                result.provider = "codex"
                if result.filter_code:
                    logger.info("Codex generated filter code (%d chars)", len(result.filter_code))
                    return result
        except FileNotFoundError:
            logger.warning("Codex CLI not found")
        except subprocess.TimeoutExpired:
            logger.warning("Codex CLI timed out after %ds", timeout)
        except Exception as exc:
            logger.warning("Codex CLI failed: %s", exc)

    logger.error("All LLM providers failed for hypothesis: %s", hypothesis.get("id", "unknown"))
    return GeneratedCode()
