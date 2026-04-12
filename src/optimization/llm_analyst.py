"""Execute claude -p with the assembled prompt and parse the structured response."""
from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_REPORTS_DIR = _PROJECT_ROOT / "reports" / "llm_analysis"


@dataclass
class AnalysisResult:
    """Parsed output from the LLM analyst."""
    reasoning: str = ""
    config_changes: dict[str, Any] = field(default_factory=dict)
    code_patches: list[dict[str, str]] = field(default_factory=list)
    raw_output: str = ""
    output_path: str = ""


class LLMAnalyst:
    """Run claude -p on the assembled prompt and parse the response."""

    def __init__(self, db_pool) -> None:
        self._pool = db_pool

    def analyze(
        self,
        instrument: str,
        epoch: int,
        prompt: str,
    ) -> AnalysisResult:
        """Send prompt to claude -p, parse response, persist to DB."""
        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        output_path = _REPORTS_DIR / f"{instrument}_epoch_{epoch}.txt"

        # Execute claude -p
        logger.info("Calling claude -p for %s epoch %d...", instrument, epoch)
        raw_output = self._call_claude(prompt)

        # Save raw output
        output_path.write_text(raw_output, encoding="utf-8")
        logger.info("LLM output saved to %s (%d chars)", output_path, len(raw_output))

        # Parse structured response
        result = self._parse_response(raw_output)
        result.raw_output = raw_output
        result.output_path = str(output_path)

        # Persist to DB
        self._persist(instrument, epoch, result)

        return result

    def _call_claude(self, prompt: str) -> str:
        """Execute claude -p with the prompt."""
        try:
            proc = subprocess.run(
                ["claude", "-p", prompt, "--dangerously-skip-permissions"],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=str(_PROJECT_ROOT),
            )
            if proc.returncode != 0:
                logger.warning("claude -p returned code %d: %s", proc.returncode, proc.stderr[:500])
            return proc.stdout or proc.stderr or ""
        except FileNotFoundError:
            logger.error("claude CLI not found — is it installed and on PATH?")
            return ""
        except subprocess.TimeoutExpired:
            logger.error("claude -p timed out after 300s")
            return ""
        except Exception as exc:
            logger.error("claude -p failed: %s", exc)
            return ""

    def _parse_response(self, raw: str) -> AnalysisResult:
        """Parse the three sections from the LLM output."""
        result = AnalysisResult()

        # Extract REASONING section
        reasoning_match = re.search(
            r"###?\s*REASONING\s*\n(.*?)(?=###?\s*CONFIG_CHANGES|$)",
            raw, re.DOTALL | re.IGNORECASE,
        )
        if reasoning_match:
            result.reasoning = reasoning_match.group(1).strip()

        # Extract CONFIG_CHANGES JSON
        config_match = re.search(
            r"```json\s*\n\s*\{[^`]*?\"config_changes\"[^`]*?\}\s*\n\s*```",
            raw, re.DOTALL,
        )
        if config_match:
            try:
                parsed = json.loads(config_match.group().strip().strip("`").strip("json").strip())
                result.config_changes = parsed.get("config_changes", {})
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse config_changes JSON: %s", exc)

        # Extract CODE_PATCHES JSON
        patches_match = re.search(
            r"```json\s*\n\s*\{[^`]*?\"code_patches\"[^`]*?\}\s*\n\s*```",
            raw, re.DOTALL,
        )
        if patches_match:
            try:
                parsed = json.loads(patches_match.group().strip().strip("`").strip("json").strip())
                result.code_patches = parsed.get("code_patches", [])
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse code_patches JSON: %s", exc)

        return result

    def _persist(self, instrument: str, epoch: int, result: AnalysisResult) -> None:
        """Save analysis to llm_analysis table."""
        try:
            with self._pool.get_cursor() as cur:
                cur.execute("""
                    INSERT INTO llm_analysis (
                        instrument, epoch, reasoning, config_changes,
                        code_patches, raw_output_path
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    instrument, epoch, result.reasoning,
                    json.dumps(result.config_changes),
                    json.dumps(result.code_patches),
                    result.output_path,
                ))
        except Exception as exc:
            logger.warning("Failed to persist LLM analysis: %s", exc)
