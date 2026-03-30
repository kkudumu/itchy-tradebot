"""
CLAUDE.md learning integration for the agentic self-optimization loop.

ClaudeMdWriter appends structured learnings to a ``## Strategy Learnings``
section after each optimization run, so future Claude sessions inherit
the accumulated knowledge.  It also parses the section back into a
structured list for programmatic consumption.

Format of each learning line
-----------------------------
``- Run {run_id}: {changes_made} → Sharpe {before:.2f}→{after:.2f} | WinRate {before:.1%}→{after:.1%} | MaxDD {before:.1%}→{after:.1%} ({verdict})``

Missing metric values are omitted gracefully to keep lines concise.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Sentinel heading that delimits the managed section
_SECTION_HEADING = "## Strategy Learnings"

# Regex to parse a previously-written learning line
_LINE_PATTERN = re.compile(
    r"^- Run (?P<run_id>\S+): (?P<changes_made>.+?) → (?P<metrics>.+?) \((?P<verdict>kept|reverted)\)\s*$"
)


def _fmt_metric(label: str, before: Optional[float], after: Optional[float],
                pct: bool = False) -> Optional[str]:
    """Return ``'Label B→A'`` string if both values present, else None."""
    if before is None or after is None:
        return None
    if pct:
        return f"{label} {before:.1%}→{after:.1%}"
    return f"{label} {before:.2f}→{after:.2f}"


class ClaudeMdWriter:
    """Append and read strategy learnings in a CLAUDE.md file.

    Parameters
    ----------
    claude_md_path:
        Path to the CLAUDE.md file.  Defaults to ``"CLAUDE.md"`` in the
        current working directory.  The file and the ``## Strategy Learnings``
        section are created automatically if absent.
    """

    def __init__(self, claude_md_path: str = "CLAUDE.md") -> None:
        self._path = Path(claude_md_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append_learning(
        self,
        run_id: str,
        changes_made: str,
        metrics_before: dict,
        metrics_after: dict,
        verdict: str,
    ) -> None:
        """Append one learning line to the Strategy Learnings section.

        Parameters
        ----------
        run_id:
            Unique identifier for this optimization run (e.g. ``"run_042"``).
        changes_made:
            Short description of the parameter changes tried.
        metrics_before:
            Dict with optional keys: ``sharpe``, ``win_rate``, ``max_dd``.
        metrics_after:
            Same keys as *metrics_before* after the optimization.
        verdict:
            ``"kept"`` if the change improved performance, ``"reverted"``
            otherwise.
        """
        if verdict not in ("kept", "reverted"):
            logger.warning(
                "ClaudeMdWriter: unexpected verdict %r — expected 'kept' or 'reverted'",
                verdict,
            )

        # Build metric delta fragments
        parts: List[str] = []

        sharpe_str = _fmt_metric(
            "Sharpe",
            metrics_before.get("sharpe"),
            metrics_after.get("sharpe"),
            pct=False,
        )
        if sharpe_str:
            parts.append(sharpe_str)

        wr_str = _fmt_metric(
            "WinRate",
            metrics_before.get("win_rate"),
            metrics_after.get("win_rate"),
            pct=True,
        )
        if wr_str:
            parts.append(wr_str)

        dd_str = _fmt_metric(
            "MaxDD",
            metrics_before.get("max_dd"),
            metrics_after.get("max_dd"),
            pct=True,
        )
        if dd_str:
            parts.append(dd_str)

        metrics_text = " | ".join(parts) if parts else "no metrics"
        changes_text = changes_made.strip() if changes_made.strip() else "(no changes)"

        line = f"- Run {run_id}: {changes_text} → {metrics_text} ({verdict})\n"

        content = self._read_file()
        updated = self._ensure_section(content)
        updated = self._append_line(updated, line)
        self._write_file(updated)
        logger.debug("ClaudeMdWriter: appended learning for run %s", run_id)

    def read_learnings(self) -> List[dict]:
        """Parse the Strategy Learnings section into a structured list.

        Returns
        -------
        List of dicts, each containing:

        ``run_id`` (str), ``changes_made`` (str), ``metrics_text`` (str),
        ``verdict`` (str).

        Returns an empty list when the section is absent or empty.
        """
        content = self._read_file()
        lines = self._extract_section_lines(content)

        results: List[dict] = []
        for raw_line in lines:
            stripped = raw_line.strip()
            if not stripped or not stripped.startswith("- "):
                continue
            match = _LINE_PATTERN.match(stripped)
            if match:
                results.append(
                    {
                        "run_id": match.group("run_id"),
                        "changes_made": match.group("changes_made"),
                        "metrics_text": match.group("metrics"),
                        "verdict": match.group("verdict"),
                    }
                )
            else:
                logger.debug(
                    "ClaudeMdWriter: could not parse learning line: %r", stripped
                )
        return results

    def get_learnings_summary(self) -> str:
        """Return all learnings as a formatted string for Claude prompts.

        Returns
        -------
        Multi-line string with a header and one line per learning.
        Returns ``"No strategy learnings recorded yet."`` when empty.
        """
        learnings = self.read_learnings()
        if not learnings:
            return "No strategy learnings recorded yet."

        lines: List[str] = ["Strategy Learnings (most recent last):"]
        for entry in learnings:
            lines.append(
                f"  Run {entry['run_id']}: {entry['changes_made']} "
                f"→ {entry['metrics_text']} ({entry['verdict']})"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_file(self) -> str:
        """Return the full content of the CLAUDE.md file, or '' if absent."""
        try:
            return self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.debug("ClaudeMdWriter: %s not found — will create it", self._path)
            return ""
        except OSError as exc:
            logger.error("ClaudeMdWriter: error reading %s: %s", self._path, exc)
            return ""

    def _write_file(self, content: str) -> None:
        """Write *content* to the CLAUDE.md file, creating parents as needed."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(content, encoding="utf-8")
        except OSError as exc:
            logger.error("ClaudeMdWriter: error writing %s: %s", self._path, exc)

    def _ensure_section(self, content: str) -> str:
        """Return *content* with ``## Strategy Learnings`` section present."""
        if _SECTION_HEADING in content:
            return content

        separator = "\n\n" if content.rstrip() else ""
        stub = f"{separator}{_SECTION_HEADING}\n"
        return content.rstrip() + stub

    def _extract_section_lines(self, content: str) -> List[str]:
        """Return the raw lines that fall inside the Strategy Learnings section.

        Stops at the next ``##`` heading or end-of-file.  Returns an empty
        list when the section is not present.
        """
        heading_idx = content.find(_SECTION_HEADING)
        if heading_idx == -1:
            return []

        # Slice content starting just after the heading line
        after_heading = content[heading_idx + len(_SECTION_HEADING):]

        # Find the next ## heading after our section
        next_heading_match = re.search(r"^##\s", after_heading, re.MULTILINE)
        if next_heading_match:
            section_body = after_heading[: next_heading_match.start()]
        else:
            section_body = after_heading

        return section_body.splitlines()

    def _append_line(self, content: str, line: str) -> str:
        """Insert *line* at the end of the Strategy Learnings section.

        The section ends at the next ``##`` heading or at end-of-file.
        """
        heading_idx = content.find(_SECTION_HEADING)
        if heading_idx == -1:
            # Shouldn't happen after _ensure_section, but guard anyway
            return content + line

        # Find the end of the section: next ## heading after the section start
        next_heading_match = re.search(
            r"^##\s", content[heading_idx + len(_SECTION_HEADING):], re.MULTILINE
        )

        if next_heading_match:
            insert_at = heading_idx + len(_SECTION_HEADING) + next_heading_match.start()
            # Back up over any trailing blank lines before the next section
            before = content[:insert_at].rstrip("\n") + "\n"
            after = content[insert_at:]
            return before + line + "\n" + after
        else:
            # Section runs to end of file
            before = content.rstrip("\n") + "\n"
            return before + line
