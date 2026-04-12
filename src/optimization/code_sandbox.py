"""Git branch sandbox for testing LLM code patches."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class CodeSandbox:
    """Test code patches on a temporary git branch."""

    def __init__(self) -> None:
        self._branch_name: str | None = None

    def test_patches(
        self,
        instrument: str,
        epoch: int,
        patches: list[dict[str, str]],
        backtest_fn: Callable,
        data: Any,
        config: dict,
        baseline_return: float,
    ) -> bool:
        """Apply patches on a temp branch, backtest, return True if improved."""
        if not patches:
            return False

        branch = f"llm-patch-{instrument.lower()}-epoch{epoch}"
        self._branch_name = branch

        try:
            # Create temp branch
            self._git("checkout", "-b", branch)

            # Apply patches
            applied = 0
            for patch in patches:
                if self._apply_patch(patch):
                    applied += 1
                else:
                    logger.warning("Patch failed: %s", patch.get("description", "unknown"))

            if applied == 0:
                logger.info("No patches applied successfully — aborting")
                self._cleanup(branch)
                return False

            # Commit patches
            self._git("add", "-A")
            desc = "; ".join(p.get("description", "LLM patch") for p in patches)
            self._git("commit", "-m", f"llm-meta: {desc}")

            # Run backtest on patched code
            logger.info("Running backtest on patched branch %s...", branch)
            result = backtest_fn(data, config, instrument, 50_000.0)

            if result is None:
                logger.warning("Patched backtest failed — reverting")
                self._cleanup(branch)
                return False

            patched_return = result.get("total_return_pct", -999)
            logger.info(
                "Patch result: %.2f%% return (baseline: %.2f%%)",
                patched_return, baseline_return,
            )

            if patched_return > baseline_return:
                # Merge to main
                self._git("checkout", "main")
                self._git("merge", branch, "--no-ff", "-m",
                          f"Merge LLM patch for {instrument} epoch {epoch}: "
                          f"{patched_return:.2f}% > {baseline_return:.2f}%")
                self._git("branch", "-d", branch)
                logger.info("Patch MERGED — improved %.2f%% -> %.2f%%",
                           baseline_return, patched_return)
                return True
            else:
                logger.info("Patch did NOT improve — discarding")
                self._cleanup(branch)
                return False

        except Exception as exc:
            logger.error("Code sandbox error: %s", exc)
            self._cleanup(branch)
            return False

    def _apply_patch(self, patch: dict[str, str]) -> bool:
        """Apply a search/replace patch to a file."""
        filepath = _PROJECT_ROOT / patch.get("file", "")
        if not filepath.exists():
            logger.warning("Patch target not found: %s", filepath)
            return False

        search = patch.get("search", "")
        replace = patch.get("replace", "")
        if not search or not replace:
            return False

        content = filepath.read_text(encoding="utf-8")
        if search not in content:
            logger.warning("Search string not found in %s", filepath)
            return False

        new_content = content.replace(search, replace, 1)
        filepath.write_text(new_content, encoding="utf-8")
        logger.info("Patched %s: %s", filepath.name, patch.get("description", ""))
        return True

    def _cleanup(self, branch: str) -> None:
        """Return to main and delete the temp branch."""
        try:
            self._git("checkout", "main")
            self._git("branch", "-D", branch)
        except Exception:
            pass
        self._branch_name = None

    @staticmethod
    def _git(*args: str) -> str:
        """Run a git command in the project root."""
        result = subprocess.run(
            ["git"] + list(args),
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr}")
        return result.stdout
