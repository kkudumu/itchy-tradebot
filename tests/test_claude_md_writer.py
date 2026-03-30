"""
Unit tests for ClaudeMdWriter — CLAUDE.md learning integration.

All tests are fully isolated: they use pytest's ``tmp_path`` fixture so no
real CLAUDE.md file is ever touched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.learning.claude_md_writer import ClaudeMdWriter, _SECTION_HEADING


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _writer(tmp_path: Path, filename: str = "CLAUDE.md") -> ClaudeMdWriter:
    """Return a ClaudeMdWriter pointing at *tmp_path/filename*."""
    return ClaudeMdWriter(str(tmp_path / filename))


def _default_before() -> dict:
    return {"sharpe": -0.5, "win_rate": 0.42, "max_dd": 0.08}


def _default_after() -> dict:
    return {"sharpe": 0.9, "win_rate": 0.55, "max_dd": 0.05}


# ---------------------------------------------------------------------------
# 1. append_learning — creates section in a brand-new file
# ---------------------------------------------------------------------------

class TestAppendLearningNewFile:
    def test_creates_file_when_absent(self, tmp_path: Path) -> None:
        w = _writer(tmp_path)
        w.append_learning("run_001", "Tightened SL", _default_before(), _default_after(), "kept")
        assert (tmp_path / "CLAUDE.md").exists()

    def test_section_heading_written(self, tmp_path: Path) -> None:
        w = _writer(tmp_path)
        w.append_learning("run_001", "Tightened SL", _default_before(), _default_after(), "kept")
        content = (tmp_path / "CLAUDE.md").read_text()
        assert _SECTION_HEADING in content

    def test_run_id_in_output(self, tmp_path: Path) -> None:
        w = _writer(tmp_path)
        w.append_learning("run_042", "Changed TP", _default_before(), _default_after(), "kept")
        content = (tmp_path / "CLAUDE.md").read_text()
        assert "Run run_042" in content

    def test_verdict_kept_in_output(self, tmp_path: Path) -> None:
        w = _writer(tmp_path)
        w.append_learning("r1", "desc", _default_before(), _default_after(), "kept")
        content = (tmp_path / "CLAUDE.md").read_text()
        assert "(kept)" in content

    def test_verdict_reverted_in_output(self, tmp_path: Path) -> None:
        w = _writer(tmp_path)
        w.append_learning("r1", "desc", _default_before(), _default_after(), "reverted")
        content = (tmp_path / "CLAUDE.md").read_text()
        assert "(reverted)" in content


# ---------------------------------------------------------------------------
# 2. append_learning — adds to existing section
# ---------------------------------------------------------------------------

class TestAppendLearningExisting:
    def test_second_append_adds_second_line(self, tmp_path: Path) -> None:
        w = _writer(tmp_path)
        w.append_learning("r1", "Change A", _default_before(), _default_after(), "kept")
        w.append_learning("r2", "Change B", _default_before(), _default_after(), "reverted")
        content = (tmp_path / "CLAUDE.md").read_text()
        assert "Run r1" in content
        assert "Run r2" in content

    def test_section_heading_not_duplicated(self, tmp_path: Path) -> None:
        w = _writer(tmp_path)
        w.append_learning("r1", "A", _default_before(), _default_after(), "kept")
        w.append_learning("r2", "B", _default_before(), _default_after(), "kept")
        content = (tmp_path / "CLAUDE.md").read_text()
        assert content.count(_SECTION_HEADING) == 1

    def test_five_sequential_appends(self, tmp_path: Path) -> None:
        w = _writer(tmp_path)
        for i in range(1, 6):
            w.append_learning(f"r{i}", f"Change {i}", _default_before(), _default_after(), "kept")
        learnings = w.read_learnings()
        assert len(learnings) == 5
        assert [e["run_id"] for e in learnings] == [f"r{i}" for i in range(1, 6)]


# ---------------------------------------------------------------------------
# 3. append_learning — preserves other content
# ---------------------------------------------------------------------------

class TestPreservesContent:
    def test_existing_content_preserved(self, tmp_path: Path) -> None:
        path = tmp_path / "CLAUDE.md"
        path.write_text("# My Project\n\nSome setup notes.\n", encoding="utf-8")
        w = ClaudeMdWriter(str(path))
        w.append_learning("r1", "desc", _default_before(), _default_after(), "kept")
        content = path.read_text()
        assert "# My Project" in content
        assert "Some setup notes." in content

    def test_subsequent_headings_preserved(self, tmp_path: Path) -> None:
        path = tmp_path / "CLAUDE.md"
        initial = (
            "# Project\n\n"
            "## Strategy Learnings\n"
            "\n"
            "## Other Section\n"
            "Important stuff here.\n"
        )
        path.write_text(initial, encoding="utf-8")
        w = ClaudeMdWriter(str(path))
        w.append_learning("r1", "change", _default_before(), _default_after(), "kept")
        content = path.read_text()
        assert "## Other Section" in content
        assert "Important stuff here." in content

    def test_learning_inserted_before_next_section(self, tmp_path: Path) -> None:
        path = tmp_path / "CLAUDE.md"
        initial = (
            "## Strategy Learnings\n"
            "\n"
            "## Notes\n"
            "keep me\n"
        )
        path.write_text(initial, encoding="utf-8")
        w = ClaudeMdWriter(str(path))
        w.append_learning("r1", "change", _default_before(), _default_after(), "kept")
        content = path.read_text()
        learning_pos = content.index("Run r1")
        notes_pos = content.index("## Notes")
        assert learning_pos < notes_pos


# ---------------------------------------------------------------------------
# 4. read_learnings — correct parsing
# ---------------------------------------------------------------------------

class TestReadLearnings:
    def test_round_trip_single_entry(self, tmp_path: Path) -> None:
        w = _writer(tmp_path)
        w.append_learning("run_007", "Raised TP to 2R", _default_before(), _default_after(), "kept")
        results = w.read_learnings()
        assert len(results) == 1
        assert results[0]["run_id"] == "run_007"
        assert results[0]["verdict"] == "kept"

    def test_round_trip_changes_made(self, tmp_path: Path) -> None:
        w = _writer(tmp_path)
        w.append_learning("r1", "Tightened SL to 1.2R", _default_before(), _default_after(), "reverted")
        results = w.read_learnings()
        assert "Tightened SL to 1.2R" in results[0]["changes_made"]

    def test_multiple_entries_ordered(self, tmp_path: Path) -> None:
        w = _writer(tmp_path)
        w.append_learning("r1", "First", _default_before(), _default_after(), "kept")
        w.append_learning("r2", "Second", _default_before(), _default_after(), "reverted")
        results = w.read_learnings()
        assert results[0]["run_id"] == "r1"
        assert results[1]["run_id"] == "r2"

    def test_verdict_kept_parsed(self, tmp_path: Path) -> None:
        w = _writer(tmp_path)
        w.append_learning("r1", "c", _default_before(), _default_after(), "kept")
        assert w.read_learnings()[0]["verdict"] == "kept"

    def test_verdict_reverted_parsed(self, tmp_path: Path) -> None:
        w = _writer(tmp_path)
        w.append_learning("r1", "c", _default_before(), _default_after(), "reverted")
        assert w.read_learnings()[0]["verdict"] == "reverted"


# ---------------------------------------------------------------------------
# 5. read_learnings — edge cases
# ---------------------------------------------------------------------------

class TestReadLearningsEdgeCases:
    def test_empty_section_returns_empty_list(self, tmp_path: Path) -> None:
        path = tmp_path / "CLAUDE.md"
        path.write_text("## Strategy Learnings\n\n", encoding="utf-8")
        w = ClaudeMdWriter(str(path))
        assert w.read_learnings() == []

    def test_no_section_returns_empty_list(self, tmp_path: Path) -> None:
        path = tmp_path / "CLAUDE.md"
        path.write_text("# Just a header\n\nSome text.\n", encoding="utf-8")
        w = ClaudeMdWriter(str(path))
        assert w.read_learnings() == []

    def test_missing_file_returns_empty_list(self, tmp_path: Path) -> None:
        w = ClaudeMdWriter(str(tmp_path / "nonexistent.md"))
        assert w.read_learnings() == []

    def test_malformed_line_skipped_gracefully(self, tmp_path: Path) -> None:
        path = tmp_path / "CLAUDE.md"
        path.write_text(
            "## Strategy Learnings\n"
            "- this line is malformed and has no proper format\n",
            encoding="utf-8",
        )
        w = ClaudeMdWriter(str(path))
        # Should not raise; malformed lines are silently skipped
        results = w.read_learnings()
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# 6. get_learnings_summary
# ---------------------------------------------------------------------------

class TestGetLearningsSummary:
    def test_empty_returns_no_learnings_message(self, tmp_path: Path) -> None:
        w = _writer(tmp_path)
        summary = w.get_learnings_summary()
        assert "No strategy learnings" in summary

    def test_summary_contains_run_ids(self, tmp_path: Path) -> None:
        w = _writer(tmp_path)
        w.append_learning("run_01", "Changed X", _default_before(), _default_after(), "kept")
        w.append_learning("run_02", "Changed Y", _default_before(), _default_after(), "reverted")
        summary = w.get_learnings_summary()
        assert "run_01" in summary
        assert "run_02" in summary

    def test_summary_is_human_readable_string(self, tmp_path: Path) -> None:
        w = _writer(tmp_path)
        w.append_learning("r1", "Test", _default_before(), _default_after(), "kept")
        summary = w.get_learnings_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0


# ---------------------------------------------------------------------------
# 7. Metric delta formatting
# ---------------------------------------------------------------------------

class TestMetricFormatting:
    def test_sharpe_delta_in_content(self, tmp_path: Path) -> None:
        w = _writer(tmp_path)
        w.append_learning("r1", "desc", {"sharpe": -0.5}, {"sharpe": 1.2}, "kept")
        content = (tmp_path / "CLAUDE.md").read_text()
        assert "Sharpe" in content
        assert "-0.50" in content
        assert "1.20" in content

    def test_win_rate_delta_formatted_as_percent(self, tmp_path: Path) -> None:
        w = _writer(tmp_path)
        w.append_learning("r1", "desc", {"win_rate": 0.42}, {"win_rate": 0.55}, "kept")
        content = (tmp_path / "CLAUDE.md").read_text()
        assert "WinRate" in content
        assert "42.0%" in content
        assert "55.0%" in content

    def test_max_dd_delta_formatted_as_percent(self, tmp_path: Path) -> None:
        w = _writer(tmp_path)
        w.append_learning("r1", "desc", {"max_dd": 0.08}, {"max_dd": 0.05}, "kept")
        content = (tmp_path / "CLAUDE.md").read_text()
        assert "MaxDD" in content

    def test_partial_metrics_some_none(self, tmp_path: Path) -> None:
        """Only sharpe provided — should not crash or emit WinRate/MaxDD."""
        w = _writer(tmp_path)
        w.append_learning("r1", "desc", {"sharpe": 0.3}, {"sharpe": 0.7}, "kept")
        content = (tmp_path / "CLAUDE.md").read_text()
        assert "Sharpe" in content
        assert "WinRate" not in content
        assert "MaxDD" not in content

    def test_empty_metrics_uses_fallback(self, tmp_path: Path) -> None:
        """No metric keys at all → falls back to 'no metrics'."""
        w = _writer(tmp_path)
        w.append_learning("r1", "desc", {}, {}, "kept")
        content = (tmp_path / "CLAUDE.md").read_text()
        assert "no metrics" in content

    def test_empty_changes_made_handled(self, tmp_path: Path) -> None:
        """Blank changes_made replaced by placeholder."""
        w = _writer(tmp_path)
        w.append_learning("r1", "   ", _default_before(), _default_after(), "kept")
        content = (tmp_path / "CLAUDE.md").read_text()
        assert "(no changes)" in content
