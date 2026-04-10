from __future__ import annotations
import pytest
from src.indicators.kihon_suchi import is_kihon_number, project_from_pivot, KIHON_NUMBERS


class TestKihonNumbers:
    def test_known_numbers(self):
        for n in [9, 17, 26, 33, 42]:
            assert is_kihon_number(n, tolerance=0)

    def test_tolerance(self):
        assert is_kihon_number(25, tolerance=1) is True
        assert is_kihon_number(27, tolerance=1) is True
        assert is_kihon_number(28, tolerance=1) is False

    def test_non_kihon_rejected(self):
        assert is_kihon_number(15, tolerance=1) is False
        assert is_kihon_number(30, tolerance=1) is False


class TestProjection:
    def test_projects_all_numbers(self):
        targets = project_from_pivot(pivot_bar=50, total_bars=300)
        assert len(targets) > 0
        assert all(t['target_bar'] > 50 for t in targets)

    def test_projects_within_range(self):
        targets = project_from_pivot(pivot_bar=50, total_bars=100)
        assert all(t['target_bar'] < 100 for t in targets)
