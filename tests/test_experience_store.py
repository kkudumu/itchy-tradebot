"""Tests for the pgvector-backed experience store (mocked DB)."""

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from src.optimization.experience_store import ExperienceStore, _vec_to_str


# ── helpers ─────────────────────────────────────────────────────────


def _make_pool():
    """Return a MagicMock that mimics DatabasePool.get_cursor()."""
    pool = MagicMock()
    cursor = MagicMock()
    # pool.get_cursor() is a context manager yielding cursor
    pool.get_cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    pool.get_cursor.return_value.__exit__ = MagicMock(return_value=False)
    return pool, cursor


def _rand_vec(dim: int) -> np.ndarray:
    return np.random.default_rng(42).random(dim).astype(np.float64)


# ── _vec_to_str helper ─────────────────────────────────────────────


class TestVecToStr:
    def test_formats_correctly(self):
        vec = np.array([0.1, 0.2, 0.3])
        result = _vec_to_str(vec)
        assert result.startswith("[")
        assert result.endswith("]")
        assert "0.100000" in result
        assert "0.200000" in result
        assert "0.300000" in result

    def test_round_trip_shape(self):
        vec = _rand_vec(20)
        result = _vec_to_str(vec)
        # Should have 20 comma-separated values
        parts = result.strip("[]").split(",")
        assert len(parts) == 20


# ── persist_trial ──────────────────────────────────────────────────


class TestPersistTrial:
    def test_inserts_and_returns_uuid(self):
        pool, cursor = _make_pool()
        expected_id = uuid.uuid4()
        cursor.fetchone.return_value = {"run_id": expected_id}

        store = ExperienceStore(pool)
        result = store.persist_trial(
            instrument="MGC",
            data_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
            data_end=datetime(2026, 3, 1, tzinfo=timezone.utc),
            market_embedding=_rand_vec(20),
            params_embedding=_rand_vec(24),
            outcome_embedding=_rand_vec(20),
            full_params={"risk": {"initial_risk_pct": 0.5}},
            active_strategies=["sss", "ichimoku"],
            outcome={"win_rate": 0.55, "total_return_pct": 8.2},
            passed_combine=True,
            passed_permutation=False,
            epoch=3,
        )

        assert result == expected_id
        cursor.execute.assert_called_once()
        sql = cursor.execute.call_args[0][0]
        assert "INSERT INTO optimization_runs" in sql
        assert "RETURNING run_id" in sql

    def test_passes_correct_param_count(self):
        pool, cursor = _make_pool()
        cursor.fetchone.return_value = {"run_id": uuid.uuid4()}

        store = ExperienceStore(pool)
        store.persist_trial(
            instrument="MNQ",
            data_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
            data_end=datetime(2026, 3, 1, tzinfo=timezone.utc),
            market_embedding=_rand_vec(20),
            params_embedding=_rand_vec(24),
            outcome_embedding=_rand_vec(20),
            full_params={},
            active_strategies=["sss"],
            outcome={},
            passed_combine=False,
            passed_permutation=False,
            epoch=0,
        )

        params = cursor.execute.call_args[0][1]
        # Should be a tuple with 13 elements (all columns including auto-generated timestamp)
        assert isinstance(params, tuple)
        assert len(params) == 13


# ── find_similar_successes ─────────────────────────────────────────


class TestFindSimilarSuccesses:
    def test_returns_parsed_rows(self):
        pool, cursor = _make_pool()
        fake_rows = [
            {
                "run_id": uuid.uuid4(),
                "instrument": "MGC",
                "full_params": {"risk": {}},
                "outcome": {"win_rate": 0.6},
                "similarity": 0.95,
                "epoch": 2,
            },
            {
                "run_id": uuid.uuid4(),
                "instrument": "MGC",
                "full_params": {"risk": {}},
                "outcome": {"win_rate": 0.5},
                "similarity": 0.88,
                "epoch": 1,
            },
        ]
        cursor.fetchall.return_value = fake_rows

        store = ExperienceStore(pool)
        results = store.find_similar_successes(
            market_embedding=_rand_vec(20),
            instrument="MGC",
            limit=10,
        )

        assert len(results) == 2
        assert results[0]["similarity"] == 0.95
        cursor.execute.assert_called_once()
        sql = cursor.execute.call_args[0][0]
        assert "<=>" in sql  # cosine distance operator
        assert "proven" in sql.lower() or "passed_combine" in sql.lower()

    def test_without_instrument_filter(self):
        pool, cursor = _make_pool()
        cursor.fetchall.return_value = []

        store = ExperienceStore(pool)
        results = store.find_similar_successes(
            market_embedding=_rand_vec(20),
            instrument=None,
            limit=5,
        )

        assert results == []
        sql = cursor.execute.call_args[0][0]
        # Without instrument, should not filter by instrument
        # The query should still work
        cursor.execute.assert_called_once()

    def test_limit_is_passed(self):
        pool, cursor = _make_pool()
        cursor.fetchall.return_value = []

        store = ExperienceStore(pool)
        store.find_similar_successes(
            market_embedding=_rand_vec(20),
            limit=7,
        )

        params = cursor.execute.call_args[0][1]
        # limit should appear in the params
        assert 7 in params


# ── find_similar_failures ──────────────────────────────────────────


class TestFindSimilarFailures:
    def test_returns_failed_trials(self):
        pool, cursor = _make_pool()
        fake_rows = [
            {
                "run_id": uuid.uuid4(),
                "instrument": "MNQ",
                "full_params": {"risk": {}},
                "outcome": {"win_rate": 0.2},
                "similarity": 0.92,
            },
        ]
        cursor.fetchall.return_value = fake_rows

        store = ExperienceStore(pool)
        results = store.find_similar_failures(
            market_embedding=_rand_vec(20),
            min_similarity=0.8,
            limit=20,
        )

        assert len(results) == 1
        sql = cursor.execute.call_args[0][0]
        assert "<=>" in sql
        # Should filter for failures (NOT passed_combine)
        assert "passed_combine" in sql.lower() or "false" in sql.lower()

    def test_min_similarity_filter(self):
        pool, cursor = _make_pool()
        cursor.fetchall.return_value = []

        store = ExperienceStore(pool)
        store.find_similar_failures(
            market_embedding=_rand_vec(20),
            min_similarity=0.9,
            limit=10,
        )

        params = cursor.execute.call_args[0][1]
        # min_similarity translates to max distance = 1 - 0.9 = 0.1
        # Should be in params somehow
        cursor.execute.assert_called_once()


# ── save_proven_config ─────────────────────────────────────────────


class TestSaveProvenConfig:
    def test_supersedes_old_and_inserts_new(self):
        pool, cursor = _make_pool()
        cursor.fetchone.return_value = {"id": 42}

        store = ExperienceStore(pool)
        run_id = uuid.uuid4()
        result = store.save_proven_config(
            instrument="MGC",
            run_id=run_id,
            params={"risk": {"initial_risk_pct": 0.5}},
            win_rate=0.55,
            total_return_pct=8.2,
            p_value=0.03,
            combine_passes=3,
            data_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
            data_end=datetime(2026, 3, 1, tzinfo=timezone.utc),
        )

        assert result == 42
        # Should have two execute calls: one UPDATE (supersede), one INSERT
        assert cursor.execute.call_count == 2

        # First call should be UPDATE (supersede)
        first_sql = cursor.execute.call_args_list[0][0][0]
        assert "UPDATE" in first_sql
        assert "superseded_at" in first_sql

        # Second call should be INSERT
        second_sql = cursor.execute.call_args_list[1][0][0]
        assert "INSERT INTO proven_configs" in second_sql
        assert "RETURNING id" in second_sql


# ── get_proven_config ──────────────────────────────────────────────


class TestGetProvenConfig:
    def test_returns_active_config(self):
        pool, cursor = _make_pool()
        expected = {
            "id": 5,
            "instrument": "MGC",
            "params": {"risk": {}},
            "win_rate": 0.55,
            "active": True,
        }
        cursor.fetchone.return_value = expected

        store = ExperienceStore(pool)
        result = store.get_proven_config("MGC")

        assert result == expected
        sql = cursor.execute.call_args[0][0]
        assert "proven_configs" in sql
        assert "active" in sql.lower()

    def test_returns_none_when_no_config(self):
        pool, cursor = _make_pool()
        cursor.fetchone.return_value = None

        store = ExperienceStore(pool)
        result = store.get_proven_config("UNKNOWN")

        assert result is None


# ── count_trials ───────────────────────────────────────────────────


class TestCountTrials:
    def test_returns_count_for_instrument(self):
        pool, cursor = _make_pool()
        cursor.fetchone.return_value = {"count": 42}

        store = ExperienceStore(pool)
        result = store.count_trials(instrument="MGC")

        assert result == 42
        sql = cursor.execute.call_args[0][0]
        assert "COUNT" in sql
        params = cursor.execute.call_args[0][1]
        assert "MGC" in params

    def test_returns_count_all(self):
        pool, cursor = _make_pool()
        cursor.fetchone.return_value = {"count": 100}

        store = ExperienceStore(pool)
        result = store.count_trials()

        assert result == 100

    def test_returns_zero_when_empty(self):
        pool, cursor = _make_pool()
        cursor.fetchone.return_value = {"count": 0}

        store = ExperienceStore(pool)
        result = store.count_trials()

        assert result == 0
