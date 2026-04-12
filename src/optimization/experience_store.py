"""pgvector-backed store for optimization trial results.

Persists trials into the ``optimization_runs`` table and queries for
similar past successes/failures using cosine similarity on the 20-dim
market context embedding.  Also manages the ``proven_configs`` table
for graduated, statistically validated parameter sets.

Usage
-----
from src.database.connection import DatabasePool
from src.optimization.experience_store import ExperienceStore

pool = DatabasePool(config)
pool.initialise()
store = ExperienceStore(pool)

run_id = store.persist_trial(
    instrument="MGC", data_start=..., data_end=...,
    market_embedding=mkt_vec, params_embedding=prm_vec,
    outcome_embedding=out_vec,
    full_params={...}, active_strategies=["sss"],
    outcome={...}, passed_combine=True, epoch=5,
)
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── helpers ─────────────────────────────────────────────────────────


def _vec_to_str(vec: np.ndarray) -> str:
    """Format a numpy vector as a pgvector literal string.

    Example: ``[0.100000,0.200000,0.300000]``
    """
    return "[" + ",".join(f"{v:.6f}" for v in vec) + "]"


# ── ExperienceStore ────────────────────────────────────────────────


class ExperienceStore:
    """pgvector-backed store for optimization trial results.

    All public methods obtain a cursor from the pool's ``get_cursor``
    context manager, which auto-commits on clean exit and rolls back
    on exception.
    """

    def __init__(self, db_pool) -> None:
        self._pool = db_pool

    # ── persist ─────────────────────────────────────────────────────

    def persist_trial(
        self,
        instrument: str,
        data_start: datetime,
        data_end: datetime,
        market_embedding: np.ndarray,
        params_embedding: np.ndarray,
        outcome_embedding: np.ndarray,
        full_params: dict,
        active_strategies: list[str],
        outcome: dict,
        passed_combine: bool = False,
        passed_permutation: bool = False,
        epoch: int = 0,
    ) -> uuid.UUID:
        """Insert one trial into ``optimization_runs``. Returns the run_id."""
        sql = """
            INSERT INTO optimization_runs (
                instrument, timestamp, data_start, data_end,
                market_embedding, params_embedding, outcome_embedding,
                full_params, active_strategies, outcome,
                passed_combine, passed_permutation, epoch
            ) VALUES (
                %s, %s, %s, %s,
                %s::vector, %s::vector, %s::vector,
                %s, %s, %s,
                %s, %s, %s
            )
            RETURNING run_id
        """
        now = datetime.now(timezone.utc)
        params = (
            instrument,
            now,
            data_start,
            data_end,
            _vec_to_str(market_embedding),
            _vec_to_str(params_embedding),
            _vec_to_str(outcome_embedding),
            json.dumps(full_params),
            active_strategies,
            json.dumps(outcome),
            passed_combine,
            passed_permutation,
            epoch,
        )

        with self._pool.get_cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()

        run_id = row["run_id"]
        logger.info("Persisted trial %s for %s (epoch %d)", run_id, instrument, epoch)
        return run_id

    # ── similarity queries ──────────────────────────────────────────

    def find_similar_successes(
        self,
        market_embedding: np.ndarray,
        instrument: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Find proven trials with similar market context via cosine similarity.

        Returns rows ordered by descending similarity (highest first).
        Only trials where ``passed_combine = TRUE`` are returned.
        """
        vec_str = _vec_to_str(market_embedding)

        if instrument is not None:
            sql = """
                SELECT run_id, instrument, full_params, outcome, epoch,
                       1 - (market_embedding <=> %s::vector) AS similarity
                FROM optimization_runs
                WHERE passed_combine = TRUE
                  AND instrument = %s
                ORDER BY market_embedding <=> %s::vector
                LIMIT %s
            """
            params = (vec_str, instrument, vec_str, limit)
        else:
            sql = """
                SELECT run_id, instrument, full_params, outcome, epoch,
                       1 - (market_embedding <=> %s::vector) AS similarity
                FROM optimization_runs
                WHERE passed_combine = TRUE
                ORDER BY market_embedding <=> %s::vector
                LIMIT %s
            """
            params = (vec_str, vec_str, limit)

        with self._pool.get_cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        results = [dict(r) for r in rows]
        logger.debug(
            "Found %d similar successes (instrument=%s)", len(results), instrument
        )
        return results

    def find_similar_failures(
        self,
        market_embedding: np.ndarray,
        min_similarity: float = 0.8,
        limit: int = 20,
    ) -> list[dict]:
        """Find failed trials in similar markets to avoid.

        Returns rows where ``passed_combine = FALSE`` and cosine
        similarity >= *min_similarity*, ordered by descending similarity.
        """
        vec_str = _vec_to_str(market_embedding)
        # cosine distance threshold: 1 - min_similarity
        max_distance = 1.0 - min_similarity

        sql = """
            SELECT run_id, instrument, full_params, outcome,
                   1 - (market_embedding <=> %s::vector) AS similarity
            FROM optimization_runs
            WHERE passed_combine = FALSE
              AND (market_embedding <=> %s::vector) <= %s
            ORDER BY market_embedding <=> %s::vector
            LIMIT %s
        """
        params = (vec_str, vec_str, max_distance, vec_str, limit)

        with self._pool.get_cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        results = [dict(r) for r in rows]
        logger.debug(
            "Found %d similar failures (min_sim=%.2f)", len(results), min_similarity
        )
        return results

    # ── proven configs ──────────────────────────────────────────────

    def save_proven_config(
        self,
        instrument: str,
        run_id: uuid.UUID,
        params: dict,
        win_rate: float,
        total_return_pct: float,
        p_value: float,
        combine_passes: int,
        data_start: datetime,
        data_end: datetime,
    ) -> int:
        """Save a proven config, superseding any previous active config.

        Steps:
        1. UPDATE all active configs for this instrument, setting
           ``active = FALSE`` and ``superseded_at = now``.
        2. INSERT the new proven config with ``active = TRUE``.

        Returns the new row's ``id``.
        """
        now = datetime.now(timezone.utc)

        supersede_sql = """
            UPDATE proven_configs
            SET active = FALSE, superseded_at = %s
            WHERE instrument = %s AND active = TRUE
        """

        insert_sql = """
            INSERT INTO proven_configs (
                instrument, timestamp, run_id,
                params, win_rate, total_return_pct, p_value,
                combine_passes, data_start, data_end, active
            ) VALUES (
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, TRUE
            )
            RETURNING id
        """

        with self._pool.get_cursor() as cur:
            # Step 1: supersede old
            cur.execute(supersede_sql, (now, instrument))

            # Step 2: insert new
            cur.execute(
                insert_sql,
                (
                    instrument,
                    now,
                    str(run_id),
                    json.dumps(params),
                    win_rate,
                    total_return_pct,
                    p_value,
                    combine_passes,
                    data_start,
                    data_end,
                ),
            )
            row = cur.fetchone()

        new_id = row["id"]
        logger.info(
            "Saved proven config #%d for %s (run_id=%s, win_rate=%.2f)",
            new_id,
            instrument,
            run_id,
            win_rate,
        )
        return new_id

    def get_proven_config(self, instrument: str) -> dict | None:
        """Get the active proven config for an instrument, or None."""
        sql = """
            SELECT id, instrument, timestamp, run_id,
                   params, win_rate, total_return_pct, p_value,
                   combine_passes, data_start, data_end, active
            FROM proven_configs
            WHERE instrument = %s AND active = TRUE
            ORDER BY timestamp DESC
            LIMIT 1
        """
        with self._pool.get_cursor() as cur:
            cur.execute(sql, (instrument,))
            row = cur.fetchone()

        if row is None:
            return None
        return dict(row)

    # ── stats ───────────────────────────────────────────────────────

    def count_trials(self, instrument: str | None = None) -> int:
        """Count total trials, optionally filtered by instrument."""
        if instrument is not None:
            sql = "SELECT COUNT(*) AS count FROM optimization_runs WHERE instrument = %s"
            params = (instrument,)
        else:
            sql = "SELECT COUNT(*) AS count FROM optimization_runs"
            params = ()

        with self._pool.get_cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()

        return row["count"]
