"""
Database connection pooling and session management.

Uses psycopg2's ThreadedConnectionPool so that multiple threads (e.g. the
live data feed thread and the decision engine thread) can each hold a
connection without contention.

Usage
-----
# Initialise once at application startup
pool = DatabasePool(config)
pool.initialise()

# In each worker
with pool.get_connection() as conn:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM trades WHERE id = %s", (trade_id,))
        row = cur.fetchone()

# Shutdown
pool.close()

Configuration keys
------------------
DB_HOST     : str   — PostgreSQL host          (default: localhost)
DB_PORT     : int   — PostgreSQL port          (default: 5432)
DB_NAME     : str   — Database name            (default: trade_agent)
DB_USER     : str   — Database user            (default: postgres)
DB_PASSWORD : str   — Database password        (default: "")
DB_MIN_CONN : int   — Minimum pool connections (default: 2)
DB_MAX_CONN : int   — Maximum pool connections (default: 10)
DB_CONNECT_TIMEOUT : int — Connection timeout in seconds (default: 10)
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DBConfig:
    """
    Holds all parameters required to open a PostgreSQL connection.
    Values are read from environment variables when not explicitly provided.
    """
    host:            str = field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    port:            int = field(default_factory=lambda: int(os.getenv("DB_PORT", "5432")))
    dbname:          str = field(default_factory=lambda: os.getenv("DB_NAME", "trade_agent"))
    user:            str = field(default_factory=lambda: os.getenv("DB_USER", "postgres"))
    password:        str = field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))
    min_connections: int = field(default_factory=lambda: int(os.getenv("DB_MIN_CONN", "2")))
    max_connections: int = field(default_factory=lambda: int(os.getenv("DB_MAX_CONN", "10")))
    connect_timeout: int = field(default_factory=lambda: int(os.getenv("DB_CONNECT_TIMEOUT", "10")))

    def dsn(self) -> str:
        """Return a libpq-compatible DSN string."""
        return (
            f"host={self.host} "
            f"port={self.port} "
            f"dbname={self.dbname} "
            f"user={self.user} "
            f"password={self.password} "
            f"connect_timeout={self.connect_timeout}"
        )


# =============================================================================
# Pool
# =============================================================================

class DatabasePool:
    """
    Wraps psycopg2's ThreadedConnectionPool with convenience helpers.

    Thread-safety: getconn/putconn are internally serialised by psycopg2.
    The context manager ensures connections are always returned to the pool,
    even when an exception is raised inside the with-block.
    """

    def __init__(self, config: Optional[DBConfig] = None) -> None:
        self._config: DBConfig = config or DBConfig()
        self._pool:   Optional[ThreadedConnectionPool] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialise(self) -> None:
        """Open the connection pool. Call once at application startup."""
        if self._pool is not None:
            logger.warning("DatabasePool.initialise() called on an already-open pool — skipping")
            return

        logger.info(
            "Opening connection pool to %s:%s/%s (min=%d, max=%d)",
            self._config.host,
            self._config.port,
            self._config.dbname,
            self._config.min_connections,
            self._config.max_connections,
        )
        self._pool = ThreadedConnectionPool(
            self._config.min_connections,
            self._config.max_connections,
            self._config.dsn(),
        )
        logger.info("Connection pool ready")

    def close(self) -> None:
        """Close all connections and release the pool."""
        if self._pool is not None:
            self._pool.closeall()
            self._pool = None
            logger.info("Connection pool closed")

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------

    @contextmanager
    def get_connection(self) -> Generator:
        """
        Context manager that yields a live psycopg2 connection.

        The connection is returned to the pool on exit.  If the block raises
        an exception the connection is rolled back before being returned so
        that it is left in a clean state for the next caller.

        Example
        -------
        with pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT 1")
        """
        if self._pool is None:
            raise RuntimeError(
                "DatabasePool has not been initialised — call pool.initialise() first"
            )

        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    @contextmanager
    def get_cursor(self, cursor_factory=RealDictCursor) -> Generator:
        """
        Shorthand context manager that yields a cursor directly.

        The parent connection is committed (or rolled back) and returned to
        the pool automatically.

        Example
        -------
        with pool.get_cursor() as cur:
            cur.execute("SELECT * FROM trades LIMIT 10")
            rows = cur.fetchall()
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=cursor_factory) as cur:
                yield cur

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """
        Perform a lightweight query to verify the pool is operational.

        Returns True when the database responds correctly, False otherwise.
        Does not raise — callers should treat False as a connectivity fault.
        """
        try:
            with self.get_cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
            return result is not None
        except Exception as exc:
            logger.error("Database health check failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        """True when the pool has been initialised and not yet closed."""
        return self._pool is not None

    @property
    def config(self) -> DBConfig:
        return self._config


# =============================================================================
# Module-level singleton (optional convenience)
# =============================================================================
# Modules that prefer a global pool can import and use `get_connection` /
# `get_cursor` directly without managing a pool instance themselves.

_default_pool: Optional[DatabasePool] = None


def init_default_pool(config: Optional[DBConfig] = None) -> DatabasePool:
    """Initialise the module-level default pool.  Idempotent."""
    global _default_pool
    if _default_pool is None:
        _default_pool = DatabasePool(config)
        _default_pool.initialise()
    return _default_pool


def get_connection():
    """Context manager for a connection from the default pool."""
    if _default_pool is None:
        raise RuntimeError("Default pool not initialised — call init_default_pool() first")
    return _default_pool.get_connection()


def get_cursor(cursor_factory=RealDictCursor):
    """Context manager for a cursor from the default pool."""
    if _default_pool is None:
        raise RuntimeError("Default pool not initialised — call init_default_pool() first")
    return _default_pool.get_cursor(cursor_factory=cursor_factory)


def close_default_pool() -> None:
    """Close the module-level pool if it is open."""
    global _default_pool
    if _default_pool is not None:
        _default_pool.close()
        _default_pool = None
