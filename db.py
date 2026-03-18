"""
Shared Postgres connection pool for the RLM service.
All RLM modules should import get_conn / put_conn from here instead of calling psycopg2.connect() directly.
"""
from __future__ import annotations

import os
import logging
from contextlib import contextmanager
from typing import Generator

import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor  # noqa: F401 — re-exported for convenience

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "")

_pool: ThreadedConnectionPool | None = None


def _get_pool() -> ThreadedConnectionPool:
    global _pool
    if _pool is None:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL not set")
        _pool = ThreadedConnectionPool(minconn=1, maxconn=5, dsn=DATABASE_URL)
        logger.info("RLM Postgres pool initialised (minconn=1, maxconn=5)")
    return _pool


def get_conn():
    """Borrow a connection from the pool. Caller must call put_conn() when done."""
    return _get_pool().getconn()


def put_conn(conn) -> None:
    """Return a connection to the pool."""
    try:
        pool = _get_pool()
        pool.putconn(conn)
    except Exception:
        logger.warning("put_conn: failed to return connection to pool", exc_info=True)


@contextmanager
def db_conn() -> Generator:
    """Context manager: borrow → yield → return (or rollback + return on error)."""
    conn = get_conn()
    try:
        yield conn
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        put_conn(conn)
