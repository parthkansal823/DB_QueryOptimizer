import os
import threading
from contextlib import contextmanager

from psycopg2 import pool

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/lqo",
)

# The pool is built on first use rather than at import.
#
# Connecting at import time made importing *any* module that touches the
# database require a running Postgres. Two things broke as a result. Pure unit
# tests could not even be collected -- `test_train_aggregation.py` exercises a
# list-of-dicts function with no database in sight, but importing `app.train`
# pulled in `app.db` and hung on a TCP timeout. And the API could not survive a
# cold start: `depends_on` in docker-compose waits for the Postgres *container*,
# not for Postgres to accept connections, so the backend could import-crash in
# the gap and never retry.
#
# Deferring construction fixes both. The first query pays the connection cost,
# which is where it belongs.
_pool: pool.SimpleConnectionPool | None = None
_pool_lock = threading.Lock()

MIN_CONNECTIONS = int(os.getenv("DB_POOL_MIN", "1"))
MAX_CONNECTIONS = int(os.getenv("DB_POOL_MAX", "10"))


def get_pool() -> pool.SimpleConnectionPool:
    """The process-wide connection pool, created on first use."""
    global _pool
    if _pool is None:
        # Double-checked under a lock: FastAPI serves requests from a thread
        # pool, so two concurrent first-requests could otherwise each build a
        # pool and leak one of them.
        with _pool_lock:
            if _pool is None:
                _pool = pool.SimpleConnectionPool(
                    MIN_CONNECTIONS, MAX_CONNECTIONS, DATABASE_URL
                )
    return _pool


def reset_pool() -> None:
    """Drop the pool so the next call reconnects. Used by tests."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.closeall()
        _pool = None


@contextmanager
def get_cursor():
    """Yield a cursor from the pool; commits on success, rolls back on error."""
    active = get_pool()
    conn = active.getconn()
    try:
        cur = conn.cursor()
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        active.putconn(conn)
