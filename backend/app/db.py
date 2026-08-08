import os
from contextlib import contextmanager

from psycopg2 import pool

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/lqo",
)

_pool = pool.SimpleConnectionPool(1, 10, DATABASE_URL)


@contextmanager
def get_cursor():
    """Yield a cursor from the pool; commits on success, rolls back on error."""
    conn = _pool.getconn()
    try:
        cur = conn.cursor()
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)
