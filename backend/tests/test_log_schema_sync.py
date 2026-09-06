"""
The two definitions of `plan_execution_log` have to agree.

`logging_store.CREATE_LOG_TABLE_SQL` says it "Mirrors
postgres/init/03_logging.sql, which only runs for the container's default
database. Pointing the optimizer at a user's own database (see app.onboard)
means creating it there too, so keep the two in sync."

`docs/WRITEUP.md` 2.78 is what a "keep the two in sync" comment is worth when
nothing checks it: `query_fingerprint`'s said the same about `QUERY_KEY_SQL` and
they disagreed on two of three cases.

This pair *does* agree, checked rather than assumed. The consequence if it ever
stops: `app.onboard` and `db_profiles.activate` build the table on somebody
else's database from the Python copy, so a column or index the init script has
and the copy lacks exists in the bundled demo and nowhere else -- and every
reader of the log works here and fails there.
"""

import re
from pathlib import Path

import pytest

from app.logging_store import CREATE_LOG_TABLE_SQL, ensure_log_table

INIT_SQL = Path(__file__).resolve().parents[2] / "postgres" / "init" / "03_logging.sql"


def _columns(create_table_sql: str) -> list[tuple[str, str]]:
    """(name, type) for each column, comments and constraints dropped."""
    body = create_table_sql[create_table_sql.index("(") + 1 : create_table_sql.rindex(")")]
    out = []
    for raw in body.split("\n"):
        line = raw.split("--")[0].strip().rstrip(",")
        if not line or line.upper().startswith(
            ("PRIMARY", "CONSTRAINT", "UNIQUE", "FOREIGN", "CHECK")
        ):
            continue
        name, _, rest = line.partition(" ")
        out.append((name.strip('"').lower(), " ".join(rest.split()).upper()))
    return out


@pytest.fixture(scope="module")
def init_script() -> str:
    if not INIT_SQL.exists():
        pytest.skip(f"{INIT_SQL} not found")
    return INIT_SQL.read_text(encoding="utf-8")


def test_the_init_script_is_where_this_thinks_it_is(init_script):
    assert "CREATE TABLE IF NOT EXISTS plan_execution_log" in init_script


def test_the_columns_match(init_script):
    table = init_script[init_script.index("CREATE TABLE") : init_script.index(");") + 1]

    assert _columns(CREATE_LOG_TABLE_SQL) == _columns(table)


def test_the_column_order_matches_too(init_script):
    """Nothing depends on it today. `SELECT *` in `_load_rows` zips column names
    to values by position, so it would if anyone stopped using the cursor
    description."""
    table = init_script[init_script.index("CREATE TABLE") : init_script.index(");") + 1]

    assert [c for c, _ in _columns(CREATE_LOG_TABLE_SQL)] == [
        c for c, _ in _columns(table)
    ]


class TestTheIndexesExistOnBothPaths:
    """They are not in `CREATE_LOG_TABLE_SQL` -- `ensure_log_table` issues them
    separately -- so comparing only the CREATE TABLE strings would report a
    difference that is not there, and comparing nothing would miss one that is.

    Every reader of this table filters on `created_at`, groups by `query_id`, or
    both: `REGRESSION_SQL`, `FAMILY_REGRESSION_SQL`, `stats.RUNS_SQL`. The
    regression guard refreshes on every settings change and every server start.
    A user's log grows with their traffic.
    """

    def _indexed_columns(self, sql: str) -> set[str]:
        """Columns indexed, from either a .sql file or Python source.

        The quotes are stripped and whitespace collapsed first, because
        `ensure_log_table` builds its statements by adjacent-string
        concatenation -- `"CREATE INDEX ... _query_id "` then a newline then
        `"ON plan_execution_log (query_id)"` -- and a pattern that assumes one
        contiguous statement finds nothing there and reports it as a difference.
        """
        flat = " ".join(sql.replace('"', " ").replace("'", " ").split())
        return {
            c.strip().lower()
            for c in re.findall(
                r"CREATE INDEX(?: IF NOT EXISTS)?\s+\S+\s+ON\s+plan_execution_log\s*\(([^)]*)\)",
                flat,
                re.I,
            )
        }

    def test_the_init_script_indexes_both_columns(self, init_script):
        assert self._indexed_columns(init_script) == {"query_id", "created_at"}

    def test_ensure_log_table_creates_the_same_ones(self):
        import inspect

        source = inspect.getsource(ensure_log_table)

        assert self._indexed_columns(source) == {"query_id", "created_at"}

    def test_they_are_if_not_exists_on_both_paths(self, init_script):
        """`ensure_log_table` runs on every profile activation, not once, so a
        bare CREATE would fail the second time a user switched databases."""
        import inspect

        assert "IF NOT EXISTS" in CREATE_LOG_TABLE_SQL
        assert inspect.getsource(ensure_log_table).count("IF NOT EXISTS") == 2
        assert init_script.upper().count("IF NOT EXISTS") == 3


def test_the_comment_still_points_at_the_right_file():
    """A stale path in the note is how the check stops being possible."""
    import inspect

    from app import logging_store

    assert "postgres/init/03_logging.sql" in inspect.getsource(logging_store)
