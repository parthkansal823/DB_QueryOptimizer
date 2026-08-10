import pytest

from app import saved_queries

URL = "postgresql://user:secret@host:5432/testdb"


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(saved_queries, "QUERIES_DIR", str(tmp_path / "queries"))


class _FakeCursor:
    """
    Plans a query, or raises the way a real planner would.

    Only the EXPLAIN fails. Transaction control around it (`SAVEPOINT`,
    `RELEASE`, `ROLLBACK TO`) succeeds, because that is how a real cursor
    behaves -- a fake that failed those too would be rejecting statements
    Postgres accepts and would misreport the savepoint handling as broken.
    """

    def __init__(self, plan=None, error=None):
        self._plan = plan
        self._error = error

    def execute(self, sql, params=None):
        if self._error and sql.lstrip().upper().startswith("EXPLAIN"):
            raise RuntimeError(self._error)

    def fetchone(self):
        return ([{"Plan": self._plan}],)


def _plan(*tables, cost=100.0, rows=10):
    node = {"Node Type": "Hash Join", "Total Cost": cost, "Plan Rows": rows, "Plans": []}
    for table in tables:
        node["Plans"].append({"Node Type": "Seq Scan", "Relation Name": table, "Plans": []})
    return node


# -- what may be saved ------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM orders",
        "UPDATE orders SET id = 1",
        "INSERT INTO orders VALUES (1)",
        "DROP TABLE orders",
        "TRUNCATE orders",
        "select 1; drop table orders",
        # PostgreSQL's spelling of CREATE TABLE AS. It carries none of the
        # other keywords, so it read as an ordinary SELECT and would have been
        # executed under every hint variant, creating a table each run.
        "SELECT * INTO evil_copy FROM orders",
        "select o.id into backup from orders o",
    ],
)
def test_mutating_queries_are_refused(sql):
    """Saved queries are executed repeatedly, under many forced plans, as
    training data. A saved DELETE would be run over and over against the
    user's own tables."""
    assert saved_queries.is_read_only(sql) is False


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM orders",
        "SELECT o.id FROM orders o JOIN users u ON o.user_id = u.id",
        "WITH x AS (SELECT 1) SELECT * FROM x",
        "SELECT 'deleted' AS status FROM orders",  # the word, not the statement
    ],
)
def test_read_only_queries_are_allowed(sql):
    assert saved_queries.is_read_only(sql) is True


def test_a_failed_validation_leaves_the_transaction_usable():
    """
    A rejected EXPLAIN aborts the whole transaction, and catching the exception
    does not undo that -- every later statement on the cursor fails with
    InFailedSqlTransaction. Validating a query the user got wrong is the
    ordinary case here, so it must not poison the connection it ran on.
    """
    executed = []

    class _AbortingCursor:
        """Fails the EXPLAIN, and refuses everything afterwards unless the
        savepoint was rolled back -- the way Postgres actually behaves."""

        def __init__(self):
            self.aborted = False

        def execute(self, sql, params=None):
            executed.append(sql)
            if sql.startswith("ROLLBACK TO SAVEPOINT"):
                self.aborted = False
                return
            if self.aborted:
                raise RuntimeError("current transaction is aborted")
            if sql.startswith("EXPLAIN"):
                self.aborted = True
                raise RuntimeError('relation "nope" does not exist')

        def fetchone(self):
            return ([{"Plan": _plan("orders")}],)

    cur = _AbortingCursor()
    result = saved_queries.validate(cur, "SELECT * FROM nope")

    assert result["ok"] is False
    assert any(s.startswith("ROLLBACK TO SAVEPOINT") for s in executed)
    cur.execute("SELECT 1")  # would raise if the savepoint had not been rolled back


def test_a_successful_validation_releases_its_savepoint():
    """Left open, savepoints accumulate for the life of the transaction."""
    executed = []

    class _RecordingCursor(_FakeCursor):
        def execute(self, sql, params=None):
            executed.append(sql)

    result = saved_queries.validate(_RecordingCursor(plan=_plan("orders", "users")), "SELECT 1")

    assert result["ok"] is True
    assert any(s.startswith("RELEASE SAVEPOINT") for s in executed)


def test_a_hand_edited_entry_without_a_name_does_not_break_the_list():
    """The store is a plain JSON file people will edit."""
    import json, os
    os.makedirs(saved_queries.QUERIES_DIR, exist_ok=True)
    with open(saved_queries._path_for(URL), "w") as f:
        json.dump([{"sql": "SELECT 1"}], f)  # no "name"

    assert saved_queries.save_query(URL, "fine", "SELECT 2")
    assert saved_queries.delete_query(URL, "fine") is not None


def test_a_keyword_hidden_in_a_comment_does_not_smuggle_a_write_through():
    """Comments are stripped before the check, so a mutation cannot be hidden
    behind one -- nor can a hint comment be mistaken for a mutation."""
    assert saved_queries.is_read_only("SELECT 1 -- delete from orders") is True
    assert saved_queries.is_read_only("/*+ Leading(a b) */ SELECT 1") is True
    assert saved_queries.is_read_only("/* nothing */ DELETE FROM orders") is False


# -- validation -------------------------------------------------------------


def test_validate_reports_what_the_query_touches():
    result = saved_queries.validate(_FakeCursor(_plan("orders", "users")), "SELECT 1")

    assert result["ok"] is True
    assert result["tables"] == ["orders", "users"]
    assert result["joins_available"] is True


def test_a_single_table_query_is_flagged_as_having_no_join_to_optimize():
    """It can never produce a candidate, so training on it teaches nothing --
    worth saying before a twenty-minute collection run, not after."""
    result = saved_queries.validate(_FakeCursor(_plan("orders")), "SELECT * FROM orders")

    assert result["ok"] is True
    assert result["joins_available"] is False


def test_a_query_that_does_not_plan_is_rejected_with_the_planner_error():
    result = saved_queries.validate(
        _FakeCursor(error='relation "nope" does not exist'), "SELECT * FROM nope"
    )

    assert result["ok"] is False
    assert "does not exist" in result["error"]


def test_an_empty_query_is_rejected_without_touching_the_database():
    assert saved_queries.validate(_FakeCursor(error="should not be reached"), "   ")["ok"] is False


# -- storage ----------------------------------------------------------------


def test_saving_then_listing_round_trips():
    saved_queries.save_query(URL, "orders", "SELECT 1", "my description")
    stored = saved_queries.list_queries(URL)

    assert [q["name"] for q in stored] == ["orders"]
    assert stored[0]["description"] == "my description"


def test_saving_the_same_name_edits_rather_than_duplicating():
    saved_queries.save_query(URL, "q", "SELECT 1")
    saved_queries.save_query(URL, "q", "SELECT 2")

    stored = saved_queries.list_queries(URL)
    assert len(stored) == 1
    assert stored[0]["sql"] == "SELECT 2"


def test_query_ids_are_namespaced_away_from_the_builtin_workload():
    """`plan_execution_log` groups by id, so a user query named like a
    benchmark one would otherwise have their histories merged."""
    saved_queries.save_query(URL, "corr_city_country", "SELECT 1")
    assert saved_queries.list_queries(URL)[0]["id"] == "user:corr_city_country"


def test_queries_are_kept_per_database():
    """A query only means anything against the schema it was written for, so
    switching connections must not present queries referencing tables that are
    not there."""
    other = "postgresql://user:secret@host:5432/otherdb"
    saved_queries.save_query(URL, "only-here", "SELECT 1")

    assert saved_queries.list_queries(other) == []
    assert len(saved_queries.list_queries(URL)) == 1


def test_the_stored_filename_does_not_contain_credentials():
    saved_queries.save_query(URL, "q", "SELECT 1")
    assert "secret" not in saved_queries._path_for(URL)


def test_deleting_removes_only_the_named_query():
    saved_queries.save_query(URL, "keep", "SELECT 1")
    saved_queries.save_query(URL, "drop", "SELECT 2")

    remaining = saved_queries.delete_query(URL, "drop")
    assert [q["name"] for q in remaining] == ["keep"]


def test_a_query_needs_a_name():
    with pytest.raises(ValueError, match="needs a name"):
        saved_queries.save_query(URL, "  ", "SELECT 1")


def test_as_workload_is_the_shape_the_collector_consumes():
    saved_queries.save_query(URL, "q", "SELECT 1", "desc")
    assert saved_queries.as_workload(URL) == [{"id": "user:q", "sql": "SELECT 1"}]


def test_a_damaged_store_reads_as_empty_rather_than_crashing():
    import os
    os.makedirs(saved_queries.QUERIES_DIR, exist_ok=True)
    with open(saved_queries._path_for(URL), "w") as f:
        f.write("{not json")

    assert saved_queries.list_queries(URL) == []
