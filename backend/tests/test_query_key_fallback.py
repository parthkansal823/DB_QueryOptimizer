r"""
`QUERY_KEY_SQL` is not `query_fingerprint`, and the docstring used to say it was.

`docs/WRITEUP.md` 2.78. Two definitions of "which query is this":

    python:  query_template(sql) if GROUP_BY_TEMPLATE else whitespace(sql)
    sql:     regexp_replace(btrim(sql_text), '\\s+', ' ', 'g')

The SQL reproduces one of the two branches, and `GROUP_BY_TEMPLATE` defaults to
the other one.

`COALESCE` prefers the recorded `query_id`, so the fallback only decides the
grouping of rows that have none -- currently 0 of 33,634. These tests pin the
difference rather than paper over it: reproducing `query_template` in SQL would
be a second implementation of query identity, and this project's last several
findings were all pairs of definitions that had to agree and stopped.
"""

import re

import pytest

from app.logging_store import QUERY_KEY_SQL, query_fingerprint
from app.query_template import query_template

_WS = re.compile(r"\s+")


def sql_normalises(text: str) -> str:
    """What `QUERY_KEY_SQL` does to `sql_text`, in Python.

    Kept beside the assertion that the SQL still says this, so the two cannot
    drift without a test failing.
    """
    return _WS.sub(" ", text.strip(" \t\r\n"))


def test_the_sql_still_does_what_this_file_thinks():
    """If `QUERY_KEY_SQL` changes, the helper above is a lie and every
    expectation below is meaningless."""
    assert "regexp_replace" in QUERY_KEY_SQL
    assert r"'\s+', ' ', 'g'" in QUERY_KEY_SQL
    assert "btrim" in QUERY_KEY_SQL
    assert "COALESCE(query_id" in QUERY_KEY_SQL


def test_the_recorded_id_wins_when_there_is_one():
    """Which is why the divergence is latent rather than live."""
    assert QUERY_KEY_SQL.startswith("COALESCE(query_id")


class TestWhereTheyDisagree:
    """Pinned, so a future reader finds a decision rather than a surprise."""

    def test_the_same_template_with_different_literals(self):
        a = "SELECT * FROM users WHERE city = 'Mumbai'"
        b = "SELECT * FROM users WHERE city = 'Delhi'"

        assert query_template(a) == query_template(b)
        assert sql_normalises(a) != sql_normalises(b)

    def test_a_hinted_query_against_its_bare_form(self):
        """The workflow that produced the finding `query_template` exists for:
        feeding the optimizer's own output back in split one query's history
        into three ids across three passes."""
        bare = "SELECT * FROM users u JOIN orders o ON o.user_id = u.id"
        hinted = f"/*+ Leading(u o) */\n{bare}"

        assert query_template(bare) == query_template(hinted)
        assert sql_normalises(bare) != sql_normalises(hinted)


class TestWhereTheyAgree:
    def test_whitespace_normalisation(self):
        a, b = "SELECT  1", "SELECT 1"

        assert query_template(a) == query_template(b)
        assert sql_normalises(a) == sql_normalises(b)

    def test_leading_and_trailing_space(self):
        a, b = "  SELECT 1  ", "SELECT 1"

        assert query_template(a) == query_template(b)
        assert sql_normalises(a) == sql_normalises(b)


class TestTheClaimIsGone:
    """The docstring said the SQL 'reproduces this same id'. It does not, and a
    false comment is worse than none -- it is the reason nobody checked."""

    def test_the_fingerprint_docstring_does_not_claim_equivalence(self):
        assert query_fingerprint.__doc__ is not None
        assert "reproduces this same" not in query_fingerprint.__doc__

    def test_the_sql_records_what_it_actually_reproduces(self):
        import inspect

        from app import logging_store

        source = inspect.getsource(logging_store)
        assert "GROUP_BY_TEMPLATE = False` branch" in source, (
            "the fallback has to say which branch it implements"
        )


@pytest.mark.parametrize("sql", [
    "SELECT 1",
    "SELECT * FROM users WHERE city = 'Mumbai'",
    "/*+ Leading(u o) */ SELECT * FROM users u",
])
def test_the_fingerprint_is_stable(sql):
    """Whatever the rule is, it has to be a function."""
    assert query_fingerprint(sql) == query_fingerprint(sql)
