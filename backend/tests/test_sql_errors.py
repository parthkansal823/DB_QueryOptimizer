"""
Explaining a failed query, and correcting it only when that is safe.

A query with a typo used to come back as an HTTP 500 and a blank panel, while
the exception already carried everything needed to do much better. PostgreSQL
reports the offending token, an exact character position, and often the
correction itself.

The rule these tests pin is that **a wrong suggestion is worse than none**. A
reader who pastes a confidently-offered correction and gets a second error has
been sent backwards and has learned not to trust the first one, so every case
below where the evidence is weak asserts that nothing is suggested.
"""

import pytest

from app.sql_errors import MIN_SIMILARITY, explain


class FakeDiag:
    def __init__(self, primary, sqlstate=None, hint=None, detail=None, position=None):
        self.message_primary = primary
        self.sqlstate = sqlstate
        self.message_hint = hint
        self.message_detail = detail
        self.statement_position = position


class FakeError(Exception):
    def __init__(self, primary, **kw):
        super().__init__(primary)
        self.diag = FakeDiag(primary, **kw)


class TestPostgresOwnHint:
    """PostgreSQL computed it against the live catalogue; nothing here beats it."""

    def test_a_column_typo_is_corrected_from_the_hint(self):
        sql = "SELECT u.nmae FROM users u"
        err = explain(
            FakeError(
                'column "u.nmae" does not exist',
                sqlstate="42703",
                hint='Perhaps you meant to reference the column "u.name".',
                position=8,
            ),
            sql,
        )
        assert err.suggestion == "u.name"
        assert err.suggestion_source == "postgresql"
        assert err.suggested_sql == "SELECT u.name FROM users u"

    def test_a_qualifier_is_not_doubled(self):
        """`u.nmae` becomes `u.name`, never `u.u.name`."""
        err = explain(
            FakeError(
                'column "u.nmae" does not exist',
                sqlstate="42703",
                hint='Perhaps you meant to reference the column "u.name".',
                position=8,
            ),
            "SELECT u.nmae FROM users u",
        )
        assert "u.u." not in (err.suggested_sql or "")

    def test_the_hint_outranks_the_catalogue(self):
        """
        Given both, the server's answer wins -- it knows which tables are in
        scope for this particular query, and a name-similarity search does not.
        """
        err = explain(
            FakeError(
                'column "nmae" does not exist',
                sqlstate="42703",
                hint='Perhaps you meant to reference the column "u.name".',
                position=8,
            ),
            "SELECT nmae FROM users u",
            known_names=["name", "email", "orders"],
        )
        assert err.suggestion == "u.name"
        assert err.suggestion_source == "postgresql"


class TestCatalogFallback:
    """PostgreSQL rarely hints for relation names, so this fills that gap."""

    def test_a_table_typo_is_matched_against_the_catalogue(self):
        err = explain(
            FakeError('relation "userss" does not exist', sqlstate="42P01", position=15),
            "SELECT * FROM userss",
            known_names=["users", "orders", "products"],
        )
        assert err.suggestion == "users"
        assert err.suggestion_source == "catalog"
        assert err.suggested_sql == "SELECT * FROM users"

    def test_nothing_is_suggested_when_nothing_is_close(self):
        """The case that protects the reader from a confident wrong answer."""
        err = explain(
            FakeError('relation "zzzzzz" does not exist', sqlstate="42P01", position=15),
            "SELECT * FROM zzzzzz",
            known_names=["users", "orders", "products"],
        )
        assert err.suggestion is None
        assert err.suggested_sql is None

    def test_an_empty_catalogue_suggests_nothing(self):
        err = explain(
            FakeError('relation "userss" does not exist', sqlstate="42P01", position=15),
            "SELECT * FROM userss",
        )
        assert err.suggestion is None

    def test_the_similarity_bar_is_meaningful(self):
        assert 0.5 <= MIN_SIMILARITY <= 0.8


class TestPointingAtTheProblem:
    def test_the_excerpt_carries_a_caret_under_the_offending_token(self):
        err = explain(
            FakeError('syntax error at or near "FORM"', sqlstate="42601", position=15),
            "SELECT id, ts FORM events",
        )
        assert err.line == 1
        assert err.column == 15
        lines = err.excerpt.split("\n")
        assert lines[0] == "SELECT id, ts FORM events"
        assert lines[1].index("^") == 14

    def test_a_position_on_a_later_line_reports_that_line(self):
        sql = "SELECT id\nFROM events\nWHERE nmae = 1"
        position = sql.index("nmae") + 1
        err = explain(
            FakeError('column "nmae" does not exist', sqlstate="42703", position=position), sql
        )
        assert err.line == 3
        assert err.column == 7
        assert err.excerpt.startswith("WHERE nmae = 1")

    @pytest.mark.parametrize("position", [None, 0, -3, 10_000])
    def test_an_unusable_position_is_dropped_rather_than_trusted(self, position):
        err = explain(
            FakeError("something went wrong", sqlstate="42601", position=position),
            "SELECT 1",
        )
        assert err.excerpt is None
        assert err.line is None


class TestPlainLanguage:
    @pytest.mark.parametrize(
        "sqlstate,fragment",
        [
            ("42703", "not in any of the tables"),
            ("42P01", "does not exist in this database"),
            ("42601", "could not parse"),
            ("42702", "more than one"),
        ],
    )
    def test_known_states_get_an_explanation(self, sqlstate, fragment):
        err = explain(FakeError("...", sqlstate=sqlstate), "SELECT 1")
        assert fragment in err.explanation

    def test_an_unknown_state_keeps_postgresqls_own_wording(self):
        """
        A paraphrase of an error nobody anticipated is a chance to be wrong
        about it. PostgreSQL's message is always more accurate.
        """
        err = explain(FakeError("something specific and rare", sqlstate="XX999"), "SELECT 1")
        assert err.explanation is None
        assert err.message == "something specific and rare"


class TestSafety:
    def test_an_ambiguous_replacement_without_a_position_is_refused(self):
        """
        `id` appears three times and only one is wrong. Without a position there
        is no way to know which, so no corrected SQL is offered -- the
        suggestion itself still is, because naming the right column is useful
        even when the substitution cannot be made.
        """
        err = explain(
            FakeError(
                'column "id" does not exist',
                sqlstate="42703",
                hint='Perhaps you meant to reference the column "o.id".',
            ),
            "SELECT id FROM orders o WHERE id > 5 ORDER BY id",
        )
        assert err.suggestion == "o.id"
        assert err.suggested_sql is None

    def test_a_suggestion_identical_to_the_original_is_not_offered(self):
        err = explain(
            FakeError(
                'column "name" does not exist',
                sqlstate="42703",
                hint='Perhaps you meant to reference the column "name".',
                position=8,
            ),
            "SELECT name FROM users",
        )
        assert err.suggested_sql is None

    def test_an_exception_with_no_diagnostics_still_explains(self):
        """Not every failure arrives from psycopg2."""
        err = explain(ValueError("connection closed"), "SELECT 1")
        assert err.message == "connection closed"
        assert err.suggestion is None


class TestUnquotedSubjects:
    """
    PostgreSQL does not quote a *qualified* column reference.

    Measured on 18.6:

        SELECT u.nmae FROM users u   ->  column u.nmae does not exist
        SELECT * FROM userss         ->  relation "userss" does not exist

    The first version of this module looked only for a quoted token, so the
    qualified case -- by far the commoner way to mistype a column in a joined
    query -- produced a suggestion from PostgreSQL's own hint and then no
    corrected SQL to apply it to, because nothing had been identified to
    replace. Caught by running real queries at the server, not by rereading the
    regex.
    """

    def test_a_qualified_column_is_identified_without_quotes(self):
        err = explain(
            FakeError(
                "column u.nmae does not exist",
                sqlstate="42703",
                hint='Perhaps you meant to reference the column "u.name".',
                position=8,
            ),
            "SELECT u.nmae FROM users u LIMIT 5",
        )
        assert err.offending == "u.nmae"
        assert err.suggested_sql == "SELECT u.name FROM users u LIMIT 5"

    def test_an_unqualified_column_still_works(self):
        err = explain(
            FakeError("column nmae does not exist", sqlstate="42703", position=8),
            "SELECT nmae FROM users",
            known_names=["name", "email"],
        )
        assert err.offending == "nmae"
        assert err.suggested_sql == "SELECT name FROM users"

    def test_a_syntax_error_has_no_subject_to_replace(self):
        """
        `syntax error at or near "orders"` names where parsing stopped, not a
        thing that is wrong -- the mistake is usually the token *before* it.
        Offering `orders` as the offending identifier would point at innocent
        text, so nothing is claimed.
        """
        err = explain(
            FakeError('syntax error at or near "orders"', sqlstate="42601", position=15),
            "SELECT id, ts FORM orders",
            known_names=["orders", "users"],
        )
        assert err.suggested_sql is None


class TestPositionIsRebasedOntoTheUsersSql:
    """
    PostgreSQL's offset is into the statement *it* received, not the one the
    user wrote.

    `/query/optimize` wraps the query in `EXPLAIN (FORMAT JSON) `, and a
    candidate carries a `/*+ ... */` hint in front; both shift every offset
    after them. Measured: `SELECT u.nmae FROM users u LIMIT 5` reported position
    30 -- 8 in the user's text plus the 22 characters of the EXPLAIN prefix --
    and the caret landed under `users u`, confidently pointing at the one part
    of the line that was correct.
    """

    EXPLAIN_PREFIX = "EXPLAIN (FORMAT JSON) "

    def test_an_explain_prefix_does_not_move_the_caret(self):
        sql = "SELECT u.nmae FROM users u LIMIT 5"
        shifted = len(self.EXPLAIN_PREFIX) + sql.index("u.nmae") + 1
        err = explain(
            FakeError("column u.nmae does not exist", sqlstate="42703", position=shifted),
            sql,
        )
        assert err.column == sql.index("u.nmae") + 1
        assert err.excerpt.split("\n")[1].index("^") == sql.index("u.nmae")

    def test_a_correct_position_is_left_alone(self):
        sql = "SELECT u.nmae FROM users u"
        err = explain(
            FakeError("column u.nmae does not exist", sqlstate="42703", position=8),
            sql,
        )
        assert err.position == 8

    def test_an_ambiguous_token_points_nowhere_rather_than_somewhere_wrong(self):
        """
        Two occurrences and no reliable offset: a caret under the wrong one is
        exactly the failure this rebasing exists to prevent, so it points at
        neither. The message and suggestion still stand.
        """
        sql = "SELECT nmae FROM users WHERE nmae IS NOT NULL"
        err = explain(
            FakeError("column nmae does not exist", sqlstate="42703", position=999),
            sql,
            known_names=["name"],
        )
        assert err.excerpt is None
        assert err.suggestion == "name"
