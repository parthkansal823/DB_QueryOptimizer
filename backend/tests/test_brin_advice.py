"""
BRIN advice, and the one statistic it turns on.

`dbadvisor` proposed B-tree for everything except the trigram case. BRIN is
core -- no extension -- and one column in this schema is exactly its shape:
`orders.created_at` on an append-only table.

The recommendation is only ever as good as the physical/logical correlation,
which PostgreSQL already computes during ANALYZE. On a well-correlated column
BRIN is a few hundred kilobytes where the B-tree is tens of megabytes. On an
uncorrelated one every page range matches and it degenerates into a sequential
scan plus wasted index reads -- so a false positive here is worse than staying
silent, which is why the threshold is strict and a missing statistic withholds
the advice rather than guessing.
"""

from app.index_advice import (
    MIN_BRIN_CORRELATION,
    MIN_BRIN_PAGES,
    brin_opportunities,
)


class FakeCursor:
    """
    Serves both queries `brin_opportunities` issues: the unique-column lookup
    and the correlation scan. Dispatching on the SQL rather than on call order
    keeps the tests from breaking the next time one is reordered.
    """

    def __init__(self, rows, unique=()):
        self._rows = rows
        self._unique = list(unique)
        self._result = []

    def execute(self, sql, params=None):
        if "indisunique" in sql:
            self._result = self._unique
            return
        # The real SQL filters on correlation and pages; emulate that so the
        # tests exercise the module's thresholds rather than trusting a string.
        _, min_corr, min_pages = params
        self._result = [
            r for r in self._rows if abs(r[2]) >= min_corr and r[3] >= min_pages
        ]

    def fetchall(self):
        return self._result


def row(table="orders", col="created_at", corr=0.99, pages=5000, typ="timestamptz"):
    return (table, col, corr, pages, typ)


class TestWhenItFires:
    def test_an_append_only_timestamp_column_is_recommended(self):
        found = brin_opportunities(FakeCursor([row()]))
        assert len(found) == 1
        assert found[0]["index_type"] == "brin"
        assert found[0]["table"] == "orders"
        assert found[0]["column"] == "created_at"

    def test_the_ddl_is_concurrent_and_names_the_access_method(self):
        """
        Every `CREATE INDEX` this project emits blocks writes for the duration
        unless it says CONCURRENTLY, and on a table large enough to want BRIN
        that is not instantaneous.
        """
        ddl = brin_opportunities(FakeCursor([row()]))[0]["ddl"]
        assert "CONCURRENTLY" in ddl
        assert "USING brin" in ddl
        assert ddl.rstrip().endswith(";")

    def test_the_note_carries_the_correlation_it_relied_on(self):
        """
        The advice is entirely a consequence of one number. A reader who cannot
        see that number cannot judge the advice.
        """
        found = brin_opportunities(FakeCursor([row(corr=0.977)]))[0]
        assert "0.977" in found["note"]
        assert found["correlation"] == 0.977


class TestWhenItStaysSilent:
    def test_an_uncorrelated_column_is_not_recommended(self):
        """The case where BRIN is actively worse than no index at all."""
        assert brin_opportunities(FakeCursor([row(corr=0.2)])) == []

    def test_a_small_table_is_not_recommended(self):
        """Too few page ranges for skipping to save anything."""
        assert brin_opportunities(FakeCursor([row(pages=100)])) == []

    def test_a_column_with_no_statistics_is_absent_rather_than_guessed(self):
        """
        `pg_stats` has no row for a column that was never analyzed. The query
        filters those out, so the advisor says nothing instead of assuming.
        """
        assert brin_opportunities(FakeCursor([])) == []

    def test_a_text_column_is_not_recommended_however_correlated(self):
        """
        A min/max summary per page range only discriminates for types with a
        meaningful range. This keeps BRIN off columns where a perfect
        correlation still buys nothing.
        """
        assert brin_opportunities(FakeCursor([row(col="status", typ="text")])) == []

    def test_a_boolean_is_not_recommended(self):
        """Every page range contains both values, so nothing is ever skipped."""
        assert brin_opportunities(FakeCursor([row(col="is_paid", typ="bool")])) == []


class TestThresholds:
    def test_negative_correlation_counts(self):
        """
        A table written in descending order is just as ordered as one written
        ascending -- `pg_stats.correlation` is signed, and BRIN does not care
        about the direction.
        """
        assert len(brin_opportunities(FakeCursor([row(corr=-0.98)]))) == 1

    def test_the_correlation_bar_is_strict(self):
        """
        Deliberately strict: a B-tree that is merely large still works, while a
        BRIN on a weakly-ordered column is a slower sequential scan.
        """
        assert MIN_BRIN_CORRELATION >= 0.85

    def test_the_page_bar_excludes_small_tables(self):
        assert MIN_BRIN_PAGES >= 1000


class TestCaveats:
    def test_the_reader_is_told_what_brin_cannot_do(self):
        caveats = " ".join(brin_opportunities(FakeCursor([row()]))[0]["caveats"]).lower()
        assert "uniqueness" in caveats or "unique" in caveats
        assert "order by" in caveats

    def test_the_reader_is_warned_that_the_benefit_decays(self):
        """
        The failure mode is silent: out-of-order inserts drop the correlation
        and the index stops skipping, with no error anywhere.
        """
        caveats = " ".join(brin_opportunities(FakeCursor([row()]))[0]["caveats"]).lower()
        assert "silently" in caveats or "degrade" in caveats


class TestKeyColumns:
    def test_a_primary_key_column_is_excluded(self):
        """
        A surrogate key on an append-only table is the most perfectly correlated
        column there is -- `order_items.id` measures 1.000 on this database --
        and the one BRIN can least afford to replace. It cannot enforce
        uniqueness and cannot serve the single-row lookup the key exists for, so
        this is advice that would break the table if followed.
        """
        rows = [row(table="order_items", col="id", corr=1.0, typ="int8")]
        assert brin_opportunities(FakeCursor(rows, unique=[("order_items", "id")])) == []

    def test_a_non_key_column_on_the_same_table_still_qualifies(self):
        """The exclusion is per column, not per table."""
        rows = [
            row(table="orders", col="id", corr=1.0, typ="int8"),
            row(table="orders", col="created_at", corr=0.98),
        ]
        found = brin_opportunities(FakeCursor(rows, unique=[("orders", "id")]))
        assert [f["column"] for f in found] == ["created_at"]
