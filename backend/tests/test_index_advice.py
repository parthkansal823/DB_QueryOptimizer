"""
Choosing the right *kind* of index, not just the right columns.

Verified against the live planner before being trusted. For all three
specialised shapes, the naive B-tree the advisor used to recommend was built and
then **ignored** — the planner went on sequentially scanning:

| predicate | naive `CREATE INDEX ON t (col)` | recommended |
|---|---|---|
| `name LIKE 'Volt%'` | Seq Scan (ignored) | Index Scan on `text_pattern_ops` |
| `name LIKE '%olti%'` | Seq Scan (ignored) | Bitmap Heap Scan on a trigram GIN |
| `lower(city) = 'x'` | Seq Scan (ignored) | Bitmap Heap Scan on the expression |

That is the failure worth preventing: the index gets built, costs disk and write
amplification, is never used, and the next advisor run reports it as unused and
suggests dropping it. Two recommendations chasing each other.
"""

import pytest

from app.index_advice import (
    already_uniquely_indexed,
    database_is_c_locale,
    expression_for,
    index_recommendation,
    predicate_shape,
)


# Filter text exactly as this server renders it, checked with EXPLAIN rather
# than assumed: LIKE is `~~`, ILIKE is `~~*`.
ANCHORED = "(name ~~ 'Volt%'::text)"
UNANCHORED = "(name ~~ '%olti%'::text)"
ILIKE = "(name ~~* '%volt%'::text)"
EXPRESSION = "(lower(city) = 'mumbai'::text)"
SCALAR = "(city = 'Mumbai'::text)"


class TestPredicateShape:
    @pytest.mark.parametrize(
        "filter_text, column, expected",
        [
            (ANCHORED, "name", "pattern_anchored"),
            (UNANCHORED, "name", "pattern_unanchored"),
            (ILIKE, "name", "pattern_unanchored"),
            (EXPRESSION, "city", "expression"),
            (SCALAR, "city", "scalar"),
            ("", "city", "scalar"),
        ],
    )
    def test_each_shape_is_recognised(self, filter_text, column, expected):
        assert predicate_shape(filter_text, column) == expected

    def test_ilike_is_unanchored_even_with_a_leading_literal(self):
        """
        §11.2.1 permits a B-tree for ILIKE only when the pattern starts with
        characters unaffected by case conversion. The index is ordered by the
        original values, so `ILIKE 'volt%'` still has to match `Volt...`.
        """
        assert predicate_shape("(name ~~* 'volt%'::text)", "name") == "pattern_unanchored"

    def test_an_underscore_wildcard_also_defeats_the_prefix(self):
        assert predicate_shape("(name ~~ '_olt%'::text)", "name") == "pattern_unanchored"

    def test_a_regex_is_anchored_only_with_a_caret(self):
        assert predicate_shape("(name ~ '^Volt'::text)", "name") == "pattern_anchored"
        assert predicate_shape("(name ~ 'Volt'::text)", "name") == "pattern_unanchored"

    def test_a_pattern_on_another_column_does_not_apply(self):
        assert predicate_shape(UNANCHORED, "category") == "scalar"

    def test_the_wrapping_expression_is_extracted(self):
        assert expression_for(EXPRESSION, "city") == "lower(city)"
        assert expression_for(SCALAR, "city") is None


class TestRecommendation:
    def test_an_anchored_like_needs_text_pattern_ops_off_the_c_locale(self):
        rec = index_recommendation("products", ["name"], ANCHORED, c_locale=False)
        assert rec["index_type"] == "text_pattern_ops"
        assert "text_pattern_ops" in rec["ddl"]

    def test_the_c_locale_needs_no_operator_class(self):
        rec = index_recommendation("products", ["name"], ANCHORED, c_locale=True)
        assert rec["index_type"] == "btree"
        assert "text_pattern_ops" not in rec["ddl"]

    def test_an_unanchored_like_needs_a_trigram_index(self):
        rec = index_recommendation("products", ["name"], UNANCHORED, c_locale=False)
        assert rec["index_type"] == "trigram"
        assert rec["requires"] == "pg_trgm"
        assert "gin_trgm_ops" in rec["ddl"]
        assert "CREATE EXTENSION IF NOT EXISTS pg_trgm" in rec["ddl"]

    def test_an_expression_predicate_indexes_the_expression(self):
        rec = index_recommendation("users", ["city"], EXPRESSION, c_locale=False)
        assert rec["index_type"] == "expression"
        assert "((lower(city)))" in rec["ddl"]

    def test_an_expression_recommendation_offers_the_cheaper_statistics_object(self):
        """
        §24.1.3: a statistics object on the same expression fixes the planner's
        selectivity estimate with no write-time cost. Worth trying first when
        the scan is not actually the bottleneck.
        """
        rec = index_recommendation("users", ["city"], EXPRESSION, c_locale=False)
        assert any("CREATE STATISTICS" in c for c in rec["caveats"])

    def test_a_scalar_predicate_still_gets_a_plain_btree(self):
        rec = index_recommendation("users", ["city"], SCALAR, c_locale=False)
        assert rec["index_type"] == "btree"
        assert rec["caveats"] == []

    def test_multi_column_predicates_stay_composite(self):
        """
        With several columns the pattern is one clause among many, and an
        ordinary composite index is still the right answer.
        """
        rec = index_recommendation(
            "products", ["name", "category"], UNANCHORED, c_locale=False
        )
        assert rec["index_type"] == "btree"

    def test_every_specialised_type_states_its_cost(self):
        """
        A GIN index is larger and slower to write; `text_pattern_ops` does not
        serve ORDER BY; an expression index is recomputed per write. The advisor
        is not allowed to hide any of that behind a one-line DDL.
        """
        for filter_text in (ANCHORED, UNANCHORED, EXPRESSION):
            rec = index_recommendation("t", ["name" if "name" in filter_text else "city"],
                                       filter_text, c_locale=False)
            assert rec["caveats"], rec["index_type"]

    def test_identifiers_are_quoted(self):
        rec = index_recommendation("Order Items", ["Full Name"], SCALAR, c_locale=False)
        assert '"Order Items"' in rec["ddl"]
        assert '"Full Name"' in rec["ddl"]

    def test_everything_is_created_concurrently(self):
        """
        A plain CREATE INDEX blocks writes for the length of the build. On a
        recommendation aimed at a live database that is an outage.
        """
        for filter_text in (SCALAR, ANCHORED, UNANCHORED, EXPRESSION):
            rec = index_recommendation("t", ["name"], filter_text, c_locale=False)
            assert "CREATE INDEX CONCURRENTLY" in rec["ddl"]


# -- the columns that need nothing -------------------------------------------


class _Cursor:
    def __init__(self, rows=(), collate="en_US.utf8", fail=False):
        self._rows = list(rows)
        self._collate = collate
        self._fail = fail
        self._mode = None

    def execute(self, sql, params=None):
        if self._fail:
            raise RuntimeError("catalog unavailable")
        self._mode = "collate" if "datcollate" in sql else "rows"

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return (self._collate,) if self._mode == "collate" else None


class TestAlreadyIndexed:
    def test_unique_columns_are_collected(self):
        found = already_uniquely_indexed(_Cursor([("orders", "id"), ("users", "id")]))
        assert found == {("orders", "id"), ("users", "id")}


class TestLocale:
    @pytest.mark.parametrize(
        "collate, expected",
        [("C", True), ("POSIX", True), ("C.UTF-8", True), ("en_US.utf8", False)],
    )
    def test_c_locale_detection(self, collate, expected):
        assert database_is_c_locale(_Cursor(collate=collate)) is expected

    def test_an_unreadable_locale_assumes_the_safer_answer(self):
        """
        Assuming "not C" recommends `text_pattern_ops`, which works in the C
        locale too — merely redundant. Assuming "C" the other way round produces
        an index the planner will never use.
        """
        assert database_is_c_locale(_Cursor(fail=True)) is False


# -- covering indexes --------------------------------------------------------


class TestCoveringOpportunities:
    """
    §11.9. An index-only scan answers from the index and never touches the heap,
    which is what makes the `IndexOnlyScan` hint family usable rather than
    theoretical — the optimizer can force one, but only a covering index makes
    one possible.

    Verified end to end: `SELECT o.status, o.created_at FROM orders o WHERE
    o.user_id = 42` planned as a **Seq Scan**, and after
    `CREATE INDEX ... (user_id) INCLUDE (status, created_at)` plus a VACUUM it
    planned as an **Index Only Scan** on that index.
    """

    def plan(self, output, filter_text="(o.user_id = 42)", cost=4000.0, node="Seq Scan"):
        return {
            "raw_plan": {
                "Node Type": node,
                "Relation Name": "orders",
                "Alias": "o",
                "Total Cost": cost,
                "Filter": filter_text,
                "Output": output,
            }
        }

    def test_a_cheap_scan_is_not_worth_an_index(self):
        """A small table is read in a few page fetches whatever the plan."""
        from app.index_advice import covering_opportunities

        assert covering_opportunities(self.plan(["status"], cost=12.0)) == []

    def test_a_selective_filter_on_a_large_table_is_still_a_candidate(self):
        """
        The gate that got this wrong first time: `WHERE user_id = 42` returns
        about four rows while sequentially scanning 200,000, and gating on rows
        *returned* rejected exactly the scans worth reporting. Cost is the
        signal.
        """
        from app.index_advice import covering_opportunities

        assert covering_opportunities(self.plan(["status"], cost=4000.0))

    def test_a_narrow_projection_is_a_candidate(self):
        from app.index_advice import covering_opportunities

        found = covering_opportunities(self.plan(["status", "created_at"]))
        assert len(found) == 1
        assert found[0]["columns"] == ["user_id"]
        assert found[0]["include"] == ["status", "created_at"]
        assert "INCLUDE" in found[0]["ddl"]

    def test_select_star_is_not_a_candidate(self):
        """
        Four payload columns on a five-column table is an index duplicating 80%
        of the table. The manual: payload columns "duplicate data from the
        index's table and bloat the size of the index, thus potentially slowing
        searches."
        """
        from app.index_advice import covering_opportunities

        assert covering_opportunities(
            self.plan(["id", "user_id", "created_at", "status", "channel"])
        ) == []

    def test_a_scan_with_no_filter_is_not_a_candidate(self):
        """Nothing to use as the index key."""
        from app.index_advice import covering_opportunities

        assert covering_opportunities(self.plan(["status"], filter_text="")) == []

    def test_an_index_only_scan_is_already_covered(self):
        from app.index_advice import covering_opportunities

        assert covering_opportunities(
            self.plan(["status"], node="Index Only Scan")
        ) == []

    def test_the_filter_column_is_not_repeated_as_payload(self):
        from app.index_advice import covering_opportunities

        found = covering_opportunities(self.plan(["user_id", "status"]))
        assert found[0]["include"] == ["status"]

    def test_it_states_the_visibility_map_condition(self):
        """
        §11.9: an index-only scan "will be a win only if a significant fraction
        of the table's heap pages have their all-visible map bits set". A
        covering index on a table whose map is stale buys nothing until VACUUM
        runs, and the Maintenance panel measures exactly that.
        """
        from app.index_advice import covering_opportunities

        caveats = covering_opportunities(self.plan(["status"]))[0]["caveats"]
        assert any("visibility map" in c for c in caveats)
        assert any("insertion fail" in c for c in caveats)

    def test_columns_from_another_relation_are_ignored(self):
        """A join node's Output lists columns from both sides."""
        from app.index_advice import covering_opportunities

        found = covering_opportunities(self.plan(["status", "u.name"]))
        assert found[0]["include"] == ["status"]
