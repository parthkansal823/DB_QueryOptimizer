"""
Integration tests against a live PostgreSQL with pg_hint_plan.

Everything else in this suite is a unit test, and that is exactly how the bug
in docs/WRITEUP.md 2.0 survived: `pg_hint_plan` was silently ignoring every
hint, so all "candidates" were byte-identical to the native plan and the
measured improvements were timing noise. No unit test can catch that, because
the string manipulation was correct -- it was the *effect on the planner* that
was absent. A hint that cannot be applied is just a SQL comment: no error, no
warning.

These tests assert the effect. They are the regression guard for the single
most expensive mistake this project made.

Skipped automatically when no database is reachable, so the unit suite still
runs anywhere.
"""

import os

import pytest

from app.optimizer.cardinality import JoinCardinalityCorrector
from app.optimizer.hints import (
    apply_hint,
    corrected_cardinality_hint,
    generate_candidates,
    plan_fingerprint,
)
from app.plan_extractor import get_plan

pytestmark = pytest.mark.integration

TWO_TABLE_SQL = (
    "SELECT o.id, u.name FROM orders o JOIN users u ON o.user_id = u.id "
    "WHERE u.country = 'US'"
)
TRAP_SQL = (
    "SELECT oi.id, p.name FROM order_items oi JOIN products p "
    "ON oi.product_id = p.id "
    "WHERE p.category = 'electronics' AND p.brand = 'Voltix'"
)
FOUR_TABLE_SQL = (
    "SELECT o.id, u.name, p.name FROM orders o "
    "JOIN users u ON o.user_id = u.id "
    "JOIN order_items oi ON oi.order_id = o.id "
    "JOIN products p ON p.id = oi.product_id "
    "WHERE u.country = 'US'"
)


@pytest.fixture(scope="module")
def cur():
    """A cursor, or skip the module if there is no database to talk to."""
    try:
        from app.db import get_cursor
    except Exception as exc:  # pragma: no cover - import-time driver problems
        pytest.skip(f"database driver unavailable: {exc}")

    try:
        with get_cursor() as cursor:
            cursor.execute("SELECT 1")
            yield cursor
    except Exception as exc:
        pytest.skip(f"no database at {os.getenv('DATABASE_URL', 'localhost')}: {exc}")


# -- the extension is actually active ---------------------------------------


def test_pg_hint_plan_is_preloaded(cur):
    """
    `CREATE EXTENSION` registers the SQL objects; only
    `shared_preload_libraries` installs the planner hooks. Without the second,
    every hint in this project is a comment.
    """
    cur.execute("SHOW shared_preload_libraries")
    assert "pg_hint_plan" in cur.fetchone()[0]


# -- hints change the plan --------------------------------------------------


def test_a_leading_hint_changes_the_join_order(cur):
    """
    The §2.0 bug in one assertion.

    Four tables, not two. A two-table join has one join node, so `Leading` has
    nothing to constrain and both orderings return the same plan -- which is
    the §2.10 finding, not a broken hint. Asserting a difference there would
    fail for a reason that has nothing to do with hints binding.
    """
    forward = get_plan(cur, apply_hint(FOUR_TABLE_SQL, "/*+ Leading(o u oi p) */"), analyze=False)
    reverse = get_plan(cur, apply_hint(FOUR_TABLE_SQL, "/*+ Leading(u p oi o) */"), analyze=False)
    assert forward["total_cost"] != reverse["total_cost"]


def test_a_two_table_leading_hint_is_a_no_op(cur):
    """
    Pins §2.10: join order alone is a weak action space on small queries.
    If this ever starts failing, the action-space argument needs revisiting.
    """
    forward = get_plan(cur, apply_hint(TWO_TABLE_SQL, "/*+ Leading(o u) */"), analyze=False)
    reverse = get_plan(cur, apply_hint(TWO_TABLE_SQL, "/*+ Leading(u o) */"), analyze=False)
    assert plan_fingerprint(forward) == plan_fingerprint(reverse)


def test_nested_paren_leading_syntax_is_silently_ignored(cur):
    """
    A trap worth pinning. pg_hint_plan accepts `Leading(a b c)` but ignores
    `Leading(((a b) c))` without raising, so a generator emitting the nested
    form would produce an action space of identical plans -- §2.0 again, in a
    new costume. `hints.py` emits the flat form; this proves why it must.
    """
    flat = get_plan(cur, apply_hint(FOUR_TABLE_SQL, "/*+ Leading(u p oi o) */"), analyze=False)
    nested = get_plan(cur, apply_hint(FOUR_TABLE_SQL, "/*+ Leading(((u p) oi) o) */"), analyze=False)
    native = get_plan(cur, FOUR_TABLE_SQL, analyze=False)
    assert flat["total_cost"] != native["total_cost"]
    assert nested["total_cost"] == native["total_cost"]


def test_an_operator_toggle_changes_the_plan(cur):
    native = get_plan(cur, TWO_TABLE_SQL, analyze=False)
    hinted = get_plan(
        cur, apply_hint(TWO_TABLE_SQL, "/*+ Set(enable_hashjoin off) */"), analyze=False
    )
    assert plan_fingerprint(native) != plan_fingerprint(hinted)


def test_the_candidate_set_contains_genuinely_different_plans(cur):
    """
    The tell that exposed §2.0 was every candidate reporting the same cost.
    An action space of N copies of one plan is not an action space.
    """
    native = get_plan(cur, TRAP_SQL, analyze=False)
    fingerprints = {plan_fingerprint(native)}
    for hint in generate_candidates(native["tables_scanned"]):
        try:
            fingerprints.add(plan_fingerprint(get_plan(cur, apply_hint(TRAP_SQL, hint), analyze=False)))
        except Exception:
            continue
    assert len(fingerprints) > 1, "every candidate produced the native plan"


def test_hint_placement_around_explain_does_not_matter_on_this_version(cur):
    """
    Records a measurement that corrects §2.0.

    That section blames the original silent failure on two causes and states
    "either bug alone is enough": the missing `shared_preload_libraries`, and
    the hint sitting *after* the `EXPLAIN` keyword. On pg_hint_plan 1.6.3 the
    second is not true -- both placements bind, to the same cost. The real
    cause was the preload alone.

    `_split_hint` is kept because hoisting is harmless and other versions may
    differ, but it is defensive, not load-bearing. If this test ever fails,
    the hoisting has become necessary and §2.0 should be corrected back.
    """
    cur.execute(f"EXPLAIN (FORMAT JSON) /*+ Set(enable_hashjoin off) */ {FOUR_TABLE_SQL}")
    after = cur.fetchone()[0][0]["Plan"]["Total Cost"]
    before = get_plan(
        cur, apply_hint(FOUR_TABLE_SQL, "/*+ Set(enable_hashjoin off) */"), analyze=False
    )["total_cost"]
    native = get_plan(cur, FOUR_TABLE_SQL, analyze=False)["total_cost"]

    assert before == after, "hoisting changed the plan; §2.0's second cause is real here"
    assert before != native, "the hint did not bind at all"


# -- the Rows correction binds too ------------------------------------------


def test_a_rows_hint_moves_the_planner_estimate(cur):
    """
    The newest candidate family is the same class of risk: a malformed `Rows`
    hint is silently ignored and the corrected-cardinality plan would quietly
    be the native plan.
    """
    native = get_plan(cur, TWO_TABLE_SQL, analyze=False)
    corrected = get_plan(
        cur, apply_hint(TWO_TABLE_SQL, "/*+ Rows(o u *100) */"), analyze=False
    )
    before = native["raw_plan"]["Plan Rows"]
    after = corrected["raw_plan"]["Plan Rows"]
    assert after == pytest.approx(before * 100, rel=0.02), f"{before} -> {after}"


def test_a_generated_rows_hint_is_accepted_by_the_planner(cur):
    """The hint this system emits, not a handwritten one."""
    class _Model:
        def predict(self, rows):
            return [2.5] * len(rows)  # ~12x underestimate

    corrector = JoinCardinalityCorrector()
    corrector.model = _Model()

    native = get_plan(cur, TRAP_SQL, analyze=False)
    hint = corrected_cardinality_hint(corrector.rows_hints(native["raw_plan"]))
    assert hint is not None, "no correction generated for a 2-table join"

    corrected = get_plan(cur, apply_hint(TRAP_SQL, hint), analyze=False)
    assert corrected["raw_plan"]["Plan Rows"] > native["raw_plan"]["Plan Rows"]


# -- the measurement path ---------------------------------------------------


def test_analyze_returns_real_actuals(cur):
    """`actual_total_time_ms` is the label everything is trained on."""
    plan = get_plan(cur, TWO_TABLE_SQL, analyze=True)
    assert plan["actual_total_time_ms"] > 0
    assert "Actual Rows" in plan["raw_plan"]


def test_explain_without_analyze_executes_nothing(cur):
    """The production path's whole economy depends on this."""
    plan = get_plan(cur, TWO_TABLE_SQL, analyze=False)
    assert plan["actual_total_time_ms"] is None
    assert "Actual Rows" not in plan["raw_plan"]


# -- every keyword this project emits is one the server recognises ----------
#
# The nastiest failure in this area is not a hint that is applied and does
# nothing. It is a hint the extension does not recognise at all: pg_hint_plan
# reports an unknown keyword at **INFO** severity, which the default
# `client_min_messages` discards, so the statement plans unhinted and looks
# exactly like a hint that bound and changed nothing.
#
# That is how `DisableIndex` got added from the current pg_hint_plan
# documentation and reached the generator before anyone noticed the extension
# installed here (1.6.3) has never had it. Documentation is not the authority on
# what this server accepts; this test is.


def _rejected_keywords(cur, hint: str, sql: str) -> list[str]:
    """Any pg_hint_plan parse complaint raised while planning `sql` with `hint`."""
    conn = cur.connection
    cur.execute("SET pg_hint_plan.debug_print = on")
    cur.execute("SET client_min_messages = 'log'")
    del conn.notices[:]
    try:
        cur.execute("EXPLAIN " + apply_hint(sql, hint))
        cur.fetchall()
        return [n for n in conn.notices if "syntax error" in n or "Unrecognized" in n]
    finally:
        cur.execute("SET pg_hint_plan.debug_print = off")
        cur.execute("SET client_min_messages = 'notice'")


def test_every_generated_candidate_is_a_keyword_this_server_knows(cur):
    """
    Generate the whole action space with every family switched on, and assert
    the server parses all of it.

    This is the guard that would have caught `DisableIndex` immediately, and it
    catches the next one for free -- a pg_hint_plan downgrade, a keyword renamed
    upstream, or a family added from documentation for a version this server is
    not running.
    """
    tables = ["o", "u", "oi", "p"]
    join_graph = {"o": ["u", "oi"], "u": ["o"], "oi": ["o", "p"], "p": ["oi"]}
    candidates = generate_candidates(
        tables,
        join_graph=join_graph,
        include_join_methods=True,
        include_tuning=True,
        include_bushy=True,
        include_scan_methods=True,
        include_negative_scans=True,
        include_negative_joins=True,
        include_memoize=True,
        include_parallel=True,
        include_enable_settings=True,
        table_indexes={"o": ["orders_pkey"], "u": ["users_pkey"]},
    )
    assert candidates

    rejected = {}
    for candidate in candidates:
        complaints = _rejected_keywords(cur, candidate, FOUR_TABLE_SQL)
        if complaints:
            rejected[candidate] = complaints[0].strip()
    assert not rejected, f"pg_hint_plan does not recognise: {rejected}"


def test_disable_index_is_not_in_the_action_space(cur):
    """
    The keyword this server rejects, pinned from both directions: the generator
    still produces it (the technique is real and it exists in newer
    pg_hint_plan), and `generate_candidates` still refuses to emit it.

    If a pg_hint_plan upgrade makes this test's first assertion fail, that is
    the signal to add an `include_disable_index` flag -- not to delete the test.
    """
    from app.optimizer.hints import generate_disable_index_candidates

    hint = generate_disable_index_candidates({"u": ["users_pkey"]})[0]
    assert _rejected_keywords(cur, hint, FOUR_TABLE_SQL), (
        "pg_hint_plan now accepts DisableIndex -- wire the family in"
    )

    everything = generate_candidates(
        ["o", "u", "oi", "p"],
        join_graph={"o": ["u", "oi"], "u": ["o"], "oi": ["o", "p"], "p": ["oi"]},
        include_bushy=True,
        include_scan_methods=True,
        include_negative_scans=True,
        include_negative_joins=True,
        include_memoize=True,
        include_parallel=True,
        include_enable_settings=True,
        table_indexes={"u": ["users_pkey"]},
    )
    assert not any("DisableIndex" in c for c in everything)


# -- the alias rule ---------------------------------------------------------


def test_a_hint_naming_the_relation_instead_of_the_alias_does_nothing(cur):
    """
    The silent 4.4x miss, pinned.

    `FROM order_items oi` makes the relation `oi` as far as pg_hint_plan is
    concerned. `Parallel(order_items 4 hard)` is not an error and not a warning
    -- it is discarded, and the plan comes back byte-identical to the baseline.
    Measured on this schema: the aliased form planned at 1774, the relation-name
    form at 7870, the same cost as no hint at all.
    """
    baseline = get_plan(cur, FOUR_TABLE_SQL, analyze=False)
    by_alias = get_plan(
        cur, apply_hint(FOUR_TABLE_SQL, "/*+ Parallel(oi 4 hard) */"), analyze=False
    )
    by_relation = get_plan(
        cur,
        apply_hint(FOUR_TABLE_SQL, "/*+ Parallel(order_items 4 hard) */"),
        analyze=False,
    )
    assert by_relation["total_cost"] == baseline["total_cost"]
    assert by_alias["total_cost"] != baseline["total_cost"]


def test_catalog_indexes_are_keyed_the_way_a_hint_needs_them(cur):
    """
    `indexes_by_alias` reads the catalog and rekeys it through the baseline
    plan's own alias mapping, so the pinning family cannot reproduce the bug
    above.
    """
    from app.optimizer.catalog import index_map, indexes_by_alias, reset_cache

    reset_cache()
    baseline = get_plan(cur, FOUR_TABLE_SQL, analyze=False)
    by_alias = indexes_by_alias(baseline.get("scan_relations"), index_map(cur))

    assert by_alias, "no indexes found for a query over four indexed tables"
    # Aliases, not relation names: the query says `FROM orders o`.
    assert "o" in by_alias
    assert "orders" not in by_alias
    assert "orders_pkey" in by_alias["o"]


@pytest.fixture
def hint_debug_cur(cur):
    """
    `cur` with pg_hint_plan's debug output on, restored afterwards.

    The cursor is module-scoped and shared, so leaving `client_min_messages` at
    `info` would hand every later test a stream of notices it did not ask for.
    """
    from app.optimizer.hint_binding import enable_hint_debug

    enable_hint_debug(cur)
    try:
        yield cur
    finally:
        cur.execute("SET pg_hint_plan.debug_print = off")
        cur.execute("RESET pg_hint_plan.message_level")
        cur.execute("RESET client_min_messages")
        del cur.connection.notices[:]


class TestEveryCandidateBinds:
    """
    Assert that every hint the optimizer generates is actually applied.

    This is the generalisation of the bug in the module docstring above. That
    one was total -- pg_hint_plan ignoring everything -- and was caught by
    checking that hinted plans differ from native. This check is finer: it asks
    pg_hint_plan directly which directives it applied, which separates the two
    cases a cost comparison cannot. `SeqScan(o)` and `SeqScan(orders)` both cost
    3979.3 on the four-table query; the first bound and did nothing, the second
    never bound. Only the debug output tells them apart.

    Writing this found a real defect: `generate_join_method_candidates` sampled
    join orders without the join graph, so `Leading(p o oi u)` produced
    `HashJoin(p o)` for two relations sharing no join predicate. pg_hint_plan
    filed it under `not used hint:` and planned that join freely -- a candidate
    labelled "hash join throughout" whose first join was not forced. 9 of 92
    candidates, all from that family, now 0.
    """

    FOUR_TABLE_SQL = (
        "SELECT o.id, u.name, p.name FROM orders o "
        "JOIN users u ON u.id = o.user_id "
        "JOIN order_items oi ON oi.order_id = o.id "
        "JOIN products p ON p.id = oi.product_id "
        "WHERE o.status = 'delivered' LIMIT 100"
    )
    ALIASES = ["o", "u", "oi", "p"]
    JOIN_GRAPH = {"o": ["u", "oi"], "u": ["o"], "oi": ["o", "p"], "p": ["oi"]}

    def _candidates(self, **overrides):
        from app.optimizer.hints import generate_candidates

        options = dict(
            max_order_candidates=4,
            include_join_methods=True,
            include_tuning=True,
            include_bushy=True,
            include_scan_methods=True,
            include_negative_scans=True,
            include_negative_joins=True,
            include_memoize=True,
            include_parallel=True,
            include_enable_settings=True,
            include_extended_toggles=True,
            include_cost_model=True,
            join_graph=self.JOIN_GRAPH,
        )
        options.update(overrides)
        return generate_candidates(self.ALIASES, **options)

    def test_every_family_binds(self, hint_debug_cur):
        from app.optimizer.hint_binding import check_binding

        notices = hint_debug_cur.connection.notices

        failures = []
        candidates = self._candidates()
        assert len(candidates) > 50, "the point is to cover every family"

        for hint in candidates:
            report = check_binding(
                hint_debug_cur,
                f"{hint} EXPLAIN (FORMAT JSON) {self.FOUR_TABLE_SQL}",
                notices,
            )
            if not report.bound:
                failures.append(f"{hint}\n    " + "\n    ".join(report.problems))

        assert not failures, (
            f"{len(failures)} of {len(candidates)} candidates did not bind:\n"
            + "\n".join(failures)
        )

    def test_a_hint_naming_a_table_instead_of_its_alias_is_caught(self, hint_debug_cur):
        """
        The check has to be able to fail, and this is the exact shape of the
        `Parallel(order_items 4 hard)` bug: cost 7870 unchanged, no error.
        """
        from app.optimizer.hint_binding import check_binding

        report = check_binding(
            hint_debug_cur,
            f"/*+ SeqScan(orders) */ EXPLAIN (FORMAT JSON) {self.FOUR_TABLE_SQL}",
            hint_debug_cur.connection.notices,
        )
        assert not report.bound
        assert report.not_used == ["SeqScan(orders)"]
