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
