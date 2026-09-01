"""
Blocking a kind of action rather than a whole query.

The whole-query guard is a blunt instrument: a query rarely regresses because
"the learned path" is bad on it, but because *one kind of action* is. Banning
hash joins on a query whose join genuinely wants one is a different mistake
from reordering that join badly, and the all-or-nothing guard throws away every
good candidate along with the bad one -- after which the query stops being
optimised at all.

The screenshot that started this was exactly that shape: the served plan was
`Set(enable_hashjoin off)` on a two-table join, 2.04 ms becoming 3.05 ms. The
join-order candidates for that query were never the problem.
"""

import pytest

from app.optimizer.hints import (
    corrected_cardinality_hint,
    generate_hint_sets,
    generate_join_method_candidates,
    generate_join_order_candidates,
    generate_tuning_candidates,
)
from app.optimizer.regression_guard import (
    RegressionGuard,
    find_regressed_families,
    hint_family,
)


class _FamilyCursor:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, query, params=None):
        pass

    def fetchall(self):
        return self._rows


# -- classification ---------------------------------------------------------


@pytest.mark.parametrize(
    ("hint", "expected"),
    [
        ("/*+ Leading(a b c) */", "join_order"),
        ("/*+ Set(enable_hashjoin off) */", "operator_toggle"),
        ("/*+ Set(enable_nestloop off) Set(enable_mergejoin off) */", "operator_toggle"),
        ("/*+ Set(work_mem 65536) */", "resource"),
        ("/*+ Set(max_parallel_workers_per_gather 4) */", "resource"),
        ("/*+ Rows(a b *10) */", "cardinality"),
        ("/*+ Leading(a b) HashJoin(a b) */", "join_method"),
    ],
)
def test_hints_are_classified_by_what_they_do(hint, expected):
    assert hint_family(hint) == expected


def test_a_join_method_hint_is_not_mistaken_for_a_join_order():
    """It carries a `Leading()` prefix as well as its method tokens, so the
    more specific pattern has to be tested first."""
    assert hint_family("/*+ Leading(a b) MergeJoin(a b) */") == "join_method"


def test_an_operator_toggle_is_not_mistaken_for_a_resource_setting():
    """Both are `Set()`, and they are completely different kinds of action."""
    assert hint_family("/*+ Set(enable_seqscan off) */") == "operator_toggle"
    assert hint_family("/*+ Set(random_page_cost 1.1) */") == "resource"


def test_no_hint_has_no_family():
    assert hint_family(None) is None
    assert hint_family("") is None


def test_every_hint_the_generator_produces_lands_in_a_real_family():
    """`other` is a fallback, not a destination. A generated hint falling into
    it would be silently ungovernable by this guard."""
    every = (
        generate_join_order_candidates(["a", "b", "c"])
        + generate_join_method_candidates(["a", "b", "c"])
        + generate_hint_sets()
        + generate_tuning_candidates()
        + [corrected_cardinality_hint(["Rows(a b *10)"])]
    )

    assert every, "the generators produced nothing to classify"
    assert all(hint_family(h) != "other" for h in every)


def test_the_sql_and_python_classifiers_agree_on_every_generated_hint():
    """
    They are two implementations of one rule -- the SQL groups the history, the
    Python filters the action space. Drift between them would block one family
    in the database and a different one at serving time, which is worse than
    either doing nothing.
    """
    import re

    from app.optimizer.regression_guard import HINT_FAMILY_SQL

    # Read the rule out of the SQL rather than restating it: `WHEN ... THEN
    # 'name'`, in order, with `%%`-escaped LIKE patterns.
    branches = re.findall(r"LIKE '%%(.+?)%%'|THEN '(\w+)'", HINT_FAMILY_SQL)
    patterns, current = [], []
    for like, name in branches:
        if like:
            current.append(like)
        else:
            patterns.append((current, name))
            current = []

    def sql_family(hint):
        for tokens, name in patterns:
            if any(t in hint for t in tokens):
                return name
        return "other"

    every = (
        generate_join_order_candidates(["a", "b", "c"])
        + generate_join_method_candidates(["a", "b", "c"])
        + generate_hint_sets()
        + generate_tuning_candidates()
        + [corrected_cardinality_hint(["Rows(a b *10)"])]
    )

    for hint in every:
        assert sql_family(hint) == hint_family(hint), hint


# -- finding regressed families ---------------------------------------------


# (query_id, family, native_avg, family_avg, n_served)
def test_a_family_slower_than_native_is_flagged():
    cur = _FamilyCursor([("q1", "operator_toggle", 100.0, 150.0, 4)])

    regressed = find_regressed_families(cur)

    assert regressed["q1"]["operator_toggle"]["regression_ratio"] == 1.5


def test_a_family_faster_than_native_is_left_alone():
    cur = _FamilyCursor([("q1", "join_order", 100.0, 60.0, 4)])

    assert find_regressed_families(cur) == {}


def test_a_wash_within_tolerance_is_not_a_regression():
    cur = _FamilyCursor([("q1", "join_order", 100.0, 105.0, 4)])

    assert find_regressed_families(cur, tolerance=0.10) == {}


def test_a_family_with_too_little_history_is_not_judged():
    cur = _FamilyCursor([("q1", "resource", 100.0, 300.0, 1)])

    assert find_regressed_families(cur, min_observations=2) == {}


def test_only_the_losing_family_is_recorded():
    cur = _FamilyCursor([
        ("q1", "operator_toggle", 100.0, 150.0, 4),
        ("q1", "join_order", 100.0, 70.0, 4),
    ])

    families = find_regressed_families(cur)["q1"]

    assert set(families) == {"operator_toggle"}


# -- filtering the action space ---------------------------------------------


def _guard_with(blocked):
    guard = RegressionGuard()
    guard.blocked_families = blocked
    return guard


HINTS = [
    "/*+ Leading(a b) */",
    "/*+ Set(enable_hashjoin off) */",
    "/*+ Set(work_mem 65536) */",
]


def test_a_blocked_family_is_dropped_and_the_rest_survive():
    """The point of the whole feature: the query keeps being optimised by the
    families that still work on it."""
    guard = _guard_with({"q1": {"operator_toggle": {}}})

    kept = guard.allowed_hints("q1", HINTS)

    assert "/*+ Set(enable_hashjoin off) */" not in kept
    assert "/*+ Leading(a b) */" in kept
    assert "/*+ Set(work_mem 65536) */" in kept


def test_a_query_with_no_blocked_families_keeps_everything():
    assert _guard_with({}).allowed_hints("q1", HINTS) == HINTS


def test_an_adhoc_query_without_an_id_is_never_filtered():
    assert _guard_with({"q1": {"join_order": {}}}).allowed_hints(None, HINTS) == HINTS


def test_blocking_every_family_falls_back_rather_than_emptying_the_space():
    """
    A query every family has failed on is one for `is_blocked` to stop
    outright. Handing the planner nothing would surface as "no candidates",
    which is a different diagnosis from the true one.
    """
    guard = _guard_with(
        {"q1": {"join_order": {}, "operator_toggle": {}, "resource": {}}}
    )

    assert guard.allowed_hints("q1", HINTS) == HINTS


def test_is_family_blocked_needs_both_a_query_and_a_hint():
    guard = _guard_with({"q1": {"join_order": {}}})

    assert guard.is_family_blocked("q1", "/*+ Leading(a b) */") is True
    assert guard.is_family_blocked("q1", None) is False
    assert guard.is_family_blocked(None, "/*+ Leading(a b) */") is False
    assert guard.is_family_blocked("q2", "/*+ Leading(a b) */") is False


def test_a_fresh_guard_blocks_nothing():
    assert RegressionGuard().allowed_hints("q1", HINTS) == HINTS
