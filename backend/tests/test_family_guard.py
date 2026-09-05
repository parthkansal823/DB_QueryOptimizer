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


# -- every hint the generator can produce must have a family ----------------


def _every_hint() -> list[str]:
    """The full action space, every opt-in family on."""
    from app.optimizer.hints import generate_candidates

    return generate_candidates(
        ["a", "b", "c", "d"],
        max_order_candidates=6,
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
        include_disable_index=True,
        table_indexes={"a": ["a_idx"], "b": ["b_idx"]},
    )


def test_no_generated_hint_falls_through_to_other():
    """
    `other` is in no `_servable` allowed set, so a hint classified that way is
    generated and served but dropped from every replay -- `app.calibrate`,
    `app.autotune`, `retrain._score_bundle`. The gate would then be calibrated
    over a smaller action space than the one being served.

    Measured before the fix: **37 of 96** hints with the full action space on,
    39% of it -- every `SeqScan`/`IndexScan`/`BitmapScan`, both `Memoize` forms,
    `Parallel`, and `DisableIndex`. Latent, because those families are off by
    default; it would have bitten the first person to turn one on.

    The existing tests asserted the Python and SQL classifiers *agree*, and they
    did -- both said `other`. Agreement on a wrong answer is still agreement.
    """
    from app.optimizer.regression_guard import hint_family

    unclassified = [h for h in _every_hint() if hint_family(h) == "other"]

    assert unclassified == [], (
        f"{len(unclassified)} hints have no family and would be invisible to "
        f"every replay: {sorted(set(unclassified))[:6]}"
    )


def test_scan_method_hints_are_servable_when_the_family_is_on():
    """Classifying them is only half the fix: `_servable` has to admit them, or
    they stay invisible under a different name."""
    from app.calibrate import _servable
    from app.optimizer import planner

    probe = [{"hint": "/*+ SeqScan(a) */"}, {"hint": "/*+ IndexScan(b) */"}]

    original = planner.INCLUDE_SCAN_METHODS
    try:
        planner.INCLUDE_SCAN_METHODS = False
        assert _servable(probe) == []
        planner.INCLUDE_SCAN_METHODS = True
        assert len(_servable(probe)) == 2
    finally:
        planner.INCLUDE_SCAN_METHODS = original


def test_negative_scan_hints_share_the_positive_family():
    """`NoSeqScan(a)` and `SeqScan(a)` are the same kind of action -- one names
    the scan to use, the other excludes the alternatives. Splitting them would
    let the guard block one while the other kept firing."""
    from app.optimizer.regression_guard import hint_family

    assert hint_family("/*+ NoSeqScan(a) */") == hint_family("/*+ SeqScan(a) */")
    assert hint_family("/*+ NoNestLoop(a b) */") == hint_family("/*+ NestLoop(a b) */")
    assert hint_family("/*+ NoMemoize(a b) */") == hint_family("/*+ Memoize(a b) */")


# -- collection must not be narrower than serving ---------------------------


def test_collection_offers_at_least_what_serving_will_score():
    """
    The invariant, and the direction matters.

    Collection may legitimately be *wider* than serving: it gathers the forced
    join-method family that `budget_for` often truncates away, and
    `calibrate._servable` narrows the replay back down. Training on plan types
    the server never offers is harmless.

    Narrower is not. The model is then asked at serving time to score plan
    families it has never seen, which is the train/serve skew README warns about
    and docs/WRITEUP.md 2.9 measured: 17 of 25 queries went into the safety veto.

    `app.collect_data` took `generate_candidates`' bare defaults --
    `max_order_candidates=8`, every opt-in family off -- while
    `planner.candidate_hints` passed the module flags and a width-scaled budget
    that reaches 96 (2.56). Collection was training on 8 join orders while
    serving scored up to 96 of them.
    """
    import inspect
    import re

    from app import collect_data
    from app.optimizer import planner

    collect_src = inspect.getsource(collect_data.collect)
    serve_src = inspect.getsource(planner.candidate_hints)

    # Every family switch the serving path forwards has to be forwarded here
    # too, or collection silently omits a family the server will offer.
    # Parameter *names* only: a signature line carries a type annotation, and
    # `include_tuning: bool | None` is not a family.
    forwarded_by_serving = {
        match.group(1)
        for match in re.finditer(r"\b(include_[a-z_]+)\s*=", serve_src)
    }
    assert forwarded_by_serving, "found no family switches to check"

    missing = sorted(f for f in forwarded_by_serving if f not in collect_src)
    assert missing == [], (
        f"serving forwards {missing} but collection does not, so the model "
        f"would be scored on families it never trained on"
    )


def test_collection_uses_the_same_budget_function_as_serving():
    """A fixed `max_order_candidates=8` in collection against a width-scaled
    budget in serving is the same skew by another route."""
    import inspect

    from app import collect_data

    src = inspect.getsource(collect_data.collect)

    assert "budget_for(" in src, (
        "collection must size its join-order sample the same way serving does"
    )


def test_the_benchmark_filters_families_exactly_as_serving_does():
    """
    `planner.plan_query` drops hint families that have regressed on a query
    before choosing (`allowed_hints`). `app.benchmark` built its candidate list
    straight from `candidate_hints` and did not, so with the guard on it
    measured a *wider* action space than production serves.

    `--protocol compare` turns the guard off entirely and is unaffected;
    `--protocol deploy` runs with it on, and that figure was taken over
    candidates the server would have removed.
    """
    import inspect

    from app import benchmark
    from app.optimizer import planner

    serve = inspect.getsource(planner.plan_query)
    bench = inspect.getsource(benchmark.run)

    assert "allowed_hints" in serve
    assert "allowed_hints" in bench, (
        "the benchmark scores candidates the serving path would have dropped"
    )


def test_family_filtering_is_tied_to_the_guard_being_on():
    """`compare` deliberately runs guardless so two configurations differ only
    in the configuration (2.30). Filtering families there would reintroduce the
    guard's state into a protocol built to exclude it."""
    import inspect

    from app import benchmark

    src = inspect.getsource(benchmark.run)
    filter_line = next(
        (line for line in src.splitlines() if "allowed_hints" in line), ""
    )
    assert filter_line, "no family filter found"

    lines = src.splitlines()
    idx = lines.index(filter_line)
    guarding = any("if use_guard" in line for line in lines[max(0, idx - 3):idx])
    assert guarding, "the family filter must be conditional on use_guard"


def test_collection_sizes_its_search_from_the_measurement_it_just_took():
    """
    `budget_for` declines to widen a query measured under half a second, because
    planning is a fixed cost per candidate and it is the ratio that matters
    (WRITEUP 2.56). Serving reads that measurement from the guard's history.

    Collection has just *executed* the baseline, so it has the same number
    directly. Not passing it means a query collected at the wide budget and
    served at the narrow one -- the same train/serve gap, in reverse.
    """
    import inspect

    from app import collect_data

    src = inspect.getsource(collect_data.collect)

    assert "observed_native_ms=" in src, (
        "collection sizes its search from the cost estimate alone while serving "
        "uses the measurement"
    )


def test_the_collection_budget_uses_a_median_not_one_run():
    """Latency has a heavy right tail; one execution colliding with autovacuum
    should not decide how wide the search gets."""
    from app.collect_data import _median_latency

    assert _median_latency([{"actual_total_time_ms": t} for t in (10, 12, 400)]) == 12
    assert _median_latency([{"actual_total_time_ms": None}]) is None
    assert _median_latency([]) is None
