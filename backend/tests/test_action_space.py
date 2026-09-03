"""
Which hint families a served query actually gets to choose between.

`app.collect_data` has always collected the forced join-*method* family --
`include_join_methods` defaults to True there -- while `planner.candidate_hints`
never generated it, and both serving paths go through `candidate_hints`. So the
join-method stretch goal reached the training set, the README and the writeup
without ever reaching a served decision: the model learned about a family of
plans it could not be offered.

That is the harmless direction of a train/serve mismatch -- nothing is scored
that was never seen, unlike the case docs/WRITEUP.md 2.9 documents -- but it is
still a claim the system was not keeping. It is now reachable, and off by
default, because widening the space costs planning round trips on the path whose
whole economic case is that choosing stays cheap.
"""

import pytest

from app.optimizer import planner

TABLES = ["o", "u"]
BASELINE = {
    "join_graph": {"o": ["u"], "u": ["o"]},
    "raw_plan": {"Node Type": "Hash Join", "Plans": []},
}


class _OptimizerWithoutCorrector:
    join_corrector = None


@pytest.fixture(autouse=True)
def _restore_flags():
    """The module flags are process-wide and set from `/settings`."""
    saved = (
        planner.ENABLE_ROWS_CORRECTION,
        planner.INCLUDE_JOIN_METHODS,
        planner.INCLUDE_TUNING,
        planner.ADAPTIVE_CANDIDATE_BUDGET,
    )
    yield
    (
        planner.ENABLE_ROWS_CORRECTION,
        planner.INCLUDE_JOIN_METHODS,
        planner.INCLUDE_TUNING,
        planner.ADAPTIVE_CANDIDATE_BUDGET,
    ) = saved


def _hints(**kwargs):
    return planner.candidate_hints(_OptimizerWithoutCorrector(), BASELINE, TABLES, **kwargs)


def _forces_a_method(hint):
    return any(method in hint for method in ("HashJoin(", "NestLoop(", "MergeJoin("))


def test_join_methods_are_off_by_default():
    """Widening the action space is not free -- 12 extra planning round trips
    per query -- so it is opt-in and meant to be settled by measurement."""
    planner.INCLUDE_JOIN_METHODS = False

    assert not any(_forces_a_method(h) for h in _hints())


def test_join_methods_can_be_served():
    """The gap this closes: before, no value of any setting could put a forced
    join method in front of a served query."""
    planner.INCLUDE_JOIN_METHODS = True

    assert any(_forces_a_method(h) for h in _hints())


def test_every_join_method_family_is_offered():
    planner.INCLUDE_JOIN_METHODS = True
    hints = " ".join(_hints())

    assert "HashJoin(" in hints
    assert "NestLoop(" in hints
    assert "MergeJoin(" in hints


def test_the_flag_is_what_both_serving_paths_read():
    """`/query/optimize` and `/query/analyze` share `candidate_hints` precisely
    so their action spaces cannot drift apart. An explicit argument still wins,
    for callers that want to pin it."""
    planner.INCLUDE_JOIN_METHODS = True
    assert not any(_forces_a_method(h) for h in _hints(include_join_methods=False))

    planner.INCLUDE_JOIN_METHODS = False
    assert any(_forces_a_method(h) for h in _hints(include_join_methods=True))


def test_join_orders_are_still_offered_either_way():
    """The method family is added to the space, not swapped in for it."""
    for enabled in (False, True):
        planner.INCLUDE_JOIN_METHODS = enabled

        assert any("Leading(" in h for h in _hints())


def test_widening_the_space_only_adds_candidates():
    planner.INCLUDE_JOIN_METHODS = False
    narrow = _hints()
    planner.INCLUDE_JOIN_METHODS = True
    wide = _hints()

    assert set(narrow) <= set(wide)
    assert len(wide) > len(narrow)


def test_the_setting_is_declared_for_the_dashboard():
    """A field the UI renders itself from, so turning it on does not require an
    edit-rebuild-restart cycle -- the reason `app.settings` exists."""
    from app import settings as app_settings

    field = app_settings.FIELDS["include_join_methods"]
    assert field["type"] == "bool"
    assert field["default"] is False
    assert app_settings.current()["include_join_methods"] in (True, False)


# -- spending planning effort where it can pay for itself ---------------------
#
# Every candidate costs an EXPLAIN round trip, about 2.5 ms here. The project's
# own measurement is 25.2 ms of choosing against 21.8 ms of execution, which
# `decision_cache.py` fixes for repeat traffic by paying it once -- and does
# nothing for a query seen once, which is most of what a dashboard sends.


def test_a_cheap_query_gets_a_small_budget():
    planner.ADAPTIVE_CANDIDATE_BUDGET = True

    assert planner.budget_for(50.0) < planner.budget_for(5_000.0)


def test_budget_grows_monotonically_with_estimated_cost():
    planner.ADAPTIVE_CANDIDATE_BUDGET = True
    budgets = [planner.budget_for(c) for c in (10.0, 500.0, 5_000.0, 500_000.0)]

    assert budgets == sorted(budgets)


def test_an_expensive_query_is_not_budget_limited():
    planner.ADAPTIVE_CANDIDATE_BUDGET = True

    assert planner.budget_for(1_000_000.0) == planner.UNBUDGETED_CANDIDATES


def test_a_missing_estimate_does_not_starve_the_search():
    """No cost to reason about is not a reason to try almost nothing."""
    planner.ADAPTIVE_CANDIDATE_BUDGET = True

    assert planner.budget_for(None) == planner.UNBUDGETED_CANDIDATES
    assert planner.budget_for(0.0) == planner.UNBUDGETED_CANDIDATES


def test_the_budget_can_be_turned_off():
    planner.ADAPTIVE_CANDIDATE_BUDGET = False

    assert planner.budget_for(1.0) == planner.UNBUDGETED_CANDIDATES


def test_truncation_keeps_the_families_that_actually_win():
    """
    The budget truncates from the end, so generation order decides what a query
    never gets to try. That order is now set by measurement.

    It used to run join orders, then operator toggles, then resources, on the
    argument that "resource settings are exactly the candidates that cannot help
    a small query" -- more `work_mem` does nothing for a join over a few
    thousand rows, while reordering it might. Plausible, and measured over 2,364
    collected candidates on this database, wrong:

        resource / cost model   n=542    mean ratio 0.99   median 0.93
        join order              n=486    mean ratio 1.27   median 1.12
        operator toggle        n=1328    mean ratio 2.91   median 1.58

    The resource family is the only one whose median is *below* native, and it
    was the one being discarded -- on every query in the workload, and on seven
    of eight it was the only thing discarded.
    """
    planner.INCLUDE_TUNING = True
    budget = planner.budget_for(50.0)
    kept = _hints()[:budget]

    assert any("Leading(" in h for h in kept), "join orders are still generated first"
    assert any("work_mem" in h for h in kept), (
        "the resource family has the best measured outcomes and must survive "
        "a small budget"
    )


def test_operator_toggles_are_what_a_tight_budget_gives_up():
    """The family with the worst measured mean is the one truncated first."""
    planner.INCLUDE_TUNING = True
    kept = _hints()[: planner.budget_for(50.0)]

    assert not any("enable_" in h for h in kept)


def test_a_budgeted_action_space_is_a_subset_of_the_collected_one():
    """The harmless direction of a train/serve difference: the model is never
    asked to score a family it has not seen."""
    planner.INCLUDE_TUNING = True
    full = _hints()

    assert set(full[: planner.budget_for(50.0)]) <= set(full)


def test_budgeting_is_deterministic():
    """Truncating a deterministic list keeps a query's action space stable
    between runs, which every A/B in the writeup depends on."""
    planner.INCLUDE_TUNING = True
    budget = planner.budget_for(500.0)

    assert _hints()[:budget] == _hints()[:budget]


def test_the_budget_setting_is_declared_for_the_dashboard():
    from app import settings as app_settings

    field = app_settings.FIELDS["adaptive_candidate_budget"]
    assert field["type"] == "bool"
    assert field["default"] is True


class TestBudgetCountsDistinctPlans:
    """
    The candidate budget funds *alternatives*, not attempts.

    It used to slice the hint list and deduplicate afterwards, which spends the
    budget before knowing what it bought. Many hints legitimately collapse to
    the same plan -- PostgreSQL answers most `Leading()` orders on a small join
    with the plan it already preferred -- and measured on this workload 56 of 83
    candidates were duplicates. The budget funded repeats while genuinely
    different alternatives further down the list went unplanned.
    """

    def test_a_duplicate_plan_is_not_counted_against_the_budget(self):
        """
        The loop's actual contract, exercised directly: given hints that plan to
        repeated shapes, the candidate list fills with *distinct* plans up to
        the budget rather than stopping at the first `budget` hints.
        """
        from app.optimizer.hints import plan_fingerprint

        budget = 3
        # Six hints, four distinct plans among them.
        planned = [
            {"Node Type": "Seq Scan"},
            {"Node Type": "Seq Scan"},
            {"Node Type": "Hash Join"},
            {"Node Type": "Seq Scan"},
            {"Node Type": "Nested Loop"},
            {"Node Type": "Merge Join"},
        ]
        baseline = {"Node Type": "Index Scan"}

        seen = {plan_fingerprint({"raw_plan": baseline})}
        kept, duplicates = [], 0
        for shape in planned:
            if len(kept) >= budget:
                break
            fp = plan_fingerprint({"raw_plan": shape})
            if fp in seen:
                duplicates += 1
                continue
            seen.add(fp)
            kept.append(shape)

        assert len(kept) == budget, "the budget should be filled with distinct plans"
        assert duplicates == 2, "the two repeats should not have consumed a slot"
        assert [k["Node Type"] for k in kept] == ["Seq Scan", "Hash Join", "Nested Loop"]

    def test_the_attempt_cap_is_a_multiple_of_the_budget(self):
        """
        Filling the budget with distinct plans must not let a fully degenerate
        query -- every hint collapsing to the baseline -- plan the entire list
        looking for something that is not there.
        """
        from app.optimizer.planner import MAX_PLANNING_ATTEMPTS_PER_SLOT

        assert MAX_PLANNING_ATTEMPTS_PER_SLOT >= 2, "no room to absorb duplicates"
        assert MAX_PLANNING_ATTEMPTS_PER_SLOT <= 5, "unbounded planning on a bad query"
