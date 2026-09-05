"""
Evidence about a query the guard will not let anyone run.

The regression guard blocks a query after three served executions of the
learned path average worse than native. From then on the query serves native,
so `is_chosen AND hint IS NOT NULL` matches nothing new for it and the learned
path accumulates **no further evidence at all**.

The block therefore lifts when the old rows age out of `evidence_since()` — a
timer, not a decision. The query is retried against whatever model happens to be
deployed, having learned nothing in between, and if that model is no better it
regresses, gets blocked, and the cycle repeats with a period equal to the
lookback window.

Shadow execution breaks it: on a sampled fraction of blocked requests, run the
learned plan *after* the user has been served native, and log it as evidence.
One extra execution on a few requests, and no user ever sees the plan.
"""

import inspect

from app.optimizer import planner
from app.optimizer.regression_guard import SHADOW_SELECTOR


# -- the guard counts it, the statistics do not -----------------------------


def test_the_guard_counts_shadow_rows_as_evidence():
    """Otherwise the measurement is taken and then ignored, which is worse than
    not taking it."""
    from app.optimizer.regression_guard import REGRESSION_SQL

    assert SHADOW_SELECTOR in REGRESSION_SQL
    assert "is_chosen OR selector_used" in REGRESSION_SQL


def test_the_served_statistics_exclude_shadow_rows():
    """The learned plan ran and was timed; the user was served native. Counting
    it as served would report plans as delivered that nobody received."""
    from app.stats import RUNS_SQL

    assert "'shadow'" in RUNS_SQL


def test_the_two_consumers_agree_on_the_marker():
    from app import stats

    assert stats.SHADOW_SELECTOR == SHADOW_SELECTOR


# -- selection is the same one that would have served -----------------------


def test_the_shadow_reuses_plan_query_rather_than_reimplementing_it():
    """Evidence about a *different* decision procedure is evidence about
    nothing. This file's own history is a list of measurements that scored a
    configuration nobody runs (WRITEUP 2.43, 2.47, 2.50)."""
    src = inspect.getsource(planner.plan_query)

    assert "bypass_block=True" in src, "the shadow must go through plan_query"
    assert "_shadow_candidate" not in src, "a hand-rolled copy would drift"


def test_bypassing_the_block_does_not_recurse():
    """The bypassed call must not itself try to shadow, or a blocked query
    spawns one measurement per level forever."""
    src = inspect.getsource(planner.plan_query)

    assert "not bypass_block and SHADOW_SAMPLE_RATE" in src


def test_the_block_still_blocks_when_not_bypassed():
    src = inspect.getsource(planner.plan_query)

    assert "not bypass_block and guard.is_blocked" in src


# -- the user is never made to wait for it ----------------------------------


def test_the_shadow_runs_after_the_served_plan():
    """A shadow measured first would put its latency in front of the user's
    answer, on the queries already identified as troublesome."""
    src = inspect.getsource(planner.optimize_and_execute)

    served = src.index("executed = execute_chosen(")
    shadow = src.index("shadow_learned = execute_chosen(")
    assert served < shadow


def test_the_shadow_gets_the_same_deadline_a_deviation_would():
    """A shadow allowed to run forever turns a protected request into a slow
    one, which is what the block exists to prevent."""
    src = inspect.getsource(planner.optimize_and_execute)
    after = src[src.index("shadow_learned = execute_chosen("):]

    assert "runtime_budget_ms" in after[:400]


def test_a_failed_shadow_does_not_fail_the_request():
    src = inspect.getsource(planner.optimize_and_execute)
    after = src[src.index("shadow_learned = execute_chosen("):]

    assert "except Exception" in after[:600]


def test_the_api_logs_it_unserved_and_unchosen():
    from app import main

    # `optimize_query` is a thin wrapper that turns SQL errors into a 400;
    # the logging lives in the function it delegates to.
    src = inspect.getsource(main._optimize_query)

    assert "shadow_learned_plan" in src
    assert "SHADOW_SELECTOR" in src
    assert "is_chosen=False" in src
