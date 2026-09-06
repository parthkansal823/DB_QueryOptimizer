"""
Before a model exists, serve native.

`docs/WRITEUP.md` 2.82. `select_plan`'s docstring says it picks "defaulting to
native unless there is *evidence* a hinted plan is better". With no model there
is no evidence, and the no-model branch deviated anyway: cost argmin over the
candidates, native not among them, `fell_back_to_baseline = False`, and only
`_is_unsafe` in the way -- whose cost test can never fire on PG18.

Replayed over the 65-query log:

    current    1.040x native   deviates 65/65   35% regress   worst 5.32x
    beat_cost  1.109x native   deviates 12/65   83% regress
    native     1.000x native   deviates  0/65

A fresh install was slower than doing nothing, and the obvious repair was worse
than the bug.
"""

import pytest

from app.optimizer.learned import LearnedOptimizer


def _plan(cost, ms, hint=None, disabled=0):
    return {
        "total_cost": cost,
        "actual_total_time_ms": ms,
        "hint": hint,
        "disabled_nodes": disabled,
        "raw_plan": {"Node Type": "Seq Scan", "Relation Name": "t", "Alias": "t",
                     "Plan Rows": 1, "Plan Width": 8},
    }


@pytest.fixture
def cold():
    """An optimizer with no model, which is what a fresh install has."""
    opt = LearnedOptimizer(policy="pairwise_rank")
    opt.model = None
    opt.ranker = None
    return opt


class TestWithNoModel:
    def test_it_serves_native(self, cold):
        native = _plan(100.0, 50.0)
        cheaper = _plan(10.0, 400.0, hint="/*+ Leading(a b) */")

        assert cold.select_plan([cheaper], baseline_plan=native) is native

    def test_it_says_why(self, cold):
        cold.select_plan([_plan(10.0, 400.0, hint="/*+ x */")],
                         baseline_plan=_plan(100.0, 50.0))

        assert cold.last_decision["fell_back_to_baseline"] is True
        assert cold.last_decision["reason"] == "no_model_yet"
        assert cold.last_decision["chosen_index"] is None

    def test_a_cheaper_candidate_does_not_tempt_it(self, cold):
        """The measured trap. PostgreSQL costing a candidate below native is
        precisely the case where its estimate is wrong -- 10 of the 12
        deviations that rule permits regress."""
        native = _plan(1000.0, 20.0)
        cheap_but_slow = _plan(1.0, 900.0, hint="/*+ Leading(a b) */")

        assert cold.select_plan([cheap_but_slow], baseline_plan=native) is native

    def test_no_candidates_still_returns_native(self, cold):
        native = _plan(100.0, 50.0)

        assert cold.select_plan([], baseline_plan=native) is native

    def test_with_no_baseline_it_falls_back_to_the_cost_argmin(self, cold):
        """Nothing to compare against and nothing to fall back to, so the cost
        ordering is the only one available."""
        a = _plan(500.0, 10.0, hint="/*+ a */")
        b = _plan(50.0, 90.0, hint="/*+ b */")

        assert cold.select_plan([a, b], baseline_plan=None) is b
        assert cold.last_decision["fell_back_to_baseline"] is False


class TestTheHeuristicIsStillTheReferencePoint:
    """Removing it from *serving* must not remove it from the evaluations that
    score against it -- it is the Phase 0 baseline the whole project is
    measured against."""

    def test_select_heuristic_still_exists(self, cold):
        chosen = cold._select_heuristic([_plan(500.0, 10.0), _plan(50.0, 90.0)])

        assert chosen == 1
        assert cold.last_decision["policy"] == "heuristic_min_cost"

    def test_training_computes_it_from_costs_directly(self):
        """Not via `select_plan`, so this change cannot move the baseline every
        capture ratio is reported against."""
        import inspect

        from app import train

        source = inspect.getsource(train._evaluate_selection)
        assert "costs.index(min(costs))" in source

    def test_the_benchmark_labels_the_cold_start_case(self):
        """`app.benchmark` has no heuristic *arm* -- it reports native, served
        and oracle, and names the mode. So this change cannot move any reference
        point there; what it does change is what a `--protocol deploy` run
        serves before a model exists, and the label is what makes those rows
        identifiable afterwards."""
        import inspect

        from app import benchmark

        source = inspect.getsource(benchmark.run)
        assert 'selector_mode = "learned" if optimizer.model is not None else "heuristic"' in source
        assert "costs.index(min(costs))" not in source


class TestAModelStillDeviates:
    """The fix must be scoped to the cold-start branch."""

    def test_a_loaded_model_takes_the_normal_path(self):
        opt = LearnedOptimizer(policy="greedy")
        if opt.model is None:
            pytest.skip("no bundle on disk to exercise the warm path")

        native = _plan(100.0, 50.0)
        opt.select_plan([_plan(90.0, 40.0, hint="/*+ x */")], baseline_plan=native)

        assert opt.last_decision.get("reason") != "no_model_yet"
