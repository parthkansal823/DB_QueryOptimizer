"""
The safety veto that the PostgreSQL 18 upgrade silently switched off.

PostgreSQL prices a node reached under `enable_x = off` so the planner avoids
it. Until 18 that pricing was `disable_cost` (~1e10) added to the plan's total,
so a forbidden plan announced itself in the cost, and `learned._is_unsafe`
detected it that way -- `DISABLE_COST_SCALE = 1e9`.

**PostgreSQL 18 removed the inflation.** Measured on 18.6, one query, with and
without three scan types banned:

    no hint                            cost 8.0   disabled []
    seq/index/bitmapscan off           cost 8.0   disabled [Seq Scan, Index Scan]

Identical cost. So no cost comparison can distinguish a plan the planner was
forbidden to use from one it chose freely, and a veto written against the cost
stopped being reachable the moment the server was upgraded -- with no error, no
failing test, and a `DISABLE_COST_SCALE` constant still sitting there looking
like it did something.

`Disabled: true` (PG17+) is the replacement, and it is better than what it
replaces: it names which nodes, and it cannot be confused by a plan that is
legitimately expensive.
"""

import pytest

from app.optimizer.learned import LearnedOptimizer
from app.plan_extractor import count_disabled_nodes


def node(node_type="Seq Scan", disabled=False, children=None):
    plan = {"Node Type": node_type, "Total Cost": 8.0}
    if disabled:
        plan["Disabled"] = True
    if children:
        plan["Plans"] = children
    return plan


class TestCountingDisabledNodes:
    def test_a_clean_plan_counts_none(self):
        assert count_disabled_nodes(node()) == 0

    def test_a_disabled_root_counts(self):
        assert count_disabled_nodes(node(disabled=True)) == 1

    def test_disabled_children_are_found(self):
        """
        The flag sits on the node that was forbidden, which is usually a scan
        deep in the tree rather than the root.
        """
        plan = node(
            "Hash Join",
            children=[node("Seq Scan", disabled=True), node("Index Scan", disabled=True)],
        )
        assert count_disabled_nodes(plan) == 2

    def test_it_recurses_to_any_depth(self):
        plan = node("Limit", children=[node("Sort", children=[node(disabled=True)])])
        assert count_disabled_nodes(plan) == 1

    def test_a_pg16_plan_has_no_flag_and_counts_zero(self):
        """
        The key is absent before PG17. Zero is the right answer there -- the
        cost signal still works on that server, and `_is_unsafe` falls through
        to it.
        """
        assert count_disabled_nodes({"Node Type": "Seq Scan", "Total Cost": 1e10}) == 0


class TestTheVeto:
    def _optimizer(self):
        opt = LearnedOptimizer(model_path="/nonexistent")
        return opt

    def test_a_plan_with_a_disabled_node_is_refused(self):
        """
        The PG18 case, and the whole point: cost says nothing, the flag says
        everything.
        """
        opt = self._optimizer()
        chosen = {"total_cost": 8.0, "disabled_nodes": 2}
        assert opt._is_unsafe(chosen, {"total_cost": 8.0})

    def test_an_ordinary_plan_is_allowed(self):
        opt = self._optimizer()
        assert not opt._is_unsafe({"total_cost": 14459.0, "disabled_nodes": 0}, {"total_cost": 7925.0})

    def test_a_costlier_plan_is_still_allowed(self):
        """
        Deliberately: the plans this system exists to find are the ones
        PostgreSQL costs *higher* than its own choice. A measured example ran
        6x faster at 1.8x the estimated cost.
        """
        opt = self._optimizer()
        assert not opt._is_unsafe({"total_cost": 1e6, "disabled_nodes": 0}, {"total_cost": 1.0})

    def test_the_pg16_cost_signal_still_works(self):
        """
        The old detection is kept, not replaced. A PG16 plan carries no flag and
        a `disable_cost`-scale total, and must still be refused.
        """
        opt = self._optimizer()
        assert opt._is_unsafe({"total_cost": 1e10}, {"total_cost": 8.0})

    @pytest.mark.parametrize("missing", [{}, {"total_cost": None}, {"total_cost": 0}])
    def test_a_plan_with_no_cost_is_not_refused_on_that_basis(self, missing):
        opt = self._optimizer()
        assert not opt._is_unsafe(missing, {"total_cost": 8.0})


class TestDisabledPlansNeverReachTheActionSpace:
    """
    The second defence, and on PostgreSQL 18 the one that matters.

    Until 18 a plan the planner was forbidden to build carried `disable_cost`
    (~1e10), so it was self-punishing: it wasted a budget slot but no
    cost-based selector would ever pick it. PG18 removed the inflation and the
    ordering inverts. Measured on 18.6, a four-table join:

        Leading(o u oi p)   connected      cost 6990.8   disabled: none
        Leading(p o oi u)   disconnected   cost   20.5   disabled: Nested Loop

    The unusable plan looks **340x cheaper** than the correct one.
    `_select_heuristic` picks the minimum cost, so it would take that plan every
    time; `total_cost` as a model feature becomes actively misleading; and the
    row reaches `plan_execution_log` as training data claiming a broken plan is
    cheap.

    `_is_unsafe` still catches it at the veto, but only after it has displaced a
    real candidate. Dropping it at planning time keeps it out of the action
    space, the model and the log.
    """

    def test_the_filter_is_keyed_on_the_flag_not_on_a_cost_threshold(self):
        """
        Pinning the mechanism, not a number.

        The veto this replaces was written against `total_cost >= 1e9`, which
        was a true fact about PostgreSQL 16 and silently false on 18 -- no
        error, no failing test, just a constant that stopped meaning anything.
        A test asserting "a forbidden plan is refused" survives that; a test
        asserting "1e10 is refused" does not.
        """
        import inspect

        from app.optimizer import planner

        source = inspect.getsource(planner.plan_query)
        assert 'plan.get("disabled_nodes")' in source, (
            "the filter must read the planner's own flag rather than infer from cost"
        )

    def test_the_count_is_reported_separately_from_duplicates(self):
        """
        A duplicate is a hint that changed nothing; a disabled plan is a hint
        the planner could not honour. Collapsing them would hide the second.
        """
        import inspect

        from app.optimizer import planner

        source = inspect.getsource(planner.plan_query)
        assert '"n_disabled_plans"' in source
        assert '"n_duplicate_plans"' in source


class TestTrainingDropsDisabledPlans:
    """
    The three places a forbidden plan has to be stopped, and why all three.

    `planner.plan_query` keeps it out of the action space, `collect_data` stops
    recording new ones, and this handles the rows collected before either filter
    existed -- which on this database is most of them.

    Skipping the third would train the model on candidates the first two
    guarantee it will never be asked to score, which is exactly the train/serve
    skew WRITEUP 2.18 traced to a gate that could not fire.
    """

    def test_a_row_with_a_disabled_plan_is_dropped(self):
        from app.train import rows_without_disabled_plans

        rows = [
            {"raw_plan": {"Node Type": "Hash Join", "Total Cost": 6990.8}},
            {"raw_plan": {"Node Type": "Nested Loop", "Disabled": True, "Total Cost": 20.5}},
        ]
        kept, report = rows_without_disabled_plans(rows)

        assert len(kept) == 1
        assert kept[0]["raw_plan"]["Node Type"] == "Hash Join"
        assert report["dropped_rows"] == 1
        assert report["filtered"] is True

    def test_a_disabled_child_is_enough_to_drop_the_row(self):
        from app.train import rows_without_disabled_plans

        rows = [
            {
                "raw_plan": {
                    "Node Type": "Limit",
                    "Total Cost": 20.5,
                    "Plans": [{"Node Type": "Nested Loop", "Disabled": True}],
                }
            }
        ]
        kept, _ = rows_without_disabled_plans(rows)
        assert kept == []

    def test_pg16_rows_are_untouched(self):
        """
        No `Disabled` key exists before PG17. Every row collected on the old
        server has to survive, or the upgrade would silently discard the entire
        history it was already invalidating for other reasons.
        """
        from app.train import rows_without_disabled_plans

        rows = [{"raw_plan": {"Node Type": "Seq Scan", "Total Cost": 1e10}}] * 5
        kept, report = rows_without_disabled_plans(rows)
        assert len(kept) == 5
        assert report["filtered"] is False

    def test_a_row_without_a_plan_is_kept_rather_than_crashing(self):
        from app.train import rows_without_disabled_plans

        kept, _ = rows_without_disabled_plans([{"raw_plan": None}, {}])
        assert len(kept) == 2

    def test_the_report_says_why(self):
        """
        Silently training on a subset is the same class of mistake as silently
        serving one, so a run that drops rows has to say so.
        """
        from app.train import rows_without_disabled_plans

        _, report = rows_without_disabled_plans(
            [{"raw_plan": {"Node Type": "Seq Scan", "Disabled": True}}]
        )
        assert "could not honour" in report["reason"]


class TestThePg18FeaturesReachTheModel:
    """
    Two signals PostgreSQL 18 added that the model can now see.

    `Index Searches` counts how many times an index was actually probed. Until
    18 nothing in the plan said this: a nested loop that searches an index once
    and one that searches it 50,000 times differ by the whole performance
    question, and `Plan Rows` times `loops` only approximates it.

    `n_disabled_nodes` is 0 for everything the optimizer now serves, since
    `planner.plan_query` filters those out. It exists for the rows already in
    the log, so the model can learn that `total_cost` means nothing on a plan
    carrying one -- WRITEUP 2.29 measured a disconnected order coming back 340x
    cheaper than the correct plan.
    """

    def _features(self, raw_plan, **extra):
        from app.optimizer.features import featurize

        return featurize({"raw_plan": raw_plan, **extra}, {"orders": 1000.0})

    def test_index_searches_are_summed_over_the_tree(self):
        plan = {
            "Node Type": "Nested Loop",
            "Plans": [
                {"Node Type": "Seq Scan"},
                {"Node Type": "Index Scan", "Index Searches": 50000},
            ],
        }
        assert self._features(plan)["n_index_searches"] == 50000.0

    def test_a_pg16_plan_reports_zero_rather_than_missing(self):
        """
        Zero is the truthful value on a server that does not report the field,
        not a gap being papered over -- and it keeps the vector the same width
        across both servers, which is what lets one model shape span them.
        """
        feats = self._features({"Node Type": "Seq Scan", "Total Cost": 10.0})
        assert feats["n_index_searches"] == 0.0
        assert feats["n_disabled_nodes"] == 0.0

    def test_disabled_nodes_are_counted_from_the_raw_tree(self):
        """
        `featurize` is handed rows read straight out of `plan_execution_log`,
        which carry only `raw_plan` -- no `disabled_nodes` key, because that is
        added by `plan_extractor` at planning time.
        """
        plan = {
            "Node Type": "Nested Loop",
            "Disabled": True,
            "Plans": [{"Node Type": "Seq Scan", "Disabled": True}],
        }
        assert self._features(plan)["n_disabled_nodes"] == 2.0

    def test_a_precomputed_count_is_used_when_present(self):
        plan = {"Node Type": "Seq Scan"}
        assert self._features(plan, disabled_nodes=3)["n_disabled_nodes"] == 3.0

    def test_both_features_are_in_the_declared_column_list(self):
        """
        A feature computed but not declared is silently dropped by
        `to_vector`, which is the quiet way to add a feature that does nothing.
        """
        from app.optimizer.features import BASE_FEATURES

        assert "n_index_searches" in BASE_FEATURES
        assert "n_disabled_nodes" in BASE_FEATURES
