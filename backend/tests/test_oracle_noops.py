"""
A plan identical to native must not be the ceiling native is measured against.

`docs/WRITEUP.md` 2.76. `app.collect_data` calls `generate_candidates` directly
while `planner.candidate_hints` runs `prefilter` first, so collection executes
and logs candidates the baseline plan *proves* cannot change it --
`Set(enable_mergejoin off)` on a plan with no Merge Join node. Measured: **188 of
1,304** servable candidates, 14.4%.

Verified rather than argued: on the ten queries whose logged oracle was one of
these hints, planning with and without gave the same `plan_fingerprint` on
**10 of 10**. They are second executions of the native plan.

The subtlety is that they belong in two places at once:

  * **training** keeps them -- dropping them made the model's typical error
    worse on 15 of 16 paired splits, p = 0.001. A row saying "this changed
    nothing and ran at 1.0x" is evidence about the level the gate reads.
  * the **oracle** cannot keep them, because it is a `min` over noisy draws, so
    a candidate that *is* native competes to be the ceiling and won on 11 of 65
    queries. One recorded 3,561 ms against 787 ms for the same plan.
"""

import pytest

from app.optimizer.prefilter import is_provable_noop
from app.train import _evaluate_selection, rows_without_provable_noops

# A plan with no Merge Join and no Incremental Sort, so hints disabling either
# are provable no-ops against it.
PLAN = {
    "Node Type": "Hash Join",
    "Relation Name": None,
    "Plans": [
        {"Node Type": "Seq Scan", "Relation Name": "orders", "Alias": "o",
         "Plan Rows": 1000, "Plan Width": 8},
        {"Node Type": "Hash", "Plans": [
            {"Node Type": "Seq Scan", "Relation Name": "users", "Alias": "u",
             "Plan Rows": 50, "Plan Width": 8},
        ]},
    ],
}
NOOP = "/*+ Set(enable_mergejoin off) */"
REAL = "/*+ Leading(o u) */"


def _row(hint, ms, baseline=False):
    return {
        "query_id": "q", "hint": hint, "is_baseline": baseline,
        "raw_plan": PLAN, "total_cost": 100.0, "actual_total_time_ms": ms,
        "sql_text": "SELECT 1", "created_at": None, "selector_used": "collection",
    }


def test_the_premise_holds_for_this_plan():
    """If `is_provable_noop` disagreed, the rest of this file would be testing
    nothing."""
    assert is_provable_noop(NOOP, PLAN)
    assert not is_provable_noop(REAL, PLAN)


class TestTheRowFilter:
    def test_it_drops_provable_noops(self):
        rows = [_row(None, 100.0, baseline=True), _row(NOOP, 40.0), _row(REAL, 80.0)]

        kept, report = rows_without_provable_noops(rows)

        assert {r["hint"] for r in kept} == {None, REAL}
        assert report["dropped_rows"] == 1

    def test_it_never_drops_a_baseline(self):
        rows = [_row(None, 100.0, baseline=True)]

        kept, _ = rows_without_provable_noops(rows)

        assert kept == rows

    def test_a_query_with_no_baseline_is_left_alone(self):
        """A missing baseline is a reason to know less, not to discard
        evidence -- an ad-hoc query logged from the dashboard has no baseline
        row of its own."""
        rows = [_row(NOOP, 40.0), _row(REAL, 80.0)]

        kept, report = rows_without_provable_noops(rows)

        assert len(kept) == 2
        assert report["dropped_rows"] == 0

    def test_it_reports_why(self):
        _, report = rows_without_provable_noops(
            [_row(None, 100.0, baseline=True), _row(NOOP, 40.0)]
        )

        assert "changes nothing" in report["reason"]


class TestTrainingKeepsThem:
    """The measurement, encoded: 15 of 16 paired splits got *worse* without
    these rows. The default has to stay off, and the reason has to stay written
    down where the default is."""

    def test_the_default_is_off(self):
        import inspect

        from app import train

        params = inspect.signature(train.train).parameters
        assert params["drop_provable_noops"].default is False

    def test_the_reason_is_recorded_next_to_it(self):
        import inspect

        from app import train

        source = inspect.getsource(train.train)
        assert "15 of 16" in source, (
            "a default set against the obvious argument needs the measurement "
            "beside it, or the next reader will 'fix' it"
        )


class TestTheOracleExcludesThem:
    def _rows(self, noop_ms):
        return {"q": [
            _row(None, 100.0, baseline=True),
            _row(NOOP, noop_ms),
            _row(REAL, 80.0),
        ]}

    def test_a_lucky_noop_does_not_become_the_ceiling(self):
        """The whole point: 20 ms is a noisy draw of the 100 ms native plan, and
        letting it set the oracle attributes 80% headroom to doing nothing."""
        result = _evaluate_selection(
            self._rows(20.0), _FlatEnsemble(), [], {}, ranker=None
        )

        assert result["avg_latency_ms"]["oracle_best_possible"] == 80.0

    def test_a_real_candidate_still_sets_the_ceiling(self):
        result = _evaluate_selection(
            self._rows(95.0), _FlatEnsemble(), [], {}, ranker=None
        )

        assert result["avg_latency_ms"]["oracle_best_possible"] == 80.0

    def test_the_query_is_still_scored(self):
        """Dropping the candidates must not drop the query -- that would change
        which queries the average is over, which is the bug 2.47 fixed."""
        result = _evaluate_selection(
            self._rows(20.0), _FlatEnsemble(), [], {}, ranker=None
        )

        assert result["n_held_out_queries"] == 1


class _FlatEnsemble:
    """Predicts the same ratio for everything, so the selector's pick is decided
    by the cost tie-break and these tests are about the oracle only.

    `sample_member` is required because `_evaluate_selection` scores every
    policy including `thompson`, which draws one ensemble member per decision.
    """

    def predict(self, X):
        return [0.0] * len(X)

    def predict_mean_std(self, X):
        return [0.0] * len(X), [0.0] * len(X)

    def sample_member(self, rng=None):
        return self


@pytest.mark.parametrize("hint", [
    "/*+ Set(enable_mergejoin off) */",
    "/*+ Set(enable_incremental_sort off) */",
])
def test_the_families_that_actually_bit(hint):
    """Both rules that produced a false oracle on the real log."""
    assert is_provable_noop(hint, PLAN)


class TestAllThreeScorersAgree:
    """2.47's lesson is that an action-space filter living in one scorer and not
    the others is how a replay reports +5.38% where the benchmark measures 0.0%.
    It was found in three places then and the fix was applied three times; this
    filter lives in one function and these check all three call it."""

    @pytest.mark.parametrize("module,function", [
        ("app.train", "_evaluate_selection"),
        ("app.retrain", "_score_bundle"),
        ("app.calibrate", "evaluate_setting"),
    ])
    def test_it_is_called(self, module, function):
        import importlib
        import inspect

        source = inspect.getsource(getattr(importlib.import_module(module), function))
        assert "without_provable_noops(" in source, (
            f"{module}.{function} computes a best-available over a pool that "
            f"still contains plans identical to native"
        )

    def test_there_is_only_one_implementation(self):
        """Two copies drift. `is_provable_noop` should be reached through the
        shared helper, not called again in a scorer."""
        import inspect

        from app import calibrate, retrain, train

        for module in (train, retrain):
            source = inspect.getsource(module)
            calls = source.count("is_provable_noop(")
            # `train.rows_without_provable_noops` is the row-level filter and
            # legitimately calls it once; nothing else should.
            assert calls <= 1, f"{module.__name__} calls is_provable_noop {calls} times"

        assert "def without_provable_noops(" in inspect.getsource(calibrate)

    def test_a_missing_baseline_plan_is_passed_through(self):
        """Every caller reaches this with whatever the log had. A query with no
        usable baseline plan must not lose its candidates."""
        from app.calibrate import without_provable_noops

        cands = [{"hint": NOOP, "actual_total_time_ms": 1.0}]

        assert without_provable_noops(cands, None) == cands
        assert without_provable_noops(cands, "not a dict") == cands

    def test_it_drops_what_it_should(self):
        from app.calibrate import without_provable_noops

        cands = [{"hint": NOOP, "actual_total_time_ms": 1.0},
                 {"hint": REAL, "actual_total_time_ms": 2.0}]

        assert without_provable_noops(cands, PLAN) == [cands[1]]

    def test_a_candidate_with_no_hint_survives(self):
        """Defensive: a row with a null hint is not a no-op, it is a row the
        caller should already have filtered as a baseline."""
        from app.calibrate import without_provable_noops

        cands = [{"hint": None, "actual_total_time_ms": 1.0}]

        assert without_provable_noops(cands, PLAN) == cands
