"""
What promotion is measured on.

`_held_out_rows` promised "the most recent queries" and delivered
`sorted(query_ids)[-n:]` -- the alphabetical tail. Nothing about it was
chronological, while `created_at` sat unused on every row. On this workload the
alphabetical tail is `gen_2*` and `shape_*`: a group defined by naming.

The consequence was not cosmetic. The challenger compared in docs/WRITEUP.md
2.40 had **trained on 10 of the 15** queries in that yardstick, so promotion was
substantially reading how well it had memorised them -- a contest the newer
model wins by construction.
"""

import pytest

from app import retrain


def _rows(qid, stamp, latency=10.0):
    plan = {"Node Type": "Seq Scan", "Relation Name": "t", "Alias": "t",
            "Plan Rows": 1, "Plan Width": 8}
    return [
        {"query_id": qid, "created_at": stamp, "is_baseline": True, "hint": None,
         "raw_plan": plan, "total_cost": 10.0, "actual_total_time_ms": latency},
        {"query_id": qid, "created_at": stamp, "is_baseline": False,
         "hint": "/*+ Leading(a b) */", "raw_plan": plan, "total_cost": 10.0,
         "actual_total_time_ms": latency / 2},
    ]


@pytest.fixture
def logged(monkeypatch):
    def _install(rows):
        monkeypatch.setattr(retrain, "_load_rows", lambda: rows)
    return _install


# -- recency is measured, not assumed ---------------------------------------


def test_the_most_recent_queries_are_the_recent_ones_not_the_last_alphabetically():
    """`zzz_old` sorts last and was logged first; `aaa_new` sorts first and was
    logged last. The alphabetical version picks exactly the wrong one."""
    rows = _rows("zzz_old", "2026-01-01") + _rows("aaa_new", "2026-09-01")

    import app.retrain as m
    original = m._load_rows
    m._load_rows = lambda: rows
    try:
        picked = m._held_out_rows(fraction=0.5)
    finally:
        m._load_rows = original

    assert set(picked) == {"aaa_new"}


def test_rows_without_a_timestamp_still_produce_a_stable_order(logged):
    """A total order, or the selection is not reproducible between runs."""
    logged(_rows("b", None) + _rows("a", None) + _rows("c", None))

    first = retrain._held_out_rows(fraction=0.5)
    second = retrain._held_out_rows(fraction=0.5)
    assert set(first) == set(second)


def test_a_querys_latest_row_decides_its_recency(logged):
    """A query re-run today is recent even if it was first logged in January."""
    logged(
        _rows("old_but_rerun", "2026-01-01")
        + _rows("old_but_rerun", "2026-09-02")
        + _rows("newer", "2026-06-01")
    )

    assert set(retrain._held_out_rows(fraction=0.5)) == {"old_but_rerun"}


# -- contamination ----------------------------------------------------------


def _bundle(held):
    return {"held_out_queries": list(held)}


def test_queries_neither_model_trained_on_are_preferred(logged):
    logged(sum((_rows(f"q{i}", f"2026-0{i}-01") for i in range(1, 9)), []))
    clean = [f"q{i}" for i in range(1, 7)]

    picked = retrain._held_out_rows(
        fraction=1.0, bundles=[_bundle(clean), _bundle(clean)]
    )

    assert set(picked) <= set(clean)


def test_a_yardstick_too_small_to_read_falls_back_to_recency(logged):
    """Champion held out 8 queries, challenger 15, and the two lists intersect
    in two. Two queries cannot separate two models, and a clean sample that
    small is worse than a contaminated one that is reported as such."""
    logged(sum((_rows(f"q{i}", f"2026-0{i}-01") for i in range(1, 9)), []))
    barely = ["q1", "q2"]

    picked = retrain._held_out_rows(
        fraction=0.5, bundles=[_bundle(barely), _bundle(barely)]
    )

    assert len(picked) > len(barely) or set(picked) != set(barely)


def test_a_query_only_one_model_held_out_is_not_clean(logged):
    """Uncontaminated means neither model saw it. One is not enough."""
    logged(sum((_rows(f"q{i}", f"2026-0{i}-01") for i in range(1, 9)), []))

    picked = retrain._held_out_rows(
        fraction=1.0,
        bundles=[_bundle([f"q{i}" for i in range(1, 9)]), _bundle(["q1"])],
    )

    assert set(picked) != {"q1"}


# -- provenance -------------------------------------------------------------


def test_provenance_counts_what_each_model_had_already_seen():
    held = {"a": [], "b": [], "c": []}
    prov = retrain.yardstick_provenance(
        held,
        {"champion": _bundle(["a"]), "challenger": _bundle(["a", "b", "c"])},
    )

    assert prov["n_queries"] == 3
    assert prov["champion_trained_on"] == 2   # b and c
    assert prov["challenger_trained_on"] == 0


def test_a_bundle_that_never_recorded_its_split_is_unknown_not_clean():
    """"We cannot check" and "we checked and it is fine" must not look alike --
    the same distinction `apply_gate` had to learn about null metrics."""
    prov = retrain.yardstick_provenance({"a": []}, {"champion": {}})

    assert prov["champion_trained_on"] is None


# -- promotion must score the action space serving actually offers ----------


def _cand(hint, latency, disabled=False, baseline=False):
    plan = {"Node Type": "Seq Scan", "Relation Name": "t", "Alias": "t",
            "Plan Rows": 1, "Plan Width": 8}
    if disabled:
        plan = {**plan, "Disabled": True}
    return {"query_id": "q", "hint": hint, "is_baseline": baseline,
            "raw_plan": plan, "total_cost": 10.0, "actual_total_time_ms": latency}


class _PickFirst:
    """Mirrors `PairwisePlanRanker`'s signature, `scores=` included.

    It did not, and `_score_bundle` passing the pre-computed scores through --
    which the real ranker has accepted since the O(n^2) sweep was deduplicated
    -- made every call raise `TypeError` inside a `try` and return `None`. A
    stub narrower than the thing it stands for turns an interface change into
    four assertions about `None`."""

    def select(self, vectors, tie_break_costs=None, scores=None):
        return 0

    def scores(self, vectors):
        return [1.0] + [0.0] * (len(vectors) - 1)


def _bundle_with_ranker():
    class _Flat:
        def predict(self, X):
            return [0.0] * len(X)

        def predict_mean_std(self, X):
            return [0.0] * len(X), [0.0] * len(X)

    return {"model": _Flat(), "ranker": _PickFirst(), "feature_columns": [],
            "table_cardinalities": {}}


def test_native_is_not_one_of_the_candidates(monkeypatch):
    """Serving passes `select_plan(candidates, baseline_plan=native)`: the policy
    chooses among candidates and the gate decides whether to use the choice.
    Leaving the baseline row in the pool let a model score well by 'picking' a
    plan it never had the option to pick."""
    rows = {"q": [_cand(None, 5.0, baseline=True),
                  _cand("/*+ Leading(a b) */", 50.0)]}

    score = retrain._score_bundle(_bundle_with_ranker(), rows)

    assert score["mean_ms"] == 50.0


def test_plans_with_disabled_nodes_are_not_scored(monkeypatch):
    """`learned._is_unsafe` refuses these outright, so crediting a model for
    them measures picks the server vetoes (docs/WRITEUP.md 2.47)."""
    rows = {"q": [_cand(None, 100.0, baseline=True),
                  _cand("/*+ Set(enable_seqscan off) */", 5.0, disabled=True),
                  _cand("/*+ Leading(a b) */", 50.0)]}

    score = retrain._score_bundle(_bundle_with_ranker(), rows)

    assert score["mean_ms"] == 50.0


def test_unservable_hint_families_are_not_scored():
    """The log records everything ever collected; collection explores wider than
    serving does. `calibrate._servable` documents an all-candidates replay
    having every gate lose 10-25% on forced nested-loop plans the live path
    drops."""
    rows = {"q": [_cand(None, 100.0, baseline=True),
                  _cand("/*+ NestLoop(a b) */", 5.0),
                  _cand("/*+ Leading(a b) */", 50.0)]}

    score = retrain._score_bundle(_bundle_with_ranker(), rows)

    assert score["mean_ms"] == 50.0


def test_a_query_with_nothing_servable_is_skipped_not_counted():
    """Scoring it as native would credit both models identically for a query
    neither can act on, diluting the comparison."""
    rows = {
        "q": [_cand(None, 100.0, baseline=True), _cand("/*+ NestLoop(a b) */", 5.0)],
        "q2": [_cand(None, 100.0, baseline=True), _cand("/*+ Leading(a b) */", 50.0)],
    }
    for row in rows["q2"]:
        row["query_id"] = "q2"

    score = retrain._score_bundle(_bundle_with_ranker(), rows)

    assert score["n_queries"] == 1
