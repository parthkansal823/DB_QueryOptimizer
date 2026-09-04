import pytest

from app import retrain


def _bundle(name):
    return {"model": name, "feature_columns": ["f"], "table_cardinalities": {"t": 100.0}}


def _rows_by_query(latencies):
    """One query, one candidate per latency.

    The hints are real `Leading(...)` syntax rather than placeholders because
    `_score_bundle` filters to the families a served query is actually offered
    (`calibrate._servable`). A placeholder classifies as no family at all and is
    dropped, which is the correct behaviour and would silently empty the
    fixture."""
    return {
        "q1": [
            {
                "query_id": "q1",
                "hint": f"/*+ Leading(a b{i}) */",
                "is_baseline": False,
                "raw_plan": {"Node Type": "Seq Scan", "Relation Name": "t", "Alias": "t", "Plan Rows": 1},
                "total_cost": 10.0,
                "actual_total_time_ms": lat,
            }
            for i, lat in enumerate(latencies)
        ]
    }


class _PickIndex:
    """Stands in for a fitted ensemble: always 'predicts' the given scores."""

    def __init__(self, scores):
        self.scores = scores

    def predict_mean_std(self, X):
        return self.scores[: len(X)], [0.0] * len(X)


@pytest.fixture
def patched_store(monkeypatch):
    state = {"current": None, "versions": {}, "promotions": []}

    monkeypatch.setattr(retrain.model_store, "current_version", lambda: state["current"])
    monkeypatch.setattr(retrain.model_store, "load_version", lambda v: state["versions"][v])

    def save_version(bundle, metrics, version_id=None):
        vid = version_id or f"v{len(state['versions']) + 1}"
        state["versions"][vid] = bundle
        return vid

    def promote(version_id, reason=""):
        state["current"] = version_id
        state["promotions"].append((version_id, reason))

    monkeypatch.setattr(retrain.model_store, "save_version", save_version)
    monkeypatch.setattr(retrain.model_store, "promote", promote)
    return state


def test_skips_retraining_when_too_little_new_data(monkeypatch):
    monkeypatch.setattr(retrain, "rows_since_last_training", lambda: 10)
    result = retrain.retrain_if_needed(min_new_rows=200)
    assert result["action"] == "skipped"
    assert result["new_rows"] == 10


def test_force_overrides_the_new_row_threshold(monkeypatch, patched_store):
    monkeypatch.setattr(retrain, "rows_since_last_training", lambda: 0)
    monkeypatch.setattr(retrain, "train", lambda **kw: {"test_mae_ms": 1.0})
    monkeypatch.setattr(retrain, "_load_challenger_bundle", lambda p: _bundle("challenger"))
    monkeypatch.setattr(retrain, "_held_out_rows", lambda **kw: _rows_by_query([10.0]))

    result = retrain.retrain_if_needed(force=True)
    # No champion yet -> unconditional promotion.
    assert result["action"] == "promoted"
    assert result["reason"] == "no incumbent to compare against"


def test_scoring_returns_none_for_a_missing_bundle():
    assert retrain._score_bundle(None, _rows_by_query([1.0])) is None



def model_store_reason(patched_store) -> str:
    """The reason recorded with the most recent promotion, or ""."""
    promotions = patched_store.get("promotions") or []
    return str(promotions[-1][1]) if promotions else ""

def test_scoring_picks_the_candidate_the_model_prefers(monkeypatch):
    bundle = _bundle("m")
    bundle["model"] = _PickIndex([99.0, 1.0])  # prefers index 1
    monkeypatch.setattr(retrain, "featurize", lambda c, card: {"f": 0.0})
    monkeypatch.setattr(retrain, "to_vector", lambda f, cols: [0.0])

    score = retrain._score_bundle(
        bundle, _rows_by_query([500.0, 20.0]), policy="greedy"
    )
    assert score["mean_ms"] == 20.0
    # 20.0 is also the best candidate available, so the worst pick threw nothing
    # away. `worst_ratio_vs_best` is the number the old float score could not
    # express, and it is what promotion now reads alongside the mean.
    assert score["worst_ratio_vs_best"] == 1.0
    assert score["n_queries"] == 1


def test_scoring_reports_how_bad_the_worst_pick_was(monkeypatch):
    """
    The measurement the mean could not make.

    A PG16-trained model left deployed after the upgrade to 18 predicted 0.291x
    for a candidate its own training data measures at 1.134x, and served it.
    Against the PG18-trained challenger it scored **1.0% apart** on the mean --
    below the promotion bar -- because on most queries the two pick the same
    plan. The whole difference lived in the few where they diverged.
    """
    bundle = _bundle("m")
    bundle["model"] = _PickIndex([1.0, 99.0])  # prefers index 0, the slow one
    monkeypatch.setattr(retrain, "featurize", lambda c, card: {"f": 0.0})
    monkeypatch.setattr(retrain, "to_vector", lambda f, cols: [0.0])

    score = retrain._score_bundle(
        bundle, _rows_by_query([500.0, 20.0]), policy="greedy"
    )
    assert score["mean_ms"] == 500.0
    assert score["worst_ratio_vs_best"] == pytest.approx(25.0)


def test_challenger_must_clear_the_improvement_bar(monkeypatch, patched_store):
    """A marginal offline win is noise (see WRITEUP 2.2.1) -- don't promote on it."""
    patched_store["current"] = "champ"
    patched_store["versions"]["champ"] = _bundle("champ")

    monkeypatch.setattr(retrain, "rows_since_last_training", lambda: 1000)
    monkeypatch.setattr(retrain, "train", lambda **kw: {"test_mae_ms": 1.0})
    monkeypatch.setattr(retrain, "_held_out_rows", lambda **kw: _rows_by_query([10.0]))
    monkeypatch.setattr(retrain, "_load_challenger_bundle", lambda p: _bundle("chal"))

    # Champion 100 ms, challenger 99 ms -- 1% better, below the 2% bar -- and
    # both pick equally well at their worst, so neither route to promotion opens.
    scores = iter([
        {"mean_ms": 100.0, "worst_ratio_vs_best": 1.5, "n_queries": 1},
        {"mean_ms": 99.0, "worst_ratio_vs_best": 1.5, "n_queries": 1},
    ])
    monkeypatch.setattr(retrain, "_score_bundle", lambda b, r: next(scores))

    result = retrain.retrain_if_needed(min_improvement=0.02)
    assert result["action"] == "rejected"
    assert patched_store["current"] == "champ"  # unchanged


def test_a_much_better_worst_case_is_promoted_on_its_own(monkeypatch, patched_store):
    """
    The second route to promotion, and the one that was missing.

    A challenger that turns a catastrophic prediction into a correct one may
    barely move the mean -- measured at 1.0% -- while being the difference
    between serving a plan 17% slower than native and declining to serve it.
    Judged on the mean alone it was rejected and the broken model kept serving.
    """
    patched_store["current"] = "champ"
    patched_store["versions"]["champ"] = _bundle("champ")

    monkeypatch.setattr(retrain, "rows_since_last_training", lambda: 1000)
    monkeypatch.setattr(retrain, "train", lambda **kw: {"test_mae_ms": 1.0})
    monkeypatch.setattr(retrain, "_held_out_rows", lambda **kw: _rows_by_query([10.0]))
    monkeypatch.setattr(retrain, "_load_challenger_bundle", lambda p: _bundle("chal"))

    scores = iter([
        {"mean_ms": 100.0, "worst_ratio_vs_best": 4.0, "n_queries": 1},
        {"mean_ms": 99.0, "worst_ratio_vs_best": 1.1, "n_queries": 1},
    ])
    monkeypatch.setattr(retrain, "_score_bundle", lambda b, r: next(scores))

    result = retrain.retrain_if_needed(min_improvement=0.02)
    assert result["action"] == "promoted"
    assert "worst pick improved" in model_store_reason(patched_store)
    assert patched_store["current"] != "champ"


def test_a_worse_worst_case_is_refused_however_good_the_mean(monkeypatch, patched_store):
    """
    The other half of the rule. A model whose average is excellent and whose bad
    case is disastrous is not an improvement to a system that exists to avoid
    disasters.
    """
    patched_store["current"] = "champ"
    patched_store["versions"]["champ"] = _bundle("champ")

    monkeypatch.setattr(retrain, "rows_since_last_training", lambda: 1000)
    monkeypatch.setattr(retrain, "train", lambda **kw: {"test_mae_ms": 1.0})
    monkeypatch.setattr(retrain, "_held_out_rows", lambda **kw: _rows_by_query([10.0]))
    monkeypatch.setattr(retrain, "_load_challenger_bundle", lambda p: _bundle("chal"))

    scores = iter([
        {"mean_ms": 100.0, "worst_ratio_vs_best": 1.2, "n_queries": 1},
        {"mean_ms": 50.0, "worst_ratio_vs_best": 6.0, "n_queries": 1},  # 50% better mean
    ])
    monkeypatch.setattr(retrain, "_score_bundle", lambda b, r: next(scores))

    result = retrain.retrain_if_needed(min_improvement=0.02)
    assert result["action"] == "rejected"
    assert "worse worst case" in result["reason"]
    assert patched_store["current"] == "champ"


def test_clearly_better_challenger_is_promoted(monkeypatch, patched_store):
    patched_store["current"] = "champ"
    patched_store["versions"]["champ"] = _bundle("champ")

    monkeypatch.setattr(retrain, "rows_since_last_training", lambda: 1000)
    monkeypatch.setattr(retrain, "train", lambda **kw: {"test_mae_ms": 1.0})
    monkeypatch.setattr(retrain, "_load_challenger_bundle", lambda p: _bundle("chal"))
    monkeypatch.setattr(retrain, "_held_out_rows", lambda **kw: _rows_by_query([10.0]))

    scores = iter([
        {"mean_ms": 100.0, "worst_ratio_vs_best": 1.5, "n_queries": 1},
        {"mean_ms": 70.0, "worst_ratio_vs_best": 1.5, "n_queries": 1},  # 30% better
    ])
    monkeypatch.setattr(retrain, "_score_bundle", lambda b, r: next(scores))

    result = retrain.retrain_if_needed(min_improvement=0.02)
    assert result["action"] == "promoted"
    assert result["improvement"] == pytest.approx(0.30)
    assert patched_store["current"] != "champ"


def test_unscoreable_models_are_never_promoted(monkeypatch, patched_store):
    patched_store["current"] = "champ"
    patched_store["versions"]["champ"] = _bundle("champ")

    monkeypatch.setattr(retrain, "rows_since_last_training", lambda: 1000)
    monkeypatch.setattr(retrain, "train", lambda **kw: {"test_mae_ms": 1.0})
    monkeypatch.setattr(retrain, "_load_challenger_bundle", lambda p: _bundle("chal"))
    monkeypatch.setattr(retrain, "_held_out_rows", lambda **kw: _rows_by_query([10.0]))
    monkeypatch.setattr(retrain, "_score_bundle", lambda b, r: None)

    result = retrain.retrain_if_needed()
    assert result["action"] == "rejected"
    assert patched_store["current"] == "champ"


# -- the served policy is one value, in one place ---------------------------


def test_the_served_policy_is_the_shipped_default():
    """
    `served_policy()` reads `SELECTION_POLICY`, and for a long time nothing set
    it where this function could see it.

    `docker-compose.yml` set it on the backend service, so the variable existed
    inside that container and nowhere else -- while `app.retrain` and
    `app.calibrate` are run from the host. Promotion and gate calibration both
    scored `greedy`; the server served `pairwise_rank`. Measured on 15 held-out
    queries (docs/WRITEUP.md 2.43) those are not near-neighbours: `greedy` lands
    at 104.1% of native and `pairwise_rank` at 95.1%, so the calibrated gate was
    tuned against the one policy that loses.

    The compose override is deleted and the default lives once. This asserts
    the deletion stays deleted.
    """
    import os

    from app.optimizer.learned import DEFAULT_SELECTION_POLICY

    assert os.getenv("SELECTION_POLICY") is None, (
        "this test describes the default; unset SELECTION_POLICY to run it"
    )
    assert retrain.served_policy() == DEFAULT_SELECTION_POLICY


def test_the_dashboard_and_the_optimizer_name_the_same_default():
    """`app.settings` restates the default as plain data for the settings UI,
    which is exactly the shape of duplication 2.22 is about. It cannot import
    the constant without a cycle, so the agreement is asserted instead."""
    from app import settings
    from app.optimizer.learned import DEFAULT_SELECTION_POLICY

    assert settings.FIELDS["selection_policy"]["default"] == DEFAULT_SELECTION_POLICY


def test_the_compose_file_does_not_reintroduce_the_override():
    """The bug was not the value, it was naming it in a second place that only
    half the system can read."""
    import os
    import re

    path = os.path.join(os.path.dirname(__file__), "..", "..", "docker-compose.yml")
    text = open(path, encoding="utf-8").read()
    uncommented = [
        line for line in text.splitlines()
        if re.match(r"\s*SELECTION_POLICY\s*:", line)
    ]
    assert uncommented == [], f"compose sets SELECTION_POLICY again: {uncommented}"


def test_the_action_space_defaults_agree_between_settings_and_planner():
    """
    `app.settings.FIELDS` restates the planner's defaults as plain data for the
    dashboard, which is exactly the duplication 2.22 is about: a constant that
    duplicates a decision made elsewhere will eventually disagree with it.

    It already had, twice. `selection_policy` said `greedy` while the container
    served `pairwise_rank` (2.43), and `include_tuning` said True after the
    planner had been measured better off without it (2.48). Neither disagreement
    was detectable by running anything.
    """
    import os

    from app.optimizer import planner
    from app import settings

    for env in ("INCLUDE_TUNING", "INCLUDE_JOIN_METHODS", "INCLUDE_PARALLEL"):
        assert os.getenv(env) is None, f"unset {env} to run this test"

    assert settings.FIELDS["include_tuning"]["default"] == planner.INCLUDE_TUNING
    assert (
        settings.FIELDS["include_join_methods"]["default"]
        == planner.INCLUDE_JOIN_METHODS
    )
