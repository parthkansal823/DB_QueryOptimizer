import pytest

from app import retrain


def _bundle(name):
    return {"model": name, "feature_columns": ["f"], "table_cardinalities": {"t": 100.0}}


def _rows_by_query(latencies):
    """One query, one candidate per latency."""
    return {
        "q1": [
            {
                "query_id": "q1",
                "hint": f"/*+ h{i} */",
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


def test_scoring_picks_the_candidate_the_model_prefers(monkeypatch):
    bundle = _bundle("m")
    bundle["model"] = _PickIndex([99.0, 1.0])  # prefers index 1
    monkeypatch.setattr(retrain, "featurize", lambda c, card: {"f": 0.0})
    monkeypatch.setattr(retrain, "to_vector", lambda f, cols: [0.0])

    score = retrain._score_bundle(bundle, _rows_by_query([500.0, 20.0]))
    assert score == 20.0


def test_challenger_must_clear_the_improvement_bar(monkeypatch, patched_store):
    """A marginal offline win is noise (see WRITEUP 2.2.1) -- don't promote on it."""
    patched_store["current"] = "champ"
    patched_store["versions"]["champ"] = _bundle("champ")

    monkeypatch.setattr(retrain, "rows_since_last_training", lambda: 1000)
    monkeypatch.setattr(retrain, "train", lambda **kw: {"test_mae_ms": 1.0})
    monkeypatch.setattr(retrain, "_held_out_rows", lambda **kw: _rows_by_query([10.0]))
    monkeypatch.setattr(retrain, "_load_challenger_bundle", lambda p: _bundle("chal"))

    # champion 100 ms, challenger 99 ms -> 1% better, below the 2% bar.
    scores = iter([100.0, 99.0])
    monkeypatch.setattr(retrain, "_score_bundle", lambda b, r: next(scores))

    result = retrain.retrain_if_needed(min_improvement=0.02)
    assert result["action"] == "rejected"
    assert patched_store["current"] == "champ"  # unchanged


def test_clearly_better_challenger_is_promoted(monkeypatch, patched_store):
    patched_store["current"] = "champ"
    patched_store["versions"]["champ"] = _bundle("champ")

    monkeypatch.setattr(retrain, "rows_since_last_training", lambda: 1000)
    monkeypatch.setattr(retrain, "train", lambda **kw: {"test_mae_ms": 1.0})
    monkeypatch.setattr(retrain, "_load_challenger_bundle", lambda p: _bundle("chal"))
    monkeypatch.setattr(retrain, "_held_out_rows", lambda **kw: _rows_by_query([10.0]))

    scores = iter([100.0, 70.0])  # 30% better
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
