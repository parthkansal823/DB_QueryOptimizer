"""
A feature constant across a query's candidates cannot separate them.

`docs/WRITEUP.md` 2.69: 23 of 67 features take one value per query, the trees
spend 6.8% of their splits on them, and dropping all 23 improved held-out
capture on 11 of 16 paired splits (signed-rank p = 0.026). The project has
shipped this mistake three times, so the check is a tool now and the tool has
tests.
"""

import pytest

from app.feature_audit import format_report
from app.train import train


def _report(features):
    return {
        "bundle": "models/test.pkl",
        "n_queries": 65,
        "n_features": len(features),
        "features": features,
    }


def _feature(name, varies, splits):
    return {
        "name": name,
        "varies_on_queries": varies,
        "varies_pct": varies / 65 * 100,
        "splits": splits,
    }


def test_a_feature_constant_within_every_query_is_named():
    text = format_report(_report([
        _feature("has_hint", 0, 255),
        _feature("cost_concentration", 65, 1069),
    ]))

    assert "constant across every query's candidates: **1/2**" in text
    assert "has_hint" in text


def test_the_splits_spent_on_them_are_counted():
    """Being constant is the defect; the splits are what it costs. A feature the
    trees ignore is harmless, so reporting only the count would overstate it."""
    text = format_report(_report([
        _feature("has_hint", 0, 255),
        _feature("num_joins", 0, 0),
        _feature("cost_concentration", 65, 745),
    ]))

    assert "255 of 1,000" in text
    assert "25.5% of the model's capacity" in text


def test_a_vector_with_nothing_constant_says_so():
    text = format_report(_report([
        _feature("cost_concentration", 65, 1069),
        _feature("total_cost", 65, 1525),
    ]))

    assert "constant across every query's candidates: **0/2**" in text


def test_a_model_that_cannot_report_splits_still_audits():
    """The variance half needs no model at all. A bundle from a class without
    split counts should lose the capacity line, not the report."""
    text = format_report(_report([
        _feature("has_hint", 0, None),
        _feature("cost_concentration", 65, None),
    ]))

    assert "constant across every query's candidates: **1/2**" in text
    assert "capacity" not in text


def test_a_feature_that_varies_but_is_never_split_on_is_reported_separately():
    """Different defect, different fix: this one is available and unused rather
    than used and useless."""
    text = format_report(_report([
        _feature("log_total_cost", 65, 0),
        _feature("cost_concentration", 65, 1069),
    ]))

    assert "varies within a query and never split on anyway: 1" in text
    assert "log_total_cost" in text


def test_excluding_an_unknown_feature_is_an_error(tmp_path, monkeypatch):
    """A typo in an ablation runs the control arm twice and reports the feature
    as worthless. Loud, because the failure produces a plausible number.

    The database is stubbed out: the ablation check runs on the column list, and
    what is being tested is that an unrecognised name stops the run rather than
    silently doing nothing."""
    from app import train as train_module

    monkeypatch.setattr(train_module, "_load_rows", lambda *a, **k: [{
        "query_id": "q", "sql_text": "SELECT 1", "hint": None, "is_baseline": True,
        "raw_plan": {"Node Type": "Seq Scan", "Relation Name": "users",
                     "Alias": "u", "Plan Rows": 1, "Plan Width": 8},
        "total_cost": 1.0, "actual_total_time_ms": 1.0,
        "created_at": None, "selector_used": "collection",
    }])
    monkeypatch.setattr(train_module, "_discover_cardinalities", lambda: {"users": 10.0})

    with pytest.raises(ValueError, match="not features of this schema"):
        train(
            model_path=str(tmp_path / "challenger.pkl"),
            eval_path=str(tmp_path / "e.json"),
            exclude_features=("no_such_feature",),
        )


def test_dropping_a_name_removes_the_column_rather_than_zeroing_it():
    """
    Why the ablation is trustworthy: `to_vector` reads features by name, so a
    name absent from `feature_columns` produces a vector that is one element
    shorter -- not a zeroed column the trees can still split on.

    A zeroing implementation would make every ablation report the feature as
    harmless, which is the answer it is trying to measure.
    """
    from app.optimizer.features import build_feature_columns, to_vector

    columns = build_feature_columns(["users"])
    assert "has_hint" in columns, (
        "the feature 2.69 names has to exist for this to mean anything"
    )
    features = {name: 1.0 for name in columns}

    kept = to_vector(features, columns)
    ablated = to_vector(features, [c for c in columns if c != "has_hint"])

    assert len(ablated) == len(kept) - 1
    assert 0.0 not in ablated
