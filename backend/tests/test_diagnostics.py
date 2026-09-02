"""
The "should I believe this model" report.

Every finding it emits is one that had actually shipped and was invisible from
the running system: a gate with `confidence_z = 0`, chosen in-sample, on a
handful of queries, over an uncertainty understated by 2.4x. `/model/status`
and `/stats/trend` could not have told you any of them.

These cover the two things a diagnostic has to get right -- it must fire when
something is wrong, and it must stay quiet when nothing is -- plus the reading
of files that may be absent, stale or corrupt, since it is pointed at whatever
`models/` happens to contain.
"""

import json
import pickle

import pytest

from app import diagnostics


class _Conformal:
    def __init__(self, q, alpha=0.1, n=200):
        self.q, self.alpha, self.n_calibration = q, alpha, n


def _write(tmp_path, *, gate=None, bundle=None, evaluation=None):
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    paths = {}
    if gate is not None:
        (models / "gate.json").write_text(json.dumps(gate))
    if evaluation is not None:
        (models / "eval.json").write_text(json.dumps(evaluation))
    if bundle is not None:
        with open(models / "m.pkl", "wb") as f:
            pickle.dump(bundle, f)
    paths["model"] = str(models / "m.pkl")
    paths["gate"] = str(models / "gate.json")
    paths["eval"] = str(models / "eval.json")
    return paths


def _codes(report):
    return {w["code"] for w in report["warnings"]}


HEALTHY_GATE = {
    "confidence_z": 1.0,
    "min_relative_gain": 0.05,
    "calibrated_on_n_queries": 40,
    "expected_regression_rate": 0.0,
}
HEALTHY_BUNDLE = {
    "target": "log_ratio_vs_native",
    "uncertainty_scale": 1.1,
    "held_out_queries": [f"q{i}" for i in range(12)],
    "conformal": _Conformal(1.8),
}
HEALTHY_EVAL = {
    "n_rows_train": 4000,
    "n_rows_test": 1200,
    "test_mae_log_ratio": 0.08,
    "test_mean_uncertainty_log_ratio": 0.07,
    "test_typical_multiplicative_error": 1.08,
}


# -- staying quiet when nothing is wrong ------------------------------------


def test_a_healthy_model_raises_nothing(tmp_path):
    paths = _write(tmp_path, gate=HEALTHY_GATE, bundle=HEALTHY_BUNDLE, evaluation=HEALTHY_EVAL)

    report = diagnostics.diagnose(paths["model"], paths["gate"], paths["eval"])

    assert report["warnings"] == []
    assert report["model_loaded"] is True
    assert report["gate"]["consults_uncertainty"] is True


# -- the four findings that had shipped -------------------------------------


def test_a_gate_that_ignores_uncertainty_is_flagged(tmp_path):
    """`confidence_z = 0` multiplies the ensemble's spread by zero. This is
    what made the dashboard print 'pessimistically 0.92x' next to a prediction
    of 0.92x."""
    paths = _write(
        tmp_path, gate={**HEALTHY_GATE, "confidence_z": 0.0},
        bundle=HEALTHY_BUNDLE, evaluation=HEALTHY_EVAL,
    )

    report = diagnostics.diagnose(paths["model"], paths["gate"], paths["eval"])

    assert "gate_ignores_uncertainty" in _codes(report)
    assert report["gate"]["consults_uncertainty"] is False


def test_a_gate_calibrated_on_a_handful_of_queries_is_flagged(tmp_path):
    paths = _write(
        tmp_path, gate={**HEALTHY_GATE, "calibrated_on_n_queries": 5},
        bundle=HEALTHY_BUNDLE, evaluation=HEALTHY_EVAL,
    )

    report = diagnostics.diagnose(paths["model"], paths["gate"], paths["eval"])

    assert "gate_calibrated_on_too_few_queries" in _codes(report)


def test_a_bundle_that_cannot_prove_it_was_calibrated_out_of_sample_is_flagged(tmp_path):
    """Without recorded held-out ids `app.calibrate` sweeps every logged query,
    including the ones the model fitted."""
    bundle = {k: v for k, v in HEALTHY_BUNDLE.items() if k != "held_out_queries"}
    paths = _write(tmp_path, gate=HEALTHY_GATE, bundle=bundle, evaluation=HEALTHY_EVAL)

    report = diagnostics.diagnose(paths["model"], paths["gate"], paths["eval"])

    assert "calibration_may_be_in_sample" in _codes(report)


def test_an_understated_uncertainty_is_flagged(tmp_path):
    paths = _write(
        tmp_path, gate=HEALTHY_GATE,
        bundle={**HEALTHY_BUNDLE, "uncertainty_scale": 2.4}, evaluation=HEALTHY_EVAL,
    )

    report = diagnostics.diagnose(paths["model"], paths["gate"], paths["eval"])

    assert "uncertainty_understated" in _codes(report)
    # The measured factor has to reach the reader; a warning that only says
    # "understated" gives them nothing to act on.
    assert "2.4x" in next(
        w["detail"] for w in report["warnings"] if w["code"] == "uncertainty_understated"
    )


def test_error_on_the_scale_of_the_headroom_is_flagged(tmp_path):
    """The project's oldest limitation, stated as a finding rather than left in
    the writeup."""
    paths = _write(
        tmp_path, gate=HEALTHY_GATE, bundle=HEALTHY_BUNDLE,
        evaluation={**HEALTHY_EVAL, "test_typical_multiplicative_error": 1.43},
    )

    report = diagnostics.diagnose(paths["model"], paths["gate"], paths["eval"])

    assert "error_comparable_to_headroom" in _codes(report)


def test_an_unusable_coverage_bound_is_reported_as_low_severity(tmp_path):
    """It is correct and not actionable, so it must not shout over the gate."""
    paths = _write(
        tmp_path, gate=HEALTHY_GATE,
        bundle={**HEALTHY_BUNDLE, "conformal": _Conformal(10.5)}, evaluation=HEALTHY_EVAL,
    )

    report = diagnostics.diagnose(paths["model"], paths["gate"], paths["eval"])
    conformal_warning = next(
        w for w in report["warnings"] if w["code"] == "no_usable_coverage_bound"
    )

    assert conformal_warning["severity"] == "low"


def test_an_infinite_conformal_bound_does_not_crash_the_report(tmp_path):
    paths = _write(
        tmp_path, gate=HEALTHY_GATE,
        bundle={**HEALTHY_BUNDLE, "conformal": _Conformal(float("inf"))},
        evaluation=HEALTHY_EVAL,
    )

    assert diagnostics.diagnose(paths["model"], paths["gate"], paths["eval"])


# -- ordering -------------------------------------------------------------


def test_findings_are_ordered_worst_first(tmp_path):
    paths = _write(
        tmp_path,
        gate={**HEALTHY_GATE, "confidence_z": 0.0, "calibrated_on_n_queries": 5},
        bundle={**HEALTHY_BUNDLE, "uncertainty_scale": 2.45, "conformal": _Conformal(10.5)},
        evaluation={**HEALTHY_EVAL, "test_typical_multiplicative_error": 1.43},
    )

    severities = [w["severity"] for w in diagnostics.diagnose(
        paths["model"], paths["gate"], paths["eval"]
    )["warnings"]]

    assert severities == sorted(severities, key=["high", "medium", "low"].index)


# -- reading whatever models/ happens to contain ----------------------------


def test_a_missing_model_is_reported_rather_than_raised(tmp_path):
    report = diagnostics.diagnose(
        str(tmp_path / "absent.pkl"), str(tmp_path / "absent.json"), str(tmp_path / "absent.json")
    )

    assert report["model_loaded"] is False
    assert report["gate"]["source"] == "built-in defaults"


def test_a_corrupt_bundle_is_a_finding_not_a_crash(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "m.pkl").write_bytes(b"not a pickle at all")

    report = diagnostics.diagnose(str(models / "m.pkl"), str(models / "g.json"), str(models / "e.json"))

    assert report["model_loaded"] is False


def test_a_damaged_gate_file_falls_back_to_defaults(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "gate.json").write_text("{ not json")

    report = diagnostics.diagnose(str(models / "m.pkl"), str(models / "gate.json"), str(models / "e.json"))

    assert report["gate"]["source"] == "built-in defaults"


def test_a_bundle_without_a_measured_scale_reads_as_one(tmp_path):
    """Bundles trained before the measurement existed must not be flagged for
    an understatement nobody measured."""
    bundle = {k: v for k, v in HEALTHY_BUNDLE.items() if k != "uncertainty_scale"}
    paths = _write(tmp_path, gate=HEALTHY_GATE, bundle=bundle, evaluation=HEALTHY_EVAL)

    report = diagnostics.diagnose(paths["model"], paths["gate"], paths["eval"])

    assert report["uncertainty"]["uncertainty_scale"] == 1.0
    assert "uncertainty_understated" not in _codes(report)


def test_the_report_is_json_serialisable(tmp_path):
    """`/model/diagnostics` returns it straight out of FastAPI."""
    paths = _write(tmp_path, gate=HEALTHY_GATE, bundle=HEALTHY_BUNDLE, evaluation=HEALTHY_EVAL)

    report = diagnostics.diagnose(paths["model"], paths["gate"], paths["eval"])

    assert json.loads(json.dumps(report, default=str))


def test_it_renders_without_a_model(tmp_path):
    report = diagnostics.diagnose(
        str(tmp_path / "a.pkl"), str(tmp_path / "b.json"), str(tmp_path / "c.json")
    )

    assert "=== safety gate ===" in diagnostics._format(report)


@pytest.mark.parametrize("severity", ["high", "medium", "low"])
def test_every_finding_carries_a_code_and_an_explanation(tmp_path, severity):
    paths = _write(
        tmp_path,
        gate={**HEALTHY_GATE, "confidence_z": 0.0, "calibrated_on_n_queries": 5},
        bundle={**HEALTHY_BUNDLE, "uncertainty_scale": 2.45, "conformal": _Conformal(10.5)},
        evaluation={**HEALTHY_EVAL, "test_typical_multiplicative_error": 1.43},
    )

    for w in diagnostics.diagnose(paths["model"], paths["gate"], paths["eval"])["warnings"]:
        assert w["code"] and len(w["detail"]) > 40


# -- is the gate keeping the promise it was calibrated on? -------------------
#
# `app.calibrate` writes `expected_regression_rate` into models/gate.json --
# measured, on a held-out sweep, at the moment the gate was chosen. That is a
# prediction about future behaviour and nothing had ever checked it. A gate
# whose regressions have climbed above what it promised is better evidence that
# the model is stale than `AUTO_RETRAIN_SECONDS` ticking over.


class _StatsCursor:
    """Stands in for the execution log via `stats.served_vs_native`."""

    def __init__(self, quality):
        self.quality = quality


def _patch_stats(monkeypatch, quality):
    from app import stats

    monkeypatch.setattr(
        stats, "served_vs_native", lambda cur, **kw: {"decision_quality": quality}
    )


def _quality(win=0, wash=0, loss=0, held=0):
    return {
        "deviated_win": win, "deviated_wash": wash, "deviated_loss": loss,
        "held_correct": held, "held_missed": 0, "regression_ms": 0.0, "saved_ms": 0.0,
    }


def test_a_gate_meeting_its_promise_raises_nothing(tmp_path, monkeypatch):
    _patch_stats(monkeypatch, _quality(win=18, wash=2, loss=0, held=5))
    paths = _write(tmp_path, gate=HEALTHY_GATE, bundle=HEALTHY_BUNDLE, evaluation=HEALTHY_EVAL)

    report = diagnostics.diagnose(paths["model"], paths["gate"], paths["eval"], cur=object())

    assert "gate_promise_not_met" not in _codes(report)
    assert report["drift"]["observed_regression_rate"] == 0.0


def test_regressions_above_the_promise_are_flagged(tmp_path, monkeypatch):
    _patch_stats(monkeypatch, _quality(win=10, wash=2, loss=8, held=5))
    paths = _write(tmp_path, gate=HEALTHY_GATE, bundle=HEALTHY_BUNDLE, evaluation=HEALTHY_EVAL)

    report = diagnostics.diagnose(paths["model"], paths["gate"], paths["eval"], cur=object())

    assert "gate_promise_not_met" in _codes(report)


def test_a_small_overshoot_is_within_tolerance(tmp_path, monkeypatch):
    """Rates over a few dozen runs are noisy, and a gate that promised 0% will
    exceed it the first time anything goes wrong."""
    _patch_stats(monkeypatch, _quality(win=19, wash=0, loss=1, held=5))
    paths = _write(tmp_path, gate=HEALTHY_GATE, bundle=HEALTHY_BUNDLE, evaluation=HEALTHY_EVAL)

    report = diagnostics.diagnose(paths["model"], paths["gate"], paths["eval"], cur=object())

    assert "gate_promise_not_met" not in _codes(report)


def test_too_few_runs_are_not_evidence_of_drift(tmp_path, monkeypatch):
    _patch_stats(monkeypatch, _quality(win=1, loss=2, held=1))
    paths = _write(tmp_path, gate=HEALTHY_GATE, bundle=HEALTHY_BUNDLE, evaluation=HEALTHY_EVAL)

    report = diagnostics.diagnose(paths["model"], paths["gate"], paths["eval"], cur=object())

    assert report["drift"]["enough_runs_to_judge"] is False
    assert "gate_promise_not_met" not in _codes(report)


def test_an_optimizer_that_never_deviates_is_flagged(tmp_path, monkeypatch):
    """A gate so strict that nothing gets through is a no-op wearing a safety
    mechanism's clothes, and the latency chart cannot tell you that."""
    _patch_stats(monkeypatch, _quality(held=25))
    paths = _write(tmp_path, gate=HEALTHY_GATE, bundle=HEALTHY_BUNDLE, evaluation=HEALTHY_EVAL)

    report = diagnostics.diagnose(paths["model"], paths["gate"], paths["eval"], cur=object())

    assert "optimizer_never_deviates" in _codes(report)


def test_the_drift_check_is_skipped_without_a_cursor(tmp_path):
    """`diagnose()` has to stay usable with no database -- the same constraint
    app/db.py exists to preserve."""
    paths = _write(tmp_path, gate=HEALTHY_GATE, bundle=HEALTHY_BUNDLE, evaluation=HEALTHY_EVAL)

    assert diagnostics.diagnose(paths["model"], paths["gate"], paths["eval"])["drift"] is None


def test_an_unreadable_log_does_not_break_the_rest_of_the_report(tmp_path, monkeypatch):
    from app import stats

    def boom(cur, **kw):
        raise RuntimeError("no such table")

    monkeypatch.setattr(stats, "served_vs_native", boom)
    paths = _write(tmp_path, gate=HEALTHY_GATE, bundle=HEALTHY_BUNDLE, evaluation=HEALTHY_EVAL)

    report = diagnostics.diagnose(paths["model"], paths["gate"], paths["eval"], cur=object())

    assert report["drift"] is None
    assert report["model_loaded"] is True


def test_a_gate_with_no_recorded_promise_cannot_drift(tmp_path, monkeypatch):
    """Gates written before calibration recorded its expectations."""
    _patch_stats(monkeypatch, _quality(win=5, loss=15, held=5))
    gate = {k: v for k, v in HEALTHY_GATE.items() if k != "expected_regression_rate"}
    paths = _write(tmp_path, gate=gate, bundle=HEALTHY_BUNDLE, evaluation=HEALTHY_EVAL)

    report = diagnostics.diagnose(paths["model"], paths["gate"], paths["eval"], cur=object())

    assert report["drift"] is None


# -- which model the evidence describes ---------------------------------------


class TestServedEvaluation:
    """
    The panel exists to answer "should I believe this model". It was answering
    it about a *different* model.

    `eval_path` is a fixed filename; the deployed model is whatever
    `model_store` last promoted. Those drifted apart, and the dashboard reported
    **-5.9% headroom captured** from a stale `models/eval_results.json` while
    the promoted version measured **+79.2%**. Worse than useless: a panel whose
    whole job is trustworthiness, quietly describing something else.
    """

    def test_the_registry_is_preferred_over_the_file(self, monkeypatch, tmp_path):
        from app import diagnostics

        registry_metrics = {"headroom_captured_vs_oracle": {"learned_greedy": 0.79}}

        class _Store:
            @staticmethod
            def current_version():
                return "v-new"

            @staticmethod
            def list_versions():
                return [
                    {"version_id": "v-old", "metrics": {"headroom_captured_vs_oracle": {}}},
                    {"version_id": "v-new", "metrics": registry_metrics},
                ]

        monkeypatch.setattr("app.model_store.current_version", _Store.current_version)
        monkeypatch.setattr("app.model_store.list_versions", _Store.list_versions)
        evaluation, source = diagnostics._served_evaluation(diagnostics.DEFAULT_EVAL_PATH)
        assert evaluation == registry_metrics
        assert "v-new" in source

    def test_the_file_is_the_fallback(self, monkeypatch, tmp_path):
        """A deployment older than the registry still has to report something."""
        from app import diagnostics

        path = tmp_path / "eval.json"
        path.write_text(json.dumps({"n_rows_train": 7}))

        class _Store:
            @staticmethod
            def current_version():
                return None

            @staticmethod
            def list_versions():
                return []

        monkeypatch.setattr("app.model_store.current_version", _Store.current_version)
        monkeypatch.setattr("app.model_store.list_versions", _Store.list_versions)
        evaluation, source = diagnostics._served_evaluation(str(path))
        assert evaluation == {"n_rows_train": 7}
        assert source == str(path)

    def test_a_broken_registry_falls_back_rather_than_failing(self, monkeypatch, tmp_path):
        from app import diagnostics

        path = tmp_path / "eval.json"
        path.write_text(json.dumps({"n_rows_train": 3}))

        def _explode():
            raise RuntimeError("registry unreadable")

        monkeypatch.setattr("app.model_store.current_version", _explode)
        evaluation, source = diagnostics._served_evaluation(str(path))
        assert evaluation == {"n_rows_train": 3}

    def test_a_promoted_version_without_metrics_falls_back(self, monkeypatch, tmp_path):
        """
        Reporting an empty evidence section as though it described the served
        model is the same mistake in a quieter form.
        """
        from app import diagnostics

        path = tmp_path / "eval.json"
        path.write_text(json.dumps({"n_rows_train": 11}))

        class _Store:
            @staticmethod
            def current_version():
                return "v1"

            @staticmethod
            def list_versions():
                return [{"version_id": "v1", "metrics": {}}]

        monkeypatch.setattr("app.model_store.current_version", _Store.current_version)
        monkeypatch.setattr("app.model_store.list_versions", _Store.list_versions)
        evaluation, _ = diagnostics._served_evaluation(str(path))
        assert evaluation == {"n_rows_train": 11}
