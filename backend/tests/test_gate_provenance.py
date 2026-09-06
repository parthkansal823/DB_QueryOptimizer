"""
A gate belongs to a model, not to a path.

`docs/WRITEUP.md` 2.73: `models/gate.json` recorded
`calibrated_against_bundle: models/plan_selector.pkl`, a promotion replaced the
model at that path, and both existing drift checks reported green. The gate file
said it was calibrated on 15 held-out queries while the served model held out
16, and nothing compared the two.
"""

import pytest

from app.calibrate import _servable_families, bundle_identity
from app.diagnostics import gate_report, warnings_for


def _bundle(held_out, columns=("a", "b"), target="log_ratio_vs_native"):
    return {
        "feature_columns": list(columns),
        "held_out_queries": list(held_out),
        "target": target,
        "uncertainty_scale": 1.1,
    }


class TestBundleIdentity:
    def test_the_same_bundle_hashes_the_same(self):
        a = _bundle(["q1", "q2"])
        assert bundle_identity(a) == bundle_identity(_bundle(["q1", "q2"]))

    def test_a_different_held_out_set_is_a_different_model(self):
        """The exact change a retrain makes, and the one that went undetected."""
        assert bundle_identity(_bundle(["q1", "q2"])) != bundle_identity(
            _bundle(["q1", "q2", "q3"])
        )

    def test_a_different_feature_vector_is_a_different_model(self):
        """2.69 dropped 23 features and the gate would otherwise have carried
        over unchanged."""
        assert bundle_identity(_bundle(["q1"], columns=("a", "b"))) != bundle_identity(
            _bundle(["q1"], columns=("a",))
        )

    def test_held_out_order_does_not_matter(self):
        """Two runs that held out the same queries are the same model, whatever
        order the list came back in."""
        assert bundle_identity(_bundle(["q2", "q1"])) == bundle_identity(
            _bundle(["q1", "q2"])
        )

    def test_it_is_short_enough_to_read_in_a_report(self):
        assert len(bundle_identity(_bundle(["q1"]))) <= 16


class TestTheDriftIsReported:
    @staticmethod
    def _warnings(gate_extra, bundle):
        gate = {
            "confidence_z": 1.0,
            "min_relative_gain": 0.05,
            "calibrated_on_n_queries": 40,
            "expected_regression_rate": 0.0,
            "calibrated_with_policy": "pairwise_rank",
            **gate_extra,
        }
        report = gate_report(gate, bundle)
        return warnings_for(report, {"uncertainty_scale": 1.0}, {"n_rows": 999})

    def test_a_gate_from_another_model_is_flagged_high(self):
        bundle = _bundle(["q1", "q2"])
        found = self._warnings(
            {"calibrated_against_model": bundle_identity(_bundle(["q1"]))}, bundle
        )

        codes = {w["code"]: w for w in found}
        assert "gate_calibrated_for_another_model" in codes
        assert codes["gate_calibrated_for_another_model"]["severity"] == "high"

    def test_a_matching_gate_is_not_flagged(self):
        bundle = _bundle(["q1", "q2"])
        found = self._warnings(
            {"calibrated_against_model": bundle_identity(bundle)}, bundle
        )

        assert not any(
            w["code"].startswith("gate_calibrated_for_another_model") for w in found
        )
        assert not any(w["code"] == "gate_does_not_record_its_model" for w in found)

    def test_an_old_gate_file_says_so_rather_than_passing_silently(self):
        """Absent is not the same as matching. A gate written before the field
        existed cannot be checked, and reporting green would be the failure this
        whole file is about."""
        found = self._warnings({"source": "models/gate.json"}, _bundle(["q1"]))

        codes = {w["code"] for w in found}
        assert "gate_does_not_record_its_model" in codes


class TestTheGateRecordsWhatItWasSweptOver:
    """`app.autotune --apply` recorded the action space and
    `app.calibrate --apply` did not, so running the simpler tool erased the
    field and the drift check switched itself off."""

    def test_servable_families_follows_the_planner_flags(self, monkeypatch):
        from app.optimizer import planner

        for name in ("INCLUDE_TUNING", "INCLUDE_JOIN_METHODS", "INCLUDE_SCAN_METHODS",
                     "INCLUDE_NEGATIVE_SCANS", "INCLUDE_INDEX_PINNING",
                     "INCLUDE_MEMOIZE", "INCLUDE_PARALLEL"):
            monkeypatch.setattr(planner, name, False)

        assert _servable_families() == {"join_order", "operator_toggle", "cardinality"}

        monkeypatch.setattr(planner, "INCLUDE_TUNING", True)
        assert "resource" in _servable_families()

    @pytest.mark.parametrize("path", ["replay", "measured"])
    def test_both_apply_paths_record_it(self, path):
        import inspect

        from app import calibrate

        source = inspect.getsource(calibrate)
        assert source.count('"calibrated_with_families": sorted(_servable_families())') == 2, (
            "both --apply routes must stamp the action space, or using one of "
            "them silently erases what the other recorded"
        )

    def test_both_apply_paths_record_the_model(self):
        import inspect

        from app import calibrate

        source = inspect.getsource(calibrate)
        assert source.count('"calibrated_against_model": bundle_identity(bundle)') == 2
