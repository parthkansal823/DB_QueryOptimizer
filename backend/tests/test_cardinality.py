import math

import pytest

from app.optimizer.cardinality import (
    QERROR_FEATURES,
    CardinalityCorrector,
    build_training_set,
    scan_nodes_with_actuals,
)


def _scan(estimated, actual, loops=1, node_type="Seq Scan", filter_text=None):
    node = {
        "Node Type": node_type,
        "Relation Name": "orders",
        "Alias": "o",
        "Plan Rows": estimated,
        "Actual Rows": actual,
        "Actual Loops": loops,
        "Plan Width": 8,
    }
    if filter_text:
        node["Filter"] = filter_text
    return node


def test_qerror_is_zero_when_the_estimate_was_right():
    (_, log_qerror), = scan_nodes_with_actuals(_scan(1000, 1000))
    assert log_qerror == pytest.approx(0.0, abs=1e-6)


def test_underestimate_gives_positive_qerror():
    """Postgres said 10 rows, reality was 10000 -- the classic failure."""
    (_, log_qerror), = scan_nodes_with_actuals(_scan(10, 10_000))
    assert log_qerror > 0
    assert math.exp(log_qerror) == pytest.approx(10_001 / 11, rel=0.01)


def test_overestimate_gives_negative_qerror():
    (_, log_qerror), = scan_nodes_with_actuals(_scan(10_000, 10))
    assert log_qerror < 0


def test_zero_row_scans_do_not_blow_up():
    """Selective filters returning nothing are common; ratio must stay finite."""
    (_, log_qerror), = scan_nodes_with_actuals(_scan(500, 0))
    assert math.isfinite(log_qerror)


def test_loops_are_multiplied_into_actual_rows():
    """A nested-loop inner scan reports per-loop rows; the total is what matters."""
    (_, single), = scan_nodes_with_actuals(_scan(100, 100, loops=1))
    (_, looped), = scan_nodes_with_actuals(_scan(100, 100, loops=50))
    assert looped > single


def test_features_capture_filter_complexity():
    (features, _), = scan_nodes_with_actuals(
        _scan(100, 100, filter_text="(a = 1) AND (b = 2) AND (c = 3)")
    )
    assert features["has_filter"] == 1.0
    assert features["n_filter_conjuncts"] == 3.0


def test_scan_type_is_encoded():
    (seq, _), = scan_nodes_with_actuals(_scan(10, 10, node_type="Seq Scan"))
    (idx, _), = scan_nodes_with_actuals(_scan(10, 10, node_type="Index Scan"))
    assert (seq["is_seq_scan"], seq["is_index_scan"]) == (1.0, 0.0)
    assert (idx["is_seq_scan"], idx["is_index_scan"]) == (0.0, 1.0)


def test_only_nodes_with_actuals_are_collected():
    """A plan that was never executed contributes no training rows."""
    unexecuted = {"Node Type": "Seq Scan", "Relation Name": "orders", "Plan Rows": 10}
    assert list(scan_nodes_with_actuals(unexecuted)) == []


def test_join_nodes_are_skipped():
    plan = {
        "Node Type": "Hash Join",
        "Plan Rows": 5,
        "Actual Rows": 5000,
        "Plans": [_scan(10, 10)],
    }
    rows = list(scan_nodes_with_actuals(plan))
    assert len(rows) == 1  # only the scan, not the join


def test_training_set_matches_declared_feature_order():
    X, y = build_training_set([_scan(10, 100), _scan(20, 20)])
    assert len(X) == len(y) == 2
    assert all(len(row) == len(QERROR_FEATURES) for row in X)


def test_fit_refuses_insufficient_data():
    corrector = CardinalityCorrector(build_model=lambda: None)
    with pytest.raises(ValueError, match="too few"):
        corrector.fit([_scan(10, 100)])


def test_corrected_rows_scales_the_estimate():
    class _AlwaysDoubles:
        def predict(self, X):
            return [math.log(2.0)] * len(X)

    corrector = CardinalityCorrector()
    corrector.model = _AlwaysDoubles()
    corrected = corrector.corrected_rows(_scan(1000, 1000))
    assert corrected["o"] == pytest.approx(2000.0, rel=1e-6)


def test_untrained_corrector_returns_nothing():
    assert CardinalityCorrector().predict_log_qerror(_scan(10, 10)) == {}
