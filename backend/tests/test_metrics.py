"""
The `/metrics` endpoint.

Everything else this API exposes answers a question when asked. This is the
half that lets the system be *watched*: a regression rate climbing past what
the gate promised, or a deviation rate that has fallen to zero and turned the
optimizer into an expensive no-op, are both things to find out from a graph
rather than by opening the dashboard and reading it.
"""

from app.metrics import render

TREND = {
    "overall": {"native_total_ms": 1000.0, "served_total_ms": 800.0},
    "decision_quality": {
        "deviated_win": 6, "deviated_wash": 2, "deviated_loss": 2,
        "held_correct": 8, "held_missed": 2,
        "saved_ms": 250.0, "regression_ms": 40.0, "missed_ms": 90.0,
    },
}
REPORT = {
    "gate": {"consults_uncertainty": True, "calibrated_on_n_queries": 36},
    "warnings": [
        {"severity": "high", "code": "a"},
        {"severity": "medium", "code": "b"},
    ],
}
STATUS = {
    "current_version": "20260901T124454Z",
    "rows_since_last_training": 412,
    "regression_guard": {"blocked_queries": {"q1": {}, "q2": {}}},
    "decision_cache": {"hit_rate": 0.75},
}


def _metric(text, name):
    for line in text.splitlines():
        if line.startswith(f"lqo_{name} ") or line.startswith(f"lqo_{name}{{"):
            return line.split(" ")[-1]
    return None


# -- the numbers that would page someone ------------------------------------


def test_the_deviation_rate_is_exported():
    """Zero means the optimizer is doing nothing, and no other number says so
    plainly."""
    assert _metric(render(TREND, REPORT, STATUS), "deviation_rate") == str(10 / 20)


def test_the_regression_rate_is_over_deviations_not_runs():
    """Declining to optimize is never a regression, and counting it as a
    success would make an inert gate look perfect."""
    assert _metric(render(TREND, REPORT, STATUS), "regression_rate") == str(2 / 10)


def test_saved_and_regressed_are_reported_separately():
    """A net figure hides the trade; these are the two halves of it."""
    text = render(TREND, REPORT, STATUS)

    assert _metric(text, "saved_ms_total") == "250.0"
    assert _metric(text, "regressed_ms_total") == "40.0"


def test_a_gate_that_ignores_uncertainty_is_visible():
    text = render(TREND, {**REPORT, "gate": {"consults_uncertainty": False}}, STATUS)

    assert _metric(text, "gate_consults_uncertainty") == "0"


def test_high_severity_findings_are_counted_separately():
    text = render(TREND, REPORT, STATUS)

    assert _metric(text, "findings_total") == "2"
    assert _metric(text, "findings_high_total") == "1"


def test_the_deployed_version_is_a_label_not_a_value():
    """The usual way to annotate a graph with what was running."""
    assert 'lqo_model_info{version="20260901T124454Z"} 1' in render(TREND, REPORT, STATUS)


# -- format and robustness --------------------------------------------------


def test_every_metric_declares_help_and_type():
    text = render(TREND, REPORT, STATUS)
    names = {
        line.split()[0].split("{")[0]
        for line in text.splitlines()
        if line and not line.startswith("#")
    }

    for name in names:
        assert f"# HELP {name} " in text
        assert f"# TYPE {name} " in text


def test_missing_inputs_produce_no_series_rather_than_zeros():
    """A metric that reports 0 when it means 'unknown' is worse than absent --
    it graphs as a real value and alerts on it."""
    text = render(None, None, None)

    assert _metric(text, "deviation_rate") is None
    assert _metric(text, "decision_cache_hit_rate") is None


def test_no_runs_does_not_divide_by_zero():
    empty = {"overall": {}, "decision_quality": dict.fromkeys(
        ("deviated_win", "deviated_wash", "deviated_loss", "held_correct", "held_missed"), 0
    )}

    text = render(empty, REPORT, STATUS)

    assert _metric(text, "runs_total") == "0"
    assert _metric(text, "deviation_rate") is None


def test_a_disabled_cache_reports_no_hit_rate():
    text = render(TREND, REPORT, {**STATUS, "decision_cache": None})

    assert _metric(text, "decision_cache_hit_rate") is None


def test_there_are_no_per_query_series():
    """Cardinality is how monitoring falls over, and this system fingerprints
    every ad-hoc statement it is handed."""
    text = render(TREND, REPORT, STATUS)

    assert "query_id" not in text
    assert text.count("{") == text.count('{version="')


def test_the_output_ends_with_a_newline():
    """Prometheus' text format requires it."""
    assert render(TREND, REPORT, STATUS).endswith("\n")
