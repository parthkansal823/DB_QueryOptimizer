"""
Measuring generalisation on predictions rather than on decisions.

`app.benchmark`'s held-out column could not answer whether the model
generalises. On this workload the gate acts on **one held-out query in six** and
vetoes the rest, so the number is a ratio whose numerator comes from one or two
queries and whose denominator is the headroom of all of them. Two measurements
of the same model gave -4.8% and +3.1%; the spread was 19.8 points. See
`docs/WRITEUP.md` 2.32.

Prediction error has an answer whether or not the gate ever fires, which is the
point of this module.
"""

import math

from app.generalisation import _reading, _summary, prediction_errors, report


class FakeModel:
    """Predicts a fixed log-ratio for every candidate."""

    def __init__(self, predicted_log: float):
        self.predicted_log = predicted_log

    def predict_mean_std(self, vectors):
        n = len(vectors)
        return [self.predicted_log] * n, [0.1] * n


def bundle(predicted_log=0.0, held_out=()):
    return {
        "model": FakeModel(predicted_log),
        "feature_columns": [],
        "table_cardinalities": {},
        "held_out_queries": list(held_out),
    }


def rows(query_id, native_ms, candidate_ms_list):
    out = [
        {
            "query_id": query_id,
            "is_baseline": True,
            "actual_total_time_ms": native_ms,
            "raw_plan": {"Node Type": "Seq Scan"},
        }
    ]
    for ms in candidate_ms_list:
        out.append(
            {
                "query_id": query_id,
                "is_baseline": False,
                "actual_total_time_ms": ms,
                "hint": "/*+ Leading(a b) */",
                "raw_plan": {"Node Type": "Hash Join"},
            }
        )
    return out


class TestPredictionError:
    def test_a_perfect_prediction_has_zero_error(self):
        """Candidate ran at exactly native speed, and the model said 1.0x."""
        by_query = {"q": rows("q", 100.0, [100.0])}
        errors = prediction_errors(bundle(predicted_log=0.0), by_query, {"q"})
        assert errors == [0.0]

    def test_error_is_measured_in_log_space(self):
        """
        Predicting 2x when the truth is 1x and 0.5x when the truth is 1x are the
        same size of mistake. A linear error would call the first one twice as
        bad, which would quietly bias every summary toward optimistic models.
        """
        by_query = {"q": rows("q", 100.0, [100.0])}
        too_high = prediction_errors(bundle(predicted_log=math.log(2)), by_query, {"q"})
        too_low = prediction_errors(bundle(predicted_log=math.log(0.5)), by_query, {"q"})
        assert too_high[0] == too_low[0]

    def test_the_baseline_is_the_median_of_its_rows(self):
        """A single native draw carries the full 18.6% run-to-run noise."""
        by_query = {
            "q": [
                {"query_id": "q", "is_baseline": True, "actual_total_time_ms": 50.0,
                 "raw_plan": {}},
                {"query_id": "q", "is_baseline": True, "actual_total_time_ms": 100.0,
                 "raw_plan": {}},
                {"query_id": "q", "is_baseline": True, "actual_total_time_ms": 150.0,
                 "raw_plan": {}},
                {"query_id": "q", "is_baseline": False, "actual_total_time_ms": 100.0,
                 "hint": "h", "raw_plan": {}},
            ]
        }
        errors = prediction_errors(bundle(predicted_log=0.0), by_query, {"q"})
        assert errors == [0.0]  # median native is 100, candidate 100 -> ratio 1.0

    def test_a_query_with_no_baseline_is_skipped_not_guessed(self):
        by_query = {"q": [{"query_id": "q", "is_baseline": False,
                           "actual_total_time_ms": 10.0, "hint": "h", "raw_plan": {}}]}
        assert prediction_errors(bundle(), by_query, {"q"}) == []

    def test_only_the_requested_queries_are_scored(self):
        by_query = {"a": rows("a", 100.0, [100.0]), "b": rows("b", 100.0, [400.0])}
        assert len(prediction_errors(bundle(), by_query, {"a"})) == 1


class TestSummary:
    def test_the_factor_is_the_readable_form_of_the_log_error(self):
        """
        Nobody can reason about 0.18 in log space. 1.20x is the same number in a
        form that says "typically 20% out".
        """
        s = _summary([math.log(2)] * 5)
        assert s["typical_factor"] == 2.0
        assert s["n"] == 5

    def test_an_empty_set_reports_nothing_rather_than_zero(self):
        """
        Zero error would read as a perfect model. No data has to look like no
        data.
        """
        assert _summary([]) == {"n": 0}


class TestReport:
    def test_equal_error_on_both_sides_reads_as_generalising(self):
        by_query = {
            "trained": rows("trained", 100.0, [200.0]),
            "unseen": rows("unseen", 100.0, [200.0]),
        }
        result = report(bundle(predicted_log=0.0, held_out=["unseen"]), by_query)
        assert result["held_out_error_ratio"] == 1.0
        assert "generalises" in result["reading"]

    def test_the_split_is_taken_from_the_bundle(self):
        """
        The model records which queries it held out. Recomputing that split here
        would be a second copy of a decision owned elsewhere -- the mistake
        WRITEUP 2.28 catalogues five instances of.
        """
        by_query = {"a": rows("a", 100.0, [100.0]), "b": rows("b", 100.0, [100.0])}
        result = report(bundle(held_out=["b"]), by_query)
        assert result["n_held_out_queries"] == 1
        assert result["n_trained_queries"] == 1

    def test_a_missing_side_is_reported_rather_than_ratioed(self):
        by_query = {"a": rows("a", 100.0, [100.0])}
        result = report(bundle(held_out=["nonexistent"]), by_query)
        assert result["held_out_error_ratio"] is None
        assert "not enough data" in result["reading"]


class TestReadings:
    def test_a_much_worse_held_out_error_reads_as_memorisation(self):
        assert "memorised" in _reading(3.0)

    def test_a_close_ratio_reads_as_generalising(self):
        assert "generalises" in _reading(1.05)

    def test_the_middle_is_not_forced_into_either_verdict(self):
        """
        A model 1.5x worse on unseen queries has not memorised and has not
        cleanly generalised. Rounding that to either would be the aggregate
        hiding the interesting case again.
        """
        reading = _reading(1.5)
        assert "partially" in reading
