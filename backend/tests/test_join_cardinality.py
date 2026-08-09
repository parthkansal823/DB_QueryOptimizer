"""
Join-level cardinality correction and the `Rows` hints it produces.

The point of these is the *hint*: unlike every other candidate, a `Rows`
correction does not force a plan shape, it changes the number the planner
reasons from. So the tests care about which corrections get emitted and which
are correctly withheld.
"""

import math

import pytest

from app.optimizer.cardinality import (
    MAX_CORRECTION_FACTOR,
    JoinCardinalityCorrector,
    build_join_training_set,
    format_rows_hint,
    join_nodes,
    join_nodes_with_actuals,
    relations_under,
)
from app.optimizer.hints import corrected_cardinality_hint


def _scan(alias, rows, actual=None, filter_text=None):
    node = {
        "Node Type": "Seq Scan",
        "Relation Name": alias.rstrip("0123456789") or alias,
        "Alias": alias,
        "Plan Rows": rows,
        "Plan Width": 8,
    }
    if actual is not None:
        node["Actual Rows"] = actual
        node["Actual Loops"] = 1
    if filter_text:
        node["Filter"] = filter_text
    return node


def _join(children, rows, actual=None, node_type="Hash Join"):
    node = {"Node Type": node_type, "Plan Rows": rows, "Plans": children}
    if actual is not None:
        node["Actual Rows"] = actual
        node["Actual Loops"] = 1
    return node


# -- addressing a join ------------------------------------------------------


def test_relations_under_collects_aliases_sorted():
    plan = _join([_scan("o", 100), _join([_scan("u", 50), _scan("p", 10)], 20)], 40)
    assert relations_under(plan) == ("o", "p", "u")


def test_relations_under_prefers_alias_over_table_name():
    node = {"Node Type": "Seq Scan", "Relation Name": "movie_info", "Alias": "mi1",
            "Plan Rows": 1}
    assert relations_under(node) == ("mi1",)


def test_join_nodes_skips_scans():
    plan = _join([_scan("a", 10), _scan("b", 10)], 5)
    assert [relations for _, _, relations in join_nodes(plan)] == [("a", "b")]


# -- the independence assumption, as a feature ------------------------------


def test_implied_selectivity_captures_the_cross_product():
    """100 x 100 estimated down to 10 rows is a selectivity of ~1/1000."""
    plan = _join([_scan("a", 100), _scan("b", 100)], 10)
    (_, features, _), = join_nodes(plan)
    assert features["log_implied_selectivity"] == pytest.approx(math.log(11 / 10000), rel=1e-6)


def test_filtered_children_are_counted():
    plan = _join([_scan("a", 100, filter_text="(x = 1)"), _scan("b", 100)], 10)
    (_, features, _), = join_nodes(plan)
    assert features["n_filtered_children"] == 1.0


def test_join_method_is_encoded():
    nl = _join([_scan("a", 10), _scan("b", 10)], 5, node_type="Nested Loop")
    (_, features, _), = join_nodes(nl)
    assert (features["is_nested_loop"], features["is_hash_join"]) == (1.0, 0.0)


# -- labels -----------------------------------------------------------------


def test_underestimated_join_gets_a_positive_label():
    plan = _join([_scan("a", 100, 100), _scan("b", 100, 100)], 10, actual=1000)
    (_, log_qerror, relations), = join_nodes_with_actuals(plan)
    assert log_qerror > 0
    assert relations == ("a", "b")


def test_a_perfect_join_estimate_has_no_error():
    plan = _join([_scan("a", 10, 10), _scan("b", 10, 10)], 50, actual=50)
    (_, log_qerror, _), = join_nodes_with_actuals(plan)
    assert log_qerror == 0.0


def test_unexecuted_plans_contribute_no_labels():
    plan = _join([_scan("a", 10), _scan("b", 10)], 5)
    assert list(join_nodes_with_actuals(plan)) == []
    assert build_join_training_set([plan]) == ([], [])


# -- hint formatting --------------------------------------------------------


def test_rows_hint_is_a_multiplier_over_the_relation_set():
    assert format_rows_hint(("a", "b"), 10.0) == "Rows(a b *10)"


def test_corrected_cardinality_hint_wraps_all_corrections():
    hint = corrected_cardinality_hint(["Rows(a b *10)", "Rows(a b c *0.2)"])
    assert hint == "/*+ Rows(a b *10) Rows(a b c *0.2) */"


def test_no_corrections_means_no_hint():
    """No hint at all, rather than an empty comment that does nothing."""
    assert corrected_cardinality_hint([]) is None


# -- the corrector ----------------------------------------------------------


class _FixedModel:
    """Predicts one constant log-ratio, so hint selection can be tested."""

    def __init__(self, log_ratio):
        self.log_ratio = log_ratio

    def fit(self, X, y):
        return self

    def predict(self, rows):
        return [self.log_ratio] * len(rows)


def _corrector(log_ratio):
    c = JoinCardinalityCorrector(build_model=lambda: _FixedModel(log_ratio))
    c.model = _FixedModel(log_ratio)
    return c


def test_a_large_underestimate_produces_a_multiplying_hint():
    plan = _join([_scan("a", 100), _scan("b", 100)], 10)
    hints = _corrector(math.log(10)).rows_hints(plan)
    assert hints == ["Rows(a b *10)"]


def test_a_large_overestimate_produces_a_shrinking_hint():
    plan = _join([_scan("a", 100), _scan("b", 100)], 10)
    (hint,) = _corrector(math.log(0.1)).rows_hints(plan)
    assert hint.startswith("Rows(a b *0.1")


def test_a_correction_inside_the_noise_band_is_withheld():
    """A hint that barely moves an estimate cannot change the plan, so
    emitting it is pure downside."""
    plan = _join([_scan("a", 100), _scan("b", 100)], 10)
    assert _corrector(math.log(1.1)).rows_hints(plan) == []


def test_corrections_are_capped():
    """An extrapolating model must not be able to move an estimate arbitrarily."""
    plan = _join([_scan("a", 100), _scan("b", 100)], 10)
    factors = _corrector(math.log(10**9)).predict_factors(plan)
    assert factors[("a", "b")] == MAX_CORRECTION_FACTOR


def test_an_untrained_corrector_stays_silent():
    plan = _join([_scan("a", 100), _scan("b", 100)], 10)
    assert JoinCardinalityCorrector().rows_hints(plan) == []


def test_every_join_level_gets_its_own_correction():
    """A 3-way join has two join nodes, addressed by different relation sets."""
    plan = _join([_join([_scan("a", 100), _scan("b", 100)], 50), _scan("c", 10)], 5)
    hints = _corrector(math.log(4)).rows_hints(plan)
    assert hints == ["Rows(a b *4)", "Rows(a b c *4)"]


def test_fitting_needs_enough_observations():
    plan = _join([_scan("a", 10, 10), _scan("b", 10, 10)], 5, actual=50)
    with pytest.raises(ValueError, match="too few"):
        JoinCardinalityCorrector(build_model=lambda: _FixedModel(0.0)).fit([plan])


# -- the action space -------------------------------------------------------


class _Optimizer:
    def __init__(self, join_corrector=None):
        self.join_corrector = join_corrector


def _baseline(plan):
    return {"raw_plan": plan}


def test_without_a_corrector_the_action_space_is_unchanged():
    from app.optimizer.hints import generate_candidates
    from app.optimizer.planner import candidate_hints

    plan = _join([_scan("a", 100), _scan("b", 100)], 10)
    assert candidate_hints(_Optimizer(), _baseline(plan), ["a", "b"]) == generate_candidates(
        ["a", "b"]
    )


def test_a_trained_corrector_adds_exactly_one_candidate():
    from app.optimizer.hints import generate_candidates
    from app.optimizer.planner import candidate_hints

    plan = _join([_scan("a", 100), _scan("b", 100)], 10)
    hints = candidate_hints(_Optimizer(_corrector(math.log(10))), _baseline(plan), ["a", "b"])
    assert len(hints) == len(generate_candidates(["a", "b"])) + 1
    assert hints[-1] == "/*+ Rows(a b *10) */"


def test_a_confident_but_tiny_correction_adds_nothing():
    """The extra candidate costs a plan cycle; it has to be worth one."""
    from app.optimizer.hints import generate_candidates
    from app.optimizer.planner import candidate_hints

    plan = _join([_scan("a", 100), _scan("b", 100)], 10)
    hints = candidate_hints(_Optimizer(_corrector(math.log(1.05))), _baseline(plan), ["a", "b"])
    assert hints == generate_candidates(["a", "b"])
