"""
Regression tests for limitations that were closed rather than documented.

Each of these describes something the system used to get wrong, so the test
name is the behaviour and the docstring is why it mattered.
"""

from app.optimizer.features import build_feature_columns, featurize
from app.optimizer.hints import generate_join_order_candidates
from app.plan_extractor import extract_join_graph

# -- 1. Candidate generation no longer wastes the budget on cross joins ------

# A chain: a - b - c - d. Only orders that walk the chain avoid a cartesian
# product; the other 16 of 24 permutations force one.
CHAIN = {"a": ["b"], "b": ["a", "c"], "c": ["b", "d"], "d": ["c"]}


def _orders(hints):
    return [h.removeprefix("/*+ Leading(").removesuffix(") */").split() for h in hints]


def test_every_generated_order_is_connected():
    """
    A `Leading()` order introducing a table that joins to nothing placed so far
    forces a cross product, which Postgres prices at disable_cost (~1e10) and
    never picks. Those candidates are budget spent on plans that cannot win.
    """
    hints = generate_join_order_candidates(list(CHAIN), 24, join_graph=CHAIN)
    for order in _orders(hints):
        placed = {order[0]}
        for table in order[1:]:
            assert set(CHAIN[table]) & placed, f"{order} breaks the chain at {table}"
            placed.add(table)


def test_the_graph_filters_out_most_permutations():
    """8 of a 4-table chain's 24 orderings are connected; the rest are waste."""
    blind = generate_join_order_candidates(list(CHAIN), 99)
    graphed = generate_join_order_candidates(list(CHAIN), 99, join_graph=CHAIN)
    assert len(blind) == 24
    assert len(graphed) == 8


def test_a_star_schema_places_the_hub_before_a_second_spoke():
    """
    Spokes join only the hub, never each other, so the hub has to be in the
    first two positions -- otherwise the second spoke has nothing to join to.
    """
    star = {"hub": ["s1", "s2", "s3"], "s1": ["hub"], "s2": ["hub"], "s3": ["hub"]}
    orders = _orders(generate_join_order_candidates(list(star), 99, join_graph=star))
    assert orders
    for order in orders:
        assert order.index("hub") <= 1, order


def test_large_queries_still_get_connected_orders():
    """Above the enumeration cutoff the sampler must still respect the graph."""
    chain = {str(i): [str(i - 1), str(i + 1)] for i in range(12)}
    tables = [str(i) for i in range(12)]
    hints = generate_join_order_candidates(tables, 8, join_graph=chain)
    assert hints
    for order in _orders(hints):
        placed = {order[0]}
        for t in order[1:]:
            assert set(chain.get(t, ())) & placed
            placed.add(t)


def test_a_genuine_cross_join_still_gets_candidates():
    """No join predicates at all must not mean no action space."""
    assert generate_join_order_candidates(["a", "b", "c"], 6, join_graph={}) != []


def test_no_graph_preserves_the_old_behaviour():
    assert len(generate_join_order_candidates(["a", "b", "c"], 99)) == 6


# -- 2. The join graph is read off the plan ---------------------------------


def test_join_graph_is_read_from_plan_conditions():
    plan = {
        "Node Type": "Hash Join", "Hash Cond": "(oi.order_id = o.id)",
        "Plans": [
            {"Node Type": "Seq Scan", "Relation Name": "order_items", "Alias": "oi"},
            {"Node Type": "Hash Join", "Hash Cond": "(o.user_id = u.id)", "Plans": [
                {"Node Type": "Seq Scan", "Relation Name": "orders", "Alias": "o"},
                {"Node Type": "Seq Scan", "Relation Name": "users", "Alias": "u"},
            ]},
        ],
    }
    graph = extract_join_graph(plan)
    assert graph == {"o": ["oi", "u"], "oi": ["o"], "u": ["o"]}


def test_index_cond_contributes_the_scanning_alias():
    """`Index Cond: (id = oi.product_id)` names only the *other* side; the
    indexed relation is the node itself and has to be added."""
    plan = {
        "Node Type": "Nested Loop",
        "Plans": [
            {"Node Type": "Seq Scan", "Relation Name": "order_items", "Alias": "oi"},
            {"Node Type": "Index Scan", "Relation Name": "products", "Alias": "p",
             "Index Cond": "(id = oi.product_id)"},
        ],
    }
    assert extract_join_graph(plan) == {"oi": ["p"], "p": ["oi"]}


def test_a_query_with_no_joins_has_an_empty_graph():
    assert extract_join_graph({"Node Type": "Seq Scan", "Alias": "t"}) == {}


# -- 3. Self-joins no longer collapse to one slot ---------------------------


def _plan(*scans):
    return {
        "raw_plan": {"Node Type": "Hash Join", "Plans": list(scans)},
        "tables_scanned": [s["Alias"] for s in scans],
        "scan_relations": {s["Alias"]: s["Relation Name"] for s in scans},
        "join_types": ["Hash Join (Inner)"],
        "total_cost": 100.0,
    }


def _scan(alias, relation, rows, node_type="Seq Scan"):
    return {"Node Type": node_type, "Relation Name": relation, "Alias": alias,
            "Plan Rows": rows, "Plan Width": 8}


CARDS = {"movie_info": 1000.0, "title": 500.0}


def test_a_self_join_is_distinguishable_from_a_single_scan():
    """
    Two aliases of one table used to overwrite each other, so `movie_info AS
    mi1, movie_info AS mi2` looked identical to a single scan of movie_info.
    """
    once = featurize(_plan(_scan("mi1", "movie_info", 100),
                           _scan("t", "title", 50)), CARDS)
    twice = featurize(_plan(_scan("mi1", "movie_info", 100),
                            _scan("mi2", "movie_info", 100),
                            _scan("t", "title", 50)), CARDS)
    assert once["movie_info_occurrences"] == 1.0
    assert twice["movie_info_occurrences"] == 2.0


def test_the_most_selective_occurrence_wins():
    """Overwriting kept whichever alias came last in the plan; the selective
    scan is the one that actually shapes the plan."""
    features = featurize(_plan(_scan("mi1", "movie_info", 900),
                               _scan("mi2", "movie_info", 10)), CARDS)
    assert features["movie_info_selectivity"] == 0.01  # 10/1000, not 900/1000


def test_the_earliest_occurrence_sets_the_join_position():
    features = featurize(_plan(_scan("mi1", "movie_info", 100),
                               _scan("t", "title", 50),
                               _scan("mi2", "movie_info", 100)), CARDS)
    assert features["movie_info_join_position"] == 1 / 3


def test_an_index_scan_on_any_occurrence_counts():
    features = featurize(_plan(_scan("mi1", "movie_info", 100),
                               _scan("mi2", "movie_info", 5, "Index Scan")), CARDS)
    assert features["movie_info_index_scan"] == 1.0


def test_absent_tables_report_zero_occurrences():
    features = featurize(_plan(_scan("t", "title", 50)), CARDS)
    assert features["movie_info_occurrences"] == 0.0
    assert features["movie_info_present"] == 0.0


def test_occurrences_is_part_of_the_column_set():
    assert "movie_info_occurrences" in build_feature_columns(list(CARDS))


# -- 4. An unseen query gets a stricter bar ---------------------------------


class _Ensemble:
    """Predicts a fixed log-ratio with a fixed spread, for every candidate."""

    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def predict_mean_std(self, vectors):
        return [self.mean] * len(vectors), [self.std] * len(vectors)

    def predict(self, vectors):
        return [self.mean] * len(vectors)


def _optimizer(mean, std):
    from app.optimizer.learned import LearnedOptimizer

    # Explicit thresholds: a calibrated `models/gate.json` would otherwise
    # change what counts as "marginal" and make this test depend on whether
    # someone has run `app.calibrate`.
    opt = LearnedOptimizer(
        model_path="/nonexistent", confidence_z=1.0, min_relative_gain=0.05
    )
    opt.model = _Ensemble(mean, std)
    opt.feature_columns = []
    opt.table_cardinalities = {}
    opt.target = "log_ratio_vs_native"
    opt.policy = "greedy"
    return opt


def _candidate(hint="/*+ Leading(a b) */"):
    return {"raw_plan": {"Node Type": "Hash Join", "Plan Rows": 1, "Total Cost": 10.0},
            "tables_scanned": ["a", "b"], "scan_relations": {}, "join_types": [],
            "total_cost": 10.0, "hint": hint}


def _baseline():
    return {"raw_plan": {"Node Type": "Seq Scan", "Plan Rows": 1, "Total Cost": 12.0},
            "tables_scanned": ["a", "b"], "scan_relations": {}, "join_types": [],
            "total_cost": 12.0}


def test_a_marginal_win_is_taken_on_a_known_query_and_refused_on_an_unseen_one():
    """
    Same prediction, same uncertainty -- only the evidence differs. A query the
    guard has history for clears the bar; a brand-new one does not.
    """
    import math

    # ~18% predicted gain with modest spread: clears the normal bar, not a 2x one.
    opt = _optimizer(mean=math.log(0.82), std=0.05)

    opt.select_plan([_candidate()], baseline_plan=_baseline(), caution=1.0)
    assert opt.last_decision["fell_back_to_baseline"] is False

    opt.select_plan([_candidate()], baseline_plan=_baseline(), caution=2.0)
    assert opt.last_decision["fell_back_to_baseline"] is True


def test_a_clear_win_still_goes_through_on_an_unseen_query():
    """Caution must not make new queries un-optimisable."""
    import math

    opt = _optimizer(mean=math.log(0.2), std=0.02)
    opt.select_plan([_candidate()], baseline_plan=_baseline(), caution=2.0)
    assert opt.last_decision["fell_back_to_baseline"] is False


def test_an_explicitly_passed_default_is_not_overridden_by_calibration():
    """
    "Explicit arguments win" was implemented as `if arg == DEFAULT`, which
    cannot tell a deliberately-passed 1.0 from an untouched one -- so asking
    for the default silently got the calibrated value, and there was no way to
    opt out of models/gate.json short of deleting it.
    """
    from app.optimizer.learned import DEFAULT_CONFIDENCE_Z, LearnedOptimizer

    opt = LearnedOptimizer(model_path="/nonexistent", confidence_z=DEFAULT_CONFIDENCE_Z)
    assert opt.confidence_z == DEFAULT_CONFIDENCE_Z
