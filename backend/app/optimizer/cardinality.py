"""
Learned cardinality-error correction.

## Why this is the root cause, not a side quest

Leis et al. ("How Good Are Query Optimizers, Really?", VLDB 2015 -- the
paper that introduced the JOB benchmark this project loads) showed that
PostgreSQL's plan choices go wrong mainly because its *cardinality
estimates* go wrong, and that the errors compound multiplicatively as joins
stack: a 10x underestimate at one join becomes a 100x underestimate two
joins later. Cost models are roughly fine; the row counts fed into them are
not.

`docs/JOB_RESULTS.md` shows the consequence directly. On held-out JOB
queries native PostgreSQL averages 3703 ms while the best available plan
averages 919 ms. The planner is not choosing badly because it reasons badly
-- it is choosing badly because it believes wrong numbers.

## What this does

Every `EXPLAIN ANALYZE` in `plan_execution_log` contains both what Postgres
*predicted* (`Plan Rows`) and what actually happened (`Actual Rows`) at
every node. That is free, abundant, perfectly-labelled supervision for the
exact quantity the optimizer gets wrong -- and this project has been
throwing it away.

This module learns to predict the **q-error** (the log-ratio of actual to
estimated rows) for a scan node from features available *before* execution.
Applied at inference, it gives a corrected row estimate:

    corrected_rows = estimated_rows * exp(predicted_log_ratio)

## Two things are learned here

**Scan-level q-error** (`CardinalityCorrector`) predicts how wrong a base
relation scan's estimate is. It feeds `features.py` as an extra signal --
"this scan's estimate is probably a 5x underestimate" is exactly the kind of
thing a plan selector should know, and Postgres's own cost model does not
have it.

**Join-level q-error** (`JoinCardinalityCorrector`) is the more interesting
one, because it can be *fed back into the planner*. Every other candidate in
this system works around a bad estimate by forcing a plan shape:
`Leading(...)` dictates a join order, `Set(enable_hashjoin off)` bans an
operator. All of them override a planner that is reasoning correctly from
wrong numbers.

pg_hint_plan's `Rows` hint takes the opposite route. `Rows(a b *10)` tells
Postgres that the join of `a` and `b` yields ten times what it thinks, and
then lets its own decades-tuned planner choose freely with a better number in
hand. So instead of arguing with the conclusion, this corrects the premise.

That distinction matters for the shape of the result. A forced join order is
one plan; a corrected estimate can produce a plan no hint in the action space
would have generated, including plan shapes the candidate generator cannot
express at all.

## Why join level is where the leverage is

Postgres estimates a join as `left_rows * right_rows * selectivity`, treating
the two sides as independent. That assumption is what breaks under correlated
predicates, and the error compounds multiplicatively as joins stack -- so the
error at a join is not merely inherited from its scans, it is *manufactured*
at the join itself. `log_implied_selectivity` below is that assumption made
into a feature: it is exactly the quantity Postgres guesses and this model
learns to correct.
"""

from __future__ import annotations

import math

QERROR_FEATURES = [
    "log_estimated_rows",
    "is_seq_scan",
    "is_index_scan",
    "has_filter",
    "n_filter_conjuncts",
    "plan_width",
    "depth_in_tree",
]


def _count_conjuncts(filter_text: str) -> float:
    """Rough predicate count. More conjuncts -> more independence assumptions
    stacked -> historically, worse estimates."""
    if not filter_text:
        return 0.0
    return float(filter_text.count(" AND ") + 1)


def scan_nodes_with_actuals(plan: dict, depth: int = 0):
    """
    Yield (features, log_qerror) for every scan node that has actuals.

    Only base-relation scans: join-node cardinality error is a *consequence*
    of scan-level error compounding, so learning it separately would mostly
    re-learn the same signal with more noise.
    """
    if "Relation Name" in plan and "Actual Rows" in plan:
        estimated = float(plan.get("Plan Rows") or 0)

        # Per-execution on both sides. PostgreSQL reports `Actual Rows` as an
        # average over `Actual Loops`, and `Plan Rows` is the estimate for a
        # single execution, so the two compare directly. Multiplying the
        # actual side by the loop count (which this did) manufactured a
        # log-qerror of log(loops) on every correctly-estimated scan sitting
        # on the inner side of a nested loop -- the exact nodes whose
        # cardinality matters most, taught with the exact wrong label.
        actual = float(plan.get("Actual Rows") or 0)

        # +1 keeps zero-row scans (very common under selective filters) from
        # producing an infinite ratio.
        log_qerror = math.log((actual + 1.0) / (estimated + 1.0))

        node_type = plan.get("Node Type", "")
        filter_text = plan.get("Filter", "") or ""
        features = {
            "log_estimated_rows": math.log1p(estimated),
            "is_seq_scan": 1.0 if node_type == "Seq Scan" else 0.0,
            "is_index_scan": 1.0 if "Index" in node_type else 0.0,
            "has_filter": 1.0 if filter_text else 0.0,
            "n_filter_conjuncts": _count_conjuncts(filter_text),
            "plan_width": float(plan.get("Plan Width") or 0),
            "depth_in_tree": float(depth),
        }
        yield features, log_qerror

    for child in plan.get("Plans", []):
        yield from scan_nodes_with_actuals(child, depth + 1)


def build_training_set(raw_plans: list[dict]) -> tuple[list[list[float]], list[float]]:
    X, y = [], []
    for plan in raw_plans:
        for features, log_qerror in scan_nodes_with_actuals(plan):
            X.append([features[c] for c in QERROR_FEATURES])
            y.append(log_qerror)
    return X, y


class CardinalityCorrector:
    """Predicts how wrong Postgres's row estimate is, before running anything."""

    def __init__(self, build_model=None):
        self.build_model = build_model
        self.model = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["build_model"] = None
        return state

    def fit(self, raw_plans: list[dict]) -> "CardinalityCorrector":
        X, y = build_training_set(raw_plans)
        if len(X) < 20:
            raise ValueError(f"only {len(X)} scan observations; too few to learn q-error")
        self.model = self.build_model()
        self.model.fit(X, y)
        return self

    def predict_log_qerror(self, plan: dict) -> dict[str, float]:
        """alias -> predicted log(actual/estimated) for each scan in the plan."""
        if self.model is None:
            return {}

        aliases, rows = [], []
        for node, features in _scan_features(plan):
            aliases.append(node.get("Alias", node.get("Relation Name")))
            rows.append([features[c] for c in QERROR_FEATURES])

        if not rows:
            return {}
        return dict(zip(aliases, (float(v) for v in self.model.predict(rows))))

    def corrected_rows(self, plan: dict) -> dict[str, float]:
        """alias -> the row count this model thinks the scan will really return."""
        corrections = self.predict_log_qerror(plan)
        corrected = {}
        for node, _ in _scan_features(plan):
            alias = node.get("Alias", node.get("Relation Name"))
            estimated = float(node.get("Plan Rows") or 0)
            if alias in corrections:
                corrected[alias] = estimated * math.exp(corrections[alias])
        return corrected


# -- join-level correction, fed back into the planner -----------------------

JOIN_NODE_TYPES = ("Nested Loop", "Hash Join", "Merge Join")

JOIN_QERROR_FEATURES = [
    "log_estimated_rows",
    "log_left_rows",
    "log_right_rows",
    # Postgres's independence assumption, made measurable: the selectivity it
    # implicitly applied to the cross product. When predicates on the two
    # sides are correlated this is the number that is wrong.
    "log_implied_selectivity",
    "n_relations",
    "depth_in_tree",
    "n_filtered_children",
    "is_nested_loop",
    "is_hash_join",
    "is_merge_join",
]

# Corrections below this are noise, and a hint that says "you were right" only
# adds risk. 1.5x in either direction is the band where re-planning is unlikely
# to change anything anyway.
MIN_CORRECTION_FACTOR = 1.5

# However confident the model is, a single hint should not be able to move an
# estimate by more than this. An unbounded multiplier on a badly extrapolated
# prediction is how a correction turns into a catastrophe.
MAX_CORRECTION_FACTOR = 1000.0


def relations_under(node: dict) -> tuple[str, ...]:
    """
    Base relation aliases beneath a plan node, sorted.

    This is the key a `Rows` hint is addressed by: `Rows(a b *10)` matches the
    join that brings together exactly the relations `a` and `b`, whatever
    shape the planner gives it.
    """
    found: list[str] = []
    if "Relation Name" in node:
        found.append(node.get("Alias", node["Relation Name"]))
    for child in node.get("Plans", []):
        found.extend(relations_under(child))
    return tuple(sorted(found))


def _join_feature_row(node: dict, depth: int) -> dict[str, float] | None:
    """Estimate-only features for one join node, or None if unusable."""
    children = node.get("Plans", [])
    if len(children) < 2:
        return None

    estimated = float(node.get("Plan Rows") or 0)
    left = float(children[0].get("Plan Rows") or 0)
    right = float(children[1].get("Plan Rows") or 0)
    cross = max(left, 1.0) * max(right, 1.0)

    node_type = node.get("Node Type", "")
    return {
        "log_estimated_rows": math.log1p(estimated),
        "log_left_rows": math.log1p(left),
        "log_right_rows": math.log1p(right),
        "log_implied_selectivity": math.log((estimated + 1.0) / cross),
        "n_relations": float(len(relations_under(node))),
        "depth_in_tree": float(depth),
        "n_filtered_children": float(
            sum(1 for c in children if c.get("Filter") or c.get("Recheck Cond"))
        ),
        "is_nested_loop": 1.0 if node_type == "Nested Loop" else 0.0,
        "is_hash_join": 1.0 if node_type == "Hash Join" else 0.0,
        "is_merge_join": 1.0 if node_type == "Merge Join" else 0.0,
    }


def join_nodes(plan: dict, depth: int = 0):
    """Yield (node, features, relations) for every join, estimate-only."""
    if plan.get("Node Type") in JOIN_NODE_TYPES:
        features = _join_feature_row(plan, depth)
        if features is not None:
            yield plan, features, relations_under(plan)
    for child in plan.get("Plans", []):
        yield from join_nodes(child, depth + 1)


def join_nodes_with_actuals(plan: dict, depth: int = 0):
    """
    Yield (features, log_qerror, relations) for joins that have actuals.

    Per-execution on both sides, for the reason spelled out in
    `scan_nodes_with_actuals`.
    """
    for node, features, relations in join_nodes(plan, depth):
        if "Actual Rows" not in node:
            continue
        estimated = float(node.get("Plan Rows") or 0)
        actual = float(node.get("Actual Rows") or 0)
        yield features, math.log((actual + 1.0) / (estimated + 1.0)), relations


def build_join_training_set(raw_plans: list[dict]) -> tuple[list[list[float]], list[float]]:
    X, y = [], []
    for plan in raw_plans:
        for features, log_qerror, _ in join_nodes_with_actuals(plan):
            X.append([features[c] for c in JOIN_QERROR_FEATURES])
            y.append(log_qerror)
    return X, y


def format_rows_hint(relations: tuple[str, ...], factor: float) -> str:
    """
    One pg_hint_plan `Rows` correction.

    The multiplier form is used rather than an absolute `#n` because the model
    predicts a *ratio*: it has learned how wrong Postgres tends to be, not what
    the answer is. Multiplying keeps whatever the planner already knows about
    the shape of the data and adjusts it.
    """
    return f"Rows({' '.join(relations)} *{factor:.4g})"


class JoinCardinalityCorrector:
    """
    Learns how wrong Postgres's join estimates are, and says so in a language
    the planner understands.
    """

    def __init__(self, build_model=None):
        self.build_model = build_model
        self.model = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["build_model"] = None
        return state

    def fit(self, raw_plans: list[dict]) -> "JoinCardinalityCorrector":
        X, y = build_join_training_set(raw_plans)
        if len(X) < 20:
            raise ValueError(f"only {len(X)} join observations; too few to learn q-error")
        self.model = self.build_model()
        self.model.fit(X, y)
        return self

    def predict_factors(self, plan: dict) -> dict[tuple[str, ...], float]:
        """relations -> the multiplier this model would apply to their join."""
        if self.model is None:
            return {}

        rows, keys = [], []
        for _, features, relations in join_nodes(plan):
            # A single-relation "join" cannot be addressed by a Rows hint, and
            # duplicate relation sets (the same join reached twice) would emit
            # contradictory hints for one node.
            if len(relations) < 2 or relations in keys:
                continue
            keys.append(relations)
            rows.append([features[c] for c in JOIN_QERROR_FEATURES])

        if not rows:
            return {}

        factors = {}
        for relations, log_ratio in zip(keys, self.model.predict(rows), strict=True):
            factor = math.exp(float(log_ratio))
            factors[relations] = min(max(factor, 1.0 / MAX_CORRECTION_FACTOR), MAX_CORRECTION_FACTOR)
        return factors

    def rows_hints(
        self, plan: dict, min_factor: float = MIN_CORRECTION_FACTOR
    ) -> list[str]:
        """
        `Rows(...)` corrections worth sending to the planner.

        Only corrections outside [1/min_factor, min_factor] are emitted. A hint
        that barely moves an estimate cannot change the plan, so issuing it
        trades no upside for the risk of being wrong.
        """
        hints = []
        for relations, factor in sorted(self.predict_factors(plan).items()):
            if factor >= min_factor or factor <= 1.0 / min_factor:
                hints.append(format_rows_hint(relations, factor))
        return hints


def _scan_features(plan: dict, depth: int = 0):
    """Like `scan_nodes_with_actuals` but estimate-only, so it works at
    inference time on a plan that has never been executed."""
    if "Relation Name" in plan:
        node_type = plan.get("Node Type", "")
        filter_text = plan.get("Filter", "") or ""
        estimated = float(plan.get("Plan Rows") or 0)
        yield plan, {
            "log_estimated_rows": math.log1p(estimated),
            "is_seq_scan": 1.0 if node_type == "Seq Scan" else 0.0,
            "is_index_scan": 1.0 if "Index" in node_type else 0.0,
            "has_filter": 1.0 if filter_text else 0.0,
            "n_filter_conjuncts": _count_conjuncts(filter_text),
            "plan_width": float(plan.get("Plan Width") or 0),
            "depth_in_tree": float(depth),
        }
    for child in plan.get("Plans", []):
        yield from _scan_features(child, depth + 1)
