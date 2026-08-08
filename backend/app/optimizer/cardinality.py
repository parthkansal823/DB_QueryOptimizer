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

## How it's used, and the honest limit

The correction feeds `features.py` as an extra signal -- "this scan's
estimate is probably a 5x underestimate" is exactly the kind of thing a plan
selector should know, and it is information Postgres's own cost model does
not have.

It is *not* fed back into Postgres's planner. Doing that properly would mean
intercepting cardinality estimation inside the planner itself (what
pg_hint_plan's `Rows` hint enables, and what a system like Bao deliberately
avoids). That is a substantially larger change and is named as future work
rather than attempted here.
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
        actual = float(plan.get("Actual Rows") or 0)
        loops = float(plan.get("Actual Loops") or 1)
        actual_total = actual * loops

        # +1 keeps zero-row scans (very common under selective filters) from
        # producing an infinite ratio.
        log_qerror = math.log((actual_total + 1.0) / (estimated + 1.0))

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
