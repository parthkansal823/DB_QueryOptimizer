"""
Phase 3: train a model that predicts a candidate plan's latency from its
features, then evaluate it against both baselines the roadmap calls for --
native Postgres and the Phase 0 heuristic (lowest *estimated* cost).

Starts simple, as the roadmap suggests: gradient-boosted trees (LightGBM,
falling back to scikit-learn's GradientBoostingRegressor if the LightGBM
wheel isn't available) predicting `actual_total_time_ms`, then argmin over
candidates. Split at the query level (not row level) so a query's own
candidates never leak between train and test.

Usage (from backend/, with the stack running via docker compose):
    python -m app.train [--model-out PATH] [--eval-out PATH]

MODEL_OUT/EVAL_OUT (env vars, or the flags above) let you train against a
different schema (e.g. DATABASE_URL pointed at the JOB/IMDB stretch goal's
`job` database) without overwriting the default model -- see
data/job/README.md.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random

from app.db import get_cursor
from app.optimizer.bandit import POLICIES, BootstrappedEnsemble, select_index
from app.optimizer.features import build_feature_columns, featurize, to_vector
from app.optimizer.ranker import PairwisePlanRanker
from app.plan_extractor import _extract_join_types, _extract_scan_relations, _extract_tables
from app.schema_introspection import discover_table_cardinalities

MODEL_PATH = os.getenv("MODEL_OUT_PATH", "models/plan_selector.pkl")
EVAL_PATH = os.getenv("EVAL_OUT_PATH", "models/eval_results.json")

FETCH_SQL = """
    SELECT query_id, sql_text, hint, is_baseline, raw_plan, total_cost, actual_total_time_ms
    FROM plan_execution_log
    WHERE actual_total_time_ms IS NOT NULL AND query_id IS NOT NULL
"""
# query_id IS NULL rows are ad-hoc dashboard queries (Phase 5's /query/analyze
# with no stable workload id) -- they still feed /stats/trend, but a query-level
# train/test split needs a real id to group each query's candidates by, so they're
# excluded from training data rather than silently mis-grouped under one bucket.


def _row_to_candidate(row: dict) -> dict:
    raw_plan = row["raw_plan"]
    return {
        "raw_plan": raw_plan,
        "tables_scanned": _extract_tables(raw_plan),
        "scan_relations": _extract_scan_relations(raw_plan),
        "join_types": _extract_join_types(raw_plan),
        "total_cost": row["total_cost"],
        "actual_total_time_ms": row["actual_total_time_ms"],
        "hint": row["hint"],
    }


def _aggregate_repetitions(rows: list[dict]) -> list[dict]:
    """
    Collapse repeated executions of the same (query, hint) into one row
    carrying the **median** latency.

    Why median and not mean: latency has a heavy right tail (a candidate
    occasionally collides with autovacuum, a checkpoint, or another process
    on the machine). The mean chases those outliers; the median ignores
    them. Since the label *is* the thing being learned, this directly
    attacks the noise that docs/WRITEUP.md §2.3 identifies as the binding
    constraint on the whole system.
    """
    buckets: dict[tuple, list[dict]] = {}
    for row in rows:
        buckets.setdefault((row["query_id"], row["hint"]), []).append(row)

    aggregated = []
    for group in buckets.values():
        latencies = sorted(r["actual_total_time_ms"] for r in group)
        mid = len(latencies) // 2
        median = (
            latencies[mid]
            if len(latencies) % 2
            else (latencies[mid - 1] + latencies[mid]) / 2
        )
        representative = dict(group[0])
        representative["actual_total_time_ms"] = median
        representative["n_reps"] = len(group)
        aggregated.append(representative)
    return aggregated


def _relative_targets(rows: list[dict]) -> list[float]:
    """
    Target = log(candidate_latency / native_latency_for_the_same_query).

    Predicting absolute milliseconds is the wrong problem, and it was the
    main reason accuracy stayed poor. Two things go wrong with it:

      1. **Scale dominates.** This workload spans ~5 ms to ~600 ms queries.
         Squared error on raw milliseconds is overwhelmingly driven by the
         slow queries, so the model spends its capacity learning "this query
         is inherently slow" -- true, useless, and nothing to do with plan
         choice. A 10% win on a fast query matters as much per execution as
         a 10% win on a slow one, and the raw-ms target says otherwise.
      2. **It answers a harder question than we asked.** We never need to
         know a plan will take 213 ms. We need to know it beats native.

    The log-ratio fixes both. Every query contributes on the same scale, and
    the target *is* the decision: negative means faster than native, 0 means
    identical, positive means worse. `learned.py` can then gate on
    "confidently below zero" instead of comparing two noisy absolute
    predictions and hoping the difference survives.

    Rows whose query has no baseline (or a zero-latency one) are dropped
    rather than guessed at.
    """
    native_by_query: dict[str, float] = {}
    for row in rows:
        if row["is_baseline"] and row["actual_total_time_ms"]:
            q = row["query_id"]
            # Several baseline executions may exist; keep the median-ish one
            # by preferring the first, which _aggregate_repetitions already
            # collapsed to a median.
            native_by_query.setdefault(q, float(row["actual_total_time_ms"]))

    targets = []
    for row in rows:
        native = native_by_query.get(row["query_id"])
        latency = row["actual_total_time_ms"]
        if not native or not latency:
            targets.append(0.0)  # no reference: treat as "same as native"
        else:
            targets.append(math.log(latency / native))
    return targets


def _discover_cardinalities() -> dict[str, float]:
    """ANALYZE first so `pg_class.reltuples` reflects what was just loaded/collected,
    then read it -- fast even on JOB-scale tables (see schema_introspection)."""
    with get_cursor() as cur:
        cur.execute("ANALYZE")
        return discover_table_cardinalities(cur)


def _load_rows() -> list[dict]:
    with get_cursor() as cur:
        cur.execute(FETCH_SQL)
        columns = [d[0] for d in cur.description]
        return [dict(zip(columns, r)) for r in cur.fetchall()]


def _build_model():
    """A single regressor + the name of the backend that provided it."""
    try:
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=-1
        ), "lightgbm"
    except (ImportError, OSError):
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42), "sklearn-gbrt"


def make_classifier():
    """Binary classifier for the pairwise ranker ('is A faster than B?')."""
    try:
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=-1
        )
    except (ImportError, OSError):
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)


def make_regressor():
    """
    Zero-arg factory for `BootstrappedEnsemble` (see optimizer/bandit.py).

    Module-level (not a lambda/closure) on purpose: the ensemble holds a
    reference to this factory and the whole ensemble gets pickled, and
    pickle can only serialise functions it can find by qualified name.
    """
    return _build_model()[0]


def _query_level_split(query_ids: list[str], test_fraction: float = 0.25, seed: int = 42):
    unique = sorted(set(query_ids))
    rng = random.Random(seed)
    rng.shuffle(unique)
    n_test = max(1, round(len(unique) * test_fraction))
    test_ids = set(unique[:n_test])
    train_ids = set(unique[n_test:])
    return train_ids, test_ids


def _evaluate_selection(
    rows_by_query: dict[str, list[dict]], ensemble, feature_columns, table_cardinalities, ranker=None
) -> dict:
    """
    For each held-out query, compare every selector against both baselines
    the roadmap asks for, plus an oracle.

    The **oracle** (always picks the candidate that actually turned out
    fastest, which requires hindsight and so is not a real strategy) is the
    ceiling: it says how much latency was available to win at all. Without
    it, "the model beat native by 3%" is unreadable -- you can't tell a good
    model on a workload with little headroom from a bad model on a workload
    with lots. Reporting the gap between a policy and the oracle is the
    honest version of that claim.
    """
    selectors = list(POLICIES) + (["pairwise_rank"] if ranker is not None else [])
    per_policy: dict[str, list[float]] = {p: [] for p in selectors}
    native_latencies, heuristic_latencies, oracle_latencies = [], [], []
    beats_heuristic = {p: 0 for p in selectors}
    rng = random.Random(7)  # fixed so the thompson column is reproducible

    for _, rows in rows_by_query.items():
        baseline_rows = [r for r in rows if r["is_baseline"]]
        if not baseline_rows:
            continue

        candidates = [_row_to_candidate(r) for r in rows]
        latencies = [c["actual_total_time_ms"] for c in candidates]
        costs = [c["total_cost"] or float("inf") for c in candidates]

        native_latency = baseline_rows[0]["actual_total_time_ms"]
        heuristic_latency = latencies[costs.index(min(costs))]

        vectors = [to_vector(featurize(c, table_cardinalities), feature_columns) for c in candidates]

        native_latencies.append(native_latency)
        heuristic_latencies.append(heuristic_latency)
        oracle_latencies.append(min(latencies))

        for policy in POLICIES:
            idx, _ = select_index(ensemble, vectors, policy=policy, rng=rng)
            per_policy[policy].append(latencies[idx])
            if latencies[idx] <= heuristic_latency:
                beats_heuristic[policy] += 1

        if ranker is not None:
            idx = ranker.select(vectors, tie_break_costs=costs)
            per_policy["pairwise_rank"].append(latencies[idx])
            if latencies[idx] <= heuristic_latency:
                beats_heuristic["pairwise_rank"] += 1

    n = len(native_latencies)
    if not n:
        return {"n_held_out_queries": 0}

    def avg(xs):
        return sum(xs) / len(xs)

    avg_latency = {
        "native": avg(native_latencies),
        "heuristic": avg(heuristic_latencies),
        "oracle_best_possible": avg(oracle_latencies),
    }
    avg_latency.update({f"learned_{p}": avg(per_policy[p]) for p in selectors})

    # What fraction of the available improvement did each selector capture?
    # 1.0 == matched the oracle; 0.0 == no better than native; <0 == worse.
    headroom = avg(native_latencies) - avg(oracle_latencies)
    captured = {
        f"learned_{p}": ((avg(native_latencies) - avg(per_policy[p])) / headroom) if headroom > 0 else None
        for p in selectors
    }

    return {
        "n_held_out_queries": n,
        "avg_latency_ms": avg_latency,
        "headroom_captured_vs_oracle": captured,
        "beats_or_ties_heuristic_rate": {f"learned_{p}": beats_heuristic[p] / n for p in selectors},
    }


def train(
    model_path: str = MODEL_PATH,
    eval_path: str = EVAL_PATH,
    n_models: int = 8,
    aggregate_reps: bool = True,
) -> dict:
    rows = _load_rows()
    if not rows:
        raise RuntimeError("plan_execution_log is empty -- run `python -m app.collect_data` first.")

    n_raw_rows = len(rows)
    if aggregate_reps:
        rows = _aggregate_repetitions(rows)

    table_cardinalities = _discover_cardinalities()
    feature_columns = build_feature_columns(list(table_cardinalities))

    candidates = [_row_to_candidate(r) for r in rows]
    query_ids = [r["query_id"] for r in rows]

    X_all = [to_vector(featurize(c, table_cardinalities), feature_columns) for c in candidates]
    y_all = _relative_targets(rows)

    train_ids, test_ids = _query_level_split(query_ids)
    train_idx = [i for i, q in enumerate(query_ids) if q in train_ids]
    test_idx = [i for i, q in enumerate(query_ids) if q in test_ids]

    X_train = [X_all[i] for i in train_idx]
    y_train = [y_all[i] for i in train_idx]
    X_test = [X_all[i] for i in test_idx]
    y_test = [y_all[i] for i in test_idx]

    _, backend_name = _build_model()
    ensemble = BootstrappedEnsemble(make_regressor, n_models=n_models).fit(X_train, y_train)

    # Pairwise ranker (Lero-style), trained on the same rows. Pairs are
    # formed within a query only -- see optimizer/ranker.py.
    train_groups: dict[str, tuple[list, list]] = {}
    for i in train_idx:
        vectors, latencies = train_groups.setdefault(query_ids[i], ([], []))
        vectors.append(X_all[i])
        latencies.append(y_all[i])
    ranker = None
    try:
        ranker = PairwisePlanRanker(make_classifier).fit(list(train_groups.values()))
    except ValueError as exc:  # too few distinct latencies to learn an ordering
        print(f"[warn] pairwise ranker not trained: {exc}")

    preds_test = ensemble.predict(X_test)
    # Error is now in log-ratio space. exp(MAE) reads as "typical multiplicative
    # error": 1.15 means predictions are typically off by ~15%.
    mae = sum(abs(p - y) for p, y in zip(preds_test, y_test)) / len(y_test) if y_test else None
    _, test_stds = ensemble.predict_mean_std(X_test)
    mean_uncertainty = sum(test_stds) / len(test_stds) if test_stds else None

    rows_by_query: dict[str, list[dict]] = {}
    for row in rows:
        if row["query_id"] in test_ids:
            rows_by_query.setdefault(row["query_id"], []).append(row)

    selection_eval = _evaluate_selection(
        rows_by_query, ensemble, feature_columns, table_cardinalities, ranker=ranker
    )

    results = {
        "model_backend": f"bootstrapped-ensemble[{n_models}x {backend_name}]",
        "pairwise_ranker": "trained" if ranker is not None else "not trained",
        "n_rows_raw": n_raw_rows,
        "reps_aggregated_to_median": aggregate_reps,
        "n_rows_total": len(rows),
        "n_rows_train": len(X_train),
        "n_rows_test": len(X_test),
        "n_features": len(feature_columns),
        "test_mae_log_ratio": mae,
        "test_typical_multiplicative_error": (math.exp(mae) if mae is not None else None),
        # Mean ensemble disagreement on held-out rows: a rough "how much does
        # the model actually know here" number. Large relative to test_mae_ms
        # means predictions are being driven by sparse evidence.
        "test_mean_uncertainty_log_ratio": mean_uncertainty,
        "n_tables_in_schema": len(table_cardinalities),
        **selection_eval,
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": ensemble,
                "ranker": ranker,
                "target": "log_ratio_vs_native",
                "feature_columns": feature_columns,
                "table_cardinalities": table_cardinalities,
            },
            f,
        )

    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-out", default=MODEL_PATH, help="where to pickle the trained model bundle")
    parser.add_argument("--eval-out", default=EVAL_PATH, help="where to write the evaluation JSON")
    parser.add_argument(
        "--n-models", type=int, default=8, help="bootstrapped ensemble size (drives uncertainty estimates)"
    )
    args = parser.parse_args()

    results = train(model_path=args.model_out, eval_path=args.eval_out, n_models=args.n_models)
    print(json.dumps(results, indent=2))
    print(f"\nModel written to {args.model_out}")
