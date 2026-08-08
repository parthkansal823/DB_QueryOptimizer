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
import os
import pickle
import random

from app.db import get_cursor
from app.optimizer.features import build_feature_columns, featurize, to_vector
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
    try:
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=-1
        ), "lightgbm"
    except (ImportError, OSError):
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42), "sklearn-gbrt"


def _query_level_split(query_ids: list[str], test_fraction: float = 0.25, seed: int = 42):
    unique = sorted(set(query_ids))
    rng = random.Random(seed)
    rng.shuffle(unique)
    n_test = max(1, round(len(unique) * test_fraction))
    test_ids = set(unique[:n_test])
    train_ids = set(unique[n_test:])
    return train_ids, test_ids


def _evaluate_selection(
    rows_by_query: dict[str, list[dict]], model, feature_columns, table_cardinalities
) -> dict:
    """For each held-out query, compare native vs. heuristic vs. model picks."""
    native_latencies, heuristic_latencies, model_latencies = [], [], []
    model_beats_heuristic = 0

    for query_id, rows in rows_by_query.items():
        baseline_rows = [r for r in rows if r["is_baseline"]]
        if not baseline_rows:
            continue
        native_latency = baseline_rows[0]["actual_total_time_ms"]

        candidates = [_row_to_candidate(r) for r in rows]
        costs = [c["total_cost"] or float("inf") for c in candidates]
        heuristic_idx = costs.index(min(costs))
        heuristic_latency = candidates[heuristic_idx]["actual_total_time_ms"]

        vectors = [to_vector(featurize(c, table_cardinalities), feature_columns) for c in candidates]
        predictions = model.predict(vectors)
        model_idx = min(range(len(predictions)), key=lambda i: predictions[i])
        model_latency = candidates[model_idx]["actual_total_time_ms"]

        native_latencies.append(native_latency)
        heuristic_latencies.append(heuristic_latency)
        model_latencies.append(model_latency)
        if model_latency <= heuristic_latency:
            model_beats_heuristic += 1

    n = len(native_latencies)
    return {
        "n_held_out_queries": n,
        "avg_latency_ms": {
            "native": sum(native_latencies) / n if n else None,
            "heuristic": sum(heuristic_latencies) / n if n else None,
            "learned": sum(model_latencies) / n if n else None,
        },
        "model_beats_or_ties_heuristic_rate": model_beats_heuristic / n if n else None,
    }


def train() -> dict:
    rows = _load_rows()
    if not rows:
        raise RuntimeError("plan_execution_log is empty -- run `python -m app.collect_data` first.")

    table_cardinalities = _discover_cardinalities()
    feature_columns = build_feature_columns(list(table_cardinalities))

    candidates = [_row_to_candidate(r) for r in rows]
    query_ids = [r["query_id"] for r in rows]

    X_all = [to_vector(featurize(c, table_cardinalities), feature_columns) for c in candidates]
    y_all = [c["actual_total_time_ms"] for c in candidates]

    train_ids, test_ids = _query_level_split(query_ids)
    train_idx = [i for i, q in enumerate(query_ids) if q in train_ids]
    test_idx = [i for i, q in enumerate(query_ids) if q in test_ids]

    X_train = [X_all[i] for i in train_idx]
    y_train = [y_all[i] for i in train_idx]
    X_test = [X_all[i] for i in test_idx]
    y_test = [y_all[i] for i in test_idx]

    model, backend_name = _build_model()
    model.fit(X_train, y_train)

    preds_test = model.predict(X_test)
    mae = sum(abs(p - y) for p, y in zip(preds_test, y_test)) / len(y_test) if y_test else None

    rows_by_query: dict[str, list[dict]] = {}
    for row in rows:
        if row["query_id"] in test_ids:
            rows_by_query.setdefault(row["query_id"], []).append(row)

    selection_eval = _evaluate_selection(rows_by_query, model, feature_columns, table_cardinalities)

    results = {
        "model_backend": backend_name,
        "n_rows_total": len(rows),
        "n_rows_train": len(X_train),
        "n_rows_test": len(X_test),
        "test_mae_ms": mae,
        "n_tables_in_schema": len(table_cardinalities),
        **selection_eval,
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(
            {"model": model, "feature_columns": feature_columns, "table_cardinalities": table_cardinalities},
            f,
        )

    with open(EVAL_PATH, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    results = train()
    print(json.dumps(results, indent=2))
    print(f"\nModel written to {MODEL_PATH}")
