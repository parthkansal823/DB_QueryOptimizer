"""
Compares model classes on the actual data instead of assuming one.

The pipeline defaulted to LightGBM because gradient-boosted trees are the
sensible first choice for tabular data, and the roadmap suggested it. That
is a reasonable prior, not evidence. Different model families have genuinely
different inductive biases, and which one wins depends on the shape of the
data in front of them:

  - **Boosted trees** handle feature interactions and skew well, but can
    overfit hard on a few hundred rows.
  - **Random / extremely-randomised forests** average many decorrelated
    trees; they usually beat boosting when the training set is small and
    noisy, which is exactly this project's regime (docs/WRITEUP.md §2.3).
  - **Ridge** is a linear baseline. If it wins, the features are basically
    linear and everything fancier is wasted capacity -- worth knowing.
  - **A small MLP** can capture smooth interactions trees approximate in
    steps, but wants more data than we have.

So this measures all of them under the same query-level split and reports a
ranking. `app.train --model auto` then uses the winner.

Two things it deliberately measures beyond raw error:

  - **Ranking accuracy** -- what fraction of held-out queries the model's
    top pick is genuinely the fastest plan. This matters more than MAE:
    the model's job is to *order* candidates, and a model with worse
    absolute error but better ordering makes better decisions.
  - **Oracle headroom captured** -- the end-to-end number, since a model
    can rank well and still pick badly when it does err.
"""

from __future__ import annotations

import argparse
import json

CANDIDATE_MODELS = ("lightgbm", "random_forest", "extra_trees", "gradient_boosting", "ridge", "mlp")


def build_model(name: str):
    """Construct one regressor by name. Unavailable backends raise ImportError."""
    if name == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            min_child_samples=5, random_state=42, verbosity=-1,
        )
    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
        )
    if name == "extra_trees":
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
        )
    if name == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(n_estimators=300, max_depth=3, random_state=42)
    if name == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        # Scaling matters here: features range from 0/1 flags to log-row-counts.
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "mlp":
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32), max_iter=1500,
                early_stopping=True, random_state=42,
            ),
        )
    raise ValueError(f"unknown model {name!r}; expected one of {CANDIDATE_MODELS}")


def evaluate_model(name: str, data: dict) -> dict | None:
    """Fit `name` on the training split and score it on held-out queries."""
    try:
        model = build_model(name)
    except (ImportError, OSError) as exc:
        return {"model": name, "error": f"unavailable: {exc}"}

    model.fit(data["X_train"], data["y_train"])
    preds = model.predict(data["X_test"])

    mae = sum(abs(p - y) for p, y in zip(preds, data["y_test"])) / len(data["y_test"])

    # Ranking + end-to-end quality on held-out queries.
    correct_top1 = 0
    native_total = served_total = oracle_total = 0.0
    for group in data["test_groups"]:
        vectors, latencies, native_ms = group["vectors"], group["latencies"], group["native_ms"]
        if not vectors:
            continue
        scores = model.predict(vectors)
        pick = min(range(len(scores)), key=lambda i: scores[i])
        if latencies[pick] == min(latencies):
            correct_top1 += 1
        native_total += native_ms
        served_total += min(latencies[pick], native_ms)  # native is always available
        oracle_total += min([native_ms] + latencies)

    n_groups = len(data["test_groups"]) or 1
    headroom = native_total - oracle_total
    return {
        "model": name,
        "test_mae": mae,
        "top1_accuracy": correct_top1 / n_groups,
        "headroom_captured": ((native_total - served_total) / headroom) if headroom > 0 else None,
    }


def compare(data: dict, models: tuple[str, ...] = CANDIDATE_MODELS) -> dict:
    results = [evaluate_model(name, data) for name in models]
    usable = [r for r in results if "error" not in r]

    # Rank by top-1 accuracy first: the model's job is to *order* candidates,
    # and a lower MAE that orders worse makes worse decisions. MAE breaks ties.
    best = max(usable, key=lambda r: (r["top1_accuracy"], -r["test_mae"]), default=None)
    return {
        "results": sorted(usable, key=lambda r: -r["top1_accuracy"]),
        "unavailable": [r for r in results if "error" in r],
        "best": best["model"] if best else None,
    }


def build_dataset() -> dict:
    """Assemble the same split `app.train` uses, plus per-query test groups."""
    from app.optimizer.features import build_feature_columns, featurize, to_vector
    from app.train import (
        _aggregate_repetitions,
        _discover_cardinalities,
        _load_rows,
        _query_level_split,
        _relative_targets,
        _row_to_candidate,
    )

    rows = _aggregate_repetitions(_load_rows())
    cardinalities = _discover_cardinalities()
    columns = build_feature_columns(list(cardinalities))

    candidates = [_row_to_candidate(r) for r in rows]
    query_ids = [r["query_id"] for r in rows]
    X = [to_vector(featurize(c, cardinalities), columns) for c in candidates]
    y = _relative_targets(rows)

    train_ids, test_ids = _query_level_split(query_ids)
    train_idx = [i for i, q in enumerate(query_ids) if q in train_ids]
    test_idx = [i for i, q in enumerate(query_ids) if q in test_ids]

    groups: dict[str, dict] = {}
    for i, row in enumerate(rows):
        if row["query_id"] not in test_ids:
            continue
        g = groups.setdefault(row["query_id"], {"vectors": [], "latencies": [], "native_ms": None})
        if row["is_baseline"]:
            g["native_ms"] = row["actual_total_time_ms"]
        else:
            g["vectors"].append(X[i])
            g["latencies"].append(row["actual_total_time_ms"])

    return {
        "X_train": [X[i] for i in train_idx],
        "y_train": [y[i] for i in train_idx],
        "X_test": [X[i] for i in test_idx],
        "y_test": [y[i] for i in test_idx],
        "test_groups": [g for g in groups.values() if g["native_ms"] and g["vectors"]],
        "n_features": len(columns),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=list(CANDIDATE_MODELS))
    args = parser.parse_args()

    data = build_dataset()
    print(
        f"{len(data['X_train'])} train / {len(data['X_test'])} test rows, "
        f"{len(data['test_groups'])} held-out queries, {data['n_features']} features\n"
    )

    report = compare(data, tuple(args.models))
    print(f"{'model':<20} {'top1_acc':>9} {'headroom':>9} {'mae':>8}")
    for r in report["results"]:
        headroom = f"{r['headroom_captured'] * 100:>8.1f}%" if r["headroom_captured"] is not None else "       -"
        print(f"{r['model']:<20} {r['top1_accuracy'] * 100:>8.0f}% {headroom} {r['test_mae']:>8.3f}")
    for r in report["unavailable"]:
        print(f"{r['model']:<20} {r['error']}")

    print(f"\nBest by top-1 ranking accuracy: {report['best']}")
    print(json.dumps({"best": report["best"]}, indent=2))
