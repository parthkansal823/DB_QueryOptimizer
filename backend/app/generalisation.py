"""
Does the model generalise? Measured on predictions, not on decisions.

`app.benchmark`'s held-out column cannot answer this. It reports captured
headroom on queries the model never trained on -- but on this workload the gate
acts on **one held-out query in six** and vetoes the rest, so the number is a
ratio whose numerator comes from one or two queries and whose denominator is the
headroom of all of them. A single win or loss moves it by ten points, which is
why two measurements of the same model gave -4.8% and +3.1%. See
`docs/WRITEUP.md` 2.32.

The gate's abstentions are correct behaviour -- an unseen query gets a stricter
bar (2.21) -- but they mean the benchmark is measuring *which query the gate
allowed*, not whether the model understands queries it has not seen.

This measures the model directly: for every logged candidate with a real
latency, how far is the predicted ratio from the actual one? Split by whether
the query was in the training set. If the two distributions are close, the model
generalises. If held-out error is much larger, it memorised.

That question has an answer whether or not the gate ever fires, which is the
whole point.
"""

from __future__ import annotations

import math
import statistics

from app.optimizer.features import featurize, to_vector


def _median_baseline(rows: list[dict]) -> float | None:
    """The query's native latency: median over its baseline rows."""
    latencies = [
        r["actual_total_time_ms"]
        for r in rows
        if r.get("is_baseline") and r.get("actual_total_time_ms")
    ]
    return statistics.median(latencies) if latencies else None


def prediction_errors(
    bundle: dict, rows_by_query: dict[str, list[dict]], query_ids: set[str]
) -> list[float]:
    """
    Absolute log-ratio error for every scoreable candidate in `query_ids`.

    Log space because the target is a ratio: predicting 2x when the truth is 1x
    and predicting 0.5x when the truth is 1x are the same size of mistake, and a
    linear error would call the first one twice as bad.
    """
    model = bundle["model"]
    columns = bundle["feature_columns"]
    cardinalities = bundle["table_cardinalities"]

    errors: list[float] = []
    for query_id, rows in rows_by_query.items():
        if query_id not in query_ids:
            continue
        native = _median_baseline(rows)
        if not native:
            continue

        candidates = [
            r
            for r in rows
            if not r.get("is_baseline") and r.get("actual_total_time_ms")
        ]
        if not candidates:
            continue

        vectors = [
            to_vector(featurize(_as_candidate(r), cardinalities), columns)
            for r in candidates
        ]
        means, _ = model.predict_mean_std(vectors)
        # strict: a length mismatch would pair a prediction with the wrong
        # row and report a plausible-looking error for a candidate nobody
        # scored. Silent mis-pairing is the failure mode this whole file
        # exists to measure; it should not be introduced measuring it.
        for row, mean in zip(candidates, means, strict=True):
            actual = row["actual_total_time_ms"] / native
            if actual <= 0:
                continue
            errors.append(abs(mean - math.log(actual)))
    return errors


def _as_candidate(row: dict) -> dict:
    """A log row in the shape `featurize` expects."""
    return {
        "raw_plan": row.get("raw_plan") or {},
        "total_cost": row.get("total_cost"),
        "hint": row.get("hint"),
        "tables_scanned": row.get("tables_scanned") or [],
        "scan_relations": row.get("scan_relations") or {},
        "join_types": row.get("join_types") or [],
    }


def _summary(errors: list[float]) -> dict:
    if not errors:
        return {"n": 0}
    errors = sorted(errors)
    median = statistics.median(errors)
    return {
        "n": len(errors),
        "median_log_error": median,
        # The same number as a multiplicative factor, which is the only form
        # anyone can reason about: 0.18 in log space is "typically 1.20x out".
        "typical_factor": math.exp(median),
        "p90_log_error": errors[int(len(errors) * 0.9)],
        "p90_factor": math.exp(errors[int(len(errors) * 0.9)]),
    }


def report(bundle: dict, rows_by_query: dict[str, list[dict]]) -> dict:
    """
    Prediction accuracy on trained-on queries against held-out ones.

    `ratio` is the headline: held-out typical error divided by trained-on
    typical error. Near 1.0 means the model does as well on queries it has never
    seen as on the ones it learned from. Much above 1.0 means it memorised.
    """
    held_out = set(bundle.get("held_out_queries") or ())
    trained = set(rows_by_query) - held_out

    held_summary = _summary(prediction_errors(bundle, rows_by_query, held_out))
    trained_summary = _summary(prediction_errors(bundle, rows_by_query, trained))

    ratio = None
    if held_summary.get("n") and trained_summary.get("n"):
        base = trained_summary["median_log_error"]
        if base > 0:
            ratio = held_summary["median_log_error"] / base

    return {
        "trained_on": trained_summary,
        "held_out": held_summary,
        "n_held_out_queries": len(held_out & set(rows_by_query)),
        "n_trained_queries": len(trained),
        "held_out_error_ratio": ratio,
        "reading": _reading(ratio),
    }


def _reading(ratio: float | None) -> str:
    if ratio is None:
        return "not enough data on one side of the split to compare"
    if ratio <= 1.25:
        return (
            "generalises: prediction error on unseen queries is close to error on "
            "trained ones"
        )
    if ratio <= 2.0:
        return (
            "partially generalises: noticeably worse on unseen queries, but the "
            "same order of accuracy"
        )
    return (
        "memorised: predictions on unseen queries are much worse than on trained "
        "ones, so headline numbers driven by trained queries will not transfer"
    )


def main() -> None:
    import json

    from app import model_store
    from app.train import (
        _aggregate_repetitions,
        _load_rows,
        rows_for_served_action_space,
        rows_without_disabled_plans,
    )

    bundle = model_store.load_version(model_store.current_version())

    # The same two filters training applies, for the same reason.
    #
    # Without them this scores the model on 567 forced join-method rows and 294
    # disabled-plan rows that `train` deliberately drops -- families it was
    # never taught and will never be asked about. That inflates the error and
    # measures the train/serve skew of 2.18 rather than generalisation, which
    # would be committing the bug while building the instrument for it.
    rows, disabled_report = rows_without_disabled_plans(_load_rows())
    rows, action_space_report = rows_for_served_action_space(rows)
    if disabled_report["filtered"] or action_space_report.get("filtered"):
        print(
            f"scoring on the served action space: dropped "
            f"{disabled_report.get('dropped_rows', 0)} disabled-plan and "
            f"{action_space_report.get('dropped_rows', 0)} unserved-family rows"
        )

    # Collapse repetitions to their median, exactly as training does, so the
    # model is scored on the shape of data it was fitted on rather than being
    # charged for the 18.6% run-to-run variation (2.18) it was never asked to
    # predict.
    #
    # This was expected to *lower* the error and it raised it: 2.21x typical on
    # raw rows, 2.93x on aggregated ones. Aggregating collapses 4,764 rows to
    # 432, and the raw set was dominated by whichever candidates happened to be
    # measured most often -- easy ones, repeatedly. One row per (query, hint)
    # gives the hard candidates their real weight. The aggregated figure is the
    # honest one, and it is worse.
    rows = _aggregate_repetitions(rows)

    rows_by_query: dict[str, list[dict]] = {}
    for row in rows:
        rows_by_query.setdefault(row["query_id"], []).append(row)

    result = report(bundle, rows_by_query)
    print(json.dumps(result, indent=2))
    print()
    print(f"trained on {result['n_trained_queries']} queries, "
          f"held out {result['n_held_out_queries']}")
    for name in ("trained_on", "held_out"):
        s = result[name]
        if s.get("n"):
            print(f"  {name:12s} typically {s['typical_factor']:.2f}x out "
                  f"(p90 {s['p90_factor']:.2f}x, n={s['n']})")
    print()
    print(result["reading"])


if __name__ == "__main__":
    main()
