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
import re
import pickle
import random

from app.db import get_cursor
from app.logging_store import ADHOC_PREFIX
from app.optimizer.bandit import POLICIES, BootstrappedEnsemble, select_index
from app.optimizer.cardinality import JoinCardinalityCorrector
from app.optimizer import calibration
from app.optimizer.conformal import ConformalGate, sigma_scale
from app.optimizer.features import build_feature_columns, featurize, to_vector
from app.optimizer.ranker import PairwisePlanRanker
from app.plan_extractor import _extract_join_types, _extract_scan_relations, _extract_tables
from app.schema_introspection import discover_table_cardinalities

# Deliberately NOT `models/plan_selector.pkl`.
#
# That file is `model_store.CURRENT_PATH` -- the one `learned.py` loads to serve
# with. `retrain.py` states the invariant plainly ("train() writes to a scratch
# path -- promotion is this module's decision, not train()'s") and honours it by
# passing an explicit path, but the CLI default did not, so
# `python -m app.train` overwrote the served model with an untested one and the
# registry's promotion decision was bypassed without a word.
#
# Measured: a retrain whose offline eval was worse than the incumbent on every
# policy (risk-averse headroom 0.045 against 0.365) was refused promotion by the
# registry and served anyway, because the pickle had already landed on the
# served path. `model_store.py`'s own docstring says the registry exists to stop
# exactly this: "Until now the pipeline overwrote models/plan_selector.pkl on
# every train".
MODEL_PATH = os.getenv("MODEL_OUT_PATH", "models/_trained.pkl")
EVAL_PATH = os.getenv("EVAL_OUT_PATH", "models/eval_results.json")

FETCH_SQL = """
    SELECT query_id, sql_text, hint, is_baseline, raw_plan, total_cost,
           actual_total_time_ms, created_at
    FROM plan_execution_log
    WHERE actual_total_time_ms IS NOT NULL
      AND query_id IS NOT NULL
      AND query_id NOT LIKE %s
"""

# Train on the action space that will actually be served.
#
# `docs/WRITEUP.md` 2.9 documents the train/serve difference and calls it
# harmless: `app.collect_data` gathers the forced join-method family
# (`HashJoin`/`NestLoop`/`MergeJoin` per join-order prefix) while
# `INCLUDE_JOIN_METHODS` is off at serving, so the model is never *asked* to
# score a family it has not seen. That is true, and it is only about coverage.
#
# It is not harmless for the predicted **level**, which is the number the safety
# gate reads. Measured over 12,826 collected candidates:
#
#     join-method candidates    n=980     mean ratio 5.49   median 3.16
#     everything else           n=11846   mean ratio 1.97   median 1.10
#
# A squared-loss model shrinks its predictions toward the mean of what it was
# trained on. Including that family moves the training mean from 1.97 to 2.24 --
# a factor of 1.137 -- and every served prediction is lifted with it. The
# measured systematic bias at serving was **x1.152**, which that accounts for
# almost exactly.
#
# The consequence was that the gate could not fire. Across 321 served
# candidates the model's most optimistic prediction anywhere was 0.817x while
# the gate deviates below 0.80x, and 64 of those candidates were genuinely
# faster than 0.8x. Not a confidence problem, and not a bad ranker: the same
# model captures 79% of the oracle headroom, because ranking is unaffected by a
# shift that applies to everything.
#
# So the rows for a family that will not be served are dropped, and which
# families those are is read from `optimizer/planner.py` -- the same flags the
# serving path uses -- rather than hardcoded here, so the two cannot drift.
# Anchored at a word boundary, because a plain substring test matches the
# *negative* forms too: "NestLoop(" is inside "NoNestLoop(". Those are a
# different family, they are served when `INCLUDE_NEGATIVE_JOINS` is on, and
# dropping their rows would be the opposite mistake -- training a model on
# fewer families than it will be asked about rather than more.
_JOIN_METHOD_RE = re.compile(r"\b(?:HashJoin|NestLoop|MergeJoin)\s*\(")


def _is_join_method_hint(hint: str | None) -> bool:
    return bool(hint) and _JOIN_METHOD_RE.search(hint) is not None


def rows_for_served_action_space(rows: list[dict]) -> tuple[list[dict], dict]:
    """
    Drop candidates from families the serving path does not offer.

    Returns the kept rows and a report, because silently training on a subset
    is the same class of mistake as silently serving one -- and a run that drops
    a thousand rows should say so.
    """
    from app.optimizer import planner

    if planner.INCLUDE_JOIN_METHODS:
        return rows, {"filtered": False, "reason": "join methods are served"}

    kept = [r for r in rows if not _is_join_method_hint(r.get("hint"))]
    dropped = len(rows) - len(kept)
    return kept, {
        "filtered": True,
        "dropped_rows": dropped,
        "kept_rows": len(kept),
        "reason": (
            "INCLUDE_JOIN_METHODS is off, so forced join-method candidates are "
            "never served. Training on them shifts every prediction upward "
            "toward their much slower mean, which is what stopped the safety "
            "gate from ever firing."
        ),
    }
# `adhoc:` rows are dashboard traffic (Phase 5's /query/analyze), which now
# carries a fingerprint id so it can be grouped for per-query statistics --
# see `logging_store.query_fingerprint`. They stay out of training regardless:
# /query/analyze runs one execution per candidate with no repetitions, and the
# noise that `_aggregate_repetitions` exists to suppress would come straight
# back in through the labels. Whether to fold live traffic into training is a
# real design question, not something to change as a side effect of giving
# those rows an id.


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


def _aggregate_repetitions(rows: list[dict], latest_run_only: bool = False) -> list[dict]:
    """
    Collapse repeated executions of the same (query, hint) into one row
    carrying the **median** latency.

    `latest_run_only` first discards everything but each query's most recent
    run. Training wants every measurement it can get and leaves this off; a
    *comparison* wants measurements taken under the same conditions, because
    the same native query in this log ranges from 2.4 ms to 13.7 ms across runs
    collected months and workloads apart. See `app.calibrate._grouped_rows`.

    Why median and not mean: latency has a heavy right tail (a candidate
    occasionally collides with autovacuum, a checkpoint, or another process
    on the machine). The mean chases those outliers; the median ignores
    them. Since the label *is* the thing being learned, this directly
    attacks the noise that docs/WRITEUP.md §2.3 identifies as the binding
    constraint on the whole system.
    """
    if latest_run_only:
        # `created_at` defaults to now() and Postgres freezes it per
        # transaction, so one collection of one query shares a single value --
        # which is what makes it an identifier for a run rather than a clock.
        newest: dict[str, object] = {}
        for row in rows:
            at = row.get("created_at")
            if at is None:
                continue
            current = newest.get(row["query_id"])
            if current is None or at > current:
                newest[row["query_id"]] = at
        if newest:
            rows = [
                row for row in rows
                if row.get("created_at") is not None
                and row["created_at"] == newest.get(row["query_id"])
            ]

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
        cur.execute(FETCH_SQL, (f"{ADHOC_PREFIX}%",))
        columns = [d[0] for d in cur.description]
        return [dict(zip(columns, r, strict=True)) for r in cur.fetchall()]


# Ensemble members are deliberately left deterministic apart from their
# bootstrap resample.
#
# Turning on row and feature sampling to make them disagree more was tried and
# measured: the spread got *smaller* (0.0264 -> 0.0247), its correlation with
# the actual error fell (+0.238 -> +0.218), and held-out MAE rose slightly. So
# did an earlier attempt at per-member seeds, which was a complete no-op.
#
# Both attempts assumed the members were too similar. They are not: they agree
# because they are all making the *same* mistake. Ensemble spread measures
# variance -- how much the fit moves when the data moves -- and this model's
# error is dominated by bias and by irreducible measurement noise, neither of
# which any amount of member diversity can reveal. That is why sigma is four
# times smaller than the error and why no arrangement of the ensemble fixes it.
#
# The consequence is in `optimizer/conformal.py`: a confidence signal here has
# to come from the residuals themselves, not from how much the members argue.
def _build_model():
    """A single regressor + the name of the backend that provided it."""
    try:
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            verbosity=-1,
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


# Risk level for the conformal safety gate: at most this share of served
# decisions should have a true log-ratio above the bound. 0.1 mirrors the
# regression tolerance the guard already runs at.
DEFAULT_CONFORMAL_ALPHA = float(os.getenv("CONFORMAL_ALPHA", "0.1"))


def train(
    model_path: str = MODEL_PATH,
    eval_path: str = EVAL_PATH,
    n_models: int = 8,
    aggregate_reps: bool = True,
    latest_run_only: bool = False,
    conformal_alpha: float = DEFAULT_CONFORMAL_ALPHA,
) -> dict:
    # Checked before any work: this used to sit beside the write, so an
    # invalid destination was only reported after a full training run.
    # A caller that names the served path is asking this function to make a
    # promotion decision it is not allowed to make. Refused loudly rather than
    # silently redirected, because a run that thinks it deployed a model and did
    # not is the same silent failure in the other direction.
    from app import model_store

    if os.path.abspath(model_path) == os.path.abspath(model_store.CURRENT_PATH):
        raise ValueError(
            f"{model_path} is the model being served. Training writes a candidate; "
            f"promoting it is `model_store.promote()`'s decision, made against the "
            f"incumbent (see app/retrain.py). Pass --model-out with another path, "
            f"or use POST /model/retrain which runs the champion/challenger comparison."
        )

    rows = _load_rows()
    if not rows:
        raise RuntimeError("plan_execution_log is empty -- run `python -m app.collect_data` first.")

    n_raw_rows = len(rows)

    # Before aggregation, so a family that is not served contributes nothing at
    # all -- not even to a median.
    rows, action_space_report = rows_for_served_action_space(rows)
    if action_space_report.get("filtered"):
        print(
            f"action space: dropped {action_space_report['dropped_rows']} rows for "
            f"families the serving path does not offer, kept "
            f"{action_space_report['kept_rows']}"
        )

    if aggregate_reps:
        # `latest_run_only` matters when the *database* changed between
        # collections, not just the workload. Training normally wants every
        # measurement it can get -- more repetitions make a better median. That
        # holds only while the thing being measured is the same thing.
        #
        # Measured here: two collections a few hours apart, with indexes created
        # and dropped, a VACUUM, a prewarm and two extensions installed in
        # between. Folding them together gives each (query, hint) a median
        # across two different databases, and the resulting model captured
        # **6.8%** of the oracle headroom against **77.9%** for one trained on a
        # single run. `_aggregate_repetitions` already warned about exactly this
        # ("the same native query in this log ranges from 2.4 ms to 13.7 ms
        # across runs collected months and workloads apart"); the option was
        # only ever wired up for calibration.
        rows = _aggregate_repetitions(rows, latest_run_only=latest_run_only)

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

    # Calibrate the safety gate on the ensemble's own out-of-bag residuals, so
    # `learned.py` can bound a prediction at a stated risk level instead of at
    # a chosen number of standard deviations. Costs no training data -- see
    # `optimizer/conformal.py` for why that matters here specifically.
    conformal = None
    uncertainty_scale = 1.0
    point_calibration = calibration.PointCalibration()
    calibration_report = None
    try:
        oob_means, oob_stds = ensemble.oob_mean_std(X_train)
        usable = [
            (mean, std, actual)
            for mean, std, actual in zip(oob_means, oob_stds, y_train, strict=True)
            if mean is not None
        ]
        if usable:
            oob_m = [row[0] for row in usable]
            oob_s = [row[1] for row in usable]
            oob_y = [row[2] for row in usable]

            # Correct the predicted *level* before anything is fitted on top of
            # it. Measured across 321 candidates on the workload, the raw model
            # predicted every one of them at 0.826x native or worse, while 54
            # were genuinely under 0.8x -- so the gate, which deviates below
            # 0.80x, could not fire on any query at any confidence setting.
            # That was "why is it never confident", and it was never about
            # confidence. See `optimizer/calibration.py`.
            #
            # The map is affine and increasing, so it cannot reorder candidates
            # and cannot change which plan the argmin picks -- only where those
            # predictions sit relative to a threshold.
            point_calibration = calibration.fit(oob_m, oob_y)
            calibration_report = calibration.report(point_calibration, oob_m, oob_y)
            if point_calibration.fitted:
                before = calibration_report["before"]
                after = calibration_report["after"]
                print(
                    f"point calibration: slope={point_calibration.slope:.3f} "
                    f"intercept={point_calibration.intercept:+.3f} "
                    f"on {point_calibration.n} out-of-bag rows"
                )
                print(
                    f"  systematic error x{before['systematic_factor']:.3f} "
                    f"-> x{after['systematic_factor']:.3f}; "
                    f"most optimistic prediction "
                    f"{before['min_predicted_ratio']:.3f}x "
                    f"-> {after['min_predicted_ratio']:.3f}x native"
                )
            else:
                print("point calibration: not fitted (too few rows, or an unusable slope)")

            # The conformal bound is fitted on the calibrated means, so the
            # interval brackets the number the gate will actually see. Fitting
            # it on raw means and then shifting the point underneath it would
            # silently invalidate its coverage guarantee.
            oob_calibrated = point_calibration.apply(oob_m)

            # Absolute rather than sigma-normalised: on this model the spread
            # is four times smaller than the error and near-zero on some rows,
            # which made the normalised bound explode. See conformal.py.
            conformal = ConformalGate(alpha=conformal_alpha).fit(
                oob_calibrated, oob_s, oob_y
            )
            # Separate from the conformal bound and used by the default gate:
            # puts `confidence_z` back into real standard deviations, so the
            # knob `app.calibrate` sweeps actually moves decisions.
            uncertainty_scale = sigma_scale(oob_calibrated, oob_s, oob_y)
    except (RuntimeError, ValueError) as exc:  # too few rows, or every row in-bag
        print(f"[warn] conformal gate not calibrated: {exc}")

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

    # Join-level cardinality correction, learned from the same executed plans.
    # Trained on the training queries only, for the same reason the selector is:
    # a corrector that has seen a held-out query's actual row counts would make
    # the evaluation of that query meaningless.
    train_plans = [rows[i]["raw_plan"] for i in train_idx]
    join_corrector = None
    try:
        join_corrector = JoinCardinalityCorrector(make_regressor).fit(train_plans)
    except ValueError as exc:  # not enough join observations yet
        print(f"[warn] join cardinality corrector not trained: {exc}")

    preds_test = ensemble.predict(X_test)
    # Error is now in log-ratio space. exp(MAE) reads as "typical multiplicative
    # error": 1.15 means predictions are typically off by ~15%.
    mae = (
        sum(abs(p - y) for p, y in zip(preds_test, y_test, strict=True)) / len(y_test)
        if y_test
        else None
    )
    test_means, test_stds = ensemble.predict_mean_std(X_test)
    mean_uncertainty = sum(test_stds) / len(test_stds) if test_stds else None

    # The guarantee, checked rather than asserted. Calibration coverage is
    # optimistic by construction; this is measured on the held-out queries, so
    # it is the number that says whether the gate is honest on data it has not
    # seen. Well below 1 - alpha means exchangeability has broken down, which is
    # exactly what a schema or workload change does.
    conformal_report = None
    if conformal is not None and conformal.fitted:
        conformal_report = {
            "alpha": conformal.alpha,
            "q": conformal.q,
            "n_calibration": conformal.n_calibration,
            "target_coverage": 1.0 - conformal.alpha,
            "held_out_coverage": (
                conformal.coverage(test_means, test_stds, y_test) if y_test else None
            ),
        }

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
        "join_cardinality_corrector": (
            "trained" if join_corrector is not None else "not trained"
        ),
        "conformal_gate": conformal_report or "not calibrated",
        # Which candidate families this model was trained on, against
        # which ones the serving path will offer it. See
        # `rows_for_served_action_space`.
        "action_space": action_space_report,
        # What correcting the predicted level changed. The headline number is
        # `after.min_predicted_ratio`: below the gate's threshold the gate can
        # fire, above it the gate is decorative whatever `confidence_z` says.
        "point_calibration": calibration_report or "not calibrated",
        # >1 means the ensemble's spread understates its real error by this
        # factor; `learned.py` multiplies sigma by it so `confidence_z` is
        # denominated in standard deviations rather than in whatever the
        # bootstrap happened to produce.
        "uncertainty_scale": uncertainty_scale,
        "n_rows_raw": n_raw_rows,
        "reps_aggregated_to_median": aggregate_reps,
        "latest_run_only": latest_run_only,
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
                "join_corrector": join_corrector,
                "conformal": conformal,
                # Applied to every prediction before the gate reads it -- see
                # `optimizer/calibration.py`. Absent in an older bundle, where
                # `learned.py` falls back to the identity.
                "point_calibration": point_calibration,
                "uncertainty_scale": uncertainty_scale,
                "target": "log_ratio_vs_native",
                # Which queries the model never saw. `app.calibrate` needs this
                # to choose a gate against honest predictions rather than
                # against the ones the model fitted.
                "held_out_queries": sorted(test_ids),
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
        "--latest-run-only",
        action="store_true",
        help="train on each query's most recent collection only -- for when the "
             "database changed between collections, since a median across two "
             "different databases is a label for neither",
    )
    parser.add_argument(
        "--n-models", type=int, default=8, help="bootstrapped ensemble size (drives uncertainty estimates)"
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="record the trained model in the version registry as a candidate. "
             "It is not promoted -- POST /model/retrain compares a challenger "
             "against the incumbent on a common held-out set and decides.",
    )
    args = parser.parse_args()

    results = train(
        model_path=args.model_out,
        eval_path=args.eval_out,
        n_models=args.n_models,
        latest_run_only=args.latest_run_only,
    )
    print(json.dumps(results, indent=2))

    print()
    print(f"Candidate written to {args.model_out}")
    if args.register:
        import pickle as _pickle

        from app import model_store

        with open(args.model_out, "rb") as fh:
            bundle = _pickle.load(fh)
        version_id = model_store.save_version(bundle, results)
        current = model_store.current_version()
        print(f"Registered as {version_id} (not promoted; serving {current}).")
    else:
        print("Not registered. Pass --register to record it as a candidate.")
    print(
        "Promotion is a comparison, not a side effect of training: "
        "POST /model/retrain scores challenger and champion on a common "
        "held-out set and promotes only on a real improvement."
    )
    print(f"\nModel written to {args.model_out}")
