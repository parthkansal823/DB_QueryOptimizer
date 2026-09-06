"""
Closes the learning loop.

Until this module existed, the system explored (Thompson sampling) but never
learned from what it found: feedback accumulated in `plan_execution_log` and
sat there until a human reran `app.train`. That is a strange thing for a
project called a *self-learning* optimizer, and it was the honest gap named
in `docs/WRITEUP.md`'s limitations.

## Champion / challenger

Retraining automatically is only safe if a worse model can't silently take
over. So a retrain never promotes on faith:

  1. Train a **challenger** on all accumulated feedback.
  2. Score the **champion** (the model currently being served) and the
     challenger on the *same* held-out queries.
  3. Promote only if the challenger wins by more than `min_improvement`.

The margin matters. `docs/WRITEUP.md` §2.2.1 shows offline evaluation is
optimistically biased and noisy; promoting on any improvement at all would
mean promoting on noise roughly half the time. Requiring a clear margin
makes the ratchet one-directional in expectation.

If there is no champion yet (first ever train), the challenger is promoted
unconditionally -- something must be served, and the cold-start heuristic is
what runs until then anyway.

Usage:
    python -m app.retrain            # retrain if enough new data, gate, maybe promote
    python -m app.retrain --force    # retrain regardless of how much is new
    python -m app.retrain --status   # what's deployed, what's pending
"""

from __future__ import annotations

import argparse
import os
import json

from app import model_store
from app.db import get_cursor
from app.logging_store import ADHOC_PREFIX
from app.optimizer.bandit import select_index
from app.optimizer.features import featurize, to_vector
from app.optimizer.learned import DEFAULT_SELECTION_POLICY, RANKER_POLICIES
from app.optimizer.ranker import borda_select
from app.train import _load_rows, _row_to_candidate, train

# Enough new executions to be worth the retrain cost. Retraining on a
# handful of new rows mostly reshuffles noise.
DEFAULT_MIN_NEW_ROWS = 200

# The challenger must beat the champion by at least this fraction of the
# champion's average latency. See the note on noise above.
DEFAULT_MIN_IMPROVEMENT = 0.02


def rows_since_last_training() -> int:
    """
    How many *trainable* executions have been logged since the deployed model
    trained.

    Counting every row overstated this badly, and not harmlessly: the
    dashboard presents it as pending feedback, and `DEFAULT_MIN_NEW_ROWS`
    gates retrains on it. Rows `app.train` filters out would trip that gate
    and buy a retrain on data the trainer never sees, so the count has to
    apply the same filter the trainer does.
    """
    registry_versions = model_store.list_versions()
    current = model_store.current_version()
    entry = next((v for v in registry_versions if v["version_id"] == current), None)

    # `NOT LIKE %s` rather than a literal pattern: psycopg2 treats `%` in the
    # query string as its own placeholder marker whenever parameters are
    # passed, so an inline 'adhoc:%' raises IndexError on the parameterised
    # branch below.
    trainable = (
        "actual_total_time_ms IS NOT NULL "
        "AND query_id IS NOT NULL AND query_id NOT LIKE %s"
    )
    adhoc_pattern = f"{ADHOC_PREFIX}%"

    with get_cursor() as cur:
        if entry is None:
            cur.execute(
                f"SELECT count(*) FROM plan_execution_log WHERE {trainable}",
                (adhoc_pattern,),
            )
            return cur.fetchone()[0]
        cur.execute(
            f"SELECT count(*) FROM plan_execution_log WHERE {trainable} AND created_at > %s",
            (adhoc_pattern, entry["created_at"]),
        )
        return cur.fetchone()[0]


def served_policy() -> str:
    """
    The selection policy this deployment actually serves with.

    Read from the environment rather than restated, because promotion has to
    score the policy that will run. It used to hardcode `risk_averse` while
    `docker-compose.yml` sets `SELECTION_POLICY: pairwise_rank` and the code
    default is `greedy` -- three policies named in three places.

    That is not academic. The model promoted on 2 September beat the champion by
    8.3% on the risk-averse score, and on the train-time eval its **pairwise_rank**
    headroom was flat (0.163 -> 0.162) while risk-averse *collapsed* from +0.365
    to -0.142. Promotion was deciding on a policy nobody runs, using a number
    that moved in the opposite direction to the one that matters.

    `docs/WRITEUP.md` 2.22 names the pattern: a constant that duplicates a
    decision made elsewhere will eventually disagree with it. This is the fourth
    instance.

    Reading the environment fixed the *hardcoding* and not the disagreement.
    `docker-compose.yml` set `SELECTION_POLICY` on the backend service, so the
    variable existed only inside that container -- and `app.retrain` and
    `app.calibrate` are run on the host, where it is unset. This function
    returned `greedy` while the server served `pairwise_rank`, which is 2.22
    again with an extra step: promotion and gate calibration both scored a
    policy nobody runs (2.43).

    The fix was to delete the duplicate rather than synchronise it. There is one
    default now, `learned.DEFAULT_SELECTION_POLICY`, and the compose override is
    gone, so host and container resolve the same value whether or not the
    environment carries it.
    """
    return os.getenv("SELECTION_POLICY", DEFAULT_SELECTION_POLICY)


def _score_bundle(
    bundle: dict,
    rows_by_query: dict[str, list[dict]],
    policy: str | None = None,
) -> float | None:
    """
    How this bundle would have performed on the held-out queries: the mean
    latency of the plans it picks, **and the worst single pick**.

    Returns `{"mean_ms", "worst_ratio_vs_best", "n_queries"}`, or None if
    unscoreable. Scores with the policy that serves (see `served_policy`), so a
    promotion decision is a statement about the system that will run rather than
    about a configuration it does not use.

    The worst case is here because the mean alone could not see the failure that
    mattered. A PostgreSQL 16-trained model, still deployed after the upgrade to
    18, predicted **0.291x** for a candidate its own training data measures at
    1.134x, and served it: 137 ms of native work became 160 ms. The PG18-trained
    challenger predicted 1.273x and would have declined. Scored on the mean, the
    two came out **1.0% apart** -- below the promotion bar -- because on most
    queries they pick the same plan, and the difference lives entirely in the
    handful where they diverge.

    An average answers "how good is this model on a typical query". Promotion
    needs "is this model ever catastrophically wrong", and a metric that cannot
    tell a uniformly mediocre model from a usually-fine-occasionally-disastrous
    one is not measuring what the safety gate exists to protect against.

    `worst_ratio_vs_best` is measured against the best candidate actually
    available for that query rather than against native, so it reads as "how
    much of the achievable win did the worst pick throw away" and stays
    comparable across queries of very different absolute cost.
    """
    if bundle is None:
        return None

    policy = policy or served_policy()
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    cardinalities = bundle["table_cardinalities"]

    # Local import: `app.calibrate` imports `served_policy` from here, so a
    # module-level import would close the cycle.
    from app.calibrate import _servable, without_provable_noops

    picked: list[float] = []
    worst_ratio = 1.0
    for rows in rows_by_query.values():
        # Three filters, all of them making this match what a served query is
        # actually offered. Without them promotion ranks two models on an action
        # space neither of them will ever be given.
        #
        #   - `is_baseline`: native is not a candidate. Serving passes
        #     `select_plan(candidates, baseline_plan=native)` -- the policy
        #     chooses among candidates and the *gate* decides whether to use
        #     the choice. Leaving native in the pool let a model score well by
        #     "picking" the plan it never had the option to pick.
        #   - `_servable`: the log records everything ever collected, and
        #     collection explores wider than serving does. `calibrate._servable`
        #     documents the cost of ignoring this: an all-candidates replay had
        #     every gate losing 10-25% because it kept choosing forced
        #     nested-loop plans the live path drops.
        #   - `disabled_nodes`: `learned._is_unsafe` refuses these outright, so
        #     scoring them credits a model for plans the server vetoes. This is
        #     the divergence that had the replay reporting +5.38% where the
        #     benchmark measured 0.0% (2.47).
        candidates = [
            c for c in _servable([_row_to_candidate(r) for r in rows if not r["is_baseline"]])
            if not c.get("disabled_nodes")
        ]
        # `best = min(latencies)` below is the number that has rejected two
        # challengers, so the pool it is taken over has to be plans a served
        # query could actually get. 2.76: 14.4% of logged candidates are
        # byte-identical to native, and one of them is the ceiling on 11 of 65
        # queries. Shared with `calibrate` and `train` so the three cannot
        # diverge again (2.47).
        baseline_rows = [r for r in rows if r["is_baseline"]]
        candidates = without_provable_noops(
            candidates,
            baseline_rows[0].get("raw_plan") if baseline_rows else None,
        )
        if not candidates:
            continue
        latencies = [c["actual_total_time_ms"] for c in candidates]
        vectors = [to_vector(featurize(c, cardinalities), feature_columns) for c in candidates]
        try:
            # `pairwise_rank` and `borda` live on the ranker, not the ensemble,
            # so they are not in `bandit.POLICIES` -- the same split
            # `calibrate.py` handles. Scoring a challenger under a policy the
            # server does not run is 2.43, and this is the yardstick that
            # decides promotions.
            if policy in RANKER_POLICIES:
                ranker = bundle.get("ranker")
                if ranker is None:
                    return None
                costs = [c["total_cost"] or float("inf") for c in candidates]
                scores = ranker.scores(vectors)
                if policy == "borda":
                    means, _ = bundle["model"].predict_mean_std(vectors)
                    index = borda_select(means, scores, tie_break_costs=costs)
                else:
                    index = ranker.select(
                        vectors, tie_break_costs=costs, scores=scores
                    )
            else:
                index, _ = select_index(model, vectors, policy=policy)
        except Exception:  # noqa: BLE001 - a stale bundle shouldn't block a retrain
            return None
        picked.append(latencies[index])
        best = min(latencies)
        if best > 0:
            worst_ratio = max(worst_ratio, latencies[index] / best)

    if not picked:
        return None
    return {
        "mean_ms": sum(picked) / len(picked),
        "worst_ratio_vs_best": worst_ratio,
        "n_queries": len(picked),
    }


def _load_challenger_bundle(path: str) -> dict:
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


def _held_out_rows(
    fraction: float = 0.25,
    bundles: "list[dict] | None" = None,
) -> dict[str, list[dict]]:
    """
    A shared yardstick for comparing two models.

    The docstring here used to read "the most recent queries' rows" and the code
    read `sorted(by_query)[-n:]` -- which sorts query **ids alphabetically**.
    Nothing about it was chronological, and `created_at` was sitting unused on
    every row. On this workload the alphabetical tail is `gen_2*` and `shape_*`,
    a group defined by naming rather than by recency or by anything a model had
    or had not seen.

    That mattered, and not subtly. The challenger being compared in 2.40 had
    **trained on 10 of the 15** queries in that yardstick, so promotion was
    largely reading how well it memorised them, and the newer model wins that
    contest by construction.

    Two changes:

    **Recency is measured, not assumed.** Queries are ordered by their most
    recent `created_at`, so "the most recent quarter" is what the name says.

    **Contamination is preferred against, and reported when unavoidable.** Given
    `bundles`, queries that none of them trained on are used if there are enough
    of them. Here there are not: the champion held out 8 queries and the
    challenger 15, and the two lists intersect in **two**
    (`corr_6w_everything`, `shape_distinct_brands`), which cannot decide
    anything. So the recency yardstick stands and `yardstick_provenance` states
    how much of it each model had already seen -- the alternative being a verdict
    that looks clean and is not.
    """
    rows = _load_rows()
    by_query: dict[str, list[dict]] = {}
    latest: dict[str, str] = {}
    for row in rows:
        qid = row["query_id"]
        by_query.setdefault(qid, []).append(row)
        stamp = str(row.get("created_at") or "")
        if stamp > latest.get(qid, ""):
            latest[qid] = stamp

    # Ties (and rows with no timestamp) fall back to the id, so the order is
    # total and the selection reproducible.
    query_ids = sorted(by_query, key=lambda q: (latest.get(q, ""), q))
    n_test = max(1, round(len(query_ids) * fraction))

    if bundles:
        unseen = [
            q for q in query_ids
            if all(q in set(b.get("held_out_queries") or ()) for b in bundles)
        ]
        # Below MIN_UNCONTAMINATED_QUERIES the rates are too coarse to separate
        # two models, and an honest contaminated sample beats a clean one too
        # small to read.
        if len(unseen) >= MIN_UNCONTAMINATED_QUERIES:
            return {q: by_query[q] for q in unseen[-n_test:]}

    return {q: by_query[q] for q in query_ids[-n_test:]}


# How many queries neither model trained on it takes before that set is
# preferred over the recency window. Chosen to match `calibrate`'s own floor for
# the same judgement: below a handful, a rate is not a measurement.
MIN_UNCONTAMINATED_QUERIES = 5


def yardstick_provenance(
    held_out: dict[str, list[dict]], bundles: "dict[str, dict]"
) -> dict:
    """
    How much of the yardstick each model had already seen.

    A promotion verdict is only as good as the sample it was measured on, and a
    sample the challenger trained on two-thirds of is measuring memorisation.
    Recording it makes that visible in `app.retrain --status` instead of leaving
    it to be rediscovered.
    """
    ids = set(held_out)
    out: dict = {"n_queries": len(ids)}
    for name, bundle in bundles.items():
        recorded = set((bundle or {}).get("held_out_queries") or ())
        if not recorded:
            # A bundle that never recorded its split cannot be checked, which is
            # not the same as a bundle that is clean.
            out[f"{name}_trained_on"] = None
            continue
        out[f"{name}_trained_on"] = len(ids - recorded)
    return out


def retrain_if_needed(
    min_new_rows: int = DEFAULT_MIN_NEW_ROWS,
    min_improvement: float = DEFAULT_MIN_IMPROVEMENT,
    force: bool = False,
    drop_constant_features: bool = False,
) -> dict:
    """Train a challenger and let the held-out comparison decide.

    `drop_constant_features` builds the challenger without the features that
    take one value across a query's own candidates (2.69). Measured over 16
    paired splits that is worth a mean +8.3 points of held-out capture,
    signed-rank p = 0.026 -- which is evidence, not a promotion. The comparison
    below is what promotes, and 2.63 is why: a challenger with a better mean and
    a worse worst case was rejected, correctly.
    """
    new_rows = rows_since_last_training()
    if not force and new_rows < min_new_rows:
        return {
            "action": "skipped",
            "reason": f"only {new_rows} new rows since last training (need {min_new_rows})",
            "new_rows": new_rows,
        }

    champion_id = model_store.current_version()
    champion = model_store.load_version(champion_id) if champion_id else None

    # Train the challenger. `train()` writes to a scratch path -- promotion
    # is this module's decision, not train()'s.
    challenger_path = "models/_challenger.pkl"
    metrics = train(
        model_path=challenger_path,
        eval_path="models/_challenger_eval.json",
        drop_constant_features=drop_constant_features,
    )
    challenger = _load_challenger_bundle(challenger_path)

    version_id = model_store.save_version(challenger, metrics)

    if champion is None:
        model_store.promote(version_id, reason="first model")
        return {
            "action": "promoted",
            "reason": "no incumbent to compare against",
            "version_id": version_id,
            "new_rows": new_rows,
            "metrics": metrics,
        }

    held_out = _held_out_rows(bundles=[champion, challenger])
    champion_score = _score_bundle(champion, held_out)
    challenger_score = _score_bundle(challenger, held_out)

    if champion_score is None or challenger_score is None:
        return {
            "action": "rejected",
            "reason": "could not score both models on a common held-out set",
            "version_id": version_id,
            "new_rows": new_rows,
        }

    improvement = (
        champion_score["mean_ms"] - challenger_score["mean_ms"]
    ) / champion_score["mean_ms"]
    champ_worst = champion_score["worst_ratio_vs_best"]
    chall_worst = challenger_score["worst_ratio_vs_best"]
    # A challenger that improves the mean while picking one plan far worse than
    # the champion's worst is not an improvement to a system whose whole purpose
    # is not being catastrophically wrong.
    worst_regressed = chall_worst > champ_worst * 1.1
    worst_improved = chall_worst < champ_worst * 0.9
    decision = {
        "version_id": version_id,
        "new_rows": new_rows,
        "champion_id": champion_id,
        "champion_avg_latency_ms": champion_score["mean_ms"],
        "challenger_avg_latency_ms": challenger_score["mean_ms"],
        "champion_worst_ratio": champ_worst,
        "challenger_worst_ratio": chall_worst,
        "improvement": improvement,
        "min_improvement": min_improvement,
        # Which policy the two models were compared under. Without it the
        # registry records a verdict with no way to tell what it measured.
        "scored_with_policy": served_policy(),
        # How much of the comparison set each model had already trained on. A
        # promotion decided on queries the challenger memorised is not a
        # promotion decision, and this is what makes that legible rather than
        # something to be rediscovered (see `_held_out_rows`).
        "yardstick": yardstick_provenance(
            held_out, {"champion": champion, "challenger": challenger}
        ),
        "metrics": metrics,
    }

    # Two ways to earn promotion, because two different things can be wrong
    # with a model.
    #
    # A clear mean improvement is the ordinary case. A materially better *worst*
    # pick is the other, and it has to count on its own: a challenger that turns
    # a catastrophic prediction into a correct one may barely move the mean --
    # measured at 1.0%, below the bar -- while being the difference between
    # serving a plan 17% slower than native and declining to serve it.
    if worst_regressed:
        decision["action"] = "rejected"
        decision["reason"] = (
            f"challenger's worst pick is {chall_worst:.2f}x the best available plan "
            f"against the champion's {champ_worst:.2f}x -- a better mean does not "
            f"buy a worse worst case"
        )
    elif improvement > min_improvement:
        model_store.promote(
            version_id,
            reason=f"beat champion by {improvement:.1%} on {served_policy()}",
        )
        decision["action"] = "promoted"
    elif worst_improved:
        model_store.promote(
            version_id,
            reason=(
                f"worst pick improved from {champ_worst:.2f}x to {chall_worst:.2f}x of "
                f"the best available plan (mean moved {improvement:+.1%}, below the "
                f"{min_improvement:.0%} bar)"
            ),
        )
        decision["action"] = "promoted"
    else:
        decision["action"] = "rejected"
        decision["reason"] = (
            f"challenger moved the mean {improvement:+.1%} (bar {min_improvement:.0%}) "
            f"and the worst pick from {champ_worst:.2f}x to {chall_worst:.2f}x -- "
            f"neither is a real improvement"
        )
    return decision


def status() -> dict:
    return {
        "current_version": model_store.current_version(),
        "rows_since_last_training": rows_since_last_training(),
        "versions": [
            {k: v[k] for k in ("version_id", "created_at", "promoted")}
            for v in model_store.list_versions()[:10]
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="retrain regardless of new-row count")
    parser.add_argument("--status", action="store_true", help="show deployment status and exit")
    parser.add_argument("--rollback", action="store_true", help="promote the previous version")
    parser.add_argument("--min-new-rows", type=int, default=DEFAULT_MIN_NEW_ROWS)
    parser.add_argument("--min-improvement", type=float, default=DEFAULT_MIN_IMPROVEMENT)
    parser.add_argument(
        "--drop-constant-features",
        action="store_true",
        help="build the challenger without features that cannot separate a "
             "query's candidates (2.69)",
    )
    args = parser.parse_args()

    if args.status:
        print(json.dumps(status(), indent=2))
    elif args.rollback:
        target = model_store.rollback()
        print(json.dumps({"action": "rollback", "promoted": target}, indent=2))
    else:
        print(
            json.dumps(
                retrain_if_needed(
                    min_new_rows=args.min_new_rows,
                    min_improvement=args.min_improvement,
                    force=args.force,
                    drop_constant_features=args.drop_constant_features,
                ),
                indent=2,
                default=str,
            )
        )
