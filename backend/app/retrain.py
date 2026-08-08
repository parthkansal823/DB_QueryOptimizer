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
import json

from app import model_store
from app.db import get_cursor
from app.logging_store import ADHOC_PREFIX
from app.optimizer.bandit import select_index
from app.optimizer.features import featurize, to_vector
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


def _score_bundle(bundle: dict, rows_by_query: dict[str, list[dict]]) -> float | None:
    """
    Average latency of the plans this bundle would have picked, over the
    given held-out queries. Lower is better; None if unscoreable.
    """
    if bundle is None:
        return None

    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    cardinalities = bundle["table_cardinalities"]

    picked: list[float] = []
    for rows in rows_by_query.values():
        candidates = [_row_to_candidate(r) for r in rows]
        latencies = [c["actual_total_time_ms"] for c in candidates]
        vectors = [to_vector(featurize(c, cardinalities), feature_columns) for c in candidates]
        try:
            index, _ = select_index(model, vectors, policy="risk_averse")
        except Exception:  # noqa: BLE001 - a stale bundle shouldn't block a retrain
            return None
        picked.append(latencies[index])

    return sum(picked) / len(picked) if picked else None


def _load_challenger_bundle(path: str) -> dict:
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


def _held_out_rows(fraction: float = 0.25) -> dict[str, list[dict]]:
    """The most recent queries' rows, grouped -- a shared yardstick for both models."""
    rows = _load_rows()
    by_query: dict[str, list[dict]] = {}
    for row in rows:
        by_query.setdefault(row["query_id"], []).append(row)

    query_ids = sorted(by_query)
    n_test = max(1, round(len(query_ids) * fraction))
    return {q: by_query[q] for q in query_ids[-n_test:]}


def retrain_if_needed(
    min_new_rows: int = DEFAULT_MIN_NEW_ROWS,
    min_improvement: float = DEFAULT_MIN_IMPROVEMENT,
    force: bool = False,
) -> dict:
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
    metrics = train(model_path=challenger_path, eval_path="models/_challenger_eval.json")
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

    held_out = _held_out_rows()
    champion_score = _score_bundle(champion, held_out)
    challenger_score = _score_bundle(challenger, held_out)

    if champion_score is None or challenger_score is None:
        return {
            "action": "rejected",
            "reason": "could not score both models on a common held-out set",
            "version_id": version_id,
            "new_rows": new_rows,
        }

    improvement = (champion_score - challenger_score) / champion_score
    decision = {
        "version_id": version_id,
        "new_rows": new_rows,
        "champion_id": champion_id,
        "champion_avg_latency_ms": champion_score,
        "challenger_avg_latency_ms": challenger_score,
        "improvement": improvement,
        "min_improvement": min_improvement,
        "metrics": metrics,
    }

    if improvement > min_improvement:
        model_store.promote(version_id, reason=f"beat champion by {improvement:.1%}")
        decision["action"] = "promoted"
    else:
        decision["action"] = "rejected"
        decision["reason"] = (
            f"challenger improved by {improvement:.1%}, below the {min_improvement:.0%} bar"
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
                ),
                indent=2,
                default=str,
            )
        )
