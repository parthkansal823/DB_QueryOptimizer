r"""
End-to-end latency, planning included, with and without the decision cache.

Every other number this project reports is an *execution* time: the plan the
optimizer chose, timed by `EXPLAIN ANALYZE`, against native timed the same way.
That leaves out the cost of choosing. `planner.plan_query` plans every candidate
with `EXPLAIN` before it picks one, and on the query profiled in WRITEUP 2.87
that planning cost 41-47% of the query's own execution time. A system that saves
3.8% of execution time and spends 40% choosing is slower, not faster, unless
the choice is reused -- which is what `DecisionCache` is for.

So this measures what a client actually waits for, three ways, per query:

    native    the query as written
    uncached  plan_query with no cache (every candidate planned), then the
              chosen plan executed
    cached    plan_query with a warm DecisionCache, then the chosen plan executed

Each arm is timed from the client (`time.perf_counter` around the round trip,
rows fetched), so PostgreSQL's own planning of the executed statement is
included in every arm alike. The order of the three arms is shuffled in every
repetition, for the reason `benchmark._paired_medians` gives, and each arm's
median over the repetitions is kept.

Two production behaviours are deliberately left out, and both are stated rather
than hidden: shadow sampling (`SHADOW_SAMPLE_RATE`) is set to 0, because it
plans or executes extra work on a random fraction of requests and would add
noise unrelated to the question; and the runtime deadline of
`execute_chosen` is not applied, because it only ever cancels a deviation that
is already running long.

Usage:
    python -m app.e2e_latency --reps 5 --json-out models/e2e_latency.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time

from app.db import get_cursor
from app.measurement import make_deterministic
from app.optimizer import planner
from app.optimizer.decision_cache import DecisionCache
from app.optimizer.hints import apply_hint
from app.optimizer.learned import DEFAULT_SELECTION_POLICY, LearnedOptimizer
from app.optimizer.regression_guard import RegressionGuard
from app.workload import WORKLOAD

ARMS = ("native", "uncached", "cached")
DEFAULT_REPS = 5
DEFAULT_STATEMENT_TIMEOUT_MS = 30_000


def _execute_ms(cur, sql: str) -> float:
    """Wall clock for one statement, rows included."""
    started = time.perf_counter()
    cur.execute(sql)
    if cur.description is not None:
        cur.fetchall()
    return (time.perf_counter() - started) * 1000.0


def _time_arm(cur, arm: str, sql: str, query_id: str, optimizer, guard, cache):
    """
    One timed request: (total_ms, choose_ms, hint). Native has no choosing.

    The uncached arm passes `cache=None`, not an empty cache, so a decision made
    in this arm can never be served to the cached arm and vice versa.
    """
    if arm == "native":
        return _execute_ms(cur, sql), None, None
    started = time.perf_counter()
    choice = planner.plan_query(
        cur, sql, optimizer, query_id=query_id, guard=guard,
        cache=cache if arm == "cached" else None,
    )
    choose_ms = (time.perf_counter() - started) * 1000.0
    hint = choice.get("hint")
    run_ms = _execute_ms(cur, apply_hint(sql, hint) if hint else sql)
    return choose_ms + run_ms, choose_ms, hint


def measure_query(cur, item: dict, optimizer, guard, cache, reps: int,
                  rng: random.Random) -> dict:
    """Medians of each arm for one query, arms shuffled every repetition."""
    query_id, sql = item["id"], item["sql"]
    # Warm the cache outside the timed repetitions, so every cached sample is a
    # hit. A miss would just be the uncached arm again under another name.
    planner.plan_query(cur, sql, optimizer, query_id=query_id, guard=guard, cache=cache)

    totals = {arm: [] for arm in ARMS}
    choosing = {"uncached": [], "cached": []}
    hints = set()
    for _ in range(reps):
        order = list(ARMS)
        rng.shuffle(order)
        for arm in order:
            total_ms, choose_ms, hint = _time_arm(
                cur, arm, sql, query_id, optimizer, guard, cache
            )
            totals[arm].append(total_ms)
            if choose_ms is not None:
                choosing[arm].append(choose_ms)
                hints.add(hint)

    row = {"query_id": query_id}
    for arm in ARMS:
        row[f"{arm}_ms"] = statistics.median(totals[arm])
    for arm, values in choosing.items():
        row[f"{arm}_choose_ms"] = statistics.median(values)
    # Deviated means a hint was served on at least one request. Recorded so a
    # reader can see how much of the difference comes from queries where the
    # optimizer only added overhead and changed nothing.
    row["deviated"] = any(h for h in hints)
    return row


def summarise(rows: list[dict], confidence: float = 0.95) -> dict:
    """
    Totals, and a bootstrap interval for each arm's difference from native.

    The interval resamples *queries*, so it describes how far the total
    difference depends on which queries happen to be in the workload. It is not
    an interval over repeated runs; the per-query medians already absorb the
    run-to-run noise of the repetitions.
    """
    from app.experiment import bootstrap_ci

    n = len(rows)
    summary = {"n_queries": n, "n_deviated": sum(r["deviated"] for r in rows)}
    native_total = sum(r["native_ms"] for r in rows)
    summary["native_total_ms"] = native_total
    for arm in ("uncached", "cached"):
        total = sum(r[f"{arm}_ms"] for r in rows)
        diffs = [r[f"{arm}_ms"] - r["native_ms"] for r in rows]
        lo, hi = bootstrap_ci(diffs, confidence=confidence)
        summary[arm] = {
            "total_ms": total,
            "choose_total_ms": sum(r[f"{arm}_choose_ms"] for r in rows),
            "diff_vs_native_ms": total - native_total,
            "diff_vs_native_pct": (total - native_total) / native_total * 100
            if native_total else None,
            # The interval is for the mean per-query difference, scaled to the
            # total so it reads in the same units as the line above it.
            "diff_ci_ms": (lo * n, hi * n),
        }
    return summary


def measure(
    reps: int = DEFAULT_REPS,
    limit: int | None = None,
    seed: int = 0,
    use_guard: bool = True,
    policy: str = DEFAULT_SELECTION_POLICY,
    statement_timeout_ms: int = DEFAULT_STATEMENT_TIMEOUT_MS,
    quiet: bool = False,
) -> dict:
    say = (lambda *a, **k: None) if quiet else print
    # See the module docstring: shadow work is production behaviour, but random
    # extra work on a fraction of requests is not what this is measuring.
    planner.SHADOW_SAMPLE_RATE = 0.0

    rng = random.Random(seed)
    workload = WORKLOAD[:limit] if limit else WORKLOAD
    optimizer = LearnedOptimizer(policy=policy, seed=0)
    cache = DecisionCache()
    rows = []
    with get_cursor() as cur:
        cur.execute("SET statement_timeout = %s", (statement_timeout_ms,))
        make_deterministic(cur)
        guard = RegressionGuard()
        if use_guard:
            guard.refresh(cur)
        for item in workload:
            cur.execute("SAVEPOINT e2e")
            try:
                row = measure_query(cur, item, optimizer, guard, cache, reps, rng)
                cur.execute("RELEASE SAVEPOINT e2e")
            except Exception as exc:  # noqa: BLE001 - one failed query must not end the run
                cur.execute("ROLLBACK TO SAVEPOINT e2e")
                say(f"{item['id']}: skipped ({type(exc).__name__}: {exc})")
                continue
            rows.append(row)
            say(
                f"{row['query_id']:32s} native {row['native_ms']:8.1f}  "
                f"uncached {row['uncached_ms']:8.1f}  cached {row['cached_ms']:8.1f}"
                + ("  (deviates)" if row["deviated"] else "")
            )

    summary = summarise(rows)
    say(f"\n=== end to end, {summary['n_queries']} queries, {reps} reps each ===")
    say(f"native total     {summary['native_total_ms']:9.1f} ms")
    for arm in ("uncached", "cached"):
        s = summary[arm]
        lo, hi = s["diff_ci_ms"]
        say(
            f"{arm:8s} total   {s['total_ms']:9.1f} ms   "
            f"vs native {s['diff_vs_native_ms']:+8.1f} ms ({s['diff_vs_native_pct']:+.1f}%)   "
            f"95% CI [{lo:+.1f}, {hi:+.1f}]   choosing {s['choose_total_ms']:.1f} ms"
        )
    say(f"queries where a hint was served: {summary['n_deviated']}/{summary['n_queries']}")
    return {"reps": reps, "seed": seed, "use_guard": use_guard, "policy": policy,
            "summary": summary, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-guard", action="store_true")
    parser.add_argument("--timeout-ms", type=int, default=DEFAULT_STATEMENT_TIMEOUT_MS)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    result = measure(
        reps=args.reps, limit=args.limit, seed=args.seed,
        use_guard=not args.no_guard, statement_timeout_ms=args.timeout_ms,
    )
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
