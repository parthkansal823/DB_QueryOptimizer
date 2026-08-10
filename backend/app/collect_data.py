"""
Phase 1: run the full workload through native Postgres and every generated
candidate (join-order only, plus join-order x join-method for the stretch
goal), and log every execution to `plan_execution_log`. This is what turns
`workload.py` into real training data for Phase 2/3.

Usage (from backend/, with the stack running via docker compose):
    python -m app.collect_data [--reps N] [--no-join-methods] [--timeout-ms N]
"""

from __future__ import annotations

import argparse
import sys
import time

from app.db import get_cursor
from app.logging_store import log_execution
from app.optimizer.hints import apply_hint, generate_candidates
from app.plan_extractor import get_plan
from app.workload import WORKLOAD

# Three, not one. Training on single executions taught the model that
# whichever candidate got the luckiest timing was genuinely fastest, and
# live runs ranged from +40% to -149% as a result. Aggregating repeated
# executions to their median removed every regression with no change to the
# model (docs/WRITEUP.md 2.2) -- so collecting a single rep makes that fix
# impossible, and shipping 1 as the default contradicted the project's own
# strongest finding. `app.onboard` already used 3; these agree with it now.
DEFAULT_REPS = 3


def collect(
    reps: int = DEFAULT_REPS,
    include_join_methods: bool = True,
    workload: list[dict] | None = None,
    statement_timeout_ms: int = 30_000,
    on_progress=None,
    should_stop=None,
) -> None:
    """
    `workload` defaults to the synthetic-schema `WORKLOAD`; pass a different
    list (e.g. real JOB queries, see `app.collect_data_job`) to collect
    against any other schema `DATABASE_URL` points at -- nothing else here
    is schema-specific. `statement_timeout_ms` caps how long one candidate
    can run: a forced join order/method on an un-indexed large table (JOB's
    tables run into the tens of millions of rows) can occasionally pick a
    pathological plan, and one hung candidate shouldn't stall the whole run.

    `on_progress(done, total, query_id, rows)` and `should_stop()` exist for
    callers that are not a terminal -- the dashboard runs this in a background
    thread, where printing to stdout tells the user nothing and there is no
    Ctrl-C to interrupt with. Both are optional and default to the previous
    behaviour exactly.
    """
    workload = workload if workload is not None else WORKLOAD
    start = time.time()
    total_rows = 0

    with get_cursor() as cur:
        cur.execute("SET statement_timeout = %s", (statement_timeout_ms,))
        for qi, item in enumerate(workload):
            # Checked between queries rather than mid-query: the per-query
            # commit below is what makes stopping here leave a consistent,
            # durable partial run rather than a half-collected one.
            if should_stop is not None and should_stop():
                print(f"Stopped after {qi} of {len(workload)} queries.", flush=True)
                break

            sql = item["sql"]
            query_id = item["id"]

            # SAVEPOINTs isolate one candidate's failure/timeout from the rest
            # of this (long-running, many-INSERT) transaction -- without one,
            # a single cancelled statement aborts the whole transaction and
            # every row logged so far in this run would be lost on rollback.
            cur.execute("SAVEPOINT candidate")
            try:
                baseline = get_plan(cur, sql)
            except Exception as exc:  # noqa: BLE001 - one bad query shouldn't kill the run
                cur.execute("ROLLBACK TO SAVEPOINT candidate")
                print(f"  [skip] {query_id} (baseline): {exc}", file=sys.stderr)
                continue
            cur.execute("RELEASE SAVEPOINT candidate")

            log_execution(
                cur,
                query_id=query_id,
                sql_text=sql,
                plan=baseline,
                hint=None,
                is_baseline=True,
                selector_used="native",
            )
            total_rows += 1

            tables = baseline["tables_scanned"]
            # The join graph must match what serving uses, or training data is
            # gathered over a different action space than the one the model is
            # later asked to score -- the train/serve skew README.md warns
            # about. Without it, roughly half the collected candidates are
            # cartesian products that no served query will ever consider, so
            # the capacity spent learning them is wasted.
            hints = generate_candidates(
                tables,
                include_join_methods=include_join_methods,
                join_graph=baseline.get("join_graph"),
            )

            for hint in hints:
                hinted_sql = apply_hint(sql, hint)
                for _ in range(reps):
                    cur.execute("SAVEPOINT candidate")
                    try:
                        plan = get_plan(cur, hinted_sql)
                    except Exception as exc:  # noqa: BLE001 - a bad/slow hint shouldn't kill the run
                        cur.execute("ROLLBACK TO SAVEPOINT candidate")
                        print(f"  [skip] {query_id} {hint!r}: {exc}", file=sys.stderr)
                        continue
                    cur.execute("RELEASE SAVEPOINT candidate")
                    log_execution(
                        cur,
                        query_id=query_id,
                        sql_text=sql,
                        plan=plan,
                        hint=hint,
                        is_baseline=False,
                        selector_used="collection",
                    )
                    total_rows += 1

            # Commit after every query, not once at the end.
            #
            # A full sweep here is 20+ minutes (24 queries x ~30 candidates x
            # reps, including slow 6-way joins). Holding it all in one
            # transaction meant any interruption -- a container restart, a
            # cancelled shell -- rolled back the entire run and left the log
            # empty. That happened twice, losing ~90% of a completed sweep
            # both times. Per-query commits make progress durable, and reruns
            # append rather than starting over.
            cur.connection.commit()

            elapsed = time.time() - start
            print(
                f"[{qi + 1}/{len(workload)}] {query_id}: "
                f"{len(hints)} candidates x {reps} rep(s), {total_rows} rows so far "
                f"({elapsed:.1f}s elapsed)",
                flush=True,  # so progress is visible when piped
            )
            if on_progress is not None:
                on_progress(qi + 1, len(workload), query_id, total_rows)

    print(f"Done. Logged {total_rows} rows in {time.time() - start:.1f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reps", type=int, default=DEFAULT_REPS,
        help="executions per candidate; the median becomes the training label",
    )
    parser.add_argument(
        "--no-join-methods",
        action="store_true",
        help="collect join-order candidates only (skip the join-method stretch goal)",
    )
    parser.add_argument("--timeout-ms", type=int, default=30_000, help="per-candidate statement_timeout")
    args = parser.parse_args()
    collect(
        reps=args.reps,
        include_join_methods=not args.no_join_methods,
        statement_timeout_ms=args.timeout_ms,
    )
