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
from app.measurement import make_deterministic

# Three, not one. Training on single executions taught the model that
# whichever candidate got the luckiest timing was genuinely fastest, and
# live runs ranged from +40% to -149% as a result. Aggregating repeated
# executions to their median removed every regression with no change to the
# model (docs/WRITEUP.md 2.2) -- so collecting a single rep makes that fix
# impossible, and shipping 1 as the default contradicted the project's own
# strongest finding. `app.onboard` already used 3; these agree with it now.
DEFAULT_REPS = 3


def _recover(cur, statement_timeout_ms: int) -> None:
    """
    Get the connection back to a usable state after a candidate failed.

    Tries the cheap thing first: `ROLLBACK TO SAVEPOINT` keeps every row logged
    so far in this transaction, which is the whole reason the savepoint exists --
    a full rollback after an hour of collection throws the hour away.

    If that fails too -- and it can, since the same cancellation that killed the
    candidate can surface on the next statement -- fall back to rolling the
    transaction back and re-applying the session settings, because a connection
    that is quietly missing `statement_timeout` will hang on the next slow
    candidate instead of skipping it.
    """
    try:
        cur.execute("ROLLBACK TO SAVEPOINT candidate")
        return
    except Exception:  # noqa: BLE001 - fall through to the heavier reset
        pass

    try:
        cur.connection.rollback()
        cur.execute("SET statement_timeout = %s", (statement_timeout_ms,))
        cur.execute("LOAD 'pg_hint_plan'")
        cur.execute("SET pg_hint_plan.enable_hint = on")
        print("  [reset] transaction rolled back; rows from this batch are lost",
              file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"  [reset] could not recover the connection: {exc}", file=sys.stderr)


def collect(
    reps: int = DEFAULT_REPS,
    include_join_methods: bool | None = None,
    include_tuning: bool = True,
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
    # Default to whatever the serving path offers, rather than hardcoding True.
    #
    # `train.rows_for_served_action_space` already drops rows for families the
    # optimizer does not serve, and reads which those are from `planner.py` so
    # the two cannot drift. Collection was the one place still asserting an
    # answer of its own: `include_join_methods` defaulted to True while
    # `INCLUDE_JOIN_METHODS` is off, so a run spent most of its time measuring
    # the forced join-method family -- the slowest in the action space, mean
    # ratio 5.49 against 1.97 for everything else -- and training then discarded
    # every row of it.
    #
    # Passing the flag explicitly still works, which is what a run meant to
    # answer "would join methods earn their place?" should do.
    if include_join_methods is None:
        from app.optimizer import planner

        include_join_methods = planner.INCLUDE_JOIN_METHODS

    workload = workload if workload is not None else WORKLOAD
    start = time.time()
    total_rows = 0
    skipped_disabled = 0

    with get_cursor() as cur:
        cur.execute("SET statement_timeout = %s", (statement_timeout_ms,))
        # Pin what would otherwise differ between runs -- see
        # `app/measurement.py::DETERMINISTIC_SESSION`.
        make_deterministic(cur)
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
            # The baseline is measured `reps` times, like every candidate.
            #
            # It used to be measured exactly once, before any candidate ran, and
            # that biased every label in the same direction. Two ways:
            #
            #   * it was a single draw carrying the full run-to-run noise, while
            #     each candidate got the median of `reps` -- and a median
            #     discards the unlucky draws a lone sample keeps;
            #   * it ran first, on the coldest cache, and every candidate after
            #     it benefited from the pages it had just pulled in. This
            #     project has been caught by cache-state bias before: a reported
            #     80.5% improvement was really 26.0%.
            #
            # The baseline is the *denominator* of every ratio the model learns,
            # so a denominator measured under worse conditions than its
            # numerators pushes every training label below 1.0. Measured
            # directly on this database, first-sample against median-of-three
            # across eight queries: **1.8%**. Small, real, free to remove, and
            # in the one direction that flatters the system.
            baseline_runs = []
            failed = False
            for _ in range(reps):
                cur.execute("SAVEPOINT candidate")
                try:
                    baseline_runs.append(get_plan(cur, sql))
                    cur.execute("RELEASE SAVEPOINT candidate")
                except Exception as exc:  # noqa: BLE001 - one bad query shouldn't kill the run
                    print(f"  [skip] {query_id} (baseline): {exc}", file=sys.stderr)
                    _recover(cur, statement_timeout_ms)
                    failed = True
                    break
            if failed or not baseline_runs:
                continue

            # Every repetition is logged, exactly as candidate repetitions are:
            # `train._aggregate_repetitions` takes the median of whatever rows a
            # (query, hint) pair has, so writing them all is what puts the
            # baseline on the same footing as everything else. `baseline` itself
            # is only used below for the table set and join graph, which do not
            # vary between repetitions.
            baseline = baseline_runs[0]
            for run in baseline_runs:
                log_execution(
                    cur,
                    query_id=query_id,
                    sql_text=sql,
                    plan=run,
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
                include_tuning=include_tuning,
                join_graph=baseline.get("join_graph"),
            )

            for hint in hints:
                hinted_sql = apply_hint(sql, hint)
                for _ in range(reps):
                    cur.execute("SAVEPOINT candidate")
                    # The recovery statements are inside the guard too, because
                    # they can fail for the same reason the candidate did.
                    #
                    # A cancelled statement does not always surface its error on
                    # the `execute` that issued it -- psycopg2 can raise it on
                    # the next one instead. `RELEASE SAVEPOINT` was outside this
                    # block, so a timeout arriving there killed the whole run
                    # after four minutes of collection:
                    #
                    #     File "app/collect_data.py", line 126, in collect
                    #       cur.execute("RELEASE SAVEPOINT candidate")
                    #     psycopg2.errors.QueryCanceled: canceling statement due
                    #     to statement timeout
                    #
                    # Losing an hour of collection to the bookkeeping around a
                    # skipped candidate is exactly what the savepoint was added
                    # to prevent.
                    try:
                        plan = get_plan(cur, hinted_sql)
                        cur.execute("RELEASE SAVEPOINT candidate")
                    except Exception as exc:  # noqa: BLE001 - a bad/slow hint shouldn't kill the run
                        print(f"  [skip] {query_id} {hint!r}: {exc}", file=sys.stderr)
                        _recover(cur, statement_timeout_ms)
                        continue
                    # A plan carrying disabled nodes is one the planner was
                    # forbidden to build and built anyway, under a constraint it
                    # could not satisfy. `planner.plan_query` drops those from
                    # the action space, so collecting them is the train/serve
                    # skew of WRITEUP 2.18 all over again -- training the model
                    # on candidates it will never be asked to score.
                    #
                    # It matters more since PostgreSQL 18. Before 18 such a plan
                    # carried `disable_cost` (~1e10) and was obvious in the
                    # cost; PG18 removed the inflation, and a disconnected
                    # four-table order now costs 20.5 against the correct
                    # plan's 6990.8. Logged as training data, that row says a
                    # structurally broken plan is 340x cheap.
                    if plan.get("disabled_nodes"):
                        skipped_disabled += 1
                        continue

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
    if skipped_disabled:
        # Said out loud, because silently collecting a subset is the same class
        # of mistake as silently serving one -- WRITEUP 2.18.
        print(
            f"Skipped {skipped_disabled} candidate(s) whose plan carried disabled "
            f"nodes: the planner could not honour the hint, so the plan is not "
            f"what the candidate claims and serving would never offer it."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reps", type=int, default=DEFAULT_REPS,
        help="executions per candidate; the median becomes the training label",
    )
    parser.add_argument(
        "--no-join-methods",
        action="store_true",
        help=(
            "skip the forced join-method family. Without this flag the run "
            "follows INCLUDE_JOIN_METHODS, which is what serving offers"
        ),
    )
    parser.add_argument(
        "--no-tuning",
        action="store_true",
        help="skip the resource/cost-model candidates (parallelism, work_mem, ...)",
    )
    parser.add_argument("--timeout-ms", type=int, default=30_000, help="per-candidate statement_timeout")
    args = parser.parse_args()
    collect(
        reps=args.reps,
        # `False` when the flag is given, otherwise `None` -- which lets
        # `collect()` read `planner.INCLUDE_JOIN_METHODS`, the flag serving
        # actually uses.
        #
        # This used to pass `not args.no_join_methods`, so the CLI hardcoded
        # `True` and defeated the very default that was added to stop collection
        # and serving disagreeing. The function was fixed and the caller was
        # not, which is the same mistake one layer up: a constant restating a
        # decision owned somewhere else (WRITEUP 2.28 lists the other four).
        #
        # Caught by watching a live run: it was executing
        # `HashJoin(o u) HashJoin(o u oi) ...` candidates, the slowest family in
        # the action space, every row of which `rows_for_served_action_space`
        # discards at training time.
        include_join_methods=False if args.no_join_methods else None,
        include_tuning=not args.no_tuning,
        statement_timeout_ms=args.timeout_ms,
    )
