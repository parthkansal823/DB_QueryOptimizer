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
from app.optimizer import planner
from app.optimizer.hints import apply_hint, generate_candidates
from app.optimizer.planner import budget_for
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


def _median_latency(runs: list[dict]) -> float | None:
    """Median measured latency across a query's baseline repetitions.

    Median rather than the first run for the reason `_aggregate_repetitions`
    gives: latency has a heavy right tail, and one execution colliding with
    autovacuum should not decide how wide the search gets.
    """
    times = [r.get("actual_total_time_ms") for r in runs]
    times = sorted(t for t in times if t is not None)
    if not times:
        return None
    return times[len(times) // 2]


def select_workload(
    workload: list[dict],
    only: str | None = None,
    held_out: bool = False,
    bundle_path: str = "models/plan_selector.pkl",
) -> list[dict]:
    """
    Narrow the workload to named queries, or to the ones the sweep can score.

    ## Why `--held-out` exists

    A collection run costs one execution per candidate per repetition and is
    routinely stopped partway. `WORKLOAD` is iterated in list order, so a run
    that dies early always covers the same alphabetical prefix -- and after the
    run that finally turned on the five unscored families, `memoize`,
    `parallel`, `join_method` and `disable_index` had rows on 16 of 65 queries,
    every one of them a `corr_*`.

    That is not a thin sample, it is the wrong 16. `calibrate.sweep` scores on
    **held-out queries only** (`_held_out_queries`, and 2.2.1 for why), and
    just **4** of those 16 were held out. Five families' worth of collection
    bought four scoreable queries, and the deltas it produced -- -15, -7, 0,
    0 and +2 ms -- sit well inside 2.18's 18.6% noise floor.

    So the rows that can change an answer are the held-out ones, and there is
    no reason to spend the budget in list order. This makes the collection
    aim at them.

    An unknown id raises rather than collecting nothing: a run that reports
    "0 queries" and exits 0 is this project's whole bug class in miniature.
    """
    if held_out:
        import pickle

        with open(bundle_path, "rb") as fh:
            wanted = set(pickle.load(fh).get("held_out_queries") or ())
        if not wanted:
            raise SystemExit(
                f"{bundle_path} records no held-out queries, so there is "
                "nothing to aim at. Train a model first."
            )
    elif only:
        wanted = {q.strip() for q in only.split(",") if q.strip()}
    else:
        return workload

    known = {item["id"] for item in workload}
    missing = sorted(wanted - known)
    if missing and not held_out:
        raise SystemExit(f"not in this workload: {', '.join(missing)}")
    selected = [item for item in workload if item["id"] in wanted]
    if not selected:
        raise SystemExit("nothing selected; refusing to run an empty collection")
    return selected


def families_at_risk(workload: list[dict]) -> dict[str, set[str]]:
    """
    Families the latest logged run has for these queries that this run will not.

    `calibrate._grouped_rows` keeps only the most recent `created_at` per
    query, for the reason its docstring gives: replaying all of history
    compares a candidate measured in one run against a native measured in
    another, and on this log that reported every gate setting losing 15-25%
    while a live benchmark of the same gate captured 40.8%.

    That rule is per **query**. Families live inside a query's rows, so a
    re-collection narrower than the last one does not add to the record, it
    replaces it, and every family the new run does not generate stops counting.
    2.93: a `--held-out` run without `INCLUDE_SCAN_METHODS` took `scan_method`
    from 65 queries to zero held-out ones. Nothing was deleted -- 43,327 rows
    were still there -- and `available_families()` would have reported the
    family as *not scored*, which reads as "never collected".

    Returns `{query_id: {family, ...}}` for the queries that would lose
    something. Empty when this run is a superset of the last, which is the
    normal case.
    """
    from app import calibrate

    wanted = {item["id"] for item in workload}
    generated = _families_this_run()
    at_risk: dict[str, set[str]] = {}
    for query_id, rows in calibrate._grouped_rows().items():
        if query_id not in wanted:
            continue
        logged = {
            calibrate.action_family(r.get("hint"))
            for r in rows if not r.get("is_baseline")
        }
        logged.discard(None)
        missing = logged - generated
        if missing:
            at_risk[query_id] = missing
    return at_risk


def _families_this_run() -> set[str]:
    """Which action families this process's flags will actually generate."""
    # `collect()` defaults `include_tuning` to True regardless of the env flag,
    # so `resource` is generated unless --no-tuning is passed. The three below
    # it are autotune's ALWAYS_ON set.
    families = {"join_order", "operator_toggle", "cardinality", "resource"}
    for flag, family in (
        ("INCLUDE_JOIN_METHODS", "join_method"),
        ("INCLUDE_SCAN_METHODS", "scan_method"),
        ("INCLUDE_MEMOIZE", "memoize"),
        ("INCLUDE_PARALLEL", "parallel"),
        ("INCLUDE_BUSHY", "bushy"),
        ("INCLUDE_DISABLE_INDEX", "disable_index"),
    ):
        if getattr(planner, flag, False):
            families.add(family)
    return families


def warn_if_narrower(workload: list[dict]) -> None:
    """Print what this run is about to stop counting. Never refuses.

    A deliberately narrow collection is a legitimate thing to want. What is not
    legitimate is it happening silently, which is what 2.93 was.
    """
    try:
        at_risk = families_at_risk(workload)
    except Exception as exc:  # noqa: BLE001 - a warning must not stop a run
        print(f"  [warn] could not check family coverage: {exc}", file=sys.stderr)
        return
    if not at_risk:
        return

    losing: dict[str, int] = {}
    for families in at_risk.values():
        for family in families:
            losing[family] = losing.get(family, 0) + 1

    print("", file=sys.stderr)
    print("  [warn] this run generates a NARROWER family set than the latest "
          "run already logged", file=sys.stderr)
    print("         for these queries. Only the latest run per query counts "
          "(calibrate._grouped_rows),", file=sys.stderr)
    print("         so the families below will stop being scored:",
          file=sys.stderr)
    for family, n in sorted(losing.items(), key=lambda kv: -kv[1]):
        print(f"           {family:<16} loses {n} of {len(workload)} queries",
              file=sys.stderr)
    print("         Set the matching INCLUDE_* flags if that is not what you "
          "want. See WRITEUP 2.93.", file=sys.stderr)
    print("", file=sys.stderr)


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
        # Imported at module scope. It used to be imported here, inside the
        # `if`, which makes `planner` a *local* name for the whole function --
        # so every other `planner.X` below would raise UnboundLocalError on any
        # call that passed `include_join_methods` explicitly, which is what
        # `app.collect_data_job` does.
        include_join_methods = planner.INCLUDE_JOIN_METHODS

    workload = workload if workload is not None else WORKLOAD
    warn_if_narrower(workload)
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
            #
            # The same applies to the *width* of the search and to which
            # families are on, and only the join-graph half of that was fixed.
            # This call took `generate_candidates`' defaults for everything
            # else: `max_order_candidates=8`, and every opt-in family off.
            # Serving takes them from `planner`, and since `budget_for` learned
            # to scale with join width (2.56) it will plan up to 96 order
            # candidates on a wide expensive query.
            #
            # So collection was training the model on 8 join orders and serving
            # was asking it to score up to 96, plus any scan-method or memoize
            # candidate the flags enabled -- a strictly *narrower* training
            # action space than the served one, which is the direction that
            # hurts. README's warning about widening the action space without
            # re-collecting is this bug, arrived at from the other side.
            #
            # Collection may legitimately be *wider* than serving -- it gathers
            # the forced join-method family that `budget_for` often truncates
            # away, and `calibrate._servable` narrows the replay back down. What
            # it must never be is narrower.
            hints = generate_candidates(
                tables,
                max_order_candidates=budget_for(
                    baseline.get("total_cost"),
                    len(tables),
                    # Collection has just *executed* the baseline, so it knows
                    # what serving has to look up from the guard's history.
                    # Passing it keeps the two sizing the same query the same
                    # way: `budget_for` declines to widen anything measured
                    # under half a second, and a query collected at the wide
                    # budget but served at the narrow one is the same
                    # train/serve gap in reverse.
                    observed_native_ms=_median_latency(baseline_runs),
                ),
                include_join_methods=include_join_methods,
                include_tuning=include_tuning,
                join_graph=baseline.get("join_graph"),
                include_bushy=planner.INCLUDE_BUSHY,
                include_scan_methods=planner.INCLUDE_SCAN_METHODS,
                include_negative_scans=planner.INCLUDE_NEGATIVE_SCANS,
                include_negative_joins=planner.INCLUDE_NEGATIVE_JOINS,
                include_memoize=planner.INCLUDE_MEMOIZE,
                include_parallel=planner.INCLUDE_PARALLEL,
                include_enable_settings=planner.INCLUDE_ENABLE_SETTINGS,
                include_extended_toggles=planner.INCLUDE_EXTENDED_TOGGLES,
                include_cost_model=planner.INCLUDE_COST_MODEL,
                # Needs the catalog, exactly as serving does. Reusing the
                # planner's own helper keeps the two from drifting: it returns
                # None when the family is off or the cursor is missing, and the
                # family then generates nothing.
                include_disable_index=planner.INCLUDE_DISABLE_INDEX,
                # Forwarded explicitly. `include_index_pinning` defaults to True
                # -- the catalog map used to be the flag -- so passing the map
                # without the flag collects a family serving does not offer.
                # `tests/test_family_guard.py` caught exactly that.
                include_index_pinning=planner.INCLUDE_INDEX_PINNING,
                table_indexes=planner._index_candidates_for(cur, baseline),
            )

            # Collection deliberately does **not** run `prefilter` here.
            #
            # It looked like an obvious bug: `planner.candidate_hints` drops
            # candidates the baseline plan proves cannot change anything, this
            # loop calls `generate_candidates` directly, so collection executes
            # and logs plans byte-identical to native -- 188 of 1,304 servable
            # candidates on the synthetic log, 14.4%, and serving never offers
            # one of them.
            #
            # Measured, dropping them from training makes the model's typical
            # error **worse on 15 of 16 paired splits** (p = 0.001). A row
            # saying "this hint changed nothing and the result was 1.0x" is
            # real evidence about the level the gate reads, and 2.70 found the
            # same thing about the baseline pairs the ranker trains on.
            #
            # So this is 2.9's harmless direction of a train/serve difference:
            # the model sees candidates it will never be *asked* about, which
            # costs collection time and buys calibration. What is not harmless
            # is letting those rows be the **oracle** -- see
            # `train._evaluate_selection`.
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
    parser.add_argument(
        "--only",
        help="comma-separated query ids to collect, instead of the whole workload",
    )
    parser.add_argument(
        "--held-out",
        action="store_true",
        help=(
            "collect only the queries the trained model held out -- the ones "
            "app.calibrate.sweep actually scores on. See select_workload()"
        ),
    )
    args = parser.parse_args()
    collect(
        workload=select_workload(WORKLOAD, only=args.only, held_out=args.held_out),
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
