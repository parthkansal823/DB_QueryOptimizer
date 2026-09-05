import asyncio
import logging
import os
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app import (
    db, db_profiles, diagnostics, experiment_job, metrics, model_store,
    retrain, saved_queries, stats, training_job, workload,
)
from app import settings as app_settings
from app import dbadvisor, hypothetical
from app.measurement import blocks_verdict, fairness
from app.advisor import analyze_plan, missing_fk_indexes
from app.index_advice import (
    already_uniquely_indexed,
    covering_opportunities,
    database_is_c_locale,
)
from app.db import get_cursor
from app import logging_store
from app.logging_store import log_execution, query_fingerprint
from app.optimizer.decision_cache import DecisionCache
from app.optimizer.hints import apply_hint, plan_fingerprint, split_hints
from app.optimizer.learned import LearnedOptimizer
from app.optimizer import planner
from app.optimizer.planner import candidate_hints, optimize_and_execute
from app.optimizer.regression_guard import SHADOW_SELECTOR, RegressionGuard
from app.optimizer.regret import regret_curve
from app.plan_extractor import get_plan
from app.schema_graph import discover_with_inference
from app.schema_graph import summarize as schema_summary

log = logging.getLogger("lqo.autolearn")

# -- runtime configuration --------------------------------------------------
#
# Everything the request path touches -- the optimizer, the regression guard,
# the decision cache, the Rows() action space -- is built from `app.settings`
# rather than read out of the environment here.
#
# `app.settings` persists every dashboard change to models/settings.json
# precisely so it survives a restart, but nothing outside `PUT /settings` ever
# read it back. The panel therefore reported one configuration while the
# process ran another, and not harmlessly: a user who *raised* the safety bar
# to 0.4 was served the calibrated gate's 0.0 instead, so the optimizer
# deviated on any predicted win at all -- the opposite of what the dashboard
# said was in force. Two of the settings were unreachable from anywhere at all:
# the guard read neither GUARD_TOLERANCE nor GUARD_MIN_OBSERVATIONS, which
# `app.settings` has always declared.
#
# The guard is mutated in place rather than rebuilt because it carries measured
# history (`blocked`, `observations`) that a settings change does not
# invalidate. The optimizer and the cache are rebuilt because everything they
# hold derives from the values that just changed.

SELECTOR_MODE = "heuristic"
guard = RegressionGuard()
decision_cache: DecisionCache | None = None


def _optimizer_from(values: dict) -> LearnedOptimizer:
    """
    Build an optimizer from settings, forwarding the gate thresholds *only*
    where someone actually chose them.

    `LearnedOptimizer` lets a measured gate (`models/gate.json`, written by
    `app.calibrate`) beat its own defaults but never beat an explicitly passed
    argument. Forwarding every current value would hand it the declared default
    1.0 as though it had been chosen, silently switching the calibration off
    for anyone who has never opened the settings panel -- so only genuine
    overrides are forwarded. See `app.settings.overrides`.
    """
    chosen = app_settings.overrides()
    # `gate_mode` joined this list when `app.calibrate` learned to sweep the
    # bound's *form* alongside its thresholds (2.42) and to write the winner
    # into models/gate.json.
    #
    # Until then the comment here read "which bound to apply is a plain choice
    # with no calibrated-file default to defer to, unlike the thresholds", and
    # it passed `values["gate_mode"]` unconditionally. That was correct while
    # nothing measured the bound. The moment something did, passing it every
    # time meant the constructor argument always won and the swept value in the
    # gate file was never read -- a calibration the server ignores, which is
    # the same shape as 2.43's policy and 2.42's unused conformal gate.
    #
    # Thresholds and the bound they were swept under have to travel together
    # anyway: a `min_relative_gain` chosen against a conformal bound is not the
    # right threshold for a fixed-z one.
    gate = {
        name: values[name]
        for name in ("confidence_z", "min_relative_gain", "gate_mode")
        if name in chosen
    }
    return LearnedOptimizer(
        policy=values["selection_policy"],
        risk_lambda=values["risk_lambda"],
        max_regression_ratio=values["max_regression_ratio"],
        **gate,
    )


def _apply_settings(values: dict) -> None:
    """
    Rebuild the parts of the running system a settings change affects.

    Constructing a fresh optimizer rather than mutating the live one: its
    thresholds are read in the constructor (including the calibrated gate), so
    assigning to attributes would apply some of a change and not the rest.
    """
    global optimizer, SELECTOR_MODE, decision_cache

    optimizer = _optimizer_from(values)
    SELECTOR_MODE = "learned" if optimizer.model is not None else "heuristic"

    guard.tolerance = values["guard_tolerance"]
    guard.min_observations = values["guard_min_observations"]

    planner.ENABLE_ROWS_CORRECTION = values["enable_rows_correction"]
    planner.INCLUDE_JOIN_METHODS = values["include_join_methods"]
    planner.INCLUDE_TUNING = values["include_tuning"]
    planner.INCLUDE_BUSHY = values["include_bushy"]
    planner.INCLUDE_SCAN_METHODS = values["include_scan_methods"]
    planner.INCLUDE_NEGATIVE_SCANS = values["include_negative_scans"]
    planner.INCLUDE_NEGATIVE_JOINS = values["include_negative_joins"]
    planner.INCLUDE_MEMOIZE = values["include_memoize"]
    planner.INCLUDE_PARALLEL = values["include_parallel"]
    planner.INCLUDE_ENABLE_SETTINGS = values["include_enable_settings"]
    planner.INCLUDE_INDEX_PINNING = values["include_index_pinning"]
    planner.INCLUDE_EXTENDED_TOGGLES = values["include_extended_toggles"]
    planner.INCLUDE_COST_MODEL = values["include_cost_model"]
    planner.ADAPTIVE_CANDIDATE_BUDGET = values["adaptive_candidate_budget"]
    planner.RUNTIME_TIMEOUT_FACTOR = values["runtime_timeout_factor"]
    planner.SHADOW_SAMPLE_RATE = values["shadow_sample_rate"]

    # Missing from this list until now, which made it a setting that was
    # declared, documented at length, shown in the dashboard and inert.
    # `logging_store` reads `GROUP_BY_TEMPLATE` from the environment once at
    # import; the settings panel writes to models/settings.json, which that
    # module never consults. Toggling "Treat parameters as the same query" did
    # nothing at all.
    #
    # `test_settings_are_applied` now asserts every declared field is either
    # applied here, consumed when the optimizer is constructed, or explicitly
    # listed as train-time-only -- so the next one cannot be silent.
    logging_store.GROUP_BY_TEMPLATE = values["group_by_template"]

    # The cache is rebuilt rather than resized: every entry in it was decided
    # by the policy and thresholds that just changed, so keeping any of them
    # would serve decisions the new settings would not have made. A TTL of 0
    # disables it, which is what a benchmark wanting every decision made from
    # scratch should use.
    ttl = values["decision_cache_seconds"]
    decision_cache = DecisionCache(ttl_seconds=ttl) if ttl > 0 else None


def _reload_optimizer() -> None:
    """
    Swap in a newly promoted (or rolled back) model, keeping the configuration.

    Rebuilt from `app.settings` rather than from the outgoing optimizer's
    attributes. Copying `policy` and `risk_lambda` off the old object silently
    dropped the safety gate with it, so every retrain-promotion, rollback and
    /model/reload reset `confidence_z` and `min_relative_gain` to their
    defaults. Lowering the safety bar is not something a model swap should be
    able to do as a side effect.
    """
    global optimizer, SELECTOR_MODE
    optimizer = _optimizer_from(app_settings.current())
    SELECTOR_MODE = "learned" if optimizer.model is not None else "heuristic"
    # Every cached decision was made by the model being replaced. Keeping them
    # would let a freshly promoted model serve its predecessor's choices and
    # appear to have changed nothing -- the retrain would look like a no-op.
    if decision_cache is not None:
        decision_cache.clear()


# Applied at import, so a restart serves what the dashboard says it serves.
_apply_settings(app_settings.current())

# The system retrains itself on accumulated feedback on this cadence. Off by
# default: a background job that silently swaps the served model is exactly
# the sort of thing that should be opted into, and it makes benchmark runs
# non-reproducible while it's on. `retrain_if_needed` still gates every
# promotion on the champion/challenger comparison, so the worst case is a
# wasted retrain, not a regression.
AUTO_RETRAIN_SECONDS = int(os.getenv("AUTO_RETRAIN_SECONDS", "0"))


class QueryRequest(BaseModel):
    sql: str


def _drifted() -> bool:
    """
    Has the deployed gate stopped delivering what its calibration promised?

    `app.calibrate` records `expected_regression_rate` in `models/gate.json` --
    a prediction about future behaviour, measured on a held-out sweep at the
    moment the gate was chosen. `app.diagnostics.drift_report` checks it against
    what has since happened.

    Used here to make retraining *evidence*-driven. The interval below is a
    clock: on a stable workload it retrains for nothing, and on one that shifted
    an hour ago it waits. A gate whose regressions have climbed past what it
    promised is a reason to retrain now; the absence of that is a reason not to.
    """
    try:
        with get_cursor() as cur:
            report = diagnostics.diagnose(cur=cur)
    except Exception:  # noqa: BLE001 - an empty log is not drift
        return False
    return any(w["code"] == "gate_promise_not_met" for w in report.get("warnings", []))


async def _auto_retrain_loop() -> None:
    """
    Fold accumulated feedback back into the served model.

    Wakes on the interval but only retrains when there is a reason to: enough
    new data (`retrain_if_needed`'s own gate) or measured drift. Promotion is
    still decided by the champion/challenger comparison either way, so the worst
    case remains a wasted retrain rather than a regression.
    """
    while True:
        await asyncio.sleep(AUTO_RETRAIN_SECONDS)
        try:
            drifted = await asyncio.to_thread(_drifted)
            if drifted:
                log.info("auto-retrain: gate is missing its promised regression rate")
            # Training is CPU-bound and synchronous; keep it off the event
            # loop so query serving isn't blocked while a retrain runs.
            result = await asyncio.to_thread(retrain.retrain_if_needed, force=drifted)
            if result.get("action") == "promoted":
                _reload_optimizer()
                with get_cursor() as cur:
                    guard.refresh(cur)
            log.info("auto-retrain: %s", result.get("action"))
        except Exception:  # noqa: BLE001 - a failed retrain must never kill the server
            log.exception("auto-retrain failed; continuing to serve the current model")


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Prime the guard so the first served query already benefits from history.
    try:
        with get_cursor() as cur:
            guard.refresh(cur)
    except Exception:  # noqa: BLE001 - an empty/absent log table is fine at boot
        log.warning("could not prime the regression guard at startup")

    task = None
    if AUTO_RETRAIN_SECONDS > 0:
        # The reference matters: asyncio only holds a *weak* one, so a bare
        # `create_task(...)` whose result nobody keeps can be garbage
        # collected mid-await. The retrain loop would then stop silently --
        # the server keeps serving, so nothing looks wrong, and the model
        # just quietly stops being updated.
        task = asyncio.create_task(_auto_retrain_loop())
        log.info("auto-retrain enabled every %ss", AUTO_RETRAIN_SECONDS)

    yield

    # Shut the loop down rather than letting it be killed mid-retrain, so a
    # promotion is never left half-applied on restart.
    if task is not None:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


# Built with the lifespan handler rather than `@app.on_event("startup")`,
# which FastAPI deprecated -- it still ran, but emitted a DeprecationWarning
# on every import and is slated for removal. The lifespan form also gives the
# shutdown half, which on_event's startup hook had no equivalent for.
app = FastAPI(title="Learned Query Optimizer", lifespan=_lifespan)

# Dev origin only -- the Vite dashboard from Phase 5. FRONTEND_ORIGIN lets
# docker-compose override this for other environments without editing code.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


# How many candidates `/query/analyze` executes for the measured comparison.
#
# It used to execute every one. Measured on a four-table query: 19 candidates,
# 48 ms each, 915 ms of a 1.2 s request -- while *planning* the same candidate
# costs 0.89 ms. The model never needed the executions to choose; it predicts
# latency from the plan tree. Only the side-by-side comparison the dashboard
# renders needs real numbers, and four rows make that comparison as well as
# nineteen do.
ANALYZE_EXECUTE_LIMIT = int(os.getenv("ANALYZE_EXECUTE_LIMIT", "4"))


def _execution_order(
    candidate_plans: list[dict], chosen_index: int | None, limit: int
) -> list[int]:
    """
    Which candidates are worth spending an execution on, best first.

    The chosen one always, because a decision the dashboard reports without a
    measured latency beside it is a claim with nothing behind it. Then the
    cheapest by the planner's own estimate, which is the only ordering
    available before anything has been run -- and the one most likely to
    contain the plan that actually wins.
    """
    order: list[int] = []
    if chosen_index is not None and 0 <= chosen_index < len(candidate_plans):
        order.append(chosen_index)

    by_cost = sorted(
        range(len(candidate_plans)),
        key=lambda i: candidate_plans[i].get("total_cost") or float("inf"),
    )
    for i in by_cost:
        if len(order) >= max(limit, 1):
            break
        if i not in order:
            order.append(i)
    return order


def _sql_error_response(exc: Exception, sql: str) -> HTTPException:
    """
    A failed query, explained -- and corrected where that is safe.

    A statement with a typo used to surface as a 500 and a blank panel, while
    the exception already carried the offending token, an exact character
    position, and frequently PostgreSQL's own correction. 400 rather than 500:
    the request was understood, the SQL in it was not valid.

    The catalogue lookup is a second connection and only happens on the failure
    path, where one extra round trip is irrelevant and a better answer is worth
    a lot. It is also allowed to fail -- blunter advice beats an error handler
    that raises.
    """
    from app.sql_errors import explain, known_names

    names: list[str] = []
    try:
        with get_cursor() as name_cur:
            names = known_names(name_cur)
    except Exception:  # noqa: BLE001 - the explanation is worth more than the extras
        pass

    return HTTPException(status_code=400, detail=explain(exc, sql, names).as_dict())


@app.post("/query/analyze")
def analyze_query(req: QueryRequest):
    """
    `_analyze_query`, with a failed statement turned into an answer.

    Any error PostgreSQL raises about the SQL itself becomes a 400
    carrying the message, the offending token, a caret under it and --
    where it can be derived safely -- a corrected statement to copy.
    Previously this was a 500 and an empty panel, which told the reader
    only that something had gone wrong somewhere.
    """
    try:
        return _analyze_query(req)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - re-raised as an explained 400
        raise _sql_error_response(exc, req.sql) from exc


def _analyze_query(req: QueryRequest):
    """
    Run the baseline plan, generate join-order candidates, run each one,
    and report which the optimizer picked -- all in one call, so a
    dashboard can show baseline vs. candidates vs. chosen side by side.
    Every execution is logged to `plan_execution_log` (Phase 4), which is
    what `/stats/trend` reports on.

    Every candidate is *planned*; only `ANALYZE_EXECUTE_LIMIT` of them are
    executed, chosen by `_execution_order`. Planning a candidate costs 0.89 ms
    against 19-48 ms to execute one, and the model predicts latency from the
    plan tree rather than needing it measured -- so executing the whole action
    space bought a slower endpoint and nothing else. `n_candidates_planned` and
    `n_candidates_executed` in the response say which numbers are measured and
    which are the planner's estimate.
    """
    # A query that arrives already carrying a hint is the second pass of an
    # obvious workflow: optimise, copy the optimised SQL, paste it back in to
    # improve it further. Two things have to happen for that to mean anything.
    #
    # First, `apply_hint` merges rather than stacks -- pg_hint_plan reads only
    # the first `/*+ ... */` comment and silently discards the rest, so the old
    # behaviour threw away the hint the user pasted in.
    #
    # Second, the *baseline* has to be the unhinted query. Planning `req.sql` as
    # it arrived makes the previous pass's chosen plan the thing every candidate
    # is compared against, and the panel labels it "native" -- so a second pass
    # would report a speedup measured from an already-optimised plan, and would
    # keep reporting "no improvement available" once it had found the best one.
    # Both numbers would be wrong in a way nothing on the page could reveal.
    incoming_hints, bare_sql = split_hints(req.sql)

    with get_cursor() as cur:
        baseline_plan = get_plan(cur, bare_sql)
        tables = baseline_plan["tables_scanned"]

        # Full action space: join orders *and* Bao-style operator toggles.
        # Deduplicated, because most hints on a simple query re-produce the
        # native plan -- counting those as candidates makes the action space
        # look bigger than it is and guarantees a 0% improvement.
        candidate_plans = []
        seen_plans = {plan_fingerprint(baseline_plan)}
        for hint in candidate_hints(optimizer, baseline_plan, tables, cur=cur):
            try:
                # Merged onto whatever the user pasted in, so a second pass
                # refines the first hint rather than replacing it. `apply_hint`
                # keeps both unless they conflict, in which case the newly
                # chosen one wins -- pg_hint_plan's own last-one-wins rule.
                #
                # Planned, not executed. Measured on a four-table query:
                # planning a candidate costs **0.89 ms**, executing it costs
                # **19-48 ms**, and this endpoint used to execute all 19 --
                # 915 ms of a 1.2 s request. The model does not need actual
                # latencies to choose: it *predicts* them from the plan tree,
                # which planning already produces. Only the measured comparison
                # the dashboard shows needs real executions, and that needs a
                # handful, not all of them. See `_execution_order` below.
                plan = get_plan(cur, apply_hint(req.sql, hint), analyze=False)
            except Exception:  # noqa: BLE001 - one bad hint shouldn't fail the request
                continue
            fingerprint = plan_fingerprint(plan)
            if fingerprint in seen_plans:
                continue
            seen_plans.add(fingerprint)
            plan["hint"] = hint
            candidate_plans.append(plan)

        # select_plan applies the safety veto: a candidate the planner costs
        # far above native never gets served, however much the model likes it.
        served_plan = optimizer.select_plan(candidate_plans, baseline_plan=baseline_plan)
        decision = optimizer.last_decision
        chosen_index = decision.get("chosen_index") if candidate_plans else None
        kept_native = bool(decision.get("fell_back_to_baseline")) or chosen_index is None

        # Now spend the expensive budget, on the few candidates whose measured
        # latency the response actually reports.
        #
        # Deliberately *not* parallelised. Running candidates concurrently would
        # cut wall time, and would also make them compete for CPU, shared
        # buffers and I/O -- so every latency would be contaminated by whatever
        # happened to run beside it, unequally, because different plan shapes
        # have different memory and parallelism profiles. Those latencies are
        # what this whole project is built on, and they are logged as training
        # data. Executing fewer of them sequentially is 12.5x faster *and*
        # leaves every number comparable; executing all of them in parallel
        # would be neither.
        executed_indices = []
        for i in _execution_order(candidate_plans, chosen_index, ANALYZE_EXECUTE_LIMIT):
            try:
                measured = get_plan(cur, apply_hint(req.sql, candidate_plans[i]["hint"]))
            except Exception:  # noqa: BLE001 - one bad hint shouldn't fail the request
                continue
            measured["hint"] = candidate_plans[i]["hint"]
            candidate_plans[i] = measured
            executed_indices.append(i)

        # Log the baseline *after* the decision, because whether it was the
        # served plan is part of that decision. When the optimizer declines to
        # deviate, the native plan is what the user got, and marking it chosen
        # is what keeps the served history complete: leaving those runs with no
        # chosen row at all was why the dashboard's "chosen average" only ever
        # covered the queries the model felt brave about.
        log_execution(
            cur, query_id=None, sql_text=req.sql, plan=baseline_plan,
            is_baseline=True, selector_used="native", is_chosen=kept_native,
        )

        # Only the executed candidates are logged. A planned-only row has no
        # `actual_total_time_ms`, and `train.FETCH_SQL` filters those out
        # anyway -- writing them would add rows that can never be training data
        # and would inflate every count the dashboard draws from the log.
        for i in executed_indices:
            plan = candidate_plans[i]
            log_execution(
                cur, query_id=None, sql_text=req.sql, plan=plan, hint=plan["hint"],
                is_baseline=False, selector_used=SELECTOR_MODE,
                is_chosen=(i == chosen_index and not kept_native),
            )

    # The best candidate *actually measured* this run, which is what a
    # developer can copy and use today. This is distinct from `chosen_plan`
    # (what the model picked) and from `served_plan` (what the gate allowed):
    # /query/analyze executes everything, so it knows the real answer with
    # hindsight, and there is no reason to withhold it just because the model
    # was not confident enough to select it.
    # The locale decides whether an anchored LIKE needs `text_pattern_ops`, and
    # the unique-index set stops the advisor recommending an index on a primary
    # key that already has one. Both are catalog reads; failing either is a
    # reason to give slightly blunter advice, not to fail the request.
    try:
        with get_cursor() as advice_cur:
            c_locale = database_is_c_locale(advice_cur)
            already_indexed = already_uniquely_indexed(advice_cur)
    except Exception:  # noqa: BLE001
        c_locale, already_indexed = False, set()
    recommendations = analyze_plan(
        baseline_plan["raw_plan"], c_locale=c_locale, already_indexed=already_indexed
    )
    # Covering-index opportunities need the query's output columns, which only
    # `EXPLAIN VERBOSE` reports. That is one extra planning round trip -- which
    # /query/analyze can afford (it already issues around thirty) and
    # /query/optimize cannot, so this is deliberately only on the analyze path.
    try:
        with get_cursor() as verbose_cur:
            verbose_cur.execute(f"EXPLAIN (FORMAT JSON, VERBOSE) {req.sql}")
            verbose_plan = verbose_cur.fetchone()[0][0]["Plan"]
        recommendations += covering_opportunities({"raw_plan": verbose_plan})
    except Exception:  # noqa: BLE001 - an extra suggestion is not worth a failed request
        pass

    best_measured = None
    # Only the candidates that were actually executed have a latency -- which is
    # what "best measured" means. Since this endpoint stopped executing the whole
    # action space, the rest carry `actual_total_time_ms = None`, and comparing
    # those raises rather than quietly picking one.
    measured_plans = [
        p for p in candidate_plans if p.get("actual_total_time_ms") is not None
    ]
    if measured_plans and baseline_plan.get("actual_total_time_ms") is not None:
        fastest = min(measured_plans, key=lambda p: p["actual_total_time_ms"])
        if fastest["actual_total_time_ms"] < baseline_plan["actual_total_time_ms"]:
            best_measured = {
                "hint": fastest["hint"],
                "optimized_sql": apply_hint(req.sql, fastest["hint"]),
                "baseline_ms": baseline_plan["actual_total_time_ms"],
                "optimized_ms": fastest["actual_total_time_ms"],
                "speedup": baseline_plan["actual_total_time_ms"] / fastest["actual_total_time_ms"],
                "percent_faster": (
                    1 - fastest["actual_total_time_ms"] / baseline_plan["actual_total_time_ms"]
                ) * 100,
                # Postgres often costs the faster plan *higher* -- that gap is
                # exactly why it rejected it, and the whole premise here.
                "baseline_cost": baseline_plan["total_cost"],
                "optimized_cost": fastest["total_cost"],
                "baseline_est_rows": baseline_plan["raw_plan"].get("Plan Rows"),
                "baseline_actual_rows": baseline_plan["raw_plan"].get("Actual Rows"),
                # Whether this speedup is about the plan or about what happened
                # to be in memory.
                #
                # This is the number the dashboard puts in front of people, and
                # it is a single-run comparison -- exactly the shape of
                # measurement that once produced a reported 80.5% improvement
                # that was really 26.0%, because one side read from memory and
                # the other from disk (docs/WRITEUP.md). The paired design in
                # experiment.py fixes that over many runs; nothing fixed it
                # here, and this panel is what most people ever look at.
                #
                # `EXPLAIN (ANALYZE, BUFFERS)` was already being collected. The
                # hit ratios say whether the two sides were comparable, and the
                # block counts are a second opinion that is stable across runs
                # where timing is not (measured: 0.0% spread against ~5%).
                "fairness": fairness(
                    baseline_plan, fastest, "native", "optimized"
                ),
                "work": blocks_verdict(
                    baseline_plan, fastest, "native", "optimized"
                ),
            }

    return {
        "baseline": baseline_plan,
        # What the query arrived with, if anything. The baseline above is the
        # *unhinted* plan whatever the input was, so "native" means native.
        "incoming_hints": incoming_hints,
        "candidates": candidate_plans,
        # Planned is the size of the action space; executed is how many carry a
        # real latency. They differ on purpose, and a reader comparing
        # candidates needs to know which numbers are measured and which are the
        # planner's estimate.
        "n_candidates_planned": len(candidate_plans),
        "n_candidates_executed": len(executed_indices),
        "executed_indices": executed_indices,
        "chosen_index": chosen_index,
        "chosen_plan": candidate_plans[chosen_index] if chosen_index is not None else None,
        "served_plan": served_plan,
        "selector_mode": SELECTOR_MODE,
        # Why this plan: policy, predicted latency, ensemble uncertainty, and
        # whether the safety net vetoed the model's pick.
        "decision": decision,
        "best_measured": best_measured,
        # Database-level fixes: hints repair one query, these repair the
        # reason a hint was needed (see app/advisor.py).
        # The locale decides whether an anchored LIKE needs `text_pattern_ops`,
        # and the unique-index set stops the advisor recommending an index on a
        # primary key that already has one.
        "recommendations": recommendations,
    }


@app.post("/query/optimize")
def optimize_query(req: QueryRequest):
    """
    `_optimize_query`, with a failed statement turned into an answer.

    Any error PostgreSQL raises about the SQL itself becomes a 400
    carrying the message, the offending token, a caret under it and --
    where it can be derived safely -- a corrected statement to copy.
    Previously this was a 500 and an empty panel, which told the reader
    only that something had gone wrong somewhere.
    """
    try:
        return _optimize_query(req)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - re-raised as an explained 400
        raise _sql_error_response(exc, req.sql) from exc


def _optimize_query(req: QueryRequest):
    """
    The **production** path: pick a plan without executing the alternatives.

    Costs N cheap `EXPLAIN`s (planning only, nothing run) plus one real
    execution, versus `/query/analyze`, which executes every candidate and is
    therefore a measurement harness rather than an optimizer. The
    `optimizer_overhead_ms` vs `execution_ms` split in the response is what
    tells you whether the decision paid for itself.
    """
    # Without an id the regression guard is inert -- `is_blocked(None)` is
    # always False -- so every query served through here bypassed the very
    # safety net /model/status reports on. A fingerprint gives ad-hoc traffic
    # the stable identity the guard needs to accumulate a history for it.
    query_id = query_fingerprint(req.sql)

    with get_cursor() as cur:
        result = optimize_and_execute(
            cur, req.sql, optimizer, query_id=query_id, guard=guard, cache=decision_cache
        )
        log_execution(
            cur, query_id=query_id, sql_text=req.sql, plan=result["executed_plan"],
            hint=result["hint"], is_baseline=result["hint"] is None,
            selector_used=SELECTOR_MODE, is_chosen=True,
        )
        # The other half of a sampled pair. Written in the same transaction, so
        # it shares the `created_at` that `app.stats` groups a run by -- that is
        # what makes the two rows a pair rather than two unrelated executions.
        if result.get("shadow_native_plan") is not None:
            log_execution(
                cur, query_id=query_id, sql_text=req.sql,
                plan=result["shadow_native_plan"],
                is_baseline=True, selector_used="native", is_chosen=False,
            )
        # The learned plan for a *blocked* query: measured, never served.
        #
        # `is_chosen=False` because nobody received it, and
        # `selector_used=SHADOW_SELECTOR` because the regression guard needs it
        # and `app.stats` must not. Without this a blocked query accumulates no
        # evidence about the learned path at all, so its block lifts when the
        # old evidence ages out rather than when the model improves -- a timer,
        # not a decision.
        if result.get("shadow_learned_plan") is not None:
            log_execution(
                cur, query_id=query_id, sql_text=req.sql,
                plan=result["shadow_learned_plan"],
                hint=result.get("shadow_learned_hint"),
                is_baseline=False, selector_used=SHADOW_SELECTOR, is_chosen=False,
            )

    return {
        "hint": result["hint"],
        "reason": result["reason"],
        "execution_ms": result["execution_ms"],
        "optimizer_overhead_ms": result["optimizer_overhead_ms"],
        "n_candidates_planned": result["n_candidates_planned"],
        "executed_plan": result["executed_plan"],
        "decision": result.get("decision"),
        # Whether the choosing was reused. Without this, a near-zero
        # `optimizer_overhead_ms` is unexplained -- it reads as the planner
        # having become mysteriously fast rather than not having run.
        "from_cache": result.get("from_cache", False),
        # Whether this request also measured PostgreSQL's plan, and what it
        # cost. Present only on sampled requests; see planner.SHADOW_SAMPLE_RATE.
        "native_ms": (
            result["shadow_native_plan"]["actual_total_time_ms"]
            if result.get("shadow_native_plan") is not None
            else None
        ),
    }


class AdvisorRequest(BaseModel):
    sql: str


@app.get("/advisor")
def advisor():
    """
    Schema-wide recommendations that don't depend on a specific query --
    currently unindexed foreign keys, which slow every join through them.
    Per-query advice comes back on /query/analyze as `recommendations`.
    """
    with get_cursor() as cur:
        return {"recommendations": missing_fk_indexes(cur)}


@app.get("/dbadvisor")
def database_advisor(include_statistics: bool = False):
    """
    What is wrong with the database, as opposed to with one query's plan.

    The rest of this API answers "given this query, which plan should run".
    That is the hard part and it is not the only reason a database is slow --
    on most databases it is not even the first one. This reports the things that
    make the planner's estimates wrong in the first place, each of which is
    fixed once and then helps every query, including ones never seen here:
    correlated columns the planner does not know about, indexes that are never
    used, stale statistics, and server settings that are defaults nobody chose.

    `include_statistics` is off by default because measuring a correlated-column
    candidate costs an `ANALYZE` per candidate, which is not something a polled
    endpoint should do. Everything else is a catalogue read.

    Nothing is changed. Each finding carries the statement that would apply it.
    """
    with get_cursor() as cur:
        # The workload is passed so the index findings can answer "what would
        # dropping this do to my plans" rather than only "has it been scanned"
        # -- a counter that resets on `pg_stat_reset()` and starts at zero on a
        # replica promoted yesterday. See `dbadvisor.index_simulation`.
        findings = dbadvisor.report(
            cur, include_statistics=include_statistics, queries=list(workload.WORKLOAD)
        )
    return {"n_findings": dbadvisor.count_findings(findings), **findings}


@app.post("/advisor/rank")
def advisor_rank(req: AdvisorRequest):
    """
    The same recommendations, ranked by what they would actually be worth.

    `/advisor` can say an index is missing but not how much it would help,
    because finding out means building it -- slow, lock-taking, and not
    something a read-only endpoint should do to someone's database. Most
    missing indexes are missing because nothing needs them, so an unranked
    list gives no basis for choosing between them.

    HypoPG registers an index in the planner's catalogue only, and `EXPLAIN`
    then plans as though it existed. What comes back is the estimated cost
    before and after, and -- the decisive part -- whether the planner would
    choose it at all. See `app/hypothetical.py` for what that does and does
    not measure.
    """
    with get_cursor() as cur:
        if not hypothetical.ensure_extension(cur):
            return {
                "available": False,
                "reason": (
                    "hypopg is not installed in this database, so index "
                    "recommendations cannot be ranked. Rebuild the Postgres image "
                    "(postgres/Dockerfile) or CREATE EXTENSION hypopg."
                ),
                "recommendations": missing_fk_indexes(cur),
            }

        candidates = missing_fk_indexes(cur)
        ranked = hypothetical.rank_candidates(cur, req.sql, candidates)

    return {
        "available": True,
        "n_candidates_considered": len(candidates),
        "recommendations": ranked,
    }


@app.get("/stats/cost-model")
def stats_cost_model():
    """
    How well PostgreSQL's cost estimates track measured latency.

    The rank correlation this returns is the project's premise stated as a
    number: the planner chooses by minimising cost, so the gap between that
    ordering and the real one is the entire space a learned optimizer has to
    work in.
    """
    with get_cursor() as cur:
        return stats.cost_vs_latency(cur)


@app.get("/stats/regret")
def stats_regret():
    """
    Cumulative regret over time -- how much slower the served plans were than
    the best available, against the same figure for native Postgres.
    A ratio below 1.0 means the learned path has been worth having.
    """
    with get_cursor() as cur:
        return regret_curve(cur)


@app.get("/schema")
def schema():
    """
    What this optimizer has discovered about the database it's pointed at.

    Present so "works on any dataset" is inspectable rather than a claim:
    point `DATABASE_URL` somewhere new and this reports the tables, join
    edges, and whether those edges were declared or inferred from naming.
    """
    with get_cursor() as cur:
        graph = discover_with_inference(cur)
    return schema_summary(graph)


@app.get("/model/status")
def model_status():
    """
    What's deployed, how stale it is, and which queries the retrospective
    guard is currently refusing to serve learned plans for.
    """
    with get_cursor() as cur:
        blocked = guard.refresh(cur)

    return {
        "selector_mode": SELECTOR_MODE,
        "policy": optimizer.policy,
        "current_version": model_store.current_version(),
        "rows_since_last_training": retrain.rows_since_last_training(),
        "versions": model_store.list_versions()[:10],
        "regression_guard": {
            "tolerance": guard.tolerance,
            "min_observations": guard.min_observations,
            "blocked_queries": blocked,
            # Finer-grained than `blocked_queries`: which *kinds* of action have
            # lost on which query, so a query can keep being optimised by the
            # families that still work on it.
            "blocked_families": guard.blocked_families,
        },
        # How much of the planning cost repeat traffic is actually avoiding.
        # A hit rate near zero means the cache is pure overhead and the TTL or
        # the traffic pattern is wrong.
        "decision_cache": decision_cache.stats() if decision_cache is not None else None,
    }


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics():
    """
    Everything worth alerting on, in Prometheus' text exposition format.

    The rest of this API answers questions when asked. This is the half that
    lets the system be *watched*: a regression rate climbing past what the gate
    promised, or a deviation rate that has fallen to zero and turned the
    optimizer into an expensive no-op, are both things you want to find out
    from a graph rather than by opening the dashboard.

    One scrape costs the same three reports the dashboard polls, so scrape
    interval should be measured in tens of seconds, not seconds.
    """
    with get_cursor() as cur:
        trend = stats.served_vs_native(cur)
        report = diagnostics.diagnose(cur=cur)
        blocked = guard.refresh(cur)

    status = {
        "current_version": model_store.current_version(),
        "rows_since_last_training": retrain.rows_since_last_training(),
        "regression_guard": {"blocked_queries": blocked},
        "decision_cache": decision_cache.stats() if decision_cache is not None else None,
    }
    return metrics.render(trend, report, status)


@app.get("/model/diagnostics")
def model_diagnostics():
    """
    Why the optimizer is behaving as it is -- as opposed to what it did.

    `/model/status` says what is deployed and `/stats/trend` says how it has
    performed; neither answers "and should I believe it". That question was
    first answered with throwaway scripts, after a dashboard showed a plan
    predicted at 0.92x native running 1.5x slower, and every one of the four
    findings (a gate with `confidence_z = 0`, calibrated in-sample, on five
    queries, over an uncertainty understated 2.4x) was invisible from the
    running system. See `app/diagnostics.py`.
    """
    with get_cursor() as cur:
        return diagnostics.diagnose(cur=cur)


@app.post("/model/retrain")
def model_retrain(force: bool = False):
    """
    Retrain on accumulated feedback and promote only if the challenger
    clearly beats the incumbent (see app.retrain). Safe to call repeatedly:
    it no-ops until enough new data has arrived.

    Note this reloads the served model in-process on promotion, so a
    successful retrain takes effect without a restart.
    """
    result = retrain.retrain_if_needed(force=force)
    if result.get("action") == "promoted":
        _reload_optimizer()
    return result


@app.post("/model/rollback")
def model_rollback():
    """Promote the previous version -- the escape hatch when a promoted
    model turns out worse in production than it looked offline."""
    target = model_store.rollback()
    if target:
        _reload_optimizer()
    return {"action": "rollback", "promoted": target}


@app.get("/stats/trend")
def stats_trend():
    """
    "Served vs. native, over time" -- the data behind the dashboard's
    historical-accuracy panel.

    Every figure here is a **matched pair**: the native plan and the served
    plan for the same query, measured in the same run. See `app.stats` for
    why the previous version of this endpoint (two averages over unrelated
    populations) reported improvements that were not real.
    """
    with get_cursor() as cur:
        result = stats.served_vs_native(cur)

    result["overall"]["selector_mode"] = SELECTOR_MODE
    return result


@app.get("/settings")
def get_settings():
    """
    Current values plus the metadata the settings UI renders itself from.

    `in_force` is the third thing, and it exists because two of these values are
    not what the optimizer applies. `confidence_z` and `min_relative_gain`
    default to 1.0 and 0.05 in `app.settings`, and `LearnedOptimizer` overrides
    both with `models/gate.json` unless someone explicitly chose them
    (`_optimizer_from`). So the panel read **z = 1.0** while the running gate was
    **1.5**, and there was no way to tell from the endpoint.

    That is the same failure this module's own header describes as fixed -- the
    panel reporting one configuration while the process runs another -- in the
    opposite direction. Deferring to the calibration is right; showing the value
    that was overridden as though it were in force is not.

    `in_force` reports what the live optimizer actually holds, and `overridden`
    names the fields where the two differ, so the UI can say *"1.0 (calibrated
    gate is using 1.5)"* rather than silently lying.
    """
    values = app_settings.current()
    live = {
        "confidence_z": getattr(optimizer, "confidence_z", None),
        "min_relative_gain": getattr(optimizer, "min_relative_gain", None),
        "gate_mode": getattr(optimizer, "gate_mode", None),
        "selection_policy": getattr(optimizer, "policy", None),
    }
    return {
        "fields": app_settings.describe(),
        "values": values,
        "in_force": live,
        "overridden": sorted(
            name for name, actual in live.items()
            if actual is not None and values.get(name) != actual
        ),
        "gate_source": getattr(optimizer, "gate_calibration", None) and "models/gate.json",
        "groups": list(dict.fromkeys(f["group"] for f in app_settings.FIELDS.values())),
    }


@app.put("/settings")
def put_settings(changes: dict[str, object]):
    """Validate, persist and apply a partial settings update."""
    try:
        values = app_settings.update(changes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    _apply_settings(values)
    return {"values": values, "applied": True, "selector_mode": SELECTOR_MODE}


@app.post("/settings/reset")
def reset_settings():
    """Back to environment/declared defaults."""
    values = app_settings.reset()
    _apply_settings(values)
    return {"values": values, "applied": True}


@app.get("/databases")
def list_databases():
    """Saved connections, which one is live, and whether the model fits it."""
    return {**db_profiles.list_profiles(), "model_fit": db_profiles.schema_matches_model(optimizer)}


class ConnectionRequest(BaseModel):
    name: str | None = None
    url: str


@app.post("/databases/test")
def test_database(req: ConnectionRequest):
    """Probe a connection without saving or switching to it."""
    return db_profiles.test_connection(req.url)


@app.post("/databases")
def add_database(req: ConnectionRequest):
    try:
        return db_profiles.add_profile(req.name or "", req.url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.delete("/databases/{name}")
def delete_database(name: str):
    return db_profiles.remove_profile(name)


@app.post("/databases/{name}/activate")
def activate_database(name: str):
    """
    Point the optimizer at a saved connection.

    The guard and the decision cache are both rebuilt: their contents are
    keyed by query against the *previous* database, and carrying them over
    would block or shortcut queries on evidence from somewhere else entirely.
    """
    try:
        result = db_profiles.activate(name)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if decision_cache is not None:
        decision_cache.clear()
    try:
        with get_cursor() as cur:
            guard.refresh(cur)
    except Exception:  # noqa: BLE001 - a fresh database has no history yet
        guard.blocked = {}

    return {**result, "model_fit": db_profiles.schema_matches_model(optimizer)}


# -- the user's own workload ------------------------------------------------


class SavedQueryRequest(BaseModel):
    name: str
    sql: str
    description: str = ""


class ValidateRequest(BaseModel):
    sql: str


@app.get("/queries")
def list_saved_queries():
    """The saved workload for whichever database is currently active."""
    return {
        "queries": saved_queries.list_queries(db.current_database_url()),
        "builtin_count": len(workload.WORKLOAD),
    }


@app.post("/queries/validate")
def validate_saved_query(req: ValidateRequest):
    """
    Plan a query without running it, so the UI can say whether it is worth
    saving before a collection run discovers it is not.
    """
    with get_cursor() as cur:
        return saved_queries.validate(cur, req.sql)


@app.post("/queries")
def save_query(req: SavedQueryRequest):
    try:
        queries = saved_queries.save_query(
            db.current_database_url(), req.name, req.sql, req.description
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"queries": queries}


@app.delete("/queries/{name}")
def delete_saved_query(name: str):
    return {"queries": saved_queries.delete_query(db.current_database_url(), name)}


# -- training from the dashboard --------------------------------------------


class TrainRequest(BaseModel):
    reps: int = 3
    promote: bool = True
    include_builtin: bool = False
    include_join_methods: bool = True


@app.get("/train/status")
def train_status():
    """Progress of the current or most recent run. Safe to poll."""
    return training_job.job.snapshot()


@app.post("/train/start")
def train_start(req: TrainRequest):
    """
    Collect training data for the saved queries, then fit and (by default)
    promote a model -- the whole loop, without leaving the dashboard.

    Promotion still goes through the champion/challenger comparison in
    `app.retrain`, so pressing this cannot install a model that is worse than
    the one already serving.
    """
    queries = saved_queries.as_workload(db.current_database_url())
    if req.include_builtin:
        queries = [{"id": q["id"], "sql": q["sql"]} for q in workload.WORKLOAD] + queries

    try:
        snapshot = training_job.job.start(
            queries,
            reps=req.reps,
            promote=req.promote,
            include_join_methods=req.include_join_methods,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return snapshot


@app.post("/train/stop")
def train_stop():
    """
    Ask the run to stop after the query it is on.

    Rows already collected are committed and kept: a rerun appends to them
    rather than starting over, so stopping is not the same as discarding.
    """
    return training_job.job.stop()


# -- settling a comparison properly -----------------------------------------


class ExperimentRequest(BaseModel):
    comparison: str = "guard"
    policy: str = "pairwise_rank"
    arm_a: str = "risk_averse"
    arm_b: str = "pairwise_rank"
    runs: int = 10
    limit: int | None = None


@app.get("/experiment/status")
def experiment_status():
    """Progress and result of the current or most recent comparison."""
    return experiment_job.job.snapshot()


@app.post("/experiment/start")
def experiment_start(req: ExperimentRequest):
    """
    Start a paired A/B comparison in the background.

    `app.experiment` is the only thing here that settles a question properly --
    interleaved arms, a bootstrap interval, a sign test, and the word
    *unresolved* when the interval spans zero. It was also the only thing that
    needed a terminal, which made the rigorous comparison the inconvenient one.
    docs/WRITEUP.md 2.4 records the cost of that: a headline result with a
    sample size of three.
    """
    try:
        return experiment_job.job.start(
            comparison=req.comparison, policy=req.policy,
            arm_a=req.arm_a, arm_b=req.arm_b, runs=req.runs, limit=req.limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/model/reload")
def model_reload():
    """Pick up a model trained out-of-band, without restarting the backend."""
    _reload_optimizer()
    return {"selector_mode": SELECTOR_MODE, "current_version": model_store.current_version()}
