import asyncio
import logging
import os
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app import model_store, retrain, stats
from app.advisor import analyze_plan, missing_fk_indexes
from app.db import get_cursor
from app.logging_store import log_execution, query_fingerprint
from app.optimizer.decision_cache import DecisionCache
from app.optimizer.hints import apply_hint, plan_fingerprint
from app.optimizer.learned import LearnedOptimizer
from app.optimizer.planner import candidate_hints, optimize_and_execute
from app.optimizer.regression_guard import RegressionGuard
from app.optimizer.regret import regret_curve
from app.plan_extractor import get_plan
from app.schema_graph import discover_with_inference
from app.schema_graph import summarize as schema_summary

optimizer = LearnedOptimizer()
SELECTOR_MODE = "learned" if optimizer.model is not None else "heuristic"
guard = RegressionGuard()

# Repeat traffic reuses the decision instead of re-deriving it. Measured on a
# 4-table join, choosing cost 25.2 ms against an execution of 21.8 ms -- the
# optimizer was more expensive than the query it optimized, and every repeat
# paid it again to reach the same answer. Set DECISION_CACHE_SECONDS=0 to
# disable, which is what a benchmark run wanting every decision made from
# scratch should do.
DECISION_CACHE_SECONDS = float(os.getenv("DECISION_CACHE_SECONDS", "300"))
decision_cache = (
    DecisionCache(ttl_seconds=DECISION_CACHE_SECONDS) if DECISION_CACHE_SECONDS > 0 else None
)

log = logging.getLogger("lqo.autolearn")

# The system retrains itself on accumulated feedback on this cadence. Off by
# default: a background job that silently swaps the served model is exactly
# the sort of thing that should be opted into, and it makes benchmark runs
# non-reproducible while it's on. `retrain_if_needed` still gates every
# promotion on the champion/challenger comparison, so the worst case is a
# wasted retrain, not a regression.
AUTO_RETRAIN_SECONDS = int(os.getenv("AUTO_RETRAIN_SECONDS", "0"))


class QueryRequest(BaseModel):
    sql: str


async def _auto_retrain_loop() -> None:
    """Periodically fold accumulated feedback back into the served model."""
    while True:
        await asyncio.sleep(AUTO_RETRAIN_SECONDS)
        try:
            # Training is CPU-bound and synchronous; keep it off the event
            # loop so query serving isn't blocked while a retrain runs.
            result = await asyncio.to_thread(retrain.retrain_if_needed)
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


@app.post("/query/analyze")
def analyze_query(req: QueryRequest):
    """
    Run the baseline plan, generate join-order candidates, run each one,
    and report which the optimizer picked -- all in one call, so a
    dashboard can show baseline vs. candidates vs. chosen side by side.
    Every execution is logged to `plan_execution_log` (Phase 4), which is
    what `/stats/trend` reports on.

    Note: this executes every candidate for comparison, which is fine for
    a dev/demo dashboard but is NOT what you'd do in production (there
    you'd only execute the chosen plan). Worth a sentence in your writeup.
    """
    with get_cursor() as cur:
        baseline_plan = get_plan(cur, req.sql)
        tables = baseline_plan["tables_scanned"]

        # Full action space: join orders *and* Bao-style operator toggles.
        # Deduplicated, because most hints on a simple query re-produce the
        # native plan -- counting those as candidates makes the action space
        # look bigger than it is and guarantees a 0% improvement.
        candidate_plans = []
        seen_plans = {plan_fingerprint(baseline_plan)}
        for hint in candidate_hints(optimizer, baseline_plan, tables):
            try:
                plan = get_plan(cur, apply_hint(req.sql, hint))
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

        for i, plan in enumerate(candidate_plans):
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
    best_measured = None
    if candidate_plans:
        fastest = min(candidate_plans, key=lambda p: p["actual_total_time_ms"])
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
            }

    return {
        "baseline": baseline_plan,
        "candidates": candidate_plans,
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
        "recommendations": analyze_plan(baseline_plan["raw_plan"]),
    }


@app.post("/query/optimize")
def optimize_query(req: QueryRequest):
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
    }


@app.get("/advisor")
def advisor():
    """
    Schema-wide recommendations that don't depend on a specific query --
    currently unindexed foreign keys, which slow every join through them.
    Per-query advice comes back on /query/analyze as `recommendations`.
    """
    with get_cursor() as cur:
        return {"recommendations": missing_fk_indexes(cur)}


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
        },
        # How much of the planning cost repeat traffic is actually avoiding.
        # A hit rate near zero means the cache is pure overhead and the TTL or
        # the traffic pattern is wrong.
        "decision_cache": decision_cache.stats() if decision_cache is not None else None,
    }


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


def _reload_optimizer() -> None:
    global optimizer, SELECTOR_MODE
    optimizer = LearnedOptimizer(policy=optimizer.policy, risk_lambda=optimizer.risk_lambda)
    SELECTOR_MODE = "learned" if optimizer.model is not None else "heuristic"
    # Every cached decision was made by the model being replaced. Keeping them
    # would let a freshly promoted model serve its predecessor's choices and
    # appear to have changed nothing -- the retrain would look like a no-op.
    if decision_cache is not None:
        decision_cache.clear()


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
