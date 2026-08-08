import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.db import get_cursor
from app.logging_store import log_execution
from app.optimizer.hints import apply_hint, generate_join_order_candidates
from app.optimizer.learned import LearnedOptimizer
from app.plan_extractor import get_plan

app = FastAPI(title="Learned Query Optimizer")

# Dev origin only -- the Vite dashboard from Phase 5. FRONTEND_ORIGIN lets
# docker-compose override this for other environments without editing code.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")],
    allow_methods=["*"],
    allow_headers=["*"],
)

optimizer = LearnedOptimizer()
SELECTOR_MODE = "learned" if optimizer.model is not None else "heuristic"


class QueryRequest(BaseModel):
    sql: str


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
        log_execution(
            cur, query_id=None, sql_text=req.sql, plan=baseline_plan,
            is_baseline=True, selector_used="native",
        )

        candidate_plans = []
        for hint in generate_join_order_candidates(tables):
            hinted_query = apply_hint(req.sql, hint)
            plan = get_plan(cur, hinted_query)
            plan["hint"] = hint
            candidate_plans.append(plan)

        chosen_index = optimizer.select(candidate_plans) if candidate_plans else None

        for i, plan in enumerate(candidate_plans):
            log_execution(
                cur, query_id=None, sql_text=req.sql, plan=plan, hint=plan["hint"],
                is_baseline=False, selector_used=SELECTOR_MODE, is_chosen=(i == chosen_index),
            )

    return {
        "baseline": baseline_plan,
        "candidates": candidate_plans,
        "chosen_index": chosen_index,
        "chosen_plan": candidate_plans[chosen_index] if chosen_index is not None else None,
        "selector_mode": SELECTOR_MODE,
    }


TREND_SQL = """
    SELECT
        date_trunc('day', created_at) AS day,
        AVG(actual_total_time_ms) FILTER (WHERE is_baseline) AS native_avg_ms,
        AVG(actual_total_time_ms) FILTER (WHERE is_chosen) AS chosen_avg_ms,
        COUNT(*) FILTER (WHERE is_chosen AND selector_used = 'learned') AS n_learned,
        COUNT(*) FILTER (WHERE is_chosen AND selector_used = 'heuristic') AS n_heuristic
    FROM plan_execution_log
    GROUP BY 1
    ORDER BY 1
"""

OVERALL_SQL = """
    SELECT
        AVG(actual_total_time_ms) FILTER (WHERE is_baseline) AS native_avg_ms,
        AVG(actual_total_time_ms) FILTER (WHERE is_chosen) AS chosen_avg_ms,
        COUNT(*) FILTER (WHERE is_baseline) AS n_native,
        COUNT(*) FILTER (WHERE is_chosen) AS n_chosen
    FROM plan_execution_log
"""


@app.get("/stats/trend")
def stats_trend():
    """
    Aggregated "learned vs. native, over time" history (Phase 4/5) -- the
    data behind the dashboard's historical-accuracy chart.
    """
    with get_cursor() as cur:
        cur.execute(OVERALL_SQL)
        o_native, o_chosen, n_native, n_chosen = cur.fetchone()

        cur.execute(TREND_SQL)
        rows = cur.fetchall()

    return {
        "overall": {
            "native_avg_latency_ms": o_native,
            "chosen_avg_latency_ms": o_chosen,
            "n_native": n_native,
            "n_chosen": n_chosen,
            "selector_mode": SELECTOR_MODE,
        },
        "by_day": [
            {
                "day": day.isoformat(),
                "native_avg_latency_ms": native_avg,
                "chosen_avg_latency_ms": chosen_avg,
                "n_learned": n_learned,
                "n_heuristic": n_heuristic,
            }
            for day, native_avg, chosen_avg, n_learned, n_heuristic in rows
        ],
    }
