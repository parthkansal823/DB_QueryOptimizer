from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.db import get_cursor
from app.optimizer.hints import apply_hint, generate_join_order_candidates
from app.optimizer.learned import LearnedOptimizer
from app.plan_extractor import get_plan

app = FastAPI(title="Learned Query Optimizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before it's a real dashboard
    allow_methods=["*"],
    allow_headers=["*"],
)

optimizer = LearnedOptimizer()


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

    Note: this executes every candidate for comparison, which is fine for
    a dev/demo dashboard but is NOT what you'd do in production (there
    you'd only execute the chosen plan). Worth a sentence in your writeup.
    """
    with get_cursor() as cur:
        baseline_plan = get_plan(cur, req.sql)
        tables = baseline_plan["tables_scanned"]

        candidate_plans = []
        for hint in generate_join_order_candidates(tables):
            hinted_query = apply_hint(req.sql, hint)
            plan = get_plan(cur, hinted_query)
            plan["hint"] = hint
            candidate_plans.append(plan)

        chosen_index = optimizer.select(candidate_plans) if candidate_plans else None

    return {
        "baseline": baseline_plan,
        "candidates": candidate_plans,
        "chosen_index": chosen_index,
        "chosen_plan": candidate_plans[chosen_index] if chosen_index is not None else None,
    }
