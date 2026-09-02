"""Does the model rank well but predict the wrong absolute level?

The eval says it captures 79% of the oracle headroom, which is a statement about
*ordering*. The gate compares a predicted ratio against a threshold, which is a
statement about *level*. Those can disagree, and if they do the gate vetoes
everything while the model is choosing correctly.

For each workload query: plan every candidate, ask the model for its predicted
ratio, execute to get the true ratio, and compare.
"""
import statistics

from app.db import get_cursor
from app.optimizer.hints import apply_hint
from app.optimizer.learned import LearnedOptimizer
from app.optimizer.planner import candidate_hints
from app.plan_extractor import get_plan
from app.workload import WORKLOAD

import os
opt = LearnedOptimizer(policy="greedy",
                       model_path=os.environ.get("PROBE_MODEL", "models/plan_selector.pkl"))

rows = []
with get_cursor() as cur:
    cur.execute("LOAD 'pg_hint_plan'")
    cur.execute("SET pg_hint_plan.enable_hint = on")
    # A forced nested loop over 500k rows runs for minutes; the point is the
    # bias, not the tail. Anything over the budget is skipped as a pair.
    cur.execute("SET statement_timeout = 3000")
    for spec in WORKLOAD[:10]:
        sql = spec["sql"]
        try:
            baseline = get_plan(cur, sql, analyze=True)
        except Exception:
            cur.connection.rollback()
            cur.execute("SET statement_timeout = 3000")
            continue
        base_ms = baseline["actual_total_time_ms"]
        if not base_ms:
            continue

        plans = []
        for hint in candidate_hints(opt, baseline, baseline["tables_scanned"], cur=cur):
            try:
                p = get_plan(cur, apply_hint(sql, hint), analyze=True)
            except Exception:
                cur.connection.rollback()
                cur.execute("SET statement_timeout = 3000")
                continue
            p["hint"] = hint
            plans.append(p)
        if not plans:
            continue

        vectors = opt._vectorize(plans)
        predicted = opt.model.predict(vectors) if opt.model is not None else None
        if predicted is None:
            continue

        for plan, pred_log in zip(plans, predicted, strict=True):
            true_ratio = plan["actual_total_time_ms"] / base_ms
            rows.append((float(pred_log), true_ratio, spec["id"]))

import math

pred_ratios = [math.exp(p) for p, _, _ in rows]
true_ratios = [t for _, t, _ in rows]
log_errors = [p - math.log(t) for p, t, _ in rows]

print(f"candidates measured: {len(rows)}")
print()
print(f"{'':<22}{'predicted':>12}{'actual':>12}")
print(f"{'median ratio':<22}{statistics.median(pred_ratios):>12.3f}{statistics.median(true_ratios):>12.3f}")
print(f"{'mean ratio':<22}{statistics.fmean(pred_ratios):>12.3f}{statistics.fmean(true_ratios):>12.3f}")
print(f"{'min ratio':<22}{min(pred_ratios):>12.3f}{min(true_ratios):>12.3f}")
print()
bias = statistics.fmean(log_errors)
print(f"mean log error (predicted - actual): {bias:+.4f}  ->  x{math.exp(bias):.3f} systematic")
print(f"median log error                   : {statistics.median(log_errors):+.4f}")
print(f"stdev of log error                 : {statistics.pstdev(log_errors):.4f}")
print()
# The question the gate actually asks.
better = [(p, t) for p, t, _ in rows if t < 0.8]
print(f"candidates genuinely faster than 0.8x native: {len(better)}")
if better:
    got_through = sum(1 for p, t in better if math.exp(p) < 0.8)
    print(f"  ...that the model also predicted under 0.8x: {got_through}")
    print(f"  their predicted ratios: {sorted(round(math.exp(p), 2) for p, _ in better)[:12]}")
