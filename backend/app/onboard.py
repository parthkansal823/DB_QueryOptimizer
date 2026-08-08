"""
One command to point this optimizer at *any* PostgreSQL database.

    DATABASE_URL=postgresql://user:pass@host:5432/yourdb \\
        python -m app.onboard --queries 25 --reps 3

It will:

    1. discover the schema  -- tables, foreign keys, columns, indexes
    2. generate a workload  -- connected joins with sampled, realistic predicates
    3. collect training data -- every candidate plan, executed and timed
    4. train + evaluate      -- and report against native Postgres and an oracle

Before this existed, using the system on a new dataset meant hand-writing a
workload module, because `workload.py` was 25 queries specific to the
synthetic e-commerce schema. Everything downstream was already schema-
agnostic (see `schema_introspection.py` and the `scan_relations`-based
featurisation); the workload was the last hardcoded piece.

The generated workload is saved to `models/workload_<db>.json` so a run can
be inspected, edited, and replayed -- auto-generation is a starting point,
not a claim that the queries are the ones you care about. Point it at your
real query log if you have one.
"""

from __future__ import annotations

import argparse
import json
import os

from app import schema_graph
from app.collect_data import collect
from app.db import get_cursor
from app.logging_store import ensure_log_table
from app.train import train
from app.workload_generator import generate_workload


def _database_name() -> str:
    url = os.getenv("DATABASE_URL", "")
    return url.rsplit("/", 1)[-1].split("?")[0] or "database"


def onboard(
    n_queries: int = 25,
    reps: int = 3,
    include_join_methods: bool = False,
    timeout_ms: int = 20_000,
    skip_training: bool = False,
) -> dict:
    db = _database_name()
    report: dict = {"database": db}

    with get_cursor() as cur:
        # The bookkeeping table lives per-database; a user's own DB won't have it.
        ensure_log_table(cur)

        print(f"==> discovering schema of '{db}'")
        graph = schema_graph.discover_with_inference(cur)
        report["schema"] = schema_graph.summarize(graph)
        origin = "inferred from naming" if graph.inferred_fks else "declared"
        print(
            f"    {report['schema']['n_tables']} tables, "
            f"{report['schema']['n_foreign_keys']} join edges ({origin}), "
            f"{report['schema']['total_rows']:,} rows"
        )

        if report["schema"]["n_foreign_keys"] == 0:
            raise SystemExit(
                "No foreign keys found. This optimizer learns join *order*, so it needs "
                "a schema whose tables reference each other. Add FK constraints (they can "
                "be NOT VALID if you don't want enforcement) or supply a workload by hand."
            )

        print(f"==> generating a {n_queries}-query workload")
        workload = generate_workload(cur, graph, n_queries=n_queries)
        report["n_queries_generated"] = len(workload)
        for item in workload[:3]:
            print(f"    e.g. {item['id']}: {item['description']}")

    if not workload:
        raise SystemExit(
            "Could not generate any queries. The schema has foreign keys but no columns "
            "this generator knows how to filter on -- supply a workload by hand."
        )

    os.makedirs("models", exist_ok=True)
    workload_path = f"models/workload_{db}.json"
    with open(workload_path, "w") as f:
        json.dump(workload, f, indent=2)
    print(f"    saved to {workload_path}")

    print(f"==> collecting training data ({reps} rep(s) per candidate)")
    collect(
        reps=reps,
        include_join_methods=include_join_methods,
        workload=workload,
        statement_timeout_ms=timeout_ms,
    )

    if skip_training:
        report["trained"] = False
        return report

    print("==> training")
    metrics = train(
        model_path=f"models/plan_selector_{db}.pkl",
        eval_path=f"models/eval_results_{db}.json",
    )
    report["trained"] = True
    report["metrics"] = metrics
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=25, help="workload size to generate")
    parser.add_argument("--reps", type=int, default=3, help="executions per candidate (median-aggregated)")
    parser.add_argument("--join-methods", action="store_true", help="also try forced join methods")
    parser.add_argument("--timeout-ms", type=int, default=20_000)
    parser.add_argument("--skip-training", action="store_true", help="collect data only")
    args = parser.parse_args()

    result = onboard(
        n_queries=args.queries,
        reps=args.reps,
        include_join_methods=args.join_methods,
        timeout_ms=args.timeout_ms,
        skip_training=args.skip_training,
    )
    print("\n" + json.dumps(result, indent=2, default=str))
