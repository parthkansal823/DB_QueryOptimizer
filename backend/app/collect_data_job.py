"""
Stretch goal: run Phase 1 data collection against the real JOB/IMDB dataset
(`data/job/`) instead of the synthetic schema. Reuses `app.collect_data`'s
`collect()` unchanged -- only the workload (real JOB query files instead of
`workload.py`) and `DATABASE_URL` differ, which is exactly what the
schema-introspection work in Phase 2 was meant to make possible.

Picks the `--n` JOB queries with the fewest tables (by counting " AS "
aliases in the FROM clause) so a smoke-test run stays fast -- JOB goes up to
17-table queries, which is a stress test for the candidate-sampling
limitation (see docs/WRITEUP.md), not a quick sanity check.

Usage (from backend/, with the stack running and DATABASE_URL pointed at
the `job` database -- see data/job/README.md):
    DATABASE_URL=postgresql://postgres:postgres@postgres:5432/job \\
        python -m app.collect_data_job --n 8
"""

from __future__ import annotations

import argparse
import glob
import os
import re

from app.collect_data import collect

QUERY_DIR = "/job_queries"  # see the backend service's volume mount in docker-compose.yml
SKIP_FILES = {"schema.sql", "fkindexes.sql"}


def load_job_workload(n: int, query_dir: str = QUERY_DIR) -> list[dict]:
    paths = sorted(
        p for p in glob.glob(os.path.join(query_dir, "*.sql")) if os.path.basename(p) not in SKIP_FILES
    )
    if not paths:
        raise RuntimeError(
            f"no query files found in {query_dir} -- see data/job/README.md to clone "
            "join-order-benchmark into data/job/queries"
        )

    scored = []
    for path in paths:
        sql = open(path, encoding="utf-8").read()
        n_tables = len(re.findall(r"\bAS\b", sql, flags=re.IGNORECASE))
        scored.append((n_tables, path, sql))
    scored.sort(key=lambda row: row[0])

    return [
        {"id": os.path.basename(path).removesuffix(".sql"), "sql": sql}
        for _, path, sql in scored[:n]
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8, help="number of (smallest) JOB queries to collect")
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--timeout-ms", type=int, default=60_000)
    args = parser.parse_args()

    workload = load_job_workload(args.n)
    print(f"Selected {len(workload)} JOB queries: {[q['id'] for q in workload]}")
    collect(
        reps=args.reps,
        include_join_methods=False,  # keep the candidate set small against 10M+ row tables
        workload=workload,
        statement_timeout_ms=args.timeout_ms,
    )
