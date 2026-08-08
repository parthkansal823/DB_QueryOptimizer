# Learned Query Optimizer

A learned plan-selection layer on top of PostgreSQL: instead of trusting
Postgres's built-in cost-based optimizer alone, this generates alternative
join orders via query hints, and (starting in Phase 3 of the roadmap) uses
a trained model to pick among them -- then benchmarks the result against
native Postgres.

## Why this design, and not a from-scratch database engine

The original framing was "a small SQL storage/query engine, or a plug-in
optimizer layer on top of SQLite/Postgres." Writing a real storage engine
-- B-trees, WAL, MVCC, a query planner -- from scratch is a multi-year
undertaking even for a small team; it would eat the whole year and leave
no room for the actual AI contribution.

Sitting on top of PostgreSQL and steering it via `pg_hint_plan` is the
same approach real research uses -- most notably Bao (Marcus, Negi, Mao,
Tatbul, Alizadeh, Kraska -- SIGMOD 2021, https://arxiv.org/abs/2004.03814),
which frames learned query optimization as picking among a small set of
Postgres plan hints rather than replacing the optimizer outright. It's
dramatically more scoped for one person in a year, and the hard, novel
part -- learning to pick well -- is still fully yours to build.

## Architecture

```
Query in --> FastAPI backend --> baseline plan (native Postgres EXPLAIN)
                   |
                   +--> candidate join orders (pg_hint_plan hints)
                   |
                   +--> LearnedOptimizer picks one
                   |
                   +--> execute chosen plan, log latency
                   |
             React dashboard <-- compare baseline vs. chosen, over time
             (Phase 5 -- not scaffolded yet)
```

## Tech stack

- **Database**: PostgreSQL 16 + `pg_hint_plan` (built from source, see
  `postgres/Dockerfile`)
- **Backend**: FastAPI + psycopg2
- **Learned component**: starts as a heuristic stub
  (`backend/app/optimizer/learned.py`), designed so a trained
  scikit-learn/LightGBM model can be dropped in without changing its
  interface
- **Frontend**: React -- Phase 5 in `docs/ROADMAP.md`, not built yet

## Quickstart

```bash
docker compose up --build
```

This builds Postgres with `pg_hint_plan`, seeds it with the synthetic
dataset in `data/schema.sql`, and starts the FastAPI backend on
`localhost:8000`.

Then, inside the backend container (or locally with `DATABASE_URL`
pointed at `localhost:5432`):

```bash
docker compose exec backend python -m app.benchmark
```

This runs the sample query through both the native optimizer and the
hint-based candidate path and prints a latency comparison.

`POST http://localhost:8000/query/analyze` with body `{"sql": "..."}`
does the same thing over HTTP and returns the full baseline plan, every
candidate plan, and which one was chosen -- this is what the dashboard
will eventually call.

## Project layout

```
learned-query-optimizer/
├── postgres/              Postgres image w/ pg_hint_plan + init scripts
├── data/schema.sql         synthetic benchmark schema + seed data
├── backend/
│   └── app/
│       ├── main.py               FastAPI endpoints
│       ├── db.py                 connection handling
│       ├── plan_extractor.py     EXPLAIN JSON -> structured metrics
│       ├── benchmark.py          CLI: baseline vs. learned, latency table
│       └── optimizer/
│           ├── hints.py          generates pg_hint_plan candidates
│           └── learned.py        plan selection (heuristic now, model later)
└── docs/ROADMAP.md         phase-by-phase plan for the rest of the year
```

See `docs/ROADMAP.md` for what comes next.

## A known limitation, on purpose

`generate_join_order_candidates` enumerates every permutation for small
queries but randomly samples for larger ones (permutations explode
factorially). Naming this limitation clearly -- and explaining why a
*learned* candidate generator would be a natural next step -- is worth
more in a viva than pretending it doesn't exist.
