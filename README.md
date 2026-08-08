# Learned Query Optimizer

A learned plan-selection layer on top of PostgreSQL: instead of trusting
Postgres's built-in cost-based optimizer alone, this generates alternative
join orders (and join methods) via query hints, trains a model on real
execution latency to pick among them, and benchmarks the result against
native Postgres -- with a dashboard to see it happen and a writeup comparing
it to the published learned-optimizer literature.

All phases in `docs/ROADMAP.md` (0 through 5) are implemented, plus both
stretch goals: join-*method* selection and a real Join Order Benchmark
(JOB/IMDB) import. See `docs/WRITEUP.md` for the literature review, results,
and an honest limitations section.

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
part -- learning to pick well -- is still fully yours to build. See
`docs/WRITEUP.md` for how this project's approach compares to Bao and Neo.

## Architecture

```
Query in --> FastAPI backend --> baseline plan (native Postgres EXPLAIN)
                   |
                   +--> candidate join orders + join methods (pg_hint_plan hints)
                   |
                   +--> featurize: per-table slots + plan-TREE structure
                   |
                   +--> bootstrapped ensemble predicts latency + uncertainty
                   |      policy: greedy | thompson (explore) | risk_averse
                   |
                   +--> SAFETY VETO: discard picks costed far above native
                   |
                   +--> execute served plan, log to plan_execution_log
                   |
             React dashboard <-- baseline vs. chosen, why it was chosen,
             latency chart, historical accuracy (/stats/trend)
```

Table identity and reference cardinalities are discovered at runtime
(`scan_relations` off each EXPLAIN plan, `pg_class` stats via
`schema_introspection.py`) rather than hardcoded -- point `DATABASE_URL` at
a different schema (the JOB/IMDB stretch goal, or any other dataset) and
the same pipeline code adapts with no changes.

## Tech stack

- **Database**: PostgreSQL 16 + `pg_hint_plan` (built from source, see
  `postgres/Dockerfile`)
- **Backend**: FastAPI + psycopg2 + LightGBM (falls back to scikit-learn's
  `GradientBoostingRegressor` if the LightGBM native lib is unavailable)
- **Learned component**: `backend/app/optimizer/learned.py` -- a bootstrapped
  ensemble (`bandit.py`) giving latency predictions *with uncertainty*, three
  selection policies (`greedy` / `thompson` for exploration / `risk_averse`),
  and a safety veto against serving regressions. Falls back to the Phase 0
  cost heuristic when no model exists (cold start)
- **Frontend**: React (Vite) + Recharts, `frontend/`

## Quickstart

```bash
docker compose up --build
```

Builds Postgres with `pg_hint_plan`, seeds the synthetic dataset
(`data/schema.sql`), and starts the FastAPI backend on `localhost:8000` and
the React dashboard on `localhost:5173`.

Then, from the repo root, run the full pipeline once to get a trained model
(skip straight to the dashboard if you just want to see the Phase 0
heuristic in action -- it works with no model too):

```bash
docker compose exec backend python -m app.collect_data   # Phase 1: populate plan_execution_log
docker compose exec backend python -m app.train           # Phase 3: train + evaluate models/plan_selector.pkl
docker compose exec backend python -m app.benchmark        # Phase 4: native vs. learned, full workload
docker compose exec backend pytest                         # backend test suite
```

Open `http://localhost:5173` for the dashboard: paste a query (or pick a
sample), see baseline vs. chosen plan side by side, a latency chart per
candidate, and historical accuracy trending across every run so far.

`POST http://localhost:8000/query/analyze` with body `{"sql": "..."}` does
the query-analysis half over HTTP directly; `GET /stats/trend` returns the
aggregated history the dashboard's trend chart reads.

## JOB/IMDB stretch goal

`data/job/` loads the real 21-table IMDB dataset and 113 JOB queries into a
second `job` database in the same container -- see `data/job/README.md` for
the download/load steps and `docs/WRITEUP.md` Section 2.3/3 for why the
pipeline needed no code changes to support it.

## Project layout

```
learned-query-optimizer/
├── postgres/                Postgres image w/ pg_hint_plan + init scripts
├── data/
│   ├── schema.sql            synthetic benchmark schema + skewed seed data
│   └── job/                  JOB/IMDB stretch goal: loader + docs
├── backend/
│   ├── tests/                 pytest suite (hints, features, learned, plan_extractor, schema_introspection)
│   └── app/
│       ├── main.py               FastAPI endpoints (/query/analyze, /stats/trend)
│       ├── db.py                 connection handling
│       ├── plan_extractor.py     EXPLAIN JSON -> structured metrics (incl. scan_relations)
│       ├── schema_introspection.py  table cardinalities from pg_class, any schema
│       ├── workload.py           Phase 1: the 25-query benchmark workload
│       ├── collect_data.py       Phase 1: offline data collection -> plan_execution_log
│       ├── train.py              Phase 3: trains + evaluates models/plan_selector.pkl
│       ├── benchmark.py          Phase 4: CLI, native vs. learned, latency table
│       └── optimizer/
│           ├── hints.py          join-order + join-method candidate generation
│           ├── features.py       Phase 2: schema-agnostic feature vectors
│           └── learned.py        plan selection (trained model, heuristic cold-start fallback)
├── frontend/                 Phase 5: React (Vite) + Recharts dashboard
└── docs/
    ├── ROADMAP.md             phase-by-phase plan
    └── WRITEUP.md             literature review, results, limitations
```

## Does it actually beat Postgres? Not yet -- and that's the finding

Honest answer, with numbers, in `docs/WRITEUP.md` §2: **no**. Across repeated
runs the learned selector does not reliably beat native Postgres, and
run-to-run variance exceeds the effect size. The diagnosis is concrete
rather than hand-wavy: the model's prediction error (~44 ms MAE) is roughly
six times larger than the total headroom available between native Postgres
and a perfect oracle (~7.5 ms/query), so no selection policy sitting on top
of it can resolve the differences it is being asked to judge. §2.3 lays out
what would have to change (far more repetitions per candidate, a quiesced
measurement environment, and a workload where Postgres's cardinality
estimates are actually wrong).

That conclusion only became visible after fixing a bug that had made every
earlier number meaningless -- `pg_hint_plan` was silently ignoring every
hint, so all "candidates" were the same plan as native and the measured
"improvements" were timing noise. `docs/WRITEUP.md` §2.0 documents the bug,
why it was silent, and the regression test that now guards it. The oracle
baseline added to `app.train` is what makes "3% better than native"
interpretable at all.

## Known limitations, on purpose

`docs/WRITEUP.md` has the full list (candidate sampling above 5 tables,
executing every candidate in the demo endpoint, prediction error exceeding
available headroom, a cost-based rather than learned safety veto, no
automatic retraining, self-joins collapsing to one feature slot) -- naming
these clearly is worth more in a viva than pretending they don't exist.
