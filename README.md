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

## Point it at *any* PostgreSQL database

One command onboards a database it has never seen — it discovers the schema,
writes its own workload, collects training data, and trains:

```bash
DATABASE_URL=postgresql://user:pass@host:5432/yourdb \
    docker compose exec backend python -m app.onboard --queries 25 --reps 3
```

It reads your tables, columns, indexes and foreign keys, builds the join
graph, and generates connected multi-table queries whose predicates are
**sampled from your actual data** (real percentiles, real string values) so
they match a realistic number of rows. No foreign keys declared? It infers
join edges from naming conventions — which is how it works on the JOB/IMDB
benchmark, whose schema declares none.

Verified on two unrelated databases with the same command:

| Database | Tables | Join edges | Rows | Best selector |
|---|---|---|---|---|
| Synthetic e-commerce | 4 | 3 declared | 755k | `pairwise_rank`, **+60%** of oracle headroom |
| **Real JOB/IMDB** | 21 | 11 *inferred* | 74.2M | `risk_averse`, **+77%** of oracle headroom |

The generated workload is written to `models/workload_<db>.json` so you can
read, edit, and replay it — auto-generation is a starting point, not a claim
that those are the queries you care about.

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
                   +--> SAFETY VETO (prospective): discard picks costed
                   |    far above native
                   +--> REGRESSION GUARD (retrospective): queries with a
                   |    measured history of regressing are served native
                   |
                   +--> execute served plan, log to plan_execution_log
                   |                              |
                   |        feedback accumulates <+
                   |                |
                   |     app.retrain: train challenger, score vs. champion
                   |     on shared held-out set, PROMOTE ONLY IF CLEARLY
                   |     BETTER; versioned, with rollback
                   |
             React dashboard <-- baseline vs. chosen, why it was chosen,
             latency chart, model health, guard state, /stats/trend
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
docker compose exec backend pytest                         # backend test suite (155 tests)
```

Compare selection policies, or the regression guard, directly:

```bash
docker compose exec backend python -m app.benchmark --policy risk_averse
docker compose exec backend python -m app.benchmark --policy pairwise_rank
docker compose exec backend python -m app.benchmark --no-guard      # A/B the guard
```

Drive the self-learning loop (also available as buttons on the dashboard):

```bash
docker compose exec backend python -m app.calibrate --apply    # measure + apply the best confidence gate
docker compose exec backend python -m app.retrain --status     # deployed version, unlearned feedback
docker compose exec backend python -m app.retrain              # retrain if enough new data, gate, maybe promote
docker compose exec backend python -m app.retrain --rollback   # restore the previous model version
```

Open `http://localhost:5173` for the dashboard: paste a query (or pick a
sample), see baseline vs. chosen plan side by side, a latency chart per
candidate, and historical accuracy trending across every run so far.

### HTTP API

| Endpoint | Purpose |
|---|---|
| `POST /query/optimize` | **Production path** — plans N candidates on estimates, executes only the chosen one |
| `POST /query/analyze` | Demo/measurement path — executes *every* candidate so they can be compared |
| `GET /schema` | What it discovered about the current database (tables, join edges, declared vs. inferred) |
| `GET /stats/regret` | Cumulative regret vs. native Postgres — `<1.0` means the optimizer is earning its keep |
| `GET /stats/trend` | Latency history for the dashboard |
| `GET /model/status` | Deployed version, unlearned feedback, blocked queries |
| `POST /model/retrain` / `POST /model/rollback` | Drive the learning loop |

The distinction between the first two matters. `/query/analyze` runs every
candidate, which costs N executions to answer one question — useful for
measuring, useless for serving. `/query/optimize` plans all candidates with
`EXPLAIN` (no `ANALYZE`, so nothing runs) and executes only the winner.
Measured on this workload: **3.8 ms of planning overhead against 138 ms of
execution**, i.e. ~2.7%.

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
│       ├── retrain.py            self-learning loop: champion/challenger gate
│       ├── model_store.py        versioned models, promotion, rollback
│       └── optimizer/
│           ├── hints.py          join-order + join-method candidate generation
│           ├── features.py       Phase 2: schema-agnostic feature vectors
│           ├── plan_tree.py      plan-TREE structural encoding (Neo/Bao-inspired)
│           ├── bandit.py         bootstrapped ensemble: Thompson sampling + uncertainty
│           ├── ranker.py         Lero-style pairwise learning-to-rank
│           ├── regression_guard.py  per-query retrospective regression blocking
│           └── learned.py        plan selection + prospective safety veto
├── frontend/                 Phase 5: React (Vite) + Recharts dashboard
└── docs/
    ├── ROADMAP.md             phase-by-phase plan
    └── WRITEUP.md             literature review, results, limitations
```

## Does it actually beat Postgres?

**On the synthetic schema, yes** -- every live run beats native PostgreSQL,
averaging **~26% of the available oracle headroom with no regressions**
(`docs/WRITEUP.md` §2.2).

**On the real JOB/IMDB benchmark, no** -- and that result is more
informative. There, 75% of latency is provably available (native 3703 ms vs.
oracle 919 ms on held-out queries) and this system captures none of it,
because 194 executions across 8 queries cannot teach a 17-table join space.
`docs/JOB_RESULTS.md` quantifies the gap rather than hiding it.

**It no longer loses to native.** The original design forced the optimizer
to deviate from PostgreSQL on *every* query — the native plan was never
something the model could choose — so on the many queries Postgres already
got right, deviating could only lose. Native is now a first-class candidate,
the model predicts **speedup relative to native** rather than absolute
milliseconds, and it only deviates when the predicted win exceeds its own
uncertainty. Live runs went from `+40%, −25%, −149%` to `+14%, +42%, +1%` —
all positive. §2.4.1 documents the two attempts that failed first.

**It optimizes ~44% of queries, and that's deliberate.** For the rest it
keeps PostgreSQL's plan, because the model isn't confident enough to gamble.
That threshold is *measured*, not guessed — `python -m app.calibrate` sweeps
it against your own logged outcomes:

| Setting | Deviates | Regresses | Net gain |
|---|---|---|---|
| calibrated | 63% | **0%** | **+14.5%** |
| forced to act more | 87% | 15% | +13.9% |

Making it optimize more queries produces *less* net improvement and starts
regressing — the extra activity is all bets the model wasn't sure about.

**Robustness mattered more than prediction quality.** A per-query regression
guard — blocking the learned path for queries with a measured history of
running slower — moved mean captured headroom from **+2% to +29%** across
paired runs and eliminated the negative runs (§2.4).

Three findings behind those numbers are worth more than the numbers:

- **Per-query instability was mostly a labelling artefact.** Training on
  single executions produced live runs ranging from +40% to **-149%**.
  Aggregating each candidate's repeated executions to their *median* before
  training removed every regression -- same model, better labels. Recent
  research treats that instability as the main barrier to deploying learned
  optimizers; a meaningful share of it here was measurement noise.
- **A bug made an entire earlier round of results meaningless.**
  `pg_hint_plan` was silently ignoring every hint (it needs
  `shared_preload_libraries`, and the hint must precede `EXPLAIN`), so all
  "candidates" were the same plan as native and the "improvements" were
  timing noise. `docs/WRITEUP.md` §2.0 documents why it was silent and the
  regression test that now guards it.

## Known limitations, on purpose

`docs/WRITEUP.md` has the full list (candidate sampling above 5 tables,
executing every candidate in the demo endpoint, prediction error exceeding
available headroom, a cost-based rather than learned safety veto, no
automatic retraining, self-joins collapsing to one feature slot) -- naming
these clearly is worth more in a viva than pretending they don't exist.
