# Learned Query Optimizer

A learned plan-picking layer on top of PostgreSQL. Instead of trusting
PostgreSQL's cost-based optimizer on its own, this generates alternative join
orders and join methods using query hints, trains a model on real execution
times to choose between them, and measures the result against plain
PostgreSQL. It ships with a dashboard so you can watch it work, and a writeup
comparing it to the published research.

Every phase in `docs/ROADMAP.md` (0 through 5) is built, plus both stretch
goals: join-*method* selection and a real Join Order Benchmark (JOB/IMDB)
import. `docs/WRITEUP.md` has the literature review, the results, and an
honest list of what does not work.

## Documentation

| Document | What it covers |
|---|---|
| [`docs/DASHBOARD.md`](docs/DASHBOARD.md) | How to read every panel in the UI |
| [`docs/METRICS.md`](docs/METRICS.md) | How each number is worked out, and why |
| [`docs/WRITEUP.md`](docs/WRITEUP.md) | Literature review, results, limitations |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | The phase-by-phase plan |
| [`docs/JOB_RESULTS.md`](docs/JOB_RESULTS.md) | Results on the real JOB/IMDB benchmark |

## The benchmark is the bottleneck (read this first)

The most important measurement here: for every query, run *every* candidate
plan and record the fastest. That gives the **oracle ceiling**, the best any
plan-picker could possibly do.

| | v1 synthetic | TPC-H | JOB/IMDB |
|---|---|---|---|
| Mean best-possible gain | **6.5%** | ~23% | **~75%** |
| Queries with >5% available | 7 / 25 | — | 3 / 4 |
| Best model's top-1 accuracy | **0%** | — | 25% |

On v1, PostgreSQL was already optimal for 18 of 25 queries, and **none of six
model types** (LightGBM, random forest, extra trees, gradient boosting, ridge,
MLP) picked the fastest plan even once. That is not six failed models. There
was no signal to learn. Every earlier attempt to fix "cost isn't going down"
by tuning thresholds and swapping models was working on the wrong variable.

So `data/schema.sql` was rebuilt to be *hard*. It targets PostgreSQL's
independence assumption: it estimates `WHERE a=x AND b=y` as sel(a) × sel(b),
which is only correct when the columns are unrelated.

```
city -> country          brand -> category
price_band ~ category    channel ~ status
```

| | v1 | **v2** |
|---|---|---|
| Mean best-possible gain | 6.5% | **22.2%** |
| Queries with >5% available | 7/25 | **14/23** |

The biggest gains land exactly on the traps that were designed in:
`brand→category` **95.4%**, `city→country` **63.1%**, and a 6-way join with
both at **93.6%**. The mechanism predicted the outcome, which is the real
evidence that the diagnosis was right. Full analysis in `docs/WRITEUP.md` §2.9.

> **If you change the action space, re-run `app.collect_data`.** Widening it
> without retraining sent 17 of 25 queries into the safety veto, because the
> model was scoring plan types it had never seen. That is train/serve skew,
> not a bug.

## Point it at *any* PostgreSQL database

One command sets up a database it has never seen. It discovers the schema,
writes its own workload, collects training data, and trains:

```bash
DATABASE_URL=postgresql://user:pass@host:5432/yourdb \
    docker compose exec backend python -m app.onboard --queries 25 --reps 3
```

It reads your tables, columns, indexes and foreign keys, builds the join
graph, and writes connected multi-table queries whose filters are **sampled
from your actual data** (real percentiles, real string values), so they match
a realistic number of rows. If the database declares no foreign keys, it works
out the join links from column names instead. That is how it runs on the
JOB/IMDB benchmark, whose schema declares none.

The same command was checked on two unrelated databases:

| Database | Tables | Join edges | Rows | Best policy |
|---|---|---|---|---|
| Synthetic e-commerce | 4 | 3 declared | 755k | `pairwise_rank`, **+60%** of oracle headroom |
| **Real JOB/IMDB** | 21 | 11 *inferred* | 74.2M | `risk_averse`, **+77%** of oracle headroom |

The generated workload is written to `models/workload_<db>.json`, so you can
read it, edit it, and replay it. Auto-generation is a starting point, not a
claim that these are the queries you care about.

## Why this design, and not a database engine from scratch

The original brief was "a small SQL storage/query engine, or a plug-in
optimizer layer on top of SQLite or Postgres." Writing a real storage engine
(B-trees, WAL, MVCC, a query planner) from scratch takes years even for a
team. It would consume the whole project and leave no room for the AI work.

Sitting on top of PostgreSQL and steering it with `pg_hint_plan` is the same
approach the research uses. The clearest example is Bao (Marcus, Negi, Mao,
Tatbul, Alizadeh, Kraska, SIGMOD 2021, https://arxiv.org/abs/2004.03814),
which treats learned query optimization as choosing among a small set of
Postgres hints rather than replacing the optimizer. It is far better scoped
for one person in a year, and the hard, original part (learning to choose
well) is still entirely yours to build. `docs/WRITEUP.md` compares this
project to Bao and Neo.

## How it works

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
                   +--> SAFETY VETO (forward-looking): drop picks costed
                   |    far above native
                   +--> REGRESSION GUARD (backward-looking): queries with a
                   |    measured history of regressing are served native
                   |
                   +--> run the chosen plan, log to plan_execution_log
                   |                              |
                   |        feedback accumulates <+
                   |                |
                   |     app.retrain: train a challenger, score it against
                   |     the champion on the same held-out set, DEPLOY ONLY
                   |     IF CLEARLY BETTER; versioned, with rollback
                   |
             React dashboard <-- what ran, why, every candidate measured,
             decision quality, model health, /stats/trend
```

Table identity and row counts are discovered at runtime, from `scan_relations`
on each EXPLAIN plan and `pg_class` statistics via `schema_introspection.py`,
rather than being hardcoded. Point `DATABASE_URL` at a different schema and
the same pipeline adapts with no code changes.

## Tech stack

- **Database**: PostgreSQL 16 + `pg_hint_plan`, built from source (see
  `postgres/Dockerfile`)
- **Backend**: FastAPI + psycopg2 + LightGBM. Falls back to scikit-learn's
  `GradientBoostingRegressor` if the LightGBM native library is missing
- **Learned part**: `backend/app/optimizer/learned.py`. A bootstrapped
  ensemble (`bandit.py`) that predicts latency *with uncertainty*, three
  selection policies (`greedy`, `thompson` for exploring, `risk_averse`), and
  a safety veto against serving a regression. Falls back to the Phase 0 cost
  heuristic when no model exists yet
- **Frontend**: React (Vite) + Recharts, in `frontend/`

## Quickstart

```bash
docker compose up --build
```

That builds Postgres with `pg_hint_plan`, loads the synthetic dataset
(`data/schema.sql`), and starts the API on `localhost:8000` and the dashboard
on `localhost:5173`.

Then run the pipeline once from the repo root to get a trained model. You can
skip straight to the dashboard if you just want to see the Phase 0 heuristic,
which works with no model at all:

```bash
docker compose exec backend python -m app.collect_data   # Phase 1: fill plan_execution_log
docker compose exec backend python -m app.train          # Phase 3: train models/plan_selector.pkl
docker compose exec backend python -m app.benchmark      # Phase 4: native vs. learned, full workload
docker compose exec backend pytest                       # backend test suite (200 tests)
```

Compare policies, or the regression guard, directly:

```bash
docker compose exec backend python -m app.benchmark --policy risk_averse
docker compose exec backend python -m app.benchmark --policy pairwise_rank
docker compose exec backend python -m app.benchmark --no-guard      # A/B the guard
```

Drive the learning loop (the dashboard has buttons for these too):

```bash
docker compose exec backend python -m app.calibrate --apply    # measure and apply the best confidence gate
docker compose exec backend python -m app.retrain --status     # deployed version, pending feedback
docker compose exec backend python -m app.retrain              # retrain if there is enough new data
docker compose exec backend python -m app.retrain --rollback   # go back to the previous version
```

Open `http://localhost:5173`. Paste a query or pick a sample, and you get
PostgreSQL's plan against the chosen one, every candidate measured, why the
decision went the way it did, and the history across every run so far.
[`docs/DASHBOARD.md`](docs/DASHBOARD.md) walks through each panel.

### HTTP API

| Endpoint | Purpose |
|---|---|
| `POST /query/optimize` | **The real path.** Plans N candidates on estimates, runs only the chosen one |
| `POST /query/analyze` | Measuring path. Runs *every* candidate so they can be compared |
| `GET /schema` | What it found in the current database (tables, join edges, declared vs. inferred) |
| `GET /stats/trend` | Served vs. native history as matched pairs: overall, per day, per query, plus decision quality |
| `GET /stats/cost-model` | How well PostgreSQL's cost estimates predict real time |
| `GET /stats/regret` | Cumulative regret. Below 1.0 means the optimizer is earning its keep |
| `GET /model/status` | Deployed version, pending feedback, blocked queries |
| `POST /model/retrain` / `POST /model/rollback` | Drive the learning loop |
| `GET /advisor` | Schema-wide fixes, currently unindexed foreign keys |

The difference between the first two matters. `/query/analyze` runs every
candidate, so it costs N executions to answer one question. That is useful for
measuring and useless for serving. `/query/optimize` plans all candidates with
`EXPLAIN` (no `ANALYZE`, so nothing runs) and executes only the winner.
Measured on this workload: **3.8 ms of planning against 138 ms of execution**,
about 2.7%.

## JOB/IMDB stretch goal

`data/job/` loads the real 21-table IMDB dataset and 113 JOB queries into a
second `job` database in the same container. See `data/job/README.md` for the
download and load steps, and `docs/WRITEUP.md` §2.3 and §3 for why the
pipeline needed no code changes to support it.

## Project layout

```
learned-query-optimizer/
├── postgres/                Postgres image with pg_hint_plan + init scripts
├── data/
│   ├── schema.sql            synthetic benchmark schema + skewed seed data
│   └── job/                  JOB/IMDB stretch goal: loader + docs
├── backend/
│   ├── tests/                 pytest suite
│   └── app/
│       ├── main.py               FastAPI endpoints
│       ├── stats.py              paired served-vs-native reporting, decision quality
│       ├── db.py                 connection handling
│       ├── logging_store.py      the feedback table and query fingerprints
│       ├── plan_extractor.py     EXPLAIN JSON -> structured metrics
│       ├── schema_introspection.py  table row counts from pg_class, any schema
│       ├── advisor.py            index and statistics recommendations
│       ├── workload.py           Phase 1: the 25-query benchmark workload
│       ├── collect_data.py       Phase 1: offline collection -> plan_execution_log
│       ├── train.py              Phase 3: trains and evaluates the model
│       ├── benchmark.py          Phase 4: CLI, native vs. learned
│       ├── retrain.py            learning loop: champion/challenger gate
│       ├── model_store.py        versioned models, promotion, rollback
│       └── optimizer/
│           ├── hints.py          join-order + join-method candidates
│           ├── features.py       Phase 2: schema-agnostic feature vectors
│           ├── plan_tree.py      plan-TREE encoding (Neo/Bao-inspired)
│           ├── bandit.py         bootstrapped ensemble: Thompson sampling
│           ├── ranker.py         Lero-style pairwise learning-to-rank
│           ├── regression_guard.py  per-query blocking, backward-looking
│           ├── planner.py        the production path: plan N, run 1
│           ├── regret.py         cumulative regret
│           └── learned.py        plan selection + forward-looking safety veto
├── frontend/                 Phase 5: React (Vite) + Recharts dashboard
└── docs/
    ├── DASHBOARD.md          how to read every panel
    ├── METRICS.md            how each number is worked out
    ├── ROADMAP.md            phase-by-phase plan
    ├── WRITEUP.md            literature review, results, limitations
    └── JOB_RESULTS.md        results on the real JOB/IMDB benchmark
```

## Does it actually beat Postgres?

**On the synthetic schema, yes.** Every live run beats native PostgreSQL,
averaging about **26% of the available oracle headroom with no regressions**
(`docs/WRITEUP.md` §2.2).

**On the real JOB/IMDB benchmark, no**, and that result is more useful. There,
75% of the latency is provably available (native 3703 ms against an oracle of
919 ms on held-out queries) and this system captures none of it, because 194
executions across 8 queries cannot teach a 17-table join space.
`docs/JOB_RESULTS.md` measures the gap rather than hiding it.

**It no longer loses to native.** The original design forced the optimizer to
deviate on *every* query. The native plan was never something the model could
choose, so on the many queries Postgres already got right, deviating could
only lose. Native is now a candidate like any other, the model predicts
**speedup relative to native** rather than absolute milliseconds, and it only
switches when the predicted win is bigger than its own uncertainty. Live runs
went from `+40%, −25%, −149%` to `+14%, +42%, +1%`, all positive. §2.4.1
documents the two attempts that failed first.

**It optimizes about 44% of queries, and that is deliberate.** For the rest it
keeps PostgreSQL's plan, because the model is not confident enough to gamble.
That threshold is *measured*, not guessed: `python -m app.calibrate` sweeps it
against your own logged results.

| Setting | Deviates | Regresses | Net gain |
|---|---|---|---|
| calibrated | 63% | **0%** | **+14.5%** |
| forced to act more | 87% | 15% | +13.9% |

Making it optimize more queries produces *less* net improvement and starts
causing regressions. The extra activity is all bets the model was unsure about.

**Robustness mattered more than prediction quality.** A per-query regression
guard, which blocks the learned path for queries with a measured history of
running slower, moved mean captured headroom from **+2% to +29%** across
paired runs and removed the negative runs (§2.4).

Three findings behind those numbers are worth more than the numbers:

- **Per-query instability was mostly a labelling problem.** Training on single
  executions produced live runs ranging from +40% to **−149%**. Aggregating
  each candidate's repeated executions to their *median* before training
  removed every regression: same model, better labels. Recent research treats
  that instability as the main barrier to deploying learned optimizers, and a
  meaningful share of it here was measurement noise.
- **A bug made an entire earlier round of results meaningless.**
  `pg_hint_plan` was silently ignoring every hint. It needs
  `shared_preload_libraries`, and the hint has to come before `EXPLAIN`. So
  every "candidate" was the same plan as native, and the "improvements" were
  timing noise. `docs/WRITEUP.md` §2.0 explains why it was silent, and the
  regression test that now guards it.
- **The dashboard's own headline number was wrong in the same way.** It
  divided the average of the plans the model chose by the average of all
  PostgreSQL plans. Those two averages cover different queries, because the
  model only deviates on queries it understands, and those skew cheap. It
  reported a 97% improvement where the paired figure was about 18%.
  `docs/METRICS.md` §2 works through it.

## Known limitations, on purpose

`docs/WRITEUP.md` has the full list: candidate sampling above 5 tables,
running every candidate on the demo endpoint, prediction error exceeding the
available headroom, a cost-based rather than learned safety veto, no automatic
retraining by default, and self-joins collapsing into one feature slot. Naming
these clearly is worth more in a viva than pretending they are not there.
