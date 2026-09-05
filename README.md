# Learned Query Optimizer

A learned plan-picking layer on top of PostgreSQL. Instead of trusting
PostgreSQL's cost-based optimizer on its own, this generates alternative join
orders and join methods using query hints, trains a model on real execution
times to choose between them, and measures the result against plain
PostgreSQL. It ships with a dashboard so you can watch it work, and a writeup
comparing it to the published research.

It runs on any PostgreSQL database, ships with a dashboard that explains every
decision it makes, and refuses to act when it isn't confident. Built end to end:
data collection, feature engineering, model training, a self-retraining loop
with rollback, two independent safety mechanisms, a conformally calibrated
safety gate, and **1,414 tests** behind CI (1,272 backend, of which 18 run
against a live PostgreSQL with `pg_hint_plan` actually loaded, plus 142
frontend).

## Documentation

| Document | What it covers |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | **Start here** — how the whole system works |
| [`docs/DASHBOARD.md`](docs/DASHBOARD.md) | How to read every panel in the UI |
| [`docs/METRICS.md`](docs/METRICS.md) | How each number is worked out, and why |
| [`docs/WRITEUP.md`](docs/WRITEUP.md) | Design notes, results, comparison to published work |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | How it was built, stage by stage |
| [`PENDING.md`](PENDING.md) | **What is not finished** — every open item, with the measurement that says why it matters |
| [`docs/JOB_RESULTS.md`](docs/JOB_RESULTS.md) | Results on the real JOB/IMDB benchmark |
| [`docs/CONFIDENCE.md`](docs/CONFIDENCE.md) | **How the optimizer decides whether to trust itself** — and the two fixes that were measured and reverted |
| [`docs/DATABASE_OPTIMIZATION.md`](docs/DATABASE_OPTIMIZATION.md) | Fixing the *database* rather than one query: correlated columns, server settings, index and table health |
| [`docs/TECHNIQUES_BACKLOG.md`](docs/TECHNIQUES_BACKLOG.md) | The PostgreSQL manual read against this system — what was built from it, and what the move to PostgreSQL 18 unblocked. Nothing in it is blocked on the server any more |
| [`famalies.txt`](famalies.txt) | Every hint family in the action space, with measured evidence that each one binds |
| [`.claude/hooks/`](.claude/hooks/) | Ruff on every backend edit, the unit suite in the background, and a gate on `git commit` |

> **Read `docs/WRITEUP.md` §2.4.0 before the numbers below.** The seed data in
> `data/schema.sql` had a bug that gave every categorical column exactly one
> distinct value — every user in London, every product one brand — so the
> correlated columns this schema exists to provide had no correlation in them.
> It is fixed, and the measurements taken before it was found are measurements
> of a database that did not contain the phenomenon they were about.

## The benchmark is the bottleneck (read this first)

The most important measurement here: for every query, run *every* candidate
plan and record the fastest. That gives the **oracle ceiling**, the best any
plan-picker could possibly do.

| | v1 synthetic | TPC-H | JOB/IMDB |
|---|---|---|---|
| Mean best-possible gain | **6.5%** | ~23% | **~82%** |
| Queries with >5% available | 7 / 25 | — | 5 / 8 |
| Best model's top-1 accuracy | **0%** | — | **38%** |

The JOB column is the 2026-09-05 re-run (8 held-out queries, `docs/JOB_RESULTS.md`);
it read 25% top-1 on the original 8-query collection.

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

> **The same applies to plan *shapes*, and that took longer to notice.** The
> workload was 24 queries all of one shape — a select over joins with filters.
> None aggregated, sorted, de-duplicated or used a subquery, so no training plan
> ever contained a `HashAggregate`, a `WindowAgg` or a semi-join, and the
> `n_aggregate` / `n_sort` features were near-zero in every row the model had
> seen. A feature that never varies teaches nothing. The workload is now 32,
> with a `shape_` group covering aggregation, DISTINCT, top-N, a window
> function, and semi- and anti-joins.

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

| Database | Tables | Join edges | Rows | Best policy, on the workload it generated |
|---|---|---|---|---|
| Synthetic e-commerce | 4 | 3 declared | 755k | `pairwise_rank`, **+60%** of oracle headroom |
| **Real JOB/IMDB** | 21 | 11 *inferred* | 74.2M | `risk_averse`, **+77%** of oracle headroom |

**Read that second row narrowly.** It says the generator produced a workload on
a 21-table schema with no declared foreign keys, and that one policy did well on
*those* queries — native averages 210.9 ms on them. That is not the Join Order
Benchmark; it is a workload this system wrote for itself on the same tables. For
the real JOB queries, see the next section.

The generated workload is written to `models/workload_<db>.json`, so you can
read it, edit it, and replay it. Auto-generation is a starting point, not a
claim that these are the queries you care about.

**And a trained model carries across schemas, which was never tested until
2026-09-05.** A quarter of the feature vector is per-table slots that read
*zero* on a database the model has not seen, so this measures whether the
schema-agnostic half — costs, tree shape, join-method counts, predicate shape —
is enough on its own:

| carried model | target | headroom captured | model trained on the target |
|---|---|---|---|
| synthetic (6 tables, 755k rows) | JOB/IMDB (21 tables, 73M rows) | **97.0%** | 98.4% |
| JOB/IMDB | synthetic | 44.1% | 69.3% |

Zero overlapping tables in both directions; cost-argmin captures 12.8% and
12.1% on the same queries. The caveat is in `docs/WRITEUP.md` §2.64: total
latency captured is not per-query accuracy, and how far a carried model gets
depends on whether the target's headroom is concentrated enough to reach without
local knowledge.

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
                   |    + resource/cost-model settings (work_mem, parallelism)
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

## What it does, in plain terms

Five things, in the order a query meets them.

**1. It writes alternative versions of your query.** Not different SQL — the
same query with a `pg_hint_plan` comment on the front that changes how
PostgreSQL executes it. Different join orders, different join methods, operators
switched off, and settings like `work_mem` and parallel workers. About 20
alternatives per query.

**2. It predicts which one is fastest — without running them.** Each alternative
is planned with `EXPLAIN` (nothing executes) and turned into a feature vector
describing the plan's shape. A model trained on real timings predicts how each
compares to PostgreSQL's own choice.

**3. It decides whether it is confident enough to act.** This is the part that
matters most, and it is where a query optimizer earns or loses trust. Two
questions: is there an expected win, and is the worst case bounded? If either
fails, PostgreSQL's plan runs unchanged. `docs/CONFIDENCE.md` is the whole
story, including two fixes that were measured and thrown away.

**4. It protects itself from being wrong.** A query with a history of getting
slower is blocked from deviating. A query never seen before gets a stricter bar,
because nothing is watching it yet. And a deviation that overruns its time
budget is cancelled and PostgreSQL's plan run instead — so "not much worse than
native" is enforced, not just predicted.

**5. It looks at the database, not only the query.** Correlated columns the
planner does not know about, indexes nobody reads, statistics describing a table
that has since changed, server settings that are defaults rather than decisions.
Each is fixed once and helps every query afterwards, including ones this system
has never seen. See `docs/DATABASE_OPTIMIZATION.md`.

Everything it decides is inspectable:

| command | answers |
|---|---|
| `python -m app.diagnostics` | should I believe this model? |
| `python -m app.dbadvisor` | what is wrong with this database? |
| `python -m app.audit_binding` | did every hint the optimizer generates actually bind? |
| `python -m app.generalisation` | does the model predict unseen queries as well as trained ones? |

The dashboard shows the first two. The third exists because a hint that is
ignored is indistinguishable from one that did nothing, and the fourth because
the benchmark's held-out column cannot answer its question when the gate
declines to act on five of six held-out queries.

### Things worth knowing before trusting a number

- **The benchmark reports two figures**, and the second is the honest one. Most
  of the built-in workload is in the model's training set, so the headline is
  substantially a memorisation score. The held-out figure below it is the one
  that says whether the model generalises.
- **A single benchmark run cannot settle a comparison.** Repeated executions of
  the *same* plan on this database differ by **18.6%** on average.
  `python -m app.experiment` interleaves two arms, puts a bootstrap interval
  around the difference, and reports **unresolved** when the interval spans
  zero; `app.benchmark --protocol compare` repeats the workload and prints the
  spread beside the median.
- **The benchmark mutates what the next benchmark reads.** The regression guard
  accumulates history as runs happen, so two runs of an identical gate gave
  15.6% and 7.2% captured headroom, differing only in what the guard had learned
  in between. `--protocol compare` turns the guard off for exactly this reason;
  `--protocol deploy` leaves it on and reports a trend point. The two numbers
  answer different questions and must never be compared. See `docs/WRITEUP.md`
  2.30.
- **`docs/WRITEUP.md` keeps the failures.** A learned cardinality correction that
  raised the ceiling by 0.0%; a conformal bound that was correct and unusable; a
  gate that calibrated itself into doing nothing; a seed-data bug that removed
  the phenomenon the whole schema was built around. Those are in the record with
  their numbers.

## Tech stack

- **Database**: PostgreSQL 18 + `pg_hint_plan` 1.8.1, built from source (see
  `postgres/Dockerfile`). Upgraded from 16.15 / 1.6.3; `docs/WRITEUP.md` 2.29
  records what that unlocked and the safety veto it silently switched off
- **Backend**: FastAPI + psycopg2 + LightGBM. Falls back to scikit-learn's
  `GradientBoostingRegressor` if the LightGBM native library is missing
- **Learned part**: `backend/app/optimizer/learned.py`. A bootstrapped
  ensemble (`bandit.py`) that predicts latency *with uncertainty*, three
  selection policies (`greedy`, `thompson` for exploring, `risk_averse`), and
  a safety veto against serving a regression. Falls back to a cost
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
skip straight to the dashboard if you just want to see the cost heuristic,
which works with no model at all:

```bash
docker compose exec backend python -m app.collect_data   # gather training data
docker compose exec backend python -m app.train          # train models/plan_selector.pkl
docker compose exec backend python -m app.benchmark      # native vs. learned, full workload
docker compose exec backend pytest                       # 1,090 unit tests, no database needed
docker compose exec backend pytest -m integration        # 17 more: hints must actually bind
docker compose exec backend ruff check app tests         # lint, same rules CI runs
```

Settle a comparison properly, with paired runs and confidence intervals
rather than eyeballing three numbers:

```bash
docker compose exec backend python -m app.experiment --runs 20            # guard on vs off
docker compose exec backend python -m app.experiment --runs 20     --compare policy --arm-a risk_averse --arm-b pairwise_rank
```

It interleaves the arms, discards a pair if either half fails, checkpoints as
it goes, and reports a bootstrap interval plus a sign test. An interval that
spans zero is reported as **unresolved**, however good the mean looks.

Compare policies, or the regression guard, in a single run:

```bash
docker compose exec backend python -m app.benchmark --policy risk_averse
docker compose exec backend python -m app.benchmark   # --policy defaults to the served one
docker compose exec backend python -m app.benchmark --no-guard      # A/B the guard
```

### Picking the settings

`app.autotune` is the one command for "what should I set". It sweeps the
**selection policy**, the **action space** and the **safety gate** together, on
the queries the model held out, under the same safety bounds, and prints one
configuration plus the variables to export:

```
        policy           extra families     z  gain  dev%  reg%   worst    net%  spread
 pairwise_rank                     none  2.00  0.00   33%   20%   1.21x   10.5%
      thompson                     none  2.00  0.02   33%   20%   1.21x    9.6%  +7.6 to +10.0 over 5 draws
        greedy                     none  2.00  0.02   33%   20%   1.21x    9.2%
   risk_averse                     none  2.00  0.02   33%   20%   1.21x    7.3%
```

Sweeping them **together** is the point. They are not independent: a gate
threshold is a property of the selection policy it was measured against, and the
action space decides what there is to select from. `docs/WRITEUP.md` 2.43 is
what happens otherwise -- the gate was calibrated under `greedy` for the life of
the project while the server selected with `pairwise_rank`, and those land at
104.1% and 95.1% of native on the same held-out queries.

Two things it does on purpose:

- **Stochastic policies are drawn five times and scored on the median**, with
  the range printed. `thompson` samples an ensemble member per decision; three
  identical sweeps of it returned +14.16%, +10.61% and +12.33%. Comparing one
  draw against a deterministic policy's exact value is how a tuner ends up
  recommending whichever option is noisiest, and the first version of this tool
  did exactly that (2.48).
- **`--apply` writes the gate and only prints the rest.** Thresholds live in
  `models/gate.json`, which this tool owns. What a running server selects with
  is not something a tuning run should change behind you.

It is a replay, not an execution, so confirm the winner with
`python -m app.benchmark --protocol compare` -- 2.32 measured two identical
3-run benchmarks landing 8 points apart.

Drive the learning loop (the dashboard has buttons for these too):

```bash
docker compose exec backend python -m app.diagnostics           # should I believe this model?
docker compose exec backend python -m app.autotune             # what are the best settings?
docker compose exec backend python -m app.autotune --apply     # ...and write the gate they imply
docker compose exec backend python -m app.calibrate --apply    # just the gate, at fixed policy
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
| `POST /query/analyze` | Measuring path. Plans every candidate, executes `ANALYZE_EXECUTE_LIMIT` of them |
| `GET /schema` | What it found in the current database (tables, join edges, declared vs. inferred) |
| `GET /stats/trend` | Served vs. native history as matched pairs: overall, per day, per query, plus decision quality |
| `GET /stats/cost-model` | How well PostgreSQL's cost estimates predict real time |
| `GET /stats/regret` | Cumulative regret. Below 1.0 means the optimizer is earning its keep |
| `GET /model/status` | Deployed version, pending feedback, blocked queries |
| `GET /model/diagnostics` | **Why** it is behaving that way: is the gate consulting uncertainty, was it calibrated on enough queries, is the uncertainty on the right scale |
| `POST /advisor/rank` | Which missing index would actually help, measured with hypothetical indexes — no index is built |
| `POST /experiment/start` / `GET /experiment/status` | Run a paired A/B comparison with a bootstrap interval and a sign test |
| `POST /model/retrain` / `POST /model/rollback` | Drive the learning loop |
| `GET /advisor` | Schema-wide fixes, currently unindexed foreign keys |

The difference between the first two matters, though it is smaller than it
was. `/query/optimize` plans all candidates with `EXPLAIN` (no `ANALYZE`, so
nothing runs) and executes only the winner.

`/query/analyze` used to execute *every* candidate, which cost N executions to
answer one question. Measured: 19 candidates at 48 ms each was 915 ms of a
1.2 s request, while *planning* a candidate costs **0.89 ms**. The model never
needed those executions -- it predicts latency from the plan tree -- so the
endpoint now plans all of them and executes only the handful whose measured
latency the response actually reports (`ANALYZE_EXECUTE_LIMIT`, default 4).
**1.19 s to 0.19 s**, and every number that remains is still a real
measurement. `n_candidates_planned` and `n_candidates_executed` in the response
say which is which.

Deliberately not parallelised: concurrent candidates compete for CPU and shared
buffers, so every latency would be contaminated by whatever ran beside it --
and those latencies are logged as training data.

## JOB/IMDB stretch goal

`data/job/` loads the real 21-table IMDB dataset and 113 JOB queries into a
second `job` database in the same container. See `data/job/README.md` for the
download and load steps, and `docs/WRITEUP.md` §2.3 and §3 for why the
pipeline needed no code changes to support it.

## Where the data lives

Everything this project produces at runtime is inside the repo directory, so
"where is my data" has one answer you can look at, back up and delete:

| path | what | size here |
|---|---|---|
| `data/pgdata/` | **the PostgreSQL data directory** — bind-mounted by `docker-compose.yml` rather than a Docker named volume | 9.5 GB |
| `data/job/csv/` | the downloaded JOB/IMDB dataset | 4.8 GB |
| `data/pgdata_pg16/` | the **PostgreSQL 16 cluster**, kept when the project moved to 18. Not regenerable from `schema.sql`: it holds the `job` database. Reachable by pointing a `postgres:16` container at it | 9.2 GB |
| `backups/` | `pg_dump` output, including the pre-upgrade snapshot | 14 MB |
| `backend/models/` | trained bundles, the calibrated gate, generated workloads, saved queries | 23 MB |

A Docker named volume would have been the conventional choice and is the reason
this is worth stating: a named volume is invisible from the filesystem, easy to
forget, and easy to destroy with a stray `docker compose down -v`. Bind-mounting
the data directory trades a little Docker idiom for being able to see, copy and
delete the database like any other directory.

All three are in `.gitignore`. The one runtime artifact that is *not* ignored is
`backend/models/eval_results*.json` — those are the numbers the README and
`docs/WRITEUP.md` report, and a result nobody can see is a result nobody can
check.

The container's `node_modules` stays a named volume on purpose. It is a build
artifact rather than data, and it has to be the Linux build: the repo is
bind-mounted into the container, so without a volume shadowing it the container
would import modules compiled for the host platform.

## Project layout

```
learned-query-optimizer/
├── postgres/                Postgres image with pg_hint_plan + init scripts
├── data/
│   ├── schema.sql            synthetic benchmark schema + skewed seed data
│   └── job/                  JOB/IMDB stretch goal: loader + docs
├── backend/
│   ├── tests/                 pytest suite (1,090 unit + 17 integration)
│   ├── native/                optional compiled encoder; falls back to Python
│   │   ├── plan_tree_native.cpp   C++ port of encode_plan_tree
│   │   └── setup.py               build_ext --inplace
│   └── app/
│       ├── main.py               FastAPI endpoints
│       ├── stats.py              paired served-vs-native reporting, decision quality
│       ├── db.py                 connection handling
│       ├── logging_store.py      the feedback table and query fingerprints
│       ├── plan_extractor.py     EXPLAIN JSON -> structured metrics
│       ├── schema_introspection.py  table row counts from pg_class, any schema
│       ├── advisor.py            index and statistics recommendations
│       ├── index_advice.py       index shape per predicate: trigram, covering, BRIN
│       ├── diagnostics.py        model/gate health: why it behaves as it does
│       ├── generalisation.py     prediction error on unseen queries vs trained ones
│       ├── sql_errors.py         a failed query, explained, with a corrected statement
│       ├── measurement.py        fairness, prewarming, a deterministic session
│       ├── audit_binding.py      asks pg_hint_plan whether every candidate bound
│       ├── calibrate.py          sweeps the gate against your own logged results
│       ├── hypothetical.py       what an index would be worth, via HypoPG
│       ├── query_template.py     parameterised statements share one identity
│       ├── experiment_job.py     paired A/B comparisons, run from the dashboard
│       ├── workload.py           the 32-query benchmark workload
│       ├── collect_data.py       offline collection -> plan_execution_log
│       ├── train.py              trains and evaluates the model
│       ├── benchmark.py          CLI: native vs. learned; --protocol compare|deploy
│       ├── retrain.py            learning loop: champion/challenger gate
│       ├── model_store.py        versioned models, promotion, rollback
│       ├── dbadvisor/            fixing the database rather than one query
│       │   ├── indexes.py            unused, duplicate, redundant, missing BRIN
│       │   ├── maintenance.py        vacuum, analyze, visibility, HOT updates
│       │   ├── extended_stats.py     correlated columns, q-error, stats targets
│       │   ├── physical.py           bloat, fill factor, partitioning
│       │   ├── observability.py      what the server is not telling you
│       │   └── query_stats.py        pg_stat_statements
│       └── optimizer/
│           ├── hints.py          the action space: 12 hint families
│           ├── hint_binding.py   did the hint actually apply? pg_hint_plan's own answer
│           ├── catalog.py        index map by *alias*, which is how hints address relations
│           ├── prefilter.py      candidates that provably cannot differ from native
│           ├── features.py       schema-agnostic feature vectors
│           ├── plan_tree.py      plan-TREE encoding (Neo/Bao-inspired)
│           ├── bandit.py         bootstrapped ensemble: Thompson sampling
│           ├── ranker.py         Lero-style pairwise learning-to-rank
│           ├── calibration.py    point calibration fitted on out-of-bag rows
│           ├── regression_guard.py  per-query blocking, backward-looking
│           ├── planner.py        the production path: plan N, run 1
│           ├── regret.py         cumulative regret
│           ├── cardinality.py    learned q-error + Rows() corrections
│           ├── conformal.py      calibrated safety bound (out-of-bag conformal)
│           └── learned.py        plan selection + forward-looking safety veto
├── frontend/                 React (Vite) + Recharts dashboard
├── .claude/hooks/            ruff on edit, tests async, a gate on git commit
└── docs/
    ├── ARCHITECTURE.md       how the whole system works — start here
    ├── DASHBOARD.md          how to read every panel
    ├── METRICS.md            how each number is worked out
    ├── ROADMAP.md            phase-by-phase plan
    ├── CONFIDENCE.md         how the optimizer decides whether to trust itself
    ├── DATABASE_OPTIMIZATION.md   fixing the database rather than one query
    ├── TECHNIQUES_BACKLOG.md the PostgreSQL manual read against this system
    ├── WRITEUP.md            literature review, results, limitations
    └── JOB_RESULTS.md        results on the real JOB/IMDB benchmark
```

## Does it actually beat Postgres?

**On the synthetic schema, yes.** The current measurement, three runs under the
`compare` protocol:

| | captured | range |
|---|---|---|
| all 65 queries | **40.8%** | 38.3 – 46.0 |
| **queries the model never trained on** | **43.0%** | 23.8 – 48.5 |
| hand-written traps and controls only | 26.0% | 23.6 – 42.2 |

The middle row is the one that matters, and it read **0.0%** on every run ever
measured until 2026-09-04 — eleven runs across three independent sets. Nothing
in the model, the features or the action space changed to move it; twelve
corrections to the measurement and configuration did, and `docs/WRITEUP.md`
§2.42–2.54 works through each.

The third row is lower because 27 of the 65 queries were generated and *kept*
for having headroom, so the first row is partly measured on a benchmark selected
to be winnable. The **14.8-point gap** between them is the size of that effect,
and it widened from 8.3 points when six hand-written six-table queries were
added *without* a headroom filter — three of which have none, by construction.
The unbiased number went down and the filtered one did not, which is what an
honest split is supposed to do.

The ranges are wide. The benchmark prints its own resolution floor and it reads
"a difference smaller than 7.7 points is inside this sample's own spread", so
the honest claim is "clearly not zero and around a quarter to a half", not
"43.0%".

**On the real JOB/IMDB benchmark, now yes — and the shape of the win matters
more than the number.** This said "no" for most of the project's life: 75% of
the latency provably available (native 3703 ms against an oracle of 919 ms) and
**none of it captured**, because 194 executions across 8 queries could not teach
a 17-table join space.

Re-run on PostgreSQL 18 with **1614 executions across all 30** of the smallest
queries and **8 held out**, on held-out queries averaging **3960 ms** native —
the same expensive regime as the original:

| selector | avg latency | headroom captured |
|---|---|---|
| native PostgreSQL | 3960.2 ms | — |
| cost heuristic | 2285.3 ms | +42% |
| oracle (best candidate) | 695.6 ms | 100% |
| **`risk_averse`** | **713.7 ms** | **+99.4%** |
| `pairwise_rank` | 722.8 ms | +99.2% |
| `greedy` | 726.9 ms | +99.0% |
| `thompson` | 863.9 ms | +94.8% |

All four policies clear 94%, and the top three sit within 0.4 points of each
other — which matters more than which one leads. One policy at 99% invites the
question of whether it got lucky; four independent selection rules landing
within half a point of the oracle is a property of the candidate set and the
model, not of the rule reading them.

**Read the breakdown, not just the total.** On three of the eight held-out
queries it picks the **oracle plan exactly** — `17c` goes from **21.6 seconds to
2.0**, an 11x speedup, and is 76% of all the saving; a fourth (`1b`) lands within
0.1 ms of it. The other four *regress*, totalling −208 ms against 26,099 ms
saved, and three of those had almost nothing to win (2.7 ms, 14.4 ms, and one
where no candidate beats native at all).

So: half the held-out queries get slightly worse, two very expensive ones carry
the aggregate, and the trade is 26 seconds saved for 0.2 lost. Those are also
*ungated* numbers — the confidence gate exists for exactly the near-tie queries
that regress here and would decline most of them.
`docs/JOB_RESULTS.md` has the per-query table and the caveats.

**It no longer loses to native.** The original design forced the optimizer to
deviate on *every* query. The native plan was never something the model could
choose, so on the many queries Postgres already got right, deviating could
only lose. Native is now a candidate like any other, the model predicts
**speedup relative to native** rather than absolute milliseconds, and it only
switches when the predicted win is bigger than its own uncertainty. Live runs
went from `+40%, −25%, −149%` to `+14%, +42%, +1%`, all positive. §2.4.1
documents the two attempts that failed first.

**It declines most queries, and how deliberate that is depends on the
calibration set.** For every query it declines, PostgreSQL's plan runs
unchanged, because the model is not confident enough to gamble. The threshold is
*measured*, not guessed: `python -m app.calibrate` sweeps it against your own
logged results.

An earlier calibration, on a larger set, produced the trade-off this section
originally reported:

| Setting | Deviates | Regresses | Net gain |
|---|---|---|---|
| calibrated | 63% | **0%** | **+14.5%** |
| forced to act more | 87% | 15% | +13.9% |

Making it optimize more produced *less* net improvement and started causing
regressions -- the extra activity was all bets the model was unsure about.

**That is not what the current calibration does, and the difference is worth
stating plainly.** The most recent sweep ran on **8 held-out queries**, over a
grid whose operating points differed by 0.4 percentage points of net gain. It
chose a bar of `min_relative_gain = 0.10`, and the measured consequence across
eleven benchmark runs was a held-out deviation rate of **exactly zero** — the
same 0.0% every run.

Those eight queries were not short of opportunity. Five of them have more than
**20%** available between PostgreSQL's plan and the best candidate the optimizer
can already reach, up to **43.4%** on a six-way join. The system took none of it.

A calibration asked to choose between options separated by less than its own
resolution will pick the one that does nothing, because doing nothing is the
only option whose downside it can bound. So the honest reading of a low
deviation rate here is **a property of the calibration set size, not a measured
statement about the model** — `docs/WRITEUP.md` §2.37 has the working, including
an A/B of a looser bar that came back *unresolved* rather than better.

**Correcting the estimate is not the same as having somewhere to go.** The
newest addition learns how wrong PostgreSQL's *join* row estimates are and
feeds corrections back with `pg_hint_plan`'s `Rows()` hint, so the planner
searches with a better premise instead of being overridden. The mechanism is
verified (a query estimated at 4 rows reports 400 under `Rows(o u *100)`) and
it beats native by up to 83% on the correlation traps. It also raises the
oracle ceiling by **0.0%**, because every one of those wins was already
reachable through the existing hints. Kept for the JOB-scale case it was built
for, disabled with `ENABLE_ROWS_CORRECTION=0`, and written up as the null
result it is (§2.8.1).

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

## Read the benchmark's second number, not its first

`app.benchmark` reports the headline capture rate and then, separately, the
same figure restricted to queries the model **never trained on**:

```
=== totals across 24 queries ===
captured 52.7% of the 393 ms available headroom

--- of those, 11 the model never trained on ---
captured 52.7% of the 247 ms available headroom
this is the number that says whether the model generalises;
the one above largely says how well it memorised.
```

Most of this workload is in the model's training set — 21 of 24 on that run —
so the headline is substantially a memorisation score. `docs/WRITEUP.md` §2.2.1
already documents offline evaluation being optimistically biased; this is that
bias in the project's own headline metric, and the split is the same correction
`app.stats` made when it stopped dividing two averages over different
populations.

On this run the two agree, which is the result worth having: the model
generalises to queries it has not seen.

## Known limitations

`docs/WRITEUP.md` §4 keeps an honest list, split into what has been fixed and
what has not. Recently closed: candidate generation now follows the query's
join graph instead of sampling blindly (usable candidates went from 48% to
100%), self-joins no longer collapse into a single feature slot, and data
collection defaults to the three repetitions the evidence calls for.

Also closed: the regression guard now raises the confidence bar on queries it
has no history for, instead of leaving first executions unprotected.

Still open: prediction error is comparable to the available headroom at this
data scale, the safety veto reasons about cost estimates rather than learned
ones, and all latency numbers come from a single-connection, otherwise-idle
database.

### Optional settings

Every row below is also editable from the dashboard, and a change is written
to `models/settings.json`. **That file is now read at startup.** It previously
was not: the settings panel displayed one configuration while the process ran
another, so a restart silently reverted every change — including a *stricter*
safety bar, which came back as the calibrated `0.0` and let the optimizer
deviate on any predicted win at all.

| Variable | Default | Effect |
|---|---|---|
| `SELECTION_POLICY` | `pairwise_rank` | `greedy`, `thompson`, `risk_averse`, `pairwise_rank`. The default lives in `learned.DEFAULT_SELECTION_POLICY` and **nothing else restates it** -- `docker-compose.yml` used to, on the backend service only, so every host-run tool measured `greedy` instead (`docs/WRITEUP.md` 2.43) |
| `GATE_MODE` | from `models/gate.json`, else `fixed_z` | `fixed_z` (mean + z·sigma) or `conformal` (measured bound, see below). `app.calibrate` sweeps both and writes the winner; setting this overrides that |
| `CONFORMAL_ALPHA` | `0.1` | Risk level for the conformal gate; applied when the model trains |
| `INCLUDE_JOIN_METHODS` | `0` (off) | Serve forced `HashJoin`/`NestLoop`/`MergeJoin` candidates as well as join orders |
| `INCLUDE_TUNING` | `1` | Resource/cost-model candidates: parallel workers, `work_mem`, `random_page_cost`, `join_collapse_limit` |
| `MAX_REGRESSION_RATIO` | `1.25` | Ceiling on the pessimistic case when deviating — the risk half of the gate |
| `RUNTIME_TIMEOUT_FACTOR` | `3.0` | Abandon a deviation once it has taken this many times what native costs |
| `GROUP_BY_TEMPLATE` | `1` | Treat `city='Mumbai'` and `city='London'` as one query |
| `ADAPTIVE_CANDIDATE_BUDGET` | `1` | Plan fewer candidates for queries PostgreSQL already prices as cheap |
| `GUARD_TOLERANCE` | `0.10` | How much slower than native a query may average before the guard blocks it |
| `GUARD_MIN_OBSERVATIONS` | `3` | Served executions needed before the guard will judge a query |
| `AUTO_RETRAIN_SECONDS` | `0` (off) | Retrain on a background interval; promotion is still gated |
| `ENABLE_ROWS_CORRECTION` | `1` | Learned `Rows()` cardinality corrections as an extra candidate |
| `INCLUDE_BUSHY` | `0` (off) | Bushy plan trees. Measured against the served space on all 24 queries: **zero wins** |
| `INCLUDE_DISABLE_INDEX` | `0` (off) | `DisableIndex(alias index)`: plan as if an index were dropped. Needs pg_hint_plan >= 1.8 |
| `INCLUDE_INDEX_PINNING` | `0` (off) | Pin scans to named indexes, read from the catalog |
| `ANALYZE_EXECUTE_LIMIT` | `4` | How many candidates `/query/analyze` executes for its measured comparison |
| `FRONTEND_ORIGIN` | `http://localhost:5173` | CORS origin the API accepts |

`GUARD_TOLERANCE` and `GUARD_MIN_OBSERVATIONS` were declared settings that
nothing read; the guard kept its constructor defaults whatever they were set to.

## The safety gate asks two questions, not one

The gate decides whether to deviate from PostgreSQL. It used to ask a single
question — would the *pessimistic* case still win?

```
predicted_ratio + z · sigma  <  1 - min_relative_gain
```

That reads as "even if the model is wrong, this still beats native", which is a
much stronger demand than it looks and throws away the best bets the system
has. A measured example from the dashboard: a six-table join where the model
predicted **0.20×** native and the truth was 0.16× — a **5.9× speedup**, with
the point estimate almost exactly right. The ensemble happened to disagree with
itself on that query, so the pessimistic figure came out at 1.13×, and the rule
refused. It declined a five-fold win because the worst case was a 13% loss.

The two things being asked are different, so they are now asked separately:

| question | asked of | condition |
|---|---|---|
| Is there an expected win? | the point estimate | `predicted < 1 - min_relative_gain` |
| Is the downside bounded? | the uncertainty | `pessimistic < MAX_REGRESSION_RATIO` |

The first is about the *size* of the opportunity; the second is a ceiling on
risk, not a demand that the worst case also be good. That is also what Bao
actually claims a learned optimizer must guarantee: never *much* worse than the
optimizer it replaces. Never worse at all is a different, unachievable promise.

Both dashboard cases come out right under it. The 5.9× win deviates (0.20
clears the gain bar, 1.13 is inside the ceiling). A candidate predicted at 0.89×
whose pessimistic case is 1.32× is held, because that one really is a coin flip
on a small edge.

Measured over the built-in workload, same harness, same model:

| gate | vetoes | oracle headroom captured |
|---|---|---|
| worst-case-must-win | 20/24 | 13.5% |
| no gate at all | 2/24 | 26.0% |
| **expected win + bounded downside** | 10/24 | **52.7%** |

The gate now beats following the model blindly, because it lets the good bets
through *and* blocks the bad ones.

## Enforcing it, not just predicting it

Every mechanism above reasons about a plan before it runs. `RUNTIME_TIMEOUT_FACTOR`
is the only one that acts once a prediction has already turned out wrong: a
deviation is given a deadline derived from what native actually costs on that
query, and if it runs past it the statement is cancelled and PostgreSQL's plan
is run instead. Queries with no measured history get no deadline, and native
itself never gets one — cancelling it would leave nothing to fall back to.

## A safety gate with a guarantee, not a constant

The gate that decides whether to deviate has always had the right shape — bound
the prediction by its own uncertainty before betting on it — with an arbitrary
number in the place that decides how much protection it buys:

```
predicted_ratio + z * sigma  <  1 - min_relative_gain
```

`z = 1.0` is a familiar quantity, not a measured one, and ensemble spread is a
useful signal rather than a calibrated standard deviation. Reading
`mean + 1·sigma` as an 84th percentile assumes Gaussian residuals; this
workload's are heavy-tailed, which is the case that assumption gets wrong in
the dangerous direction.

`GATE_MODE=conformal` replaces the constant with **split conformal prediction**
(Vovk et al.; Lei et al., JASA 2018): score held-out examples by how wrong the
model was, take the (1−α) quantile, use it as the multiplier. The bound then
holds at least 1−α of the time whatever shape the residuals have — no normality
assumption, finite-sample, distribution-free.

Ordinary split conformal spends training data on calibration, which is the
worst thing to spend here (§2.3 — data scarcity is the binding constraint). So
the residuals come from the ensemble's own **out-of-bag** predictions: each
member is fit on a bootstrap resample, so ~37% of rows are unseen by any given
member, and averaging only the members that missed a row gives an honest
held-out prediction for *every* training row at zero cost in data. That is
jackknife+-after-bootstrap (Kim, Xu & Barber, NeurIPS 2020), which exists for
ensembles built exactly the way `bandit.py` builds this one.

`app.train` reports the achieved coverage on held-out queries, so the guarantee
is checked rather than asserted.

**Which bound is used is measured, not chosen.** `app.calibrate` sweeps
`fixed_z` and `conformal` as one grid over the same held-out queries, and
`--apply` writes the winner into `models/gate.json` alongside the thresholds it
was swept under -- the two have to travel together, because a
`min_relative_gain` chosen against a conformal bound is not the right threshold
for a fixed-z one.

That sweep is worth running rather than trusting the default. Measured on a
model with 15 held-out queries (docs/WRITEUP.md 2.42), `fixed_z`'s net result
moved from **-4.04% to +1.17%** depending on where `min_relative_gain` was set,
with a worst case of 1.73x native; `conformal` returned **+1.10%** at every
value of that knob with a worst case of 1.07x. The best single number belongs
to `fixed_z` by 0.07 points, which is far below what this project's measurement
noise can resolve (2.18, 2.32) -- the 5-point swing is not.

Precedence: explicit constructor argument, then `GATE_MODE`, then
`models/gate.json`, then `fixed_z`. The environment variable outranks the gate
file here (unlike `CONFIDENCE_Z`) because exporting it is an override of the
method itself, and a stale sweep should not silently undo that.

### What it actually measured, which was not what it was built for

On a calibration run here (936 executions, 17 queries, 237 training rows) the
calibration comes back with **q = 8.5** — the bound needs
*eight and a half* sigmas to cover 90% of held-out outcomes, where `z = 1.0`
had been assumed sufficient. The reason is visible in the same eval output:

| Spread against error | log-ratio units |
|---|---|
| mean ensemble spread on held-out rows | 0.175 |
| actual mean absolute error | **0.403** |

The ensemble members agree with each other roughly 2.3× more than they agree
with reality. Bootstrap resampling measures *sampling* variability; it does not
see model bias or irreducible measurement noise, and both dominate here. So
`mean + 1·sigma` was never an 84% bound. It was a number that looked like one.

The practical consequence is that conformal mode is **not usable on this
dataset**: at the model's typical uncertainty a 90% bound demands a predicted
4.2× speedup before it will deviate, which essentially never occurs, so the
optimizer would decline almost every query. That is a real answer, not a
failure of the method — it is the project's own stated limitation ("prediction
error is comparable to the available headroom at this data scale") converted
from an assertion into a measurement with a guarantee attached. Kept, off by
default, and written up as the null result it is, exactly like the `Rows()`
correction in §2.8.1. `docs/WRITEUP.md` §2.4.5 has the full working.

Two things it is worth for, regardless: 229 of 237 training rows produced an
out-of-bag residual (96.6%), so the claim that it costs no training data holds;
and the gap it exposes between believed and actual uncertainty is the thing to
fix if the deviation rate is ever to be raised safely.

What it does *not* claim: coverage is marginal and assumes exchangeability. A
workload or schema that stops resembling the training queries breaks that
assumption, and the guarantee with it — the same assumption the rest of the
model already runs on, stated rather than buried.
