# How it works

A walkthrough of the whole system: the problem it solves, what happens when a
query goes through it, what each piece does, and where to find the code.

Read this first if you want to understand the project. `METRICS.md` covers how
the numbers are calculated, `DASHBOARD.md` covers the UI, and `WRITEUP.md`
covers how it compares to published research and what the results were.

---

## 1. The problem

When you run a SQL query, the database has to decide *how* to run it. For a
four-table join there are dozens of valid strategies: which tables to join
first, whether to use a hash join or a nested loop, whether to scan a table or
use an index. They all return the same rows. Some take a millisecond, some take
ten seconds.

PostgreSQL picks by estimating a cost for each strategy and taking the cheapest.
The estimate is built from column statistics, and it makes one assumption that
often fails:

```sql
WHERE city = 'Mumbai' AND country = 'IN'
```

PostgreSQL treats these as unrelated. It multiplies their individual
selectivities — 1% of rows match the city, 20% match the country, so it expects
0.2% to match both. In reality Mumbai is *in* India, so every Mumbai row already
matches. The estimate is 100x too low.

Row estimates feed the cost model, and the errors multiply at each join. A plan
built on a row count that is 100x wrong is not slightly wrong; it is often a
completely different, much slower plan.

This is the failure mode Leis et al. documented in *"How Good Are Query
Optimizers, Really?"* (VLDB 2015). Cost models are roughly fine. The row counts
fed into them are not.

## 2. The approach

Two ways to fix this:

1. **Replace the planner.** Write a new optimizer that chooses plans itself.
   This is what Neo (VLDB 2019) does. Highest ceiling, hardest to build, and it
   throws away decades of tuning in PostgreSQL's planner.
2. **Steer the existing planner.** Let PostgreSQL plan, but generate a handful
   of *alternative* plans, run them, measure which is actually fastest, and
   learn to pick. This is Bao's framing (SIGMOD 2021), and it's what this
   project does.

The steering is done with [pg_hint_plan](https://github.com/ossc-db/pg_hint_plan),
a PostgreSQL extension that reads special comments and constrains the planner:

```sql
/*+ Leading(orders users items) */   -- join in this order
/*+ Set(enable_hashjoin off) */      -- don't use hash joins
/*+ Rows(orders users *10) */        -- that join returns 10x what you think
SELECT ...
```

Same query, same results, different plan. That gives us an *action space*: a
set of alternatives to choose between.

The key insight that makes learning possible: **we can measure.** Every
`EXPLAIN ANALYZE` tells us exactly how long a plan took. That's a perfect
training label for the thing the cost model is guessing at.

---

## 3. What happens when you run a query

The measurement path (`POST /query/analyze`, used by the dashboard):

```
  Your SQL
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. BASELINE            plan_extractor.get_plan()            │
│    Run it the way PostgreSQL wants. Record the plan tree,    │
│    the estimated cost, and how long it really took.          │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. GENERATE ALTERNATIVES          optimizer/hints.py         │
│    Build candidate hints from the tables involved:           │
│      • join orders      Leading(a b c)                       │
│      • join methods     HashJoin(a b) / NestLoop(a b)        │
│      • operator toggles Set(enable_seqscan off)              │
│      • learned row corrections  Rows(a b *10)                │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. DEDUPLICATE                hints.plan_fingerprint()       │
│    Most hints on a simple query reproduce the native plan.   │
│    Identical plans are dropped, so the candidate count       │
│    reflects real choices instead of copies.                  │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. FEATURIZE          optimizer/features.py + plan_tree.py   │
│    Turn each plan into a fixed-length vector, using only     │
│    estimate-side fields (never actual timings — see §6).     │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. PREDICT AND CHOOSE      optimizer/learned.py + bandit.py  │
│    An ensemble predicts each candidate's speedup vs native,  │
│    with an uncertainty estimate. A policy picks one.         │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. TWO SAFETY CHECKS                                         │
│    Cost veto (forward)  — is this priced far above native?   │
│    Regression guard (back) — has this query burned us before?│
│    Either one → serve PostgreSQL's plan instead.             │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. LOG EVERYTHING              logging_store.py              │
│    Every execution → plan_execution_log. This is both the    │
│    training data and the source for every dashboard number.  │
└─────────────────────────────────────────────────────────────┘
```

### The production path is different, and that matters

`/query/analyze` **executes every candidate** so the dashboard can compare
them. That costs N executions to answer one question — it is a measuring
instrument, not an optimizer. Serving real traffic with it would be slower than
having no optimizer at all.

`POST /query/optimize` (`optimizer/planner.py`) is the real path:

```
N x EXPLAIN without ANALYZE    →  PostgreSQL plans, runs nothing
1 x actual execution           →  only the plan that was chosen
```

Planning is cheap (sub-millisecond); execution is not. Measured on this
workload: **3.8 ms of planning against 138 ms of execution**, about 2.7%
overhead.

---

## 4. The data model

Everything revolves around one table (`postgres/init/03_logging.sql`):

```sql
plan_execution_log (
    query_id              TEXT,        -- workload id, or adhoc:<hash of SQL>
    sql_text              TEXT,
    hint                  TEXT,        -- NULL for PostgreSQL's own plan
    is_baseline           BOOLEAN,     -- this was PostgreSQL's choice
    is_chosen             BOOLEAN,     -- this is what actually ran
    selector_used         TEXT,        -- native | learned | heuristic | collection
    raw_plan              JSONB,       -- the full EXPLAIN tree
    total_cost            DOUBLE,      -- what PostgreSQL estimated
    actual_total_time_ms  DOUBLE,      -- what it really took
    created_at            TIMESTAMPTZ  -- now(), frozen per transaction
)
```

Three things about this design are load-bearing:

**`is_baseline` and `is_chosen` can both be true.** When the optimizer decides
to keep PostgreSQL's plan, that one execution *is* both the baseline and the
served plan. Recording it as only a baseline is what made an earlier version of
the dashboard report a 97% improvement that was not real — see `METRICS.md` §2.

**`created_at` is frozen per transaction.** PostgreSQL holds `now()` constant
for the whole transaction, so every row a single request writes shares one
timestamp. That is what lets us reconstruct "one decision" after the fact by
grouping on `(query, created_at)`.

**`raw_plan` keeps the whole tree.** Storing only summary numbers would have
made the later cardinality work impossible; the join-level row estimates and
actuals are in there.

---

## 5. Turning a plan into numbers

`optimizer/features.py` builds a fixed-length vector from three blocks:

| Block | Contents |
|---|---|
| **Scalars** | table count, join count, estimated cost, join-method counts |
| **Plan-tree structure** | depth, bushiness, estimated row blowup, operator mix |
| **Per-table slots** | for each table: present, join position, selectivity, index scan |

The plan-tree block (`optimizer/plan_tree.py`) is a hand-built stand-in for the
tree convolution Neo and Bao use — cheaper, and workable with far less data.

**The per-table slots are what make this schema-agnostic.** Table identity is
read from each plan's own `scan_relations` (PostgreSQL tells you which alias
maps to which table), and table sizes come from `pg_class`. Nothing is
hardcoded. Point `DATABASE_URL` at a different database and the same code
adapts — which is how the 21-table, 74M-row JOB/IMDB benchmark ran through the
pipeline with no code changes.

---

## 6. The one constraint that makes production inference possible

**No feature may read an actual measurement.** Only estimate-side fields:
`Plan Rows`, `Total Cost`, `Plan Width`, node types.

This looks like an arbitrary restriction until you need to score a plan
*before* running it. If any feature needed `Actual Rows`, the only way to
score a candidate would be to execute it — and then there is no point
predicting anything. The production path exists because this rule was followed.

---

## 7. The model

`optimizer/bandit.py` holds a **bootstrapped ensemble**: eight LightGBM
regressors, each trained on a different resample of the data. Their spread is a
usable uncertainty estimate — where they agree, evidence is strong; where they
disagree, the model is guessing.

### It predicts a ratio, not a duration

The target is `log(candidate_latency / native_latency)` for the same query, not
raw milliseconds. Two reasons:

1. **Scale would dominate.** The workload spans 5 ms to 600 ms queries. Squared
   error on raw milliseconds is driven almost entirely by the slow ones, so the
   model spends its capacity learning "this query is slow" — true, and nothing
   to do with plan choice.
2. **It answers an easier question.** We never need to know a plan takes 213 ms.
   We need to know it beats native. Negative prediction = faster than native.

### Four ways to choose

| Policy | Behaviour |
|---|---|
| `greedy` | Take the lowest predicted latency |
| `thompson` | Sample one ensemble member per decision — keeps exploring |
| `risk_averse` | Penalise candidates the ensemble disagrees about |
| `pairwise_rank` | Lero-style: a classifier answering "is A faster than B?" |

Which one wins depends on how much data you have. On the small synthetic schema
`pairwise_rank` leads; on data-starved JOB every policy that trusts its own
point predictions collapses and only `risk_averse` survives. Set with
`SELECTION_POLICY`.

---

## 8. Two safety layers

An optimizer that is faster on average but occasionally three times slower on
one query cannot be deployed — one user-facing query getting slower outweighs a
diffuse average win.

**Cost veto — looks forward** (`optimizer/learned.py`). Before running anything,
discard any candidate PostgreSQL prices far above the native plan. Catches
catastrophes, but it reasons about the very estimates this project distrusts, so
it cannot catch a plan that looks cheap and runs slow.

**Regression guard — looks backward** (`optimizer/regression_guard.py`). Reads
the system's own history and blocks the learned path for queries where it has
measurably been slower:

```
blocked  ⟺  avg(served) > avg(native) × 1.10  AND  at least 3 observations
```

Deliberately asymmetric: a query must *prove* a regression to be blocked, but it
keeps being re-measured and can recover. A brake, not a ban.

**And a confidence gate.** Even with a candidate in hand, the optimizer only
deviates if the predicted win exceeds its own uncertainty about that win. The
thresholds are not guessed — `app/calibrate.py` sweeps them against your logged
outcomes and writes the best pair to `models/gate.json`.

The result is that its failure mode is *"no better than native"* rather than
*"worse than native"*.

---

## 9. The learning loop

```
   collect_data.py          train.py              retrain.py
        │                      │                      │
   run every           learn from the           train a challenger,
   candidate, 3x,      accumulated log          score it against the
   log the results          │                   deployed model on the
        │                   ▼                   same held-out queries
        │            models/plan_selector.pkl          │
        │                   │                          ▼
        │                   │                   better by >2%?
        │                   ▼                    ├── yes → deploy
        └──────────  serving (main.py)  ◄────────┤
                            │                    └── no  → keep current
                            ▼
                    plan_execution_log
                    (feedback accumulates)
```

**Three repetitions, median-aggregated.** This is the single most important
detail in the pipeline. Training on one execution per candidate taught the model
that whichever plan got the luckiest timing was genuinely fastest; live runs
swung from +40% to −149%. Taking the median of three removed every regression
with no change to the model at all. Same architecture, better labels.

**Promotion requires a margin.** A challenger must beat the deployed model by
more than 2% on the same held-out queries. Offline scores here are noisy enough
that promoting on any improvement would send the model on a random walk rather
than improving it. Every version is timestamped and `--rollback` restores the
previous one.

---

## 10. Measuring honestly

Two ideas do most of the work here, both covered in detail in `METRICS.md`.

**The oracle ceiling.** For each query, run *every* candidate and record the
fastest. That is the best any selector could possibly have done. Results are
reported as a fraction of it, because "3% faster than PostgreSQL" is unreadable
on its own — you cannot tell whether 3% was everything available or a sliver of
a large opportunity.

This is not a detail. Measuring the ceiling is how we discovered that on the
first version of the benchmark, PostgreSQL was already optimal on 18 of 25
queries and **no model could have learned anything**. Six model types scored 0%
top-1 accuracy. The benchmark was rebuilt around correlated predicates as a
direct result, which is where the interesting results come from.

**Matched pairs.** Every before/after figure compares the native plan and the
served plan *for the same query in the same run*. Comparing the average of the
plans the model chose against the average of all PostgreSQL plans is not a
comparison — the two averages cover different queries. That mistake reported a
97% improvement where the true figure was ~18%.

---

## 11. Where the code lives

```
backend/app/
├── main.py                 HTTP API — the endpoints below
├── db.py                   connection pool (built lazily, see note)
├── plan_extractor.py       EXPLAIN JSON → structured metrics
├── logging_store.py        writes plan_execution_log; query fingerprints
├── schema_introspection.py table sizes from pg_class, any schema
├── schema_graph.py         discovers tables and join edges
├── advisor.py              suggests CREATE INDEX / CREATE STATISTICS
├── workload.py             the 25-query benchmark
├── workload_generator.py   writes a workload for an unknown database
├── onboard.py              one command: discover → generate → collect → train
├── collect_data.py         offline training-data sweep
├── train.py                trains the model, writes the bundle
├── retrain.py              champion/challenger gate
├── model_store.py          versioned models, promote, rollback
├── calibrate.py            measures the best confidence thresholds
├── benchmark.py            native vs learned over the workload
├── experiment.py           paired A/B with confidence intervals
├── stats.py                matched-pair reporting for the dashboard
└── optimizer/
    ├── hints.py            builds the action space
    ├── features.py         plan → feature vector
    ├── plan_tree.py        structural features
    ├── bandit.py           bootstrapped ensemble + policies
    ├── ranker.py           pairwise learning-to-rank
    ├── cardinality.py      learned row-estimate correction
    ├── learned.py          plan selection + cost veto
    ├── regression_guard.py per-query blocking
    ├── planner.py          production path: plan N, run 1
    └── regret.py           cumulative regret
```

### API

| Endpoint | Purpose |
|---|---|
| `POST /query/optimize` | Production path — plans N, executes 1 |
| `POST /query/analyze` | Measurement path — executes every candidate |
| `GET /stats/trend` | Matched-pair history and decision quality |
| `GET /stats/cost-model` | How well PostgreSQL's estimates predict reality |
| `GET /stats/regret` | Cumulative regret vs native |
| `GET /model/status` | Deployed version, pending feedback, blocked queries |
| `POST /model/retrain`, `POST /model/rollback` | Drive the learning loop |
| `GET /schema` | What it discovered about your database |
| `GET /advisor` | Schema-level fixes (unindexed foreign keys) |

### A note on the connection pool

`db.py` builds its pool on first use rather than at import. Connecting at import
time meant importing *any* module that touched the database required a running
PostgreSQL — pure unit tests could not even be collected without one, and the
API could crash on a cold start because `depends_on` waits for the container,
not for PostgreSQL to accept connections.

---

## 12. Testing

```bash
pytest -m "not integration"   # 247 tests, no database required
pytest -m integration         # 11 tests, needs a live PostgreSQL
```

The split exists for a reason. Early on, a configuration mistake meant
pg_hint_plan was silently ignoring every hint — so every "candidate" was
identical to PostgreSQL's own plan and the measured improvements were timing
noise. An entire round of results was meaningless.

No unit test can catch that. The string manipulation was correct; it was the
*effect on the planner* that was missing, and a hint that cannot be applied is
just a SQL comment — no error, no warning.

The integration tests assert the effect: hints bind, candidates are genuinely
different plans, `Rows(a b *100)` really does multiply the estimate by 100. CI
runs the unit suite with no database at all (which keeps them honestly
decoupled) and the integration suite against a purpose-built PostgreSQL image.

---

## 13. Running it

```bash
docker compose up --build
```

PostgreSQL with pg_hint_plan on `:5432`, API on `:8000`, dashboard on `:5173`.
It works immediately using a cost-based heuristic; to get the learned model:

```bash
docker compose exec backend python -m app.collect_data   # gather training data
docker compose exec backend python -m app.train          # train
docker compose exec backend python -m app.benchmark      # native vs learned
```

To point it at your own database, one command discovers the schema, writes a
workload sampled from your real data, collects, and trains:

```bash
DATABASE_URL=postgresql://user:pass@host:5432/yourdb \
    docker compose exec backend python -m app.onboard --queries 25 --reps 3
```
