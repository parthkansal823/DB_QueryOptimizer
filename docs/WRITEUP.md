# Writeup: literature review, results, and limitations

This document covers what `docs/ROADMAP.md`'s Phase 5 asks for: how this
project relates to the published learned-query-optimization literature, what
the evaluation actually showed, and an honest account of where it falls
short. See `README.md` for architecture and quickstart.

## 1. Literature review

Query optimization -- picking an execution plan for a declarative query --
has been cost-based and hand-tuned since System R. The systems below
represent three different answers to "what should replace, or augment, the
cost-based optimizer's choice."

**Native Postgres (cost-based optimizer, CBO).** Dynamic programming over
join orders for small queries (falling back to a genetic algorithm, GEQO,
past `geqo_threshold` tables), driven by selectivity estimates from
per-column histograms and independence assumptions across predicates. No
training required and well-tuned over decades, but the independence
assumption breaks down under correlated predicates and data skew --
cardinality estimation errors compound multiplicatively across joins. This
is precisely the failure mode documented in Leis et al., ["How Good Are
Query Optimizers, Really?"](http://www.vldb.org/pvldb/vol9/p204-leis.pdf)
(VLDB 2015) -- the paper that introduced the Join Order Benchmark this
project's stretch goal loads (Section 4).

**Neo** (Marcus et al., ["Neo: A Learned Query Optimizer"](https://arxiv.org/abs/1904.03711),
VLDB 2019). Builds a full plan bottom-up with a neural network (a tree-convolutional
value network) trained via experience replay against real execution latency,
bootstrapped from an existing optimizer's plans and then improving through
exploration. Action space: essentially everything a classical optimizer
controls -- join order, join method, access path -- learned end to end.
Highest ceiling of the three, but the largest action space also means the
slowest convergence, the most execution feedback needed, and the least
interpretable failure mode: a bad exploration step can regress badly before
the model corrects.

**Bao** (Marcus et al., ["Bao: Making Learned Query Optimization
Practical"](https://arxiv.org/abs/2004.03814), SIGMOD 2021). Reframes the
problem as a contextual bandit over a small, fixed set of coarse-grained
query hints (~48 hint-sets in the paper, each toggling whole classes of
operators on/off), using a tree-convolutional network to predict which
hint-set will produce the best plan for a given query, with Thompson
sampling for exploration and a safe fallback to the native optimizer. The
small, bounded action space is the whole point: it makes the system safe to
run in production (worst case, you get a plan the native optimizer could
have produced anyway) and fast to retrain. **This project follows Bao's
framing directly** -- hints over full plan construction -- but narrows the
action space further (join order, plus join method as a stretch goal,
instead of Bao's operator-class toggles) and replaces the contextual bandit
with a simpler regression model, trading Bao's online regret bounds for
something easier to train and explain with a small amount of data.

**This project.** Candidates are `pg_hint_plan` `Leading()` join-order hints
(exhaustive permutation for <=5 tables, random sampling above that -- a
named limitation, see below), extended for the stretch goal with forced
join-method hints (`HashJoin`/`NestLoop`/`MergeJoin`) applied at every
prefix of the join order. A gradient-boosted regression tree (LightGBM)
predicts each candidate's latency from a feature vector (join position,
selectivity, scan type, join-method counts per table -- see
`backend/app/optimizer/features.py`); the optimizer executes every
candidate (dev/demo only -- see limitations) and picks the argmin. Cold
start falls back to the Phase 0 heuristic (lowest Postgres-*estimated*
cost) until a model has been trained.

| | Native Postgres CBO | Neo | Bao | This project |
|---|---|---|---|---|
| **Approach** | Cost-based DP / GEQO | Learned, builds full plan | Learned, picks among coarse hint-sets | Learned, picks among join-order/-method hints |
| **Action space** | Full plan space (implicit) | Full plan space (explicit search) | ~48 fixed hint-sets | Join-order permutations x join-method (this schema: <=20/query) |
| **Training signal** | None (static heuristics + stats) | Real latency, experience replay | Real latency, contextual bandit reward | Real latency, supervised regression |
| **Exploration** | None | Epsilon-greedy over tree search | Thompson sampling | None (argmin over a precomputed candidate set) |
| **Cold start** | N/A | Bootstrapped from CBO plans | Falls back to native optimizer | Falls back to Phase 0 heuristic |
| **Key strength** | Zero training cost, decades of tuning | Highest ceiling -- learns operators too | Bounded risk, safe in production | Simple to train and explain; query-level train/test split |
| **Key limitation** | Cardinality-estimation errors compound across joins | Slow to converge, large feedback requirement | Coarser control (whole operator classes, not per-join) | No online exploration; candidate sampling above 5 tables; small training set here |

## 2. Results

### 2.1 Offline evaluation (`app.train`, query-level held-out split)

Trained on `plan_execution_log` after Phase 1 data collection (25 workload
queries x native baseline + join-order candidates + join-method candidates,
one rep each -- see `backend/app/workload.py`). Split at the *query* level
(75/25) so a query's own candidates never leak between train and test --
with 25 queries that's 6 held-out queries, small enough that these numbers
are indicative, not statistically strong (see limitations).

| Metric | Value |
|---|---|
| Model | LightGBM (`LGBMRegressor`, 200 trees, depth 5) |
| Training rows | 741 (across 25 queries) |
| Held-out queries | 6 |
| Test MAE (latency prediction) | ~14.3 ms |
| Avg. latency -- native Postgres | ~93.8 ms |
| Avg. latency -- Phase 0 heuristic (argmin estimated cost) | ~95.3 ms |
| Avg. latency -- trained model (argmin predicted latency) | ~92.5 ms |
| Model picks <= heuristic's pick | 6/6 held-out queries |

The trained model matched or beat the cost-based heuristic on every
held-out query, and both beat native Postgres's un-hinted default plan on
average -- modest margins, consistent with a workload that Postgres's own
estimator already handles reasonably well most of the time (see `data/schema.sql`'s
skew, which is what makes the gap nonzero at all).

### 2.2 Live benchmark (`app.benchmark`, full 25-query workload)

Running the live `LearnedOptimizer` (trained-model path) against the full
workload in one pass:

| | Total latency, 25 queries |
|---|---|
| Native Postgres | 2837.7 ms |
| Learned path (chosen candidate) | 2552.4 ms |
| **Improvement** | **~10%** |

This number moves run to run (system load, Postgres's plan cache, autovacuum
timing) -- `/stats/trend` and the dashboard's "Historical accuracy" panel
track it over repeated runs rather than trusting a single number, which is
the Phase 4 "trending over time" requirement.

### 2.3 JOB/IMDB stretch goal

See `data/job/README.md` for the load procedure and `docs/JOB_RESULTS.md`
(generated once the import completes) for results against the real dataset.
The pipeline itself required zero code changes to point at JOB -- see
Section 3.

## 3. Stretch goals

**Join-method selection.** `generate_join_method_candidates`
(`backend/app/optimizer/hints.py`) pairs each sampled join order with a
forced method (`HashJoin`/`NestLoop`/`MergeJoin`) applied at every prefix of
that order's left-deep join tree, since `Leading(a b c d)`'s join nodes are
exactly the prefixes `(a b)`, `(a b c)`, `(a b c d)`. This roughly
approximates "use this method throughout the plan" without a full per-node
hint-tree generator. `features.py` counts join methods actually used per
plan (`n_hash_join`/`n_nestloop_join`/`n_merge_join`), so the model can learn
method-sensitive patterns, not just order-sensitive ones.

**Dataset-agnostic pipeline.** `optimizer/features.py` originally hardcoded
the synthetic schema's 4 tables and their aliases. It's now schema-driven:
table identity comes from each EXPLAIN plan's own `scan_relations` (alias ->
real table name, read directly off the plan by `plan_extractor.py`), and
reference cardinalities come from `schema_introspection.py` querying
Postgres's own `pg_class` statistics. Point `DATABASE_URL` at a different
database and `hints.py`, `plan_extractor.py`, `features.py`, and `train.py`
all adapt with no code change -- `feature_columns` and `table_cardinalities`
are computed at training time and pickled alongside the model so inference
stays consistent with whatever schema it was trained on. This is what let
the JOB/IMDB import (Section 2.3) reuse the exact same pipeline.

## 4. Limitations, named on purpose

- **Candidate sampling above 5 tables.** `generate_join_order_candidates`
  enumerates every permutation for <=5 tables but randomly samples above
  that (permutations explode factorially -- 10 tables is 3.6M orderings).
  JOB queries go up to 17 tables; a learned *candidate generator* (rather
  than exhaustive/random search) is the natural next step, and is closer to
  what Neo actually does.
- **Every candidate is executed in the demo path.** `/query/analyze` runs
  the baseline and every candidate so the dashboard can show them side by
  side. That's fine for a dev/demo tool; a production system would only
  execute the chosen plan and use the optimizer's own cost estimate (or a
  learned cost model, as in Bao) to pick without running the alternatives.
- **Small training set.** 741 rows across 25 queries, 6 held-out, is enough
  to prove the pipeline works end to end but not enough to draw strong
  statistical conclusions -- the offline evaluation numbers in 2.1 should be
  read as indicative, not definitive. More workload queries (Phase 1's
  "20-30" was a floor, not a target) and more repetitions per candidate
  would tighten this.
- **No online exploration.** Unlike Bao's Thompson sampling, this system
  doesn't explore -- it trains once on collected data and serves argmin
  predictions. It also doesn't retrain automatically as new live traffic
  accumulates in `plan_execution_log`; that's a manual `app.train` rerun
  today, an obvious Phase 4 follow-up (periodic retraining) it does not
  attempt.
- **Cold start is honest, not solved.** Before a model is trained,
  `LearnedOptimizer` falls back to the Phase 0 cost heuristic. That's a
  reasonable default but means the system provides zero learned benefit
  until someone runs `app.collect_data` + `app.train` -- there's no
  incremental/bootstrapped warm-up like Neo's.
- **Self-joins collapse to one feature slot.** A query that joins the same
  table twice under different aliases (common in JOB, e.g. `movie_info AS
  mi1, movie_info AS mi2`) gets one feature slot for that table, not two --
  see `features.py`'s docstring. The vector stays fixed-length by table
  *identity*; a per-occurrence (positional) encoding would fix this at the
  cost of a variable-length or much larger feature space.
- **Single-node Postgres, no replication, no concurrent-load testing.**
  All latency numbers are single-query, single-connection, otherwise-idle
  measurements. Real workloads have concurrent queries competing for
  buffer cache and I/O, which changes the calculus around candidate
  execution cost.
- **Synthetic skew is still synthetic.** `data/schema.sql`'s power-users /
  popular-products skew is deliberately simple (an 80/20-ish split) compared
  to the long-tailed, correlated skew real IMDB data has -- which is
  exactly why the JOB/IMDB stretch goal (Section 2.3) matters for external
  validity, and also why its numbers shouldn't be assumed to match the
  synthetic-schema numbers in Section 2.1/2.2.
