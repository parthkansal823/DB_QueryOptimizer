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
prefix of the join order. Each candidate is featurized two ways: per-table
slots (selectivity, join position, scan type) and *plan-tree structure*
(depth, bushiness, estimated intermediate-result blowup, operator mix --
`backend/app/optimizer/plan_tree.py`), the latter being a cheap stand-in for
the tree convolution Neo and Bao use. Selection is a **bootstrapped
ensemble** of LightGBM regressors (`backend/app/optimizer/bandit.py`),
which supports three policies -- `greedy` (argmin of the mean),
`thompson` (sample one ensemble member per decision: Bao's bootstrapped
Thompson sampling, giving genuine exploration), and `risk_averse` (argmin
of mean + λ·σ, penalising plans the ensemble disagrees about). A **safety
veto** discards any learned pick the planner costs far above the native
plan, so the system cannot knowingly serve a large regression -- Bao's
central practical claim. Cold start falls back to the Phase 0 heuristic
(lowest Postgres-*estimated* cost) until a model has been trained.

| | Native Postgres CBO | Neo | Bao | This project |
|---|---|---|---|---|
| **Approach** | Cost-based DP / GEQO | Learned, builds full plan | Learned, picks among coarse hint-sets | Learned, picks among join-order/-method hints |
| **Action space** | Full plan space (implicit) | Full plan space (explicit search) | ~48 fixed hint-sets | Join-order permutations x join-method (this schema: <=20/query) |
| **Training signal** | None (static heuristics + stats) | Real latency, experience replay | Real latency, contextual bandit reward | Real latency, supervised regression |
| **Plan encoding** | N/A (cost formulas) | Tree convolution over plan tree | Tree convolution over plan tree | Per-table slots + hand-built plan-tree structure features |
| **Exploration** | None | Epsilon-greedy over tree search | Bootstrapped Thompson sampling | Bootstrapped Thompson sampling (same mechanism, smaller action space) |
| **Uncertainty** | None | Implicit | Ensemble spread | Ensemble spread (drives the risk-averse policy) |
| **Safety** | N/A | None explicit | Falls back to native optimizer | Cost-ratio veto vs. the native plan |
| **Cold start** | N/A | Bootstrapped from CBO plans | Falls back to native optimizer | Falls back to Phase 0 heuristic |
| **Key strength** | Zero training cost, decades of tuning | Highest ceiling -- learns operators too | Bounded risk, safe in production | Simple to train and explain; oracle-relative evaluation; query-level split |
| **Key limitation** | Cardinality-estimation errors compound across joins | Slow to converge, large feedback requirement | Coarser control (whole operator classes, not per-join) | Prediction error exceeds available headroom at this data scale (see §2) |

## 2. Results

### 2.0 A correctness bug that invalidated an earlier set of results

Worth recording, because it is the single most important thing this project
learned and because the failure was **silent**.

`pg_hint_plan` installs its planner hooks when the library is *loaded*, so
it must appear in `shared_preload_libraries`. `postgres/init/01_extensions.sql`
ran `CREATE EXTENSION pg_hint_plan`, which registers the SQL objects but
does **not** preload the library. Separately, `plan_extractor.get_plan`
wrapped queries as `EXPLAIN (...) /*+ Leading(...) */ SELECT ...`, putting
the hint *after* the `EXPLAIN` keyword, where pg_hint_plan does not look for
it.

Either bug alone is enough to make every hint a no-op. And a hint that
cannot be applied is just a SQL comment -- it raises nothing, warns nothing.
The pipeline ran happily and produced plausible-looking numbers for the
entire candidate set while **every "candidate" was byte-identical to the
native plan**. The measured "improvements" were run-to-run timing noise.

The tell was that all 8 candidates for a 4-table query reported the exact
same `Total Cost` (16134) and the same join order. Both bugs are fixed
(`shared_preload_libraries=pg_hint_plan` in `docker-compose.yml`; hint
hoisted ahead of `EXPLAIN` in `plan_extractor._split_hint`) and
`tests/test_plan_extractor.py::test_hint_is_hoisted_ahead_of_the_explain_keyword`
guards the regression. Data collection time went from 51s to 366s once the
hints actually bound -- candidates were finally *different plans*, some of
them slow. **Everything in §2.1/§2.2 below is from after the fix.**

The generalisable lesson: an experiment whose treatment silently does
nothing still produces a full set of numbers. Verifying that the
intervention *changed anything at all* (here: do candidate plans differ
from each other?) belongs in the pipeline, not in a reviewer's intuition.

### 2.1 Offline evaluation (`app.train`, query-level held-out split)

Trained on `plan_execution_log` after Phase 1 data collection (25 workload
queries x native baseline + join-order + join-method candidates). Split at
the *query* level (75/25) so a query's own candidates never leak between
train and test -- with 25 queries that's 6 held-out queries.

Two collection runs, to show what more data buys. Both post-bug-fix; the
only difference is repetitions per candidate (`--reps 1` vs `--reps 3`):

| Metric | 1 rep (403 rows) | **3 reps (2525 rows)** |
|---|---|---|
| Test MAE (latency prediction) | 44.4 ms | **25.5 ms** |
| Mean ensemble uncertainty | 27.1 ms | **10.6 ms** |
| Avg. latency -- native Postgres | 108.4 ms | 108.4 ms |
| Avg. latency -- Phase 0 heuristic | 107.4 ms | 107.9 ms |
| Avg. latency -- **oracle (best possible)** | 100.9 ms | 86.3 ms |
| Avg. latency -- learned, `greedy` | 105.5 ms | 98.9 ms |
| Avg. latency -- learned, `thompson` | 107.7 ms | 101.9 ms |
| Avg. latency -- learned, `risk_averse` | 106.0 ms | **92.1 ms** |
| Headroom captured -- `risk_averse` | 32% | **74%** |

Model: bootstrapped ensemble of 8 LightGBM regressors, 41 features, split at
the query level (6 held-out queries).

Tripling the repetitions nearly halved prediction error, and `risk_averse`
-- which penalises candidates the ensemble disagrees about -- captures ~74%
of the available headroom on held-out queries, beating both `greedy` (43%)
and `thompson` (29%). That ordering is what theory predicts when a model is
accurate on average but unreliable in places: distrusting your own
high-variance predictions is worth more than chasing their minimum.
`thompson` trailing both is also expected -- it is *spending* performance to
gather information, which is the trade exploration makes.

### 2.2 Live benchmark, and why it disagrees with §2.1

Re-running the full workload live (`app.benchmark`), three runs per policy,
share of oracle headroom captured (negative = worse than plain native):

| Run | `greedy` | `risk_averse` |
|---|---|---|
| 1 | +39.7% | +25.9% |
| 2 | +36.7% | -6.3% |
| 3 | -30.7% | -148.8% |

**These do not reproduce §2.1's result, and the gap is the most interesting
finding in this document.**

The offline evaluation replays *logged* latencies: for a held-out query it
picks among candidate latencies that were each measured exactly once, and
scores itself against those same recorded numbers. So a selector is rewarded
partly for identifying which candidate got the *luckiest measurement*, not
which plan is genuinely fastest. The oracle column has the same bias --
"best possible" is really "best single sample observed."

The live benchmark re-executes every plan, drawing fresh samples, and the
advantage largely evaporates. That is the honest signal: **offline
replay-based evaluation of a plan selector is optimistically biased, and a
query-level train/test split does not fix it** -- the split prevents leakage
between queries, but the noise lives inside each query's candidate
measurements.

The general lesson, which applies well beyond this project: when your label
noise is comparable to the effect you are trying to measure, held-out
evaluation on logged outcomes will flatter you. Only re-execution tests the
thing you actually care about. Anything reported here on the strength of
§2.1 alone would have been overclaiming.

The safety veto fired on 2-5 of 25 queries per run, doing real work: with
hints binding, some forced join orders carry Postgres's `disable_cost`
penalty (~1e10, signalling a cartesian product), and those are discarded
before execution rather than served.

### 2.3 Where this leaves things, and what would actually fix it

Honest summary: **the learned selector does not yet reliably beat native
PostgreSQL under live re-execution.** Offline it looks strong (74% of
headroom); live it is inconsistent. Given §2.2, the offline number should be
treated as an upper bound contaminated by measurement noise, not as a
result.

The evidence says this remains a measurement problem before it is a
modelling problem:

- **Label noise is still the binding constraint.** Going from 1 to 3
  repetitions cut MAE from 44 ms to 25 ms -- a big move from a small change,
  which is the signature of noise-dominated labels rather than an
  under-powered model. There is likely more to gain the same way.
- **Single samples are the wrong label.** Each candidate's target is one
  timing. The fix is to execute each candidate many times and train on the
  *median*, which is exactly what §2.2's bias analysis argues for. This is
  the single highest-value next change, and it is cheap to implement --
  `collect_data.py` already takes `--reps`; nothing aggregates them yet.
- **The measurement environment is not quiesced.** These runs share a laptop
  with Docker, a browser, and an editor. Latency measured this way has a
  heavy right tail (see the -148.8% outlier in §2.2, almost certainly one
  query hitting contention).
- **The workload may be too easy.** Postgres already picks a near-oracle
  plan for most of these 25 queries. A learned optimizer can only win where
  the CBO is *wrong*, which is precisely why Leis et al. built JOB out of
  deliberately hard, correlated-predicate queries.

In priority order: aggregate repetitions into median labels; quiesce the
measurement environment; expand the workload, selecting for queries where
Postgres's cardinality estimates are known to be poor; and re-run the
evaluation as live re-execution rather than offline replay. The
`headroom_captured_vs_oracle` metric is what makes any of this diagnosable
-- without an oracle column, "3% better than native" is unreadable.

### 2.3 JOB/IMDB stretch goal

The real 21-table, ~74.5M-row IMDB dataset was downloaded and imported in
full (`data/job/load_job.sh`). A smoke test -- 8 real JOB queries through
the unmodified `app.collect_data`/`app.train` pipeline, `DATABASE_URL`
repointed at the `job` database, zero code changes -- confirmed the whole
pipeline works against it: `schema_introspection` discovered all 21 real
tables automatically, and both the heuristic and the trained model
correctly deferred to Postgres's own plan on both held-out queries (i.e.
no regression, though 2 held-out queries is too small a sample to claim
more than that). Full details, honestly including what a complete JOB
evaluation would still need (all 113 queries, more reps, join-method
candidates enabled): `docs/JOB_RESULTS.md`. Load procedure:
`data/job/README.md`. Why the pipeline needed no code changes: Section 3
below.

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
- **The headline limitation: prediction error exceeds available headroom.**
  See §2.3. ~400 rows across 25 queries, one execution per candidate, gives
  a 44 ms MAE against ~7.5 ms/query of actual opportunity. Until that ratio
  inverts, no selection policy can demonstrate a real win, and any positive
  result on a single run should be assumed to be noise.
- **No automatic retraining.** Thompson sampling gives the system genuine
  *exploration*, but the loop is not closed: new live traffic accumulates in
  `plan_execution_log` and is only used when someone reruns `app.train` by
  hand. A real bandit retrains (or updates incrementally) on the feedback it
  gathers; this one explores and then forgets.
- **The safety veto is cost-based, not learned.** It compares the candidate's
  *estimated* cost against the native plan's -- and distrusting those
  estimates is the entire premise of the project. It reliably catches the
  catastrophic cases (Postgres's `disable_cost` marks them at ~1e10), but it
  cannot catch a plan that is cheap on paper and slow in reality, which is
  precisely the case a learned optimizer exists to handle.
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
