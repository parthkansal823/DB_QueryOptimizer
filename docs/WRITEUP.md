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

Tripling the repetitions nearly halved prediction error. `risk_averse` --
which penalises candidates the ensemble disagrees about -- consistently
beats `greedy`, which is what theory predicts when a model is accurate on
average but unreliable in places: distrusting your own high-variance
predictions is worth more than chasing their minimum. `thompson` trailing
both is also expected -- it *spends* performance to gather information,
which is the trade exploration makes.

**A fourth selector: pairwise ranking.** Given that prediction error is
comparable to the spread between candidates, predicting absolute latency is
harder than the problem requires. `optimizer/ranker.py` implements the Lero
(VLDB 2023) approach instead -- a classifier trained on *pairs* of
same-query candidates answering only "is A faster than B?", scored at
inference by how many pairwise duels each candidate wins. On median-
aggregated labels it is the strongest offline selector:

| Selector | Headroom captured (held-out) |
|---|---|
| `greedy` | -52% |
| `thompson` | -51% |
| `risk_averse` | +31% |
| **`pairwise_rank`** | **+46%** |

Worth noting honestly: `pairwise_rank` has the best *average* latency but
the lowest beats-the-heuristic *rate* (33% of queries). It wins big when it
wins and loses often but cheaply -- a different risk profile from
`risk_averse` (83% win rate, smaller average gain), and which you'd prefer
depends on whether you care about mean latency or per-query predictability.

### 2.2 Live benchmark: the instability, and what fixed it

Re-running the full workload live (`app.benchmark`) is the real test, since
it re-executes every plan rather than replaying logged timings. Three runs
per policy, share of oracle headroom captured (negative = worse than plain
native):

**Before median aggregation** (training on single-execution labels):

| Run | `greedy` | `risk_averse` |
|---|---|---|
| 1 | +39.7% | +25.9% |
| 2 | +36.7% | -6.3% |
| 3 | -30.7% | **-148.8%** |

**After median aggregation** (`_aggregate_repetitions` collapses each
query/hint's repeated executions to their median before training):

| Run | `greedy` | `risk_averse` | `pairwise_rank` |
|---|---|---|---|
| 1 | +35.7% | +36.1% | +34.2% |
| 2 | +33.2% | +1.4% | +12.5% |
| 3 | +14.9% | +37.6% | -- |

**Every run is now positive, averaging ~26% of oracle headroom, with no
regressions.** The catastrophic -148.8% outlier is gone.

This is the most useful result in the project, and it is a *data* fix rather
than a model fix. Training on single noisy executions taught the model that
whichever candidate got the luckiest timing was genuinely fastest; it then
confidently served those plans and sometimes lost badly. Taking the median
of three executions per candidate removed that failure mode entirely. The
model architecture did not change.

This connects directly to the current research direction on learned
optimizers: recent work targets the *per-query instability* that makes
learned optimizers hard to deploy, since an optimizer that is 30% faster on
average but occasionally 150% slower is unshippable. The evidence here says
a meaningful share of that instability can be measurement noise in the
training labels rather than anything intrinsic to the policy.

### 2.2.1 Offline evaluation is optimistically biased

One caveat that survives the fix. §2.1's offline numbers are computed by
replaying *logged* latencies: for a held-out query the selector picks among
candidate latencies measured once each, and is scored against those same
recorded numbers. So it is rewarded partly for identifying which candidate
got the luckiest measurement. The oracle column shares the bias -- "best
possible" is really "best single sample observed."

Live re-execution consistently comes in below the offline estimate. **A
query-level train/test split does not fix this**: the split prevents leakage
*between* queries, but the noise lives *inside* each query's candidate
measurements. Only re-execution measures the thing that matters.

The general lesson, which applies well beyond this project: when label noise
is comparable to the effect you are trying to measure, held-out evaluation
on logged outcomes will flatter you. Only re-execution tests the thing you
actually care about.

The safety veto fired on 2-5 of 25 queries per run, doing real work: with
hints binding, some forced join orders carry Postgres's `disable_cost`
penalty (~1e10, signalling a cartesian product), and those are discarded
before execution rather than served.

### 2.3 Where this leaves things

**On the synthetic schema it works**: every live run beats native
PostgreSQL, averaging ~26% of available oracle headroom with no regressions
(§2.2). The honest qualifiers are that the workload is small (25 queries),
the absolute headroom is modest (~8 ms/query), and the measurement
environment is a laptop.

**On the real JOB benchmark it does not** (`docs/JOB_RESULTS.md`): 75% of
latency is available there and the system captures none of it, because 194
executions across 8 queries cannot teach a 17-table join space. This is the
more informative of the two results -- it quantifies how far the system is
from the regime the literature operates in.

What the evidence says to do next, in priority order:

1. **More repetitions.** The single biggest win so far came from medians,
   not modelling. Going 1 -> 3 reps cut MAE 44 -> 25 ms and eliminated every
   live regression. 5-10 reps is the obvious next step.
2. **Far more JOB data.** The opportunity is demonstrably there (75%); the
   data is not. This is a compute-time problem, not a research problem.
3. **A quiesced measurement environment.** These runs share a laptop with
   Docker, a browser, and an editor; latency measured this way has a heavy
   right tail.
4. **Close the bandit loop.** Thompson sampling explores, but nothing
   retrains on what it learns -- see limitations.
5. **Real tree convolution.** `plan_tree.py` hand-builds structural features;
   Neo and Bao learn them with a tree-convolutional network. That is the
   principled version, and worth attempting once (2) supplies the data to
   justify the capacity.

The `headroom_captured_vs_oracle` metric is what makes any of this
diagnosable -- without an oracle column, "3% better than native" is
unreadable.

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

## 2.4 Closing the loop: a system that actually keeps learning

Everything above describes a system that is *trained*, then *serves*. That
is not what "self-learning" means, and the gap was the honest limitation
this document previously carried: Thompson sampling explored, but nothing
ever learned from what the exploration found. Feedback accumulated in
`plan_execution_log` until a human reran `app.train`.

Three components close it. All three are motivated by measurements above
rather than by a feature list.

### Automatic retraining with a champion/challenger gate (`app/retrain.py`)

`python -m app.retrain` checks how many executions have been logged since
the deployed model was trained, and if there is enough new feedback:

1. Trains a **challenger** on everything accumulated.
2. Scores the **champion** (currently served) and the challenger on the
   *same* held-out queries.
3. Promotes only if the challenger wins by more than a margin (default 2%).

The margin is the important part, and §2.2.1 is why it exists. Offline
evaluation here is noisy and optimistically biased; promoting on *any*
measured improvement would mean promoting on noise roughly half the time,
and the model would random-walk rather than improve. Requiring a clear
margin makes the ratchet one-directional in expectation.

Verified end to end: the first retrain promoted (no incumbent to compare
against), and the immediately following retrain was **rejected** --
`"challenger improved by 0.0%, below the 2% bar"`. The system declines to
replace a model with a statistically indistinguishable one, which is the
behaviour that makes unattended retraining safe.

### Versioned models with rollback (`app/model_store.py`)

Previously each train overwrote `models/plan_selector.pkl`. For a system
that retrains itself unattended that is unacceptable -- one bad automated
promotion and the good model is gone. Every trained model is now stored
under a timestamped version with its metrics; promotion is an explicit
recorded act; `--rollback` restores the previous version. Given that §2.2.1
establishes offline scores can mislead, an escape hatch for "this looked
better offline and is worse live" is a requirement, not a nicety.

### Per-query regression guard (`app/optimizer/regression_guard.py`)

The safety veto in `learned.py` is *prospective* -- it judges a plan before
running it, using the cost estimates this project exists because it
distrusts. So it cannot catch a plan that is cheap on paper and slow in
reality, which is precisely the interesting case.

The regression guard is *retrospective*. It reads the system's own history
and finds queries where the learned path has, in measured fact, been slower
than plain native PostgreSQL (beyond a 10% tolerance, over at least 3
executions). Those queries are served the native plan regardless of what
the model currently believes.

On the accumulated history it flags 6 of 25 workload queries, e.g.:

| Query | Native avg | Learned avg | Ratio |
|---|---|---|---|
| `4w_full_clothing_recent` | 209.1 ms | 265.2 ms | 1.27x |
| `4w_full_country_us_electronics` | 106.3 ms | 124.3 ms | 1.17x |
| `4w_full_broad` | 592.3 ms | 682.5 ms | 1.15x |

This targets the deployment blocker named in the recent literature: an
optimizer that is faster on average but occasionally much slower on a
*specific* query is unshippable, because one user-facing query getting 30%
slower outweighs a diffuse average win. Measuring per-query rather than in
aggregate is the entire point -- an aggregate mean hides exactly the queries
that would get someone paged.

The guard is deliberately asymmetric and recoverable: a query must
demonstrate a regression over several executions to be blocked, and it keeps
being re-evaluated from history, so it un-blocks itself if the model
improves. It is a brake, not a ban.

**Does the guard help? Measured, not assumed** -- three paired runs of the
same workload and policy, differing only in `--no-guard`:

| Run | Guard ON | Guard OFF | Delta |
|---|---|---|---|
| 1 | +13.6% | +19.4% | -5.8 pp |
| 2 | +40.4% | +9.3% | +31.1 pp |
| 3 | +34.1% | **-23.7%** | +57.8 pp |
| **mean** | **+29.4%** | **+1.7%** | **+27.7 pp** |

Two things stand out. The guard raised mean captured headroom from ~2% to
~29%. More importantly, **the guarded runs never went negative** (worst
+13.6%) while the unguarded runs did (-23.7%) -- which is precisely the
behaviour it was built for. Blocking the queries with a track record of
regressing removes the left tail, and on this workload the left tail was
large enough that removing it also moved the mean.

Three pairs is still a small sample and run 1 went the other way, so this
is a supported direction rather than a settled effect size. But it is the
first evidence here that a *robustness* mechanism, not a better predictor,
is what turns this from "sometimes faster, sometimes much slower" into
something that could be deployed.

## 2.4.1 Fixing "native sometimes beats the learned path"

The most common complaint about this system was also its most legitimate:
on plenty of queries plain PostgreSQL won. Three attempts were needed, and
the first two failed in instructive ways.

### Attempt 1 -- the design flaw

The optimizer scored the *hinted* candidates and served the argmin. The
native plan was used only for the safety veto; it was never something the
model could choose. So the optimizer was **forced to deviate from PostgreSQL
on every single query**, including the many where PostgreSQL was already
right. On those, deviating can only lose.

The model wasn't choosing native badly. It was never allowed to choose
native at all.

### Attempt 2 -- correct idea, useless result

Native became a first-class candidate, plus a confidence gate: only deviate
if the predicted gain exceeds the model's own uncertainty about that gain
(`gain > z * sigma`). Regressions vanished. So did everything else:

    run 1  0.0%    run 2  0.0%    run 3  0.0%

Served latency equalled native *exactly*, on every run. The gate never
fired. That is not a bug in the gate -- it is the gate correctly reporting
that with ~25 ms prediction error against ~8-20 ms of available gain, the
model could not distinguish any candidate from native. A principled
confidence test on a model this uncertain refuses to act, and it was right
to. Trading regressions for having no optimizer is not a fix.

### Attempt 3 -- fix the target, not the threshold

The real problem was what the model was asked to predict. It regressed on
**absolute milliseconds**, which is wrong twice over:

  1. **Scale dominates.** The workload spans ~5 ms to ~600 ms queries, so
     squared error is overwhelmingly driven by the slow ones. The model
     spent its capacity learning "this query is inherently slow" -- true,
     useless, and nothing to do with plan choice.
  2. **It answers a harder question than we asked.** We never need to know a
     plan will take 213 ms. We only need to know it beats native.

The target is now `log(candidate_latency / native_latency)` for the same
query (`train._relative_targets`). Every query contributes on the same
scale, and the prediction *is* the decision: negative means faster than
native. The gate becomes directly readable -- "is this confidently below
1.0x native?" -- instead of hoping a difference between two noisy absolute
predictions survives subtraction.

One further fix: the gate initially picked the candidate itself, which
silently bypassed the policy and made `risk_averse`, `thompson` and
`pairwise_rank` all behave like `greedy`. Selection (which candidate) and
authorisation (is it worth deviating) are now separate steps.

### Result

| Configuration | Live runs (headroom captured) |
|---|---|
| Absolute target, native not a candidate | +40%, -25%, **-149%** |
| + confidence gate | 0.0%, 0.0%, 0.0% |
| **+ ratio target, policy-aware gate** | **+14.4%, +42.4%, +1.3%** |

All runs positive, no regressions, and the optimizer still acts. The
failure mode is now "no better than native" rather than "worse than
native", which is the property that makes a learned optimizer deployable
at all.

The generalisable lesson is that **the loss function encodes the question**.
Two attempts went into thresholds and safety logic when the actual defect
was that the model was being trained on the wrong quantity. A gate can only
withhold a bad decision; it cannot manufacture a good one.

## 2.4.2 Calibrating the gate instead of guessing it

Fixing the regressions (§2.4.1) produced the opposite complaint: the
optimizer now declined to act on many queries. Both complaints are about the
same knob, and both are legitimate -- a gate set too loose regresses, a gate
set too tight is an expensive no-op.

The right threshold is not something to reason about from first principles,
because it depends entirely on how accurate the model happens to be on the
data in front of it. So `app/calibrate.py` measures it: replay every logged
query at a grid of `(confidence_z, min_relative_gain)` settings and record
how often each deviates, how often those deviations actually regress, and
the net latency saved.

| z | min_gain | deviates | regresses | net gain |
|---|---|---|---|---|
| 0.50 | 0.05 | 63% | **0%** | **+14.5%** |
| 1.00 | 0.05 | 60% | 0% | +14.5% |
| 0.50 | 0.00 | 77% | 9% | +14.0% |
| 0.00 | 0.02 | **87%** | **15%** | +13.9% |

The bottom row is the answer to "why doesn't it optimize more queries."
Forcing the optimizer to act on 87% of queries instead of 63% produces
*less* net improvement (+13.9% vs +14.5%) and introduces a 15% regression
rate. The extra 24 percentage points of activity are all bets the model
wasn't sure about, and they lose slightly more than they win.

So the ~40% of queries where it keeps native are not a failure. They are
queries where PostgreSQL was already right and the model correctly declines
to gamble.

The recommendation is chosen by maximising net improvement **subject to** a
regression-rate bound, not by maximising net improvement alone. Optimising
the mean without that constraint would happily accept "usually much faster,
occasionally catastrophic" -- exactly the per-query instability that makes
learned optimizers undeployable. `app.calibrate --apply` writes the winning
setting to `models/gate.json`, which `LearnedOptimizer` prefers over its
hardcoded defaults, so the threshold is re-derived per dataset rather than
inherited from whatever happened to work here.

Live result with the calibrated gate, three runs:

| Run | Headroom captured | Queries optimized |
|---|---|---|
| 1 | +46.4% | 12 / 25 |
| 2 | +14.3% | 11 / 25 |
| 3 | +17.1% | 10 / 25 |

## 2.5 Working on any dataset, and what that revealed

The system now onboards an arbitrary PostgreSQL database in one command
(`app/onboard.py`): discover schema -> generate workload -> collect -> train.
The last hardcoded piece was the *workload* -- `workload.py` is 25 queries
written for the synthetic schema, which made "works on any dataset" untrue
in practice even though the feature layer was schema-agnostic.

`schema_graph.py` reads tables, columns, indexes and foreign keys and builds
the join graph; `workload_generator.py` walks it for connected table subsets
and writes queries whose predicates are **sampled from the data itself**
(real percentiles for numerics, values actually present for text). Inventing
predicate values is the obvious trap: a filter matching nothing makes every
join order equally instant, and one matching everything makes the filter
irrelevant. Either way the query teaches the model nothing.

**Schemas without declared foreign keys.** JOB/IMDB declares none -- and
neither do plenty of production databases, which enforce integrity in the
application or drop constraints for bulk-load speed. Requiring FKs would
have meant not working on the literature's own benchmark. `infer_foreign_keys`
falls back to naming conventions (`keyword_id` -> `keyword.id`;
`kind_id` -> `kind_type.id` when exactly one table matches the prefix),
requiring integer types on both sides so a coincidental name match can't
manufacture a nonsense join. On JOB this recovers 11 join edges from 21
tables -- enough to generate a workload. Inferred edges are flagged as such
in `/schema` rather than presented as fact.

### The result: same command, two unrelated databases

| | Synthetic e-commerce | Real JOB/IMDB |
|---|---|---|
| Tables | 4 | 21 |
| Join edges | 3 (declared) | 11 (**inferred**) |
| Rows | 755k | 74.2M |
| Native avg | 161.3 ms | 210.9 ms |
| Oracle avg | 110.7 ms | 164.3 ms |
| `greedy` | +21.5% | **-76%** |
| `thompson` | +27.3% | **-661%** |
| `risk_averse` | +31.7% | **+77.2%** |
| `pairwise_rank` | **+59.6%** | -504% |

Both rows are from auto-generated workloads, so this is the system solving a
problem it set itself.

**The disagreement between the two columns is the interesting part.** On the
small, well-sampled synthetic schema, pairwise ranking wins. On JOB -- 21
tables, 74M rows, only 103 training rows after aggregation -- every selector
that trusts its point predictions collapses (`thompson` is 7x *worse* than
native), and only `risk_averse`, which explicitly penalises candidates the
ensemble disagrees about, survives. It captures 77% of headroom there.

That is the clearest evidence in this project for a claim that recurs
throughout it: **when data is thin relative to the problem, modelling
uncertainty matters more than modelling the target.** It also means "which
policy is best" is not a constant -- it depends on the data regime, and a
system that ships a single hardcoded policy will be wrong on half its
deployments. The policy is configurable (`SELECTION_POLICY`) for exactly
this reason.

## 2.6 Production inference: choosing without executing

`/query/analyze` and `app.benchmark` execute *every* candidate and then
report which was fastest. That is a measurement harness, not an optimizer --
it spends N executions to answer a question asked once, so serving traffic
with it would be strictly slower than having no optimizer at all. This was a
named limitation from the first draft of this document.

`optimizer/planner.py` is the real path. For N candidates it issues N
`EXPLAIN`s **without** `ANALYZE` (Postgres plans but runs nothing), scores
them, and executes only the winner. Measured live: **3.8 ms of planning
overhead against 138 ms of execution**, ~2.7%.

This only works because the feature layer never reads actuals.
`plan_tree.py` and `features.py` were deliberately restricted to
estimate-side fields (`Plan Rows`, `Total Cost`, `Plan Width`, node types).
That looked like an arbitrary constraint when written; it is what makes
production inference possible at all.

## 2.7 Regret: the summary number that isn't a summary

"Captured 29% of headroom" describes a run. Cumulative regret -- how much
slower the served plans were than the best available, summed over time --
describes a *trajectory*, and distinguishes a learner that converged from
one making the same mistake repeatedly. Averages hide that completely.

Over 500 logged decisions:

| | Cumulative regret |
|---|---|
| Learned path | 5,016 ms |
| Native PostgreSQL | 6,261 ms |
| **Ratio** | **0.80** |

The learned optimizer has accumulated 20% less regret than always trusting
Postgres. This is the single most defensible number in the project: it is
computed over every decision actually made, not a favourable subset, and a
ratio above 1.0 would have meant the system was actively harmful.

Regret is measurable here only *because* the harness executes every
candidate. In production you never learn what the plans you didn't run would
have cost, so this is an offline diagnostic computed from
`plan_execution_log`, not a live signal.

## 2.8 Learned cardinality correction

Leis et al. showed that PostgreSQL's plans go wrong mainly because its
*cardinality estimates* go wrong, and that the errors compound
multiplicatively across joins. `docs/JOB_RESULTS.md` shows the consequence:
native averages 3703 ms where the best plan averages 919 ms.

Every `EXPLAIN ANALYZE` already in `plan_execution_log` contains both the
predicted (`Plan Rows`) and the actual (`Actual Rows`) count at every node --
abundant, perfectly-labelled supervision for the exact quantity the
optimizer gets wrong, which this project had been discarding.
`optimizer/cardinality.py` learns to predict the log-ratio (q-error) from
pre-execution features, giving a corrected row estimate.

Honest scope: it is wired as an additional *signal*, not fed back into
Postgres's planner. Doing the latter properly means intercepting cardinality
estimation inside the planner (pg_hint_plan's `Rows` hint is the hook);
that is a substantially larger change and is named as future work.

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
- **Retraining is triggered, not scheduled.** §2.4 closes the loop --
  `app.retrain` retrains on accumulated feedback and gates promotion on a
  champion/challenger comparison -- but something still has to *call* it (a
  cron entry, or the dashboard button). There is no background scheduler in
  the container, and no incremental/online update: each retrain is a full
  refit from history.
- **The regression guard's effect size is not settled.** The paired A/B in
  §2.4 supports it clearly (mean +29% vs +2%, and no negative runs), but
  three pairs is a small sample and one of them went the other way. Treat
  the direction as supported and the magnitude as provisional.
- **The guard needs history before it can protect anything.** It blocks on
  measured regressions, so a brand-new query gets no protection on its first
  few executions -- exactly when the model is least informed about it. The
  prospective cost veto is the only cover there.
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
