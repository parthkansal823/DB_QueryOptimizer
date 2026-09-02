# Design notes and results

How this project relates to published work on learned query optimization, what
the evaluation actually measured, and the things that turned out to matter more
than expected.

It is written as a running log rather than a report: several sections record a
result that was wrong, how it was found, and what replaced it, because those
were the most useful parts of building it.

See `ARCHITECTURE.md` for how the system works, `METRICS.md` for how each number
is calculated, and `DASHBOARD.md` for the UI.

## 1. Literature review

Query optimization means picking an execution plan for a declarative query.
It has been cost-based and hand-tuned since System R. The systems below give
three different answers to one question: what should replace, or add to, the
cost-based optimizer's choice?

**Native Postgres (cost-based optimizer, CBO).** Uses dynamic programming
over join orders for small queries, falling back to a genetic algorithm
(GEQO) past `geqo_threshold` tables. It is driven by selectivity estimates
from per-column histograms, and it assumes filters are independent of each
other. It needs no training and has been tuned for decades. But the
independence assumption breaks down when filters are correlated or the data is
skewed, and row-count errors then multiply across joins. This is exactly the
failure mode documented in Leis et al., ["How Good Are Query Optimizers,
Really?"](http://www.vldb.org/pvldb/vol9/p204-leis.pdf) (VLDB 2015), the paper
that introduced the Join Order Benchmark this project's stretch goal loads
(Section 4).

**Neo** (Marcus et al., ["Neo: A Learned Query Optimizer"](https://arxiv.org/abs/1904.03711),
VLDB 2019). Builds a full plan from the bottom up with a neural network, a
tree-convolutional value network trained by experience replay against real
execution time. It starts from an existing optimizer's plans and improves
through exploration. Its action space covers essentially everything a
classical optimizer controls, learned end to end: join order, join method and
access path. That gives it the highest ceiling of the three. It also makes it
the slowest to converge, the hungriest for execution feedback, and the hardest
to debug when it fails, because a bad exploration step can regress badly
before the model corrects itself.

**Bao** (Marcus et al., ["Bao: Making Learned Query Optimization
Practical"](https://arxiv.org/abs/2004.03814), SIGMOD 2021). Treats the
problem as a contextual bandit over a small, fixed set of coarse query hints:
about 48 hint-sets in the paper, each switching whole classes of operators on
or off. A tree-convolutional network predicts which hint-set will give the
best plan for a query, with Thompson sampling for exploration and a safe
fallback to the native optimizer.

The small, bounded action space is the whole point. It makes the system safe
to run in production, because the worst case is a plan the native optimizer
could have produced anyway, and it makes retraining fast. **This project
follows Bao's framing directly**, choosing among hints rather than building
plans. It narrows the action space further, to join order plus join method as
a stretch goal instead of Bao's operator-class toggles, and swaps the
contextual bandit for a simpler regression model. That trades Bao's online
regret bounds for something easier to train and explain on a small amount of
data.

**This project.** Candidates come from `pg_hint_plan` `Leading()`
join-order hints: every permutation for 5 tables or fewer, random sampling
above that -- see §4 for why a learned candidate generator is the way past
that. The stretch goal adds forced
join-method hints (`HashJoin`, `NestLoop`, `MergeJoin`) at every prefix of the
join order.

Each candidate is featurized two ways. Per-table slots carry selectivity, join
position and scan type. *Plan-tree structure* carries depth, bushiness,
estimated blowup of intermediate results, and operator mix
(`backend/app/optimizer/plan_tree.py`). The second is a cheap stand-in for the
tree convolution Neo and Bao use.

Selection uses a **bootstrapped ensemble** of LightGBM regressors
(`backend/app/optimizer/bandit.py`) with three policies: `greedy` takes the
argmin of the mean; `thompson` samples one ensemble member per decision, which
is Bao's bootstrapped Thompson sampling and gives real exploration; and
`risk_averse` takes the argmin of mean + λ·σ, penalising plans the ensemble
disagrees about.

A **safety veto** throws out any learned pick the planner costs far above the
native plan, so the system cannot knowingly serve a large regression. That is
Bao's central practical claim. On a cold start it falls back to a cost
heuristic -- the lowest Postgres-*estimated* cost -- until a model has been
trained.

| | Native Postgres CBO | Neo | Bao | This project |
|---|---|---|---|---|
| **Approach** | Cost-based DP / GEQO | Learned, builds full plan | Learned, picks among coarse hint-sets | Learned, picks among join-order/-method hints |
| **Action space** | Full plan space (implicit) | Full plan space (explicit search) | ~48 fixed hint-sets | Join-order permutations x join-method (this schema: <=20/query) |
| **Training signal** | None (static heuristics + stats) | Real latency, experience replay | Real latency, contextual bandit reward | Real latency, supervised regression |
| **Plan encoding** | N/A (cost formulas) | Tree convolution over plan tree | Tree convolution over plan tree | Per-table slots + hand-built plan-tree structure features |
| **Exploration** | None | Epsilon-greedy over tree search | Bootstrapped Thompson sampling | Bootstrapped Thompson sampling (same mechanism, smaller action space) |
| **Uncertainty** | None | Implicit | Ensemble spread | Ensemble spread (drives the risk-averse policy) |
| **Safety** | N/A | None explicit | Falls back to native optimizer | Cost-ratio veto vs. the native plan |
| **Cold start** | N/A | Bootstrapped from CBO plans | Falls back to native optimizer | Falls back to a cost heuristic |
| **Key strength** | Zero training cost, decades of tuning | Highest ceiling -- learns operators too | Bounded risk, safe in production | Simple to train and explain; oracle-relative evaluation; query-level split |
| **Key limitation** | Cardinality-estimation errors compound across joins | Slow to converge, large feedback requirement | Coarser control (whole operator classes, not per-join) | Prediction error exceeds available headroom at this data scale (see §2) |

## 2. Results

### 2.0 A correctness bug that invalidated an earlier set of results

Worth recording, because it is the single most important thing this project
learned, and because the failure was **silent**.

`pg_hint_plan` installs its planner hooks when the library is *loaded*, so
it must appear in `shared_preload_libraries`. `postgres/init/01_extensions.sql`
ran `CREATE EXTENSION pg_hint_plan`, which registers the SQL objects but
does **not** preload the library. Separately, `plan_extractor.get_plan`
wrapped queries as `EXPLAIN (...) /*+ Leading(...) */ SELECT ...`, putting
the hint *after* the `EXPLAIN` keyword, where pg_hint_plan does not look for
it.

Either bug on its own is enough to make every hint do nothing. And a hint
that cannot be applied is just a SQL comment: it raises no error and prints no
warning. The pipeline ran happily and produced believable numbers for the
whole candidate set, while **every "candidate" was byte-identical to the
native plan**. The measured "improvements" were run-to-run timing noise.

The tell was that all 8 candidates for a 4-table query reported the exact
same `Total Cost` (16134) and the same join order. Both bugs are fixed
(`shared_preload_libraries=pg_hint_plan` in `docker-compose.yml`; hint
hoisted ahead of `EXPLAIN` in `plan_extractor._split_hint`) and
`tests/test_plan_extractor.py::test_hint_is_hoisted_ahead_of_the_explain_keyword`
guards the regression. Data collection time went from 51s to 366s once the
hints actually bound -- candidates were finally *different plans*, some of
them slow. **Everything in §2.1/§2.2 below is from after the fix.**

The general lesson: an experiment whose treatment silently does nothing
still produces a full set of numbers. Checking that the intervention *changed
anything at all* (here, do the candidate plans actually differ?) belongs in
the pipeline, not in a reviewer's intuition.

#### A correction to the diagnosis above

The paragraph about hint placement is wrong, and finding that out took
measuring it, which is the point.

On pg_hint_plan 1.6.3 -- the version pinned in `postgres/Dockerfile` -- a hint
placed *after* the `EXPLAIN` keyword binds exactly as well as one placed
before it. Measured on a 4-table query:

| | Total cost |
|---|---|
| No hint | 11,699 |
| `Set(enable_hashjoin off)` before `EXPLAIN` | 42,094 |
| `Set(enable_hashjoin off)` after `EXPLAIN` | 42,094 |
| `Leading(u p oi o)` before `EXPLAIN` | 2.00000e10 |
| `Leading(u p oi o)` after `EXPLAIN` | 2.00000e10 |

So "either bug alone is enough to make every hint a no-op" is false. There was
one bug, not two: the missing `shared_preload_libraries`. Hoisting the hint in
`plan_extractor._split_hint` fixed nothing, and the fact that the results
improved after doing both changes made it look causal.

This is the same error as the original, one level up. The diagnosis of a
silent failure was itself never checked against a planner -- two plausible
causes were identified, both were fixed at once, the numbers moved, and the
explanation was written up as established. `_split_hint` stays, because
hoisting is harmless and other pg_hint_plan versions may genuinely require it,
but it is defensive rather than load-bearing.

Both claims are now pinned by
`tests/test_integration_hints.py`, which asserts against a live planner that
hints bind, that the candidate set contains genuinely different plans, and
that placement does not matter on this version. Those are the tests that would
have caught §2.0 in the first place, and their absence is why a unit suite of
200 passing tests could sit on top of a pipeline whose central mechanism did
nothing.

A third trap surfaced while writing them: pg_hint_plan accepts
`Leading(a b c)` but **silently ignores** `Leading(((a b) c))`. A generator
emitting the nested form would produce an action space of identical plans and
no error at all -- §2.0 exactly, waiting to happen again. `hints.py` emits the
flat form; a test now pins that it must.

### 2.1 Offline evaluation (`app.train`, query-level held-out split)

Trained on `plan_execution_log` after data collection (25 workload
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
| Avg. latency -- cost heuristic | 107.4 ms | 107.9 ms |
| Avg. latency -- **oracle (best possible)** | 100.9 ms | 86.3 ms |
| Avg. latency -- learned, `greedy` | 105.5 ms | 98.9 ms |
| Avg. latency -- learned, `thompson` | 107.7 ms | 101.9 ms |
| Avg. latency -- learned, `risk_averse` | 106.0 ms | **92.1 ms** |
| Headroom captured -- `risk_averse` | 32% | **74%** |

Model: bootstrapped ensemble of 8 LightGBM regressors, 41 features, split at
the query level (6 held-out queries).

Tripling the repetitions nearly halved the prediction error. `risk_averse`,
which penalises candidates the ensemble disagrees about, consistently beats
`greedy`. That is what theory predicts when a model is accurate on average but
unreliable in places: distrusting your own high-variance predictions is worth
more than chasing their minimum. `thompson` trailing both is expected too. It
*spends* performance to gather information, which is the trade exploration
makes.

**A fourth selector: pairwise ranking.** Prediction error here is about as
large as the spread between candidates, which means predicting absolute
latency is a harder problem than we actually need to solve.
`optimizer/ranker.py` uses the Lero (VLDB 2023) approach instead: a classifier
trained on *pairs* of candidates from the same query, answering only "is A
faster than B?". At inference each candidate is scored by how many pairwise
duels it wins. On median-aggregated labels it is the strongest offline
selector:

| Selector | Headroom captured (held-out) |
|---|---|
| `greedy` | -52% |
| `thompson` | -51% |
| `risk_averse` | +31% |
| **`pairwise_rank`** | **+46%** |

Worth noting honestly: `pairwise_rank` has the best *average* latency but
the lowest rate of beating the heuristic, at 33% of queries. It wins big when
it wins, and loses often but cheaply. That is a different risk profile from
`risk_averse`, which wins on 83% of queries for a smaller average gain. Which
you prefer depends on whether you care about mean latency or about per-query
predictability.

### 2.2 Live benchmark: the instability, and what fixed it

Re-running the full workload live (`app.benchmark`) is the real test,
because it re-executes every plan instead of replaying logged timings. Three
runs per policy, showing the share of oracle headroom captured (negative means
worse than plain native):

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

This is the most useful result in the project, and it is a *data* fix, not a
model fix. Training on single noisy executions taught the model that whichever
candidate got the luckiest timing was genuinely the fastest. It then served
those plans confidently, and sometimes lost badly. Taking the median of three
executions per candidate removed that failure mode completely. The model
architecture did not change at all.

This connects directly to where the research is heading. Recent work targets
the *per-query instability* that makes learned optimizers hard to deploy: an
optimizer that is 30% faster on average but occasionally 150% slower cannot be
shipped. The evidence here says a meaningful share of that instability can be
measurement noise in the training labels, rather than anything inherent to the
policy.

### 2.2.1 Offline evaluation is optimistically biased

One caveat survives the fix. The offline numbers in §2.1 are computed by
replaying *logged* latencies. For a held-out query, the selector picks among
candidate latencies that were each measured once, and is then scored against
those same recorded numbers. So it gets rewarded partly for spotting which
candidate got the luckiest measurement. The oracle column shares the bias:
"best possible" really means "best single sample observed."

Live re-execution consistently comes in below the offline estimate. **A
query-level train/test split does not fix this.** The split stops leakage
*between* queries, but the noise lives *inside* each query's candidate
measurements. Only re-execution measures the thing that matters.

The general lesson goes well beyond this project. When label noise is as
large as the effect you are trying to measure, held-out evaluation on logged
outcomes will flatter you. Only re-execution tests what you actually care
about.

The safety veto fired on 2 to 5 of the 25 queries per run, and it was doing
real work. Now that hints bind, some forced join orders carry Postgres's
`disable_cost` penalty of about 1e10, which signals a cartesian product. Those
are thrown out before execution rather than served.

### 2.2.2 Where this leaves things

**On the synthetic schema it works.** Every live run beats native
PostgreSQL, averaging about 26% of the available oracle headroom with no
regressions (§2.2). The honest qualifiers: the workload is small at 25
queries, the absolute headroom is modest at roughly 8 ms per query, and the
measurements were taken on a laptop.

**On the real JOB benchmark it does not** (`docs/JOB_RESULTS.md`). There,
75% of the latency is available and the system captures none of it, because
194 executions across 8 queries cannot teach a 17-table join space. This is
the more informative of the two results: it puts a number on how far the
system is from the scale the research operates at.

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
diagnosable. Without an oracle column, "3% better than native" tells you
nothing about whether 3% was all there was or a small slice of a large
opportunity.

### 2.3 JOB/IMDB stretch goal

The real 21-table, roughly 74.5M-row IMDB dataset was downloaded and
imported in full (`data/job/load_job.sh`). A smoke test then ran 8 real JOB
queries through the unchanged `app.collect_data` and `app.train` pipeline,
with `DATABASE_URL` pointed at the `job` database and zero code changes.

It confirmed the pipeline works against a real research benchmark:
`schema_introspection` found all 21 tables automatically, and both the
heuristic and the trained model kept Postgres's own plan on both held-out
queries. So there was no regression, though two held-out queries is too small
a sample to claim anything more than that.

`docs/JOB_RESULTS.md` has the full details, including what a complete JOB
evaluation would still need: all 113 queries, more repetitions, and
join-method candidates enabled. `data/job/README.md` has the load procedure,
and Section 3 explains why the pipeline needed no code changes.

## 2.4 Closing the loop: a system that actually keeps learning

Everything above describes a system that is *trained*, and then *serves*.
That is not what "self-learning" means, and the gap was an honest limitation
this document used to carry. Thompson sampling explored, but nothing ever
learned from what it found. Feedback piled up in `plan_execution_log` until a
human reran `app.train`.

Three components close it. All three are motivated by measurements above
rather than by a feature list.

### Automatic retraining with a champion/challenger gate (`app/retrain.py`)

`python -m app.retrain` checks how many executions have been logged since
the deployed model was trained, and if there is enough new feedback:

1. Trains a **challenger** on everything accumulated.
2. Scores the **champion** (currently served) and the challenger on the
   *same* held-out queries.
3. Promotes only if the challenger wins by more than a margin (default 2%).

The margin is the important part, and §2.2.1 explains why it exists.
Offline evaluation here is noisy and biased towards optimism. Promoting on
*any* measured improvement would mean promoting on noise about half the time,
and the model would wander rather than improve. Requiring a clear margin makes
the ratchet turn one way.

Checked end to end: the first retrain promoted, since there was no
incumbent to compare against, and the very next retrain was **rejected** with
`"challenger improved by 0.0%, below the 2% bar"`. The system refuses to
replace a model with one it cannot tell apart, and that refusal is what makes
unattended retraining safe.

### Versioned models with rollback (`app/model_store.py`)

Each train used to overwrite `models/plan_selector.pkl`. For a system that
retrains itself unattended, that is unacceptable: one bad automatic promotion
and the good model is gone. Every trained model is now saved under a
timestamped version with its metrics, promotion is an explicit recorded act,
and `--rollback` restores the previous version. Since §2.2.1 shows offline
scores can mislead, an escape hatch for "this looked better offline and is
worse live" is a requirement, not a nicety.

### Per-query regression guard (`app/optimizer/regression_guard.py`)

The safety veto in `learned.py` looks *forward*. It judges a plan before
running it, using the very cost estimates this project exists because it
distrusts. So it cannot catch a plan that is cheap on paper and slow in
reality, which is exactly the interesting case.

The regression guard looks *backward*. It reads the system's own history
and finds queries where the learned path has actually measured slower than
plain native PostgreSQL, by more than a 10% tolerance and over at least 3
executions. Those queries get the native plan no matter what the model
currently believes.

On the accumulated history it flags 6 of 25 workload queries, e.g.:

| Query | Native avg | Learned avg | Ratio |
|---|---|---|---|
| `4w_full_clothing_recent` | 209.1 ms | 265.2 ms | 1.27x |
| `4w_full_country_us_electronics` | 106.3 ms | 124.3 ms | 1.17x |
| `4w_full_broad` | 592.3 ms | 682.5 ms | 1.15x |

This targets the deployment blocker the recent literature names. An
optimizer that is faster on average but occasionally much slower on a
*specific* query cannot be shipped, because one user-facing query getting 30%
slower outweighs a diffuse average win. Measuring per query rather than in
aggregate is the entire point: an average hides exactly the queries that would
get someone paged.

The guard is deliberately asymmetric, and a query can recover. It has to
show a regression over several executions before it is blocked, and it keeps
being re-checked against history, so it unblocks itself if the model improves.
It is a brake, not a ban.

**Does the guard help? Measured, not assumed** -- three paired runs of the
same workload and policy, differing only in `--no-guard`:

| Run | Guard ON | Guard OFF | Delta |
|---|---|---|---|
| 1 | +13.6% | +19.4% | -5.8 pp |
| 2 | +40.4% | +9.3% | +31.1 pp |
| 3 | +34.1% | **-23.7%** | +57.8 pp |
| **mean** | **+29.4%** | **+1.7%** | **+27.7 pp** |

Two things stand out. The guard raised mean captured headroom from about 2%
to about 29%. More importantly, **the guarded runs never went negative**, with
a worst case of +13.6%, while the unguarded runs did, at -23.7%. That is
exactly the behaviour it was built for. Blocking the queries with a track
record of regressing removes the left tail, and on this workload that tail was
big enough that removing it also moved the mean.

Three pairs is a small sample, and run 1 went the other way, so this is a
supported direction rather than a settled effect size. Even so, it is the
first evidence here that a *robustness* mechanism, rather than a better
predictor, is what turns this from "sometimes faster, sometimes much slower"
into something you could deploy.

**Why it stayed at three, and what replaced the excuse.** `app.benchmark`
printed its numbers and returned nothing, so repeating it meant copying figures
off a terminal by hand. Three runs is about as many as anyone does that way,
and the sample size was set by the ergonomics of the tool rather than by the
question.

`app.experiment` now runs the comparison properly: arms interleaved so drift
hits both equally, paired differences rather than two independent means,
percentile-bootstrap confidence intervals (latency is too right-tailed to
assume normality), and a sign test -- deliberately the weakest test available,
because with runs this noisy anything stronger would report more confidence
than the data holds. Failed runs discard the whole pair rather than silently
unpairing the comparison, and results checkpoint after every pair so an
hour-long experiment survives a dropped connection.

A smoke run of the harness (4 pairs, first 6 queries only, `pairwise_rank`)
already shows why the tooling mattered:

| arm | mean | 95% CI | negative runs |
|---|---|---|---|
| guard on | 80.9% | [74.1, 85.0] | 0/4 |
| guard off | 74.0% | [69.5, 78.5] | 0/4 |

Paired difference **+6.9 pp, 95% CI [-2.6, +14.0], sign test p = 0.63**. The
mean points the same way as the original result; the interval spans zero, so
the verdict is *unresolved*.

That is not a refutation of §2.4 -- it is 4 pairs on a 6-query subset, a
smaller and easier sample than the 3 pairs on the full workload above, and the
guard has fewer queries to block. What it does establish is that the honest
verdict on this evidence is "unresolved", and that a mean of +6.9 or +27.7
should not be read as an effect until an interval says so. The full experiment
(`--runs 20`, whole workload, roughly two hours) is the one that settles it and
has not been run here.

## 2.4.1 Fixing "native sometimes beats the learned path"

The most common complaint about this system was also the most fair: on
plenty of queries, plain PostgreSQL won. Three attempts were needed, and the
first two failed in useful ways.

### Attempt 1 -- the design flaw

The optimizer scored the *hinted* candidates and served the best of them.
The native plan was used only by the safety veto; the model could never choose
it. So the optimizer was **forced to deviate from PostgreSQL on every single
query**, including the many where PostgreSQL was already right. On those,
deviating can only lose.

The model wasn't choosing native badly. It was never allowed to choose
native at all.

### Attempt 2 -- correct idea, useless result

Native became a first-class candidate, plus a confidence gate: only deviate
if the predicted gain exceeds the model's own uncertainty about that gain
(`gain > z * sigma`). Regressions vanished. So did everything else:

    run 1  0.0%    run 2  0.0%    run 3  0.0%

Served latency equalled native *exactly* on every run. The gate never
fired. That is not a bug in the gate. It is the gate correctly reporting that
with about 25 ms of prediction error against 8 to 20 ms of available gain, the
model could not tell any candidate apart from native. A sound confidence test
on a model this uncertain refuses to act, and it was right to. But trading
regressions for having no optimizer at all is not a fix.

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
query (`train._relative_targets`). Every query now contributes on the same
scale, and the prediction *is* the decision: negative means faster than
native. The gate becomes something you can read directly, "is this confidently
below 1.0x native?", instead of hoping that the difference between two noisy
absolute predictions survives subtraction.

One further fix. The gate originally picked the candidate itself, which
quietly bypassed the policy and made `risk_averse`, `thompson` and
`pairwise_rank` all behave like `greedy`. Choosing a candidate and authorising
the deviation are now two separate steps.

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

Fixing the regressions in §2.4.1 produced the opposite complaint: the
optimizer now refused to act on many queries. Both complaints are about the
same knob, and both are fair. A gate set too loose causes regressions; a gate
set too tight is an expensive way to do nothing.

The right threshold cannot be worked out from first principles, because it
depends entirely on how accurate the model happens to be on the data in front
of it. So `app/calibrate.py` measures it. It replays every logged query across
a grid of `(confidence_z, min_relative_gain)` settings and records how often
each setting deviates, how often those deviations actually regress, and how
much time is saved overall.

| z | min_gain | deviates | regresses | net gain |
|---|---|---|---|---|
| 0.50 | 0.05 | 63% | **0%** | **+14.5%** |
| 1.00 | 0.05 | 60% | 0% | +14.5% |
| 0.50 | 0.00 | 77% | 9% | +14.0% |
| 0.00 | 0.02 | **87%** | **15%** | +13.9% |

The bottom row answers "why doesn't it optimize more queries?". Forcing the
optimizer to act on 87% of queries instead of 63% produces *less* net
improvement, +13.9% against +14.5%, and adds a 15% regression rate. The extra
24 percentage points of activity are all bets the model was unsure about, and
they lose slightly more than they win.

So the roughly 40% of queries where it keeps the native plan are not a
failure. They are queries where PostgreSQL was already right, and the model
correctly declined to gamble.

The recommended setting maximises net improvement **subject to** a limit on
the regression rate, rather than maximising net improvement alone. Optimising
the mean without that limit would happily accept "usually much faster,
occasionally catastrophic", which is exactly the per-query instability that
makes learned optimizers undeployable. `app.calibrate --apply` writes the
winning setting to `models/gate.json`, and `LearnedOptimizer` prefers that
over its built-in defaults. The threshold is therefore re-derived for each
dataset instead of being inherited from whatever happened to work here.

## 2.4.0 The benchmark schema had no correlations in it

This section is placed before the others in 2.4 because it invalidates parts of
what they measured, and because it is the most embarrassing thing in this
document.

The schema in `data/schema.sql` was rebuilt specifically to be hard (§2.9). Its
whole purpose is correlated columns — `city` determines `country`, `brand`
determines `category` — because PostgreSQL estimates `WHERE a = x AND b = y` as
sel(a) x sel(b), which is only right when the columns are unrelated. Version 1
of the workload had 6.5% mean available headroom and nothing for a plan-picker
to find; version 2 was built around that failure and reported 22.2%.

The seed data never had those correlations, because it never had any variety.

```
SELECT count(DISTINCT city)     FROM users;      -- 1
SELECT count(DISTINCT country)  FROM users;      -- 1
SELECT count(DISTINCT brand)    FROM products;   -- 1
SELECT count(DISTINCT category) FROM products;   -- 1
SELECT count(DISTINCT status)   FROM orders;     -- 1
```

One value per column, across 50,000 users and 20,000 products. Every user lived
in London; every product was a Voltix; every order was completed. A filter on
any of those columns matched either every row or none.

### The bug

All three inserts that build a correlated pair drew their values like this:

```sql
FROM generate_series(1, 50000) AS i
CROSS JOIN LATERAL (
    SELECT country, city, language
    FROM (VALUES ('IN','Mumbai','hi'), ('UK','London','en'), ...) AS v
    ORDER BY random()
    LIMIT 1
) AS c
```

Which reads as "pick a random row for each generated row", and is not. The
subquery references nothing from the outer row, so it is not correlated in the
SQL sense, and PostgreSQL is free to evaluate it **once** and reuse the result
for all 50,000 — which it does.

`random()` being volatile is not enough; volatility governs whether the planner
may cache a value *within* one evaluation, not how many times an uncorrelated
subquery is evaluated. Both of the obvious rewrites were tested and are also
wrong: pushing the `random()` into a `LATERAL (SELECT ... AS idx)` collapses the
same way. What works is computing the index in a plain subquery over the series
and subscripting parallel arrays with it, which is what the file now does. One
index shared across the columns keeps them correlated; computing it per row is
what gives them variety.

### What it explains

The dashboard had been reporting estimates like *"it expected 75 rows and got
0"* and *"expected 4 rows and got 0"*, which reads as a spectacular
mis-estimate and is really the generated workload filtering on `city =
'Mumbai'` in a database with no Mumbai in it. Every such query returned nothing.

It also means the model was trained on a workload where every categorical
predicate was all-or-nothing. There was no independence-assumption error to
learn, because there were no two columns whose joint selectivity could differ
from the product of the marginals. Whatever the model learned, it was not the
thing this project exists to learn.

### After the fix

| | before | after |
|---|---|---|
| `users.city` distinct values | 1 | 11 |
| `users.country` | 1 | 5 |
| `products.brand` | 1 | 10 |
| `orders.status` | 1 | 4 |

And the trap the schema was designed around is measurable again:

```
city = 'Mumbai' AND country = 'IN'       planner expected 7722, got 14919   (1.9x)
brand = 'Hearthware' AND category = 'home'  expected 421, got 2051          (4.9x)
```

with `CREATE STATISTICS` closing it exactly:

```
brand = 'Hearthware' AND category = 'home'  expected 2051, got 2051         (1.0x)
```

which is the whole argument for `app/dbadvisor/extended_stats.py`
(`docs/DATABASE_OPTIMIZATION.md` §1).

### What this costs the rest of the document

Every measurement in §2.1 through §2.9 was taken against the degenerate
database. The *mechanisms* they describe are unaffected — the pg_hint_plan
binding bug in §2.0 was real, the unpaired-averages bug in §2.2 was real, the
median-aggregation finding in §2.3 was real — but the headline numbers attached
to them are measurements of a database that did not contain the phenomenon they
were about.

They are left in place rather than deleted, with this section pointing at them,
because rewriting them would need the whole sequence re-run and because a result
that turned out to rest on bad data is itself the finding.

## 2.4.3 How the calibrated gate ended up switching itself off

§2.4.2 describes `app.calibrate` as bounding the regression rate first and
maximising net improvement within that. Deployed, it wrote this to
`models/gate.json`:

```json
{"confidence_z": 0.0, "min_relative_gain": 0.0,
 "expected_deviation_rate": 0.78, "expected_regression_rate": 0.11}
```

`confidence_z = 0.0` multiplies the model's uncertainty by zero. The gate was
`predicted_ratio < 1.0` -- deviate whenever the point estimate is any amount
below native, with no margin and no reference to how sure the model is. The
dashboard reported it faithfully and nobody read it: "Predicted speed vs
PostgreSQL: 0.92x (**pessimistically 0.92x**)". The pessimistic figure equalling
the estimate exactly is what an inert safety mechanism looks like.

It showed up as a served plan predicted at 0.92x that ran at 1.5x native
(2.04 ms to 3.05 ms). Replaying that decision under both gates:

| gate | pessimistic bound | required | outcome |
|---|---|---|---|
| as deployed (z=0, min_gain=0) | 0.92x | < 1.00x | **deviates** |
| defaults (z=1.0, min_gain=0.05) | 1.10x | < 0.95x | holds native |

Three separate faults produced it.

**The bound never bound.** `max_regression_rate` defaulted to 0.34. Every
setting on the grid passed, so the constraint the module docstring calls "the
important one" excluded nothing and the recommendation collapsed to plain
`max(net_improvement)`.

**Ties in a sum are not ties.** The winning and runner-up settings were

    z=0.00 min_gain=0.00 -> 83% deviate, 10% regress, +35.0%
    z=0.00 min_gain=0.05 -> 58% deviate,  0% regress, +34.6%

Net improvement is a sum dominated by a handful of expensive queries, so four
tenths of a percentage point of it is noise. A 0% to 10% regression rate is not.
The sweep now treats settings within 5% of each other as equivalent and breaks
the tie on regression rate, then on the worst single regression, then toward
the stricter gate -- strictness being free among equals and not free later,
because the sample the gate was chosen on is small.

**It was calibrated in-sample.** The sweep replayed *every* logged query,
almost all of which the model had trained on, so a gate with no margin was
scored against predictions the model had already fitted. §2.2.1 documents how
optimistic that is; this is the same bias steering a safety threshold. The
bundle now records which queries were held out and the sweep scores only those,
falling back to the whole set -- and saying so -- when too few exist to
distinguish settings.

Re-run out-of-sample, the same grid shows what was hidden: `z=0, min_gain=0`
regresses on **17%** of held-out queries with a worst case of 1.23x, and the
recommendation moves to `min_gain=0.10` at 0% regressions.

## 2.4.4 Why `confidence_z` did nothing, and what it cost

The sweep above had a stranger symptom: the whole `z` column was identical from
0.0 to 1.0. The knob could not change a single decision, which is why the
calibration kept setting it to zero -- correctly, given what it could measure.

The reason is a units error rather than a bug. Measured out-of-bag on this
model:

| | log-ratio units |
|---|---|
| mean \|out-of-bag residual\| | 0.143 |
| mean out-of-bag sigma | 0.071 |
| **spearman(sigma, \|residual\|)** | **+0.37** |

The rank correlation is the important one, and it is the good news: the
ensemble genuinely does know where it is unsure. It just reports it in units
about **2.4x** smaller than the error it is meant to bound, so a full unit of
`z` bought less than half a standard deviation of protection -- too little to
flip any decision, hence the flat column.

`conformal.sigma_scale` measures that factor from the same out-of-bag residuals
and `learned.py` applies it before the gate reads sigma. `confidence_z` is then
denominated in standard deviations, and the sweep immediately becomes
informative: deviation rate falls 83% -> 67% -> 50% across z = 0, 0.25, 0.5,
where it had previously not moved at all.

This is the mild counterpart to §2.4.5's conformal result. Conformal corrects
the *tail* and needs q = 10.5, which is rigorous and unusable; the mean-matched
scale corrects the *centre* at 2.4x, which is approximate and actionable. They
answer different questions and both are kept.

## 2.4.5 The gate's uncertainty was overconfident by 8x

§2.4.2 measures the *threshold* but leaves the quantity it is applied to
unexamined. The gate is

    predicted_ratio + z * sigma  <  1 - min_relative_gain

and `sigma` is the spread across the bootstrapped ensemble. Sweeping `z`
assumes that spread is a calibrated standard deviation. Nothing had checked it.

`optimizer/conformal.py` checks it. Split conformal prediction (Vovk et al.;
Lei et al., JASA 2018) turns any point predictor into one with a
distribution-free, finite-sample bound: score held-out examples by how wrong
the model was, take the (1-alpha) quantile, use it as the multiplier. Ordinary
split conformal spends training data on calibration, which is the worst thing
to spend here (§2.3), so the residuals come from the ensemble's own
**out-of-bag** predictions instead -- each member is fit on a bootstrap
resample, so ~37% of rows are unseen by any given member, and averaging only
the members that missed a row gives an honest held-out prediction for every
training row at no cost in data. That is jackknife+-after-bootstrap (Kim, Xu &
Barber, NeurIPS 2020), which exists for ensembles built exactly this way.

These figures come from a **separate calibration run** on this machine (936
raw executions across 17 queries, 237 training rows after median aggregation),
not from the larger sweep behind 2.4.2's table -- so read the two as answering
different questions rather than as one series. The direction of the finding is
not sensitive to that: a smaller sample makes the conformal multiplier *more*
conservative, not less, so a fuller collection moves `q` down, not up.

229 of 237 training rows (96.6%) produced an out-of-bag residual, so the
no-cost claim holds. The calibration then returns:

| | value |
|---|---|
| alpha | 0.10 |
| **conformal multiplier q** | **8.51** |
| calibration rows | 229 |
| held-out coverage (target 0.90) | 1.00 |

**q = 8.51 against an assumed z = 1.0.** Covering 90% of outcomes takes eight
and a half sigmas, not one. The reason is in the same eval output:

| | log-ratio units |
|---|---|
| mean ensemble spread, held-out | 0.175 |
| actual mean absolute error | 0.403 |

The members agree with each other about 2.3x more than they agree with
reality. Bootstrap resampling measures how much the fit moves when the *sample*
changes; it is blind to model bias and to the irreducible measurement noise
§2.3 spends most of its length on, and here those dominate. So `mean + 1*sigma`
was never the 84% bound a normal reading implies -- and the deployed
`gate.json`, at `z = 0.0`, is not a bound at all.

### Why it is off by default

Applied honestly, the gate stops working. At the model's typical uncertainty a
90% bound requires a predicted speedup of about 4.2x before deviating, which
this workload essentially never offers, so the optimizer would hold native on
almost every query. Turning it on would not make the system safer in any useful
sense; it would make it inert.

That is worth stating plainly rather than burying, because it is the same
finding as §2.8.1 in a different guise: the honest version of a mechanism can
be correct and still not pay. It also converts the limitation this writeup has
been asserting -- "prediction error is comparable to the available headroom at
this data scale" -- into something measured, with a coverage guarantee behind
it, rather than an impression from a table of averages.

The actionable reading is not "use conformal". It is that the deviation rate in
§2.4.2 rests on an uncertainty estimate that is wrong by most of an order of
magnitude, and that closing *that* gap -- more data, or an uncertainty estimate
that models noise rather than resampling -- is the prerequisite for raising the
deviation rate safely. The sweep in §2.4.2 found a threshold that works; it
could not have told us the quantity underneath it was miscalibrated.

`GATE_MODE=conformal` selects it; `CONFORMAL_ALPHA` sets the risk level, applied
when the model trains. Bundles trained before it existed fall back to `fixed_z`.

Live result with the calibrated gate, three runs:

| Run | Headroom captured | Queries optimized |
|---|---|---|
| 1 | +46.4% | 12 / 25 |
| 2 | +14.3% | 11 / 25 |
| 3 | +17.1% | 10 / 25 |

## 2.5 Working on any dataset, and what that revealed

The system now sets up any PostgreSQL database in one command
(`app/onboard.py`): discover the schema, generate a workload, collect data,
train. The last hardcoded piece was the *workload*. `workload.py` holds 25
queries written for the synthetic schema, which made "works on any dataset"
untrue in practice, even though the feature layer was already
schema-agnostic.

`schema_graph.py` reads tables, columns, indexes and foreign keys and builds
the join graph. `workload_generator.py` walks that graph for connected sets of
tables and writes queries whose filters are **sampled from the data itself**:
real percentiles for numbers, values that actually appear for text.

Inventing filter values is the obvious trap. A filter matching nothing makes
every join order equally instant; a filter matching everything makes the
filter irrelevant. Either way, the query teaches the model nothing.

**Schemas without declared foreign keys.** JOB/IMDB declares none, and
neither do plenty of production databases, which enforce integrity in the
application or drop constraints to speed up bulk loads. Requiring foreign keys
would have meant not working on the literature's own benchmark.

`infer_foreign_keys` falls back to naming conventions instead: `keyword_id`
maps to `keyword.id`, and `kind_id` maps to `kind_type.id` when exactly one
table matches the prefix. It requires integer types on both sides, so a
coincidental name match cannot invent a nonsense join. On JOB this recovers 11
join edges across 21 tables, which is enough to generate a workload. Inferred
edges are labelled as inferred in `/schema` rather than presented as fact.

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
small, well-sampled synthetic schema, pairwise ranking wins. On JOB, with 21
tables, 74M rows and only 103 training rows after aggregation, every selector
that trusts its own point predictions collapses. `thompson` ends up 7x *worse*
than native. Only `risk_averse` survives, because it explicitly penalises
candidates the ensemble disagrees about, and it captures 77% of the headroom
there.

That is the clearest evidence in this project for a claim that keeps coming
up: **when the data is thin relative to the problem, modelling uncertainty
matters more than modelling the target.** It also means "which policy is best"
is not a fixed answer. It depends on the data, and a system that ships one
hardcoded policy will be wrong on half its deployments. That is exactly why
the policy is configurable through `SELECTION_POLICY`.

## 2.6 Production inference: choosing without executing

`/query/analyze` and `app.benchmark` run *every* candidate and then report
which was fastest. That is a measuring tool, not an optimizer. It spends N
executions to answer a question asked once, so serving traffic with it would
be strictly slower than having no optimizer at all. This was a named
limitation in the first draft of this document.

`optimizer/planner.py` is the real path. For N candidates it issues N
`EXPLAIN`s **without** `ANALYZE` (Postgres plans but runs nothing), scores
them, and executes only the winner. Measured live: **3.8 ms of planning
overhead against 138 ms of execution**, ~2.7%.

This only works because the feature layer never reads actual measurements.
`plan_tree.py` and `features.py` were deliberately restricted to estimate-side
fields: `Plan Rows`, `Total Cost`, `Plan Width` and node types. That looked
like an arbitrary restriction when it was written. It is what makes production
inference possible at all.

## 2.7 Regret: the summary number that isn't a summary

"Captured 29% of headroom" describes a run. Cumulative regret describes a
*trajectory*: how much slower the served plans were than the best available,
added up over time. It tells apart a learner that has converged from one
repeating the same mistake. Averages hide that completely.

Over 500 logged decisions:

| | Cumulative regret |
|---|---|
| Learned path | 5,016 ms |
| Native PostgreSQL | 6,261 ms |
| **Ratio** | **0.80** |

The learned optimizer has built up 20% less regret than always trusting
Postgres. This is the most defensible number in the project. It is computed
over every decision actually made, not a favourable subset, and a ratio above
1.0 would have meant the system was doing active harm.

Regret is measurable here only *because* the harness runs every candidate.
In production you never learn what the plans you skipped would have cost, so
this is an offline diagnostic computed from `plan_execution_log`, not a live
signal. `docs/METRICS.md` §5 gives the exact definition.

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

A correction to an earlier version of this section, which claimed the scan
corrector was "wired as an additional signal" into `features.py`. It was not.
`CardinalityCorrector` was written, tested, and imported by nothing. The claim
sat here unchallenged because nobody grepped for the call site. It is recorded
rather than quietly deleted, because a documented capability that does not
exist is the same class of error as §2.0: a plausible-looking write-up
describing something that never ran.

What *is* wired is the join-level correction described in §2.8.1.

### 2.8.1 Correcting the estimate instead of overriding the plan

Every candidate up to this point overrides the planner. `Leading(...)`
dictates a join order, `Set(enable_hashjoin off)` bans an operator. All of them
argue with the conclusion of a planner that is reasoning correctly from wrong
numbers.

pg_hint_plan's `Rows` hint allows the opposite move. `Rows(a b *10)` tells
Postgres that the join of `a` and `b` yields ten times what it thinks, and then
lets its own decades-tuned search run with a better premise.
`JoinCardinalityCorrector` learns the join-level q-error from
`plan_execution_log` and emits exactly those corrections, added as one extra
candidate alongside the hint families.

Join level rather than scan level is deliberate. Postgres estimates a join as
`left * right * selectivity`, treating the sides as independent, so the error
is not merely inherited from the scans -- it is *manufactured at the join*.
The `log_implied_selectivity` feature is that assumption made measurable.

**Verification first.** §2.0's lesson is that an unbound hint is silently a
comment, so the hint was checked against a live planner before anything was
measured: the same query estimated 4 rows unhinted and 400 rows under
`Rows(o u *100)`. The hint binds.

**The mechanism works.** Trained on 4,678 join observations, the largest
logged q-error being a 195x underestimate, the corrected plan beats native on
exactly the queries the v2 schema was built to break:

| Query | Native | Corrected | |
|---|---|---|---|
| `corr_brand_category` | 29.9 ms | 5.2 ms | **+82.6%** |
| `corr_city_country` | 17.9 ms | 5.0 ms | **+71.8%** |
| `corr_brand_band` | 10.5 ms | 6.3 ms | **+39.5%** |

**And it changes nothing.** Applied to every query unconditionally it is worse
than native more often than better (4 faster, 7 slower across 12). More
decisively, as a *candidate* -- which is how it is actually used -- it raises
the oracle ceiling by **0.0%**:

| | Total | Headroom |
|---|---|---|
| Native | 292.2 ms | -- |
| Best of hint candidates | 184.4 ms | 36.9% |
| Best, including Rows correction | 184.3 ms | 36.9% |

Every win it finds against native was already reachable through the existing
`Leading()`/`Set()` candidates. On a 4-to-6 table schema the hint action space
is close to exhaustive, so there is nothing left for a smarter premise to
reach. One query improved by 0.03 ms, which is noise.

This is the §2.9 result again in a different costume: the mechanism was never
the binding constraint. Publishing the 82.6% figure without the ceiling
measurement beside it would have been the same error this document opens with
-- a real effect, measured honestly, that changes nothing.

It is kept, and left on, for one reason that is a hypothesis rather than a
finding: the case it is built for is JOB scale. At 17 tables the candidate
generator samples a handful of orderings out of millions and cannot cover the
space, so "correct the estimate and let the planner search" reaches plans no
enumeration will produce. That is untested here -- it needs the JOB collection
§2.3 says is missing -- and it is why the feature ships behind
`ENABLE_ROWS_CORRECTION=0` rather than being presented as a win.

## 2.9 The benchmark was the bottleneck

The most persistent complaint about this system was that it did not reduce
query cost. Investigating it properly produced the most important result in
the project, and it had nothing to do with the model.

### Measuring the ceiling

For every query in the v1 workload, run *every* candidate plan and record
the fastest. That gives the **oracle ceiling**: the best any selector could
possibly do.

| | v1 synthetic | TPC-H | JOB/IMDB |
|---|---|---|---|
| Tables | 4 | 8 | 21 |
| Mean best-possible gain | **6.5%** | ~23% | **~75%** |
| Queries with >5% available | **7 / 25** | -- | 3 / 4 |
| Best model's top-1 accuracy | **0%** | -- | 25% |

On v1, PostgreSQL was already choosing the best plan for 18 of the 25
queries. Six model types were compared (LightGBM, random forest, extremely
randomised trees, gradient boosting, ridge, and an MLP) and **not one picked
the fastest plan even once**. That is not six failed models. There was no
signal to learn. Where signal did exist, on JOB, extremely randomised trees
reached 25% top-1 accuracy and captured 73% of the headroom.

**A benchmark where the baseline is already optimal cannot measure an
improvement.** Every earlier attempt to fix "cost isn't reducing" by adjusting
thresholds, targets and model types was working on the wrong variable.

### Rebuilding the dataset to be hard

So `data/schema.sql` was rewritten to target the exact mechanism Leis et al.
identified: PostgreSQL's **independence assumption**. Given
`WHERE a = x AND b = y`, it estimates sel(a) x sel(b), which is only correct
when the columns are unrelated. v2 makes them related on purpose:

    city    -> country      (Mumbai implies IN)
    brand   -> category     (Voltix implies electronics)
    price_band ~ category   (electronics skew premium)
    channel ~ status        (cancellations cluster in one channel)

Filtering on both halves of a dependency now produces a row estimate several
times too small, and that error compounds through the joins into a genuinely
wrong join order. Two extra tables widen the graph to six, so the workload
reaches 5- and 6-way joins, where ordering matters far more than it does at
two or three. `workload.py` was rewritten to exercise each trap, with
uncorrelated **control** queries alongside, so the contrast is something you
can see rather than something you have to assume.

One thing is deliberately missing: any `CREATE STATISTICS` object.
Multi-column statistics are exactly how a DBA would *fix* these correlations,
so leaving them out is what preserves the errors the benchmark exists to
exploit. Adding them is the obvious controlled experiment, and the headroom
should collapse when you do.

### Result

| | v1 | **v2** |
|---|---|---|
| Mean best-possible gain | 6.5% | **22.2%** |
| Queries with >5% available | 7 / 25 | **14 / 23** |

with the largest gains landing exactly on the designed traps:

| Query | Trap | Headroom |
|---|---|---|
| `corr_brand_category` | brand -> category | **95.4%** |
| `corr_brand_band` | brand -> category + band | **95.2%** |
| `corr_6w_everything` | both, over 6 tables | **93.6%** |
| `corr_city_country` | city -> country | **63.1%** |

The mechanism predicted the outcome. That is the strongest evidence
available that the diagnosis was right, rather than just convenient.

### A regression this exposed

Widening the action space (§2.10) without re-collecting training data sent
**17 of 25 queries into the safety veto**. The model was being asked to score
`Set(enable_*)` plans it had never seen in training, so its predictions were
guesses beyond its data, and the confidence gate correctly refused them. This
is classic train/serve skew. The fix is not a code change: re-run
`app.collect_data` whenever the action space changes, which the README now
says.

## 2.10 Widening the action space

`Leading()` join-order hints on their own turned out to be nearly useless on
small queries. A two-table join has only two orderings, and PostgreSQL already
picks the better one, so every "candidate" came back as the *same plan at the
same cost*. You could see it in the dashboard: native and chosen both read
4673.9. The optimizer looked like it was running while having nothing to
choose between.

`hints.py` now also emits Bao's actual action space: **operator toggles**
such as `Set(enable_nestloop off)`, `Set(enable_indexscan off)`, and
combinations of them. Switching off a class of operator forces the planner to
re-plan under that restriction, which produces genuinely different plans.
`plan_fingerprint` removes candidates that come back structurally identical,
so the candidate count reflects the real size of the action space rather than
counting copies.

On JOB this immediately found wins join-order hints could not reach:

| Query | Native | Best | Gain | Winning hint |
|---|---|---|---|---|
| `auto_2w_01` | 525.0 ms | 108.6 ms | **79.3%** | `Set(enable_indexscan off)` |
| `auto_3w_03` | 320.9 ms | 180.2 ms | **43.8%** | `Set(enable_indexscan off)` |

In both cases the winning hint is a scan-method toggle, not a join order.
That is an action the previous design could not even express.

## 2.11 Reading the manual, and what it cost to check

The PostgreSQL manual (3,156 pages) was read against this system rather than
summarised. Findings are in `docs/TECHNIQUES_BACKLOG.md`; the ones that changed
code are below. The recurring lesson is the one this project keeps learning:
**the manual describes a version, and the server is the authority.** Four
documented features turned out not to exist on the running server, and every one
of them would have shipped as a silent no-op:

| documented | reality here |
|---|---|
| `DisableIndex(t idx)` | not in pg_hint_plan 1.6.3 — `Unrecognized hint keyword`, at **INFO** severity, which the default `client_min_messages` discards |
| `pg_overexplain` | PG18 |
| `parallel_workers_to_launch` in `pg_stat_statements` | PG18; this is 1.10 |
| `shared_blk_read_time` | named `blk_read_time` here |

`test_integration_hints.py` now generates the entire action space and asserts
the server recognises every keyword in it, so the next one is caught on the run
that introduces it rather than by inspection.

## 2.12 The optimizer was paying for candidates it could prove were useless

Profiling the serving path, twenty runs, median:

| stage | ms | share |
|---|---|---|
| baseline `EXPLAIN` | 0.57 | 2% |
| generate candidate hints (Python) | 0.66 | 3% |
| `EXPLAIN` × 31 candidates | **19.13** | **75%** |
| features + LightGBM inference | 5.16 | 20% |

31 candidates produced **12 distinct plans**. `plan_fingerprint` discards the
duplicates — after the round trip has been paid for.

`optimizer/prefilter.py` discards a provable subset first. PostgreSQL does not
remove a disabled operator; it prices it at `disable_cost`. So if the baseline
plan contains no node of the type being disabled, its cost is unchanged by the
ban and it was already the cheapest — `Set(enable_mergejoin off)` on a plan with
no Merge Join cannot do anything.

The prefilter was verified by planning **every candidate it dropped**, across
three query shapes, and comparing against the baseline fingerprint and cost.
That caught two bugs in the first version:

1. `Set(enable_nestloop off) Set(enable_mergejoin off)` was dropped because the
   *second* flag was a no-op, while the first took a 7870 plan to 12234.
   Reasoning per-flag is wrong precisely where the flags are meant to interact.
2. `Set(enable_hashagg off)` was dropped on a plan that was hash aggregating.
   EXPLAIN's **JSON** writes `"Node Type": "Aggregate"` with a separate
   `"Strategy": "Hashed"`; `HashAggregate` is the text format's name and matches
   nothing. Measured 492.05 against a real 1970.82.

A third came from the unit tests: an **empty** plan was read as "contains no
Merge Join", so a stub baseline stripped candidates on no information at all.

After the fixes: **0 wrongly dropped** on all three shapes, 32 → 28 round trips,
25.52 → 21.01 ms, same 12 distinct plans.

**What it does not reach**: 56 of the 83 remaining candidates are still
duplicates, and they are `Leading()` variants. Whether two join orders produce
the same plan is what the planner decides. Anything less than a proof there
shrinks the action space silently, which is the failure this project keeps
rediscovering.

### And on rewriting it in a compiled language

The question came up and is answerable from the table above. 77% of the time is
PostgreSQL planning and socket round trips, which no language change touches.
Of the 23% that is Python-side, most is LightGBM inference — already compiled
C++ behind a binding. Genuinely interpreted Python is **0.66 ms, 3%**.

The way to stop paying for round trips is to be inside the planner, via
`planner_hook` — and PostgreSQL's extension API is **C, not C++**: extensions
build against `postgres.h` with C linkage, and `elog(ERROR)` unwinds with
`longjmp`, which walks straight past C++ destructors.

## 2.13 Measuring the plan instead of the cache

The most expensive measurement mistake here was reporting an 80.5% improvement
that was really 26.0% — the two sides ran at different cache states.
`experiment.py`'s paired, order-flipping design fixes that over many runs, but
makes fairness *likely* rather than *checkable*, and says nothing about the
single `/query/analyze` call the dashboard actually shows people.

`EXPLAIN (ANALYZE, BUFFERS)` was already being requested and the numbers thrown
away. Measured, twelve runs each:

| plan | runtime | spread | blocks | spread |
|---|---|---|---|---|
| native | 25.01 ms | 5.3% | 418 | **0.0%** |
| `Parallel(oi 4 hard)` | 31.15 ms | 4.9% | 197 | **0.0%** |
| `Set(enable_nestloop off)` | 30.38 ms | 4.8% | 1024 | **0.0%** |

Block counts are a property of the plan and the data. Runtime is a property of
the moment.

`app/measurement.py` uses them two ways: `fairness()` compares the two cache hit
ratios and says whether the gap alone could explain the result, and
`blocks_verdict()` is a tie-breaker for a runtime difference inside the noise
band. `prewarm()` loads both sides into shared buffers first.

**It is a second opinion, not a verdict**, and the table above says why: the
parallel plan reads 2.1× fewer pages and is 25% *slower*, because worker startup
dominates on a query this size. The note it emits says so.

## 2.14 The advisor was recommending indexes the planner ignores

`analyze_plan` found a wasteful sequential scan and proposed
`CREATE INDEX ON t (col)` for every predicate shape. For three shapes that index
is built, costs disk and write amplification, and is **never used** — so the
next advisor run reports it as unused and suggests dropping it.

Measured against the live planner, building both the old and new recommendation:

| predicate | naive `CREATE INDEX ON t (col)` | recommended |
|---|---|---|
| `name LIKE 'Volt%'` | Seq Scan — **ignored** | Index Scan, `text_pattern_ops` |
| `name LIKE '%olti%'` | Seq Scan — **ignored** | Bitmap Heap Scan, trigram GIN |
| `lower(city) = 'x'` | Seq Scan — **ignored** | Bitmap Heap Scan, expression index |

Why each: §11.2.1 requires `text_pattern_ops` for an anchored `LIKE` outside the
C locale (this database is `en_US.utf8`); an unanchored pattern has no prefix to
descend on and is not B-tree-indexable at all; and §11.7, an index on `city`
cannot serve `lower(city)`.

`app/index_advice.py` also caps recommendations at three columns (§11.3), states
the §11.5 trade-off against index combination, proposes covering indexes with
`INCLUDE` — verified turning a Seq Scan into an **Index Only Scan** — and never
recommends indexing a column that already carries a unique index (§11.6).

## 2.15 Optimising the optimizer's own output

Reported by a user, and the most valuable bug found in this round because it
sits on the obvious workflow: optimise a query, copy the SQL, paste it back to
improve it further.

`apply_hint` prepended unconditionally, producing two hint comments.
**pg_hint_plan reads only the first and discards the rest** — not as an error,
not in the "not used" or "error hint" sections of its own debug output. Measured:

| input | cost | `used hint:` |
|---|---|---|
| `Leading(u o)` | 5332.3 | `Leading(u o)` |
| `Set(enable_hashjoin off)` | 14365.1 | `Set(enable_hashjoin off)` |
| A then B, stacked | 5332.3 | **`Leading(u o)` only** |
| B then A, stacked | 14365.1 | **`Set(...)` only** |
| merged into one comment | 14365.1 | **both** |

So the second pass applied its new hint and silently threw away the one the user
had pasted in. `apply_hint` now merges into a single comment, with last-one-wins
on conflict — pg_hint_plan's own rule — so chained passes converge.

It had a second half. `query_template` kept the hint comment, so the fingerprint
changed every pass, and everything keyed on `query_id` — the **regression
guard**, the **decision cache**, the **regret curve** — saw each pass as a
brand-new query. The guard could not have blocked a bad hint however many times
it had seen one. Hints are now stripped from the identity: a hint says how to
run a query, not which query it is, which is also why `main.py` already logs the
baseline and every candidate under the same `sql_text`.

Fixing it exposed two more: a bushy `Leading(((a b) (c d)))` has two levels of
nesting and the directive pattern allowed one, so it vanished from the merge;
and a JSX block comment describing hint syntax contained the sequence that ends
a block comment.

## 2.16 The trustworthiness panel described a different model

`/model/diagnostics` exists to answer "should I believe this model". It was
answering it about a different one.

`diagnose()` read `models/eval_results.json`, a fixed filename. The deployed
model is whatever `model_store` last promoted. They had drifted apart: the panel
reported **-5.9% headroom captured** while the promoted version measured
**+79.2%**.

Two causes, both fixed. The evidence now comes from the registry entry for the
version actually being served, and the report says which — `evaluation_source`
— so the question is checkable rather than assumed. An explicitly passed
`eval_path` still wins, because a caller naming a file means that file.

And the served model was itself stale: promoted at 04:54, before the
fixed-schema collection finished at 08:08. Retrained and promoted, it captures
**79.2%** of the oracle headroom (`pairwise_rank`) against the previous
**11.6%**.

## 2.17 "Why is it never confident" was never about confidence

The complaint that drove most of this work: the model finds large improvements
and then declines to use them. The gate's own explanation, printed on every
benchmark line, looked like a threshold problem:

    model predicted 1.39x native (pessimistically 2.14x, needs < 0.80x to deviate)

Thirteen of thirteen queries vetoed, every prediction between 1.10x and 1.66x.
The obvious readings are "the confidence bar is too high" or "the model is bad".
Both are wrong.

### What it actually is

321 candidates on the live workload, predicted against measured:

| | predicted | actual |
|---|---|---|
| median ratio | 1.464 | 1.075 |
| mean ratio | 1.816 | 2.038 |
| **minimum ratio** | **0.826** | **0.483** |

```
mean log error (predicted - actual)   +0.1025   ->  x1.108 systematic
median log error                      +0.1779   ->  x1.19

candidates genuinely faster than 0.8x native            54
...that the model also predicted under 0.8x              0
```

**The model's most optimistic prediction anywhere was 0.826x, and the gate
deviates below 0.80x.** It could not have fired on any query, at any
`confidence_z`, at any `min_relative_gain` above 0.175. The safety gate was not
being cautious; it was unreachable.

Note the shape of the error, because it names the cause: the model
*over*-predicts in the middle (median 1.464 against 1.075) and *under*-predicts
at the top (mean 1.816 against 2.038). That is not a constant offset, it is
shrinkage — a squared-loss model hedging toward the mean of what it was trained
on rather than committing to the extremes. The extremes are where the wins are.

### None of this contradicts the model being good

The same model captures **79% of the oracle headroom**. That number is a
statement about **ordering**, and ordering is all the argmin uses. Level is what
the gate uses, and nothing had ever checked it. Two entirely different
properties of one set of predictions, and only one of them had a test.

### The correction

`optimizer/calibration.py` fits a least-squares line on out-of-bag predictions:

    calibrated_log_ratio = intercept + slope * raw_log_ratio

`slope > 1` undoes the compression, `intercept` removes the shift. It costs no
training data, for the same reason `conformal.py` uses out-of-bag rows: a bagged
ensemble already holds out about a third of the rows for every member.

The property that makes it safe to apply is that an increasing affine map is
**strictly monotone** — it cannot reorder candidates, so the argmin picks the
same plan and the 79% headroom capture is preserved exactly. Only the position
relative to the threshold moves. A fit producing a non-positive slope is refused
rather than applied, because a calibration that reorders candidates is worse
than none.

The conformal bound is fitted on *calibrated* means, so the interval brackets the
number the gate will actually see. Fitting it on raw means and then shifting the
point underneath would silently invalidate its coverage guarantee.

### What the calibration measured, which was the more useful result

Fitted on out-of-bag rows, it came out at **slope 1.018, intercept -0.005,
systematic factor 0.999** — and the most optimistic out-of-bag prediction was
**0.530x**, comfortably inside the gate.

So the model is *already* well calibrated against its own training
distribution, and badly calibrated against the serving one. Those two numbers
disagreeing on the same model is not a bad regressor. It is train/serve skew,
and the calibration report is what made it visible: before this, there was no
number anywhere that said "the most optimistic thing this model will ever
predict is 0.826x", which is the single fact that explained the behaviour.

`app.train` now prints it on every run:

    point calibration: slope=1.018 intercept=-0.005 on 2194 out-of-bag rows
      systematic error x0.999 -> x1.000; most optimistic prediction
      0.530x -> 0.522x native

If that last number is above the gate's threshold, the gate is decorative — and
now you can see it without instrumenting anything.

### The remaining skew

`app.collect_data` gathers **30 candidates per query** (join methods and tuning
both on by default). A served query sees roughly **12** after
`INCLUDE_JOIN_METHODS` is off, the adaptive budget truncates, and
`optimizer/prefilter.py` drops the provable no-ops. §2.9 documents that
difference as harmless for *coverage* — serving is a subset, so the model is
never asked to score a family it has not seen.

It is not harmless for *calibration*. Shrinkage pulls predictions toward the
mean of the training distribution, and the training distribution contains a
large tail of deliberately bad candidates that serving never offers. Every
serving prediction is therefore lifted toward a mean that does not apply to it.

That is the live hypothesis for the 0.826x floor, and it is testable by
collecting with the flags serving actually uses.

## 2.18 The gate could not fire, and the reason was in the training set

§2.17 established that the model's predicted *level* was wrong at serving —
across 321 candidates its most optimistic prediction anywhere was 0.817×, the
gate deviates below 0.80×, and 64 of those candidates were genuinely faster than
0.8×. It also established that the point calibration fitted on out-of-bag rows
came out at slope 1.02, systematic factor 0.999: **the model is well calibrated
against its own training distribution and badly calibrated against the serving
one.** That is train/serve skew, and the calibration report is what made it
visible.

This section finds it.

### The measurement

Over the 12,826 candidates in the log, split by whether the candidate belongs to
a family the serving path offers:

| family | n | mean ratio | median | p95 |
|---|---|---|---|---|
| forced join methods | 980 | **5.49** | 3.16 | 17.4 |
| everything else | 11,846 | **1.97** | 1.10 | 6.4 |

`app.collect_data` gathers the forced join-method family by default.
`INCLUDE_JOIN_METHODS` is **off** at serving. §2.9 documents that difference and
calls it harmless, on the grounds that the model is never *asked* to score a
family it has not seen. That is true, and it is a statement about **coverage**.

It is not harmless for **level**. A squared-loss model shrinks its predictions
toward the mean of what it was trained on. Including that family moves the
training mean from 1.97 to 2.24 — a factor of **1.137**. The measured systematic
bias at serving was **×1.152**.

Those two numbers agreeing to within 1.5% is the whole argument.

### The fix

`train.rows_for_served_action_space` drops rows for families the serving path
does not offer, and reads *which* families from `optimizer/planner.py` — the same
flags serving uses — so the two cannot drift. It reports what it dropped, because
silently training on a subset is the same class of mistake as silently serving
one.

Writing the test for it caught a bug in the first version: `"NestLoop("` is a
substring of `"NoNestLoop("`, so the *negative* join family — which **is** served
under `INCLUDE_NEGATIVE_JOINS` — would have been dropped too. That is the
opposite mistake, training on fewer families than will be asked about rather than
more, and it is the one that produces a model scoring plans it has never seen.

### And a second contamination, self-inflicted

Retraining on freshly collected data made the model *worse*: 78% of the oracle
headroom down to 3%. Two causes, both mine, and both worth recording because
they are easy to repeat.

**Repetitions.** The fresh collection used `--reps 2` to save time. Measured on
this database, repeated executions of the *same* plan differ by **18.6% on
average** and up to 125%, so a two-sample median is barely better than a single
draw. §2.2 already established that aggregating repetitions to a median is what
removed every regression from this system; collecting two of them gives that
mechanism almost nothing to work with.

**A moving database.** The two collections ran hours apart, and between them this
session had created and dropped indexes, run `VACUUM`, prewarmed shared buffers
and installed two extensions. Folding them together gives each (query, hint)
pair a median across two *different* databases — a label for neither.

`_aggregate_repetitions` already had a `latest_run_only` flag, with a docstring
naming this exact hazard: *"a comparison wants measurements taken under the same
conditions, because the same native query in this log ranges from 2.4 ms to 13.7
ms across runs collected months and workloads apart."* It had only ever been
wired up for calibration. It is a training option now.

The lesson generalises past this project: **a plan-latency log is only training
data while the database it measured is still the database you are serving.**
Every schema change, index build and vacuum silently invalidates part of it, and
nothing in the log records that it happened.

### What it changed, measured

Retrained on the serving action space, from a single collection of the current
database, three repetitions per candidate:

| | before | after |
|---|---|---|
| most optimistic prediction, anywhere | 0.817× | **0.752×** |
| candidates predicted under the 0.80× gate | **0** of 321 | 3 of 40 truly-fast |
| spread of log error | 0.523 | **0.286** |
| benchmark: queries where the gate fired | 0 of 13 | **1 of 24** |
| benchmark: headroom captured | 0.0% | **4.5%** of 600 ms |

The threshold is crossed: the gate can now fire, and does. That is the
difference between a safety mechanism and a decoration, and it was the thing
being asked about.

It is not a solve. 23 of 24 queries are still vetoed, the residual systematic
bias is still **×1.15**, and on the nine held-out queries the model captures
**0.0%** — the number that says whether it generalises rather than memorises.
The action-space skew was one cause of the level being wrong, quantified and
removed; something else is still worth about 15%.

### A caveat on every before/after number in this section

They are not strictly comparable, and the reason is the same lesson again. Over
this session the database changed underneath the measurements — indexes built
and dropped, a `VACUUM`, shared buffers prewarmed, two extensions installed. The
effect is visible in the evaluations themselves:

    before   native 247.1 ms   oracle 190.7 ms   headroom 56.4 ms (22.8%)
    after    native 183.9 ms   oracle 151.2 ms   headroom 32.7 ms (17.8%)

Native got 26% faster and the available headroom shrank by 42%. A model
"capturing 78% of the headroom" on the first database and one capturing 4.5% on
the second are not being scored on the same problem. The benchmark numbers above
are self-consistent — it measures native and served in the same run, seconds
apart — but the two evaluation files are not, and reading them as a regression
would be wrong.

### The policy barely matters while the gate vetoes almost everything

The evaluation ranks the four selection policies by headroom captured, and on
this model it preferred `risk_averse` (0.365) over the configured
`pairwise_rank` (0.163) by more than a factor of two. Benchmarked, that ordering
does not survive:

| policy | vetoes | headroom captured |
|---|---|---|
| greedy | 23/24 | **4.5%** of 600 ms |
| risk_averse | 23/24 | 2.9% of 750 ms |

Both save under 1% of total runtime, and the difference between them is inside
the noise this database produces run to run.

The two numbers disagree because they measure different things. The evaluation
takes the model's argmin and asks how good that pick was — a question about the
**policy**. The benchmark applies the safety gate, which discards the pick on 23
of 24 queries — so what it mostly measures is the **gate**.

While the veto rate is 96%, tuning the policy is optimising the part of the
system that is almost never consulted. That is worth stating plainly, because
the policy is the knob the dashboard makes most prominent and the eval report
makes it look like the most consequential one.

## 2.19 The calibration was refusing, and blaming the wrong thing

§2.18 got the gate firing again. Running `app.calibrate` on the result printed
a grid and then this:

    No setting met the regression-rate bound while still optimizing anything.
    That means the model isn't accurate enough to act on yet -- collect more data.

Both sentences were doing damage. The second is a diagnosis, and it was being
issued without checking which of the two bounds had actually refused.

### Two bounds, and only one of them is a safety property

`sweep()` enforces `max_worst_regression` (no single query may end up more than
1.25x slower than native) and `max_regression_rate` (no more than 25% of the
queries it acts on may regress at all). Its docstring already records, from
measurement, that these are not peers:

> `max_worst_regression` is the real safety bound... `max_regression_rate`
> stays as a much looser sanity check on frequency. Together they say "a few
> small losses are an acceptable price for many wins; one large loss is not".

A refusal by the first means a query got materially slower and the model needs
to be better. A refusal by the second, *while the first is satisfied*, means the
losses are real but individually negligible -- and whether to accept many tiny
losses in exchange for the wins is a policy choice belonging to whoever runs
the tool. No quantity of additional training data moves a threshold comparison.

`refusal_lines()` now distinguishes three cases and names the bound. It also
declines to suggest raising the rate bound when the net is negative: there,
loosening it would buy a setting that loses time overall, so the small losses
have no wins to pay for them and the advice reverts to collecting more data.
That last branch was a bug in the first version of this fix, and the run that
caught it was a real one, not a test.

### The verdict was one query

With honest reporting the same run said the *magnitude* bound had refused it:
one query 1.64x slower. Worth looking at, so:

| held-out query | gate acts? | net | worst |
|---|---|---|---|
| `auto_4w_72_oi_quantity_lt` | yes | **-158.1 ms** | 1.64x |
| `corr_6w_status_premium` | yes | +2.2 ms | 1.00x |
| the other 13 | no | -- | -- |

Every number in that grid -- 13% deviation, 50% regression, 1.64x worst,
-7.3% net -- was computed from **two events**, and the verdict from one of them.
That is not a measurement of the model. It is a measurement of one query.

### The population was the documented defect, shipped as the default

`_grouped_rows`'s docstring spends two hundred words on why generated queries
are the wrong basis for choosing a threshold: they carry **12.3%** mean
available headroom against the built-in workload's **40.5%**, while the model's
typical error is around **24%**. On the first group the error exceeds the
headroom and declining is correct; on the second there is three times more to
win than the model is typically wrong by. A sweep over both picks a threshold
right for neither. `measure_settings` lists the same thing as one of three ways
the offline replay had been wrong.

The flag to fix it existed. Its help text told you to pass `auto_`. And the
default was `None`, so the documented defect was what you got by running the
tool the obvious way. Measured over the same 15 held-out queries:

| population | outcome |
|---|---|
| including `auto_*` | nothing qualified at any of 50 settings; best net **-7.3%** |
| excluding `auto_*` | net **+24.0%**, **0%** regressions, worst case **1.00x** |

Documenting a defect and providing an opt-out is not the same as fixing it.
`DEFAULT_EXCLUDE_PREFIX = "auto_"` makes the documented-correct usage the
default; `--exclude-prefix ''` is the way back.

Generated queries remain excellent *training* data -- §2.6 records that they
took the model's typical error from 43% to 24%. This is only about where a
threshold that will be spent on the real workload gets chosen.

### The expensive check was not checking the pick

`--measure` exists because the replay has been wrong three times, and it is the
honest instrument: it runs `app.benchmark` once per setting with that setting's
thresholds forced, so every number it returns came from executing the plans it
describes. It costs a full workload sweep per row.

`MEASURED_GRID` was a fixed tuple of five settings. The replay's new
recommendation -- `z=2.0, min_gain=0.02` -- was not one of them. The grid held
`z=2.0, min_gain=0.20`, the incumbent, and nothing between 0.05 and 0.20. So
the expensive instrument was about to spend five full benchmarks without
measuring the thing it was run to check.

Adding the point to the constant would have worked exactly once, until the next
retrain moved the recommendation. `grid_with_recommendation()` computes it
instead: `--measure` now runs the fixed grid *plus* whatever the replay
currently recommends, deduplicated. The incumbent stays in deliberately -- a
measurement of a candidate without a measurement of what it would replace is
not a comparison, only a number.

### The deployed gate was claiming a score nobody had computed

`models/gate.json` is what the serving path reads and what anyone deciding
whether to trust this system looks at. It held:

```json
"expected_regression_rate": 0.0,
"expected_worst_regression_ratio": 1.0
```

Those came from the `--measure --apply` path, which passed them as literals:

```python
apply_gate({**best, "deviation_rate": 0.0, "regression_rate": 0.0,
            "worst_regression_ratio": 1.0})
```

A benchmark reports captured headroom. It does not produce a per-query
regression breakdown, so there was nothing to put in those fields -- and what
went in was a perfect score. Worse, `app.main` retrains when observed
regressions exceed `expected_regression_rate`, so a fabricated 0.0 turns the
first regression of any size into a retrain trigger.

They are `null` now. **"Not measured" and "measured as flawless" must not look
the same to whoever reads the file next**, and `drift_report` already declines
to judge a null promise, so an unmeasured claim cannot be broken.

The same file also never recorded *which queries* it was calibrated on -- the
one variable that moved this result from -7.3% to +24.0%. `apply_gate` now
takes a `provenance` dict and writes `chosen_by`, `evaluated_on` and
`excluded_query_prefix` alongside the numbers. Expected rates without a stated
population are close to meaningless.

### What generalises

Three things, none specific to this project:

* **A refusal has to name its cause.** A tool that says "no" and then guesses
  at why sends people to fix things that are not broken. The guess here cost
  nothing to make and would have cost a data-collection run to act on.
* **A default is a recommendation.** Knowing a default is wrong, writing that
  down, and adding a flag leaves the wrong thing happening by default.
* **Check n before believing a rate.** "50% of deviations regress" and "2
  deviations, 1 regressed" are the same sentence, and only one of them sounds
  like evidence.
* **A verification grid has to contain the thing being verified.** A fixed set
  of test points drifts away from a moving recommendation, and the failure is
  quiet: the expensive check still runs, still prints a table, and still tells
  you nothing about the decision you were making.
* **A default value is a claim.** Writing `0.0` where nothing was measured
  states a result, and every reader downstream -- including the retraining
  trigger -- believes it. `null` is the honest value and costs nothing.

## 3. Stretch goals

**Join-method selection.** `generate_join_method_candidates`
(`backend/app/optimizer/hints.py`) pairs each sampled join order with a forced
method (`HashJoin`, `NestLoop` or `MergeJoin`) applied at every prefix of that
order's left-deep join tree. The join nodes of `Leading(a b c d)` are exactly
the prefixes `(a b)`, `(a b c)` and `(a b c d)`, so this approximates "use
this method throughout the plan" without needing a full per-node hint-tree
generator. `features.py` counts the join methods each plan actually used
(`n_hash_join`, `n_nestloop_join`, `n_merge_join`), so the model can learn
patterns that depend on method, not just on order.

**Dataset-agnostic pipeline.** `optimizer/features.py` originally hardcoded
the synthetic schema's four tables and their aliases. It is now driven by the
schema instead. Table identity comes from each EXPLAIN plan's own
`scan_relations`, an alias-to-table-name map that `plan_extractor.py` reads
straight off the plan, and reference row counts come from
`schema_introspection.py` querying Postgres's own `pg_class` statistics.

Point `DATABASE_URL` at a different database and `hints.py`,
`plan_extractor.py`, `features.py` and `train.py` all adapt with no code
change. `feature_columns` and `table_cardinalities` are computed at training
time and pickled alongside the model, so inference stays consistent with
whatever schema it was trained on. This is what let the JOB/IMDB import
(Section 2.3) reuse the exact same pipeline.

## 4. Limitations, and what was done about them

Every item here was written down as a known weakness first. Several turned out
to be fixable once they were stated precisely, which is the main argument for
keeping a list like this at all.

### Closed

**Candidate generation wasted most of its budget.** Join orders were sampled
blindly from all permutations, and an order that introduces a table sharing no
join predicate with those already placed forces a cartesian product -- which
Postgres prices at `disable_cost` (~1e10) and never chooses. The optimizer was
handed a list of alternatives most of which it could not use.

`plan_extractor.extract_join_graph` now reads the query's join graph off the
baseline plan's own conditions (`Hash Cond`, `Merge Cond`, `Index Cond`), and
generation only produces connected orders. Measured over the workload's 3- and
4-table queries:

| | Usable candidates |
|---|---|
| Blind permutation | 23 / 48 (48%) |
| Graph-aware | 40 / 40 (**100%**) |

The effect is larger than the rate suggests, because the absolute count rose
too. On `corr_4w_premium_us` blind sampling produced **one** usable candidate
out of eight; the graph-aware generator produces eight. That query effectively
had no action space at all. Note also that this bit at four tables, not "above
five" as originally written.

**Self-joins collapsed to one feature slot.** Per-table slots were assigned,
not aggregated, so `movie_info AS mi1, movie_info AS mi2` was described by
whichever alias appeared last in the plan and the other vanished. The vector
could not distinguish a self-join from a single scan -- and self-joins are
routine in JOB, the benchmark this most needed to work on.

Slots now aggregate across occurrences: earliest join position, most selective
scan, index-scan if any occurrence uses one, plus an `occurrences` count. The
vector stays fixed-length and one-slot-per-table, so it still transfers across
schemas.

**The guard could not protect a query it had never seen.** Blocking is
retrospective: a query has to *demonstrate* a regression over several
executions before it is stopped, so the first few runs of a new query had
nothing watching them -- exactly when the model is most likely to be
extrapolating.

Refusing to act on unseen queries would make the optimizer useless on any
fresh workload, so the confidence bar is scaled instead.
`RegressionGuard.caution_multiplier` returns 2x with no history and eases
linearly to 1x once `min_observations` served executions exist, and
`select_plan` multiplies both halves of the gate by it. A marginal prediction
on an unknown query no longer clears the bar; a clear win still does.

**Explicit thresholds were silently ignored.** Found while testing the above.
"Explicit constructor arguments win over `models/gate.json`" was implemented as
`if arg == DEFAULT`, which cannot distinguish a deliberately-passed default
from an untouched argument -- so a caller asking for `confidence_z=1.0` got
whatever the calibration file said, and there was no way to opt out short of
deleting the file. A sentinel now separates the two cases. This mattered more
than it looks: the deployed `gate.json` holds `confidence_z=0.0,
min_relative_gain=0.0`, so anything relying on the documented defaults was
running with the gate effectively disabled.

**Training on single executions.** `collect_data` defaulted to `--reps 1`, the
value §2.2 identifies as the cause of live runs ranging from +40% to -149%.
The default is now 3, matching `app.onboard`, so the median aggregation that
removed every regression is available by default rather than opt-in.

**The regression guard's effect size could not be settled.** Not because the
question was hard, but because `app.benchmark` printed its results and returned
nothing, so repeating it meant copying numbers off a terminal. `app.experiment`
now runs the comparison with interleaved arms, bootstrap confidence intervals
and a sign test, and reports *unresolved* when the interval spans zero. §2.4
has the first output.

**Non-reproducible action spaces.** Sampling used the unseeded global `random`,
so a query above the candidate budget got a different action space on every
call -- training saw one subset, inference another, and two benchmark runs
meant to differ only in a flag were also comparing different candidates.
Seeding from the table list makes generation deterministic per query.

### Already present, just undocumented

**Scheduled retraining.** The list said retraining was "triggered, not
scheduled". A background scheduler has been there all along:
`AUTO_RETRAIN_SECONDS` runs `retrain_if_needed` on an interval, off the event
loop, holding a reference to the task so it cannot be garbage collected
mid-await, and cancelling it cleanly on shutdown. It is off by default because
a job that silently swaps the served model should be opted into. The limitation
was a documentation gap, not a missing feature.

### Still open

**Prediction error against available headroom.** ~400 rows across 25 queries
gives an error comparable to the opportunity being measured. More repetitions
help and are now the default; the real fix is more data, which is compute time
rather than a design question.

**The safety veto is cost-based, not learned.** It compares estimated costs,
and distrusting those estimates is the premise of the project. It reliably
catches catastrophes (`disable_cost` marks them clearly) but cannot catch a
plan that is cheap on paper and slow in reality.

**Cold start.** Before a model exists, selection falls back to the cost
heuristic. Reasonable, but there is no bootstrapped warm-up of the kind Neo
uses.

**Single-node, single-connection measurement.** Every latency here is measured
on an otherwise-idle database. Real workloads have concurrent queries competing
for buffer cache and I/O, which changes the arithmetic on candidate execution
cost.

**Synthetic skew is still synthetic.** `data/schema.sql`'s correlations are
deliberately clean compared to real data's long tails, which is exactly why the
JOB results in §2.3 matter for external validity -- and why the synthetic
numbers should not be assumed to carry over.
