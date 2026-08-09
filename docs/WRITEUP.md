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
